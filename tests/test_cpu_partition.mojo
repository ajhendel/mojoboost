"""The row arena: exact order equality with the shipped partition.

`tree.RowArena` replaces the two freshly allocated `List[Int]` sides that
`tree.partition_rows_into` produces per split, and the fresh `List[Int]` of
every row that a tree starts from, with two `Int32` buffers allocated once and
ping-ponged. Everything a consumer of a row list can observe has to be
unchanged by that, and exactly one property makes it so:

    **the arena's left window, followed by its right window, equals
    `partition_rows_into`'s `left` followed by its `right`, index for index.**

Histogram accumulation walks a node's row list in order and sums `Float64` in
that order, so a side that came out permuted -- even holding the same set --
would move every histogram cell beneath that node and every leaf value under
it. There is no tolerance anywhere in this file: every comparison is integer
equality or `Float64.to_bits()`, because a comparison that needed a tolerance
would not have established the property.

What is checked, in order:

1. `test_arena_partition_matches_two_list_exactly` -- the equality above, over
   the routing matrix `partition_rows_into` already handles: numerical splits
   with a missing bin, numerical splits on a feature with no missing bin (-1),
   both default directions, and categorical splits by set membership. Root
   spans and nested spans, on both arena buffers, at serial and forced-parallel
   worker counts. The parent's own window is asserted untouched, which is what
   makes ping-pong safe for a caller that still holds the parent.
2. `test_arena_partition_deterministic_across_workers` -- byte-identical arena
   contents at `MOJOTREES_NUM_WORKERS` 1, 3 and 8, and a *gate assertion* that
   3 and 8 actually planned more than one block. Without that assertion this
   file would compare three runs of the serial path and prove nothing, which
   is a mistake this project has shipped before.
3. `test_arena_replays_tree_membership_exactly` -- the tree-level statement.
   A tree is grown by the shipped grower, and every split it took is then
   replayed through the arena from the root span. Each leaf's arena window is
   asserted equal, element for element, to the `LeafMembership` row list the
   grower handed back. Since the histograms, the leaf values, and the splits
   are all functions of those lists and of nothing else the arena touches,
   equality of the lists is what makes a grower on the arena grow the same
   tree bit for bit -- and it is a stronger statement than comparing two leaf
   values, because it holds at every internal node and not only at the leaves.
   The leaf values are compared with `to_bits()` on top of it, at each of
   those three worker counts, which is the grower-level determinism check for
   the one change this lane did wire in.
4. `test_fill_identity_rows_matches_serial_loop` -- that change itself.
   `fill_identity_rows` replaced the grower's serial
   `for r in range(n): root_rows[r] = r`, so the list it produces is asserted
   equal to that loop's at 1, 3 and 8 workers, with the same gate assertion.
5. `test_inplace_variant_matches_ping_pong_and_two_list` -- LightGBM's
   ordered copy-back (`DataPartition::Split`, and `gpu_active_rows.
   partition_range_host` for the device) layered on the same partition, so
   both children come back on their parent's side. Asserted to change only
   the address: same bytes, index for index, as both the ping-pong form and
   the two-list form.
6. `test_arena_root_from_bag_copies_in_order`,
   `test_arena_reuse_across_trees_allocates_once`,
   `test_arena_rejects_bad_spans` and `test_arena_empty_span_is_a_no_op` --
   the bagged root, the grow-only buffer contract a booster-scoped arena rests
   on, the range refusals, and the degenerate window.

Negative control, run once by hand and recorded here because the file cannot
assert it: reversing the left side's scatter order inside `_partition_span_into`
-- which leaves both *sets* exactly right and moves only the order -- fails
tests 1, 2 and 3. An order-only regression is what this file exists to catch,
and it does.

The `RowArena` itself is not yet wired into `grow_tree_leaves_profiled`: two
consumers of a node's rows take a whole `List[Int]` and live in files this
lane does not own (see the lane report). There is therefore no conditional
path in the grower to gate-check -- `fill_identity_rows` is unconditional --
and the arena is exercised here directly rather than through a flag that would
have to be proven open.

Shapes are odd against every plausible SIMD width and against the block
geometry, so block boundaries fall in different places relative to the data on
each fixture.
"""

from std.os import setenv
from std.testing import assert_equal, assert_true, TestSuite

from mojotrees.binning import BinnedMatrix, bin_equal_width
from mojotrees.boosting import SQUARED_ERROR, fill_grad_hess
from mojotrees.categorical import cat_add, cat_empty
from mojotrees.cegb import CegbLedger
from mojotrees.parallel import DispatchSettings, plan_row_blocks_with
from mojotrees.split import SplitInfo
from mojotrees.tree import (
    GrowScratch,
    LeafMembership,
    LeafSpan,
    RowArena,
    Tree,
    TreeParams,
    fill_identity_rows,
    grow_tree_leaves,
    partition_arena_span,
    partition_arena_span_inplace,
    partition_rows_into,
)
from support import _uniform


def _workers(n: String):
    _ = setenv("MOJOTREES_NUM_WORKERS", n)


def _auto():
    _ = setenv("MOJOTREES_NUM_WORKERS", "")


def _make_data(
    n_rows: Int, n_features: Int, n_bins: Int, seed: UInt64
) raises -> BinnedMatrix:
    """A binned matrix from the mapper-free path, which reserves no missing
    bin.

    Both missing-bin arms are still covered, and covered more directly:
    `missing_bin` is a plain argument of `partition_rows_into` and of
    `partition_arena_span`, so the routing grid below passes a real bin id for
    the "this feature has one" arm and -1 for the "it has none" arm, against
    the same data. That tests the rule rather than testing `fit_bins`.
    """
    var features = List[Float64](capacity=n_rows * n_features)
    for k in range(n_rows * n_features):
        features.append(_uniform(seed + UInt64(k)))
    return bin_equal_width(features, n_rows, n_features, n_bins)


def _grad_hess(
    n_rows: Int, seed: UInt64
) raises -> Tuple[List[Float64], List[Float64]]:
    var target = List[Float64](capacity=n_rows)
    var raw = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        target.append(_uniform(seed + UInt64(r)) * 4.0 - 2.0)
        raw.append(_uniform(seed + UInt64(7_000_000 + r)) - 0.5)
    var grad = List[Float64]()
    var hess = List[Float64]()
    var weights = List[Float64]()
    fill_grad_hess(raw, target, SQUARED_ERROR, weights, 0.7, grad, hess)
    return (grad^, hess^)


def _split_grid(data: BinnedMatrix) raises -> List[SplitInfo]:
    """One split of every shape `partition_rows_into` documents."""
    var out = List[SplitInfo]()
    # Numerical, threshold in the middle of the range, missing rows left.
    out.append(SplitInfo(0, data.n_bins // 2, 1.0, True, False))
    # Numerical, same threshold, missing rows right.
    out.append(SplitInfo(0, data.n_bins // 2, 1.0, False, False))
    # Numerical, threshold at the bottom, so one side is nearly empty.
    out.append(SplitInfo(1, 0, 1.0, True, False))
    # Numerical, threshold at the top, so the other side is.
    out.append(SplitInfo(1, data.n_bins - 1, 1.0, False, False))
    # Categorical, membership of an odd, non-contiguous bin set.
    var bits = cat_empty()
    for b in range(0, data.n_bins, 3):
        cat_add(bits, b)
    out.append(SplitInfo.categorical(2, 1.0, bits))
    var bits2 = cat_empty()
    for b in range(1, data.n_bins, 5):
        cat_add(bits2, b)
    out.append(SplitInfo.categorical(3, 1.0, bits2))
    return out^


def _missing_for(data: BinnedMatrix, split: SplitInfo, force_none: Bool) -> Int:
    """The `missing_bin` argument for one fixture.

    `force_none` is the "this feature reserves no missing bin" arm, which is
    what -1 means everywhere in this package. Otherwise a real, populated bin
    is named -- one *below* the threshold on some fixtures and above it on
    others -- so the missing rule and the threshold rule disagree and a
    partition that ignored the missing rule would be caught.
    """
    if split.is_categorical or force_none:
        return -1
    return data.n_bins // 3


def _assert_span_equals_list(
    arena: RowArena, span: LeafSpan, want: List[Int]
) raises:
    """The arena window and the `List[Int]` side hold the same row ids in the
    same positions. Integer equality, no tolerance, and the length is checked
    first so a short window cannot pass by vacuity."""
    assert_equal(span.count, len(want))
    for i in range(len(want)):
        assert_equal(arena.row_at(span, i), want[i])


def test_arena_partition_matches_two_list_exactly() raises:
    """The whole design constraint, over the full routing matrix.

    For each split shape, and for both the whole-row span and a nested span
    that is itself the output of a previous partition (so the source lives in
    buffer `b` and not `a`), the arena's two windows equal
    `partition_rows_into`'s two lists index for index.
    """
    var n_rows = 1009
    var n_features = 5
    var data = _make_data(n_rows, n_features, 19, UInt64(4_242))
    var splits = _split_grid(data)

    for forced_none in range(2):
        for s in range(len(splits)):
            var split = splits[s].copy()
            var missing = _missing_for(data, split, forced_none == 1)

            for parallel in range(2):
                if parallel == 0:
                    _workers("1")
                else:
                    _workers("4")
                var settings = DispatchSettings.resolve()

                # --- level 0: the root span, source side 0 -----------------
                var arena = RowArena()
                var root = arena.root_identity(n_rows, settings)
                assert_equal(root.side, 0)
                assert_equal(root.count, n_rows)

                var rows = List[Int]()
                fill_identity_rows(rows, n_rows, settings)

                var want_left = List[Int]()
                var want_right = List[Int]()
                partition_rows_into(
                    want_left, want_right, data, rows, split, missing,
                    settings,
                )
                var got = partition_arena_span(
                    arena, root, data, split, missing, settings
                )

                # Ping-pong: the children live in the other buffer, they
                # partition the parent's window with no gap, and the parent's
                # own window is untouched.
                assert_equal(got.left.side, 1)
                assert_equal(got.right.side, 1)
                assert_equal(got.left.begin, root.begin)
                assert_equal(got.right.begin, got.left.end())
                assert_equal(got.right.end(), root.end())
                _assert_span_equals_list(arena, got.left, want_left)
                _assert_span_equals_list(arena, got.right, want_right)
                for i in range(n_rows):
                    assert_equal(arena.row_at(root, i), i)

                # --- level 1: a nested span, source side 1 -----------------
                # The left child is re-split by the *next* shape in the grid,
                # so the second partition is not a repeat of the first and its
                # window does not start at zero when the right child is taken.
                var next_split = splits[(s + 1) % len(splits)].copy()
                var next_missing = _missing_for(
                    data, next_split, forced_none == 1
                )
                for which in range(2):
                    var parent = (
                        got.left.copy() if which == 0 else got.right.copy()
                    )
                    var parent_rows = arena.span_rows(parent)
                    var sub_left = List[Int]()
                    var sub_right = List[Int]()
                    partition_rows_into(
                        sub_left, sub_right, data, parent_rows, next_split,
                        next_missing, settings,
                    )
                    var sub = partition_arena_span(
                        arena, parent, data, next_split, next_missing,
                        settings,
                    )
                    assert_equal(sub.left.side, 0)
                    assert_equal(sub.left.begin, parent.begin)
                    assert_equal(sub.right.begin, sub.left.end())
                    assert_equal(sub.right.end(), parent.end())
                    _assert_span_equals_list(arena, sub.left, sub_left)
                    _assert_span_equals_list(arena, sub.right, sub_right)
                    # The parent window in buffer 1 still reads as it did.
                    _assert_span_equals_list(arena, parent, parent_rows)
    _auto()


def test_arena_partition_deterministic_across_workers() raises:
    """Identical arena contents at 1, 3 and 8 workers -- and proof that 3 and
    8 planned more than one block, so this is not three runs of the serial
    path agreeing with themselves."""
    var n_rows = 2003
    var data = _make_data(n_rows, 4, 23, UInt64(9_001))
    var split = SplitInfo(0, data.n_bins // 2, 1.0, True, True)
    var missing = data.n_bins // 3

    var counts = ["1", "3", "8"]
    var reference = List[Int]()
    var ref_n_left = -1
    for k in range(len(counts)):
        _workers(counts[k])
        var settings = DispatchSettings.resolve()
        var blocks = plan_row_blocks_with(settings, n_rows, 3 * n_rows)
        if k == 0:
            assert_equal(blocks.n_blocks, 1)
        else:
            # The gate. Without this the loop below compares three serial
            # runs and establishes nothing about the parallel scatter.
            assert_true(blocks.n_blocks > 1)

        var arena = RowArena()
        var root = arena.root_identity(n_rows, settings)
        var got = partition_arena_span(
            arena, root, data, split, missing, settings
        )
        var flat = List[Int](capacity=n_rows)
        for i in range(got.left.count):
            flat.append(arena.row_at(got.left, i))
        for i in range(got.right.count):
            flat.append(arena.row_at(got.right, i))
        assert_equal(len(flat), n_rows)
        if k == 0:
            reference = flat^
            ref_n_left = got.left.count
        else:
            assert_equal(got.left.count, ref_n_left)
            for i in range(n_rows):
                assert_equal(flat[i], reference[i])
    _auto()


def _replay_tree_through_arena(
    mut arena: RowArena,
    tree: Tree,
    data: BinnedMatrix,
    settings: DispatchSettings,
) raises -> Tuple[List[Int], List[LeafSpan]]:
    """Walk `tree`'s nodes in id order, partitioning the arena at every
    internal node exactly as growth did, and return the leaf node ids beside
    their spans.

    Growth appends both children immediately after their parent's split is
    recorded, so node ids increase down the tree and one ascending pass
    reproduces every partition in an order consistent with the one growth
    used. The order the *splits* were taken in does not matter here: each
    node's span depends only on its parent's span and its parent's split, and
    the arena's windows for distinct nodes are disjoint.

    The matrix this file grows on is numerical throughout, so every internal
    node carries a threshold split; categorical routing is covered directly in
    `test_arena_partition_matches_two_list_exactly`, against the same
    `SplitInfo.goes_left` the grower uses.
    """
    var n_nodes = len(tree.feature)
    var spans = List[LeafSpan]()
    for _ in range(n_nodes):
        spans.append(LeafSpan(0, 0, 0))
    spans[0] = arena.root_identity(data.n_rows, settings)

    var leaf_nodes = List[Int]()
    var leaf_spans = List[LeafSpan]()
    for node in range(n_nodes):
        if tree.feature[node] < 0:
            leaf_nodes.append(node)
            leaf_spans.append(spans[node].copy())
            continue
        var split = SplitInfo(
            tree.feature[node],
            tree.threshold_bin[node],
            0.0,
            True,
            tree.default_left[node],
        )
        var missing = tree.missing_bin[node]
        var part = partition_arena_span(
            arena, spans[node].copy(), data, split, missing, settings
        )
        spans[tree.left[node]] = part.left.copy()
        spans[tree.right[node]] = part.right.copy()
    return (leaf_nodes^, leaf_spans^)


def test_arena_replays_tree_membership_exactly() raises:
    """A grown tree's leaf membership, reproduced by the arena element for
    element, and its leaf values bit-identical across worker counts.

    This is the statement that matters: `LeafMembership.rows[l]` is exactly
    the list every histogram beneath leaf `l` was accumulated from, in the
    order it was accumulated in. If the arena reproduces those lists position
    for position at every leaf, then a grower driven from the arena
    accumulates the same `Float64` in the same order at every node, and the
    tree it grows is the tree here bit for bit.
    """
    var n_rows = 1531
    var n_features = 6
    var data = _make_data(n_rows, n_features, 21, UInt64(31_337))
    var gh = _grad_hess(n_rows, UInt64(555))
    var grad = gh[0].copy()
    var hess = gh[1].copy()

    var params = TreeParams(12, 5, 1.0, 1e-3)

    var counts = ["1", "3", "8"]
    var ref_bits = List[UInt64]()
    for k in range(len(counts)):
        _workers(counts[k])
        var settings = DispatchSettings.resolve()

        var leaves = LeafMembership()
        var ledger = CegbLedger.none()
        var scratch = GrowScratch(data.n_features, data.n_bins)
        var tree = grow_tree_leaves(
            leaves, ledger, scratch, data, grad, hess, params
        )
        assert_true(leaves.n_leaves() > 1)
        assert_true(leaves.covers_all_rows)
        # The replay reconstructs threshold splits, so a categorical node
        # would make it compare the wrong routing rule and still pass by
        # coincidence on some fixtures. There are none on this matrix, and
        # this is the assertion that says so rather than assuming it.
        for node in range(len(tree.feature)):
            assert_true(tree.cat_offset[node] < 0)

        var arena = RowArena()
        var replay = _replay_tree_through_arena(arena, tree, data, settings)
        var replay_nodes = replay[0].copy()
        var replay_spans = replay[1].copy()
        assert_equal(len(replay_nodes), leaves.n_leaves())

        # The grower hands leaves back in frontier order, which is not node
        # order, so each membership entry is matched to the replay by node id.
        var total = 0
        for l in range(leaves.n_leaves()):
            var node = leaves.node[l]
            var found = -1
            for j in range(len(replay_nodes)):
                if replay_nodes[j] == node:
                    found = j
            assert_true(found >= 0)
            _assert_span_equals_list(
                arena, replay_spans[found], leaves.rows[l]
            )
            total += len(leaves.rows[l])
        assert_equal(total, n_rows)

        # And the tree itself, bit for bit, at every worker count.
        var bits = List[UInt64]()
        for node in range(len(tree.value)):
            bits.append(UInt64(tree.value[node].to_bits()))
        if k == 0:
            ref_bits = bits^
        else:
            assert_equal(len(bits), len(ref_bits))
            for i in range(len(bits)):
                assert_equal(bits[i], ref_bits[i])
    _auto()


def test_fill_identity_rows_matches_serial_loop() raises:
    """The one grower change this lane wired: the root list's serial fill.

    Asserted against the loop it replaced, at 1, 3 and 8 workers, with the
    gate assertion that 3 and 8 planned more than one block. An empty and a
    one-row fill are included because a block plan degenerates at both.
    """
    var sizes = [0, 1, 2, 63, 1009, 4001]
    var counts = ["1", "3", "8"]
    for k in range(len(counts)):
        _workers(counts[k])
        var settings = DispatchSettings.resolve()
        for s in range(len(sizes)):
            var n = sizes[s]
            var rows = List[Int]()
            fill_identity_rows(rows, n, settings)
            assert_equal(len(rows), n)
            for r in range(n):
                assert_equal(rows[r], r)
            # Refilling a list that already has contents overwrites all of
            # them, which is what a caller reusing a buffer relies on.
            fill_identity_rows(rows, n, settings)
            assert_equal(len(rows), n)
            for r in range(n):
                assert_equal(rows[r], r)
        # The gate, on the largest size only: the small ones are meant to stay
        # serial and asserting otherwise would be asserting the dispatch rule.
        var blocks = plan_row_blocks_with(settings, 4001, 4001)
        if k == 0:
            assert_equal(blocks.n_blocks, 1)
        else:
            assert_true(blocks.n_blocks > 1)
    _auto()

    # Negative counts are refused rather than resizing to garbage.
    var raised = False
    try:
        var bad = List[Int]()
        fill_identity_rows(bad, -1)
    except:
        raised = True
    assert_true(raised)


def test_inplace_variant_matches_ping_pong_and_two_list() raises:
    """LightGBM's ordered copy-back, and that it changes only the address.

    `partition_arena_span_inplace` is the single-canonical-array contract --
    LightGBM's `DataPartition::Split`, and `gpu_active_rows.
    partition_range_host` -- layered on the same partition. Both children come
    back on their parent's side, and the bytes are what the ping-pong form
    produced and what `partition_rows_into` produced, index for index. Run at
    both worker settings and to two levels, so the copy-back is exercised from
    each buffer.
    """
    var n_rows = 1301
    var data = _make_data(n_rows, 5, 17, UInt64(60_601))
    var splits = _split_grid(data)

    for parallel in range(2):
        if parallel == 0:
            _workers("1")
        else:
            _workers("4")
        var settings = DispatchSettings.resolve()
        for s in range(len(splits)):
            var split = splits[s].copy()
            var missing = _missing_for(data, split, False)

            var rows = List[Int]()
            fill_identity_rows(rows, n_rows, settings)
            var want_left = List[Int]()
            var want_right = List[Int]()
            partition_rows_into(
                want_left, want_right, data, rows, split, missing, settings
            )

            var arena = RowArena()
            var root = arena.root_identity(n_rows, settings)
            var got = partition_arena_span_inplace(
                arena, root, data, split, missing, settings
            )
            # Same side as the parent, which is the whole point of the copy.
            assert_equal(got.left.side, root.side)
            assert_equal(got.right.side, root.side)
            assert_equal(got.left.begin, root.begin)
            assert_equal(got.right.begin, got.left.end())
            assert_equal(got.right.end(), root.end())
            _assert_span_equals_list(arena, got.left, want_left)
            _assert_span_equals_list(arena, got.right, want_right)

            # A second level, so the copy-back runs from buffer `b` too --
            # the ping-pong writes into `b` and copies back into `a`, so the
            # source of the copy alternates even though the spans do not.
            var next_split = splits[(s + 1) % len(splits)].copy()
            var next_missing = _missing_for(data, next_split, False)
            var parent_rows = arena.span_rows(got.left)
            var sub_left = List[Int]()
            var sub_right = List[Int]()
            partition_rows_into(
                sub_left, sub_right, data, parent_rows, next_split,
                next_missing, settings,
            )
            var sub = partition_arena_span_inplace(
                arena, got.left.copy(), data, next_split, next_missing,
                settings,
            )
            assert_equal(sub.left.side, got.left.side)
            _assert_span_equals_list(arena, sub.left, sub_left)
            _assert_span_equals_list(arena, sub.right, sub_right)
            # The right sibling's window is outside the copied range and must
            # be untouched by it.
            _assert_span_equals_list(arena, got.right, want_right)
    _auto()


def test_arena_root_from_bag_copies_in_order() raises:
    """The bagged root. `sampling.check_row_set` has already made the bag
    ascending and duplicate-free; this copies it position for position, so the
    arena root is the bag in the bag's own order."""
    var n_rows = 997
    var bag = List[Int]()
    for r in range(1, n_rows, 3):
        bag.append(r)
    _workers("4")
    var settings = DispatchSettings.resolve()
    var arena = RowArena()
    var root = arena.root_from_bag(bag, settings)
    assert_equal(root.count, len(bag))
    assert_equal(root.side, 0)
    for i in range(len(bag)):
        assert_equal(arena.row_at(root, i), bag[i])
    var back = arena.span_rows(root)
    assert_equal(len(back), len(bag))
    for i in range(len(bag)):
        assert_equal(back[i], bag[i])
    _auto()


def test_arena_reuse_across_trees_allocates_once() raises:
    """An arena reused for a second, smaller root keeps its buffers.

    The property a booster-scoped arena rests on: `ensure` grows and never
    shrinks, so the second tree of a fit finds the memory the first one
    allocated. Checked through the buffer lengths, which is the only
    observable of an allocation this file can reach without a timer.
    """
    var arena = RowArena()
    _ = arena.root_identity(4096)
    assert_equal(len(arena.a), 4096)
    assert_equal(len(arena.b), 4096)
    var small = arena.root_identity(100)
    assert_equal(arena.n, 100)
    assert_equal(len(arena.a), 4096)
    assert_equal(len(arena.b), 4096)
    assert_equal(small.count, 100)
    for i in range(100):
        assert_equal(arena.row_at(small, i), i)
    var bigger = arena.root_identity(8192)
    assert_equal(len(arena.a), 8192)
    assert_equal(len(arena.b), 8192)
    assert_equal(bigger.count, 8192)
    for i in range(8192):
        assert_equal(arena.row_at(bigger, i), i)


def test_arena_rejects_bad_spans() raises:
    """Range refusals, so a span that escapes its buffer is named rather than
    written past."""
    var data = _make_data(101, 3, 11, UInt64(77))
    var split = SplitInfo(0, 5, 1.0, True, False)
    var arena = RowArena()
    var root = arena.root_identity(101)

    var cases = [
        LeafSpan(-1, 10, 0),
        LeafSpan(0, -1, 0),
        LeafSpan(50, 100, 0),
        LeafSpan(0, 101, 2),
    ]
    for c in range(len(cases)):
        var raised = False
        try:
            _ = partition_arena_span(
                arena, cases[c].copy(), data, split, data.missing_bin[0]
            )
        except:
            raised = True
        assert_true(raised)

    # A feature outside the matrix is refused before any bin is read.
    var raised2 = False
    try:
        _ = partition_arena_span(
            arena,
            root.copy(),
            data,
            SplitInfo(data.n_features, 5, 1.0, True, False),
            -1,
        )
    except:
        raised2 = True
    assert_true(raised2)

    # An arena is refused a row count it cannot address in Int32.
    var raised3 = False
    try:
        var big = RowArena()
        big.ensure(2147483648)
    except:
        raised3 = True
    assert_true(raised3)


def test_arena_empty_span_is_a_no_op() raises:
    """A zero-row span partitions into two zero-row spans on the other side,
    and touches nothing. Growth cannot produce one (a gain-chosen split has
    `min_data_in_leaf` on both sides), but the range arithmetic is easier to
    trust when the degenerate case is written down."""
    var data = _make_data(64, 3, 9, UInt64(5))
    var arena = RowArena()
    var root = arena.root_identity(64)
    var empty = LeafSpan(64, 0, 0)
    var got = partition_arena_span(
        arena, empty, data, SplitInfo(0, 4, 1.0, True, False), -1
    )
    assert_equal(got.left.count, 0)
    assert_equal(got.right.count, 0)
    assert_equal(got.left.side, 1)
    assert_equal(got.left.begin, 64)
    assert_equal(got.right.begin, 64)
    for i in range(64):
        assert_equal(arena.row_at(root, i), i)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
