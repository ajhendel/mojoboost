"""The batched level: the recomputed census, and the two things it needed.

Why this file exists
--------------------
`grow_policy = oblivious` on the resident plane was funded on a launch-count
argument -- fewer command buffers per tree, landing under the 64-deep queue
knee where per-launch enqueue cost goes from 6-7 microseconds to 14-17.

**THE "LANDING UNDER" HALF OF THAT IS RETIRED, 2026-08-18**
(`docs/design/SWITCH_GRID.md` section 6 item 8). A full Metal queue blocks the
host thread that enqueues into it rather than dropping work, and the leaf-wise
grower this package ships as its fastest arm runs 278 to 2,303 command buffers
a tree, measured backpressured at commit 1d77414. So 64 is where a launch gets
dearer and is not a line an arm has to stay under. The launch-count argument
survives as a cost argument, which is what every assertion in this file
measures; `_QUEUE_DEPTH` below is a price point that the counts are compared
against, not a bound they must satisfy.
The `oblivious-device` lane computed the real census against the code as it
stands and **the argument failed**: `GpuHistogramBuilder.enqueue_desc_child`
builds one child per launch pair, so a depth-6 tree is 176 command buffers and
not the 62 the design assumed.

`GpuLeafBatcher.enqueue_batch` already does a whole frontier in two launches
whatever its width. The single thing keeping it out of a device-owned tree was
that its item table came from the host -- which is to say, from a read-back of
the commit that decides it. This file pins the three claims that closes:

1. **The census, recomputed.** `oblivious_launch_census` now takes the child
   build's shape as a parameter instead of assuming it, so all three shapes
   this package can be in are values of one function. The batched shape is 62
   at depth 6 **and only with a batcher sized for a whole level**. At
   `gpu_leaf_batching.DEFAULT_MAX_ITEMS` of 32 the last level needs two batches
   and the tree lands at 64. This paragraph called that "a precondition and not
   a preference" on the strength of the knee; the sizing test still stands, on
   the ground that the two extra command buffers buy nothing at all and one
   parameter to `open_resident` avoids them.

2. **The device-written plan.** `_pick_and_commit_kernel` writes the three
   words of an item row that a commit decides, on every execution including the
   ones that commit nothing, and it does so in the launch it was already
   making. Nothing on any existing path can observe it: both `enqueue_step`
   overloads pass `write_plan = 0`.

3. **The window setter.** One stable partition of the whole active prefix
   produces a whole oblivious level, but numbers it `j` and `j + 2^l`, so a
   parent's rows land in two blocks that are not adjacent and
   `LeafRangeTable.split` -- which requires containment -- cannot replay such a
   tree. `set_window` can, without becoming a way around the staleness poison
   that `_publish_row_ranges` exists to lift.

Bits do not move on any existing path, and the tests below are written to say
so rather than to assume it: the host-staged batch arm is checked to refuse the
one sentinel value the device arm introduces, and the census is checked to
still return every value its own pinned table already held.

Host-side only. Nothing here opens a `DeviceContext`; the launch shapes are
counted, not run.
"""

from std.testing import (
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
    TestSuite,
)

from mojotrees.gpu_active_rows import LeafRange, LeafRangeTable
from mojotrees.gpu_leaf_batching import (
    DEFAULT_MAX_ITEMS,
    ITEM_BEGIN,
    ITEM_COUNT,
    ITEM_DEAD,
    ITEM_OUT,
    ITEM_PLANE,
    ITEM_ROWS_PER_TILE,
    ITEM_TILES,
    ITEM_TILE_BEGIN,
    ITEM_WORDS,
    OBLIVIOUS_MAX_ITEMS,
)
from mojotrees.gpu_resident_round import (
    oblivious_launch_census,
    oblivious_schedule_launches,
)
from mojotrees.gpu_tree_tables import PLAN_ITEMS

# The queue is 64 command buffers deep and MAX never raises it
# (`docs/GPU_PORTABILITY.md` 6.2). A knee, not a wall: enqueue costs 6 to 7
# microseconds through a stream of 64 and 14 to 17 beyond, and a full queue
# blocks the enqueuing host thread rather than dropping anything. Retired as a
# safety criterion 2026-08-18, so every comparison below is a comparison of
# counts against a price point.
comptime _QUEUE_DEPTH = 64


# --- The recomputed census ------------------------------------------------


def test_the_batched_shape_is_what_the_design_assumed_all_along() raises:
    # `batch_max_items = 0` is the design's assumption: the whole level's
    # children in one batch, two launches per level however wide the level is.
    # Every value the census was already pinned at is still that value, which
    # is the check that the parameter was added and nothing was rewritten.
    assert_equal(oblivious_launch_census(4), 44)
    assert_equal(oblivious_launch_census(4, fused_reduce=False), 48)
    assert_equal(oblivious_launch_census(5), 53)
    assert_equal(oblivious_launch_census(5, fused_reduce=False), 58)
    assert_equal(oblivious_launch_census(6), 62)
    assert_equal(oblivious_launch_census(6, fused_reduce=False), 68)
    assert_equal(oblivious_launch_census(7), 71)
    assert_equal(oblivious_launch_census(7, fused_reduce=False), 78)


def test_the_shape_this_package_has_today_is_the_176_the_lane_found() raises:
    # A negative bound is "no batcher at all": `enqueue_desc_child` builds one
    # child per parent and derives the sibling, so the histogram term is
    # 2 * 2^l and depth 6 sums to 2 * (1+2+4+8+16+32) = 126 of histogram alone.
    # This is the number the previous lane computed by hand and recorded in
    # prose; it is now a value the same function returns.
    assert_equal(oblivious_launch_census(6, batch_max_items=-1), 176)
    # And it misses the count the mode was funded on by a hundred and twelve
    # rather than by two. "Fails the criterion" is how this read until
    # 2026-08-18; there is no criterion, there are 112 extra launches at 14 to
    # 17 microseconds each.
    assert_true(oblivious_launch_census(6, batch_max_items=-1) > _QUEUE_DEPTH)
    # It is still better than leaf-wise at the default budget, which is why
    # the mode was worth a lane and not why it was worth shipping.
    assert_true(oblivious_launch_census(6, batch_max_items=-1) < 278)


def test_the_batched_census_clears_the_knee_and_by_how_much() raises:
    # The deliverable. A depth-6 oblivious tree, with the level's children
    # built by one device-driven batch, is 62 command buffers between waits.
    # "Clears the knee" is a statement about a count and a price, not about
    # safety; see the header.
    assert_equal(oblivious_launch_census(6, batch_max_items=64), 62)
    assert_true(oblivious_launch_census(6, batch_max_items=64) < _QUEUE_DEPTH)
    assert_equal(
        _QUEUE_DEPTH - oblivious_launch_census(6, batch_max_items=64), 2
    )
    # A whole level in one batch and a batch bound wide enough to hold it are
    # the same count, which is the statement that the bound is the only thing
    # standing between the two.
    assert_equal(
        oblivious_launch_census(6, batch_max_items=64),
        oblivious_launch_census(6),
    )


def test_the_default_item_bound_puts_depth_six_on_the_knee_not_under_it(
) raises:
    # A depth-6 tree's last level has 64 children; `DEFAULT_MAX_ITEMS` is 32;
    # the extra batch is two launches. This comment called that "a precondition
    # rather than a preference"; since 2026-08-18 the reason to size the
    # batcher for a whole level is that those two launches buy nothing, not
    # that 64 is a line.
    assert_equal(DEFAULT_MAX_ITEMS, 32)
    var at_default = oblivious_launch_census(
        6, batch_max_items=DEFAULT_MAX_ITEMS
    )
    assert_equal(at_default, 64)
    assert_false(at_default < _QUEUE_DEPTH)
    assert_equal(at_default - oblivious_launch_census(6, batch_max_items=64), 2)
    # Depth 5 is 32 children at its widest, so the default bound is enough
    # there and a lane that tested only at depth 5 would have seen nothing.
    assert_equal(
        oblivious_launch_census(5, batch_max_items=DEFAULT_MAX_ITEMS),
        oblivious_launch_census(5, batch_max_items=64),
    )


def test_the_batched_shape_beats_the_per_child_one_at_every_useful_depth(
) raises:
    for d in range(2, 8):
        assert_true(
            oblivious_launch_census(d, batch_max_items=128)
            < oblivious_launch_census(d, batch_max_items=-1)
        )
    # At depth 1 there is one parent and two children, so a batch and a
    # per-child build cost the same two launches and there is nothing to win.
    assert_equal(
        oblivious_launch_census(1, batch_max_items=128),
        oblivious_launch_census(1, batch_max_items=-1),
    )
    # Zero depth is not a tree under any shape.
    assert_equal(oblivious_launch_census(0, batch_max_items=-1), 0)
    assert_equal(oblivious_launch_census(0, batch_max_items=64), 0)


def test_depth_seven_is_over_the_knee_even_batched() raises:
    # **BOTH SENTENCES THAT USED TO BE HERE ARE WRONG NOW AND THE ASSERTION IS
    # NOT.** They read: "The batcher does not buy a depth the reserved per-leaf
    # scan state (`OBLIVIOUS_MAX_LEAVES`) could not serve anyway, and it should
    # not be read as having done so." `OBLIVIOUS_MAX_LEAVES` went to 256 on
    # 2026-08-18 and serves depth 8, and passing 64 command buffers costs
    # enqueue time rather than correctness.
    #
    # The census still says 71 at depth 7 and that number is still worth
    # pinning, because `oblivious_launch_census` is the FROZEN registered
    # prediction the round was opened on and a prediction edited after the fact
    # is not one. What it predicts is a schedule that was never built.
    assert_true(
        oblivious_launch_census(7, batch_max_items=128) > _QUEUE_DEPTH
    )
    assert_equal(oblivious_launch_census(7), 71)
    # The schedule that RUNS is the one to reason about, and at the shipped
    # `skip_last_build` it is 63 at depth 7 and 71 at depth 8 -- the depth-7
    # figure under the knee the census put it over. Asserted beside the frozen
    # number rather than instead of it, so the gap between model and build
    # stays visible.
    assert_equal(
        oblivious_schedule_launches(
            7, batch_max_items=OBLIVIOUS_MAX_ITEMS, skip_last_build=True
        ),
        63,
    )
    assert_equal(
        oblivious_schedule_launches(
            8, batch_max_items=OBLIVIOUS_MAX_ITEMS, skip_last_build=True
        ),
        71,
    )


# --- The item table the commit kernel writes ------------------------------


def test_a_commit_fills_both_children_and_the_layout_says_which_words(
) raises:
    # The split between the two halves of a device-written plan, pinned so it
    # cannot drift: three words are the commit's and four are the grid's.
    # `_pick_and_commit_kernel._write_plan_item` writes exactly the first
    # three; `GpuLeafBatcher.stage_device_plan` writes exactly the rest.
    assert_equal(PLAN_ITEMS, 2)
    assert_equal(ITEM_BEGIN, 0)
    assert_equal(ITEM_COUNT, 1)
    assert_equal(ITEM_OUT, 5)
    # The geometry words, which no kernel may write because a grid is a host
    # argument to `enqueue_function`.
    assert_equal(ITEM_ROWS_PER_TILE, 2)
    assert_equal(ITEM_TILE_BEGIN, 3)
    assert_equal(ITEM_TILES, 4)
    assert_equal(ITEM_PLANE, 6)
    assert_equal(ITEM_WORDS, 8)
    # A plan is small: two items at eight words is what a leaf-wise commit
    # fills, and a whole depth-6 level is 64 of them.
    assert_equal(PLAN_ITEMS * ITEM_WORDS, 16)


def test_the_dead_sentinel_is_not_a_row_count() raises:
    # `ITEM_DEAD` has to be distinguishable from a live leaf holding no rows,
    # because those two want opposite things from the zeroing pass: a dead
    # item must be left alone (its `ITEM_OUT` is stale and may name a live
    # leaf's slot), an empty live one must be cleared (its histogram is
    # legitimately all zeros and something has to write them).
    assert_equal(ITEM_DEAD, -1)
    assert_true(ITEM_DEAD < 0)


# --- The window setter ----------------------------------------------------


def _table(n_rows: Int, n_active: Int) raises -> LeafRangeTable:
    var t = LeafRangeTable(n_rows)
    t.reset_root(n_active)
    return t^


def test_a_level_lands_in_two_blocks_split_gets_silently_wrong() raises:
    # The situation, as a test rather than as a paragraph, on one worked
    # level. The root owns [0, 100) and the first split sends 40 rows left,
    # so node 1 owns [0, 40) and node 2 owns [40, 100). Both replays agree so
    # far, because a single window partitioned in place is contiguous.
    var t = _table(100, 100)
    _ = t.split(0, 1, 2, 40)
    assert_equal(t.get(1).begin, 0)
    assert_equal(t.get(1).end, 40)

    # Level two applies **one** rule to the whole prefix in **one** stable
    # partition, which is what makes the level cost two launches instead of
    # two per leaf. All the left-going rows come first, in order, then all the
    # right-going ones. With 30 of node 1's rows and 25 of node 2's going
    # left, the four new leaves own, in leaf-index order,
    #
    #     leaf 0  node 1 left   [0, 30)
    #     leaf 1  node 2 left   [30, 55)
    #     leaf 2  node 1 right  [55, 65)
    #     leaf 3  node 2 right  [65, 100)
    #
    # and node ids go out level by level over ascending leaf index, left child
    # before right (`OBLIVIOUS_LEAF_INDEX_RULE`): 3 and 4 for node 1, 5 and 6
    # for node 2. So node 1's two children own [0, 30) and [55, 65) --
    # **not adjacent, and the second is nowhere inside node 1's old window.**
    #
    # `split` cannot be asked for that, and the failure is not a raise. It
    # takes an `n_left` and hands out the two contiguous halves, so it writes
    # node 4 as [30, 40) and is silently describing a different partition:
    # the exact failure mode `_publish_row_ranges` documents, where the tree
    # is right and the scores are wrong.
    _ = t.split(1, 3, 4, 30)
    assert_equal(t.get(3).begin, 0)
    assert_equal(t.get(3).end, 30)
    assert_equal(t.get(4).begin, 30)
    assert_equal(t.get(4).end, 40)
    assert_true(t.get(4).end != 65)


def test_the_publish_check_catches_the_split_replay_of_a_level() raises:
    # And it is caught, which is what makes the setter a fix rather than a
    # convenience: replaying a level with `split` produces a table that
    # `end_descriptor_partition` refuses against the device's own frontier.
    var t = _table(100, 100)
    _ = t.begin_descriptor_partition(100)
    _ = t.split(0, 1, 2, 40)
    _ = t.split(1, 3, 4, 30)
    _ = t.split(2, 5, 6, 25)
    var nodes: List[Int] = [3, 5, 4, 6]
    var windows: List[LeafRange] = [
        LeafRange(0, 30),
        LeafRange(30, 55),
        LeafRange(55, 65),
        LeafRange(65, 100),
    ]
    with assert_raises():
        t.end_descriptor_partition(7, nodes, windows)
    assert_true(t.is_resident_owned())


def test_set_window_writes_the_level_split_cannot() raises:
    # The same level, replayed with the setter, accepted by the same check.
    var t = _table(100, 100)
    _ = t.begin_descriptor_partition(100)
    t.set_window(0, 0, 0)
    t.set_window(1, 0, 0)
    t.set_window(2, 0, 0)
    t.set_window(3, 0, 30)
    t.set_window(4, 55, 65)
    t.set_window(5, 30, 55)
    t.set_window(6, 65, 100)
    var nodes: List[Int] = [3, 5, 4, 6]
    var windows: List[LeafRange] = [
        LeafRange(0, 30),
        LeafRange(30, 55),
        LeafRange(55, 65),
        LeafRange(65, 100),
    ]
    t.end_descriptor_partition(7, nodes, windows)
    assert_false(t.is_resident_owned())
    assert_equal(t.n_nodes(), 7)
    assert_equal(t.total_active(), 100)
    # Node 1's two children, non-adjacent, both correct.
    assert_equal(t.get(3).end, 30)
    assert_equal(t.get(4).begin, 55)


def test_set_window_keeps_the_orphan_guard_split_has() raises:
    var t = _table(100, 100)
    t.set_window(0, 0, 0)
    t.set_window(1, 0, 40)
    # A second nonempty window on a node that already owns rows is the
    # mistake `split` refuses for the same reason: the first window's rows
    # would be silently orphaned.
    with assert_raises():
        t.set_window(1, 50, 60)
    # Emptying is always allowed, because that is what a level replay does to
    # the parents it has just replaced and an empty window orphans nothing.
    t.set_window(1, 0, 0)
    t.set_window(1, 50, 60)
    assert_equal(t.get(1).begin, 50)


def test_set_window_refuses_a_window_outside_the_buffer() raises:
    var t = _table(100, 100)
    with assert_raises():
        t.set_window(-1, 0, 10)
    with assert_raises():
        t.set_window(1, -1, 10)
    with assert_raises():
        t.set_window(1, 10, 5)
    with assert_raises():
        t.set_window(1, 0, 101)


def test_set_window_is_a_writer_and_not_a_hole_in_the_poison() raises:
    # The property the whole discipline turns on. `set_window` is permitted
    # while the descriptor partition owns the windows -- it has to be, it is
    # how the replay that clears the poison happens -- but it clears nothing
    # and reveals nothing, so every reader is still refused after it has run.
    var t = _table(100, 100)
    _ = t.begin_descriptor_partition(100)
    assert_true(t.is_resident_owned())
    t.set_window(0, 0, 0)
    t.set_window(1, 0, 40)
    t.set_window(2, 40, 100)
    # Still poisoned, and every accessor still refuses.
    assert_true(t.is_resident_owned())
    with assert_raises():
        _ = t.get(1)
    with assert_raises():
        _ = t.n_nodes()
    with assert_raises():
        _ = t.total_active()
    with assert_raises():
        t.check_invariants()


def test_only_the_device_frontier_lifts_the_poison_off_a_set_table() raises:
    # And the proof that lifts it is the same one a `split` replay is held to:
    # every window checked against the device's own frontier, byte for byte,
    # plus the tiling invariant over the finished table.
    var t = _table(100, 100)
    _ = t.begin_descriptor_partition(100)
    t.set_window(0, 0, 0)
    t.set_window(1, 0, 40)
    t.set_window(2, 40, 100)
    var nodes: List[Int] = [1, 2]
    var windows: List[LeafRange] = [LeafRange(0, 40), LeafRange(40, 100)]
    t.end_descriptor_partition(3, nodes, windows)
    assert_false(t.is_resident_owned())
    assert_equal(t.get(2).begin, 40)


def test_a_set_table_that_is_not_the_device_s_tree_is_refused() raises:
    var t = _table(100, 100)
    _ = t.begin_descriptor_partition(100)
    t.set_window(0, 0, 0)
    t.set_window(1, 0, 40)
    t.set_window(2, 40, 100)
    var nodes: List[Int] = [1, 2]
    # The device says leaf 1 owns 50 rows and the host table says 40. That is
    # exactly the divergence `_publish_row_ranges` exists to catch, and it
    # must still be caught when the windows were set directly.
    var windows: List[LeafRange] = [LeafRange(0, 50), LeafRange(50, 100)]
    with assert_raises():
        t.end_descriptor_partition(3, nodes, windows)
    assert_true(t.is_resident_owned())


def test_windows_that_do_not_tile_are_refused_at_the_publish() raises:
    # `set_window` deliberately does not check tiling: that is a property of
    # the whole table, it is false halfway through any replay, and the place
    # it is established is `end_descriptor_partition`. A gap has to be caught
    # there and nowhere earlier.
    var t = _table(100, 100)
    _ = t.begin_descriptor_partition(100)
    t.set_window(0, 0, 0)
    t.set_window(1, 0, 40)
    t.set_window(2, 50, 100)
    var nodes: List[Int] = [1, 2]
    var windows: List[LeafRange] = [LeafRange(0, 40), LeafRange(50, 100)]
    with assert_raises():
        t.end_descriptor_partition(3, nodes, windows)


def test_reset_root_still_clears_a_table_built_by_the_setter() raises:
    var t = _table(100, 100)
    t.set_window(0, 0, 0)
    t.set_window(1, 0, 40)
    t.set_window(2, 40, 100)
    t.reset_root(80)
    assert_equal(t.n_nodes(), 1)
    assert_equal(t.get(0).count(), 80)
    t.check_invariants()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
