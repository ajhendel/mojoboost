"""`grow_policy = oblivious` on the device: the launch census, and the
cross-leaf reduction that census makes mandatory.

Two things are asserted here and they answer two different questions.

**The census.** `docs/design/OBLIVIOUS.md` B5 registers one kill criterion
ahead of any number: unless a level schedule brings the per-tree command-buffer
count under 64 -- the measured queue-depth knee -- the queue-depth argument for
oblivious evaporates and only the accuracy question is left. The arithmetic is
`gpu_resident_round.oblivious_launch_census` and the tests below pin it to the
table the round was opened on, so that "62 fused, 68 standalone, at CatBoost's
own default depth of 6" is a value something checks rather than a sentence in a
docstring. They also pin the refusals, because the census found two things the
design assumed and the code does not provide, and a refusal nothing names is a
refusal that gets forgotten.

**The reduction.** `gpu_split_search._scan_slot_oblivious_kernel` is what makes
the fused arm possible: the sum over a level's leaves is the innermost loop of
the scan that already runs, so a level search is two launches exactly as a node
search is. It is checked against a host replica that mirrors it statement for
statement, and against the ordinary node search on the degenerate level of one
leaf, where the two must agree exactly and not merely closely.

Bits move only in the new mode. Nothing here touches leaf-wise or depth-wise
growth, and no golden fixture changes.

The device tests skip (passing) with no accelerator, so the census layer still
runs on a CPU-only machine.
"""

from std.os import setenv
from std.sys import has_accelerator
from std.testing import (
    assert_equal,
    assert_false,
    assert_true,
    TestSuite,
)

from mojotrees.binning import bin_equal_width, BinnedMatrix
from mojotrees.categorical import CategoricalParams, CategoricalSpec
from mojotrees.growth_policy import GROW_LEAFWISE, GROW_OBLIVIOUS
from mojotrees.gpu_leaf_batching import DEFAULT_MAX_ITEMS, OBLIVIOUS_MAX_ITEMS
from mojotrees.gpu_resident_round import (
    OBLIVIOUS_CATEGORICAL,
    OBLIVIOUS_DEPTH,
    OBLIVIOUS_LEAF_INDEX_RULE,
    OBLIVIOUS_LEVEL_HISTOGRAM,
    OBLIVIOUS_LEVEL_LAUNCHES,
    OBLIVIOUS_NO_CPU_PEER,
    OBLIVIOUS_OK,
    OBLIVIOUS_RECORDS,
    OBLIVIOUS_ROW_RANGES,
    OBLIVIOUS_SPECULATION,
    OBLIVIOUS_TABLES,
    OBLIVIOUS_TRACE_MARK,
    oblivious_device_supported,
    oblivious_launch_census,
    oblivious_leaf_budget,
    oblivious_open_blockers,
    oblivious_reason_name,
    oblivious_records_needed,
    oblivious_schedule_launches,
)
from mojotrees.gpu_tree_tables import OBLIVIOUS_PLAN_ITEMS
from mojotrees.histogram_gpu import GpuHistogramBuilder
from mojotrees.train_gpu import grow_tree_gpu
from mojotrees.tree import Tree, TreeParams, grow_tree
from mojotrees.gpu_split_search import (
    DEFAULT_GAIN_FORM,
    GpuSplitParams,
    GpuSplitRecord,
    GpuSplitSearcher,
    MONOTONE_FREE,
    OBLIVIOUS_MAX_LEAVES,
    gpu_cross_node_s,
    gpu_cross_offset,
    gpu_leaf_score,
    gpu_resolve_gain_form,
    gpu_right_sum,
    gpu_soft_threshold_l1,
    gpu_split_gain,
)

# The queue is 64 command buffers deep and MAX never raises it
# (`docs/GPU_PORTABILITY.md` 6.2). Per-launch enqueue cost is measured flat at
# 6 to 7 microseconds through 64 and 14 to 17 beyond
# (`bench/results/session3_2026-08-16/RESULTS.md`). A knee, not a wall, which
# is why the tests below say "under by two" rather than "safe".
comptime _QUEUE_DEPTH = 64


# --- The census -----------------------------------------------------------


def test_the_census_reproduces_the_table_the_round_was_opened_on() raises:
    # Depth, fused, standalone. Every row computed statically off the launch
    # shapes in `gpu_resident_round`'s own per-step census.
    assert_equal(oblivious_launch_census(4), 44)
    assert_equal(oblivious_launch_census(4, fused_reduce=False), 48)
    assert_equal(oblivious_launch_census(5), 53)
    assert_equal(oblivious_launch_census(5, fused_reduce=False), 58)
    assert_equal(oblivious_launch_census(6), 62)
    assert_equal(oblivious_launch_census(6, fused_reduce=False), 68)
    assert_equal(oblivious_launch_census(7), 71)
    assert_equal(oblivious_launch_census(7, fused_reduce=False), 78)


def test_depth_six_is_the_knife_edge_and_the_fusion_is_what_decides_it(
) raises:
    # The whole reason the reduction is a loop inside a kernel instead of a
    # kernel of its own. At CatBoost's default depth the fused schedule is
    # under the measured knee and the standalone one is over it, and the gap
    # between them is exactly the six launches a per-level reduce would add.
    assert_true(oblivious_launch_census(6) < _QUEUE_DEPTH)
    assert_true(oblivious_launch_census(6, fused_reduce=False) > _QUEUE_DEPTH)
    assert_equal(
        oblivious_launch_census(6, fused_reduce=False)
        - oblivious_launch_census(6),
        6,
    )
    # Depth 5 is under either way, so a lane that built the standalone version
    # and tested it at depth 5 would have seen nothing wrong.
    assert_true(oblivious_launch_census(5, fused_reduce=False) < _QUEUE_DEPTH)
    # Depth 7 is over either way, which is the other half of why 6 is the
    # bound `OBLIVIOUS_MAX_LEAVES` reserves state for.
    assert_true(oblivious_launch_census(7) > _QUEUE_DEPTH)
    assert_equal(OBLIVIOUS_MAX_LEAVES, 64)


def test_the_census_beats_leaf_wise_at_every_depth_it_admits() raises:
    # The leaf-wise count at the default budget, from the per-step census in
    # `gpu_resident_round`: 7 fixed + 30 steps x 9 + 1 tail.
    var leaf_wise = 7 + 30 * 9 + 1
    assert_equal(leaf_wise, 278)
    for d in range(1, 8):
        assert_true(oblivious_launch_census(d) < leaf_wise)
    # A zero or negative depth is not a tree and is not counted as one.
    assert_equal(oblivious_launch_census(0), 0)


def test_the_batch_item_bound_is_the_census_precondition() raises:
    # The one sizing number this mode cannot be left to a default on. A
    # depth-6 level's last generation has 64 children; at the default bound of
    # 32 that level needs two batches, which costs two command buffers and puts
    # the tree exactly ON the measured knee.
    assert_equal(DEFAULT_MAX_ITEMS, 32)
    assert_equal(OBLIVIOUS_MAX_ITEMS, 64)
    assert_equal(oblivious_launch_census(6, batch_max_items=64), 62)
    assert_equal(oblivious_launch_census(6, batch_max_items=32), 64)
    assert_true(oblivious_launch_census(6, batch_max_items=64) < _QUEUE_DEPTH)
    assert_false(oblivious_launch_census(6, batch_max_items=32) < _QUEUE_DEPTH)
    # And the reason validating at depth 5 would have shown nothing: a depth-5
    # level tops out at 32 children, so the two bounds agree there.
    assert_equal(
        oblivious_launch_census(5, batch_max_items=32),
        oblivious_launch_census(5, batch_max_items=64),
    )
    # The plan is twice a level wide, because a level builds both children of
    # every leaf rather than one and a sibling.
    assert_equal(OBLIVIOUS_PLAN_ITEMS, 2 * OBLIVIOUS_MAX_ITEMS)


def test_the_schedule_as_built_is_counted_and_the_gap_is_named() raises:
    # `oblivious_launch_census` is the registered model and is not edited after
    # the fact; `oblivious_schedule_launches` is the built thing. They differ by
    # one launch per level, which is the record-filing phase a level does not
    # need, and both are under the knee at `max_items >= 64`.
    assert_equal(OBLIVIOUS_LEVEL_LAUNCHES, 6)
    assert_equal(oblivious_schedule_launches(6), 56)
    assert_equal(oblivious_schedule_launches(6, batch_max_items=32), 58)
    assert_equal(
        oblivious_launch_census(6, batch_max_items=64)
        - oblivious_schedule_launches(6),
        6,
    )
    for d in range(1, 7):
        assert_true(oblivious_schedule_launches(d) < _QUEUE_DEPTH)
        assert_true(
            oblivious_schedule_launches(d)
            <= oblivious_launch_census(d, batch_max_items=64)
        )
    assert_equal(oblivious_schedule_launches(0), 0)
    # A schedule with no batcher at all is the 176 shape the census names, and
    # this is the same arithmetic from the other side.
    assert_true(oblivious_schedule_launches(6, batch_max_items=-1) > 100)


def test_num_leaves_does_not_bind_and_the_budget_says_so() raises:
    # `num_leaves` is ignored under this mode. Every table the plane sizes comes
    # from here, and the point of the assertion is that the answer does not
    # move when `num_leaves` does.
    for leaves in [2, 3, 31, 255]:
        var params = _tree_params(max_depth=4, num_leaves=leaves)
        assert_equal(oblivious_leaf_budget(params), 16)
        assert_equal(oblivious_records_needed(params), 17)
    assert_equal(
        oblivious_leaf_budget(_tree_params(max_depth=6, num_leaves=31)), 64
    )
    assert_equal(
        oblivious_records_needed(_tree_params(max_depth=6, num_leaves=31)), 65
    )
    # The leaf-wise default budget of 31 asks for 33 records and a depth-6
    # oblivious tree needs 65, which is why the sizing had to be re-asked rather
    # than inherited.
    assert_true(
        oblivious_records_needed(_tree_params(max_depth=6, num_leaves=31))
        > 31 + 2
    )


def test_every_refusal_names_itself() raises:
    assert_equal(oblivious_reason_name(OBLIVIOUS_OK), String("ok"))
    var reasons = [
        OBLIVIOUS_DEPTH,
        OBLIVIOUS_LEVEL_HISTOGRAM,
        OBLIVIOUS_ROW_RANGES,
        OBLIVIOUS_CATEGORICAL,
        OBLIVIOUS_NO_CPU_PEER,
        OBLIVIOUS_TABLES,
        OBLIVIOUS_SPECULATION,
        OBLIVIOUS_RECORDS,
    ]
    for i in range(len(reasons)):
        var name = oblivious_reason_name(reasons[i])
        assert_true(name.byte_length() > 0)
        assert_true(name != String("unknown"))
        assert_true(name != String("ok"))
    # The codes are distinct, so a caller cannot report one refusal as
    # another.
    for i in range(len(reasons)):
        for j in range(len(reasons)):
            if i != j:
                assert_true(reasons[i] != reasons[j])
    assert_equal(oblivious_reason_name(99), String("unknown"))


def test_the_standing_blockers_are_all_closed() raises:
    # The list held three: the whole-level histogram build, the host row-range
    # table, and the missing CPU comparator. Each is closed by something that
    # now runs -- the batched level build behind
    # `GpuHistogramBuilder.enqueue_desc_level_children`,
    # `_publish_level_row_ranges` through `LeafRangeTable.set_window`, and
    # `tree._grow_oblivious_levels`. An empty list something asserts is stronger
    # than a deleted function nobody can ask.
    assert_equal(len(oblivious_open_blockers()), 0)
    # The two codes that can no longer be returned still name themselves, so a
    # trace written before the closure still reads back.
    assert_true(
        oblivious_reason_name(OBLIVIOUS_ROW_RANGES) != String("unknown")
    )
    assert_true(
        oblivious_reason_name(OBLIVIOUS_NO_CPU_PEER) != String("unknown")
    )


def test_the_leaf_index_rule_is_stated_in_one_place() raises:
    # The one thing the two oblivious growers have to agree about. The test is
    # not that the sentence is well written; it is that there is exactly one
    # of it and that it says which end the first level's bit goes.
    assert_true(String(OBLIVIOUS_LEAF_INDEX_RULE).byte_length() > 0)
    assert_true("least significant" in OBLIVIOUS_LEAF_INDEX_RULE)
    assert_true("ascending leaf" in OBLIVIOUS_LEAF_INDEX_RULE)


# --- A host replica of the cross-leaf scan --------------------------------


@fieldwise_init
struct _LevelChoice(Copyable, Movable):
    """What a level search decides, which is one split for the whole level."""

    var found: Bool
    var feature: Int
    var bin: Int
    var default_left: Bool
    var gain: Float32


def _zeroed(n: Int) -> List[Int32]:
    var out = List[Int32](capacity=n)
    out.resize(n, Int32(0))
    return out^


def _one(a: List[Int]) -> List[List[Int]]:
    """A one-leaf level's plane. A list literal of `List[Int]` will not
    convert to `List[List[Int]]`, so the outer list is built rather than
    written."""
    var out = List[List[Int]]()
    out.append(a.copy())
    return out^


def _pair(a: List[Int], b: List[Int]) -> List[List[Int]]:
    """A two-leaf level's plane, leaf 0 first. The order is the summation
    order; see `OBLIVIOUS_LEAF_INDEX_RULE`."""
    var out = List[List[Int]]()
    out.append(a.copy())
    out.append(b.copy())
    return out^


def _level_words(
    n_features: Int,
    n_bins: Int,
    g: List[List[Int]],
    h: List[List[Int]],
    c: List[List[Int]],
) raises -> List[Int32]:
    """`n_slots` consecutive `[grad | hess | count]` histograms, leaf 0 first,
    from per-leaf flat `[f * n_bins + b]` lists. The slot stride is
    `3 * n_features * n_bins`, which is what the level search indexes by."""
    var size = n_features * n_bins
    var n_slots = len(g)
    if len(h) != n_slots or len(c) != n_slots:
        raise Error("every leaf needs all three planes")
    var words = _zeroed(3 * size * n_slots)
    for s in range(n_slots):
        if len(g[s]) != size or len(h[s]) != size or len(c[s]) != size:
            raise Error("plane length must equal n_features * n_bins")
        var base = s * 3 * size
        for i in range(size):
            words[base + i] = Int32(g[s][i])
            words[base + size + i] = Int32(h[s][i])
            words[base + 2 * size + i] = Int32(c[s][i])
    return words^


def _reference_level(
    words: List[Int32],
    n_features: Int,
    n_bins: Int,
    n_leaves: Int,
    params: GpuSplitParams,
    missing_bins: List[Int] = [],
) raises -> _LevelChoice:
    """`_scan_slot_oblivious_kernel` on the host, statement for statement.

    Both fixed-point scales are 1.0 in these tests, so every quantized word
    dequantizes to itself exactly and the only arithmetic that can differ
    between this and the kernel is the Float32 summation -- which is the thing
    being checked, so it is written in the same order: features ascending,
    bins ascending, missing-left before missing-right, leaves ascending and
    innermost.
    """
    var size = n_features * n_bins
    var slot_cells = 3 * size
    var g_inv = Float32(1.0)
    var h_inv = Float32(1.0)
    var lambda_l2 = Float32(params.lambda_l2)
    var lambda_l1 = Float32(params.lambda_l1)
    var min_child_hess = Float32(params.min_child_hess)
    var min_data = Int32(params.min_data_in_leaf)
    var form = gpu_resolve_gain_form(Int32(DEFAULT_GAIN_FORM), lambda_l1)

    var best = _LevelChoice(False, -1, -1, False, Float32(0.0))
    # The cross-feature reduction downstream walks slots ascending and accepts
    # a new best only on a strictly greater gain, so the feature loop here
    # carries the same rule.
    for f in range(n_features):
        var missing_bin = -1
        if len(missing_bins) > 0:
            missing_bin = missing_bins[f]
        var n_scan = missing_bin if missing_bin >= 0 else n_bins

        var tot_g = List[Int32]()
        var tot_h = List[Int32]()
        var tot_c = List[Int32]()
        var par = List[Float32]()
        var nss = List[Float32]()
        var off = List[Float32]()
        var mis_g = List[Int32]()
        var mis_h = List[Int32]()
        var mis_c = List[Int32]()
        var run_g = List[Int32]()
        var run_h = List[Int32]()
        var run_c = List[Int32]()
        for l in range(n_leaves):
            var lb = l * slot_cells + f * n_bins
            var tg = Int32(0)
            var th = Int32(0)
            var tc = Int32(0)
            for b in range(n_bins):
                tg += words[lb + b]
                th += words[size + lb + b]
                tc += words[2 * size + lb + b]
            tot_g.append(tg)
            tot_h.append(th)
            tot_c.append(tc)
            var tgf = tg.cast[DType.float32]() * g_inv
            var thf = th.cast[DType.float32]() * h_inv
            par.append(gpu_leaf_score(tgf, thf, lambda_l1, lambda_l2))
            var ns = gpu_cross_node_s(thf, lambda_l2)
            nss.append(ns)
            off.append(
                gpu_cross_offset(tgf, thf, lambda_l1, lambda_l2, lambda_l2, ns)
            )
            if missing_bin >= 0:
                mis_g.append(words[lb + missing_bin])
                mis_h.append(words[size + lb + missing_bin])
                mis_c.append(words[2 * size + lb + missing_bin])
            else:
                mis_g.append(Int32(0))
                mis_h.append(Int32(0))
                mis_c.append(Int32(0))
            run_g.append(Int32(0))
            run_h.append(Int32(0))
            run_c.append(Int32(0))

        for b in range(n_scan):
            for l in range(n_leaves):
                var lb = l * slot_cells + f * n_bins
                run_g[l] += words[lb + b]
                run_h[l] += words[size + lb + b]
                run_c[l] += words[2 * size + lb + b]
            for d in range(2):
                var want_default_left = d == 0
                if want_default_left and missing_bin < 0:
                    continue
                if not want_default_left and missing_bin >= 0:
                    var any_missing = False
                    for l in range(n_leaves):
                        if mis_c[l] > Int32(0):
                            any_missing = True
                    if not any_missing:
                        continue
                var total = Float32(0.0)
                for l in range(n_leaves):
                    var lg = run_g[l]
                    var lh = run_h[l]
                    var lc = run_c[l]
                    if want_default_left:
                        lg += mis_g[l]
                        lh += mis_h[l]
                        lc += mis_c[l]
                    var tg = tot_g[l]
                    var th = tot_h[l]
                    var tc = tot_c[l]
                    var tgf = tg.cast[DType.float32]() * g_inv
                    var thf = th.cast[DType.float32]() * h_inv
                    var lhf = lh.cast[DType.float32]() * h_inv
                    var rhf = gpu_right_sum(thf, lhf, th, lh, h_inv, form)
                    var lgf = lg.cast[DType.float32]() * g_inv
                    var rgf = gpu_right_sum(tgf, lgf, tg, lg, g_inv, form)
                    if (
                        lc < min_data
                        or tc - lc < min_data
                        or lhf < min_child_hess
                        or rhf < min_child_hess
                    ):
                        continue
                    total += gpu_split_gain(
                        gpu_soft_threshold_l1(lgf, lambda_l1),
                        lhf,
                        gpu_soft_threshold_l1(rgf, lambda_l1),
                        rhf,
                        lambda_l2,
                        par[l],
                        Int32(MONOTONE_FREE),
                        Float32(0.0),
                        Float32(0.0),
                        False,
                        nss[l],
                        off[l],
                        form,
                    )
                if total > best.gain:
                    best = _LevelChoice(
                        True, f, b, want_default_left, total
                    )
    return best^


def _assert_matches(record: GpuSplitRecord, want: _LevelChoice) raises:
    """No tolerance. The gain is compared as a Float32 bit pattern, because
    the whole claim of the fused reduction is that it is the same sum in the
    same order as a host loop and not a close one."""
    assert_true(record.found == want.found)
    if not want.found:
        return
    assert_equal(record.feature, want.feature)
    assert_equal(record.bin, want.bin)
    assert_true(record.default_left == want.default_left)
    assert_equal(
        Float32(record.gain).to_bits(), want.gain.to_bits()
    )


def _params(
    lambda_l2: Float64 = 1.0,
    lambda_l1: Float64 = 0.0,
    min_child_hess: Float64 = 0.0,
    min_data_in_leaf: Int = 0,
) -> GpuSplitParams:
    return GpuSplitParams(
        lambda_l2,
        lambda_l1,
        min_child_hess,
        min_data_in_leaf,
        CategoricalParams.default(),
    )


def _search_level(
    words: List[Int32],
    n_features: Int,
    n_bins: Int,
    n_leaves: Int,
    params: GpuSplitParams,
    missing_bins: List[Int] = [],
) raises -> GpuSplitRecord:
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var searcher = GpuSplitSearcher(
            n_features,
            n_bins,
            missing_bins,
            CategoricalSpec.none(),
            n_leaves + 1,
        )
        searcher.set_monotone([])
        searcher.upload_level_histogram(words, n_leaves)
        var slots = List[Int](capacity=n_leaves)
        for l in range(n_leaves):
            slots.append(l)
        return searcher.search_oblivious_level(
            params, 1.0, 1.0, slots, level_record=0, leaf_base=1
        )


# --- The device kernel ----------------------------------------------------


def test_a_one_leaf_level_matches_the_host_replica() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        # The degenerate level. A sum over one leaf is that leaf's own gain,
        # so this pins the per-leaf arithmetic before any summing is asked of
        # it, and it is the case where a bug in the cross-leaf loop cannot
        # hide behind a bug in the gain.
        var words = _level_words(
            1,
            4,
            _one([-4, -2, 2, 4]),
            _one([1, 1, 1, 1]),
            _one([10, 10, 10, 10]),
        )
        var got = _search_level(words, 1, 4, 1, _params())
        assert_true(got.found)
        _assert_matches(got, _reference_level(words, 1, 4, 1, _params()))


def test_the_level_split_is_not_either_leafs_own_best() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        # Two leaves that disagree about the best bin of feature 0 and agree
        # about a middling bin of feature 1. The summed score has to pick the
        # agreement, which no per-leaf search would, so this is the assertion
        # that the reduction is cross-leaf at all rather than a search over
        # leaf 0 with extra steps.
        var a_g: List[Int] = [-6, 1, 1, 1, -3, -3, 3, 3]
        var b_g: List[Int] = [1, 1, 1, -6, -3, -3, 3, 3]
        var ones: List[Int] = [1, 1, 1, 1, 1, 1, 1, 1]
        var tens: List[Int] = [10, 10, 10, 10, 10, 10, 10, 10]
        var words = _level_words(
            2, 4, _pair(a_g, b_g), _pair(ones, ones), _pair(tens, tens)
        )
        var want = _reference_level(words, 2, 4, 2, _params())
        var got = _search_level(words, 2, 4, 2, _params())
        assert_true(got.found)
        _assert_matches(got, want)
        # Feature 1 is the one both leaves agree about, and it is what the
        # level takes. If this ever flips, the level is being decided by one
        # leaf and the test above would still pass.
        assert_equal(got.feature, 1)


def test_a_leaf_that_fails_min_data_contributes_nothing() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        # Leaf 1 is small enough that `min_data_in_leaf` rejects every
        # candidate for it. Under the rule this mode records -- CatBoost
        # scores a failing leaf as zero contribution -- the level still takes
        # leaf 0's split, and the summed gain is leaf 0's alone.
        var big_g: List[Int] = [-4, -2, 2, 4]
        var big_h: List[Int] = [4, 4, 4, 4]
        var big_c: List[Int] = [20, 20, 20, 20]
        var tiny_g: List[Int] = [-1, 0, 0, 1]
        var tiny_h: List[Int] = [1, 1, 1, 1]
        var tiny_c: List[Int] = [1, 1, 1, 1]
        var words = _level_words(
            1,
            4,
            _pair(big_g, tiny_g),
            _pair(big_h, tiny_h),
            _pair(big_c, tiny_c),
        )
        var p = _params(min_data_in_leaf=5)
        var want = _reference_level(words, 1, 4, 2, p)
        var got = _search_level(words, 1, 4, 2, p)
        assert_true(got.found)
        _assert_matches(got, want)
        # And the level's answer is exactly the level-of-one answer over the
        # leaf that passed, bit for bit, which is what "contributes zero"
        # means and is stronger than agreeing with the replica.
        var alone = _level_words(
            1, 4, _one(big_g), _one(big_h), _one(big_c)
        )
        var solo = _reference_level(alone, 1, 4, 1, p)
        assert_equal(got.bin, solo.bin)
        assert_equal(Float32(got.gain).to_bits(), solo.gain.to_bits())


def test_a_level_no_leaf_can_split_finds_nothing() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        # Every candidate refused by every leaf sums to exactly 0.0, and 0.0
        # never beats the initial best, so the level writes no record rather
        # than committing a split no child could take.
        var flat: List[Int] = [1, 1, 1, 1]
        var ones: List[Int] = [1, 1, 1, 1]
        var counts: List[Int] = [2, 2, 2, 2]
        var words = _level_words(
            1, 4, _pair(flat, flat), _pair(ones, ones), _pair(counts, counts)
        )
        var p = _params(min_data_in_leaf=100)
        var got = _search_level(words, 1, 4, 2, p)
        assert_false(got.found)
        assert_false(_reference_level(words, 1, 4, 2, p).found)


def test_missing_rows_choose_one_direction_for_the_whole_level() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        # Bin 3 is the missing bin of feature 0, so the candidate set is
        # bins 0..2 crossed with a direction, and the direction is one
        # decision for the level rather than one per leaf.
        var a_g: List[Int] = [-4, -2, 2, 6]
        var b_g: List[Int] = [-3, -1, 3, 5]
        var hess: List[Int] = [2, 2, 2, 2]
        var cnt: List[Int] = [10, 10, 10, 10]
        var words = _level_words(
            1, 4, _pair(a_g, b_g), _pair(hess, hess), _pair(cnt, cnt)
        )
        var missing: List[Int] = [3]
        var want = _reference_level(words, 1, 4, 2, _params(), missing)
        var got = _search_level(words, 1, 4, 2, _params(), missing)
        assert_true(got.found)
        _assert_matches(got, want)
        # The missing bin itself is never a threshold.
        assert_true(got.bin < 3)


def test_a_full_depth_six_level_matches_the_host_replica() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        # 64 leaves, which is the level `OBLIVIOUS_MAX_LEAVES` reserves state
        # for and the last level of a depth-6 tree. The summation is 64 terms
        # deep here, which is where a reassociating reduction would show up as
        # a differing low bit and where the bit comparison earns its keep.
        var n_bins = 8
        var n_features = 3
        var size = n_features * n_bins
        var g = List[List[Int]]()
        var h = List[List[Int]]()
        var c = List[List[Int]]()
        for l in range(OBLIVIOUS_MAX_LEAVES):
            var gl = List[Int](capacity=size)
            var hl = List[Int](capacity=size)
            var cl = List[Int](capacity=size)
            for i in range(size):
                # Deliberately not round numbers: the point is a sum whose
                # order matters, so the terms must not be exactly
                # representable multiples of one another.
                gl.append((i * 7 + l * 13) % 23 - 11)
                hl.append(1 + (i * 5 + l * 3) % 7)
                cl.append(3 + (i + l) % 11)
            g.append(gl^)
            h.append(hl^)
            c.append(cl^)
        var words = _level_words(n_features, n_bins, g, h, c)
        var p = _params(lambda_l2=1.0, min_data_in_leaf=1)
        var want = _reference_level(
            words, n_features, n_bins, OBLIVIOUS_MAX_LEAVES, p
        )
        var got = _search_level(
            words, n_features, n_bins, OBLIVIOUS_MAX_LEAVES, p
        )
        assert_true(got.found)
        assert_true(want.found)
        _assert_matches(got, want)


def test_the_level_search_is_deterministic() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        # Two runs of the same level in one process. The scan is one thread
        # per feature and the summation is a loop, so there is no scheduling
        # for an answer to depend on; this is the assertion that says so.
        var a_g: List[Int] = [-5, -2, 1, 4, 2, -3, 3, 1]
        var b_g: List[Int] = [1, -4, 2, 3, -2, 4, -1, 2]
        var hess: List[Int] = [1, 2, 1, 2, 1, 2, 1, 2]
        var cnt: List[Int] = [7, 8, 9, 10, 11, 12, 13, 14]
        var words = _level_words(
            2, 4, _pair(a_g, b_g), _pair(hess, hess), _pair(cnt, cnt)
        )
        var one = _search_level(words, 2, 4, 2, _params())
        var two = _search_level(words, 2, 4, 2, _params())
        assert_true(one.found == two.found)
        assert_equal(one.feature, two.feature)
        assert_equal(one.bin, two.bin)
        assert_equal(
            Float32(one.gain).to_bits(), Float32(two.gain).to_bits()
        )


def test_a_level_wider_than_the_reservation_is_refused() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        # Depth 7 is 128 leaves and is over the queue's knee whatever this
        # kernel does, so the refusal is a named one at the launch rather than
        # a silently truncated scan.
        var flat: List[Int] = [1, 1, 1, 1]
        var words = _level_words(1, 4, _one(flat), _one(flat), _one(flat))
        var searcher = GpuSplitSearcher(1, 4)
        searcher.set_monotone([])
        searcher.upload_level_histogram(words, 1)
        var slots = List[Int]()
        for _ in range(OBLIVIOUS_MAX_LEAVES + 1):
            slots.append(0)
        var refused = False
        try:
            _ = searcher.search_oblivious_level(_params(), 1.0, 1.0, slots)
        except:
            refused = True
        assert_true(refused)


# --- The level schedule, end to end ---------------------------------------
#
# Everything above tests one launch. What follows tests the schedule: the level
# commit, the whole-prefix partition, the batched level build, and the host
# row-range replay, by growing a tree with them and comparing it to the tree the
# CPU oblivious grower builds from the same gradients.
#
# **The comparison is against `tree._grow_oblivious_levels`, not against another
# device path**, and that is the accuracy gate for the mode. What can and cannot
# be exact between the two backends is stated at `_assert_same_shape`: a
# decision is discrete and is compared with no tolerance; a leaf value is a
# Float64 Newton step on one side and a Float32 one over fixed-point sums on the
# other, and no schedule can make those the same bits.


comptime _OB_TRACE_PATH = "./.test_gpu_oblivious_device_trace.tmp"
"""Where the level schedule's trace lands, so a test can count the trees it
grew rather than assume it grew any."""


def _truncate_trace() raises:
    with open(_OB_TRACE_PATH, "w") as handle:
        handle.write(String(""))


def _read_trace() raises -> String:
    return open(_OB_TRACE_PATH, "r").read()


def _tree_params(
    max_depth: Int,
    num_leaves: Int = 31,
    min_data: Int = 1,
    policy: Int = GROW_OBLIVIOUS,
) -> TreeParams:
    # `num_leaves` is deliberately unrelated to the depth: it does not bind
    # under this policy and two tests above pin that.
    return TreeParams(
        num_leaves,
        min_data,
        1.0,
        1e-3,
        max_depth=max_depth,
        grow_policy=policy,
    )


def _dense(n_rows: Int, n_features: Int) -> List[Float64]:
    """Column-major pseudo-random features, the shape `bin_equal_width`
    takes. The same generator `tests/test_oblivious.mojo` uses, so the two
    files disagree about nothing but the backend."""
    var out = List[Float64](capacity=n_rows * n_features)
    var state = UInt64(20260816)
    for _ in range(n_rows * n_features):
        state = state * 6364136223846793005 + 1442695040888963407
        out.append(Float64(state >> 11) * (1.0 / 9007199254740992.0))
    return out^


def _grads(n_rows: Int, features: List[Float64]) -> List[Float64]:
    """A gradient with real structure in the first four features, so a level
    has something to disagree about and the chosen split is decided by the data
    rather than by a tie. A near-tie is the one thing that can legitimately move
    a split between a Float64 host scan and a Float32 device one, so the fixture
    is built to avoid one rather than to hope."""
    var out = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        out.append(
            -(
                3.0 * features[r]
                - 2.0 * features[n_rows + r]
                + 1.5 * features[2 * n_rows + r] * features[3 * n_rows + r]
            )
        )
    return out^


def _ones(n: Int) -> List[Float64]:
    var out = List[Float64](capacity=n)
    for _ in range(n):
        out.append(1.0)
    return out^


@fieldwise_init
struct _Pair(Movable):
    var host: Tree
    var device: Tree
    var trace: String


def _both_growers(
    n_rows: Int, n_features: Int, n_bins: Int, params: TreeParams
) raises -> _Pair:
    """Grow one tree on each backend from the same binned matrix and the same
    Float64 gradients, and return both with the device arm's trace.

    The gradients are host Float64 on both sides. That is the point: the only
    thing that differs between the two arms is the grower, not the objective,
    not the round, and not the initialization. `upload_gradients` quantizes them
    into the fixed-point domain the device accumulates in, which is the one
    numeric difference the comparison has to survive.
    """
    var features = _dense(n_rows, n_features)
    var data = bin_equal_width(features, n_rows, n_features, n_bins)
    var grad = _grads(n_rows, features)
    var hess = _ones(n_rows)
    var host = grow_tree(data, grad, hess, params)

    _ = setenv("MOJOTREES_GPU_TREE_RESIDENT_TRACE", _OB_TRACE_PATH)
    _truncate_trace()
    var builder = GpuHistogramBuilder(data)
    builder.upload_gradients(grad, hess)
    var device = grow_tree_gpu(builder, params)
    var trace = _read_trace()
    _ = setenv("MOJOTREES_GPU_TREE_RESIDENT_TRACE", "")
    return _Pair(host^, device^, trace)


def _assert_plane_ran(trace: String, label: String) raises:
    """The level schedule executed, exactly once.

    The positive control, and it is not a formality here. A refused
    configuration raises rather than falling back, but a *routing* mistake --
    the AUTO decision sending an oblivious fit to the host-scan grower, say --
    would produce a tree that is still symmetric and still correct, and every
    structural assertion below would pass while this function never ran.
    """
    assert_equal(
        trace.count(OBLIVIOUS_TRACE_MARK),
        1,
        label
        + ": the oblivious level schedule traced "
        + String(trace.count(OBLIVIOUS_TRACE_MARK))
        + " trees; zero means the fit never reached it and this comparison"
        + " proves nothing",
    )
    assert_equal(
        trace.count("status=running"),
        0,
        label + ": a tree came home while growth was still running",
    )
    assert_equal(
        trace.count("status=pool_full") + trace.count("status=overflow"),
        0,
        label + ": the device tables were too small for the depth budget",
    )
    # The batched level build is what pays the partition's deferred copy-back,
    # once per level, and the counter is incremented by the code that takes the
    # fused branch rather than by a prediction of how often it should.
    assert_equal(
        trace.count("plane=device-oblivious"),
        1,
        label + ": the trace mark is not the one this plane writes",
    )


def _node_depths(tree: Tree) -> List[Int]:
    var depths = List[Int](capacity=len(tree.feature))
    depths.resize(len(tree.feature), 0)
    for n in range(len(tree.feature)):
        if tree.feature[n] >= 0:
            depths[tree.left[n]] = depths[n] + 1
            depths[tree.right[n]] = depths[n] + 1
    return depths^


def _assert_symmetric(tree: Tree, label: String) raises:
    """Every internal node at one depth carries the same split.

    THE marker of an oblivious tree in an ordinary binary representation, and
    the assertion that keeps a leaf-wise grower from passing everything else in
    this file."""
    var depths = _node_depths(tree)
    for d in range(len(tree.feature)):
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
            assert_equal(
                tree.feature[n], feature, label + ": level feature"
            )
            assert_equal(
                tree.threshold_bin[n], threshold, label + ": level threshold"
            )
            assert_true(
                tree.default_left[n] == default_left,
                label + ": level missing direction",
            )


def _assert_same_shape(host: Tree, device: Tree, label: String) raises:
    """Node for node, with no tolerance on anything discrete.

    **What is exact and what is not, stated rather than assumed.** The split a
    node makes -- its feature, its threshold bin, its missing direction -- and
    the shape of the tree -- its node count, its child ids, the order node ids
    were assigned in -- are *decisions*, and a decision is discrete. Those are
    compared with `assert_equal` and no tolerance, because a different split is
    not corrected by a later round the way a leaf value is. Node row counts are
    exact integers on both sides and are compared the same way.

    Leaf **values** are not compared as bit patterns and cannot be. The host
    computes a Float64 Newton step from Float64 histogram sums; the device
    computes a Float32 one from fixed-point Int32 sums dequantized by a power of
    two. Those are two different arithmetics over two different quantizations of
    the same data, and no amount of scheduling makes them the same bits. The
    device plane's own bit-level comparison is against another *device* path
    (`tests/test_gpu_tree_resident.mojo`); across backends this package's
    standing convention is structural equality plus a tolerance, which is what
    `tests/test_backend_equivalence.mojo` also asserts.
    """
    assert_equal(len(device.feature), len(host.feature), label + ": n_nodes")
    assert_equal(device.n_leaves, host.n_leaves, label + ": n_leaves")
    for i in range(len(host.feature)):
        assert_equal(device.feature[i], host.feature[i], label + ": feature")
        assert_equal(
            device.threshold_bin[i],
            host.threshold_bin[i],
            label + ": threshold_bin",
        )
        assert_equal(device.left[i], host.left[i], label + ": left")
        assert_equal(device.right[i], host.right[i], label + ": right")
        assert_true(
            device.default_left[i] == host.default_left[i],
            label + ": default_left",
        )
        assert_equal(
            device.missing_bin[i], host.missing_bin[i], label + ": missing_bin"
        )
        assert_equal(
            Int(device.count[i]), Int(host.count[i]), label + ": node count"
        )
        var want = host.value[i]
        var got = device.value[i]
        var scale = abs(want) if abs(want) > 1.0 else 1.0
        assert_true(
            abs(got - want) <= 1e-4 * scale,
            label
            + ": leaf value "
            + String(got)
            + " against "
            + String(want),
        )


def test_a_depth_three_tree_matches_the_cpu_grower_node_for_node() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        # The smallest depth at which the leaf numbering and the node-id order
        # can disagree: at level 2 the leaves in node-id order carry leaf
        # indices 0, 2, 1, 3, so a backend that numbered ascending node id
        # would build a different tree here and an identical one at depth 2.
        var run = _both_growers(400, 5, 16, _tree_params(max_depth=3))
        _assert_plane_ran(run.trace, String("depth 3"))
        _assert_symmetric(run.device, String("depth 3 device"))
        _assert_same_shape(run.host, run.device, String("depth 3"))
        assert_equal(run.device.n_leaves, 8)


def test_a_depth_six_tree_matches_the_cpu_grower_node_for_node() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        # CatBoost's default depth, the widest level this mode admits, and the
        # only depth at which the `max_items` precondition can bite. 64 leaves,
        # 127 nodes, and the last level's 64 children built by one batch.
        var run = _both_growers(2000, 6, 32, _tree_params(max_depth=6))
        _assert_plane_ran(run.trace, String("depth 6"))
        _assert_symmetric(run.device, String("depth 6 device"))
        _assert_same_shape(run.host, run.device, String("depth 6"))
        assert_equal(run.device.n_leaves, 64)
        assert_equal(len(run.device.feature), 127)


def test_the_windows_the_device_published_tile_the_prefix() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        # `_publish_level_row_ranges` is what makes the tree usable for a second
        # round: `update_raw_device` reads the host row-range table, and a level
        # partition leaves a parent's rows in two blocks that are not adjacent,
        # so `LeafRangeTable.split` cannot replay it. The proof is that
        # `end_descriptor_partition` lifted the staleness refusal at all -- it
        # checks every window against the device's own frontier and runs the
        # tiling invariant before it does -- so a table that can be READ here is
        # a table that passed all four of its checks.
        var n_rows = 600
        var features = _dense(n_rows, 4)
        var data = bin_equal_width(features, n_rows, 4, 16)
        var grad = _grads(n_rows, features)
        var builder = GpuHistogramBuilder(data)
        builder.upload_gradients(grad, _ones(n_rows))
        var tree = grow_tree_gpu(builder, _tree_params(max_depth=4))
        # Reading at all is the assertion: every accessor refuses while the
        # descriptor partition owns the windows.
        var total = builder.rows.ranges.total_active()
        assert_equal(total, n_rows)
        assert_equal(builder.rows.ranges.n_nodes(), len(tree.feature))
        builder.rows.ranges.check_invariants()
        # And the windows are the leaves' own row counts, which is the number
        # the raw-score update walks.
        var leaf_rows = 0
        for n in range(len(tree.feature)):
            if tree.feature[n] < 0:
                leaf_rows += builder.rows.ranges.get(n).count()
        assert_equal(leaf_rows, n_rows)


def test_a_second_round_of_gradients_still_grows_the_same_tree() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        # The failure `_publish_level_row_ranges` exists against was silent for
        # exactly one round: correct trees, wrong scores, and every round after
        # the first diverging. Growing two trees on one builder is the cheapest
        # thing that would have caught it, because the second tree's
        # `begin_tree` refuses outright if the copy-back debt or the staleness
        # poison survived the first.
        var n_rows = 500
        var features = _dense(n_rows, 5)
        var data = bin_equal_width(features, n_rows, 5, 16)
        var grad = _grads(n_rows, features)
        var hess = _ones(n_rows)
        var params = _tree_params(max_depth=4)
        var host = grow_tree(data, grad, hess, params)
        var builder = GpuHistogramBuilder(data)
        builder.upload_gradients(grad, hess)
        var first = grow_tree_gpu(builder, params, [], 0)
        var second = grow_tree_gpu(builder, params, [], 1)
        _assert_same_shape(host, first, String("round 1"))
        _assert_same_shape(host, second, String("round 2"))


def test_the_default_item_bound_refuses_rather_than_landing_on_the_knee(
) raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        # A builder whose batcher was opened for the leaf-wise plane holds 32
        # items. A depth-6 level wants 64, and the refusal is by name rather
        # than a silent second batch.
        var n_rows = 300
        var features = _dense(n_rows, 4)
        var data = bin_equal_width(features, n_rows, 4, 16)
        var builder = GpuHistogramBuilder(data)
        builder.upload_gradients(_grads(n_rows, features), _ones(n_rows))
        assert_true(builder.open_resident(64, DEFAULT_MAX_ITEMS))
        assert_false(builder.oblivious_level_fits(64))
        assert_equal(
            oblivious_device_supported(_tree_params(max_depth=6), builder),
            OBLIVIOUS_LEVEL_HISTOGRAM,
        )
        # Depth 5 wants 32 and fits at the default, which is exactly why a
        # depth-5 validation would have shown nothing.
        assert_true(builder.oblivious_level_fits(32))
        assert_equal(
            oblivious_device_supported(_tree_params(max_depth=5), builder),
            OBLIVIOUS_OK,
        )


def test_the_speculation_and_a_level_build_refuse_to_combine() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        # `_pick_runner_up_kernel` publishes a leaf that is still live, and a
        # plan that builds both of its children overwrites the histogram the
        # next pick reads on a miss. Individually correct, jointly wrong, and
        # refused rather than ordered.
        var n_rows = 300
        var features = _dense(n_rows, 4)
        var data = bin_equal_width(features, n_rows, 4, 16)
        var builder = GpuHistogramBuilder(data)
        builder.upload_gradients(_grads(n_rows, features), _ones(n_rows))
        assert_true(builder.open_resident(64, OBLIVIOUS_MAX_ITEMS))
        assert_equal(
            oblivious_device_supported(_tree_params(max_depth=6), builder),
            OBLIVIOUS_OK,
        )
        _ = setenv("MOJOTREES_GPU_SPECULATION", "1")
        var refused = oblivious_device_supported(
            _tree_params(max_depth=6), builder
        )
        _ = setenv("MOJOTREES_GPU_SPECULATION", "")
        assert_equal(refused, OBLIVIOUS_SPECULATION)


def test_the_other_refusals_are_reachable_by_name() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 300
        var features = _dense(n_rows, 4)
        var data = bin_equal_width(features, n_rows, 4, 16)
        var builder = GpuHistogramBuilder(data)
        builder.upload_gradients(_grads(n_rows, features), _ones(n_rows))
        assert_true(builder.open_resident(64, OBLIVIOUS_MAX_ITEMS))
        # Depth 7 is 128 leaves: over the scan's per-leaf reservation and over
        # the queue's knee whatever the schedule does.
        assert_equal(
            oblivious_device_supported(_tree_params(max_depth=7), builder),
            OBLIVIOUS_DEPTH,
        )
        assert_equal(
            oblivious_device_supported(_tree_params(max_depth=0), builder),
            OBLIVIOUS_DEPTH,
        )
        # A policy that is not this one is not this plane's to grow.
        assert_equal(
            oblivious_device_supported(
                _tree_params(max_depth=4, policy=GROW_LEAFWISE), builder
            ),
            OBLIVIOUS_TABLES,
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
