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

from std.sys import has_accelerator
from std.testing import (
    assert_equal,
    assert_false,
    assert_true,
    TestSuite,
)

from mojotrees.categorical import CategoricalParams, CategoricalSpec
from mojotrees.gpu_resident_round import (
    OBLIVIOUS_CATEGORICAL,
    OBLIVIOUS_DEPTH,
    OBLIVIOUS_LEAF_INDEX_RULE,
    OBLIVIOUS_LEVEL_HISTOGRAM,
    OBLIVIOUS_NO_CPU_PEER,
    OBLIVIOUS_OK,
    OBLIVIOUS_ROW_RANGES,
    oblivious_launch_census,
    oblivious_open_blockers,
    oblivious_reason_name,
)
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


def test_every_refusal_names_itself() raises:
    assert_equal(oblivious_reason_name(OBLIVIOUS_OK), String("ok"))
    var reasons = [
        OBLIVIOUS_DEPTH,
        OBLIVIOUS_LEVEL_HISTOGRAM,
        OBLIVIOUS_ROW_RANGES,
        OBLIVIOUS_CATEGORICAL,
        OBLIVIOUS_NO_CPU_PEER,
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


def test_the_standing_blockers_are_listed_largest_first() raises:
    # The census found two things the design assumed and the code does not
    # provide, and a third that is not this backend's to close. The list is
    # what a reader needs; the test is that it is not empty, that the census
    # blocker is first, and that every entry names itself.
    var open = oblivious_open_blockers()
    assert_equal(len(open), 3)
    assert_equal(open[0], OBLIVIOUS_LEVEL_HISTOGRAM)
    assert_equal(open[1], OBLIVIOUS_ROW_RANGES)
    assert_equal(open[2], OBLIVIOUS_NO_CPU_PEER)
    for i in range(len(open)):
        assert_true(oblivious_reason_name(open[i]) != String("unknown"))
    # The cross-leaf reduction is not on the list: it is built and checked.
    for i in range(len(open)):
        assert_true(open[i] != OBLIVIOUS_OK)


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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
