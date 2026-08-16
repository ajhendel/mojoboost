"""`score_function = Cosine` under `grow_policy = oblivious`, on the device.

The one thing this file exists to establish is that a level's Cosine score is
**a ratio of two cross-leaf accumulators with a single square root taken at the
end**, and not a sum of per-leaf Cosine gains. `_launch_oblivious_search` used
to refuse Cosine outright on exactly that ground, and the refusal was correct
for as long as `_scan_slot_oblivious_kernel`'s leaf loop could only do
`total += gain`. Retiring it means the kernel now carries what it stood in for,
and three things below check that rather than assert it.

1. **The shape.** The winning candidate's cross-leaf ratio is compared against
   the sum of the same leaves' individual Cosine gains, and the two are
   required to DIFFER. If they did not, the distinction would be unobservable
   on this fixture and every other test here would be vacuous.

2. **The arithmetic.** The device record is compared against a host replica
   that mirrors the kernel statement for statement, as a Float32 bit pattern
   and at no tolerance. Both fixed-point scales are 1.0 in these fixtures, so
   every quantized word dequantizes to itself exactly and the only thing that
   can differ between replica and kernel is the Float32 summation order --
   which is the thing being checked. The replica writes the same two `fma`
   spellings the kernel writes, because an unfused `+` is a different number.

3. **Node identity with the CPU grower.** `split.find_best_split_shared` with
   `score_function=SCORE_COSINE` is the specification, and the device is
   required to choose the same (feature, bin, missing direction).
   **The gains are not compared and cannot be**: the host works in Float64
   over exact sums and the device in Float32 over a fixed-point histogram,
   which is the divergence this module already had under L2, and Cosine adds
   exactly one operation to it -- the square root, correctly rounded under
   IEEE-754. Node identity is the bar the oblivious mode holds itself to and
   it is the bar asserted here. Every node-identity fixture is built with a
   clearly dominant candidate, so what is being tested is the rule and not the
   last bit of a near tie.

The illegal-leaf case has its own test, because it is the one place the L2 arm
and the Cosine arm genuinely differ rather than merely rounding differently:
the L2 arm adds `0.0` for a leaf that cannot take the candidate and the Cosine
arm substitutes that leaf's *unsplit* terms, since `0.0` is a legitimate
numerator once the sum is a ratio. Its fixture makes the illegality a matter of
integer counts, so the two backends agree exactly about WHICH leaves are
illegal and the comparison is about what an illegal leaf contributes.

The launch census is asserted unchanged, because a correct kernel that cost a
launch per leaf per level would have to be rewritten before it could ship.

Bits move only under `score_function=Cosine`. Nothing here touches an L2 level,
and the L2 regression below is what says so.

The device tests skip (passing) with no accelerator.
"""

from std.math import fma
from std.sys import has_accelerator
from std.testing import (
    assert_equal,
    assert_false,
    assert_true,
    TestSuite,
)

from mojotrees.categorical import CategoricalParams, CategoricalSpec
from mojotrees.growth_policy import SharedSplitAudit
from mojotrees.gpu_resident_round import oblivious_launch_census
from mojotrees.histogram import Histogram
from mojotrees.split import (
    SCORE_COSINE,
    SCORE_L2,
    SplitInfo,
    find_best_split_shared,
)
from mojotrees.gpu_split_search import (
    DEFAULT_GAIN_FORM,
    GpuSplitParams,
    GpuSplitRecord,
    GpuSplitSearcher,
    MONOTONE_FREE,
    gpu_cosine_out,
    gpu_cosine_score,
    gpu_cosine_unsplit,
    gpu_cross_node_s,
    gpu_cross_offset,
    gpu_leaf_score,
    gpu_resolve_gain_form,
    gpu_right_sum,
    gpu_soft_threshold_l1,
    gpu_split_gain,
)


# --- Fixtures -------------------------------------------------------------


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
    order; see `gpu_resident_round.OBLIVIOUS_LEAF_INDEX_RULE`."""
    var out = List[List[Int]]()
    out.append(a.copy())
    out.append(b.copy())
    return out^


def _quad(
    a: List[Int], b: List[Int], c: List[Int], d: List[Int]
) -> List[List[Int]]:
    """A four-leaf level's plane, leaf 0 first."""
    var out = List[List[Int]]()
    out.append(a.copy())
    out.append(b.copy())
    out.append(c.copy())
    out.append(d.copy())
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


def _histograms(
    words: List[Int32], n_features: Int, n_bins: Int, n_leaves: Int
) raises -> List[Histogram]:
    """The same level, as the CPU grower's own histograms.

    Both fixed-point scales are 1.0 in these fixtures, so a word dequantizes
    to itself and the two backends are handed the same numbers exactly. That
    is what makes a node-identity comparison a statement about the two search
    loops rather than about quantization."""
    var size = n_features * n_bins
    var out = List[Histogram]()
    for l in range(n_leaves):
        var base = l * 3 * size
        var grad = List[Float64](capacity=size)
        var hess = List[Float64](capacity=size)
        var count = List[Int](capacity=size)
        for i in range(size):
            grad.append(Float64(Int(words[base + i])))
            hess.append(Float64(Int(words[base + size + i])))
            count.append(Int(words[base + 2 * size + i]))
        out.append(
            Histogram.from_planes(grad^, hess^, count^, n_features, n_bins)
        )
    return out^


def _params(
    lambda_l2: Float64 = 3.0,
    lambda_l1: Float64 = 0.0,
    min_child_hess: Float64 = 0.0,
    min_data_in_leaf: Int = 0,
) -> GpuSplitParams:
    """CatBoost's `l2_leaf_reg` default is 3 and it is the default here,
    because Cosine's whole difference from L2 is a function of `lambda_l2`
    and a fixture at 0 would be testing the degenerate point."""
    return GpuSplitParams(
        lambda_l2,
        lambda_l1,
        min_child_hess,
        min_data_in_leaf,
        CategoricalParams.default(),
    )


# --- The device, and the host replica of it -------------------------------


def _search_level(
    words: List[Int32],
    n_features: Int,
    n_bins: Int,
    n_leaves: Int,
    params: GpuSplitParams,
    score_function: Int,
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
            params,
            1.0,
            1.0,
            slots,
            level_record=0,
            leaf_base=1,
            score_function=score_function,
        )


def _reference_level(
    words: List[Int32],
    n_features: Int,
    n_bins: Int,
    n_leaves: Int,
    params: GpuSplitParams,
    score_function: Int,
    missing_bins: List[Int] = [],
    sum_of_ratios: Bool = False,
    illegal_adds_zero: Bool = False,
) raises -> _LevelChoice:
    """`_scan_slot_oblivious_kernel` on the host, statement for statement.

    `sum_of_ratios` and `illegal_adds_zero` are the two WRONG spellings of a
    Cosine level, written here so that a test can show they are different
    numbers rather than assert that they would be. The first takes one root
    per leaf and sums, which is what a leaf loop that could only do
    `total += gain` would have computed; the second adds nothing for an
    illegal leaf, which is the L2 arm's rule applied where it does not hold.
    Neither is reachable from the kernel; both are reachable from here.
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
    var cosine = score_function == SCORE_COSINE

    var best = _LevelChoice(False, -1, -1, False, Float32(0.0))
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
        var un_num = List[Float32]()
        var un_den = List[Float32]()
        var mis_g = List[Int32]()
        var mis_h = List[Int32]()
        var mis_c = List[Int32]()
        var run_g = List[Int32]()
        var run_h = List[Int32]()
        var run_c = List[Int32]()
        var p_num = Float32(0.0)
        var p_den = Float32(0.0)
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
            var ut = gpu_cosine_unsplit(tgf, thf, lambda_l1, lambda_l2)
            un_num.append(ut[0])
            un_den.append(ut[1])
            p_num += ut[0]
            p_den += ut[1]
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

        var level_parent = Float32(0.0)
        if cosine:
            level_parent = gpu_cosine_score(p_num, p_den)

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
                var cos_num = Float32(0.0)
                var cos_den = Float32(0.0)
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
                        if cosine and not illegal_adds_zero:
                            cos_num += un_num[l]
                            cos_den += un_den[l]
                        continue
                    if cosine:
                        var tl = gpu_soft_threshold_l1(lgf, lambda_l1)
                        var tr = gpu_soft_threshold_l1(rgf, lambda_l1)
                        var lo = gpu_cosine_out(tl, lhf, lambda_l2)
                        var ro = gpu_cosine_out(tr, rhf, lambda_l2)
                        # The kernel's two `fma` spellings, which are not the
                        # same numbers an unfused `+` would give.
                        var num = fma(-ro, tr, -lo * tl)
                        var den = fma(ro * ro, rhf, lo * lo * lhf)
                        if sum_of_ratios:
                            total += gpu_cosine_score(num, den)
                            total -= gpu_cosine_score(un_num[l], un_den[l])
                        else:
                            cos_num += num
                            cos_den += den
                    else:
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
                if cosine and not sum_of_ratios:
                    total = gpu_cosine_score(cos_num, cos_den) - level_parent
                if total > best.gain:
                    best = _LevelChoice(True, f, b, want_default_left, total)
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
    assert_equal(Float32(record.gain).to_bits(), want.gain.to_bits())


def _assert_node_identical(record: GpuSplitRecord, want: SplitInfo) raises:
    """The bar this mode holds itself to: the same node, not the same number.

    The gain is deliberately NOT compared. The host computes in Float64 over
    exact sums and the device in Float32 over a fixed-point histogram, which
    is the divergence this module had under L2 before Cosine existed; Cosine
    adds one operation to it, the square root, and takes it correctly rounded.
    A tolerance on the gain would be a claim about the SIZE of that
    divergence, which no fixture here measures."""
    assert_true(record.found == want.found)
    if not want.found:
        return
    assert_equal(record.feature, want.feature)
    assert_equal(record.bin, want.bin)
    assert_true(record.default_left == want.default_left)


def _cpu_level(
    mut audit: SharedSplitAudit,
    words: List[Int32],
    n_features: Int,
    n_bins: Int,
    n_leaves: Int,
    params: GpuSplitParams,
    score_function: Int,
    missing_bins: List[Int] = [],
) raises -> SplitInfo:
    """The CPU oblivious grower's own answer for the same level."""
    var hists = _histograms(words, n_features, n_bins, n_leaves)
    return find_best_split_shared(
        audit,
        hists,
        lambda_reg=params.lambda_l2,
        min_child_hess=params.min_child_hess,
        min_data_in_leaf=params.min_data_in_leaf,
        lambda_l1=params.lambda_l1,
        missing_bins=missing_bins,
        score_function=score_function,
    )


# --- The launch shape, which is a precondition and not a result -----------


def test_the_level_still_costs_two_launches_and_the_census_is_unmoved() raises:
    # 360 trees of 6 levels is the shape the shipped default will run, so the
    # cross-leaf accumulation had to fit inside the launch the scan already
    # makes. It does: a second accumulator beside the first, in the same leaf
    # loop of the same kernel. Depth 6 is 62 command buffers per tree fused
    # and 68 standalone, exactly what it was before Cosine.
    assert_equal(oblivious_launch_census(6), 62)
    assert_equal(oblivious_launch_census(6, fused_reduce=False), 68)
    assert_equal(oblivious_launch_census(4), 44)
    assert_equal(oblivious_launch_census(5), 53)


# --- The shape of the score ------------------------------------------------


def test_a_level_is_one_ratio_and_not_a_sum_of_per_leaf_ratios() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        # The claim the retired refusal rested on, made observable. Four
        # leaves that disagree about where the gradient sits, so no leaf's own
        # ratio is the level's.
        var words = _level_words(
            1,
            4,
            _quad(
                [-6, -3, 3, 6], [-1, -5, 5, 1], [-8, 2, -2, 8], [-2, -2, 4, 4]
            ),
            _quad([2, 2, 2, 2], [3, 1, 1, 3], [1, 4, 4, 1], [2, 3, 3, 2]),
            _quad([8, 8, 8, 8], [9, 7, 7, 9], [6, 9, 9, 6], [8, 8, 8, 8]),
        )
        var p = _params()
        var got = _search_level(words, 1, 4, 4, p, SCORE_COSINE)
        assert_true(got.found)
        var ratio = _reference_level(words, 1, 4, 4, p, SCORE_COSINE)
        _assert_matches(got, ratio)

        # And the wrong spelling is a different number, so nothing above is
        # vacuous. Compared as values rather than as an argmax, because an
        # argmax comparison could coincide on a fixture and then say nothing.
        var summed = _reference_level(
            words, 1, 4, 4, p, SCORE_COSINE, sum_of_ratios=True
        )
        assert_true(summed.found)
        assert_true(ratio.gain.to_bits() != summed.gain.to_bits())


def test_a_one_leaf_level_matches_the_replica_exactly() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        # The degenerate level, where a sum over one leaf is that leaf's own
        # ratio. This pins the per-leaf Cosine terms before any summing is
        # asked of them, so a bug in the cross-leaf fold cannot hide behind a
        # bug in the terms.
        var words = _level_words(
            1,
            4,
            _one([-4, -2, 2, 4]),
            _one([1, 1, 1, 1]),
            _one([10, 10, 10, 10]),
        )
        var p = _params()
        var got = _search_level(words, 1, 4, 1, p, SCORE_COSINE)
        assert_true(got.found)
        _assert_matches(got, _reference_level(words, 1, 4, 1, p, SCORE_COSINE))


def test_an_l2_level_is_the_number_it_was_before_cosine_existed() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        # The regression that says bits move only in the new mode: the L2 arm
        # takes the branch it took before the selector existed.
        var words = _level_words(
            1,
            4,
            _pair([-6, -3, 3, 6], [-1, -5, 5, 1]),
            _pair([2, 2, 2, 2], [3, 1, 1, 3]),
            _pair([8, 8, 8, 8], [9, 7, 7, 9]),
        )
        var p = _params()
        var got = _search_level(words, 1, 4, 2, p, SCORE_L2)
        assert_true(got.found)
        _assert_matches(got, _reference_level(words, 1, 4, 2, p, SCORE_L2))


# --- Node identity with the CPU oblivious grower --------------------------


def test_the_device_picks_the_cpu_growers_node_on_a_two_leaf_level() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        # Both leaves put their negative gradient in bins 0 and 1, so bin 1 is
        # the level's split by a wide margin and the comparison is about the
        # rule rather than about the last bit of a near tie.
        var words = _level_words(
            1,
            4,
            _pair([-6, -4, 0, 3], [-5, -3, 1, 2]),
            _pair([2, 3, 4, 3], [2, 2, 3, 2]),
            _pair([20, 20, 20, 20], [15, 15, 15, 15]),
        )
        var p = _params()
        var got = _search_level(words, 1, 4, 2, p, SCORE_COSINE)
        _assert_matches(got, _reference_level(words, 1, 4, 2, p, SCORE_COSINE))
        var audit = SharedSplitAudit.none()
        var want = _cpu_level(audit, words, 1, 4, 2, p, SCORE_COSINE)
        assert_true(want.found)
        assert_equal(audit.n_leaves, 2)
        assert_equal(audit.n_illegal, 0)
        _assert_node_identical(got, want)


def test_a_full_depth_six_level_picks_the_cpu_growers_node() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        # 64 leaves, which is CatBoost's default depth and the widest level
        # `OBLIVIOUS_MAX_LEAVES` reserves state for. Every leaf carries its
        # own tilt, so the cross-leaf accumulators are genuinely folded rather
        # than dominated by one leaf, and every leaf agrees that bin 1 is the
        # threshold, so the level's answer does not rest on a tie.
        var g = List[List[Int]]()
        var h = List[List[Int]]()
        var c = List[List[Int]]()
        for l in range(64):
            var gl = List[Int]()
            var hl = List[Int]()
            var cl = List[Int]()
            gl.append(-10 - (l % 4))
            gl.append(-8 - (l % 3))
            gl.append(1)
            gl.append(1)
            hl.append(2 + (l % 3))
            hl.append(3)
            hl.append(4)
            hl.append(3)
            cl.append(10 + (l % 5))
            cl.append(10)
            cl.append(10)
            cl.append(10)
            g.append(gl^)
            h.append(hl^)
            c.append(cl^)
        var words = _level_words(1, 4, g, h, c)
        var p = _params()
        var got = _search_level(words, 1, 4, 64, p, SCORE_COSINE)
        _assert_matches(
            got, _reference_level(words, 1, 4, 64, p, SCORE_COSINE)
        )
        var audit = SharedSplitAudit.none()
        var want = _cpu_level(audit, words, 1, 4, 64, p, SCORE_COSINE)
        assert_true(want.found)
        assert_equal(audit.n_leaves, 64)
        _assert_node_identical(got, want)


def test_a_level_with_missing_rows_picks_the_cpu_growers_direction() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        # Bin 3 is the missing bin and holds a positive gradient in both
        # leaves, so the level routes it right and splits at bin 1. The
        # direction is level-wide because the candidate is.
        var missing: List[Int] = [3]
        var words = _level_words(
            1,
            4,
            _pair([-10, -8, 1, 4], [-9, -7, 2, 3]),
            _pair([4, 4, 4, 2], [3, 3, 3, 2]),
            _pair([10, 10, 10, 4], [10, 10, 10, 4]),
        )
        var p = _params()
        var got = _search_level(
            words, 1, 4, 2, p, SCORE_COSINE, missing_bins=missing
        )
        _assert_matches(
            got,
            _reference_level(
                words, 1, 4, 2, p, SCORE_COSINE, missing_bins=missing
            ),
        )
        var audit = SharedSplitAudit.none()
        var want = _cpu_level(
            audit, words, 1, 4, 2, p, SCORE_COSINE, missing_bins=missing
        )
        assert_true(want.found)
        _assert_node_identical(got, want)


# --- The illegal leaf, which is where the two arms genuinely differ -------


def test_an_illegal_leaf_contributes_its_unsplit_terms_and_not_zero() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        # Leaf 1 is narrow at both ends and holds twenty of its twenty-six
        # rows in one bin, so `min_data_in_leaf = 4` puts it out of reach at
        # every candidate while leaf 0 stays admissible at all of them. The
        # legality test is on integer counts, so host and device agree exactly
        # about WHICH leaves are illegal and the comparison is about what an
        # illegal leaf contributes.
        var words = _level_words(
            1,
            5,
            _pair([-6, -4, 0, 3, 7], [-2, -1, 0, 1, 4]),
            _pair([2, 3, 4, 3, 3], [1, 1, 1, 1, 1]),
            _pair([20, 20, 20, 20, 20], [1, 2, 20, 2, 1]),
        )
        var p = _params(min_data_in_leaf=4)
        var audit = SharedSplitAudit.none()
        var want = _cpu_level(audit, words, 1, 5, 2, p, SCORE_COSINE)
        assert_true(want.found)
        # The fixture has to actually exercise the substitution, or the test
        # asserts nothing. The CPU grower's own audit is what says it does.
        assert_equal(audit.n_leaves, 2)
        assert_equal(audit.n_illegal, 1)

        var got = _search_level(words, 1, 5, 2, p, SCORE_COSINE)
        _assert_matches(got, _reference_level(words, 1, 5, 2, p, SCORE_COSINE))
        _assert_node_identical(got, want)

        # And "adds nothing", which is the L2 arm's rule, is a different
        # number here. Compared as a value for the reason the sum-of-ratios
        # comparison is.
        var zeroed = _reference_level(
            words, 1, 5, 2, p, SCORE_COSINE, illegal_adds_zero=True
        )
        var right = _reference_level(words, 1, 5, 2, p, SCORE_COSINE)
        assert_true(zeroed.gain.to_bits() != right.gain.to_bits())


def test_a_level_no_leaf_can_split_finds_nothing_under_cosine() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        # Every leaf illegal at every candidate, so every candidate
        # accumulates exactly the level's own unsplit terms in exactly the
        # order `level_parent` was accumulated in, and scores an exact 0.0.
        # `0.0` never beats a `best_gain` that starts at 0.0 under a strict
        # `>`, so nothing is found -- which is the property that lets the
        # top-threshold rule stay unbranched under Cosine.
        var words = _level_words(
            1, 4, _one([-4, -2, 2, 4]), _one([1, 1, 1, 1]), _one([3, 3, 3, 3])
        )
        var p = _params(min_data_in_leaf=100)
        var got = _search_level(words, 1, 4, 1, p, SCORE_COSINE)
        assert_false(got.found)


# --- The refusal, retired, and the one that is not ------------------------


def test_the_cosine_refusal_is_retired_and_an_unknown_code_is_not() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var words = _level_words(
            1,
            4,
            _one([-4, -2, 2, 4]),
            _one([1, 1, 1, 1]),
            _one([10, 10, 10, 10]),
        )
        var p = _params()
        # Retired, because the kernel now carries what it stood in for.
        var got = _search_level(words, 1, 4, 1, p, SCORE_COSINE)
        assert_true(got.found)
        # Not retired: an unknown selector must fail rather than receive an L2
        # answer under its own label.
        var raised = False
        try:
            _ = _search_level(words, 1, 4, 1, p, 7)
        except:
            raised = True
        assert_true(raised)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
