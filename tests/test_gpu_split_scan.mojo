"""The collective reductions choose the split the hand-rolled loops chose.

`gpu_split_search` now spells its reductions as `gpu.primitives.block`
collectives: `block.sum` and `block.prefix_sum` over the fixed-point Int32
sums, `block.max` and `block.min` over the Float32 gains and the candidate
positions that break their ties. `MOJOTREES_GPU_SPLIT_PRIMITIVES=0`, or
`GpuSplitSearcher.set_primitives(False)`, keeps the loops those replaced.

The claim under test is not that both arms find a good split. It is that
they find the *same* split, field for field, because every reduction that
was replaced is exact under reassociation: integer addition is associative,
and so are `max` and `min` on the values these kernels produce. A reduction
that moved one last bit, or that resolved one tie the other way, would be a
different tree and not a rounding difference, so the discrete fields are
compared exactly and so are the gains, which no arm recomputes.

Both arms are driven through `enqueue` rather than `search`, because
`enqueue` is the entry point that honors the searcher's `wide_scan` field
and therefore the only one that can reach the wide scan kernel at all.

Every histogram here is hand-built in the fixed-point layout the GPU
histogram kernels produce, with both scales set to 1.0 so every quantized
word dequantizes to itself exactly, which is the same convention
`tests/test_gpu_split_search.mojo` uses.

With no accelerator the device tests skip (passing) and the two host-side
tests, which pin the switch's default and its parameter form, still run.
"""

from std.sys import has_accelerator
from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_true,
    TestSuite,
)
from max.gpu.host import DeviceContext

from mojotrees.categorical import (
    CategoricalParams,
    CategoricalSpec,
    cat_contains,
)
from mojotrees.gpu_split_search import (
    GpuSplitParams,
    GpuSplitRecord,
    GpuSplitSearcher,
    MAX_SPLIT_BINS,
    REDUCE_SLOT_THREADS,
    WIDE_SCAN_THREADS,
    reference_search,
    split_primitives_requested,
)
from mojotrees.monotone import OutputBounds

comptime _TOL = 1e-4


# --- Tiny fixed-point histograms -----------------------------------------


def _zeroed(n: Int) -> List[Int32]:
    var out = List[Int32](capacity=n)
    out.resize(n, Int32(0))
    return out^


def _histogram_words(
    n_features: Int, n_bins: Int, g: List[Int], h: List[Int], c: List[Int]
) raises -> List[Int32]:
    """A `[grad | hess | count]` buffer from flat `[f * n_bins + b]` lists."""
    var size = n_features * n_bins
    if len(g) != size or len(h) != size or len(c) != size:
        raise Error("plane length must equal n_features * n_bins")
    var words = _zeroed(3 * size)
    for i in range(size):
        words[i] = Int32(g[i])
        words[size + i] = Int32(h[i])
        words[2 * size + i] = Int32(c[i])
    return words^


def _repeat_feature(
    n_features: Int,
    n_bins: Int,
    g: List[Int],
    h: List[Int],
    c: List[Int],
) raises -> List[Int32]:
    """The same one-feature histogram copied into every feature, so every
    feature slot scores exactly the same gain and the cross-feature tie-break
    is the only thing that can decide the winner."""
    var gg = List[Int](capacity=n_features * n_bins)
    var hh = List[Int](capacity=n_features * n_bins)
    var cc = List[Int](capacity=n_features * n_bins)
    for _ in range(n_features):
        for b in range(n_bins):
            gg.append(g[b])
            hh.append(h[b])
            cc.append(c[b])
    return _histogram_words(n_features, n_bins, gg, hh, cc)


def _params(
    lambda_l2: Float64 = 1.0,
    lambda_l1: Float64 = 0.0,
    min_child_hess: Float64 = 0.0,
    min_data_in_leaf: Int = 0,
    cat: CategoricalParams = CategoricalParams.default(),
) -> GpuSplitParams:
    return GpuSplitParams(
        lambda_l2, lambda_l1, min_child_hess, min_data_in_leaf, cat.copy()
    )


def _one_categorical(n_features: Int, n_categories: Int) -> CategoricalSpec:
    """Feature 0 categorical with `n_categories` codes, the rest numerical."""
    var flags = List[Bool](capacity=n_features)
    var offsets = List[Int](capacity=n_features + 1)
    var codes = List[Int](capacity=n_categories)
    for i in range(n_categories):
        codes.append(i)
    offsets.append(0)
    for f in range(n_features):
        flags.append(f == 0)
        offsets.append(n_categories if f == 0 else offsets[f])
    return CategoricalSpec(flags^, codes^, offsets^)


# --- The two arms ---------------------------------------------------------


def _arm(
    words: List[Int32],
    n_features: Int,
    n_bins: Int,
    params: GpuSplitParams,
    primitives: Bool,
    wide: Bool = False,
    features: List[Int] = [],
    allowed: List[Bool] = [],
    missing_bins: List[Int] = [],
    monotone: List[Int] = [],
    cats: CategoricalSpec = CategoricalSpec.none(),
    bounds: OutputBounds = OutputBounds.unbounded(),
) raises -> GpuSplitRecord:
    """One search, on the arm named by `primitives` and `wide`.

    The comptime guard keeps the device instantiation out of CPU-only
    builds: module-level helpers compile unconditionally, so without it a
    machine with no accelerator fails the arch constraint at compile time
    even though only guarded tests call this.

    `enqueue` rather than `search`, because `search` leaves the scan kernel
    at its serial default whatever `wide_scan` says, and this file has to
    reach both scan kernels. The histogram is a caller-owned buffer on the
    searcher's own context, which is the zero-copy shape the trainer uses.
    """
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var ctx = DeviceContext()
        var searcher = GpuSplitSearcher(
            ctx, n_features, n_bins, missing_bins, cats
        )
        searcher.set_primitives(primitives)
        searcher.wide_scan = wide
        if len(features) > 0:
            searcher.set_features(features)
        searcher.set_monotone(monotone)
        searcher.set_allowed(allowed)
        var hist = ctx.enqueue_create_buffer[DType.int32](len(words))
        ctx.enqueue_copy(dst_buf=hist, src_ptr=words.unsafe_ptr())
        ctx.synchronize()
        searcher.enqueue(hist, params, 1.0, 1.0, bounds)
        return searcher.download(0)


def _assert_identical(
    got: GpuSplitRecord, want: GpuSplitRecord, n_bins: Int
) raises:
    """Field for field, exactly.

    No tolerance anywhere, and that is the point of the file. The two arms
    run the same scan expressions over the same fixed-point words; what
    differs is only how the partial results are folded together, and every
    fold that was replaced is exact under reassociation. A tolerance here
    would hide precisely the failure this test exists to catch.
    """
    assert_equal(got.found, want.found)
    assert_equal(got.feature, want.feature)
    assert_equal(got.bin, want.bin)
    assert_equal(got.ordinal, want.ordinal)
    assert_equal(got.default_left, want.default_left)
    assert_equal(got.is_categorical, want.is_categorical)
    assert_equal(got.left.count, want.left.count)
    assert_equal(got.right.count, want.right.count)
    assert_equal(got.total.count, want.total.count)
    assert_equal(got.gain, want.gain)
    assert_equal(got.runner_gain, want.runner_gain)
    assert_equal(got.left.grad, want.left.grad)
    assert_equal(got.left.hess, want.left.hess)
    assert_equal(got.right.grad, want.right.grad)
    assert_equal(got.right.hess, want.right.hess)
    assert_equal(got.total.grad, want.total.grad)
    assert_equal(got.total.hess, want.total.hess)
    assert_equal(got.left_value, want.left_value)
    assert_equal(got.right_value, want.right_value)
    assert_equal(got.parent_value, want.parent_value)
    for b in range(n_bins):
        assert_equal(
            cat_contains(got.cat_bitset, b), cat_contains(want.cat_bitset, b)
        )


def _assert_arms_agree(
    words: List[Int32],
    n_features: Int,
    n_bins: Int,
    params: GpuSplitParams,
    features: List[Int] = [],
    allowed: List[Bool] = [],
    missing_bins: List[Int] = [],
    monotone: List[Int] = [],
    cats: CategoricalSpec = CategoricalSpec.none(),
    bounds: OutputBounds = OutputBounds.unbounded(),
) raises -> GpuSplitRecord:
    """Both reduction arms on the serial scan, plus the host reference as a
    third opinion on the discrete fields. Returns the primitive arm's
    record so a caller can assert what it actually chose."""
    var fast = _arm(
        words,
        n_features,
        n_bins,
        params,
        True,
        False,
        features,
        allowed,
        missing_bins,
        monotone,
        cats,
        bounds,
    )
    var slow = _arm(
        words,
        n_features,
        n_bins,
        params,
        False,
        False,
        features,
        allowed,
        missing_bins,
        monotone,
        cats,
        bounds,
    )
    _assert_identical(fast, slow, n_bins)
    var host = reference_search(
        words,
        n_features,
        n_bins,
        1.0,
        1.0,
        params,
        features,
        allowed,
        missing_bins,
        monotone,
        cats,
        bounds,
    )
    assert_equal(fast.found, host.found)
    if host.found:
        assert_equal(fast.feature, host.feature)
        assert_equal(fast.bin, host.bin)
        assert_equal(fast.ordinal, host.ordinal)
        assert_equal(fast.default_left, host.default_left)
        assert_equal(fast.is_categorical, host.is_categorical)
        assert_equal(fast.left.count, host.left.count)
        assert_almost_equal(fast.gain, host.gain, atol=_TOL)
        for b in range(n_bins):
            assert_equal(
                cat_contains(fast.cat_bitset, b),
                cat_contains(host.cat_bitset, b),
            )
    return fast^


def _assert_wide_arms_agree(
    words: List[Int32],
    n_features: Int,
    n_bins: Int,
    params: GpuSplitParams,
    features: List[Int] = [],
    allowed: List[Bool] = [],
    missing_bins: List[Int] = [],
    monotone: List[Int] = [],
    bounds: OutputBounds = OutputBounds.unbounded(),
) raises -> GpuSplitRecord:
    """The same comparison on the wide scan kernel, which is where
    `block.sum` and `block.prefix_sum` actually replaced something. Ordinal
    features only, which is the bar `wide_scan_for` applies."""
    var fast = _arm(
        words,
        n_features,
        n_bins,
        params,
        True,
        True,
        features,
        allowed,
        missing_bins,
        monotone,
        CategoricalSpec.none(),
        bounds,
    )
    var slow = _arm(
        words,
        n_features,
        n_bins,
        params,
        False,
        True,
        features,
        allowed,
        missing_bins,
        monotone,
        CategoricalSpec.none(),
        bounds,
    )
    _assert_identical(fast, slow, n_bins)
    return fast^


# --- The switch itself ----------------------------------------------------


def test_primitives_are_on_unless_refused() raises:
    # Host-side, so it runs with no accelerator. The variable follows the
    # package's `MOJOTREES_` contract: unset means the default, and the
    # default here is on, because both arms return the same record and only
    # the timing is unmeasured.
    # The test runner sets no `MOJOTREES_` variable, so this reads the
    # default, and the default is on. Only the exact string "0" turns the
    # collectives off, which is the package's contract for these switches:
    # a misspelled value is not a silent opt-out.
    assert_true(split_primitives_requested())
    # The threadgroup widths the collectives are instantiated at are warp
    # multiples on every supported backend; a non-multiple would silently
    # cost a partial warp in every reduction.
    assert_equal(REDUCE_SLOT_THREADS % 32, 0)
    assert_equal(WIDE_SCAN_THREADS % 32, 0)
    assert_true(REDUCE_SLOT_THREADS >= 32)


def test_the_setter_overrides_the_environment() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        # A benchmark has to hold both arms in one process, because
        # benchmarks on this machine drift across time windows and only
        # interleaved repeats compare. The setter is that handle, and it has
        # to survive being flipped between searches on one searcher.
        var ctx = DeviceContext()
        var searcher = GpuSplitSearcher(ctx, 1, 4)
        searcher.set_primitives(False)
        assert_false(searcher.use_primitives)
        assert_true(searcher.describe_scan().find("reduce=serial") >= 0)
        searcher.set_primitives(True)
        assert_true(searcher.use_primitives)
        assert_true(
            searcher.describe_scan().find("reduce=block-primitives") >= 0
        )


# --- Ties -----------------------------------------------------------------


def test_an_exact_tie_inside_one_feature_keeps_the_lower_bin() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        # One feature, four bins, lambda_l2 = 1, no L1, no missing bin.
        #   grad  -4  0  0  4     total 0
        #   hess   1  1  1  1     total 4
        #   count 10 10 10 10     total 40
        # parent_score = 0, so a candidate's gain is the sum of its two child
        # scores:
        #   bin 0: 16/(1+1) + 16/(3+1) = 8 + 4 = 12
        #   bin 1: 16/(2+1) + 16/(2+1) = 10.667
        #   bin 2: 16/(3+1) + 16/(1+1) = 4 + 8 = 12
        #   bin 3: every ordinary bin left with no missing rows, not a
        #          candidate at all.
        # Bins 0 and 2 tie at the maximum. The serial scan takes a candidate
        # only on a strictly greater gain and walks bins ascending, so bin 0
        # wins; a `max` reduction that resolved the tie the other way would
        # return a different tree with the same gain, which is exactly the
        # failure this asserts against.
        var words = _histogram_words(
            1, 4, [-4, 0, 0, 4], [1, 1, 1, 1], [10, 10, 10, 10]
        )
        var rec = _assert_arms_agree(words, 1, 4, _params())
        assert_true(rec.found)
        assert_equal(rec.bin, 0)
        assert_equal(rec.ordinal, 1)
        assert_almost_equal(rec.gain, 12.0, atol=_TOL)
        # The runner-up is the other tied candidate, so the record reports a
        # zero margin and `is_near_tie` sees it.
        assert_almost_equal(rec.runner_gain, 12.0, atol=_TOL)
        assert_true(rec.is_near_tie())

        # The same tie under the wide scan, where the two tied candidates
        # may land in different threads and the winner is decided by
        # `block.max` on the gain and `block.min` on the ordinal.
        var wide = _assert_wide_arms_agree(words, 1, 4, _params())
        assert_equal(wide.bin, 0)
        assert_equal(wide.ordinal, 1)


def test_an_exact_tie_across_features_keeps_the_lower_slot() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        # Every feature carries the identical histogram, so every feature
        # slot scores the identical maximum gain and the winner is decided
        # entirely by the cross-feature tie-break: the lowest slot, which is
        # what a single thread walking slots ascending on a strict `>`
        # gives. Seventy features against a sixty-four-thread reduction is
        # the ragged case: threads 0 through 5 hold two slots each, the rest
        # hold one, and the maximum is held by all of them at once.
        var words = _repeat_feature(
            70, 4, [-4, 0, 0, 4], [1, 1, 1, 1], [10, 10, 10, 10]
        )
        var rec = _assert_arms_agree(words, 70, 4, _params())
        assert_true(rec.found)
        assert_equal(rec.feature, 0)
        assert_equal(rec.bin, 0)
        # Every losing slot ties the winner, so the node's runner-up equals
        # its gain: a tie across features is a genuine near tie.
        assert_equal(rec.gain, rec.runner_gain)


def test_a_unique_winner_in_a_high_slot_survives_the_strided_fold() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        # The winner sits at slot 65, which thread 1 owns and thread 0 never
        # sees, so a fold that quietly kept thread 0's answer would fail
        # here and pass everywhere the winner happens to be early. The
        # runner-up sits at slot 3, owned by a third thread.
        #
        #   grad  -a  0  a  0     total 0, hess 4, parent_score 0
        #   bin 0: a^2/2 + a^2/4 = 0.75 a^2   <- unique maximum
        #   bin 1: a^2/3 + a^2/3 = 0.667 a^2
        #   bin 2: 0 + 0 = 0
        # so each feature has one strict best and the margins across
        # features are real rather than ties, which is what makes the
        # runner-up assertions below mean something.
        var n_features = 70
        var n_bins = 4
        var g = List[Int](capacity=n_features * n_bins)
        var h = List[Int](capacity=n_features * n_bins)
        var c = List[Int](capacity=n_features * n_bins)
        for f in range(n_features):
            for b in range(n_bins):
                var amplitude = 1
                if f == 65:
                    amplitude = 4
                elif f == 3:
                    amplitude = 2
                var value = 0
                if b == 0:
                    value = -amplitude
                elif b == 2:
                    value = amplitude
                g.append(value)
                h.append(1)
                c.append(10)
        var words = _histogram_words(n_features, n_bins, g, h, c)
        var rec = _assert_arms_agree(words, n_features, n_bins, _params())
        assert_true(rec.found)
        assert_equal(rec.feature, 65)
        assert_equal(rec.bin, 0)
        assert_almost_equal(rec.gain, 12.0, atol=_TOL)
        # The node's runner-up is the better of the best losing feature
        # (slot 3, at 0.75 * 4 = 3) and the winning feature's own second
        # candidate (0.667 * 16 = 10.667), so it is the latter, and it is
        # strictly below the winner. Both halves of that fold are collective
        # reductions now, and getting either wrong moves this number.
        assert_almost_equal(rec.runner_gain, 32.0 / 3.0, atol=_TOL)
        assert_true(rec.gain > rec.runner_gain)


# --- The missing bin ------------------------------------------------------


def test_the_missing_bin_routes_right_and_both_arms_agree() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        # Four bins with bin 3 reserved for missing rows, so the ordinary
        # scan covers bins 0..2 and each one is scored twice, missing-left
        # first.
        #   grad  -4  0  4  0     total 0, hess 4, parent_score 0
        #   bin 0 missing-left : 16/3 + 16/3 = 10.667
        #   bin 0 missing-right: 16/2 + 16/4 = 12
        #   bin 1 missing-left : 16/4 + 16/2 = 12
        #   bin 1 missing-right: 16/3 + 16/3 = 10.667
        # The two twelves tie, and the earlier one in scan order is bin 0's
        # missing-right candidate, ordinal 1. That pins two rules at once:
        # the direction ordering inside a bin, and the ascending bin order
        # across bins.
        var missing: List[Int] = [3]
        var right = _histogram_words(
            1, 4, [-4, 0, 4, 0], [1, 1, 1, 1], [10, 10, 10, 10]
        )
        var rec = _assert_arms_agree(
            right, 1, 4, _params(), [], [], missing
        )
        assert_true(rec.found)
        assert_false(rec.default_left)
        assert_equal(rec.bin, 0)
        assert_equal(rec.ordinal, 1)
        assert_equal(rec.left.count, 10)
        _ = _assert_wide_arms_agree(right, 1, 4, _params(), [], [], missing)

        # The mirror case, where grouping the missing rows with the left
        # child is what wins:
        #   grad  -4 -1  4 -3     total -4, hess 4, parent_score 16/5 = 3.2
        #   bin 1 missing-left: 64/4 + 16/2 - 3.2 = 20.8   <- winner
        # and no other candidate comes close, so `default_left` is decided
        # by the gain and not by a tie.
        var left = _histogram_words(
            1, 4, [-4, -1, 4, -3], [1, 1, 1, 1], [10, 10, 10, 10]
        )
        var rec2 = _assert_arms_agree(
            left, 1, 4, _params(), [], [], missing
        )
        assert_true(rec2.found)
        assert_true(rec2.default_left)
        assert_equal(rec2.bin, 1)
        assert_equal(rec2.ordinal, 2)
        assert_equal(rec2.left.count, 30)
        assert_almost_equal(rec2.gain, 20.8, atol=_TOL)
        _ = _assert_wide_arms_agree(left, 1, 4, _params(), [], [], missing)


# --- Categorical features -------------------------------------------------


def test_a_one_vs_rest_partition_survives_the_collective_fold() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        # Bin 0 is the reserved missing/unseen bin and bins 1..3 are the
        # categories; three categories is at or below the default
        # max_cat_to_onehot, so every category is tried one against the
        # rest. The category set travels as a bitset in the record's integer
        # words, and the fold copies the winning slot's words verbatim, so a
        # fold that picked the wrong slot would come back with the wrong
        # set and not merely the wrong gain.
        var words = _histogram_words(
            1, 4, [0, -6, 1, 5], [1, 1, 1, 1], [5, 10, 10, 10]
        )
        var cats = _one_categorical(1, 3)
        var rec = _assert_arms_agree(
            words, 1, 4, _params(), [], [], [], [], cats
        )
        assert_true(rec.found)
        assert_true(rec.is_categorical)
        assert_equal(rec.bin, -1)
        assert_true(cat_contains(rec.cat_bitset, 1))
        assert_false(cat_contains(rec.cat_bitset, 0))
        assert_false(cat_contains(rec.cat_bitset, 2))
        assert_false(cat_contains(rec.cat_bitset, 3))
        assert_almost_equal(rec.gain, 27.0, atol=_TOL)


def test_a_sorted_partition_survives_the_collective_fold() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        # Four categories, above max_cat_to_onehot, so the search sorts them
        # by grad / (hess + cat_smooth) and walks prefixes from both ends.
        # The backward walk wins with {3, 4}, which is a two-word bitset
        # written inside the scan rather than at the end of it.
        var words = _histogram_words(
            1, 5, [0, -8, -2, 2, 9], [1, 1, 1, 1, 1], [5, 10, 10, 10, 10]
        )
        var cats = _one_categorical(1, 4)
        var params = _params(cat=CategoricalParams(2, 32, 1.0, 0.0, 1))
        var rec = _assert_arms_agree(
            words, 1, 5, params, [], [], [], [], cats
        )
        assert_true(rec.found)
        assert_true(rec.is_categorical)
        assert_true(cat_contains(rec.cat_bitset, 3))
        assert_true(cat_contains(rec.cat_bitset, 4))
        assert_false(cat_contains(rec.cat_bitset, 1))
        assert_false(cat_contains(rec.cat_bitset, 2))


def test_a_categorical_feature_competing_with_numerical_ones() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        # Feature 0 categorical, features 1 and 2 numerical, all reduced by
        # the same fold, so the collective has to compare a categorical
        # slot's gain against ordinal ones and carry the winner's flags
        # through.
        var words = _histogram_words(
            3,
            4,
            [0, -6, 1, 5, -4, -2, 2, 4, -1, -1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
        )
        var cats = _one_categorical(3, 3)
        var rec = _assert_arms_agree(
            words, 3, 4, _params(), [], [], [], [], cats
        )
        assert_true(rec.found)
        assert_equal(rec.feature, 0)
        assert_true(rec.is_categorical)


# --- Node shape -----------------------------------------------------------


def test_a_node_whose_feature_set_is_a_strict_subset() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        # The strongest feature is 2, and the node's scan does not include
        # it, so the fold has to run over a slot count narrower than the
        # searcher's feature count and never look at the tail slots, whose
        # per-slot records the launch leaves alone. An interaction mask
        # removes one more feature after the subset is drawn, which is a
        # slot that reports no candidate at all.
        var words = _histogram_words(
            4,
            4,
            [-1, 0, 0, 1, -3, 0, 0, 3, -9, 0, 0, 9, -2, 0, 0, 2],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
        )
        var subset: List[Int] = [0, 1, 3]
        var rec = _assert_arms_agree(words, 4, 4, _params(), subset)
        assert_true(rec.found)
        assert_equal(rec.feature, 1)

        var mask: List[Bool] = [True, False, True, True]
        var rec2 = _assert_arms_agree(words, 4, 4, _params(), subset, mask)
        assert_true(rec2.found)
        assert_equal(rec2.feature, 3)

        # A node whose smaller row range is what the histogram counts:
        # the same shape, a tenth of the rows, and the scan and the fold
        # must behave identically on it.
        var child = _histogram_words(
            4,
            4,
            [-1, 0, 0, 1, -3, 0, 0, 3, -9, 0, 0, 9, -2, 0, 0, 2],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        )
        var rec3 = _assert_arms_agree(child, 4, 4, _params(), subset)
        assert_true(rec3.found)
        assert_equal(rec3.feature, 1)
        assert_equal(rec3.total.count, 4)


def test_a_node_too_small_for_min_data_in_leaf_finds_nothing() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        # Every candidate is rejected, on every feature, so no slot reports
        # a find and the fold takes its empty branch. That branch still owes
        # the caller the parent's totals and the parent's leaf value, which
        # is the part a reduction that returned early would drop.
        var words = _repeat_feature(
            70, 4, [-4, 0, 0, 4], [1, 1, 1, 1], [2, 2, 2, 2]
        )
        var params = _params(min_data_in_leaf=10)
        var rec = _assert_arms_agree(words, 70, 4, params)
        assert_false(rec.found)
        assert_equal(rec.feature, -1)
        assert_equal(rec.bin, -1)
        assert_equal(rec.ordinal, -1)
        assert_equal(rec.total.count, 8)
        assert_almost_equal(rec.total.grad, 0.0, atol=_TOL)
        assert_almost_equal(rec.total.hess, 4.0, atol=_TOL)
        assert_almost_equal(rec.parent_value, 0.0, atol=_TOL)
        assert_almost_equal(rec.gain, 0.0, atol=_TOL)

        # A hessian floor rejects everything for the other reason, and the
        # empty record has to look the same.
        var rec2 = _assert_arms_agree(
            words, 70, 4, _params(min_child_hess=100.0)
        )
        assert_false(rec2.found)
        assert_equal(rec2.total.count, 8)


# --- Shapes ---------------------------------------------------------------


def test_several_bin_counts_and_feature_counts() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        # Feature counts on both sides of the reduction's threadgroup width
        # and one that leaves a ragged tail: 1 and 7 are a partial first
        # wave, 64 fills it exactly, 65 and 100 leave threads holding two
        # slots while the rest hold one. Bin counts run from the narrowest
        # histogram a split can come out of up to the widest the module
        # accepts.
        var feature_counts: List[Int] = [1, 7, 64, 65, 100]
        var bin_counts: List[Int] = [2, 3, 16, 65, MAX_SPLIT_BINS]
        for fi in range(len(feature_counts)):
            var nf = feature_counts[fi]
            for bi in range(len(bin_counts)):
                var nb = bin_counts[bi]
                var g = List[Int](capacity=nf * nb)
                var h = List[Int](capacity=nf * nb)
                var c = List[Int](capacity=nf * nb)
                for f in range(nf):
                    for b in range(nb):
                        # A gradient that changes sign partway along, so the
                        # best threshold sits in the interior, with the
                        # amplitude varying by feature so the winner is a
                        # different slot for a different feature count, and
                        # with exact repeats along the bins so ties are
                        # reached rather than avoided.
                        g.append((1 + f % 3) * ((b * 4) // nb - 1))
                        h.append(1 + (b % 3))
                        c.append(4 + (b % 5))
                var words = _histogram_words(nf, nb, g, h, c)
                var rec = _assert_arms_agree(words, nf, nb, _params())
                assert_true(rec.found)


def test_the_wide_scan_agrees_across_bin_counts() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        # The wide scan is where `block.sum` and
        # `block.prefix_sum[exclusive=True]` replaced a halving tree and a
        # per-thread walk over the chunk sums. Fewer bins than threads
        # leaves most chunks empty; more bins than threads gives every
        # thread a multi-bin chunk, which is the only case where the
        # exclusive prefix carries anything at all.
        var bin_counts: List[Int] = [
            2,
            16,
            WIDE_SCAN_THREADS,
            WIDE_SCAN_THREADS + 1,
            200,
            MAX_SPLIT_BINS,
        ]
        for bi in range(len(bin_counts)):
            var nb = bin_counts[bi]
            var g = List[Int](capacity=2 * nb)
            var h = List[Int](capacity=2 * nb)
            var c = List[Int](capacity=2 * nb)
            for f in range(2):
                for b in range(nb):
                    g.append((1 + f) * (b - nb // 2))
                    h.append(1 + (b % 3))
                    c.append(4 + (b % 5))
            var words = _histogram_words(2, nb, g, h, c)
            var rec = _assert_wide_arms_agree(words, 2, nb, _params())
            assert_true(rec.found)
            if nb > 2:
                assert_true(rec.bin > 0 and rec.bin < nb - 1)

        # A monotone constraint, which rejects candidates inside the scan
        # rather than before it, so the per-thread bests it leaves behind
        # are sparser than the chunk partition.
        var mono: List[Int] = [1]
        var shaped = _histogram_words(
            1, 4, [4, 2, -2, -4], [1, 1, 1, 1], [10, 10, 10, 10]
        )
        _ = _assert_wide_arms_agree(
            shaped, 1, 4, _params(), [], [], [], mono
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
