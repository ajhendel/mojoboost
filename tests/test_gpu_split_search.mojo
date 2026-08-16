"""Device-side split search.

Every histogram here is tiny and hand-built in the fixed-point layout the GPU
histogram kernels produce, with both scales set to 1.0 so every quantized word
dequantizes to itself exactly. Each expected gain is therefore an ordinary
fraction, worked out by hand from the second-order formula, and each expected
winner is a decision that can be checked by reading the numbers rather than by
trusting another implementation.

Three layers are asserted:

- The arithmetic. Hand-computed gains, child statistics, and leaf values
  against `reference_search`, which shares its Float32 helpers with the
  kernels.
- The decisions. `reference_search` against the host `find_best_split` on the
  same histogram, so missing-value routing, tie-breaking, rejection rules,
  interaction masks, monotone constraints, and both categorical searches are
  pinned to the semantics the CPU trainer already ships.
- The device. When an accelerator is present, `GpuSplitSearcher` against
  `reference_search`, and against itself twice over, since the GPU path must
  be bit-deterministic run to run. Skips (passing) with no accelerator, so
  the first two layers still run on a CPU-only machine.
"""

from std.sys import has_accelerator
from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_true,
    TestSuite,
)

from mojotrees.categorical import (
    CategoricalParams,
    CategoricalSpec,
    cat_contains,
)
from mojotrees.gpu_split_search import (
    GpuSplitParams,
    GpuSplitRecord,
    GpuSplitSearcher,
    reference_search,
    wide_scan_for,
    wide_scan_requested,
)
from mojotrees.histogram import Histogram
from mojotrees.monotone import OutputBounds
from mojotrees.split import find_best_split

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


def _as_histogram(
    words: List[Int32], n_features: Int, n_bins: Int
) -> Histogram:
    """The same numbers as the Float64 `Histogram` the host scan takes. Both
    scales are 1.0 in these tests, so this conversion is exact and the two
    searches see identical values."""
    var size = n_features * n_bins
    var grad = List[Float64](capacity=size)
    var hess = List[Float64](capacity=size)
    var count = List[Int](capacity=size)
    for i in range(size):
        grad.append(Float64(words[i]))
        hess.append(Float64(words[size + i]))
        count.append(Int(words[2 * size + i]))
    return Histogram.from_planes(grad^, hess^, count^, n_features, n_bins)


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


# --- Cross-checks ---------------------------------------------------------


def _assert_agrees_with_host(
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
) raises:
    """The device search's decision must be the host's decision. Gains agree
    to Float32 precision; everything discrete agrees exactly."""
    var got = reference_search(
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
    var want = find_best_split(
        _as_histogram(words, n_features, n_bins),
        lambda_reg=params.lambda_l2,
        min_child_hess=params.min_child_hess,
        min_data_in_leaf=params.min_data_in_leaf,
        lambda_l1=params.lambda_l1,
        allowed=allowed,
        features=features,
        missing_bins=missing_bins,
        monotone=monotone,
        bounds=bounds,
        cats=cats,
        cat_params=params.cat,
    )
    assert_equal(got.found, want.found)
    if not want.found:
        return
    assert_equal(got.feature, want.feature)
    assert_equal(got.is_categorical, want.is_categorical)
    assert_equal(got.bin, want.bin)
    assert_equal(got.default_left, want.default_left)
    assert_almost_equal(got.gain, want.gain, atol=_TOL)
    if want.is_categorical:
        for b in range(n_bins):
            assert_equal(
                cat_contains(got.cat_bitset, b),
                cat_contains(want.cat_bitset, b),
            )


def _assert_same_record(
    got: GpuSplitRecord, want: GpuSplitRecord, n_bins: Int
) raises:
    """Discrete fields exactly, Float32 quantities to tolerance: the device
    may contract a multiply and an add into one fused instruction, which the
    host reference does not."""
    assert_equal(got.found, want.found)
    assert_equal(got.feature, want.feature)
    assert_equal(got.bin, want.bin)
    assert_equal(got.ordinal, want.ordinal)
    assert_equal(got.default_left, want.default_left)
    assert_equal(got.is_categorical, want.is_categorical)
    assert_equal(got.left.count, want.left.count)
    assert_equal(got.right.count, want.right.count)
    assert_equal(got.total.count, want.total.count)
    assert_almost_equal(got.gain, want.gain, atol=_TOL)
    assert_almost_equal(got.left.grad, want.left.grad, atol=_TOL)
    assert_almost_equal(got.left.hess, want.left.hess, atol=_TOL)
    assert_almost_equal(got.right.grad, want.right.grad, atol=_TOL)
    assert_almost_equal(got.right.hess, want.right.hess, atol=_TOL)
    assert_almost_equal(got.total.grad, want.total.grad, atol=_TOL)
    assert_almost_equal(got.total.hess, want.total.hess, atol=_TOL)
    assert_almost_equal(got.left_value, want.left_value, atol=_TOL)
    assert_almost_equal(got.right_value, want.right_value, atol=_TOL)
    assert_almost_equal(got.parent_value, want.parent_value, atol=_TOL)
    for b in range(n_bins):
        assert_equal(
            cat_contains(got.cat_bitset, b), cat_contains(want.cat_bitset, b)
        )


def _device_search(
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
    # The comptime guard keeps the device instantiation out of CPU-only
    # builds: module-level helpers compile unconditionally, so without it a
    # machine with no accelerator fails the arch constraint at compile time
    # even though only guarded tests call this.
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var searcher = GpuSplitSearcher(
            n_features, n_bins, missing_bins, cats
        )
        if len(features) > 0:
            searcher.set_features(features)
        searcher.set_monotone(monotone)
        searcher.set_allowed(allowed)
        searcher.upload_histogram(words)
        return searcher.search(params, 1.0, 1.0, bounds)


def _device_search_wide(
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
    """`_device_search` on the wide scan kernel, forced on rather than asked
    for through `MOJOTREES_GPU_SPLIT_WIDE`, so both kernels are exercised in
    one process and the comparison is between them and nothing else. Only
    ordinal datasets reach this, which is the same bar `wide_scan_for`
    applies."""
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var searcher = GpuSplitSearcher(n_features, n_bins, missing_bins)
        searcher.wide_scan = True
        if len(features) > 0:
            searcher.set_features(features)
        searcher.set_monotone(monotone)
        searcher.set_allowed(allowed)
        searcher.upload_histogram(words)
        return searcher.search(params, 1.0, 1.0, bounds)


# --- Numerical scan -------------------------------------------------------


def test_numerical_scan_matches_hand_computed_gains() raises:
    # One feature, four bins, no missing bin. lambda_l2 = 1, no L1.
    #   grad  -4 -2  2  4     total 0
    #   hess   1  1  1  1     total 4
    #   count 10 10 10 10     total 40
    # parent_score = 0^2 / (4 + 1) = 0, so each candidate's gain is just the
    # sum of its two child scores:
    #   bin 0:  16/(1+1) + 16/(3+1) =  8   +  4   = 12
    #   bin 1:  36/(2+1) + 36/(2+1) = 12   + 12   = 24
    #   bin 2:  16/(3+1) + 16/(1+1) =  4   +  8   = 12
    #   bin 3:  every ordinary bin left, no missing rows to fill the right
    #           child, so it is not a candidate at all.
    var words = _histogram_words(
        1, 4, [-4, -2, 2, 4], [1, 1, 1, 1], [10, 10, 10, 10]
    )
    var rec = reference_search(words, 1, 4, 1.0, 1.0, _params())
    assert_true(rec.found)
    assert_equal(rec.feature, 0)
    assert_equal(rec.bin, 1)
    assert_false(rec.is_categorical)
    assert_false(rec.default_left)
    # Missing rows to the right is the second candidate at bin 1.
    assert_equal(rec.ordinal, 3)
    assert_almost_equal(rec.gain, 24.0, atol=_TOL)

    # Child statistics come off the exact integer histogram.
    assert_almost_equal(rec.left.grad, -6.0, atol=_TOL)
    assert_almost_equal(rec.left.hess, 2.0, atol=_TOL)
    assert_equal(rec.left.count, 20)
    assert_almost_equal(rec.right.grad, 6.0, atol=_TOL)
    assert_almost_equal(rec.right.hess, 2.0, atol=_TOL)
    assert_equal(rec.right.count, 20)
    assert_equal(rec.total.count, 40)

    # Leaf values are the unclamped Newton steps -T(G) / (H + lambda_l2).
    assert_almost_equal(rec.left_value, 2.0, atol=_TOL)
    assert_almost_equal(rec.right_value, -2.0, atol=_TOL)
    assert_almost_equal(rec.parent_value, 0.0, atol=_TOL)

    _assert_agrees_with_host(words, 1, 4, _params())


def test_l1_soft_thresholds_every_gradient_sum() raises:
    # Same histogram with lambda_l1 = 1: every child gradient sum shrinks
    # toward zero by 1 before it is squared.
    #   bin 1: T(-6) = -5, T(6) = 5 -> 25/3 + 25/3 = 16.6667
    #          parent T(0) = 0, so the shift stays 0.
    var words = _histogram_words(
        1, 4, [-4, -2, 2, 4], [1, 1, 1, 1], [10, 10, 10, 10]
    )
    var params = _params(lambda_l1=1.0)
    var rec = reference_search(words, 1, 4, 1.0, 1.0, params)
    assert_true(rec.found)
    assert_equal(rec.bin, 1)
    assert_almost_equal(rec.gain, 50.0 / 3.0, atol=_TOL)
    # -T(-6) / (2 + 1) = 5/3.
    assert_almost_equal(rec.left_value, 5.0 / 3.0, atol=_TOL)
    _assert_agrees_with_host(words, 1, 4, params)


def test_ties_go_to_the_lowest_bin_and_the_lowest_feature() raises:
    # grad -3 0 0 3 gives bin 0 and bin 2 the same gain, 9/2 + 9/4 = 6.75,
    # and bin 1 a strictly smaller one, 9/3 + 9/3 = 6. The first candidate in
    # scan order has to win.
    var words = _histogram_words(
        1, 4, [-3, 0, 0, 3], [1, 1, 1, 1], [10, 10, 10, 10]
    )
    var rec = reference_search(words, 1, 4, 1.0, 1.0, _params())
    assert_true(rec.found)
    assert_equal(rec.bin, 0)
    assert_almost_equal(rec.gain, 6.75, atol=_TOL)
    _assert_agrees_with_host(words, 1, 4, _params())

    # Two identical features tie on every candidate, so the lower feature id
    # wins the cross-feature reduction.
    var two = _histogram_words(
        2,
        4,
        [-3, 0, 0, 3, -3, 0, 0, 3],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [10, 10, 10, 10, 10, 10, 10, 10],
    )
    var pair = reference_search(two, 2, 4, 1.0, 1.0, _params())
    assert_true(pair.found)
    assert_equal(pair.feature, 0)
    assert_equal(pair.bin, 0)
    _assert_agrees_with_host(two, 2, 4, _params())


def test_min_child_hess_and_min_data_in_leaf_reject_candidates() raises:
    var words = _histogram_words(
        1, 4, [-4, -2, 2, 4], [1, 1, 1, 1], [10, 10, 10, 10]
    )
    # min_child_hess = 1.5 leaves only bin 1, whose children hold 2 each.
    var by_hess = reference_search(
        words, 1, 4, 1.0, 1.0, _params(min_child_hess=1.5)
    )
    assert_true(by_hess.found)
    assert_equal(by_hess.bin, 1)
    _assert_agrees_with_host(words, 1, 4, _params(min_child_hess=1.5))

    # min_data_in_leaf = 25 rejects every candidate: the three splits give
    # (10, 30), (20, 20), and (30, 10).
    var none = reference_search(
        words, 1, 4, 1.0, 1.0, _params(min_data_in_leaf=25)
    )
    assert_false(none.found)
    assert_equal(none.feature, -1)
    assert_equal(none.total.count, 40)
    _assert_agrees_with_host(words, 1, 4, _params(min_data_in_leaf=25))


def test_no_split_still_reports_the_parent_leaf_value() raises:
    # A single-bin feature has no threshold at all, but the node still needs
    # its own value: -5 / (2 + 1).
    var words = _histogram_words(1, 1, [5], [2], [10])
    var rec = reference_search(words, 1, 1, 1.0, 1.0, _params())
    assert_false(rec.found)
    assert_almost_equal(rec.parent_value, -5.0 / 3.0, atol=_TOL)
    assert_almost_equal(rec.total.grad, 5.0, atol=_TOL)
    assert_almost_equal(rec.total.hess, 2.0, atol=_TOL)
    assert_equal(rec.total.count, 10)
    assert_equal(rec.to_split_info().found, False)


# --- Missing values -------------------------------------------------------


def test_missing_rows_choose_their_direction() raises:
    # Three bins, bin 2 reserved for missing rows, so bins 0 and 1 are the
    # ordinary ones.
    #   grad  -5  5 -5    ordinary total 0, missing -5, total -5
    #   hess   1  1  1
    #   count 10 10 10
    # parent_score = 25 / (3 + 1) = 6.25.
    #   bin 0, missing left:  100/(2+1) + 25/(1+1) - 6.25 = 39.5833
    #   bin 0, missing right:  25/(1+1) +  0/(2+1) - 6.25 =  6.25
    # so the winner sends missing rows left.
    var words = _histogram_words(1, 3, [-5, 5, -5], [1, 1, 1], [10, 10, 10])
    var missing: List[Int] = [2]
    var rec = reference_search(
        words, 1, 3, 1.0, 1.0, _params(), [], [], missing
    )
    assert_true(rec.found)
    assert_equal(rec.bin, 0)
    assert_true(rec.default_left)
    # Missing left is the first candidate at bin 0.
    assert_equal(rec.ordinal, 0)
    assert_almost_equal(rec.gain, 100.0 / 3.0 + 12.5 - 6.25, atol=_TOL)
    # The left child holds the missing rows.
    assert_almost_equal(rec.left.grad, -10.0, atol=_TOL)
    assert_equal(rec.left.count, 20)
    _assert_agrees_with_host(words, 1, 3, _params(), [], [], missing)


def test_top_ordinary_bin_is_a_candidate_only_with_missing_rows() raises:
    # Bins 0 and 1 ordinary, bin 2 missing and non-empty. The best split
    # puts both ordinary bins left and the missing rows alone on the right:
    #   parent_score = 64 / (3 + 1) = 16
    #   bin 1, missing right: 4/(2+1) + 100/(1+1) - 16 = 35.3333
    var words = _histogram_words(1, 3, [-1, -1, 10], [1, 1, 1], [10, 10, 10])
    var missing: List[Int] = [2]
    var rec = reference_search(
        words, 1, 3, 1.0, 1.0, _params(), [], [], missing
    )
    assert_true(rec.found)
    assert_equal(rec.bin, 1)
    assert_false(rec.default_left)
    assert_equal(rec.ordinal, 3)
    assert_almost_equal(rec.gain, 4.0 / 3.0 + 50.0 - 16.0, atol=_TOL)
    assert_equal(rec.right.count, 10)
    _assert_agrees_with_host(words, 1, 3, _params(), [], [], missing)

    # With the missing bin empty the same threshold disappears, because the
    # right child would be empty: only bin 0 remains a candidate.
    var empty = _histogram_words(1, 3, [-4, 4, 0], [1, 1, 0], [10, 10, 0])
    var no_miss = reference_search(
        empty, 1, 3, 1.0, 1.0, _params(), [], [], missing
    )
    assert_true(no_miss.found)
    assert_equal(no_miss.bin, 0)
    # A node holding no missing rows records the left default, which is the
    # direction an unseen missing value follows at prediction time.
    assert_true(no_miss.default_left)
    assert_almost_equal(no_miss.gain, 16.0, atol=_TOL)
    _assert_agrees_with_host(empty, 1, 3, _params(), [], [], missing)


# --- Constraints ----------------------------------------------------------


def test_monotone_constraint_rejects_a_candidate() raises:
    # Unconstrained, bin 0 wins with 16/2 + 16/3 = 13.3333. Under a
    # nondecreasing constraint its children's outputs are 2 and -1.3333,
    # which run the wrong way, so the candidate scores zero and no split
    # survives.
    var words = _histogram_words(1, 3, [-4, 4, 0], [1, 1, 1], [10, 10, 10])
    var free = reference_search(words, 1, 3, 1.0, 1.0, _params())
    assert_true(free.found)
    assert_equal(free.bin, 0)
    assert_almost_equal(free.gain, 8.0 + 16.0 / 3.0, atol=_TOL)

    var signs: List[Int] = [1]
    var held = reference_search(
        words, 1, 3, 1.0, 1.0, _params(), [], [], [], signs
    )
    assert_false(held.found)
    _assert_agrees_with_host(words, 1, 3, _params(), [], [], [], signs)

    # A nonincreasing constraint accepts the same candidate.
    var down: List[Int] = [-1]
    var ok = reference_search(
        words, 1, 3, 1.0, 1.0, _params(), [], [], [], down
    )
    assert_true(ok.found)
    assert_equal(ok.bin, 0)
    _assert_agrees_with_host(words, 1, 3, _params(), [], [], [], down)
    # And a bounded node scores its candidates from clamped outputs.
    _assert_agrees_with_host(
        words,
        1,
        3,
        _params(),
        [],
        [],
        [],
        down,
        CategoricalSpec.none(),
        OutputBounds(-1.0, 1.0),
    )


def test_interaction_mask_and_feature_subset_narrow_the_scan() raises:
    # Feature 0 holds the stronger split (24 at bin 1) and feature 1 a weaker
    # one (9/3 + 9/3 = 6, also at bin 1).
    var words = _histogram_words(
        2,
        4,
        [-4, -2, 2, 4, -1, -2, 2, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [10, 10, 10, 10, 10, 10, 10, 10],
    )
    var both = reference_search(words, 2, 4, 1.0, 1.0, _params())
    assert_equal(both.feature, 0)

    # Masked off by the node's interaction constraints.
    var masked = reference_search(
        words, 2, 4, 1.0, 1.0, _params(), [], [False, True]
    )
    assert_true(masked.found)
    assert_equal(masked.feature, 1)
    _assert_agrees_with_host(words, 2, 4, _params(), [], [False, True])

    # Or simply not drawn by feature subsampling.
    var subset = reference_search(words, 2, 4, 1.0, 1.0, _params(), [1])
    assert_true(subset.found)
    assert_equal(subset.feature, 1)
    _assert_agrees_with_host(words, 2, 4, _params(), [1])


# --- Categorical features -------------------------------------------------


def test_categorical_onehot_partition() raises:
    # Four bins: bin 0 is the reserved missing/unseen/dropped bin and bins
    # 1..3 are the categories. Three categories is at or below the default
    # max_cat_to_onehot, so every category is tried one against the rest:
    #   parent_score = 0
    #   {1}: 36/(1+1) + 36/(3+1) = 27
    #   {2}:  1/(1+1) +  1/(3+1) =  0.75
    #   {3}: 25/(1+1) + 25/(3+1) = 18.75
    var words = _histogram_words(
        1, 4, [0, -6, 1, 5], [1, 1, 1, 1], [5, 10, 10, 10]
    )
    var cats = _one_categorical(1, 3)
    var rec = reference_search(
        words, 1, 4, 1.0, 1.0, _params(), [], [], [], [], cats
    )
    assert_true(rec.found)
    assert_true(rec.is_categorical)
    assert_equal(rec.feature, 0)
    assert_equal(rec.bin, -1)
    assert_false(rec.default_left)
    assert_almost_equal(rec.gain, 27.0, atol=_TOL)
    assert_true(cat_contains(rec.cat_bitset, 1))
    assert_false(cat_contains(rec.cat_bitset, 0))
    assert_false(cat_contains(rec.cat_bitset, 2))
    assert_false(cat_contains(rec.cat_bitset, 3))
    assert_almost_equal(rec.left.grad, -6.0, atol=_TOL)
    assert_equal(rec.left.count, 10)
    # The unknown bin's rows route right with everything not in the set.
    assert_equal(rec.right.count, 25)
    _assert_agrees_with_host(
        words, 1, 4, _params(), [], [], [], [], cats
    )

    var info = rec.to_split_info()
    assert_true(info.is_categorical)
    assert_true(info.goes_left(1))
    assert_false(info.goes_left(0))
    assert_false(info.goes_left(3))


def test_categorical_sorted_partition() raises:
    # Five bins: the unknown bin plus four categories, above
    # max_cat_to_onehot, so the search sorts categories by
    # grad / (hess + cat_smooth) and walks prefixes from both ends.
    #   keys: 1 -> -4, 2 -> -1, 3 -> 1, 4 -> 4.5, so the order is 1,2,3,4
    #   total grad 1, hess 5; parent_score = 1 / (5 + 1) = 0.16667
    #   max_cat_threshold caps the walk at (4 + 1) // 2 = 2 steps
    #   forward  {1}   : 64/2 + 81/5 - 0.16667 = 48.0333
    #   forward  {1,2} : 100/3 + 121/4 - 0.16667 = 63.4167
    #   backward {4}   : 81/2 + 64/5 - 0.16667 = 53.1333
    #   backward {3,4} : 121/3 + 100/4 - 0.16667 = 65.1667   <- winner
    var words = _histogram_words(
        1, 5, [0, -8, -2, 2, 9], [1, 1, 1, 1, 1], [5, 10, 10, 10, 10]
    )
    var cats = _one_categorical(1, 4)
    var cat_params = CategoricalParams(2, 32, 1.0, 0.0, 1)
    var params = _params(cat=cat_params)
    var rec = reference_search(
        words, 1, 5, 1.0, 1.0, params, [], [], [], [], cats
    )
    assert_true(rec.found)
    assert_true(rec.is_categorical)
    assert_almost_equal(
        rec.gain, 121.0 / 3.0 + 25.0 - 1.0 / 6.0, atol=_TOL
    )
    assert_true(cat_contains(rec.cat_bitset, 3))
    assert_true(cat_contains(rec.cat_bitset, 4))
    assert_false(cat_contains(rec.cat_bitset, 1))
    assert_false(cat_contains(rec.cat_bitset, 2))
    assert_false(cat_contains(rec.cat_bitset, 0))
    assert_almost_equal(rec.left.grad, 11.0, atol=_TOL)
    assert_equal(rec.left.count, 20)
    assert_equal(rec.right.count, 25)
    _assert_agrees_with_host(words, 1, 5, params, [], [], [], [], cats)


def test_categorical_and_numerical_compete_on_one_gain() raises:
    # Feature 0 categorical, feature 1 numerical, scored by the same formula
    # and reduced in one pass; the stronger candidate wins whichever kind it
    # is.
    var words = _histogram_words(
        2,
        4,
        [0, -6, 1, 5, -4, -2, 2, 4],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [5, 10, 10, 10, 10, 10, 10, 10],
    )
    var cats = _one_categorical(2, 3)
    var rec = reference_search(
        words, 2, 4, 1.0, 1.0, _params(), [], [], [], [], cats
    )
    assert_true(rec.found)
    # 27 (categorical, feature 0) beats 24 (numerical, feature 1).
    assert_equal(rec.feature, 0)
    assert_true(rec.is_categorical)
    _assert_agrees_with_host(words, 2, 4, _params(), [], [], [], [], cats)


def test_record_round_trips_to_split_info() raises:
    var words = _histogram_words(
        1, 4, [-4, -2, 2, 4], [1, 1, 1, 1], [10, 10, 10, 10]
    )
    var rec = reference_search(words, 1, 4, 1.0, 1.0, _params())
    var info = rec.to_split_info()
    assert_true(info.found)
    assert_equal(info.feature, 0)
    assert_equal(info.bin, 1)
    assert_false(info.is_categorical)
    assert_true(info.goes_left(1))
    assert_false(info.goes_left(2))
    assert_almost_equal(info.gain, rec.gain, atol=_TOL)


# --- Device ---------------------------------------------------------------


def test_device_search_matches_the_reference() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        # Numerical, no missing bin.
        var plain = _histogram_words(
            2,
            4,
            [-4, -2, 2, 4, -1, -2, 2, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [10, 10, 10, 10, 10, 10, 10, 10],
        )
        _assert_same_record(
            _device_search(plain, 2, 4, _params()),
            reference_search(plain, 2, 4, 1.0, 1.0, _params()),
            4,
        )

        # Missing bin, both directions exercised.
        var missing: List[Int] = [2]
        var miss = _histogram_words(
            1, 3, [-5, 5, -5], [1, 1, 1], [10, 10, 10]
        )
        _assert_same_record(
            _device_search(miss, 1, 3, _params(), [], [], missing),
            reference_search(
                miss, 1, 3, 1.0, 1.0, _params(), [], [], missing
            ),
            3,
        )

        # Rejection rules and an interaction mask.
        _assert_same_record(
            _device_search(
                plain, 2, 4, _params(min_data_in_leaf=25), [], [False, True]
            ),
            reference_search(
                plain,
                2,
                4,
                1.0,
                1.0,
                _params(min_data_in_leaf=25),
                [],
                [False, True],
            ),
            4,
        )

        # A monotone constraint that rejects the best candidate.
        var mono = _histogram_words(
            1, 3, [-4, 4, 0], [1, 1, 1], [10, 10, 10]
        )
        var signs: List[Int] = [1]
        _assert_same_record(
            _device_search(mono, 1, 3, _params(), [], [], [], signs),
            reference_search(
                mono, 1, 3, 1.0, 1.0, _params(), [], [], [], signs
            ),
            3,
        )

        # One-vs-rest categories.
        var onehot = _histogram_words(
            1, 4, [0, -6, 1, 5], [1, 1, 1, 1], [5, 10, 10, 10]
        )
        var cats3 = _one_categorical(1, 3)
        _assert_same_record(
            _device_search(
                onehot, 1, 4, _params(), [], [], [], [], cats3
            ),
            reference_search(
                onehot, 1, 4, 1.0, 1.0, _params(), [], [], [], [], cats3
            ),
            4,
        )

        # The sorted many-vs-many walk.
        var sorted_words = _histogram_words(
            1, 5, [0, -8, -2, 2, 9], [1, 1, 1, 1, 1], [5, 10, 10, 10, 10]
        )
        var cats4 = _one_categorical(1, 4)
        var sorted_params = _params(cat=CategoricalParams(2, 32, 1.0, 0.0, 1))
        _assert_same_record(
            _device_search(
                sorted_words, 1, 5, sorted_params, [], [], [], [], cats4
            ),
            reference_search(
                sorted_words,
                1,
                5,
                1.0,
                1.0,
                sorted_params,
                [],
                [],
                [],
                [],
                cats4,
            ),
            5,
        )


def test_device_search_is_deterministic() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var words = _histogram_words(
            2,
            4,
            [-4, -2, 2, 4, -1, -2, 2, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [10, 10, 10, 10, 10, 10, 10, 10],
        )
        var searcher = GpuSplitSearcher(2, 4)
        searcher.upload_histogram(words)
        var first = searcher.search(_params(), 1.0, 1.0)
        var second = searcher.search(_params(), 1.0, 1.0)
        # No atomics and a fixed reduction order, so the two runs have to
        # agree bit for bit, not merely to tolerance.
        assert_equal(first.gain, second.gain)
        assert_equal(first.left_value, second.left_value)
        assert_equal(first.feature, second.feature)
        assert_equal(first.bin, second.bin)
        assert_equal(first.ordinal, second.ordinal)


def test_device_frontier_pick_best() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        # Two leaves' records held on the device at once, reduced there to
        # the one the grower would split next.
        var weak = _histogram_words(
            1, 4, [-1, -2, 2, 1], [1, 1, 1, 1], [10, 10, 10, 10]
        )
        var strong = _histogram_words(
            1, 4, [-4, -2, 2, 4], [1, 1, 1, 1], [10, 10, 10, 10]
        )
        var searcher = GpuSplitSearcher(1, 4, [], CategoricalSpec.none(), 3)

        searcher.upload_histogram(weak)
        var weak_rec = searcher.search(_params(), 1.0, 1.0, record=0)
        searcher.upload_histogram(strong)
        var strong_rec = searcher.search(_params(), 1.0, 1.0, record=1)
        assert_true(strong_rec.gain > weak_rec.gain)

        searcher.enqueue_pick_best(2, record=2)
        var best = searcher.download(2)
        _assert_same_record(best, strong_rec, 4)


def test_wide_scan_is_refused_for_a_categorical_dataset() raises:
    # Host-side, so it runs without an accelerator: the wide kernel scans
    # ordinal features only, and the gate is per dataset rather than per
    # feature so one launch is one kernel.
    assert_false(wide_scan_for(True))
    assert_equal(wide_scan_for(False), wide_scan_requested())


def test_wide_scan_matches_the_serial_scan() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        # Every rule the ordinal scan enforces, run through both kernels.
        # The wide one splits a feature's bins across a threadgroup and the
        # serial one walks them on a lane, so agreement here is the claim in
        # `_scan_slot_wide_kernel`'s docstring: the same record, not a
        # similar one.
        var plain = _histogram_words(
            2,
            4,
            [-4, -2, 2, 4, -1, -2, 2, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [10, 10, 10, 10, 10, 10, 10, 10],
        )
        _assert_same_record(
            _device_search_wide(plain, 2, 4, _params()),
            _device_search(plain, 2, 4, _params()),
            4,
        )

        # Missing bin, so both candidate directions are scored per bin and
        # the default-left tie rule is exercised.
        var missing: List[Int] = [2]
        var miss = _histogram_words(
            1, 3, [-5, 5, -5], [1, 1, 1], [10, 10, 10]
        )
        _assert_same_record(
            _device_search_wide(miss, 1, 3, _params(), [], [], missing),
            _device_search(miss, 1, 3, _params(), [], [], missing),
            3,
        )

        # Rejection rules and an interaction mask, which skip a feature
        # before any of its candidates is scored.
        _assert_same_record(
            _device_search_wide(
                plain, 2, 4, _params(min_data_in_leaf=25), [], [False, True]
            ),
            _device_search(
                plain, 2, 4, _params(min_data_in_leaf=25), [], [False, True]
            ),
            4,
        )

        # A monotone constraint, which rejects candidates inside the scan
        # rather than before it.
        var mono: List[Int] = [1]
        _assert_same_record(
            _device_search_wide(
                _histogram_words(
                    1, 4, [4, 2, -2, -4], [1, 1, 1, 1], [10, 10, 10, 10]
                ),
                1,
                4,
                _params(),
                [],
                [],
                [],
                mono,
            ),
            _device_search(
                _histogram_words(
                    1, 4, [4, 2, -2, -4], [1, 1, 1, 1], [10, 10, 10, 10]
                ),
                1,
                4,
                _params(),
                [],
                [],
                [],
                mono,
            ),
            4,
        )

        # More bins than the threadgroup is wide, so a thread's chunk is
        # several bins long and the exclusive prefix over the chunk sums is
        # what carries the running left totals across the boundaries. This
        # is the case a one-bin-per-thread scan would never reach.
        var wide_bins = 200
        var g = List[Int](capacity=wide_bins)
        var h = List[Int](capacity=wide_bins)
        var c = List[Int](capacity=wide_bins)
        for b in range(wide_bins):
            # A gradient that changes sign in the middle, so the best split
            # sits well away from either end, and a few exact ties.
            g.append(b - wide_bins // 2)
            h.append(1 + (b % 3))
            c.append(4 + (b % 5))
        var many = _histogram_words(1, wide_bins, g, h, c)
        var wide_rec = _device_search_wide(many, 1, wide_bins, _params())
        _assert_same_record(
            wide_rec, _device_search(many, 1, wide_bins, _params()), wide_bins
        )
        # The scan really did find a split in the interior, so the case
        # above is not passing on two empty records.
        assert_true(wide_rec.found)
        assert_true(wide_rec.bin > 0 and wide_rec.bin < wide_bins - 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
