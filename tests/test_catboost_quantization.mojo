"""CatBoost's border quantization and its one-hot threshold.

Two opt-in modes, and the first thing every case here has to establish is
that they are opt-in: `border_type` defaults to `BORDER_QUANTILE`, which is
the LightGBM `GreedyFindBin` port `binning.mojo` has always been, and
`CategoricalParams.one_hot_max_size` defaults to off, which is the LightGBM
`max_cat_to_onehot` test `categorical.mojo` has always run.

The border values asserted below are not eyeballed. They come from a port of
CatBoost's `TWeightedFeatureBin` + `GreedySplit`
(`library/cpp/grid_creator/binarization.cpp:1428-1520`) run on the same
columns, which is the same algorithm this file's subject implements; they are
a check that the Mojo and the reading agree, NOT a measurement against a
running CatBoost. See `docs/design/CATBOOST_CATALOG.md` A11 and A12 for what
was verified from source and what diverges on purpose.
"""

from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_raises,
    assert_true,
    TestSuite,
)

from mojotrees.binning import (
    BORDER_GREEDY_LOG_SUM,
    BORDER_GREEDY_MIN_ENTROPY,
    BORDER_MAX_LOG_SUM,
    BORDER_MEDIAN,
    BORDER_MIN_ENTROPY,
    BORDER_QUANTILE,
    BORDER_UNIFORM,
    BORDER_UNIFORM_AND_QUANTILES,
    BinMapper,
    border_type_name,
    check_border_type,
    fit_bins,
    parse_border_type,
)
from mojotrees.categorical import (
    CATBOOST_DEFAULT_ONE_HOT_MAX_SIZE,
    CategoricalParams,
    CategoricalSpec,
    ONE_HOT_MAX_SIZE_OFF,
    cat_contains,
)
from mojotrees.histogram import Histogram
from mojotrees.split import find_best_split
from support import _uniform


def _nan() -> Float64:
    var zero = _uniform(UInt64(0)) * 0.0
    return zero / zero


def _edges(mapper: BinMapper, feature: Int) -> List[Float64]:
    var out = List[Float64]()
    for i in range(
        mapper.edge_offsets[feature], mapper.edge_offsets[feature + 1]
    ):
        out.append(mapper.edges[i])
    return out^


def _ramp(n: Int) -> List[Float64]:
    """`0.0, 1.0, ... n-1.0`: every value distinct, evenly spaced."""
    var out = List[Float64]()
    for i in range(n):
        out.append(Float64(i))
    return out^


def _skewed() -> List[Float64]:
    """Six levels with counts 89, 5, 65, 10, 4, 5, which is the column the
    two greedy penalties disagree on. Total 178 rows."""
    var counts: List[Int] = [89, 5, 65, 10, 4, 5]
    var out = List[Float64]()
    for level in range(len(counts)):
        for _ in range(counts[level]):
            out.append(Float64(level))
    return out^


# --- The mode is opt-in ----------------------------------------------------


def test_default_border_type_is_the_quantile_fit_unchanged() raises:
    var col = _ramp(400)
    var stock = fit_bins(col, 400, 1, 32)
    var explicit = fit_bins(col, 400, 1, 32, border_type=BORDER_QUANTILE)
    assert_true(stock.matches(explicit))
    # And it is not vacuous: the CatBoost arm is a different fit. The column
    # has to be one that separates them, which a uniform ramp at a
    # power-of-two budget does not; see the next case.
    var skew = _skewed()
    var flat = fit_bins(skew, 178, 1, 4, min_data_in_bin=1)
    var greedy = fit_bins(
        skew, 178, 1, 4, min_data_in_bin=1, border_type=BORDER_GREEDY_LOG_SUM
    )
    assert_true(not flat.matches(greedy))


def test_the_two_rules_coincide_on_a_ramp_at_a_power_of_two_budget() raises:
    # Worth pinning because it is the case a careless comparison would be run
    # on. Equal-frequency boundaries sit at `b * n / bins` and the recursive
    # median lands on the dyadic grid, so at 400 evenly spaced values and 32
    # bins both are the 1/32 grid and the fits are edge for edge identical.
    # Nothing about the algorithms is the same; only this column is.
    var col = _ramp(400)
    var quantile = fit_bins(col, 400, 1, 32, min_data_in_bin=1)
    var greedy = fit_bins(
        col, 400, 1, 32, min_data_in_bin=1, border_type=BORDER_GREEDY_LOG_SUM
    )
    assert_true(quantile.matches(greedy))


def test_border_type_names_round_trip() raises:
    var names: List[String] = [
        "quantile",
        "GreedyLogSum",
        "GreedyMinEntropy",
        "Uniform",
        "Median",
        "UniformAndQuantiles",
        "MinEntropy",
        "MaxLogSum",
    ]
    for i in range(len(names)):
        assert_equal(border_type_name(parse_border_type(names[i])), names[i])
    with assert_raises():
        _ = parse_border_type("greedylogsum")
    with assert_raises():
        _ = border_type_name(99)


def test_exact_border_types_are_refused_by_name() raises:
    # CatBoost's TExactBinarizer dynamic program. Refused, not approximated,
    # because its approximation is already here as the Greedy* pair.
    with assert_raises():
        check_border_type(BORDER_MIN_ENTROPY)
    with assert_raises():
        check_border_type(BORDER_MAX_LOG_SUM)
    with assert_raises():
        check_border_type(-1)
    check_border_type(BORDER_QUANTILE)
    check_border_type(BORDER_GREEDY_LOG_SUM)
    var col = _ramp(50)
    with assert_raises():
        _ = fit_bins(
            col, 50, 1, 8, min_data_in_bin=1, border_type=BORDER_MAX_LOG_SUM
        )


def test_min_data_in_bin_is_refused_beside_a_catboost_border_type() raises:
    # LightGBM's parameter, no CatBoost analogue, so the combination is
    # refused rather than silently ignored. The default of 3 means every
    # CatBoost-arm caller has to say so.
    var col = _ramp(50)
    with assert_raises():
        _ = fit_bins(col, 50, 1, 8, border_type=BORDER_GREEDY_LOG_SUM)
    var ok = fit_bins(
        col, 50, 1, 8, min_data_in_bin=1, border_type=BORDER_GREEDY_LOG_SUM
    )
    assert_true(len(_edges(ok, 0)) > 0)


# --- GreedyLogSum ----------------------------------------------------------


def test_greedy_log_sum_borders_on_a_ramp() raises:
    # 100 distinct values, budget 5 bins, so 4 borders. Golden values from
    # the CatBoost port; the recursive median split cuts 49.5 first, then
    # 24.5 and 74.5, then breaks the four-way tie leftmost at 11.5.
    var col = _ramp(100)
    var mapper = fit_bins(
        col, 100, 1, 5, min_data_in_bin=1, border_type=BORDER_GREEDY_LOG_SUM
    )
    var got = _edges(mapper, 0)
    var want: List[Float64] = [11.5, 24.5, 49.5, 74.5]
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        assert_almost_equal(got[i], want[i])


def test_greedy_log_sum_border_count_is_min_budget_distinct_minus_one() raises:
    # From `GreedySplit`: the loop stops either at the budget or once every
    # bin holds a single level, and an unsplittable bin scores -inf so it
    # sinks. There is no third outcome, so the count is exactly this.
    var skew = _skewed()
    # Six levels, budget 255 bins: under budget, five borders is impossible
    # and four is forced.
    var wide = fit_bins(
        skew, 178, 1, 255, min_data_in_bin=1, border_type=BORDER_GREEDY_LOG_SUM
    )
    assert_equal(len(_edges(wide, 0)), 5)
    # Same column, budget 4 bins: the budget binds at three borders.
    var tight = fit_bins(
        skew, 178, 1, 4, min_data_in_bin=1, border_type=BORDER_GREEDY_LOG_SUM
    )
    assert_equal(len(_edges(tight, 0)), 3)
    # All distinct and far above the budget: fills it.
    var ramp = _ramp(400)
    var full = fit_bins(
        ramp, 400, 1, 32, min_data_in_bin=1, border_type=BORDER_GREEDY_LOG_SUM
    )
    assert_equal(len(_edges(full, 0)), 31)


def test_greedy_penalties_disagree_on_a_skewed_column() raises:
    # Counts 89, 5, 65, 10, 4, 5 at budget 4 bins. MaxSumLog's gain is
    # log(L*R/(L+R)) and MinEntropy's scales with the bin's size, so they
    # prioritize different bins and split differently. If these two ever
    # match, the penalty is not wired.
    var skew = _skewed()
    var log_sum = fit_bins(
        skew, 178, 1, 4, min_data_in_bin=1, border_type=BORDER_GREEDY_LOG_SUM
    )
    var entropy = fit_bins(
        skew,
        178,
        1,
        4,
        min_data_in_bin=1,
        border_type=BORDER_GREEDY_MIN_ENTROPY,
    )
    var want_log: List[Float64] = [0.5, 2.5, 3.5]
    var want_ent: List[Float64] = [0.5, 1.5, 2.5]
    var got_log = _edges(log_sum, 0)
    var got_ent = _edges(entropy, 0)
    assert_equal(len(got_log), 3)
    assert_equal(len(got_ent), 3)
    for i in range(3):
        assert_almost_equal(got_log[i], want_log[i])
        assert_almost_equal(got_ent[i], want_ent[i])


# --- The other implemented types -------------------------------------------


def test_uniform_borders_are_evenly_spaced_between_min_and_max() raises:
    # `minValue + (i + 1) * (maxValue - minValue) / (maxBordersCount + 1)`,
    # inserted raw. min 0, max 99, four borders.
    var col = _ramp(100)
    var mapper = fit_bins(
        col, 100, 1, 5, min_data_in_bin=1, border_type=BORDER_UNIFORM
    )
    var got = _edges(mapper, 0)
    assert_equal(len(got), 4)
    for i in range(4):
        assert_almost_equal(got[i], Float64(i + 1) * 99.0 / 5.0)


def test_median_borders_sit_below_their_rank() raises:
    # Rank `(i + 1) * total / (max_borders + 1)` is 20, 40, 60, 80 here, and
    # `RegularBorder` turns the value at that rank into the midpoint below
    # it, so the borders are the ranks minus a half.
    var col = _ramp(100)
    var mapper = fit_bins(
        col, 100, 1, 5, min_data_in_bin=1, border_type=BORDER_MEDIAN
    )
    var got = _edges(mapper, 0)
    var want: List[Float64] = [19.5, 39.5, 59.5, 79.5]
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        assert_almost_equal(got[i], want[i])


def test_uniform_and_quantiles_splits_the_budget_in_two() raises:
    # `halfBorders = maxBordersCount / 2` goes to the uniform half and the
    # MEDIAN half gets the rest, so a budget of 5 borders is 3 median plus 2
    # uniform. Both land in one set.
    var col = _ramp(100)
    var mapper = fit_bins(
        col,
        100,
        1,
        6,
        min_data_in_bin=1,
        border_type=BORDER_UNIFORM_AND_QUANTILES,
    )
    var got = _edges(mapper, 0)
    var want: List[Float64] = [24.5, 32.5, 49.5, 65.5, 74.5]
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        assert_almost_equal(got[i], want[i])


# --- Invariants the rest of the package depends on -------------------------


def test_catboost_edges_are_strictly_increasing_and_bin() raises:
    var skew = _skewed()
    var types: List[Int] = [
        BORDER_GREEDY_LOG_SUM,
        BORDER_GREEDY_MIN_ENTROPY,
        BORDER_UNIFORM,
        BORDER_MEDIAN,
        BORDER_UNIFORM_AND_QUANTILES,
    ]
    for t in range(len(types)):
        var mapper = fit_bins(
            skew, 178, 1, 8, min_data_in_bin=1, border_type=types[t]
        )
        var e = _edges(mapper, 0)
        for i in range(1, len(e)):
            assert_true(e[i] > e[i - 1])
        # The scratch stride is `max_bins - 1`, so an arm that overran it
        # would be writing into the next feature's slice.
        assert_true(len(e) <= 7)
        # And the edges actually bin: `bin_value` is monotone in the value
        # and never leaves the ordinary range.
        var data = mapper.transform(skew, 178)
        var prev = 0
        for level in range(6):
            var b = mapper.bin_value(0, Float64(level))
            assert_true(b >= prev)
            assert_true(b <= len(e))
            prev = b
        assert_equal(data.n_rows, 178)


def test_catboost_arm_reserves_a_missing_bin() raises:
    # NaN costs one bin here exactly as it does on the quantile arm, which is
    # also what CatBoost does (`--nonNanValuesBorderCount`) and what LightGBM
    # does (`FindBin(..., max_bin - 1)` then push the NaN bound).
    var col = _ramp(99)
    col.append(_nan())
    var mapper = fit_bins(
        col, 100, 1, 5, min_data_in_bin=1, border_type=BORDER_GREEDY_LOG_SUM
    )
    var e = _edges(mapper, 0)
    # Five bins minus the missing one leaves four ordinary bins, three
    # borders.
    assert_equal(len(e), 3)
    assert_true(mapper.has_missing())
    assert_equal(mapper.missing_bin[0], 4)
    assert_equal(mapper.bin_value(0, _nan()), 4)


def test_catboost_fit_does_not_depend_on_how_features_are_dispatched() raises:
    # Determinism across `MOJOTREES_NUM_WORKERS` is the contract; a
    # multi-feature fit splits across tasks and reuses per-task scratch, so
    # each column has to come out of it identical to a fit of that column
    # alone.
    var a = _ramp(100)
    var b = _skewed()
    var c = _ramp(60)
    var wide = List[Float64]()
    for i in range(100):
        wide.append(a[i])
    for i in range(100):
        wide.append(b[i])
    for i in range(100):
        wide.append(c[i] if i < 60 else Float64(59))
    var many = fit_bins(
        wide, 100, 3, 8, min_data_in_bin=1, border_type=BORDER_GREEDY_LOG_SUM
    )
    for f in range(3):
        var one_col = List[Float64]()
        for i in range(100):
            one_col.append(wide[f * 100 + i])
        var one = fit_bins(
            one_col,
            100,
            1,
            8,
            min_data_in_bin=1,
            border_type=BORDER_GREEDY_LOG_SUM,
        )
        var got = _edges(many, f)
        var want = _edges(one, 0)
        assert_equal(len(got), len(want))
        for i in range(len(want)):
            assert_equal(got[i], want[i])


def test_catboost_fit_repeats_exactly() raises:
    var skew = _skewed()
    var first = fit_bins(
        skew, 178, 1, 16, min_data_in_bin=1, border_type=BORDER_GREEDY_LOG_SUM
    )
    var second = fit_bins(
        skew, 178, 1, 16, min_data_in_bin=1, border_type=BORDER_GREEDY_LOG_SUM
    )
    assert_true(first.matches(second))


def test_constant_and_tiny_columns_produce_no_borders() raises:
    var flat = List[Float64]()
    for _ in range(50):
        flat.append(7.0)
    var types: List[Int] = [
        BORDER_GREEDY_LOG_SUM,
        BORDER_GREEDY_MIN_ENTROPY,
        BORDER_UNIFORM,
        BORDER_MEDIAN,
        BORDER_UNIFORM_AND_QUANTILES,
    ]
    for t in range(len(types)):
        var mapper = fit_bins(
            flat, 50, 1, 8, min_data_in_bin=1, border_type=types[t]
        )
        assert_equal(len(_edges(mapper, 0)), 0)
        # A single level is one bin, so every row lands in bin 0.
        assert_equal(mapper.bin_value(0, 7.0), 0)


# --- one_hot_max_size ------------------------------------------------------


def test_categorical_params_five_argument_construction_is_unchanged() raises:
    var p = CategoricalParams(4, 32, 10.0, 10.0, 100)
    assert_equal(p.max_cat_to_onehot, 4)
    assert_equal(p.min_data_per_group, 100)
    assert_equal(p.one_hot_max_size, ONE_HOT_MAX_SIZE_OFF)
    assert_true(not p.uses_catboost_one_hot())
    var d = CategoricalParams.default()
    assert_equal(d.one_hot_max_size, ONE_HOT_MAX_SIZE_OFF)
    assert_true(not d.uses_catboost_one_hot())
    # CatBoost's own default is recorded and is NOT ours.
    assert_equal(CATBOOST_DEFAULT_ONE_HOT_MAX_SIZE, 2)


def test_one_hot_max_size_is_validated() raises:
    CategoricalParams(4, 32, 10.0, 10.0, 100).check_one_hot()
    CategoricalParams(4, 32, 10.0, 10.0, 100, 0).check_one_hot()
    with assert_raises():
        CategoricalParams(4, 32, 10.0, 10.0, 100, -2).check_one_hot()
    with assert_raises():
        CategoricalParams(4, 32, 10.0, 10.0, 100, 257).check_one_hot()


def _alternating_hist() -> Histogram:
    """Six categories in bins 1..6 whose gradient signs alternate, a grouping
    no single category can express. One-vs-rest can only isolate one of them;
    the many-vs-many walk finds the three-and-three split."""
    var grad = List[Float64]()
    var hess = List[Float64]()
    var count = List[Int]()
    grad.append(0.0)
    hess.append(0.0)
    count.append(0)
    for b in range(1, 7):
        grad.append(-6.0 if b % 2 == 0 else 6.0)
        hess.append(6.0)
        count.append(6)
    return Histogram.from_planes(grad^, hess^, count^, 1, 7)


def _set_size(bitset: SIMD[DType.uint64, 4]) -> Int:
    var n = 0
    for b in range(256):
        if cat_contains(bitset, b):
            n += 1
    return n


def test_one_hot_max_size_off_leaves_the_lightgbm_threshold_in_charge() raises:
    # max_cat_to_onehot = 6 with six categories, so the one-vs-rest path, and
    # the winning set is a single category.
    var spec = CategoricalSpec([True], [0, 1, 2, 3, 4, 5], [0, 6])
    var split = find_best_split(
        _alternating_hist(),
        lambda_reg=0.0,
        min_data_in_leaf=0,
        cat_params=CategoricalParams(6, 32, 1.0, 0.0, 1),
        cats=spec,
    )
    assert_true(split.found and split.is_categorical)
    assert_equal(_set_size(split.cat_bitset), 1)


def test_one_hot_max_size_takes_over_the_one_hot_decision() raises:
    var spec = CategoricalSpec([True], [0, 1, 2, 3, 4, 5], [0, 6])
    # Same max_cat_to_onehot = 6, but CatBoost's threshold is 2 and six
    # categories is above it, so the feature falls to the many-vs-many
    # search and the alternating grouping wins. This is the whole point of
    # the parameter: it, and not max_cat_to_onehot, decides.
    var wide = find_best_split(
        _alternating_hist(),
        lambda_reg=0.0,
        min_data_in_leaf=0,
        cat_params=CategoricalParams(
            6, 32, 1.0, 0.0, 1, CATBOOST_DEFAULT_ONE_HOT_MAX_SIZE
        ),
        cats=spec,
    )
    assert_true(wide.found and wide.is_categorical)
    var evens_in = cat_contains(wide.cat_bitset, 2)
    for b in range(1, 7):
        assert_equal(
            cat_contains(wide.cat_bitset, b),
            evens_in if b % 2 == 0 else not evens_in,
        )
    # And it works the other way: raise CatBoost's threshold to six and the
    # one-vs-rest path comes back even though nothing else changed.
    var narrow = find_best_split(
        _alternating_hist(),
        lambda_reg=0.0,
        min_data_in_leaf=0,
        cat_params=CategoricalParams(1, 32, 1.0, 0.0, 1, 6),
        cats=spec,
    )
    assert_true(narrow.found and narrow.is_categorical)
    assert_equal(_set_size(narrow.cat_bitset), 1)


def test_one_hot_max_size_zero_sends_everything_to_the_sorted_search() raises:
    # CatBoost's guard is `count <= 1` on the low side and `> oneHotMaxSize`
    # on the high side, so a threshold of 0 leaves no feature one-hot.
    var spec = CategoricalSpec([True], [0, 1, 2, 3, 4, 5], [0, 6])
    var split = find_best_split(
        _alternating_hist(),
        lambda_reg=0.0,
        min_data_in_leaf=0,
        cat_params=CategoricalParams(6, 32, 1.0, 0.0, 1, 0),
        cats=spec,
    )
    assert_true(split.found and split.is_categorical)
    assert_true(_set_size(split.cat_bitset) > 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
