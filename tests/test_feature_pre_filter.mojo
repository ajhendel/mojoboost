"""LightGBM's `feature_pre_filter`, end to end.

The filter drops a feature that has no usable split at all, and dropping it
means removing it from the pool `feature_fraction` samples, not merely
declining to split on it. So there are three things to establish and this
suite establishes all three separately, because any one of them can hold while
another is broken:

1. **`filter_count` and `need_filter` are LightGBM's, not something like
   them.** Both are asserted on hand-computed integers, including the
   truncation in the scaling and the two branches of `NeedFilter`. No fit is
   involved: a transcription is tested as a transcription.
2. **The filter actually fires.** The fixture is built so that it must: a
   constant column and a 995/5 column cannot survive `filter_cnt` of 20, and
   a balanced binary column and a continuous one must. The assertion is on the
   *usable count falling*, and on exactly which ids survive. A fixture of
   dense continuous columns would filter nothing at all and prove nothing --
   at 255 bins over a million rows `filter_cnt` is about 4 and every prefix
   past the fifth bin clears it -- which is why nothing here is continuous
   except the control column.
3. **Off is the fit that preceded the option.** Every edge is compared by
   `to_bits()` between a fit that says nothing, a fit that says
   `feature_pre_filter=False`, and a fit that says `True`. The filter must not
   move one bit of one edge; it may only shorten `usable`.

Determinism across `MOJOTREES_NUM_WORKERS` is asserted on the filter's own
output, because the bin counts are accumulated per feature inside the
feature-parallel fit and the surviving list is assembled from them.
"""

from std.os import setenv
from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from mojotrees.binning import (
    BinMapper,
    DEFAULT_MIN_DATA_IN_LEAF,
    all_features,
    filter_count,
    fit_bins,
    need_filter,
)
from mojotrees.sampling import select_tree_features, selection_count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _workers(n: Int):
    _ = setenv("MOJOTREES_NUM_WORKERS", String(n))


def _auto():
    _ = setenv("MOJOTREES_NUM_WORKERS", "0")


def _assert_same_edges(a: BinMapper, b: BinMapper, what: String) raises:
    """Edge for edge, bit for bit. Deliberately does *not* compare `usable`:
    this is the assertion that the filter changed nothing about the binning
    itself."""
    assert_equal(a.n_features, b.n_features, String(what, ": n_features"))
    assert_equal(len(a.edges), len(b.edges), String(what, ": edge count"))
    for i in range(len(a.edges)):
        assert_equal(
            a.edges[i].to_bits().cast[DType.uint64](),
            b.edges[i].to_bits().cast[DType.uint64](),
            String(what, ": edge ", i),
        )
    for i in range(len(a.edge_offsets)):
        assert_equal(
            a.edge_offsets[i], b.edge_offsets[i], String(what, ": offset ", i)
        )
    for i in range(len(a.missing_bin)):
        assert_equal(
            a.missing_bin[i], b.missing_bin[i], String(what, ": missing ", i)
        )


comptime FIXTURE_ROWS = 1000


def _fixture() -> List[Float64]:
    """Four columns, column-major, 1000 rows, chosen so the filter's answer is
    arithmetic rather than luck at `filter_cnt = 20`:

    - f0 constant 0.0. One level, so one bin, so trivial before `NeedFilter`
      is even consulted -- LightGBM's `num_bin_ <= 1` rule.
    - f1 near-constant: 995 zeros and 5 ones. Two bins holding 995 and 5. The
      only prefix is 995, which leaves 5 on the right, and 5 < 20. Filtered.
    - f2 balanced binary: 500 and 500. The prefix 500 leaves 500. Kept.
    - f3 continuous 0..999. More distinct values than bins, so ~255 bins of
      ~4 rows; the prefix clears 20 by the fifth bin with 980 left. Kept.
    """
    var out = List[Float64](capacity=4 * FIXTURE_ROWS)
    for _ in range(FIXTURE_ROWS):
        out.append(0.0)
    for r in range(FIXTURE_ROWS):
        out.append(1.0 if r >= 995 else 0.0)
    for r in range(FIXTURE_ROWS):
        out.append(1.0 if r >= 500 else 0.0)
    for r in range(FIXTURE_ROWS):
        out.append(Float64(r))
    return out^


# ---------------------------------------------------------------------------
# The transcribed rules
# ---------------------------------------------------------------------------


def test_filter_count_is_min_data_in_leaf_scaled_to_the_sample() raises:
    """`static_cast<data_size_t>(double(min_data_in_leaf * total_sample_size)
    / num_dist_data)`, from `src/io/dataset_loader.cpp`. Truncation toward
    zero, not rounding, and `min_data_in_leaf` itself only when the fit read
    every row."""
    # LightGBM's own stock numbers: 20 in a leaf, a 200,000-row bin sample of
    # a 1,000,000-row dataset.
    assert_equal(filter_count(20, 200_000, 1_000_000), 4, "stock scaling")
    # A fit that reads every row is unscaled.
    assert_equal(filter_count(20, 1_000, 1_000), 20, "unsampled")
    assert_equal(filter_count(20, 5_000, 1_000), 100, "sample above num_data")
    # Truncation, not rounding: 19.98 is 19 and 0.999 is 0.
    assert_equal(filter_count(20, 999, 1_000), 19, "truncates down")
    assert_equal(filter_count(1, 999, 1_000), 0, "truncates to zero")
    # A filter_cnt of 0 filters nothing, because every prefix clears it.
    assert_equal(filter_count(0, 1_000, 1_000), 0, "min_data_in_leaf 0")
    with assert_raises():
        _ = filter_count(20, 100, 0)


def test_need_filter_numerical_branch() raises:
    """LightGBM walks prefixes and stops one bin short of the end, so the last
    bin's population never forms a prefix."""
    var even: List[Int] = [5, 5, 5]
    var tail_heavy: List[Int] = [1, 100]
    var head_heavy: List[Int] = [30, 1]
    var interior: List[Int] = [1, 4, 26]
    var one_bin: List[Int] = [1000]
    var empty_head: List[Int] = [0, 1000]
    # A prefix that leaves enough on both sides: keep.
    assert_true(not need_filter(even, 15, 5, False), "5|10 clears 5 both ways")
    # Nothing on the left until the last bin, which is never a prefix.
    assert_true(
        need_filter(tail_heavy, 101, 5, False),
        "bin 1 holds 100 but is the last bin, so it is never a prefix",
    )
    # The right side runs out.
    assert_true(need_filter(head_heavy, 31, 5, False), "1 on the right")
    # An interior prefix succeeds where the first one failed.
    assert_true(not need_filter(interior, 31, 5, False), "prefix 5 leaves 26")
    assert_true(need_filter(interior, 31, 27, False), "no prefix reaches 27")
    # One bin is trivial before any prefix exists.
    assert_true(need_filter(one_bin, 1000, 1, False), "one bin")
    # filter_cnt 0 keeps everything with at least two bins.
    assert_true(not need_filter(empty_head, 1000, 0, False), "filter_cnt 0")


def test_need_filter_categorical_branch_differs() raises:
    """A categorical split is one category against the rest, not a prefix, so
    LightGBM returns False without looking once there are more than two bins.
    Asserted against the numerical answer on the *same* counts, so the branch
    is proved taken rather than assumed."""
    var three: List[Int] = [1, 1, 28]
    var two_short: List[Int] = [1, 29]
    var two_fine: List[Int] = [10, 20]
    assert_true(
        need_filter(three, 30, 5, False), "numerical: no prefix reaches 5"
    )
    assert_true(
        not need_filter(three, 30, 5, True),
        "categorical: more than two bins is never filtered",
    )
    # At two bins the categorical branch tests the bin, which is the prefix.
    assert_true(need_filter(two_short, 30, 5, True), "categorical two bins")
    assert_true(
        not need_filter(two_fine, 30, 5, True), "categorical two bins, kept"
    )


# ---------------------------------------------------------------------------
# The filter fires
# ---------------------------------------------------------------------------


def test_off_keeps_every_feature() raises:
    """The default, and the explicit `False`, both keep the whole pool. This
    is the assertion that makes the next test mean something: the count has to
    have somewhere to fall from."""
    var x = _fixture()
    var off = fit_bins(x, FIXTURE_ROWS, 4, 255, bin_construct_sample_cnt=0)
    assert_equal(len(off.usable), 4, "default keeps every feature")
    for f in range(4):
        assert_equal(off.usable[f], f, String("default usable ", f))
        assert_true(off.is_usable(f), String("default is_usable ", f))


def test_on_drops_the_features_that_cannot_split() raises:
    """The used-feature count falls, and falls to exactly the two columns that
    have a qualifying prefix."""
    var x = _fixture()
    var on = fit_bins(
        x,
        FIXTURE_ROWS,
        4,
        255,
        bin_construct_sample_cnt=0,
        feature_pre_filter=True,
        min_data_in_leaf=20,
    )
    assert_equal(len(on.usable), 2, "two of four features survive")
    assert_equal(on.usable[0], 2, "the balanced binary column survives")
    assert_equal(on.usable[1], 3, "the continuous column survives")
    assert_true(not on.is_usable(0), "the constant column is trivial")
    assert_true(not on.is_usable(1), "the 995/5 column has no usable prefix")


def test_min_data_in_leaf_moves_the_boundary() raises:
    """The same column is filtered or kept depending only on
    `min_data_in_leaf`, which is what proves the number being tested is the
    scaled count rather than a constant baked into the fit.

    Column 0 is continuous and survives either way, so the flip is column 1's
    alone and the fit never runs out of features.
    """
    var n = 1000
    var x = List[Float64](capacity=2 * n)
    for r in range(n):
        x.append(Float64(r))
    for r in range(n):
        x.append(1.0 if r >= 990 else 0.0)

    # 990 | 10. At min_data_in_leaf 20 the right side is short of 20; at 5 it
    # is not.
    var strict = fit_bins(
        x,
        n,
        2,
        255,
        bin_construct_sample_cnt=0,
        feature_pre_filter=True,
        min_data_in_leaf=20,
    )
    assert_equal(len(strict.usable), 1, "at 20 the 990|10 column is dropped")
    assert_equal(strict.usable[0], 0, "the continuous column survives")

    var lenient = fit_bins(
        x,
        n,
        2,
        255,
        bin_construct_sample_cnt=0,
        feature_pre_filter=True,
        min_data_in_leaf=5,
    )
    assert_equal(len(lenient.usable), 2, "at 5 both columns survive")
    assert_equal(lenient.usable[1], 1, "the 990|10 column survives at 5")


def test_a_categorical_column_takes_lightgbms_other_branch() raises:
    """The same 995/5 split that is filtered as a numerical column survives as
    a categorical one, because a categorical split is one category against the
    rest and LightGBM stops looking once there are more than two bins.

    This is the gate proved open rather than assumed: column 1 here holds
    exactly the values `_fixture`'s column 1 holds, and that one is dropped.
    """
    var n = FIXTURE_ROWS
    var x = List[Float64](capacity=2 * n)
    for r in range(n):
        x.append(Float64(r))
    for r in range(n):
        x.append(1.0 if r >= 995 else 0.0)
    var cats: List[Int] = [1]

    var as_numerical = fit_bins(
        x,
        n,
        2,
        255,
        bin_construct_sample_cnt=0,
        feature_pre_filter=True,
        min_data_in_leaf=20,
    )
    assert_equal(len(as_numerical.usable), 1, "995|5 is dropped as numerical")

    var as_categorical = fit_bins(
        x,
        n,
        2,
        255,
        categorical_features=cats,
        bin_construct_sample_cnt=0,
        feature_pre_filter=True,
        min_data_in_leaf=20,
    )
    assert_equal(len(as_categorical.usable), 2, "995|5 is kept as categorical")
    assert_equal(as_categorical.usable[1], 1, "the categorical column is 1")


def test_all_features_filtered_is_refused() raises:
    """LightGBM warns and continues. There is nowhere to warn to and no tree
    to grow, so this raises instead, and the divergence is on the record in
    `fit_bins`'s docstring."""
    var n = 500
    var x = List[Float64](capacity=2 * n)
    for _ in range(n):
        x.append(0.0)
    for _ in range(n):
        x.append(7.5)
    with assert_raises():
        _ = fit_bins(
            x,
            n,
            2,
            255,
            bin_construct_sample_cnt=0,
            feature_pre_filter=True,
            min_data_in_leaf=20,
        )


# ---------------------------------------------------------------------------
# Off is bit-identical
# ---------------------------------------------------------------------------


def test_the_filter_moves_no_edge() raises:
    """Three fits, one comparison each way. Saying nothing, saying `False`,
    and saying `True` must all produce the same edges bit for bit; only
    `usable` may differ."""
    var x = _fixture()
    var silent = fit_bins(x, FIXTURE_ROWS, 4, 255, bin_construct_sample_cnt=0)
    var explicit_off = fit_bins(
        x,
        FIXTURE_ROWS,
        4,
        255,
        bin_construct_sample_cnt=0,
        feature_pre_filter=False,
        min_data_in_leaf=20,
    )
    var on = fit_bins(
        x,
        FIXTURE_ROWS,
        4,
        255,
        bin_construct_sample_cnt=0,
        feature_pre_filter=True,
        min_data_in_leaf=20,
    )
    _assert_same_edges(silent, explicit_off, "default vs explicit false")
    _assert_same_edges(silent, on, "default vs true")
    # `matches` must still separate them, because a prefiltered mapper and an
    # unfiltered one do not offer a tree the same features.
    assert_true(silent.matches(explicit_off), "off matches off")
    assert_true(not silent.matches(on), "on does not match off")


def test_the_matrix_carries_the_pool() raises:
    """`transform` is how the list reaches a grower, which is handed a
    `BinnedMatrix` and nothing else."""
    var x = _fixture()
    var on = fit_bins(
        x,
        FIXTURE_ROWS,
        4,
        255,
        bin_construct_sample_cnt=0,
        feature_pre_filter=True,
        min_data_in_leaf=20,
    )
    var data = on.transform(x, FIXTURE_ROWS)
    var pool = data.usable_features()
    assert_equal(len(pool), 2, "the matrix carries the filtered pool")
    assert_equal(pool[0], 2, "pool 0")
    assert_equal(pool[1], 3, "pool 1")
    # Every column is still there and still binned: the filter is a pool, not
    # a renumbering, which is what keeps feature ids and importance slots
    # stable (LightGBM sizes FeatureImportance by num_total_features).
    assert_equal(data.n_features, 4, "no column was removed")


# ---------------------------------------------------------------------------
# The pool reaches feature_fraction
# ---------------------------------------------------------------------------


def test_passing_every_feature_is_the_same_draw_as_passing_nothing() raises:
    """The default is not a special case: an explicit full pool must give the
    identical sample, or the filtered path and the unfiltered path are two
    different samplers."""
    for tree_index in range(4):
        var implicit = select_tree_features(20, 0.35, 7, tree_index)
        var explicit = select_tree_features(
            20, 0.35, 7, tree_index, all_features(20)
        )
        assert_equal(
            len(implicit), len(explicit), String("count ", tree_index)
        )
        for i in range(len(implicit)):
            assert_equal(
                implicit[i], explicit[i], String("id ", i, " ", tree_index)
            )


def test_a_filtered_feature_is_never_drawn() raises:
    """The pool is the survivors, so nothing else can come out however many
    trees are drawn."""
    var pool: List[Int] = [1, 3, 5, 7, 9, 11]
    for tree_index in range(32):
        var chosen = select_tree_features(12, 0.5, 3, tree_index, pool)
        for i in range(len(chosen)):
            var found = False
            for j in range(len(pool)):
                if pool[j] == chosen[i]:
                    found = True
            assert_true(
                found, String("tree ", tree_index, " drew ", chosen[i])
            )
            if i > 0:
                assert_true(chosen[i] > chosen[i - 1], "ascending")


def test_the_fraction_is_of_the_survivors() raises:
    """LightGBM's `ColSampler` sizes the draw by
    `valid_feature_indices_.size()`, not by the total. Six survivors at 0.5
    give three, where twelve features at 0.5 give six."""
    var pool: List[Int] = [1, 3, 5, 7, 9, 11]
    assert_equal(selection_count(12, 0.5), 6, "unfiltered count")
    assert_equal(selection_count(len(pool), 0.5), 3, "filtered count")
    var chosen = select_tree_features(12, 0.5, 3, 0, pool)
    assert_equal(len(chosen), 3, "the draw is three of six, not six of twelve")


def test_a_bad_pool_is_refused() raises:
    var descending: List[Int] = [3, 1, 5]
    var repeated: List[Int] = [1, 1, 5]
    var past_the_end: List[Int] = [1, 5, 10]
    var negative: List[Int] = [-1, 5]
    var empty = List[Int]()
    with assert_raises():
        _ = select_tree_features(10, 0.5, 1, 0, descending)
    with assert_raises():
        _ = select_tree_features(10, 0.5, 1, 0, repeated)
    with assert_raises():
        _ = select_tree_features(10, 0.5, 1, 0, past_the_end)
    with assert_raises():
        _ = select_tree_features(10, 0.5, 1, 0, negative)
    # An empty pool is the default, meaning "nothing was filtered", so it is
    # the full draw rather than an error.
    assert_equal(
        len(select_tree_features(10, 1.0, 1, 0, empty)), 10, "empty is all"
    )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_the_filter_is_worker_independent() raises:
    """The bin counts are accumulated inside the feature-parallel fit and the
    surviving list is assembled from them, so the worker count is the thing
    most able to disturb it."""
    var x = _fixture()
    _workers(1)
    var serial = fit_bins(
        x,
        FIXTURE_ROWS,
        4,
        255,
        bin_construct_sample_cnt=0,
        feature_pre_filter=True,
        min_data_in_leaf=20,
    )
    var counts: List[Int] = [3, 8]
    for w in range(len(counts)):
        _workers(counts[w])
        var again = fit_bins(
            x,
            FIXTURE_ROWS,
            4,
            255,
            bin_construct_sample_cnt=0,
            feature_pre_filter=True,
            min_data_in_leaf=20,
        )
        _assert_same_edges(
            serial, again, String("edges at ", counts[w], " workers")
        )
        assert_equal(
            len(again.usable),
            len(serial.usable),
            String("usable count at ", counts[w], " workers"),
        )
        for i in range(len(serial.usable)):
            assert_equal(
                again.usable[i],
                serial.usable[i],
                String("usable ", i, " at ", counts[w], " workers"),
            )
        assert_true(serial.matches(again), String("matches at ", counts[w]))
    _auto()


def test_the_default_min_data_in_leaf_is_lightgbms() raises:
    assert_equal(DEFAULT_MIN_DATA_IN_LEAF, 20, "LightGBM's min_data_in_leaf")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
