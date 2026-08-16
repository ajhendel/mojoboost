"""Focused tests for quantile bin fitting and bin assignment.

Two things are asserted throughout, and they are the whole contract the
optimized paths in `binning.mojo` have to keep:

- **The selected path equals the sorted path.** `fit_bins` resolves order
  statistics without sorting once a column passes `SELECT_MIN_ROWS`. Every
  test that reaches that path fits the same data twice, once above the
  threshold and once with the threshold pushed out of reach, and compares
  edge for edge. The sort path is the reference; nothing here trusts the
  selected one on its own.
- **The padded search equals the plain search.** `BinMapper.transform` walks
  a power-of-two padded table branchlessly; `BinMapper.bin_value` keeps the
  plain binary search. Every transform test compares the whole matrix against
  `bin_value` cell by cell.

Determinism across worker counts is asserted directly, because the fit is
feature-parallel and the selection is not: workers must not be able to change
an edge.
"""

from std.os import setenv
from std.testing import TestSuite, assert_equal, assert_raises, assert_true
from std.utils.numerics import inf, nan

from mojotrees.binning import (
    DEFAULT_BIN_CONSTRUCT_SAMPLE_CNT,
    DEFAULT_DATA_RANDOM_SEED,
    DEFAULT_MIN_DATA_IN_BIN,
    SELECT_MIN_ROWS,
    value_from_key,
    BinMapper,
    bin_construct_sample_rows,
    bin_equal_width,
    collect_distinct,
    count_levels,
    distinct_levels_sorted,
    emit_quantile_edges,
    fit_bins,
    order_key,
    quantile_boundary_indices,
    resolve_ranks,
)
from support import _splitmix64, _uniform


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


comptime NAN = nan[DType.float64]()
comptime INF = inf[DType.float64]()


def _serial():
    _ = setenv("MOJOTREES_NUM_WORKERS", "1")


def _workers(n: Int):
    _ = setenv("MOJOTREES_NUM_WORKERS", String(n))


def _auto():
    _ = setenv("MOJOTREES_NUM_WORKERS", "0")


def _force_sorted_path():
    """Push the selection threshold past any column these tests build, so
    `fit_bins` takes the plain sort. Restored by `_restore_paths`."""
    _ = setenv("MOJOTREES_BINNING_SELECT_MIN_ROWS", "1000000000")


def _restore_paths():
    _ = setenv("MOJOTREES_BINNING_SELECT_MIN_ROWS", "0")


def _assert_same_binning(a: BinMapper, b: BinMapper, what: String) raises:
    assert_equal(a.n_features, b.n_features, String(what, ": n_features"))
    assert_equal(len(a.edges), len(b.edges), String(what, ": edge count"))
    for i in range(len(a.edges)):
        assert_equal(
            a.edges[i].to_bits().cast[DType.uint64](),
            b.edges[i].to_bits().cast[DType.uint64](),
            String(what, ": edge ", i),
        )
    assert_equal(
        len(a.edge_offsets), len(b.edge_offsets), String(what, ": offsets")
    )
    for i in range(len(a.edge_offsets)):
        assert_equal(
            a.edge_offsets[i], b.edge_offsets[i], String(what, ": offset ", i)
        )
    for i in range(len(a.missing_bin)):
        assert_equal(
            a.missing_bin[i], b.missing_bin[i], String(what, ": missing ", i)
        )
    assert_true(a.matches(b), String(what, ": matches"))


def _assert_transform_matches_bin_value(
    mapper: BinMapper, features: List[Float64], n_rows: Int, what: String
) raises:
    """The padded branchless search must agree with the plain binary search
    on every cell."""
    var data = mapper.transform(features, n_rows)
    for f in range(mapper.n_features):
        for r in range(n_rows):
            # Raised rather than asserted: these loops run over whole
            # matrices, and building a message per cell would cost more than
            # the comparison it describes.
            if data.bin_at(r, f) != mapper.bin_value(
                f, features[f * n_rows + r]
            ):
                raise Error(
                    what, ": transform disagrees at cell (", r, ", ", f, ")"
                )


def _fit_both_ways(
    features: List[Float64],
    n_rows: Int,
    n_features: Int,
    max_bins: Int,
    what: String,
    min_data_in_bin: Int = 3,
) raises -> BinMapper:
    """Fit with the shipped policy and with the sort forced, assert the two
    agree, and return the shipped one.

    `min_data_in_bin` defaults to the package default so that a test which
    says nothing is testing what a user gets. A test about the one-bin-per-
    level rule itself passes 1, because that rule is stated at 1 and the
    stock 3 merges the rare levels the rule exists to find."""
    var fast = fit_bins(
        features, n_rows, n_features, max_bins, min_data_in_bin=min_data_in_bin
    )
    _force_sorted_path()
    var exact = fit_bins(
        features, n_rows, n_features, max_bins, min_data_in_bin=min_data_in_bin
    )
    _restore_paths()
    _assert_same_binning(fast, exact, what)
    return fast^


def _random_column(n_rows: Int, seed: UInt64) -> List[Float64]:
    var out = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        out.append(_uniform(seed + UInt64(r)))
    return out^


# ---------------------------------------------------------------------------
# order_key, boundaries, the edge rule
# ---------------------------------------------------------------------------


def test_order_key_is_monotone() raises:
    var values = [
        -INF,
        -1e300,
        -1.5,
        -1.0,
        -1e-300,
        -0.0,
        0.0,
        1e-300,
        1.0,
        1.5,
        1e300,
        INF,
    ]
    for i in range(1, len(values)):
        assert_true(
            order_key(values[i - 1]) < order_key(values[i]),
            String("order_key must increase at ", i),
        )
    # The one pair it separates that `<` does not.
    assert_true(-0.0 == 0.0, "the two zeros are equal as values")
    assert_true(
        order_key(-0.0) < order_key(0.0), "and ordered as keys"
    )

    # `value_from_key` inverts it bit for bit, which is what lets a bucket
    # holding one repeated key answer its ranks without keeping a value.
    for i in range(len(values)):
        assert_equal(
            value_from_key(order_key(values[i])).to_bits().cast[
                DType.uint64
            ](),
            values[i].to_bits().cast[DType.uint64](),
            String("round trip at ", i),
        )


def test_quantile_boundary_indices_drops_the_ends() raises:
    var idxs = List[Int]()
    quantile_boundary_indices(100, 4, idxs)
    assert_equal(len(idxs), 3, "three interior boundaries for four bins")
    assert_equal(idxs[0], 25, "first boundary")
    assert_equal(idxs[1], 50, "second boundary")
    assert_equal(idxs[2], 75, "third boundary")

    # Fewer values than bins: boundaries repeat and rank 0 is dropped.
    idxs.clear()
    quantile_boundary_indices(3, 8, idxs)
    for j in range(len(idxs)):
        assert_true(idxs[j] > 0, "no boundary at rank 0")
        assert_true(idxs[j] < 3, "no boundary at rank n_valid")

    # Nothing at all when there is nothing to cut.
    quantile_boundary_indices(0, 8, idxs)
    assert_equal(len(idxs), 0, "no valid values, no boundaries")
    quantile_boundary_indices(1, 8, idxs)
    assert_equal(len(idxs), 0, "one valid value, no boundaries")


def test_emit_quantile_edges_rules() raises:
    var out = List[Float64]()

    # Midpoints, in order.
    emit_quantile_edges([0.0, 2.0], [2.0, 4.0], out)
    assert_equal(len(out), 2, "two gaps, two edges")
    assert_equal(out[0], 1.0, "first midpoint")
    assert_equal(out[1], 3.0, "second midpoint")

    # A boundary whose value is the column maximum has nothing above it, so
    # the caller passes `below` back and it produces nothing.
    emit_quantile_edges([1.0, 1.0], [1.0, 3.0], out)
    assert_equal(len(out), 1, "maximum-valued boundary dropped")
    assert_equal(out[0], 2.0, "only the real gap")

    # A repeated boundary cannot repeat an edge.
    emit_quantile_edges([0.0, 0.0], [2.0, 2.0], out)
    assert_equal(len(out), 1, "duplicate boundary dropped")

    # Infinite midpoints clamp, and the clamp cannot break monotonicity.
    emit_quantile_edges([1e308, 1e308], [INF, INF], out)
    assert_equal(len(out), 1, "clamped once")
    assert_equal(out[0], 1e300, "clamped to MAX_EDGE")


def test_resolve_ranks_matches_a_sort() raises:
    # Big enough to take every branch of the selection: the column is far
    # above SELECT_SMALL_SEGMENT and its values repeat.
    var n = 1 << 14
    var col = List[Float64](capacity=n)
    for i in range(n):
        col.append(Float64(Int(_splitmix64(UInt64(i)) % 500)) * 0.25)

    var reference = col.copy()
    sort(reference)

    var ranks = List[Int]()
    var r = 0
    while r < n:
        ranks.append(r)
        r += 37
    var vals = List[Float64]()
    vals.resize(len(ranks), 0.0)
    var counts = List[Int]()
    var keys = List[UInt64]()
    var seg = col.copy()
    resolve_ranks(seg, 0, ranks, 0, len(ranks), counts, keys, vals, 0)
    for i in range(len(ranks)):
        assert_equal(
            vals[i].to_bits().cast[DType.uint64](),
            reference[ranks[i]].to_bits().cast[DType.uint64](),
            String("rank ", ranks[i]),
        )


# ---------------------------------------------------------------------------
# Column shapes
# ---------------------------------------------------------------------------


def test_constant_column() raises:
    var n_rows = SELECT_MIN_ROWS + 3
    var features = List[Float64](capacity=n_rows)
    features.resize(n_rows, 7.5)
    var mapper = _fit_both_ways(features, n_rows, 1, 255, "constant")
    assert_equal(mapper.edge_offsets[1], 0, "a constant column has no edges")
    assert_equal(mapper.missing_bin[0], -1, "and reserves nothing")
    var data = mapper.transform(features, n_rows)
    for r in range(n_rows):
        assert_equal(data.bin_at(r, 0), 0, "every row in bin 0")


def test_binary_and_low_cardinality_columns() raises:
    var n_rows = SELECT_MIN_ROWS + 11
    # Feature 0: two values. Feature 1: five values. Feature 2: repeated
    # values with a heavy mode.
    var features = List[Float64](capacity=n_rows * 3)
    for r in range(n_rows):
        features.append(1.0 if (r % 2) == 0 else -4.0)
    for r in range(n_rows):
        features.append(Float64(r % 5))
    for r in range(n_rows):
        features.append(0.0 if (r % 10) != 0 else 100.0)

    var mapper = _fit_both_ways(features, n_rows, 3, 255, "low cardinality")
    # A column with k distinct values and k <= max_bins gets its k - 1 gaps
    # cut, every one of them. Every boundary here lands inside a run, so
    # before ties were cut at the end of their run these came out with no
    # edges at all and one unsplittable bin.
    assert_equal(
        mapper.edge_offsets[1] - mapper.edge_offsets[0], 1, "binary"
    )
    assert_equal(
        mapper.edge_offsets[2] - mapper.edge_offsets[1], 4, "five levels"
    )
    assert_equal(
        mapper.edge_offsets[3] - mapper.edge_offsets[2], 1, "two levels"
    )
    _assert_transform_matches_bin_value(
        mapper, features, n_rows, "low cardinality"
    )


def test_all_missing_column() raises:
    var n_rows = SELECT_MIN_ROWS + 5
    var features = List[Float64](capacity=n_rows)
    features.resize(n_rows, NAN)
    var mapper = _fit_both_ways(features, n_rows, 1, 255, "all missing")
    assert_equal(mapper.edge_offsets[1], 0, "no edges without a value")
    assert_equal(mapper.missing_bin[0], 1, "missing bin still reserved")
    var data = mapper.transform(features, n_rows)
    for r in range(n_rows):
        assert_equal(data.bin_at(r, 0), 1, "every row missing")
        assert_true(data.is_missing(r, 0), "and reported as missing")


def test_mixed_missing_and_finite() raises:
    # Sized so the non-missing values alone still clear SELECT_MIN_ROWS: one
    # row in seven is missing, so the column has to be a seventh larger than
    # the threshold for the selected path to run at all.
    var n_rows = SELECT_MIN_ROWS + SELECT_MIN_ROWS // 4
    var features = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        features.append(NAN if (r % 7) == 0 else _uniform(UInt64(r)))
    var mapper = _fit_both_ways(features, n_rows, 1, 32, "mixed missing")
    assert_true(mapper.missing_bin[0] > 0, "a missing bin is reserved")
    assert_true(
        mapper.edge_offsets[1] <= 30,
        "a reserving feature fits inside max_bins - 1 ordinary bins",
    )
    _assert_transform_matches_bin_value(
        mapper, features, n_rows, "mixed missing"
    )
    var data = mapper.transform(features, n_rows)
    for r in range(n_rows):
        assert_equal(
            data.is_missing(r, 0),
            (r % 7) == 0,
            String("missingness of row ", r),
        )


def test_infinities_are_finite_side_extremes() raises:
    var n_rows = SELECT_MIN_ROWS + 8
    var features = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        features.append(_uniform(UInt64(r)))
    features[0] = -INF
    features[1] = INF
    features[2] = -INF
    features[3] = INF

    var mapper = _fit_both_ways(features, n_rows, 1, 64, "infinities")
    assert_equal(mapper.missing_bin[0], -1, "infinity is not missing")
    var n_edges = mapper.edge_offsets[1]
    assert_true(n_edges > 0, "the column still has edges")
    for i in range(n_edges):
        assert_true(
            mapper.edges[i] >= -1e300 and mapper.edges[i] <= 1e300,
            String("edge ", i, " is clamped"),
        )
    var data = mapper.transform(features, n_rows)
    assert_equal(data.bin_at(0, 0), 0, "-inf in the lowest bin")
    assert_equal(data.bin_at(1, 0), n_edges, "+inf in the highest bin")
    _assert_transform_matches_bin_value(
        mapper, features, n_rows, "infinities"
    )


def test_values_exactly_on_boundaries() raises:
    var n_rows = SELECT_MIN_ROWS + 100
    var features = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        features.append(Float64(r % 251))
    var mapper = _fit_both_ways(features, n_rows, 1, 255, "on boundaries")

    # Feed the edges themselves back through, plus the values either side.
    var n_edges = mapper.edge_offsets[1]
    for i in range(n_edges):
        var e = mapper.edges[i]
        assert_equal(
            mapper.bin_value(0, e), i, String("edge ", i, " lands in bin ", i)
        )
        assert_true(
            mapper.bin_value(0, e + 1e-9) >= i,
            String("just above edge ", i),
        )

    # And through the vectorized path, on a matrix built from the edges.
    var probe = List[Float64](capacity=n_edges)
    for i in range(n_edges):
        probe.append(mapper.edges[i])
    if n_edges > 0:
        _assert_transform_matches_bin_value(
            mapper, probe, n_edges, "edges as data"
        )


def test_max_bin_values() raises:
    var n_rows = SELECT_MIN_ROWS + 37
    var n_features = 2
    var features = List[Float64](capacity=n_rows * n_features)
    for f in range(n_features):
        for r in range(n_rows):
            features.append(_uniform(UInt64(f) * 7919 + UInt64(r)))
    var bins = [2, 15, 31, 63, 127, 255]
    for i in range(len(bins)):
        var mb = bins[i]
        var mapper = _fit_both_ways(
            features, n_rows, n_features, mb, String("max_bin ", mb)
        )
        for f in range(n_features):
            var k = mapper.edge_offsets[f + 1] - mapper.edge_offsets[f]
            assert_true(
                k <= mb - 1,
                String("max_bin ", mb, ": feature ", f, " has ", k, " edges"),
            )
        # The cell-by-cell comparison is quadratic in what this loop already
        # covers, so it runs at the two ends of the range: the shortest
        # padded search table and the longest one. The middle bin counts are
        # still fitted both ways and bounds-checked above.
        if mb == 2 or mb == 255:
            _assert_transform_matches_bin_value(
                mapper, features, n_rows, String("max_bin ", mb)
            )


def test_ties_are_cut_where_the_run_ends() raises:
    """The regression this rule exists for: a column whose every quantile
    boundary lands inside a run of equal values still gets its gaps cut."""
    var n_rows = 100_000

    # Balanced binary. 254 boundaries, both runs 50k long, so not one
    # boundary straddles the change.
    var flag = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        flag.append(1.0 if (r % 2) == 0 else 0.0)
    var m = _fit_both_ways(flag, n_rows, 1, 255, "balanced binary")
    assert_equal(m.edge_offsets[1], 1, "a binary column has one gap to cut")
    assert_true(
        m.edges[0] > 0.0 and m.edges[0] < 1.0, "and cuts between the two"
    )
    var d = m.transform(flag, n_rows)
    assert_equal(d.bin_at(0, 0), 1, "a one lands in the upper bin")
    assert_equal(d.bin_at(1, 0), 0, "a zero in the lower bin")

    # Skewed binary: the rare level is still a bin of its own.
    var rare = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        rare.append(1.0 if (r % 10) == 0 else 0.0)
    var m2 = _fit_both_ways(rare, n_rows, 1, 255, "skewed binary")
    assert_equal(m2.edge_offsets[1], 1, "10/90 binary still has its gap")

    # More levels than bins: the budget binds, not the ties.
    var many = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        many.append(Float64(r % 300))
    var m3 = _fit_both_ways(many, n_rows, 1, 255, "300 levels")
    assert_true(
        m3.edge_offsets[1] > 200 and m3.edge_offsets[1] <= 254,
        String("300 levels into 255 bins gave ", m3.edge_offsets[1], " edges"),
    )


def test_distinct_columns_are_unchanged_by_the_tie_rule() raises:
    """With no ties the next distinct value above rank idx - 1 is the value at
    rank idx, so the rule is a no-op: this is why continuous data bins exactly
    as it always did."""
    var n_rows = SELECT_MIN_ROWS + 91
    var col = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        col.append(Float64(r) * 1.5)
    var m = _fit_both_ways(col, n_rows, 1, 255, "strictly increasing")
    assert_equal(m.edge_offsets[1], 254, "every boundary cuts")
    for i in range(1, m.edge_offsets[1]):
        assert_true(
            m.edges[i] > m.edges[i - 1], String("edge ", i, " increases")
        )
    _assert_transform_matches_bin_value(m, col, n_rows, "strictly increasing")


def test_every_rank_a_boundary_is_unchanged_by_the_tie_rule() raises:
    """The other no-op case: with n_valid <= n_ordinary every rank is a
    boundary, so a run's own end is one too and the old rule already found
    it."""
    var features: List[Float64] = [
        0.0, 0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 5.0, 9.0, 9.0
    ]
    # At `min_data_in_bin=1`, which is where "every rank is a boundary" is
    # the whole rule. The stock 3 is a second rule on top of it and is
    # asserted separately below.
    var m = _fit_both_ways(features, 10, 1, 255, "every rank a boundary", 1)
    assert_equal(m.edge_offsets[1], 3, "four levels, three gaps")
    assert_equal(m.edges[0], 0.5, "0 | 1")
    assert_equal(m.edges[1], 3.0, "1 | 5")
    assert_equal(m.edges[2], 7.0, "5 | 9")

    # Counts are 3, 2, 3, 2, so at the stock minimum the accumulator clears
    # on level 0, cannot reach three on level 1, and reaches five on level 2:
    # the 1 | 5 cut is the one that goes.
    var stock = _fit_both_ways(features, 10, 1, 255, "stock minimum")
    assert_equal(stock.edge_offsets[1], 2, "one gap merged away")
    assert_equal(stock.edges[0], 0.5, "0 | 1 survives")
    assert_equal(stock.edges[1], 7.0, "5 | 9 survives")


# ---------------------------------------------------------------------------
# Determinism and reuse
# ---------------------------------------------------------------------------


def test_edges_are_independent_of_worker_count() raises:
    var n_rows = SELECT_MIN_ROWS + 13
    var n_features = 5
    var features = List[Float64](capacity=n_rows * n_features)
    for f in range(n_features):
        for r in range(n_rows):
            features.append(_uniform(UInt64(f) * 104729 + UInt64(r)))

    _serial()
    var serial = fit_bins(features, n_rows, n_features, 255)
    var serial_data = serial.transform(features, n_rows)
    var counts = [2, 3, 4, 8]
    for i in range(len(counts)):
        _workers(counts[i])
        var m = fit_bins(features, n_rows, n_features, 255)
        _assert_same_binning(
            serial, m, String("workers ", counts[i])
        )
        var d = m.transform(features, n_rows)
        for c in range(len(serial_data.bins)):
            if d.bins[c] != serial_data.bins[c]:
                raise Error(
                    "workers ", counts[i], ": bin ", c, " differs from serial"
                )
    _auto()


def test_reference_mapper_reuse_bins_new_data() raises:
    """A mapper fitted once bins later data through the same edges: no refit,
    and the bins mean what the model was trained against."""
    var n_rows = SELECT_MIN_ROWS + 21
    var train = _random_column(n_rows, 11)
    var mapper = _fit_both_ways(train, n_rows, 1, 64, "reference fit")

    var valid = List[Float64](capacity=64)
    for r in range(64):
        valid.append(_uniform(UInt64(900_000) + UInt64(r)))
    var binned = mapper.transform(valid, 64)
    assert_equal(binned.n_bins, 64, "the reference's bin count carries over")
    for r in range(64):
        assert_equal(
            binned.bin_at(r, 0),
            mapper.bin_value(0, valid[r]),
            String("reused mapper, row ", r),
        )
    # Refitting on the validation rows would have produced different edges,
    # which is exactly the leak reuse exists to prevent.
    var refit = fit_bins(valid, 64, 1, 64)
    assert_true(
        len(refit.edges) != len(mapper.edges)
        or refit.edges[0] != mapper.edges[0],
        "the validation refit is a different binning",
    )


def test_small_column_still_takes_the_sort_path() raises:
    """Below the threshold the selection is a no-op, at either minimum.

    Eight rows against four bins is the quantile branch, and it is small
    enough that `min_data_in_bin` reaches the budget itself: the stock 3 caps
    it at `max(1, 8 // 3) = 2` bins, so a column this short gets one cut where
    at 1 it got two. Both are asserted, because the point of the test is that
    forcing the sort changes nothing and that has to hold under either.
    """
    var features: List[Float64] = [3.0, 1.0, 2.0, 5.0, 4.0, 5.0, 5.0, 0.0]
    # Sorted: 0 1 2 3 4 5 5 5. At min_data_in_bin=1 the boundaries are at
    # ranks 2, 4 and 6; the third falls inside the run of fives and cuts
    # nothing.
    var loose = _fit_both_ways(features, 8, 1, 4, "small column at 1", 1)
    assert_equal(loose.edge_offsets[1], 2, "two edges, the third had no gap")
    _assert_transform_matches_bin_value(loose, features, 8, "small at 1")

    var mapper = _fit_both_ways(features, 8, 1, 4, "small column")
    assert_equal(mapper.edge_offsets[1], 1, "the budget capped at two bins")
    _assert_transform_matches_bin_value(mapper, features, 8, "small column")


def test_equal_width_path_is_untouched() raises:
    var features: List[Float64] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    var data = bin_equal_width(features, 8, 1, 4)
    assert_equal(data.bin_at(0, 0), 0, "first row")
    assert_equal(data.bin_at(7, 0), 3, "last row")


def test_few_features_split_by_rows_and_still_match_serial() raises:
    """`transform` on a matrix with fewer features than workers.

    That shape used to cap itself at `n_features` workers; it now cuts each
    feature's rows into blocks. Binning a cell depends on that cell alone, so
    the bins must come out byte for byte the same as the serial run at every
    worker count. Row counts are deliberately not multiples of the block
    sizes, so the last block of each feature is short and a mistake in the
    tail clamp would leave trailing rows unwritten (bin 0) rather than binned.
    """
    var widths = [1, 2, 3]
    var row_counts = [SELECT_MIN_ROWS + 7, SELECT_MIN_ROWS + 1]
    var counts = [2, 3, 4, 8, 16]
    for wi in range(len(widths)):
        var n_features = widths[wi]
        for ri in range(len(row_counts)):
            var n_rows = row_counts[ri]
            var features = List[Float64](capacity=n_rows * n_features)
            for f in range(n_features):
                for r in range(n_rows):
                    features.append(
                        _uniform(UInt64(f) * 7919 + UInt64(r) * 31 + 1)
                    )

            _serial()
            var mapper = fit_bins(features, n_rows, n_features, 255)
            var serial_data = mapper.transform(features, n_rows)
            var what = String(n_features, " features x ", n_rows, " rows")

            # A tail left unwritten reads as bin 0, and so does a genuinely
            # smallest value, so pin the last row against the reference search
            # rather than against a constant.
            for f in range(n_features):
                assert_equal(
                    serial_data.bin_at(n_rows - 1, f),
                    mapper.bin_value(f, features[f * n_rows + n_rows - 1]),
                    String(what, ": serial last row of feature ", f),
                )

            for ci in range(len(counts)):
                _workers(counts[ci])
                var d = mapper.transform(features, n_rows)
                for c in range(len(serial_data.bins)):
                    if d.bins[c] != serial_data.bins[c]:
                        _auto()
                        raise Error(
                            what,
                            ", workers ",
                            counts[ci],
                            ": cell ",
                            c,
                            " differs from serial",
                        )
    _auto()


def test_row_split_bins_a_categorical_feature_the_same_way() raises:
    """The categorical branch of `transform` is tiled by rows too, and a
    category's bin is a table lookup on its own cell, so the tiles cannot
    disagree with the serial pass or with `bin_value`."""
    var n_rows = SELECT_MIN_ROWS + 5
    var n_features = 2
    var features = List[Float64](capacity=n_rows * n_features)
    # Feature 0 is declared categorical and takes seven levels; feature 1 is
    # ordinary, so one call covers both branches of the tiled kernel.
    for r in range(n_rows):
        features.append(Float64(r % 7))
    for r in range(n_rows):
        features.append(_uniform(UInt64(r) * 6151 + 17))

    _serial()
    var mapper = fit_bins(features, n_rows, n_features, 255, [0])
    var serial_data = mapper.transform(features, n_rows)

    var counts = [2, 4, 8]
    for ci in range(len(counts)):
        _workers(counts[ci])
        var d = mapper.transform(features, n_rows)
        for c in range(len(serial_data.bins)):
            if d.bins[c] != serial_data.bins[c]:
                _auto()
                raise Error(
                    "workers ", counts[ci], ": cell ", c, " differs from serial"
                )
    _auto()
    _assert_transform_matches_bin_value(
        mapper, features, n_rows, "categorical plus numeric, row split"
    )


def test_collect_distinct_answers_and_leaves_its_table_clean() raises:
    """The helper on its own: it reports the levels when there are few
    enough, refuses as soon as there is one more, and one table serves a
    whole run of columns. A stale key left behind by an earlier column would
    make a later one lose a level, which is why the refusals below are
    interleaved with the answers rather than run at the end.

    The counts are checked through the same reuse, because `hits` is cleared
    the same way `table` is -- by the slots that were recorded -- and a
    counter left behind by an earlier column would inflate a later one's
    level and change where `min_data_in_bin` cuts."""
    var table = List[UInt64]()
    var slots = List[Int]()
    var hits = List[Int]()
    var out = List[Float64]()
    var cnts = List[Int]()

    var three: List[Float64] = [2.0, 5.0, 2.0, -1.0, 5.0, -1.0, 2.0]
    assert_true(
        collect_distinct(three, 7, 8, table, slots, hits, out, cnts),
        "three levels fit",
    )
    assert_equal(len(out), 3, "three levels")
    assert_equal(out[0], -1.0, "ascending, first")
    assert_equal(out[1], 2.0, "ascending, second")
    assert_equal(out[2], 5.0, "ascending, third")
    assert_equal(len(cnts), 3, "one count per level")
    assert_equal(cnts[0], 2, "-1.0 twice")
    assert_equal(cnts[1], 3, "2.0 three times")
    assert_equal(cnts[2], 2, "5.0 twice")

    # Exactly at the limit is an answer; one past it is not.
    var eight = List[Float64](capacity=8)
    for i in range(8):
        eight.append(Float64(i) * 3.0)
    assert_true(
        collect_distinct(eight, 8, 8, table, slots, hits, out, cnts),
        "eight levels fit",
    )
    assert_equal(len(out), 8, "eight levels")
    for i in range(8):
        assert_equal(cnts[i], 1, "each of the eight is one row")
    assert_true(
        not collect_distinct(eight, 8, 7, table, slots, hits, out, cnts),
        "eight levels do not fit in seven",
    )
    assert_equal(len(out), 0, "a refusal reports nothing")
    assert_equal(len(cnts), 0, "and counts nothing")

    # The same three-level column again, through the table that just took a
    # refusal and an eight-level answer. The counts are the point: they came
    # off slots an earlier column had already used.
    assert_true(
        collect_distinct(three, 7, 8, table, slots, hits, out, cnts),
        "three levels again",
    )
    assert_equal(len(out), 3, "three levels after reuse")
    assert_equal(out[0], -1.0, "ascending after reuse")
    assert_equal(cnts[0], 2, "counts after reuse, first")
    assert_equal(cnts[1], 3, "counts after reuse, second")
    assert_equal(cnts[2], 2, "counts after reuse, third")

    # Negative and positive zero are one level, not two: every comparison an
    # edge is built from treats them as equal.
    var zeros: List[Float64] = [0.0, -0.0, 0.0, -0.0]
    assert_true(
        collect_distinct(zeros, 4, 8, table, slots, hits, out, cnts),
        "zeros fit",
    )
    assert_equal(len(out), 1, "-0.0 and 0.0 are one level")
    assert_equal(cnts[0], 4, "and one counter")

    # An empty column has no levels to report and is not an answer.
    assert_true(
        not collect_distinct(three, 0, 8, table, slots, hits, out, cnts),
        "no rows",
    )


def test_a_rare_level_gets_its_own_bin() raises:
    """A level too rare for any quantile boundary to land in.

    Boundaries are ranks. Three rows in sixty thousand hold no rank that any
    of 254 boundaries asks about, so under the boundary rule alone those
    three rows were swallowed by the zeros however far away they sat: one
    edge, two bins, and no split that could separate a 3 from a 0. Counting
    the levels finds them.
    """
    var n_rows = SELECT_MIN_ROWS + 1000
    var features = List[Float64](capacity=n_rows)
    for _ in range(n_rows):
        features.append(0.0)
    features[5] = 1.0
    features[900] = 2.0
    features[40000] = 3.0

    var mapper = _fit_both_ways(
        features, n_rows, 1, 255, "one rare level", 1
    )
    assert_equal(mapper.edge_offsets[1], 3, "one edge per adjacent pair")
    for level in range(4):
        assert_equal(
            mapper.bin_value(0, Float64(level)),
            level,
            String("level ", level, " has its own bin"),
        )
    _assert_transform_matches_bin_value(
        mapper, features, n_rows, "one rare level"
    )

    # And what the stock default does to the same column, asserted rather
    # than left implied, because it is a real narrowing of the guarantee
    # above. Counts are (n-3, 1, 1, 1): the accumulator clears on level 0
    # and then never reaches 3 again, so 1, 2 and 3 end up sharing one bin.
    # That is LightGBM's GreedyFindBin at min_data_in_bin=3, and the rare
    # levels are separated from the zeros but no longer from each other.
    var stock = _fit_both_ways(features, n_rows, 1, 255, "rare level, stock")
    assert_equal(stock.edge_offsets[1], 1, "one edge at the stock minimum")
    assert_equal(stock.bin_value(0, 0.0), 0, "the zeros keep their own bin")
    for level in range(1, 4):
        assert_equal(
            stock.bin_value(0, Float64(level)),
            1,
            String("stock: level ", level, " shares the second bin"),
        )


def test_levels_stop_getting_their_own_bin_past_the_budget() raises:
    """The handover. At the budget every level is a bin; one level past it
    the column goes back to quantile boundaries, which cannot give each level
    a bin because there are more levels than bins to give."""
    var n_rows = SELECT_MIN_ROWS + 64

    # Eight levels into eight ordinary bins: one edge per adjacent pair.
    var eight = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        # Level 7 is rare on purpose; it still earns a bin.
        eight.append(7.0 if r == 3 else Float64(r % 7))
    var m8 = _fit_both_ways(eight, n_rows, 1, 8, "eight levels")
    assert_equal(m8.edge_offsets[1], 7, "seven edges for eight levels")
    for level in range(8):
        assert_equal(
            m8.bin_value(0, Float64(level)),
            level,
            String("eight levels, level ", level),
        )

    # Nine levels into eight ordinary bins: the budget binds. Which pair of
    # levels ends up sharing is the quantile rule's business and is not
    # asserted here; that some pair must is the point.
    var nine = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        nine.append(8.0 if r == 3 else Float64(r % 8))
    var m9 = _fit_both_ways(nine, n_rows, 1, 8, "nine levels")
    assert_true(
        m9.edge_offsets[1] <= 7, "nine levels cannot exceed the bin budget"
    )
    var seen = List[Int]()
    for level in range(9):
        var b = m9.bin_value(0, Float64(level))
        var known = False
        for i in range(len(seen)):
            if seen[i] == b:
                known = True
        if not known:
            seen.append(b)
    assert_true(
        len(seen) < 9, "nine levels cannot each have a bin in eight of them"
    )


def test_a_continuous_column_never_takes_the_level_path() raises:
    """More distinct values than bins is the ordinary case, and it must reach
    the quantile boundaries unchanged: a full budget of edges, and a helper
    that refuses before it has read any meaningful part of the column."""
    var n_rows = SELECT_MIN_ROWS + 33
    var col = _random_column(n_rows, 4242)
    var table = List[UInt64]()
    var slots = List[Int]()
    var out = List[Float64]()
    var hits = List[Int]()
    var cnts = List[Int]()
    assert_true(
        not collect_distinct(col, n_rows, 255, table, slots, hits, out, cnts),
        "a continuous column has far more than 255 levels",
    )
    var mapper = _fit_both_ways(col, n_rows, 1, 255, "continuous")
    assert_equal(mapper.edge_offsets[1], 254, "a full budget of edges")


def test_a_reserved_missing_bin_costs_the_levels_one() raises:
    """The level count is compared against the ordinary bins, not against
    `max_bins`, so a column that reserves a missing bin has one fewer level
    it can spend. Eight levels plus a NaN at `max_bins = 8` therefore falls
    back to boundaries, while seven levels plus a NaN still fits."""
    var n_rows = SELECT_MIN_ROWS + 64

    var seven = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        seven.append(NAN if r == 0 else Float64(r % 7))
    var m7 = _fit_both_ways(seven, n_rows, 1, 8, "seven levels and a NaN")
    assert_equal(m7.edge_offsets[1], 6, "six edges for seven levels")
    assert_equal(m7.missing_bin[0], 7, "the missing bin sits above them")

    var eight = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        eight.append(NAN if r == 0 else (7.0 if r == 3 else Float64(r % 7)))
    var m8 = _fit_both_ways(eight, n_rows, 1, 8, "eight levels and a NaN")
    assert_true(
        m8.edge_offsets[1] <= 6,
        "eight levels do not fit in seven ordinary bins",
    )


def test_the_two_level_counters_agree() raises:
    """`collect_distinct` (dense, unsorted) and `distinct_levels_sorted`
    (sparse, sorted) have to give the same answer on every column, because a
    sparse matrix and its dense form are required to bin identically. Checked
    over columns that stress the ways they could differ: repeats, both signs,
    both zeros, extremes, and both sides of the limit.
    """
    var table = List[UInt64]()
    var slots = List[Int]()
    var a = List[Float64]()
    var b = List[Float64]()
    var hits = List[Int]()
    var cnts = List[Int]()

    var limits = [1, 2, 3, 7, 8, 64, 255, 256]
    for li in range(len(limits)):
        var limit = limits[li]
        for shape in range(6):
            var col = List[Float64]()
            if shape == 0:
                for i in range(500):
                    col.append(Float64(i % 9) - 4.0)
            elif shape == 1:
                # Both zeros, repeatedly, with values either side.
                for i in range(200):
                    col.append(0.0 if i % 4 == 0 else -0.0)
                    if i % 7 == 0:
                        col.append(-2.5)
                    if i % 11 == 0:
                        col.append(2.5)
            elif shape == 2:
                col.append(-INF)
                col.append(INF)
                for i in range(100):
                    col.append(Float64(i % 3))
            elif shape == 3:
                # Exactly `limit` levels, then exactly one more.
                for i in range(limit * 3):
                    col.append(Float64(i % limit))
            elif shape == 4:
                for i in range((limit + 1) * 3):
                    col.append(Float64(i % (limit + 1)))
            else:
                for i in range(300):
                    col.append(_uniform(UInt64(i) * 977 + UInt64(limit)))

            var n = len(col)
            var fit_a = collect_distinct(col, n, limit, table, slots, hits, a, cnts)
            sort(col)
            var fit_b = distinct_levels_sorted(col, n, limit, b)
            var what = String("limit ", limit, ", shape ", shape)
            assert_equal(fit_a, fit_b, String(what, ": same verdict"))
            assert_equal(len(a), len(b), String(what, ": same level count"))
            for i in range(len(a)):
                assert_equal(a[i], b[i], String(what, ": level ", i))


# ---------------------------------------------------------------------------
# min_data_in_bin and bin_construct_sample_cnt
#
# Both options change which bins exist, so every test here is written to fail
# if the option were silently ignored: each one asserts an edge *count* (and
# where it is knowable, each edge's bits) against a fixture built so that the
# option must move it. A fixture on which the option happens to change nothing
# would pass whatever the binner did, which is the one thing these must not be.
# ---------------------------------------------------------------------------


def _edge_count(mapper: BinMapper, f: Int) -> Int:
    return mapper.edge_offsets[f + 1] - mapper.edge_offsets[f]


def _assert_edges_are(
    mapper: BinMapper, expected: List[Float64], what: String
) raises:
    assert_equal(
        _edge_count(mapper, 0), len(expected), String(what, ": edge count")
    )
    for i in range(len(expected)):
        assert_equal(
            mapper.edges[i].to_bits().cast[DType.uint64](),
            expected[i].to_bits().cast[DType.uint64](),
            String(what, ": edge ", i),
        )


def _eight_levels_column() -> List[Float64]:
    """Levels 0..7 with populations 1,1,1,1,4,4,4,4 (twenty rows).

    Chosen so that `min_data_in_bin = 3` demonstrably merges: the four rare
    levels cannot each hold three rows, so three of the seven cuts between
    adjacent levels have to disappear. On a column where every level already
    held three rows the option would change nothing and the test would prove
    nothing.
    """
    var col = List[Float64]()
    for v in range(4):
        col.append(Float64(v))
    for v in range(4, 8):
        for _ in range(4):
            col.append(Float64(v))
    return col^


def test_min_data_in_bin_merges_levels() raises:
    _serial()
    var col = _eight_levels_column()
    assert_equal(len(col), 20, "twenty rows")

    # Eight levels inside a 255-bin budget, so this is the levels path.
    var loose = fit_bins(col, len(col), 1, 255, min_data_in_bin=1)
    _assert_edges_are(
        loose,
        [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
        "min_data_in_bin=1 cuts between every adjacent level",
    )

    # Spelling the stock default out has to be the default.
    var three_default = fit_bins(col, len(col), 1, 255)
    var three_named = fit_bins(col, len(col), 1, 255, min_data_in_bin=3)
    _assert_same_binning(
        three_default, three_named, "min_data_in_bin=3 is the default"
    )

    # LightGBM's GreedyFindBin walks the levels accumulating counts and cuts
    # only when the accumulator reaches the minimum, resetting it there:
    # counts 1,1,1 reach 3 at level 2, then 4 at each of levels 4, 5 and 6.
    var three = three_named.copy()
    _assert_edges_are(
        three,
        [2.5, 4.5, 5.5, 6.5],
        "min_data_in_bin=3 merges the four rare levels into one bin",
    )
    # The gate, stated as a count as well as as values: seven cuts became
    # four, so an ignored option cannot pass this.
    assert_equal(_edge_count(loose, 0), 7, "seven edges at one")
    assert_equal(_edge_count(three, 0), 4, "four edges at the stock three")

    # Levels 0, 1 and 2 are now one bin, and level 3 -- which is still too
    # rare to earn a cut of its own -- shares the next one with level 4. Both
    # of those pairings are impossible at one, where every level has its own
    # bin, so this is the merge asserted through the bins rather than through
    # the edges.
    var data = three.transform(col, len(col))
    for r in range(3):
        assert_equal(data.bin_at(r, 0), 0, String("rare level ", r, " merged"))
    assert_equal(data.bin_at(3, 0), 1, "level 3 leaves the first bin")
    assert_equal(data.bin_at(4, 0), 1, "and shares the second with level 4")
    assert_equal(data.bin_at(8, 0), 2, "level 5 keeps a bin of its own")
    _assert_transform_matches_bin_value(three, col, len(col), "merged levels")

    var loose_data = loose.transform(col, len(col))
    assert_equal(
        loose_data.bin_at(3, 0), 3, "at the default level 3 is its own bin"
    )
    assert_true(
        loose_data.bin_at(3, 0) != loose_data.bin_at(4, 0),
        "and does not share one with level 4",
    )

    # A minimum no level can reach leaves one bin and no edge at all.
    var huge = fit_bins(col, len(col), 1, 255, min_data_in_bin=20)
    assert_equal(_edge_count(huge, 0), 0, "no cut survives a minimum of 20")
    _auto()


def test_min_data_in_bin_caps_the_quantile_budget() raises:
    """The other branch of `GreedyFindBin`: more distinct values than bins, so
    LightGBM shrinks the budget to `total_cnt / min_data_in_bin` before it
    cuts. A hundred distinct values in an eight-bin budget takes that branch.
    """
    _serial()
    var col = List[Float64]()
    for r in range(100):
        col.append(Float64(r))

    var loose = fit_bins(col, len(col), 1, 8)
    _assert_edges_are(
        loose,
        [11.5, 24.5, 36.5, 49.5, 61.5, 74.5, 86.5],
        "seven quantile boundaries in an eight-bin budget",
    )

    # 100 // 30 = 3 ordinary bins, so two boundaries at ranks 33 and 66.
    var capped = fit_bins(col, len(col), 1, 8, min_data_in_bin=30)
    _assert_edges_are(
        capped, [32.5, 65.5], "the budget shrinks to three bins"
    )
    assert_equal(_edge_count(loose, 0), 7, "seven edges at the default")
    assert_equal(_edge_count(capped, 0), 2, "two edges at thirty")

    # The same, through the rank-selection path rather than the sort, because
    # the two are required to resolve the same order statistics.
    _ = setenv("MOJOTREES_BINNING_SELECT_MIN_ROWS", "1")
    var selected = fit_bins(col, len(col), 1, 8, min_data_in_bin=30)
    _restore_paths()
    _assert_same_binning(capped, selected, "selection agrees with the sort")

    # A minimum larger than the column leaves one bin: `max(1, 100 // 200)`.
    var one_bin = fit_bins(col, len(col), 1, 8, min_data_in_bin=200)
    assert_equal(_edge_count(one_bin, 0), 0, "one bin, no edges")
    _auto()


def test_count_levels_counts_what_the_levels_hold() raises:
    """`min_data_in_bin`'s merge rule is only as good as its counts, and the
    two zeros are the case that could silently split one level in two.

    Both counters, checked against each other. `collect_distinct` counts by
    hash slot inside the scan it was already doing and `count_levels` counts
    by binary search in a pass of its own; the first is what `fit_bins` uses
    and the second is the only thing that keeps it from being checked against
    itself.
    """
    var col = List[Float64]()
    for i in range(30):
        col.append(-0.0 if i % 3 == 0 else (0.0 if i % 3 == 1 else 5.0))
    var table = List[UInt64]()
    var slots = List[Int]()
    var hits = List[Int]()
    var levels = List[Float64]()
    var fused = List[Int]()
    assert_true(
        collect_distinct(col, len(col), 8, table, slots, hits, levels, fused),
        "two levels",
    )
    assert_equal(len(levels), 2, "the two zeros are one level")
    var counts = List[Int]()
    count_levels(col, len(col), levels, counts)
    assert_equal(len(counts), 2, "one count per level")
    assert_equal(counts[0], 20, "both signs of zero counted together")
    assert_equal(counts[1], 10, "and the rest")
    assert_equal(counts[0] + counts[1], len(col), "the counts are the column")

    # The fused counts are the ones `fit_bins` cuts on, so they are the ones
    # that have to be right.
    assert_equal(len(fused), 2, "one fused count per level")
    for j in range(len(counts)):
        assert_equal(fused[j], counts[j], String("fused count ", j))

    # And over a set of shapes, not just this one: singletons, a heavy level,
    # every level distinct, and one level per row.
    var shapes = [0, 1, 2, 3]
    for si in range(len(shapes)):
        var shape = shapes[si]
        var c = List[Float64]()
        for i in range(97):
            if shape == 0:
                c.append(Float64(i))
            elif shape == 1:
                c.append(0.0 if i < 90 else Float64(i))
            elif shape == 2:
                c.append(Float64(i % 5))
            else:
                c.append(Float64(i % 2) - 0.5)
        var lv = List[Float64]()
        var fz = List[Int]()
        var what = String("shape ", shape)
        assert_true(
            collect_distinct(c, len(c), 128, table, slots, hits, lv, fz),
            String(what, ": fits"),
        )
        var ref_counts = List[Int]()
        count_levels(c, len(c), lv, ref_counts)
        assert_equal(len(fz), len(lv), String(what, ": one count per level"))
        var total = 0
        for j in range(len(lv)):
            assert_equal(fz[j], ref_counts[j], String(what, ": count ", j))
            total += fz[j]
        assert_equal(total, len(c), String(what, ": counts are the column"))


def test_bin_construct_sample_rows_is_exact_and_deterministic() raises:
    """Exactly k rows, ascending, in range, and the same k rows at every
    worker count. The sample is drawn once per fit and every feature is fit
    from it, so a sample that moved with the schedule would move the model."""
    var ns = [1000, 1000, 1000, 64, 5]
    var ks = [20, 999, 1, 32, 5]
    for i in range(len(ns)):
        var n = ns[i]
        var k = ks[i]
        var rows = bin_construct_sample_rows(n, k, 1)
        var what = String("n ", n, ", k ", k)
        if k >= n:
            assert_equal(len(rows), 0, String(what, ": empty means all rows"))
            continue
        assert_equal(len(rows), k, String(what, ": exactly k rows"))
        for j in range(len(rows)):
            assert_true(
                rows[j] >= 0 and rows[j] < n, String(what, ": in range")
            )
            if j > 0:
                assert_true(
                    rows[j - 1] < rows[j], String(what, ": ascending")
                )

    # 0 and a count covering the matrix both mean every row.
    assert_equal(len(bin_construct_sample_rows(1000, 0, 1)), 0, "0 is all")
    assert_equal(
        len(bin_construct_sample_rows(1000, 5000, 1)), 0, "over n is all"
    )

    # Different seeds are different samples, or the seed is not being read.
    var a = bin_construct_sample_rows(1000, 20, 1)
    var b = bin_construct_sample_rows(1000, 20, 7)
    var same = len(a) == len(b)
    if same:
        for j in range(len(a)):
            if a[j] != b[j]:
                same = False
                break
    assert_true(not same, "a different seed draws a different sample")

    # The draw depends on the row index alone, so the schedule cannot reach
    # it; asserted rather than assumed.
    var workers = [1, 3, 8]
    for w in range(len(workers)):
        _workers(workers[w])
        var again = bin_construct_sample_rows(1000, 20, 1)
        assert_equal(len(again), len(a), "same length at every worker count")
        for j in range(len(a)):
            assert_equal(again[j], a[j], "same row at every worker count")
    _auto()


def test_bin_construct_sample_cnt_fits_from_the_sample() raises:
    """A thousand distinct values fit from twenty of them, with the edges
    recomputed independently from the sample the module reports.

    The full fit takes the quantile branch (a thousand values against 255
    bins) and the sampled one takes the levels branch (twenty against 255), so
    an ignored `bin_construct_sample_cnt` cannot produce these nineteen edges
    by any other route.
    """
    _serial()
    var n_rows = 1000
    var col = List[Float64]()
    for r in range(n_rows):
        col.append(Float64(r))

    # `min_data_in_bin=1` throughout: this test is about which rows the edges
    # come off, and at the stock 3 the twenty sampled singletons would merge
    # in threes and the arithmetic below would be testing two rules at once.
    var full = fit_bins(col, n_rows, 1, 255, min_data_in_bin=1)
    var sampled = fit_bins(
        col,
        n_rows,
        1,
        255,
        min_data_in_bin=1,
        bin_construct_sample_cnt=20,
        data_random_seed=1,
    )

    var rows = bin_construct_sample_rows(n_rows, 20, 1)
    assert_equal(len(rows), 20, "twenty sampled rows")
    var expected = List[Float64]()
    for j in range(len(rows) - 1):
        # col[r] == r, and the sampled rows are already ascending and
        # distinct, so the levels path cuts at each adjacent midpoint. Both
        # operands are integers below 2048, so the midpoint is exact.
        expected.append((Float64(rows[j]) + Float64(rows[j + 1])) / 2.0)
    _assert_edges_are(sampled, expected, "edges come off the sample")

    assert_true(
        _edge_count(full, 0) > 19,
        "the full fit spends far more of the budget",
    )
    # Every row is still binned, sample or no sample.
    _assert_transform_matches_bin_value(sampled, col, n_rows, "sampled fit")
    var data = sampled.transform(col, n_rows)
    assert_equal(data.n_rows, n_rows, "every row is binned")

    # 0 and a count at or above the row count are the full fit, and so is the
    # stock 200,000 on a matrix of a thousand rows: nothing under the stock
    # count is sampled at all.
    var zero = fit_bins(
        col, n_rows, 1, 255, min_data_in_bin=1, bin_construct_sample_cnt=0
    )
    _assert_same_binning(full, zero, "0 means every row")
    var over = fit_bins(
        col, n_rows, 1, 255, min_data_in_bin=1, bin_construct_sample_cnt=n_rows
    )
    _assert_same_binning(full, over, "a sample of every row is every row")
    var stock_cnt = fit_bins(col, n_rows, 1, 255, min_data_in_bin=1)
    _assert_same_binning(
        full, stock_cnt, "the stock 200,000 does not sample a thousand rows"
    )
    _auto()


def test_the_new_options_are_worker_independent() raises:
    """Both new paths, fit at one, three, and eight workers, compared exactly.

    Four features rather than one, because the fit is feature-parallel: a
    single feature is one task at any worker count and would prove nothing.
    """
    var n_rows = 800
    var n_features = 4
    var feats = List[Float64]()
    for f in range(n_features):
        for r in range(n_rows):
            if f == 0:
                feats.append(Float64(r % 11))
            elif f == 1:
                feats.append(Float64(r))
            elif f == 2:
                feats.append(NAN if r % 97 == 0 else _uniform(UInt64(r) * 31))
            else:
                feats.append(Float64((r * 7) % 200))

    _serial()
    var base = fit_bins(
        feats,
        n_rows,
        n_features,
        64,
        min_data_in_bin=20,
        bin_construct_sample_cnt=300,
        data_random_seed=1,
    )
    var workers = [3, 8]
    for w in range(len(workers)):
        _workers(workers[w])
        var other = fit_bins(
            feats,
            n_rows,
            n_features,
            64,
            min_data_in_bin=20,
            bin_construct_sample_cnt=300,
            data_random_seed=1,
        )
        _assert_same_binning(
            base, other, String("workers ", workers[w])
        )
    _auto()

    # And the two options together are not the same fit as either alone,
    # which is what says both are still being read when both are set.
    _serial()
    var neither = fit_bins(feats, n_rows, n_features, 64)
    var only_min = fit_bins(feats, n_rows, n_features, 64, min_data_in_bin=20)
    var only_sample = fit_bins(
        feats,
        n_rows,
        n_features,
        64,
        bin_construct_sample_cnt=300,
        data_random_seed=1,
    )
    assert_true(
        not neither.matches(only_min), "min_data_in_bin moved the binning"
    )
    assert_true(
        not neither.matches(only_sample),
        "bin_construct_sample_cnt moved the binning",
    )
    assert_true(
        not only_min.matches(base) and not only_sample.matches(base),
        "both options are read when both are set",
    )
    _auto()


def test_the_new_options_are_range_checked() raises:
    var col = List[Float64]()
    for r in range(50):
        col.append(Float64(r))
    with assert_raises():
        _ = fit_bins(col, len(col), 1, 255, min_data_in_bin=0)
    with assert_raises():
        _ = fit_bins(col, len(col), 1, 255, min_data_in_bin=-3)
    with assert_raises():
        _ = fit_bins(col, len(col), 1, 255, bin_construct_sample_cnt=-1)


# ---------------------------------------------------------------------------
# The stock defaults
#
# mojotrees's binning defaults are LightGBM's, not the ones at which this
# module is the module it was. Three things have to be true and each is
# asserted here rather than read off the source: the constants are the stock
# numbers, an unqualified fit is a fit at those numbers, and the old fit is
# still reachable and still exactly the old fit.
# ---------------------------------------------------------------------------


def test_the_defaults_are_lightgbms_stock_values() raises:
    """The numbers, and that `fit_bins` actually uses them.

    Two assertions rather than one. The constants can be read; whether the
    signature still defaults to them cannot, and a signature that had drifted
    back to 1 and 0 would leave the constants perfectly correct and the
    package binning differently from what it documents. The fixture is chosen
    so that all three of (1, 0), (3, 0) and (3, 200000) would give different
    edge counts if the row count reached the sample, and so that (1, 0) and
    (3, 0) differ here.
    """
    assert_equal(DEFAULT_MIN_DATA_IN_BIN, 3, "LightGBM's min_data_in_bin")
    assert_equal(
        DEFAULT_BIN_CONSTRUCT_SAMPLE_CNT,
        200_000,
        "LightGBM's bin_construct_sample_cnt",
    )
    assert_equal(DEFAULT_DATA_RANDOM_SEED, 1, "LightGBM's data_random_seed")

    _serial()
    var col = _eight_levels_column()
    var silent = fit_bins(col, len(col), 1, 255)
    var named = fit_bins(
        col,
        len(col),
        1,
        255,
        min_data_in_bin=DEFAULT_MIN_DATA_IN_BIN,
        bin_construct_sample_cnt=DEFAULT_BIN_CONSTRUCT_SAMPLE_CNT,
        data_random_seed=DEFAULT_DATA_RANDOM_SEED,
    )
    _assert_same_binning(silent, named, "saying nothing is saying stock")

    # And the fixture does distinguish the two settings, so the comparison
    # above is not vacuous.
    var old = fit_bins(
        col, len(col), 1, 255, min_data_in_bin=1, bin_construct_sample_cnt=0
    )
    assert_true(
        not silent.matches(old),
        "the fixture separates the stock defaults from the old ones",
    )
    _auto()


def test_the_pre_stock_binning_is_still_reachable_and_identical() raises:
    """`min_data_in_bin=1, bin_construct_sample_cnt=0` is the old binner.

    Not "close to" and not "the same shape": the same edges, bit for bit,
    computed here from the pre-option rules rather than from a second call
    into the same code. Those rules are, for a column with no more distinct
    values than ordinary bins, one edge at the midpoint of every adjacent
    pair of levels; and for a column with more, the full budget of quantile
    boundaries with none of them capped.

    This exists because the stock defaults are a re-baseline, and a
    re-baseline is only defensible if the thing it replaced can still be
    asked for and still is what it was.
    """
    _serial()

    # A levels column: eight levels, four of them too rare for the stock
    # minimum, so this is exactly where the two settings part company.
    var col = _eight_levels_column()
    var old = fit_bins(
        col, len(col), 1, 255, min_data_in_bin=1, bin_construct_sample_cnt=0
    )
    _assert_edges_are(
        old,
        [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
        "one edge per adjacent pair, as before the options existed",
    )
    # Every level lands in its own bin, which is the whole of the old rule.
    var data = old.transform(col, len(col))
    for r in range(4):
        assert_equal(data.bin_at(r, 0), r, String("old rule, level ", r))

    # A quantile column: a thousand distinct values against eight bins. The
    # old rule spends the whole budget; the stock minimum cannot cap it here
    # (1000 // 3 = 333 > 8), so this half is a check that reachability did
    # not change the branch that was never affected.
    var wide = List[Float64]()
    for r in range(1000):
        wide.append(Float64(r))
    var old_wide = fit_bins(
        wide, 1000, 1, 8, min_data_in_bin=1, bin_construct_sample_cnt=0
    )
    var stock_wide = fit_bins(wide, 1000, 1, 8)
    assert_equal(_edge_count(old_wide, 0), 7, "the full budget of edges")
    _assert_same_binning(
        old_wide, stock_wide, "the quantile branch is untouched here"
    )

    # Reachable at every worker count, not only serially.
    var workers = [3, 8]
    for w in range(len(workers)):
        _workers(workers[w])
        var again = fit_bins(
            col,
            len(col),
            1,
            255,
            min_data_in_bin=1,
            bin_construct_sample_cnt=0,
        )
        _assert_same_binning(
            old, again, String("old rule at ", workers[w], " workers")
        )
    _auto()


def _lgbm_num_bins_levels(
    counts: List[Int], min_data_in_bin: Int
) raises -> Int:
    """`GreedyFindBin`'s `num_distinct_values <= max_bin` branch, transcribed
    from LightGBM's `src/io/bin.cpp`, returning `bin_upper_bound.size()`.

    The transcription, so a reader can check it against the C++ rather than
    against a description of it:

        int cur_cnt_inbin = 0;
        for (int i = 0; i < num_distinct_values - 1; ++i) {
          cur_cnt_inbin += counts[i];
          if (cur_cnt_inbin >= min_data_in_bin) {
            auto val = GetDoubleUpperBound((v[i] + v[i+1]) / 2.0);
            if (empty || !CheckDoubleEqualOrdered(back, val)) {
              bin_upper_bound.push_back(val);
              cur_cnt_inbin = 0;
            }
          }
        }
        bin_upper_bound.push_back(infinity);

    The `CheckDoubleEqualOrdered` guard drops a midpoint that collides with
    the previous one. The fixtures below use small integer levels, whose
    midpoints are exactly representable and pairwise distinct, so it never
    fires and the count is the count of cuts plus one. A fixture where it
    could fire would be comparing two dedupe rules rather than two binning
    rules, and mojotrees's dedupe lives in `emit_quantile_edges`.

    This is a second implementation, not a call into the first. That is the
    only thing that makes it a parity check.
    """
    var bounds = 0
    var cur = 0
    for i in range(len(counts) - 1):
        cur += counts[i]
        if cur >= min_data_in_bin:
            bounds += 1
            cur = 0
    return bounds + 1


def test_bin_counts_match_lightgbms_greedy_find_bin() raises:
    """Per-column bin counts against LightGBM's rule on a fixed matrix.

    Six columns, each a different shape of level population, and each fit
    through `fit_bins` at both minima. The comparison is the *number of
    bins*, which is what LightGBM's `num_bin_` reports and what decides how
    much histogram work a column costs.

    Scope, stated so the pass is not read as more than it is: the levels
    branch only. Past the bin budget mojotrees uses quantile boundaries
    inside LightGBM's shrunken budget and does not implement the per-level
    greedy or `is_big_count_value`, which `docs/LIGHTGBM_PARITY.md` records
    as the reason that row says `partial`. That gap is asserted below as a
    bound, not as equality, because equality is not claimed.
    """
    _serial()
    var n_rows = 240
    var n_features = 6
    var feats = List[Float64]()
    # Level populations per column, by construction, and each column's levels
    # are the small integers 0, 1, 2, ... so the midpoints are exact.
    var expected_counts = List[List[Int]]()
    for f in range(n_features):
        var counts = List[Int]()
        if f == 0:
            # Twelve levels of twenty: nothing merges at either minimum.
            for r in range(n_rows):
                feats.append(Float64(r % 12))
            for _ in range(12):
                counts.append(20)
        elif f == 1:
            # 236 singletons then one level of four: every merge is a merge
            # of singletons, which is the case the minimum is for.
            for r in range(n_rows):
                feats.append(Float64(r) if r < 236 else 236.0)
            for _ in range(236):
                counts.append(1)
            counts.append(4)
        elif f == 2:
            # One heavy level and three rare ones: the accumulator clears on
            # the heavy level and then cannot reach three again.
            for r in range(n_rows):
                feats.append(0.0 if r < 237 else Float64(r - 236))
            counts.append(237)
            counts.append(1)
            counts.append(1)
            counts.append(1)
        elif f == 3:
            # Alternating populations of two, so every bin takes two levels.
            for r in range(n_rows):
                feats.append(Float64(r // 2))
            for _ in range(120):
                counts.append(2)
        elif f == 4:
            # A binary column: two levels, neither rare.
            for r in range(n_rows):
                feats.append(Float64(r % 2))
            counts.append(120)
            counts.append(120)
        else:
            # A constant column: one level, no cut possible at any minimum.
            for _ in range(n_rows):
                feats.append(7.0)
            counts.append(240)
        expected_counts.append(counts^)

    var minima = [1, 2, 3, 5]
    for mi in range(len(minima)):
        var m = minima[mi]
        var mapper = fit_bins(
            feats,
            n_rows,
            n_features,
            255,
            min_data_in_bin=m,
            bin_construct_sample_cnt=0,
        )
        for f in range(n_features):
            var want = _lgbm_num_bins_levels(expected_counts[f], m)
            var got = _edge_count(mapper, f) + 1
            assert_equal(
                got,
                want,
                String("min_data_in_bin ", m, ", column ", f, ": bin count"),
            )

    # The gate: at least one column must actually be moved by the minimum, or
    # the loop above would pass on a binner that ignored the option entirely.
    var at_one = fit_bins(
        feats, n_rows, n_features, 255, min_data_in_bin=1,
        bin_construct_sample_cnt=0,
    )
    var at_three = fit_bins(
        feats, n_rows, n_features, 255, min_data_in_bin=3,
        bin_construct_sample_cnt=0,
    )
    var moved = 0
    for f in range(n_features):
        if _edge_count(at_one, f) != _edge_count(at_three, f):
            moved += 1
    assert_true(moved >= 2, "the minimum moved at least two of the columns")

    # Past the budget, where mojotrees is documented as `partial`. LightGBM
    # shrinks the budget to `max(1, min(max_bin, n / min_data_in_bin))` and
    # then runs its greedy inside it; mojotrees applies the same cap and then
    # cuts at quantile boundaries. What is claimed, and all that is claimed,
    # is that the cap binds the same way.
    var wide = List[Float64]()
    for r in range(n_rows):
        wide.append(Float64(r))
    # 240 distinct values against 64 bins, so the column is past the budget
    # and takes the quantile branch rather than the levels one.
    var capped = fit_bins(
        wide, n_rows, 1, 64, min_data_in_bin=7, bin_construct_sample_cnt=0
    )
    var lgbm_budget = n_rows // 7
    if lgbm_budget > 64:
        lgbm_budget = 64
    if lgbm_budget < 1:
        lgbm_budget = 1
    assert_equal(lgbm_budget, 34, "LightGBM's shrunken budget on this column")
    # The cap must bind, or the bound below would hold whatever the binner
    # did with `min_data_in_bin` in this branch.
    var uncapped = fit_bins(
        wide, n_rows, 1, 64, min_data_in_bin=1, bin_construct_sample_cnt=0
    )
    assert_true(
        _edge_count(uncapped, 0) + 1 > lgbm_budget,
        "without the cap this column spends more than the shrunken budget",
    )
    assert_true(
        _edge_count(capped, 0) + 1 <= lgbm_budget,
        "mojotrees stays inside LightGBM's shrunken budget",
    )
    _auto()


def test_the_stock_sample_fires_and_is_worker_independent() raises:
    """The default sample, on a matrix large enough for it to bind.

    Everything else about `bin_construct_sample_cnt` is tested at explicit
    small counts, which proves the mechanism and not the default. This one
    says nothing at all to `fit_bins` and asserts that 200,000 rows out of
    240,000 were used, by requiring the result to differ from the full fit --
    the gate, proved rather than assumed -- and then that it is identical at
    one, three, and eight workers.
    """
    var n_rows = 240_000
    var col = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        col.append(Float64(r))

    _serial()
    var stock = fit_bins(col, n_rows, 1, 255)
    var full = fit_bins(col, n_rows, 1, 255, bin_construct_sample_cnt=0)
    assert_true(
        not stock.matches(full),
        "the stock sample fired: 200,000 of 240,000 rows is not every row",
    )

    var workers = [3, 8]
    for w in range(len(workers)):
        _workers(workers[w])
        var again = fit_bins(col, n_rows, 1, 255)
        _assert_same_binning(
            stock, again, String("stock fit at ", workers[w], " workers")
        )
    _auto()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
