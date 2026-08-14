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
from std.testing import TestSuite, assert_equal, assert_true
from std.utils.numerics import inf, nan

from mojoboost.binning import (
    SELECT_MIN_ROWS,
    value_from_key,
    BinMapper,
    bin_equal_width,
    emit_quantile_edges,
    fit_bins,
    order_key,
    quantile_boundary_indices,
    resolve_ranks,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _splitmix64(state: UInt64) -> UInt64:
    var z = state + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _uniform(counter: UInt64) -> Float64:
    return Float64(_splitmix64(counter) >> 11) * (1.0 / 9007199254740992.0)


comptime NAN = nan[DType.float64]()
comptime INF = inf[DType.float64]()


def _serial():
    _ = setenv("MOJOBOOST_NUM_WORKERS", "1")


def _workers(n: Int):
    _ = setenv("MOJOBOOST_NUM_WORKERS", String(n))


def _auto():
    _ = setenv("MOJOBOOST_NUM_WORKERS", "0")


def _force_sorted_path():
    """Push the selection threshold past any column these tests build, so
    `fit_bins` takes the plain sort. Restored by `_restore_paths`."""
    _ = setenv("MOJOBOOST_BINNING_SELECT_MIN_ROWS", "1000000000")


def _restore_paths():
    _ = setenv("MOJOBOOST_BINNING_SELECT_MIN_ROWS", "0")


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
) raises -> BinMapper:
    """Fit with the shipped policy and with the sort forced, assert the two
    agree, and return the shipped one."""
    var fast = fit_bins(features, n_rows, n_features, max_bins)
    _force_sorted_path()
    var exact = fit_bins(features, n_rows, n_features, max_bins)
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
    var m = _fit_both_ways(features, 10, 1, 255, "every rank a boundary")
    assert_equal(m.edge_offsets[1], 3, "four levels, three gaps")
    assert_equal(m.edges[0], 0.5, "0 | 1")
    assert_equal(m.edges[1], 3.0, "1 | 5")
    assert_equal(m.edges[2], 7.0, "5 | 9")


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
    """Below the threshold nothing changed: assert the shape a small fit has
    always had, and that forcing the sort is a no-op there."""
    var features: List[Float64] = [3.0, 1.0, 2.0, 5.0, 4.0, 5.0, 5.0, 0.0]
    var mapper = _fit_both_ways(features, 8, 1, 4, "small column")
    # Sorted: 0 1 2 3 4 5 5 5. Boundaries at ranks 2, 4 and 6; the third
    # falls inside the run of fives and cuts nothing.
    assert_equal(mapper.edge_offsets[1], 2, "two edges, the third had no gap")
    _assert_transform_matches_bin_value(mapper, features, 8, "small column")


def test_equal_width_path_is_untouched() raises:
    var features: List[Float64] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    var data = bin_equal_width(features, 8, 1, 4)
    assert_equal(data.bin_at(0, 0), 0, "first row")
    assert_equal(data.bin_at(7, 0), 3, "last row")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
