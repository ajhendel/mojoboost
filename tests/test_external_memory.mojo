"""External-memory dataset construction against the resident path.

The claim `external_memory.mojo` makes is that streaming a source through
its census, block-binning, and transform passes produces the same binning
and the same model the resident `Dataset` produces from the same matrix.
These are the checks its design document listed as unrun: multi-pass edges
equal single-pass edges, a chunk binned alone equals its slice of the whole,
the cache round-trips exactly through its checksums, the row-coverage rule
rejects gaps and overlaps, and a cancelled token stops a build.
"""

from std.testing import (
    assert_equal,
    assert_raises,
    assert_true,
    TestSuite,
)

from mojotrees import (
    SQUARED_ERROR,
    BoosterParams,
    Dataset,
    TreeParams,
    train_dataset,
)
from mojotrees.binning import fit_bins
from mojotrees.external_memory import (
    CacheLayout,
    ExternalMemoryParams,
    build_external_dataset_from_raw,
    open_external_dataset,
    train_external,
)
from mojotrees.raw_data import RawData
from mojotrees.sequence import CancelToken, RowIdRange, check_row_coverage
from support import _make_features as _features


# Cache files land in the working directory as `.test_extmem_*.mbx`
# (`.gitignore` covers `.test_*.tmp` and `*.mbx`); each test discards its
# cache, which truncates the files.
comptime _CACHE_PREFIX = ".test_extmem_"


def _target(features: List[Float64], n_rows: Int) -> List[Float64]:
    var y = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        var x0 = features[0 * n_rows + r]
        var x1 = features[1 * n_rows + r]
        var x2 = features[2 * n_rows + r]
        y.append(4.0 * x0 - 3.0 * x1 + 2.0 * (x2 - 0.5) * (x2 - 0.5))
    return y^


def _row(
    features: List[Float64], n_rows: Int, n_features: Int, r: Int
) -> List[Float64]:
    var row = List[Float64](capacity=n_features)
    for f in range(n_features):
        row.append(features[f * n_rows + r])
    return row^


def _params(n_rounds: Int) -> BoosterParams:
    var tree = TreeParams.default()
    tree.num_leaves = 8
    tree.min_data_in_leaf = 5
    return BoosterParams(n_rounds, 0.1, tree^)


def _layout(prefix: String) raises -> CacheLayout:
    return CacheLayout(String(""), String(_CACHE_PREFIX) + prefix + ".tmp")


def test_streamed_bins_equal_resident_bins() raises:
    # 130 rows cut into chunks of 17 leaves a short last chunk, and a block
    # width of one feature forces one binning pass per column.
    var n_rows = 130
    var n_features = 4
    var features = _features(n_rows, n_features)
    var label = _target(features, n_rows)

    var params = ExternalMemoryParams(
        max_bin=16, chunk_rows=17, bin_memory_budget=8 * n_rows
    )
    var cancel = CancelToken.none()
    var ext = build_external_dataset_from_raw(
        RawData.dense(features.copy(), n_rows, n_features),
        _layout("bins"),
        params,
        cancel,
        label.copy(),
    )
    var resident = fit_bins(
        Span(features), n_rows, n_features, max_bins=16
    )

    assert_equal(ext.mapper.n_features, resident.n_features)
    assert_equal(ext.mapper.n_bins, resident.n_bins)
    assert_equal(len(ext.mapper.edges), len(resident.edges))
    for i in range(len(resident.edges)):
        assert_equal(ext.mapper.edges[i], resident.edges[i])
    for i in range(len(resident.edge_offsets)):
        assert_equal(ext.mapper.edge_offsets[i], resident.edge_offsets[i])

    # The whole matrix materialized equals the resident transform, and each
    # chunk read alone equals its slice.
    var whole = ext.materialize_binned(n_rows * n_features)
    var expected = resident.transform(features, n_rows)
    assert_equal(whole.n_rows, n_rows)
    assert_equal(len(whole.bins), len(expected.bins))
    for i in range(len(expected.bins)):
        assert_equal(whole.bins[i], expected.bins[i])
    var seen = 0
    for c in range(len(ext.manifest.chunks)):
        var chunk = ext.binned_chunk(c)
        var base = ext.manifest.chunks[c].base
        for f in range(n_features):
            for r in range(chunk.n_rows):
                assert_equal(
                    chunk.bins[f * chunk.n_rows + r],
                    expected.bins[f * n_rows + base + r],
                )
        seen += chunk.n_rows
    assert_equal(seen, n_rows)
    assert_true(cancel.polls > 0)
    _ = ext.discard()


def test_cache_reopens_and_refuses_the_wrong_mapper() raises:
    var n_rows = 90
    var n_features = 3
    var features = _features(n_rows, n_features)
    var params = ExternalMemoryParams(max_bin=12, chunk_rows=25)
    var cancel = CancelToken.none()
    var ext = build_external_dataset_from_raw(
        RawData.dense(features.copy(), n_rows, n_features),
        _layout("reopen"),
        params,
        cancel,
        _target(features, n_rows),
    )
    var reopened = open_external_dataset(_layout("reopen"), ext.mapper)
    assert_equal(reopened.manifest.n_rows, n_rows)
    assert_equal(len(reopened.manifest.chunks), len(ext.manifest.chunks))
    var a = ext.materialize_binned(n_rows * n_features)
    var b = reopened.materialize_binned(n_rows * n_features)
    for i in range(len(a.bins)):
        assert_equal(a.bins[i], b.bins[i])
    var fields = reopened.row_fields()
    assert_equal(len(fields.label), n_rows)

    var other = fit_bins(Span(features), n_rows, n_features, max_bins=7)
    with assert_raises():
        _ = open_external_dataset(_layout("reopen"), other)
    with assert_raises():
        _ = ext.materialize_binned(1)
    _ = ext.discard()
    with assert_raises():
        _ = ext.materialize_binned(n_rows * n_features)


def test_external_training_matches_resident_training() raises:
    var n_rows = 160
    var n_features = 4
    var features = _features(n_rows, n_features)
    var label = _target(features, n_rows)
    var params = ExternalMemoryParams(max_bin=32, chunk_rows=50)
    var cancel = CancelToken.none()
    var ext = build_external_dataset_from_raw(
        RawData.dense(features.copy(), n_rows, n_features),
        _layout("train"),
        params,
        cancel,
        label.copy(),
    )
    var streamed = train_external(
        ext, SQUARED_ERROR, _params(20), n_rows * n_features
    )
    var ds = Dataset(features, n_rows, n_features, label.copy(), max_bin=32)
    var resident = train_dataset(ds, SQUARED_ERROR, _params(20))
    assert_equal(
        len(streamed.booster.trees), len(resident.booster.trees)
    )
    for r in range(n_rows):
        var row = _row(features, n_rows, n_features, r)
        assert_equal(streamed.predict(row), resident.predict(row))
    _ = ext.discard()


def test_row_coverage_rejects_gaps_and_overlaps() raises:
    var ok = List[RowIdRange]()
    ok.append(RowIdRange(0, 4))
    ok.append(RowIdRange(4, 6))
    check_row_coverage(ok, 10)
    with assert_raises():
        var gap = List[RowIdRange]()
        gap.append(RowIdRange(0, 4))
        gap.append(RowIdRange(5, 5))
        check_row_coverage(gap, 10)
    with assert_raises():
        var overlap = List[RowIdRange]()
        overlap.append(RowIdRange(0, 6))
        overlap.append(RowIdRange(4, 6))
        check_row_coverage(overlap, 10)
    with assert_raises():
        check_row_coverage(ok, 11)


def test_cancelled_token_stops_a_build() raises:
    var n_rows = 40
    var n_features = 2
    var features = _features(n_rows, n_features)
    var cancel = CancelToken.none()
    cancel.cancel()
    with assert_raises():
        _ = build_external_dataset_from_raw(
            RawData.dense(features.copy(), n_rows, n_features),
            _layout("cancel"),
            ExternalMemoryParams(chunk_rows=10),
            cancel,
        )
    var reasoned = CancelToken.live()
    assert_true(not reasoned.is_cancelled())
    reasoned.cancel(String("user asked"))
    assert_equal(reasoned.why(), String("user asked"))
    with assert_raises():
        reasoned.check(String("tree 3"))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
