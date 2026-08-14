"""Active-row compaction: stable partition and range invariants.

The claims worth testing here are structural rather than numeric.

- A split's two children own contiguous halves of the parent's range, the
  live ranges keep tiling the active prefix with no gap and no overlap, and
  a range never escapes the buffer.
- The partition is *stable*: both sides come out in the relative order they
  had in the parent, so a compacted range is the CPU grower's row list for
  the same node, index for index. That is asserted directly against
  `tree.partition_rows`, the shipped CPU partition, for numerical splits,
  for both missing-bin directions, and for categorical set splits.
- Splitting one leaf touches nothing outside its own range, so its siblings'
  rows survive it untouched.
- The device kernels reproduce the host reference model exactly, and a
  histogram built over a compacted range equals the CPU histogram over that
  node's rows: exactly in the integer counts, and to Float32 precision in
  the gradient and hessian sums.

The device half skips (passing) when no accelerator is present, so the file
stays green on CPU-only machines; the host half runs everywhere.
"""

from std.sys import has_accelerator
from std.testing import assert_equal, assert_raises, assert_true, TestSuite
from max.gpu.host import DeviceBuffer, DeviceContext

from mojoboost.binning import BinnedMatrix
from mojoboost.categorical import CategoricalSpec, cat_add, cat_empty
from mojoboost.gpu_active_rows import (
    FEATURE_GROUP_MAX,
    GpuActiveRows,
    LeafRange,
    LeafRangeTable,
    RowRouting,
    partition_range_host,
)
from mojoboost.gpu_tiling import (
    STRATEGY_ATOMIC,
    STRATEGY_TILED,
    query_device_caps,
)
from mojoboost.histogram import build_histogram_subset
from mojoboost.split import SplitInfo
from mojoboost.tree import partition_rows


def _splitmix64(state: UInt64) -> UInt64:
    var z = state + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _make_data(
    n_rows: Int, n_features: Int, n_bins: Int, missing_bin: Int = -1
) raises -> BinnedMatrix:
    """A column-major binned matrix of pseudorandom bins. With
    `missing_bin >= 0` every feature reserves that bin, and the draw puts a
    meaningful share of rows in it."""
    var bins = List[UInt8](capacity=n_rows * n_features)
    for f in range(n_features):
        for r in range(n_rows):
            var v = _splitmix64(UInt64(f * n_rows + r) + 0x51ED2701)
            bins.append(UInt8(Int(v % UInt64(n_bins))))
    if missing_bin < 0:
        return BinnedMatrix(bins^, n_rows, n_features, n_bins)
    var table = List[Int](capacity=n_features)
    for _ in range(n_features):
        table.append(missing_bin)
    return BinnedMatrix(
        bins^, n_rows, n_features, n_bins, CategoricalSpec.none(), table^
    )


def _identity_rows(n: Int) -> List[Int32]:
    var rows = List[Int32](capacity=n)
    for r in range(n):
        rows.append(Int32(r))
    return rows^


def _zeros(n: Int) -> List[Int32]:
    var out = List[Int32](capacity=n)
    for _ in range(n):
        out.append(Int32(0))
    return out^


def _as_ints(rows: List[Int32], window: LeafRange) -> List[Int]:
    var out = List[Int](capacity=window.count())
    for j in range(window.begin, window.end):
        out.append(Int(rows[j]))
    return out^


def _assert_same(got: List[Int], want: List[Int]) raises:
    assert_equal(len(got), len(want))
    for i in range(len(got)):
        assert_equal(got[i], want[i])


def test_leaf_range_geometry() raises:
    var r = LeafRange(3, 10)
    assert_equal(r.count(), 7)
    assert_true(not r.is_empty())
    assert_true(r.contains(3))
    assert_true(r.contains(9))
    assert_true(not r.contains(10))
    assert_true(not r.contains(2))

    var empty = LeafRange.empty()
    assert_equal(empty.count(), 0)
    assert_true(empty.is_empty())
    # An empty range overlaps nothing, including itself.
    assert_true(not empty.overlaps(r))
    assert_true(not r.overlaps(empty))
    # Adjacent halves of one parent must not count as overlapping, or every
    # split would look like a violation.
    assert_true(not LeafRange(3, 6).overlaps(LeafRange(6, 10)))
    assert_true(LeafRange(3, 7).overlaps(LeafRange(6, 10)))


def test_children_take_contiguous_halves() raises:
    var table = LeafRangeTable(100)
    table.reset_root(40)
    assert_equal(table.get(0).begin, 0)
    assert_equal(table.get(0).end, 40)

    var left = table.split(0, 1, 2, 12)
    assert_equal(left.begin, 0)
    assert_equal(left.end, 12)
    assert_equal(table.get(1).begin, 0)
    assert_equal(table.get(1).end, 12)
    assert_equal(table.get(2).begin, 12)
    assert_equal(table.get(2).end, 40)
    # The parent kept nothing: its rows belong to the children now.
    assert_true(table.get(0).is_empty())
    table.check_invariants()

    # Splitting a child only subdivides that child's own range.
    _ = table.split(2, 3, 4, 5)
    assert_equal(table.get(3).begin, 12)
    assert_equal(table.get(3).end, 17)
    assert_equal(table.get(4).begin, 17)
    assert_equal(table.get(4).end, 40)
    assert_equal(table.get(1).begin, 0)
    assert_equal(table.get(1).end, 12)
    table.check_invariants()
    assert_equal(table.total_active(), 40)


def test_degenerate_splits_keep_the_invariants() raises:
    var table = LeafRangeTable(16)
    table.reset_root(16)
    # Everything left, then everything right: both are legal outcomes of a
    # split whose feature routes uniformly, and both must keep tiling.
    _ = table.split(0, 1, 2, 16)
    assert_equal(table.get(1).count(), 16)
    assert_equal(table.get(2).count(), 0)
    table.check_invariants()
    _ = table.split(1, 3, 4, 0)
    assert_equal(table.get(3).count(), 0)
    assert_equal(table.get(4).count(), 16)
    table.check_invariants()
    assert_equal(table.total_active(), 16)


def test_range_table_rejects_bad_wiring() raises:
    var table = LeafRangeTable(50)
    table.reset_root(20)
    with assert_raises():
        _ = table.split(0, 1, 2, 21)
    with assert_raises():
        _ = table.split(0, 1, 2, -1)
    with assert_raises():
        _ = table.split(0, 1, 1, 5)
    with assert_raises():
        _ = table.split(0, 0, 2, 5)
    with assert_raises():
        _ = table.get(7)
    with assert_raises():
        table.reset_root(51)

    _ = table.split(0, 1, 2, 5)
    # Reusing a live leaf id would orphan the rows it already owns.
    with assert_raises():
        _ = table.split(2, 1, 3, 2)
    table.check_invariants()


def test_live_ranges_tile_the_active_prefix() raises:
    var table = LeafRangeTable(64)
    table.reset_root(64)
    var next_node = 1
    var to_split = List[Int]()
    to_split.append(0)
    var cuts = [7, 19, 3, 11, 1, 23, 5]
    var c = 0
    # Leaf-wise growth: always split the frontier's first leaf, exactly as
    # the grower does, and check the tiling after every one.
    while len(to_split) > 0 and c < len(cuts):
        var parent = to_split[0]
        _ = to_split.pop(0)
        var n = table.get(parent).count()
        if n < 2:
            continue
        var n_left = cuts[c] % n
        c += 1
        var left = next_node
        var right = next_node + 1
        next_node += 2
        _ = table.split(parent, left, right, n_left)
        table.check_invariants()
        assert_equal(table.total_active(), 64)
        to_split.append(left)
        to_split.append(right)


def test_host_partition_is_stable() raises:
    var data = _make_data(64, 3, 8)
    var rows = _identity_rows(64)
    var scratch = _zeros(64)
    var routing = RowRouting.numerical(1, 3)
    var n_left = partition_range_host(
        rows, scratch, data, LeafRange(0, 64), routing
    )

    # Every row is still present exactly once.
    var seen = List[Int](capacity=64)
    seen.resize(64, 0)
    for j in range(64):
        seen[Int(rows[j])] += 1
    for r in range(64):
        assert_equal(seen[r], 1)

    # The left rows come first and both sides keep ascending order, which
    # for an identity-seeded root is the original relative order.
    var left_count = 0
    for j in range(64):
        var goes_left = routing.goes_left(data.bin_at(Int(rows[j]), 1))
        if j < n_left:
            assert_true(goes_left)
            left_count += 1
        else:
            assert_true(not goes_left)
    assert_equal(left_count, n_left)
    for j in range(1, n_left):
        assert_true(rows[j - 1] < rows[j])
    for j in range(n_left + 1, 64):
        assert_true(rows[j - 1] < rows[j])


def test_host_partition_matches_the_cpu_row_lists() raises:
    """The compacted range has to be the CPU grower's two row lists laid end
    to end, index for index, or the backends cannot be compared node by
    node."""
    var data = _make_data(96, 4, 16, missing_bin=0)
    var root = List[Int](capacity=96)
    for r in range(96):
        root.append(r)

    var cases = List[SplitInfo]()
    cases.append(SplitInfo(2, 7, 1.0, True, False))
    cases.append(SplitInfo(2, 7, 1.0, True, True))
    cases.append(SplitInfo(0, 0, 1.0, True, True))
    cases.append(SplitInfo(3, 15, 1.0, True, False))
    var bitset = cat_empty()
    cat_add(bitset, 3)
    cat_add(bitset, 5)
    cat_add(bitset, 11)
    cases.append(SplitInfo.categorical(1, 1.0, bitset))

    for i in range(len(cases)):
        var split = cases[i].copy()
        var missing_bin = -1
        if not split.is_categorical:
            missing_bin = data.missing_bin[split.feature]
        var want = partition_rows(data, root, split, missing_bin)

        var rows = _identity_rows(96)
        var scratch = _zeros(96)
        var routing = RowRouting.from_split(split, missing_bin)
        var n_left = partition_range_host(
            rows, scratch, data, LeafRange(0, 96), routing
        )
        assert_equal(n_left, len(want.left))
        _assert_same(_as_ints(rows, LeafRange(0, n_left)), want.left)
        _assert_same(_as_ints(rows, LeafRange(n_left, 96)), want.right)


def test_host_partition_leaves_the_rest_of_the_buffer_alone() raises:
    """A leaf's split may only touch that leaf's own slots; a sibling's rows
    have to survive it byte for byte."""
    var data = _make_data(40, 2, 8)
    var rows = _identity_rows(40)
    var scratch = _zeros(40)
    var before = rows.copy()

    var window = LeafRange(10, 25)
    var n_left = partition_range_host(
        rows, scratch, data, window, RowRouting.numerical(0, 4)
    )
    assert_true(n_left >= 0 and n_left <= window.count())
    for j in range(40):
        if j < window.begin or j >= window.end:
            assert_equal(rows[j], before[j])

    # And the window still holds exactly the rows it started with.
    var got = _as_ints(rows, window)
    var want = _as_ints(before, window)
    var seen = List[Int](capacity=40)
    seen.resize(40, 0)
    for i in range(len(got)):
        seen[got[i]] += 1
    for i in range(len(want)):
        seen[want[i]] -= 1
    for r in range(40):
        assert_equal(seen[r], 0)


def test_host_partition_rejects_ranges_and_routings_it_cannot_serve() raises:
    var data = _make_data(20, 2, 8)
    var rows = _identity_rows(20)
    var scratch = _zeros(20)
    with assert_raises():
        _ = partition_range_host(
            rows, scratch, data, LeafRange(0, 21), RowRouting.numerical(0, 3)
        )
    with assert_raises():
        _ = partition_range_host(
            rows, scratch, data, LeafRange(-1, 5), RowRouting.numerical(0, 3)
        )
    with assert_raises():
        _ = partition_range_host(
            rows, scratch, data, LeafRange(0, 20), RowRouting.numerical(2, 3)
        )
    with assert_raises():
        _ = partition_range_host(
            rows, scratch, data, LeafRange(0, 20), RowRouting.numerical(0, 8)
        )
    var short = _zeros(4)
    with assert_raises():
        _ = partition_range_host(
            rows, short, data, LeafRange(0, 20), RowRouting.numerical(0, 3)
        )


def test_routing_mirrors_the_split_it_came_from() raises:
    var numerical = SplitInfo(1, 5, 1.0, True, True)
    var routing = RowRouting.from_split(numerical, 0)
    for bin in range(8):
        assert_true(routing.goes_left(bin) == (bin <= 5))
    # The missing bin follows the default direction, not the threshold.
    assert_true(routing.goes_left(0))
    var right_default = RowRouting.from_split(SplitInfo(1, 5, 1.0, True), 7)
    assert_true(not right_default.goes_left(7))
    assert_true(right_default.goes_left(5))

    var bitset = cat_empty()
    cat_add(bitset, 2)
    cat_add(bitset, 70)
    var cat = RowRouting.from_split(
        SplitInfo.categorical(0, 1.0, bitset), -1
    )
    assert_true(cat.goes_left(2))
    assert_true(cat.goes_left(70))
    assert_true(not cat.goes_left(3))
    # Bin 0 is never a member, so missing/unseen/dropped route right.
    assert_true(not cat.goes_left(0))


def _upload_bins(
    mut ctx: DeviceContext, data: BinnedMatrix
) raises -> DeviceBuffer[DType.uint8]:
    # The comptime guard keeps the device instantiation out of CPU-only
    # builds: module-level helpers compile unconditionally, so without it a
    # machine with no accelerator fails the arch constraint at compile time
    # even though only guarded tests ever call this.
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var buf = ctx.enqueue_create_buffer[DType.uint8](len(data.bins))
        ctx.enqueue_copy(dst_buf=buf, src_ptr=data.bins.unsafe_ptr())
        ctx.synchronize()
        return buf^


def _assert_device_matches_host(
    mut rows: GpuActiveRows,
    mut bins: DeviceBuffer[DType.uint8],
    data: BinnedMatrix,
    mut host_rows: List[Int32],
    mut scratch: List[Int32],
    parent: Int,
    left: Int,
    right: Int,
    routing: RowRouting,
) raises:
    """Run one split on both the device and the reference model and require
    the whole buffer to agree, not just the partitioned window."""
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var window = rows.range_of(parent)
        var want_left = partition_range_host(
            host_rows, scratch, data, window, routing
        )
        var got_left = rows.partition(
            bins.unsafe_ptr(), parent, left, right, routing
        )
        assert_equal(got_left, want_left)
        var got = rows.download_rows()
        for j in range(len(host_rows)):
            assert_equal(Int(got[j]), Int(host_rows[j]))
        rows.ranges.check_invariants()


def test_device_partition_matches_the_host_reference() raises:
    comptime if not has_accelerator():
        return
    else:
        var n_rows = 3000
        var data = _make_data(n_rows, 3, 16, missing_bin=0)
        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var bins = _upload_bins(ctx, data)
        var rows = GpuActiveRows(
            ctx, n_rows, data.n_features, data.n_bins, caps
        )
        rows.begin_tree()
        assert_equal(rows.n_active(), n_rows)

        var host_rows = _identity_rows(n_rows)
        var scratch = _zeros(n_rows)
        # The identity seed has to survive the round trip before any split.
        var seeded = rows.download_rows()
        for r in range(n_rows):
            assert_equal(Int(seeded[r]), r)

        # A missing bin routing left, then each child split again, so both a
        # range starting at 0 and one starting mid-buffer are exercised.
        _assert_device_matches_host(
            rows,
            bins,
            data,
            host_rows,
            scratch,
            0,
            1,
            2,
            RowRouting.numerical(0, 7, missing_bin=0, default_left=True),
        )
        _assert_device_matches_host(
            rows,
            bins,
            data,
            host_rows,
            scratch,
            2,
            3,
            4,
            RowRouting.numerical(1, 3, missing_bin=0, default_left=False),
        )
        # Node 4 sits in the middle of the buffer and is two splits deep,
        # so this exercises a range anchored neither at 0 nor at the root.
        _assert_device_matches_host(
            rows,
            bins,
            data,
            host_rows,
            scratch,
            4,
            5,
            6,
            RowRouting.numerical(2, 11, missing_bin=0, default_left=True),
        )

        # Each live leaf's compacted range is the CPU grower's row list for
        # the same node: the same rows, in the same order.
        var root = List[Int](capacity=n_rows)
        for r in range(n_rows):
            root.append(r)
        var top = partition_rows(
            data, root, SplitInfo(0, 7, 1.0, True, True), 0
        )
        _assert_same(rows.download_range(1), top.left)
        var right_split = partition_rows(
            data, top.right, SplitInfo(1, 3, 1.0, True, False), 0
        )
        _assert_same(rows.download_range(3), right_split.left)
        var deep = partition_rows(
            data, right_split.right, SplitInfo(2, 11, 1.0, True, True), 0
        )
        _assert_same(rows.download_range(5), deep.left)
        _assert_same(rows.download_range(6), deep.right)
        # A split leaf owns nothing: its rows belong to its children now.
        assert_equal(rows.range_of(4).count(), 0)


def test_device_partition_handles_bags_and_categoricals() raises:
    comptime if not has_accelerator():
        return
    else:
        var n_rows = 1500
        var data = _make_data(n_rows, 2, 16)
        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var bins = _upload_bins(ctx, data)
        var rows = GpuActiveRows(
            ctx, n_rows, data.n_features, data.n_bins, caps
        )

        # An ascending bag, as the sampler draws them.
        var bag = List[Int]()
        for r in range(n_rows):
            if r % 3 == 0:
                bag.append(r)
        rows.begin_tree(bag)
        assert_equal(rows.n_active(), len(bag))
        assert_equal(rows.range_of(0).count(), len(bag))

        var host_rows = _zeros(n_rows)
        for i in range(len(bag)):
            host_rows[i] = Int32(bag[i])
        var scratch = _zeros(n_rows)
        var bitset = cat_empty()
        cat_add(bitset, 3)
        cat_add(bitset, 4)
        cat_add(bitset, 9)
        cat_add(bitset, 12)
        _assert_device_matches_host(
            rows,
            bins,
            data,
            host_rows,
            scratch,
            0,
            1,
            2,
            RowRouting.categorical(1, bitset),
        )

        # Out-of-bag rows are simply not in any range: no sentinel, and the
        # bagged node's rows are the CPU grower's bagged row lists.
        var want = partition_rows(
            data, bag, SplitInfo.categorical(1, 1.0, bitset), -1
        )
        _assert_same(rows.download_range(1), want.left)
        _assert_same(rows.download_range(2), want.right)
        assert_equal(
            rows.range_of(1).count() + rows.range_of(2).count(), len(bag)
        )


def test_device_partition_verifies_a_supplied_left_count() raises:
    comptime if not has_accelerator():
        return
    else:
        var n_rows = 700
        var data = _make_data(n_rows, 2, 8)
        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var bins = _upload_bins(ctx, data)
        var rows = GpuActiveRows(
            ctx, n_rows, data.n_features, data.n_bins, caps
        )
        rows.verify_counts = True
        rows.begin_tree()

        var routing = RowRouting.numerical(0, 3)
        var host_rows = _identity_rows(n_rows)
        var scratch = _zeros(n_rows)
        var want_left = partition_range_host(
            host_rows, scratch, data, LeafRange(0, n_rows), routing
        )
        # The grower passes the count it read off the parent histogram; the
        # device must agree with it.
        var got = rows.partition(
            bins.unsafe_ptr(), 0, 1, 2, routing, expected_left=want_left
        )
        assert_equal(got, want_left)
        assert_equal(rows.range_of(1).count(), want_left)

        # A wrong count is caught rather than silently corrupting the ranges.
        var wrong = want_left - 1
        if wrong < 0:
            wrong = want_left + 1
        rows.begin_tree()
        with assert_raises():
            _ = rows.partition(
                bins.unsafe_ptr(), 0, 1, 2, routing, expected_left=wrong
            )


def _fixed_scale(values: List[Float64]) -> Float32:
    var total = 0.0
    for i in range(len(values)):
        total += abs(values[i])
    if total < 1e-12:
        total = 1e-12
    return Float32(Float64(1 << 30) / total)


def _range_histogram_case(strategy: Int) raises:
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var n_rows = 2048
        var n_features = 3
        var n_bins = 16
        var data = _make_data(n_rows, n_features, n_bins)
        var grad = List[Float64](capacity=n_rows)
        var hess = List[Float64](capacity=n_rows)
        for r in range(n_rows):
            grad.append(
                Float64(Int(_splitmix64(UInt64(r) + 7) % 2000)) * 0.001 - 1.0
            )
            hess.append(
                Float64(Int(_splitmix64(UInt64(r) + 991) % 1000)) * 0.001
                + 0.25
            )

        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var bins = _upload_bins(ctx, data)
        var rows = GpuActiveRows(ctx, n_rows, n_features, n_bins, caps)
        rows.begin_tree()

        var g_scale = _fixed_scale(grad)
        var h_scale = _fixed_scale(hess)
        var grad32 = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
        var hess32 = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
        for r in range(n_rows):
            grad32.unsafe_ptr().unsafe_store(r, Float32(grad[r]))
            hess32.unsafe_ptr().unsafe_store(r, Float32(hess[r]))
        var grad_dev = ctx.enqueue_create_buffer[DType.float32](n_rows)
        var hess_dev = ctx.enqueue_create_buffer[DType.float32](n_rows)
        ctx.enqueue_copy(dst_buf=grad_dev, src_ptr=grad32.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=hess_dev, src_ptr=hess32.unsafe_ptr())

        var feat_dev = ctx.enqueue_create_buffer[DType.int32](n_features)
        with feat_dev.map_to_host() as host:
            for f in range(n_features):
                host.unsafe_ptr().unsafe_store(f, Int32(f))

        var hist_size = n_features * n_bins
        var out_dev = ctx.enqueue_create_buffer[DType.int32](3 * hist_size)
        var host_out = ctx.enqueue_create_host_buffer[DType.int32](
            3 * hist_size
        )

        # Split once so the histogram is built over a proper sub-range that
        # does not start at 0, which is the case the leaf-id filter never
        # exercised.
        var routing = RowRouting.numerical(0, 6)
        _ = rows.partition(bins.unsafe_ptr(), 0, 1, 2, routing)
        var node = 2
        var node_rows = rows.download_range(node)
        assert_true(len(node_rows) > 0)

        var tiling = rows.range_tiling(
            caps, node, n_features, strategy, 1 << 20
        )
        assert_equal(tiling.strategy, strategy)
        var part_cells = tiling.partial_cells
        if part_cells < 1:
            part_cells = 1
        var part_dev = ctx.enqueue_create_buffer[DType.int32](3 * part_cells)

        rows.enqueue_range_histogram(
            tiling,
            node,
            bins.unsafe_ptr(),
            grad_dev.unsafe_ptr(),
            hess_dev.unsafe_ptr(),
            feat_dev.unsafe_ptr(),
            out_dev.unsafe_ptr(),
            part_dev.unsafe_ptr(),
            n_features,
            g_scale,
            h_scale,
        )
        ctx.enqueue_copy(dst_ptr=host_out.unsafe_ptr(), src_buf=out_dev)
        ctx.synchronize()

        var want = build_histogram_subset(data, grad, hess, node_rows)
        var src = host_out.unsafe_ptr()
        var g_inv = 1.0 / Float64(g_scale)
        var h_inv = 1.0 / Float64(h_scale)
        var total_count = 0
        for i in range(hist_size):
            # Counts are exact integers on both sides.
            assert_equal(
                Int(src.unsafe_load(2 * hist_size + i)), want.count[i]
            )
            total_count += want.count[i]
            var got_g = Float64(src.unsafe_load(i)) * g_inv
            var got_h = Float64(src.unsafe_load(hist_size + i)) * h_inv
            assert_true(abs(got_g - want.grad[i]) < 1e-3)
            assert_true(abs(got_h - want.hess[i]) < 1e-3)
        # Every row of the node landed in exactly one bin of every feature.
        assert_equal(total_count, len(node_rows) * n_features)


def test_range_histogram_matches_the_cpu_subset_atomic() raises:
    comptime if not has_accelerator():
        return
    else:
        _range_histogram_case(STRATEGY_ATOMIC)


def test_range_histogram_matches_the_cpu_subset_tiled() raises:
    comptime if not has_accelerator():
        return
    else:
        _range_histogram_case(STRATEGY_TILED)


def test_paired_feature_group_builds_the_identical_histogram() raises:
    """Two features per threadgroup changes the launch, not the histogram.

    The paired kernel exists to spend one row index, one gradient, and one
    hessian on two features instead of one, so the only thing it may change
    is how long the build takes. Both kernels accumulate fixed-point Int32
    and integer addition does not care in what order it happens, so the two
    outputs have to agree bit for bit, not to a tolerance. Both an even and
    an odd active-feature count are built: the odd one leaves the last
    threadgroup unpaired, which is the case that could drop or double a
    feature.
    """
    comptime if not has_accelerator():
        return
    else:
        var n_rows = 2048
        var n_features = 3
        var n_bins = 16
        var data = _make_data(n_rows, n_features, n_bins)
        var grad = List[Float64](capacity=n_rows)
        var hess = List[Float64](capacity=n_rows)
        for r in range(n_rows):
            grad.append(
                Float64(Int(_splitmix64(UInt64(r) + 7) % 2000)) * 0.001 - 1.0
            )
            hess.append(
                Float64(Int(_splitmix64(UInt64(r) + 991) % 1000)) * 0.001
                + 0.25
            )

        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var bins = _upload_bins(ctx, data)
        var rows = GpuActiveRows(ctx, n_rows, n_features, n_bins, caps)
        rows.begin_tree()

        var g_scale = _fixed_scale(grad)
        var h_scale = _fixed_scale(hess)
        var grad32 = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
        var hess32 = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
        for r in range(n_rows):
            grad32.unsafe_ptr().unsafe_store(r, Float32(grad[r]))
            hess32.unsafe_ptr().unsafe_store(r, Float32(hess[r]))
        var grad_dev = ctx.enqueue_create_buffer[DType.float32](n_rows)
        var hess_dev = ctx.enqueue_create_buffer[DType.float32](n_rows)
        ctx.enqueue_copy(dst_buf=grad_dev, src_ptr=grad32.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=hess_dev, src_ptr=hess32.unsafe_ptr())

        var feat_dev = ctx.enqueue_create_buffer[DType.int32](n_features)
        with feat_dev.map_to_host() as host:
            # Feature 1 is left out of the two-slot pass on purpose, so the
            # slots are not the identity and the pairing has to follow
            # `feat_ids` rather than `block_idx.x`.
            host.unsafe_ptr().unsafe_store(0, Int32(0))
            host.unsafe_ptr().unsafe_store(1, Int32(2))
            host.unsafe_ptr().unsafe_store(2, Int32(1))

        var hist_size = n_features * n_bins
        var cells = 3 * hist_size
        var one_dev = ctx.enqueue_create_buffer[DType.int32](cells)
        var two_dev = ctx.enqueue_create_buffer[DType.int32](cells)
        var host_one = ctx.enqueue_create_host_buffer[DType.int32](cells)
        var host_two = ctx.enqueue_create_host_buffer[DType.int32](cells)
        var part_dev = ctx.enqueue_create_buffer[DType.int32](1)

        # A node whose range does not start at 0, as in the other range
        # tests: the pairing must not disturb the row window.
        var routing = RowRouting.numerical(0, 6)
        _ = rows.partition(bins.unsafe_ptr(), 0, 1, 2, routing)
        var node = 2
        assert_true(len(rows.download_range(node)) > 0)

        for step in range(2):
            # Two active features, then three: the odd count is the one that
            # leaves a threadgroup unpaired.
            var slots = 2 + step
            var tiling = rows.range_tiling(
                caps, node, slots, STRATEGY_ATOMIC, 1
            )
            assert_equal(tiling.strategy, STRATEGY_ATOMIC)

            rows.set_feature_group(1)
            assert_equal(rows.feature_group, 1)
            rows.enqueue_range_histogram(
                tiling,
                node,
                bins.unsafe_ptr(),
                grad_dev.unsafe_ptr(),
                hess_dev.unsafe_ptr(),
                feat_dev.unsafe_ptr(),
                one_dev.unsafe_ptr(),
                part_dev.unsafe_ptr(),
                slots,
                g_scale,
                h_scale,
            )
            rows.set_feature_group(2)
            assert_equal(rows.feature_group, 2)
            rows.enqueue_range_histogram(
                tiling,
                node,
                bins.unsafe_ptr(),
                grad_dev.unsafe_ptr(),
                hess_dev.unsafe_ptr(),
                feat_dev.unsafe_ptr(),
                two_dev.unsafe_ptr(),
                part_dev.unsafe_ptr(),
                slots,
                g_scale,
                h_scale,
            )
            ctx.enqueue_copy(dst_ptr=host_one.unsafe_ptr(), src_buf=one_dev)
            ctx.enqueue_copy(dst_ptr=host_two.unsafe_ptr(), src_buf=two_dev)
            ctx.synchronize()

            var a = host_one.unsafe_ptr()
            var b = host_two.unsafe_ptr()
            var populated = 0
            for i in range(cells):
                assert_equal(Int(a.unsafe_load(i)), Int(b.unsafe_load(i)))
                if a.unsafe_load(i) != 0:
                    populated += 1
            # A comparison of two all-zero buffers would pass for the wrong
            # reason.
            assert_true(populated > 0)
        rows.set_feature_group(1)


def test_feature_group_rejects_a_width_it_has_no_kernel_for() raises:
    comptime if not has_accelerator():
        return
    else:
        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var rows = GpuActiveRows(ctx, 32, 2, 4, caps)
        with assert_raises():
            rows.set_feature_group(0)
        with assert_raises():
            rows.set_feature_group(FEATURE_GROUP_MAX + 1)


def _fused_subtract_case(strategy: Int, group: Int = 1) raises:
    """A build that folds the sibling subtraction in leaves the pool holding
    exactly what a build followed by a separate subtraction would.

    `group` is the histogram launch shape, and running the whole case at 2 is
    what keeps the paired kernel's copy of the fused subtraction honest: it
    subtracts per slice, so a block that owns two slices has to subtract
    twice, and an unpaired tail block once. Three features makes the last
    block unpaired.

    The fused form is what a device-resident frontier spends per split, and
    what it saves is a launch plus a pass over two whole slots, so the claim
    it has to earn is that it changes nothing. This builds the same node
    twice into a two-slot pool: once plain, to get the child histogram and to
    subtract on the host, and once fused against a parent slot seeded with a
    known histogram. Both are exact fixed-point Int32, so the comparison is
    bit for bit rather than to a tolerance.
    """
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var n_rows = 2048
        var n_features = 3
        var n_bins = 16
        var data = _make_data(n_rows, n_features, n_bins)
        var grad = List[Float64](capacity=n_rows)
        var hess = List[Float64](capacity=n_rows)
        for r in range(n_rows):
            grad.append(
                Float64(Int(_splitmix64(UInt64(r) + 7) % 2000)) * 0.001 - 1.0
            )
            hess.append(
                Float64(Int(_splitmix64(UInt64(r) + 991) % 1000)) * 0.001
                + 0.25
            )

        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var bins = _upload_bins(ctx, data)
        var rows = GpuActiveRows(ctx, n_rows, n_features, n_bins, caps)
        rows.set_feature_group(group)
        rows.begin_tree()

        var g_scale = _fixed_scale(grad)
        var h_scale = _fixed_scale(hess)
        var grad32 = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
        var hess32 = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
        for r in range(n_rows):
            grad32.unsafe_ptr().unsafe_store(r, Float32(grad[r]))
            hess32.unsafe_ptr().unsafe_store(r, Float32(hess[r]))
        var grad_dev = ctx.enqueue_create_buffer[DType.float32](n_rows)
        var hess_dev = ctx.enqueue_create_buffer[DType.float32](n_rows)
        ctx.enqueue_copy(dst_buf=grad_dev, src_ptr=grad32.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=hess_dev, src_ptr=hess32.unsafe_ptr())

        var feat_dev = ctx.enqueue_create_buffer[DType.int32](n_features)
        with feat_dev.map_to_host() as host:
            for f in range(n_features):
                host.unsafe_ptr().unsafe_store(f, Int32(f))

        var hist_size = n_features * n_bins
        var cells = 3 * hist_size
        # Two slots of one pool, exactly as the resident frontier holds them:
        # slot 0 is the parent to derive from, slot 1 is the child to build.
        var pool = ctx.enqueue_create_buffer[DType.int32](2 * cells)
        var host_pool = ctx.enqueue_create_host_buffer[DType.int32](2 * cells)

        _ = rows.partition(
            bins.unsafe_ptr(), 0, 1, 2, RowRouting.numerical(0, 6)
        )
        var node = 2
        assert_true(rows.range_of(node).count() > 0)

        var tiling = rows.range_tiling(
            caps, node, n_features, strategy, 1 << 20
        )
        assert_equal(tiling.strategy, strategy)
        var part_cells = tiling.partial_cells
        if part_cells < 1:
            part_cells = 1
        var part_dev = ctx.enqueue_create_buffer[DType.int32](3 * part_cells)

        # The child alone, into slot 1, with nothing to subtract.
        rows.enqueue_range_histogram(
            tiling,
            node,
            bins.unsafe_ptr(),
            grad_dev.unsafe_ptr(),
            hess_dev.unsafe_ptr(),
            feat_dev.unsafe_ptr(),
            pool.unsafe_ptr().unsafe_offset(cells),
            part_dev.unsafe_ptr(),
            n_features,
            g_scale,
            h_scale,
        )
        ctx.enqueue_copy(dst_ptr=host_pool.unsafe_ptr(), src_buf=pool)
        ctx.synchronize()
        var child = List[Int32](capacity=cells)
        for i in range(cells):
            child.append(host_pool.unsafe_ptr().unsafe_load(cells + i))

        # Seed a parent with values a subtraction cannot hide in, then build
        # the same node again with the subtraction folded in.
        with pool.map_to_host() as host:
            var p = host.unsafe_ptr()
            for i in range(cells):
                p.unsafe_store(i, Int32(1000 + 7 * i))
                p.unsafe_store(cells + i, Int32(0))
        rows.enqueue_range_histogram(
            tiling,
            node,
            bins.unsafe_ptr(),
            grad_dev.unsafe_ptr(),
            hess_dev.unsafe_ptr(),
            feat_dev.unsafe_ptr(),
            pool.unsafe_ptr().unsafe_offset(cells),
            part_dev.unsafe_ptr(),
            n_features,
            g_scale,
            h_scale,
            sub_offset=-cells,
            subtract=True,
        )
        ctx.enqueue_copy(dst_ptr=host_pool.unsafe_ptr(), src_buf=pool)
        ctx.synchronize()

        var got = host_pool.unsafe_ptr()
        for i in range(cells):
            # The build itself is untouched by folding the subtraction in.
            assert_equal(Int(got.unsafe_load(cells + i)), Int(child[i]))
            # And the parent slot now holds parent - child, bin for bin.
            assert_equal(
                Int(got.unsafe_load(i)), 1000 + 7 * i - Int(child[i])
            )


def test_fused_subtract_matches_a_separate_subtraction_atomic() raises:
    comptime if not has_accelerator():
        return
    else:
        _fused_subtract_case(STRATEGY_ATOMIC)


def test_fused_subtract_matches_a_separate_subtraction_tiled() raises:
    comptime if not has_accelerator():
        return
    else:
        _fused_subtract_case(STRATEGY_TILED)


def test_fused_subtract_matches_a_separate_subtraction_paired() raises:
    comptime if not has_accelerator():
        return
    else:
        _fused_subtract_case(STRATEGY_ATOMIC, group=2)


def test_fused_subtract_needs_a_slot_other_than_its_own() raises:
    """A zero offset would aim the subtraction at the build's own slot, which
    is not a sibling subtraction but a self-cancelling one."""
    comptime if not has_accelerator():
        return
    else:
        var n_rows = 128
        var n_features = 2
        var n_bins = 8
        var data = _make_data(n_rows, n_features, n_bins)
        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var bins = _upload_bins(ctx, data)
        var rows = GpuActiveRows(ctx, n_rows, n_features, n_bins, caps)
        rows.begin_tree()
        var grad_dev = ctx.enqueue_create_buffer[DType.float32](n_rows)
        var hess_dev = ctx.enqueue_create_buffer[DType.float32](n_rows)
        var feat_dev = ctx.enqueue_create_buffer[DType.int32](n_features)
        var hist_size = n_features * n_bins
        var out_dev = ctx.enqueue_create_buffer[DType.int32](3 * hist_size)
        var part_dev = ctx.enqueue_create_buffer[DType.int32](1)
        var tiling = rows.range_tiling(caps, 0, n_features, STRATEGY_ATOMIC, 0)
        var raised = False
        try:
            rows.enqueue_range_histogram(
                tiling,
                0,
                bins.unsafe_ptr(),
                grad_dev.unsafe_ptr(),
                hess_dev.unsafe_ptr(),
                feat_dev.unsafe_ptr(),
                out_dev.unsafe_ptr(),
                part_dev.unsafe_ptr(),
                n_features,
                Float32(1.0),
                Float32(1.0),
                sub_offset=0,
                subtract=True,
            )
        except:
            raised = True
        assert_true(raised)


def test_range_histogram_of_an_empty_leaf_is_zero() raises:
    comptime if not has_accelerator():
        return
    else:
        var n_rows = 256
        var n_features = 2
        var n_bins = 8
        var data = _make_data(n_rows, n_features, n_bins)
        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var bins = _upload_bins(ctx, data)
        var rows = GpuActiveRows(ctx, n_rows, n_features, n_bins, caps)
        rows.begin_tree()
        # Threshold at the top bin sends every row left, so the right child
        # owns nothing at all.
        _ = rows.partition(
            bins.unsafe_ptr(), 0, 1, 2, RowRouting.numerical(0, n_bins - 1)
        )
        assert_equal(rows.range_of(2).count(), 0)
        assert_equal(rows.range_of(1).count(), n_rows)

        var ones = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
        for r in range(n_rows):
            ones.unsafe_ptr().unsafe_store(r, Float32(1.0))
        var grad_dev = ctx.enqueue_create_buffer[DType.float32](n_rows)
        var hess_dev = ctx.enqueue_create_buffer[DType.float32](n_rows)
        ctx.enqueue_copy(dst_buf=grad_dev, src_ptr=ones.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=hess_dev, src_ptr=ones.unsafe_ptr())
        var feat_dev = ctx.enqueue_create_buffer[DType.int32](n_features)
        with feat_dev.map_to_host() as host:
            for f in range(n_features):
                host.unsafe_ptr().unsafe_store(f, Int32(f))
        var hist_size = n_features * n_bins
        var out_dev = ctx.enqueue_create_buffer[DType.int32](3 * hist_size)
        var host_out = ctx.enqueue_create_host_buffer[DType.int32](
            3 * hist_size
        )
        var part_dev = ctx.enqueue_create_buffer[DType.int32](1)

        var tiling = rows.range_tiling(caps, 2, n_features, STRATEGY_ATOMIC, 0)
        rows.enqueue_range_histogram(
            tiling,
            2,
            bins.unsafe_ptr(),
            grad_dev.unsafe_ptr(),
            hess_dev.unsafe_ptr(),
            feat_dev.unsafe_ptr(),
            out_dev.unsafe_ptr(),
            part_dev.unsafe_ptr(),
            n_features,
            Float32(1.0),
            Float32(1.0),
        )
        ctx.enqueue_copy(dst_ptr=host_out.unsafe_ptr(), src_buf=out_dev)
        ctx.synchronize()
        var src = host_out.unsafe_ptr()
        for i in range(3 * hist_size):
            assert_equal(Int(src.unsafe_load(i)), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
