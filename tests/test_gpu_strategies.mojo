"""GPU histogram strategy equivalence.

The scalable tiled path (per-threadgroup partials plus a deterministic
reduction kernel) and the preserved atomic fallback accumulate the same
exact fixed-point integers, so they must agree bit for bit, not merely to a
tolerance. Everything here asserts exact equality between the two, and
tolerance-based agreement only against the Float64 CPU builder.

Skips (passing) when no accelerator is present so the suite stays green on
CPU-only machines.
"""

from std.os import setenv
from std.sys import has_accelerator
from std.testing import assert_equal, assert_true, TestSuite

from mojoboost.binning import bin_equal_width, BinnedMatrix
from mojoboost.gpu_tiling import (
    STRATEGY_ATOMIC,
    STRATEGY_TILED,
    query_device_caps,
    strategy_name,
)
from mojoboost.histogram import (
    Histogram,
    build_histogram,
    build_histogram_subset,
)
from mojoboost.histogram_gpu import GpuHistogramBuilder


def _splitmix64(state: UInt64) -> UInt64:
    var z = state + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _uniform(counter: UInt64) -> Float64:
    return Float64(_splitmix64(counter) >> 11) * (1.0 / 9007199254740992.0)


def _make_data(
    n_rows: Int, n_features: Int, n_bins: Int
) raises -> BinnedMatrix:
    var features = List[Float64](capacity=n_rows * n_features)
    for k in range(n_rows * n_features):
        features.append(_uniform(UInt64(k)))
    return bin_equal_width(features, n_rows, n_features, n_bins)


def _make_grad_hess(
    n_rows: Int, base: UInt64
) -> Tuple[List[Float64], List[Float64]]:
    var grad = List[Float64](capacity=n_rows)
    var hess = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        grad.append(2.0 * _uniform(base + UInt64(r)) - 1.0)
        hess.append(_uniform(base + UInt64(n_rows + r)) + 0.01)
    return (grad^, hess^)


def _assert_identical(a: Histogram, b: Histogram) raises:
    assert_equal(a.n_features, b.n_features)
    assert_equal(a.n_bins, b.n_bins)
    for i in range(a.n_features * a.n_bins):
        assert_equal(a.grad[i], b.grad[i])
        assert_equal(a.hess[i], b.hess[i])
        assert_equal(a.count[i], b.count[i])


def test_device_capabilities_are_usable() raises:
    """Whatever the backend reports or refuses to report, the numbers the
    tiling policy consumes have to be positive and launchable."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var data = _make_data(1_000, 2, 16)
        var builder = GpuHistogramBuilder(data)
        assert_true(builder.caps.sm_count >= 1)
        assert_true(builder.caps.max_threads_per_block >= 64)
        assert_true(builder.caps.max_shared_memory_per_block >= 16 * 12)
        assert_true(
            builder.tiling.block_threads
            <= builder.caps.max_threads_per_block
        )
        assert_true(builder.tiling.n_tiles >= 1)


def test_strategies_agree_bit_exactly() raises:
    """The headline invariant: the reduction stage changes how partials
    combine, never what they sum to."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        # Shapes on both sides of the one-tile boundary: wide-and-short
        # gets one tile per feature, narrow-and-tall gets many.
        var shapes = [(20_000, 13, 64), (200_000, 4, 64), (5_000, 3, 255)]
        for i in range(len(shapes)):
            var n_rows = shapes[i][0]
            var n_features = shapes[i][1]
            var n_bins = shapes[i][2]
            var data = _make_data(n_rows, n_features, n_bins)
            var gh = _make_grad_hess(n_rows, UInt64(n_rows * n_features))

            var atomic = GpuHistogramBuilder(data, STRATEGY_ATOMIC)
            var tiled = GpuHistogramBuilder(data, STRATEGY_TILED)
            assert_equal(atomic.strategy(), STRATEGY_ATOMIC)
            assert_equal(tiled.strategy(), STRATEGY_TILED)
            _assert_identical(
                atomic.build(gh[0], gh[1]), tiled.build(gh[0], gh[1])
            )


def test_strategies_agree_on_leaf_filtered_builds() raises:
    """Equality has to hold for a node's rows, not just the whole dataset:
    the leaf filter is what every tree node after the root uses."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 100_000
        var n_features = 5
        var n_bins = 64
        var data = _make_data(n_rows, n_features, n_bins)
        var gh = _make_grad_hess(n_rows, UInt64(7_000_000))

        var atomic = GpuHistogramBuilder(data, STRATEGY_ATOMIC)
        var tiled = GpuHistogramBuilder(data, STRATEGY_TILED)
        assert_true(tiled.tiling.n_tiles > 1)

        atomic.upload_gradients(gh[0], gh[1])
        atomic.begin_tree()
        atomic.apply_split(0, 31, 0, 1, 2)
        tiled.upload_gradients(gh[0], gh[1])
        tiled.begin_tree()
        tiled.apply_split(0, 31, 0, 1, 2)

        var a_left = atomic.build_leaf(1)
        var t_left = tiled.build_leaf(1)
        var a_right = atomic.build_leaf(2)
        var t_right = tiled.build_leaf(2)
        _assert_identical(a_left, t_left)
        _assert_identical(a_right, t_right)

        # And both match the CPU builder's exact counts on the same split.
        var left_rows = List[Int]()
        var right_rows = List[Int]()
        for r in range(n_rows):
            if data.bin_at(r, 0) <= 31:
                left_rows.append(r)
            else:
                right_rows.append(r)
        var cpu_left = build_histogram_subset(data, gh[0], gh[1], left_rows)
        var cpu_right = build_histogram_subset(data, gh[0], gh[1], right_rows)
        for i in range(n_features * n_bins):
            assert_equal(cpu_left.count[i], t_left.count[i])
            assert_equal(cpu_right.count[i], t_right.count[i])


def test_tiled_matches_the_cpu_builder() raises:
    """Float32 fixed point, so tolerance against the Float64 CPU sums;
    counts are integer on both sides and must match exactly."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 200_000
        var n_features = 4
        var n_bins = 64
        var data = _make_data(n_rows, n_features, n_bins)
        var gh = _make_grad_hess(n_rows, UInt64(11_000_000))

        var g_mag = 0.0
        var h_mag = 0.0
        for r in range(n_rows):
            g_mag += abs(gh[0][r])
            h_mag += abs(gh[1][r])

        var cpu = build_histogram(data, gh[0], gh[1])
        var tiled = GpuHistogramBuilder(data, STRATEGY_TILED)
        assert_true(tiled.tiling.n_tiles > 1)
        var gpu = tiled.build(gh[0], gh[1])

        for i in range(n_features * n_bins):
            assert_equal(cpu.count[i], gpu.count[i])
            assert_true(abs(cpu.grad[i] - gpu.grad[i]) <= 1e-4 * g_mag + 1e-6)
            assert_true(abs(cpu.hess[i] - gpu.hess[i]) <= 1e-4 * h_mag + 1e-6)


def test_tiled_repeat_builds_are_identical() raises:
    """The reduction sums tiles in a fixed order over exact integers, so
    repeat builds are bit-identical, not merely close."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var data = _make_data(150_000, 3, 64)
        var gh = _make_grad_hess(150_000, UInt64(13_000_000))
        var tiled = GpuHistogramBuilder(data, STRATEGY_TILED)
        assert_true(tiled.tiling.n_tiles > 1)
        _assert_identical(tiled.build(gh[0], gh[1]), tiled.build(gh[0], gh[1]))


def test_strategies_agree_under_feature_subsampling() raises:
    """A narrowed grid.x re-derives the tiling against the already
    allocated partial buffer, and inactive features stay zero on both
    paths."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 100_000
        var n_features = 8
        var n_bins = 32
        var data = _make_data(n_rows, n_features, n_bins)
        var gh = _make_grad_hess(n_rows, UInt64(17_000_000))
        var active = List[Int]()
        active.append(1)
        active.append(2)
        active.append(5)

        var atomic = GpuHistogramBuilder(data, STRATEGY_ATOMIC)
        var tiled = GpuHistogramBuilder(data, STRATEGY_TILED)
        atomic.set_features(active)
        tiled.set_features(active)
        assert_true(
            tiled.tiling.partial_cells
            <= tiled.part_capacity
        )

        var a = atomic.build(gh[0], gh[1])
        var t = tiled.build(gh[0], gh[1])
        _assert_identical(a, t)

        # Inactive features are zero; active ones carry the dataset's rows.
        for f in range(n_features):
            var total = 0
            for b in range(n_bins):
                total += t.count[f * n_bins + b]
            if f == 1 or f == 2 or f == 5:
                assert_equal(total, n_rows)
            else:
                assert_equal(total, 0)


def test_env_overrides_change_geometry_not_results() raises:
    # One test owns all env mutation so no other test sees a dirty
    # environment regardless of suite ordering; empty string means unset.
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 60_000
        var n_features = 4
        var data = _make_data(n_rows, n_features, 64)
        var gh = _make_grad_hess(n_rows, UInt64(19_000_000))

        var default_builder = GpuHistogramBuilder(data)
        var expected = default_builder.build(gh[0], gh[1])

        # A forced row tile splits the rows differently and still sums the
        # same integers.
        _ = setenv("MOJOBOOST_GPU_ROW_TILE", "997")
        var forced = GpuHistogramBuilder(data, STRATEGY_TILED)
        # The request sets the tile count; rows per tile is then rebalanced
        # so no trailing tile is short, which can only shrink it.
        assert_true(forced.tiling.rows_per_tile <= 997)
        assert_true(forced.tiling.n_tiles > default_builder.tiling.n_tiles)
        _assert_identical(expected, forced.build(gh[0], gh[1]))
        _ = setenv("MOJOBOOST_GPU_ROW_TILE", "")

        # A forced block size changes how many rows each thread walks.
        _ = setenv("MOJOBOOST_GPU_BLOCK_THREADS", "64")
        var narrow = GpuHistogramBuilder(data, STRATEGY_TILED)
        assert_equal(narrow.tiling.block_threads, 64)
        _assert_identical(expected, narrow.build(gh[0], gh[1]))
        _ = setenv("MOJOBOOST_GPU_BLOCK_THREADS", "")

        # The strategy override reaches the builder through AUTO.
        _ = setenv("MOJOBOOST_GPU_HIST_STRATEGY", "atomic")
        var forced_atomic = GpuHistogramBuilder(data)
        assert_equal(forced_atomic.strategy(), STRATEGY_ATOMIC)
        _assert_identical(expected, forced_atomic.build(gh[0], gh[1]))

        _ = setenv("MOJOBOOST_GPU_HIST_STRATEGY", "tiled")
        var forced_tiled = GpuHistogramBuilder(data)
        assert_equal(forced_tiled.strategy(), STRATEGY_TILED)
        _assert_identical(expected, forced_tiled.build(gh[0], gh[1]))
        _ = setenv("MOJOBOOST_GPU_HIST_STRATEGY", "")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
