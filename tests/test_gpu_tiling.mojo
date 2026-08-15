"""GPU histogram tiling policy.

`derive_tiling` is pure host-side arithmetic over a `DeviceCaps` value, so
the launch-geometry policy is tested here on every machine, with or without
an accelerator. The device-specific half (`query_device_caps`) is exercised
by the GPU tests in test_gpu_strategies.mojo.

The invariants that matter are structural rather than numeric: the tiles
have to cover every row exactly once with no empty trailing tile, the block
size has to be launchable on the device, the partial buffer has to stay
inside whatever bound it was given, and a caller that asks for one strategy
has to get it.
"""

from std.os import setenv
from std.testing import assert_equal, assert_true, TestSuite

from mojotrees.gpu_tiling import (
    MAX_GRID_DIM_Y,
    STRATEGY_ATOMIC,
    STRATEGY_AUTO,
    STRATEGY_TILED,
    WARP_GRANULARITY,
    DeviceCaps,
    derive_block_threads,
    derive_tiling,
    env_strategy,
    shared_bytes_for,
    strategy_name,
)


def _small_gpu() -> DeviceCaps:
    """Integrated class: Apple M4 reports these exactly."""
    return DeviceCaps(10, 1024, 32768)


def _large_gpu() -> DeviceCaps:
    """Discrete class: an A100-shaped device."""
    return DeviceCaps(108, 1024, 49152)


def _assert_covers_rows(
    n_rows: Int, n_tiles: Int, rows_per_tile: Int
) raises:
    """Tiles must cover every row, and the last tile must not be empty."""
    assert_true(n_tiles >= 1)
    assert_true(rows_per_tile >= 1)
    assert_true(n_tiles * rows_per_tile >= n_rows)
    assert_true((n_tiles - 1) * rows_per_tile < n_rows)


def test_tiles_cover_every_row_across_shapes() raises:
    var caps = _small_gpu()
    var rows = [1, 2, 999, 8_192, 100_000, 1_000_000]
    var features = [1, 3, 50, 100]
    for r in range(len(rows)):
        for f in range(len(features)):
            var t = derive_tiling(caps, rows[r], features[f], 255)
            _assert_covers_rows(rows[r], t.n_tiles, t.rows_per_tile)
            assert_true(t.n_tiles <= MAX_GRID_DIM_Y)


def test_block_threads_are_launchable() raises:
    # A warp multiple, at least one warp, never above the device maximum.
    var caps = [_small_gpu(), _large_gpu(), DeviceCaps(4, 128, 16384)]
    for i in range(len(caps)):
        var threads = derive_block_threads(caps[i])
        assert_equal(threads % WARP_GRANULARITY, 0)
        assert_true(threads >= WARP_GRANULARITY)
        assert_true(threads <= caps[i].max_threads_per_block)


def test_more_multiprocessors_buy_more_tiles() raises:
    """The whole point of deriving from capabilities: a wider device gets
    more threadgroups for the same shape, as long as the rows support it."""
    var small = derive_tiling(_small_gpu(), 1_000_000, 4, 255)
    var large = derive_tiling(_large_gpu(), 1_000_000, 4, 255)
    assert_true(large.n_tiles > small.n_tiles)
    assert_true(large.rows_per_tile < small.rows_per_tile)


def test_tiny_datasets_do_not_fragment() raises:
    """A tile has to scan enough rows to pay for its partial histogram, so
    a small dataset stays on one tile even on a wide device."""
    var t = derive_tiling(_large_gpu(), 512, 4, 255)
    assert_equal(t.n_tiles, 1)
    assert_equal(t.rows_per_tile, 512)


def test_partial_buffer_respects_an_explicit_capacity() raises:
    """Feature subsampling re-derives the tiling against an already
    allocated buffer, so the cap has to bind."""
    var caps = _large_gpu()
    var full = derive_tiling(caps, 1_000_000, 8, 255)
    assert_equal(full.strategy, STRATEGY_TILED)
    assert_true(full.partial_cells > 0)

    var capped = derive_tiling(
        caps, 1_000_000, 8, 255, STRATEGY_TILED, 4 * 8 * 255
    )
    assert_true(capped.n_tiles <= 4)
    assert_true(capped.partial_cells <= 4 * 8 * 255)
    _assert_covers_rows(1_000_000, capped.n_tiles, capped.rows_per_tile)


def test_auto_falls_back_to_atomics_when_partials_do_not_fit() raises:
    """The preserved fallback has to be reachable by policy, not only by
    hand: a histogram so wide that not even two tiles fit the budget gets
    the atomic path, which needs no partial buffer at all."""
    var caps = _large_gpu()
    var t = derive_tiling(caps, 1_000_000, 8_000_000, 255)
    assert_equal(t.strategy, STRATEGY_ATOMIC)
    assert_equal(t.partial_cells, 0)
    _assert_covers_rows(1_000_000, t.n_tiles, t.rows_per_tile)


def test_requested_strategy_is_honored() raises:
    var caps = _small_gpu()
    var atomic = derive_tiling(caps, 100_000, 10, 255, STRATEGY_ATOMIC)
    assert_equal(atomic.strategy, STRATEGY_ATOMIC)
    assert_equal(atomic.partial_cells, 0)

    var tiled = derive_tiling(caps, 100_000, 10, 255, STRATEGY_TILED)
    assert_equal(tiled.strategy, STRATEGY_TILED)
    assert_true(tiled.partial_cells > 0)
    assert_equal(tiled.partial_cells, tiled.n_tiles * 10 * 255)


def test_shape_and_shared_memory_are_validated() raises:
    var caps = _small_gpu()
    var raised = False
    try:
        _ = derive_tiling(caps, 0, 4, 255)
    except:
        raised = True
    assert_true(raised)

    raised = False
    try:
        _ = derive_tiling(caps, 100, 0, 255)
    except:
        raised = True
    assert_true(raised)

    # 255 bins need 255 * 12 bytes of shared memory per threadgroup, which
    # a device advertising 1 KiB cannot give.
    assert_true(shared_bytes_for(255) > 1024)
    raised = False
    try:
        _ = derive_tiling(DeviceCaps(8, 1024, 1024), 100_000, 4, 255)
    except:
        raised = True
    assert_true(raised)


def test_strategy_names_round_trip() raises:
    assert_equal(strategy_name(STRATEGY_ATOMIC), "atomic")
    assert_equal(strategy_name(STRATEGY_TILED), "tiled")
    assert_equal(strategy_name(STRATEGY_AUTO), "auto")


def test_env_overrides() raises:
    # One test owns all env mutation so no other test sees a dirty
    # environment regardless of suite ordering; empty string means unset.
    var caps = _small_gpu()

    _ = setenv("MOJOTREES_GPU_HIST_STRATEGY", "atomic")
    assert_equal(env_strategy(), STRATEGY_ATOMIC)
    assert_equal(
        derive_tiling(caps, 100_000, 10, 255).strategy, STRATEGY_ATOMIC
    )

    _ = setenv("MOJOTREES_GPU_HIST_STRATEGY", "tiled")
    assert_equal(env_strategy(), STRATEGY_TILED)
    assert_equal(
        derive_tiling(caps, 100_000, 10, 255).strategy, STRATEGY_TILED
    )

    # An explicit argument outranks the environment.
    assert_equal(
        derive_tiling(caps, 100_000, 10, 255, STRATEGY_ATOMIC).strategy,
        STRATEGY_ATOMIC,
    )

    _ = setenv("MOJOTREES_GPU_HIST_STRATEGY", "nonsense")
    assert_equal(env_strategy(), STRATEGY_AUTO)
    _ = setenv("MOJOTREES_GPU_HIST_STRATEGY", "")

    # A forced row tile still has to cover the rows and stay in budget.
    _ = setenv("MOJOTREES_GPU_ROW_TILE", "1000")
    var forced = derive_tiling(caps, 100_000, 4, 255)
    assert_equal(forced.rows_per_tile, 1000)
    assert_equal(forced.n_tiles, 100)
    _assert_covers_rows(100_000, forced.n_tiles, forced.rows_per_tile)
    _ = setenv("MOJOTREES_GPU_ROW_TILE", "")

    _ = setenv("MOJOTREES_GPU_BLOCK_THREADS", "128")
    assert_equal(derive_block_threads(caps), 128)
    # Still clamped to the device maximum and rounded to a warp.
    _ = setenv("MOJOTREES_GPU_BLOCK_THREADS", "100000")
    assert_equal(derive_block_threads(caps), 1024)
    _ = setenv("MOJOTREES_GPU_BLOCK_THREADS", "3")
    assert_equal(derive_block_threads(caps), WARP_GRANULARITY)
    _ = setenv("MOJOTREES_GPU_BLOCK_THREADS", "")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
