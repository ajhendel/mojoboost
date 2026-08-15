"""The row-tile floor in the GPU histogram tiling policy.

`gpu_tiling.resolve_tiling` used to spend the device-wide threadgroup target
across the features in `grid.x`, taking `ceil(target_blocks / n_slots)` row
tiles, which is a ceiling on total threadgroups wearing the name of an
occupancy target. `row_tile_floor` replaces it with a floor: the row
dimension alone should be able to fill the device, and the feature count no
longer spends the budget before the rows see any of it.

This is pure host arithmetic over a synthetic `DeviceCaps`, so nothing here
needs an accelerator and nothing here measures one. The assertions are the
geometry the policy claims, not a speed: what the tile count is at each of
the four corners of the (few, many) x (features, rows) square, that the
floor never pushes a tile below the amortization bound that `tiles_by_rows`
enforces, that it stays inside the partial-buffer budget and `grid.y`, that
the tiles still cover every row with no empty last tile, and that all four
`MOJOTREES_GPU_*` overrides still win.

tests/test_gpu_tiling.mojo owns the invariants the floor did not change and
is left alone; this file only covers what the floor added.
"""

from std.os import setenv
from std.testing import assert_equal, assert_raises, assert_true, TestSuite

from mojotrees.gpu_tiling import (
    MAX_GRID_DIM_Y,
    MIN_ROWS_PER_TILE_BIN_FACTOR,
    MIN_ROWS_PER_TILE_THREAD_FACTOR,
    PARTIAL_BUDGET_BYTES,
    BYTES_PER_PARTIAL_CELL,
    STRATEGY_ATOMIC,
    STRATEGY_TILED,
    TARGET_BLOCKS_PER_SM,
    DeviceCaps,
    blocks_per_sm_for,
    derive_block_threads,
    derive_tiling,
    env_min_tiles,
    row_tile_floor,
    target_blocks_for,
)


def _m4() -> DeviceCaps:
    """The one Apple triple this repository has read from hardware, as
    tests/test_gpu_tiling.mojo already pins it. Ten cores make
    `target_blocks` 80, which is small enough that every number in the
    corner table below can be checked by hand."""
    return DeviceCaps(10, 1024, 32768)


def _discrete() -> DeviceCaps:
    """An A100-shaped device: 108 multiprocessors, so `target_blocks` is
    864 and the row bound is what binds on almost every shape."""
    return DeviceCaps(108, 1024, 49152)


def _ceil_div(a: Int, b: Int) -> Int:
    return (a + b - 1) // b


def _min_rows_per_tile(n_bins: Int, block_threads: Int) -> Int:
    """The amortization bound, re-derived from the exported constants rather
    than copied as a number, so a change to either factor moves this test
    with it."""
    var by_bins = MIN_ROWS_PER_TILE_BIN_FACTOR * n_bins
    var by_threads = MIN_ROWS_PER_TILE_THREAD_FACTOR * block_threads
    if by_threads > by_bins:
        return by_threads
    return by_bins


def _tiles_by_rows(caps: DeviceCaps, n_rows: Int, n_bins: Int) raises -> Int:
    var tiles = _ceil_div(
        n_rows, _min_rows_per_tile(n_bins, derive_block_threads(caps))
    )
    if tiles < 1:
        return 1
    return tiles


def _previous_tile_count(
    caps: DeviceCaps, n_rows: Int, n_features: Int, n_bins: Int
) raises -> Int:
    """What the rule chose before the floor: the device-wide target divided
    among the features, clamped by the row bound. Kept here so the corner
    table states the change rather than only the new value."""
    var by_occupancy = _ceil_div(
        caps.sm_count * TARGET_BLOCKS_PER_SM, n_features
    )
    if by_occupancy < 1:
        by_occupancy = 1
    var by_rows = _tiles_by_rows(caps, n_rows, n_bins)
    if by_rows < by_occupancy:
        return by_rows
    return by_occupancy


def test_the_m4_million_row_root_fills_the_device() raises:
    """The shape end-to-end GPU training is slowest on, asserted exactly.

    Fifty features and a million-row root on the M4: the old rule spent the
    80-threadgroup target across 50 features, took 2 row tiles, and launched
    100 threadgroups each scanning 500,000 rows. The floor asks the row
    dimension for a full device on its own.
    """
    var caps = _m4()
    var t = derive_tiling(caps, 1_000_000, 50, 256)

    assert_equal(_previous_tile_count(caps, 1_000_000, 50, 256), 2)
    assert_equal(t.n_tiles, 80)
    assert_equal(t.rows_per_tile, 12_500)
    assert_equal(t.n_tiles * 50, 4_000)
    assert_true(t.n_tiles > _previous_tile_count(caps, 1_000_000, 50, 256))

    # The row bound is nowhere near binding here, which is what made the old
    # value a policy choice rather than a physical limit.
    assert_equal(_tiles_by_rows(caps, 1_000_000, 256), 489)
    assert_true(t.n_tiles < _tiles_by_rows(caps, 1_000_000, 256))


def test_the_four_corners() raises:
    """Few or many features against few or many rows, on the M4, at 256
    bins. This table is the argument that the rule is right: it may only
    raise the tile count, and only where the rows leave room."""
    var caps = _m4()
    # features, rows, tiles before, tiles after
    var corners = [
        4, 8_192, 4, 4,
        4, 1_000_000, 20, 80,
        100, 8_192, 1, 4,
        100, 1_000_000, 1, 80,
    ]
    var i = 0
    while i + 3 < len(corners):
        var features = corners[i]
        var rows = corners[i + 1]
        var t = derive_tiling(caps, rows, features, 256)
        assert_equal(
            _previous_tile_count(caps, rows, features, 256), corners[i + 2]
        )
        assert_equal(t.n_tiles, corners[i + 3])
        assert_true(t.n_tiles >= corners[i + 2])
        i += 4

    # Few features and few rows is the corner that must not move: the row
    # bound is what set it, and the floor never pushes past the row bound.
    assert_equal(
        derive_tiling(caps, 8_192, 4, 256).n_tiles,
        _tiles_by_rows(caps, 8_192, 256),
    )


def test_more_features_than_the_block_target_no_longer_collapse() raises:
    """The pathological case: at or above 80 features on a 10-core device the
    old occupancy term was `ceil(80 / n_features)`, which is 1 for every one
    of these, so a node got a single tile whatever its row count."""
    var caps = _m4()
    var features = [80, 100, 200, 500]
    for f in range(len(features)):
        assert_equal(
            _previous_tile_count(caps, 1_000_000, features[f], 256), 1
        )
        var t = derive_tiling(caps, 1_000_000, features[f], 256)
        assert_true(t.n_tiles > 1)
        assert_true(t.rows_per_tile < 1_000_000)

    # 80 features leave the partial buffer plenty of room, so the floor is
    # reached exactly; 500 features do not, and the memory bound clamps the
    # floor rather than the floor overriding it.
    assert_equal(derive_tiling(caps, 1_000_000, 80, 256).n_tiles, 80)
    var wide = derive_tiling(caps, 1_000_000, 500, 256)
    var cell_limit = PARTIAL_BUDGET_BYTES // BYTES_PER_PARTIAL_CELL
    assert_true(wide.n_tiles < 80)
    assert_true(wide.partial_cells <= cell_limit)


def test_the_floor_never_shortens_a_tile_past_amortization() raises:
    """A tile below the amortization threshold cannot pay for writing or
    folding its own partial histogram. The floor raises the tile count toward
    `tiles_by_rows` and never past it, so that bound holds unchanged, and
    `rows_per_tile` is never shorter than the bound alone would have made
    it."""
    var devices = [_m4(), _discrete(), DeviceCaps(1, 1024, 16384)]
    var rows = [1, 2, 999, 8_192, 100_000, 1_000_000, 50_000_000]
    var features = [1, 4, 50, 100, 1_000]
    var bins = [16, 64, 255]

    for d in range(len(devices)):
        var caps = devices[d].copy()
        for r in range(len(rows)):
            for f in range(len(features)):
                for b in range(len(bins)):
                    var t = derive_tiling(caps, rows[r], features[f], bins[b])
                    var by_rows = _tiles_by_rows(caps, rows[r], bins[b])
                    assert_true(
                        t.n_tiles <= by_rows,
                        "the floor pushed the tile count past the"
                        " amortization bound",
                    )
                    assert_true(
                        t.rows_per_tile >= _ceil_div(rows[r], by_rows),
                        "the floor produced a tile shorter than the"
                        " amortization bound alone would have",
                    )


def test_the_floor_stays_inside_the_partial_budget_and_the_grid() raises:
    """The floor is a request; the partial buffer and `grid.y` are limits.

    A caller that hands in an already allocated buffer is the case that
    matters: feature subsampling re-derives the tiling for a narrower
    `grid.x` without reallocating, and a floor that ignored the capacity it
    was given would plan a launch that cannot be backed."""
    var caps = _discrete()
    var capacity = 4 * 8 * 255
    var capped = derive_tiling(
        caps, 1_000_000, 8, 255, STRATEGY_TILED, capacity
    )
    assert_true(capped.n_tiles <= 4)
    assert_true(capped.partial_cells <= capacity)
    # Uncapped, the same shape reaches much further, so the cap is what bound
    # it rather than the floor happening to be small.
    assert_true(derive_tiling(caps, 1_000_000, 8, 255).n_tiles > 4)

    # A device claiming an implausible number of multiprocessors puts the
    # floor past `grid.y`. The atomic strategy allocates no partial buffer,
    # so the grid bound is the only one left to catch it.
    var vast = DeviceCaps(100_000, 1024, 32768)
    assert_true(target_blocks_for(vast) > MAX_GRID_DIM_Y)
    var t = derive_tiling(vast, 2_000_000_000, 1, 256, STRATEGY_ATOMIC)
    assert_true(t.n_tiles <= MAX_GRID_DIM_Y)
    assert_equal(t.partial_cells, 0)
    assert_true(row_tile_floor(target_blocks_for(vast), 1) <= MAX_GRID_DIM_Y)


def test_tiles_still_cover_every_row() raises:
    """The re-derivation after the clamps guarantees it: `rows_per_tile`
    comes from the final tile count and the count then comes back from the
    rows, so the last tile is never empty. More tiles is exactly the
    direction that would expose a gap, which is why it is asserted again
    here rather than left to the tiling suite."""
    var devices = [_m4(), _discrete()]
    var rows = [1, 3, 2_047, 8_192, 100_000, 1_000_000, 12_345_678]
    var features = [1, 7, 50, 300]

    for d in range(len(devices)):
        var caps = devices[d].copy()
        for r in range(len(rows)):
            for f in range(len(features)):
                var t = derive_tiling(caps, rows[r], features[f], 255)
                assert_true(t.n_tiles >= 1)
                assert_true(t.rows_per_tile >= 1)
                assert_true(t.n_tiles * t.rows_per_tile >= rows[r])
                assert_true((t.n_tiles - 1) * t.rows_per_tile < rows[r])


def test_the_block_target_is_derivable_from_a_footprint() raises:
    """`TARGET_BLOCKS_PER_SM` stays 8, as the default and as the ceiling.

    A caller with no footprint to offer gets the fixed target. A caller that
    knows its kernel's threadgroup footprint can lower it, because shared
    memory can prove that this many blocks do not fit. It cannot raise it:
    thread slots and the register file bound residency too and `DeviceCaps`
    reports neither, so a small footprint is not evidence of more residency.

    That is the contract with a kernel sized to its bin count rather than to
    `MAX_BINS`: a 32-bin histogram occupying 384 bytes changes nothing here
    until an occupancy measurement justifies raising the ceiling.
    """
    assert_equal(blocks_per_sm_for(32768, 0), TARGET_BLOCKS_PER_SM)
    assert_equal(blocks_per_sm_for(32768, 3072), TARGET_BLOCKS_PER_SM)
    assert_equal(blocks_per_sm_for(32768, 384), TARGET_BLOCKS_PER_SM)
    assert_equal(blocks_per_sm_for(32768, 8192), 4)
    assert_equal(blocks_per_sm_for(32768, 16384), 2)
    assert_equal(blocks_per_sm_for(32768, 65536), 1)

    var caps = _m4()
    assert_equal(target_blocks_for(caps), 80)
    assert_equal(target_blocks_for(caps, 384), 80)
    assert_equal(target_blocks_for(caps, 8192), 40)

    # A footprint that halves the target halves the floor, and a bin-sized
    # one leaves the geometry exactly where it was.
    assert_equal(derive_tiling(caps, 1_000_000, 50, 256).n_tiles, 80)
    var halved = derive_tiling(
        caps, 1_000_000, 50, 256, STRATEGY_TILED, 0, 8192
    )
    assert_equal(halved.n_tiles, 40)
    var narrow = derive_tiling(
        caps, 1_000_000, 50, 32, STRATEGY_TILED, 0, 384
    )
    var unchanged = derive_tiling(caps, 1_000_000, 50, 32, STRATEGY_TILED)
    assert_equal(narrow.n_tiles, unchanged.n_tiles)


def test_row_tile_floor_refuses_impossible_bounds() raises:
    """`resolve_tiling` validates the same two quantities, so the floor
    refusing them keeps a caller that computed a bound wrongly from reaching
    an empty grid by a different route."""
    assert_equal(row_tile_floor(80, 50), 80)
    with assert_raises():
        _ = row_tile_floor(0, 50)
    with assert_raises():
        _ = row_tile_floor(80, 0)


def test_env_overrides() raises:
    """One test owns all environment mutation, as in
    tests/test_gpu_tiling.mojo, so no other test sees a dirty environment
    whatever order the suite runs in. Empty string means unset."""
    var caps = _m4()

    # Unset is zero, which is the derived floor.
    assert_equal(env_min_tiles(), 0)

    # A floor supplied by hand replaces the derived one, downward.
    _ = setenv("MOJOTREES_GPU_MIN_TILES", "4")
    assert_equal(env_min_tiles(), 4)
    var few = derive_tiling(caps, 1_000_000, 50, 256)
    assert_equal(few.n_tiles, 4)
    assert_equal(few.rows_per_tile, 250_000)

    # `1` restores the tile count the rule chose before the floor existed,
    # because the old occupancy term survives underneath as a second floor.
    _ = setenv("MOJOTREES_GPU_MIN_TILES", "1")
    assert_equal(
        derive_tiling(caps, 1_000_000, 50, 256).n_tiles,
        _previous_tile_count(caps, 1_000_000, 50, 256),
    )
    assert_equal(derive_tiling(caps, 1_000_000, 50, 256).n_tiles, 2)

    # Upward it is still a request: the row bound clamps it here.
    _ = setenv("MOJOTREES_GPU_MIN_TILES", "1000000")
    var by_rows = derive_tiling(caps, 1_000_000, 4, 256)
    assert_equal(by_rows.n_tiles, _tiles_by_rows(caps, 1_000_000, 256))

    # And the partial buffer clamps it on a wider histogram, where the row
    # bound would have allowed more.
    var by_memory = derive_tiling(caps, 1_000_000, 50, 256)
    var cell_limit = PARTIAL_BUDGET_BYTES // BYTES_PER_PARTIAL_CELL
    assert_true(by_memory.n_tiles < _tiles_by_rows(caps, 1_000_000, 256))
    assert_true(by_memory.partial_cells <= cell_limit)
    assert_true(by_memory.n_tiles * 50 * 256 <= cell_limit)

    # An explicit row tile outranks the floor: it names the tile length, and
    # the floor only names how many the policy wanted.
    _ = setenv("MOJOTREES_GPU_MIN_TILES", "8")
    _ = setenv("MOJOTREES_GPU_ROW_TILE", "1000")
    var forced = derive_tiling(caps, 100_000, 4, 255)
    assert_equal(forced.rows_per_tile, 1000)
    assert_equal(forced.n_tiles, 100)
    _ = setenv("MOJOTREES_GPU_ROW_TILE", "")
    _ = setenv("MOJOTREES_GPU_MIN_TILES", "")
    assert_equal(env_min_tiles(), 0)

    # A narrower threadgroup lowers the amortization bound, so the floor can
    # reach further before the rows stop it.
    _ = setenv("MOJOTREES_GPU_BLOCK_THREADS", "128")
    assert_equal(derive_block_threads(caps), 128)
    assert_equal(derive_tiling(caps, 1_000_000, 50, 256).block_threads, 128)
    _ = setenv("MOJOTREES_GPU_BLOCK_THREADS", "")

    # The strategy override still decides which path a floored tile count
    # runs on, and the atomic path still allocates nothing.
    _ = setenv("MOJOTREES_GPU_HIST_STRATEGY", "atomic")
    var atomic = derive_tiling(caps, 1_000_000, 50, 256)
    assert_equal(atomic.strategy, STRATEGY_ATOMIC)
    assert_equal(atomic.partial_cells, 0)
    assert_equal(atomic.n_tiles, 80)
    _ = setenv("MOJOTREES_GPU_HIST_STRATEGY", "")

    assert_equal(derive_tiling(caps, 1_000_000, 50, 256).n_tiles, 80)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
