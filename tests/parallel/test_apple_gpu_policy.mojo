"""Apple-silicon tuning policy.

`derive_policy` is pure host-side arithmetic over a `GpuProfile`, so every
decision it makes is checkable here on any machine, with or without an
accelerator, and on a machine that has none of the Apple parts involved.

What this file is actually guarding, beyond the structural invariants the
tiling suite already covers for `derive_tiling`:

- The plan tracks what the device reported. A different core count, a
  different threadgroup memory, or a different memory budget has to produce
  a different plan, because the entire justification for this layer is that
  it does not carry one chip's numbers to another chip.
- The developer's M4 is not universal. `apple_m4_observed()` is a named
  fixture; nothing falls back to it, and a device reporting other numbers
  plans from those.
- The synthetic fixtures stay labeled. They are conservative
  specifications, not measurements, and the flag that says so has to be set
  on all five.
- No crossover threshold is invented. `min_cells` is disabled everywhere,
  matching device.mojo, until a benchmark says otherwise.
- The portable geometry names this layer re-exports are gpu_tiling.mojo's
  own, so the two layers cannot disagree about a warp, a budget, or a
  strategy code.
"""

from std.testing import assert_equal, assert_false, assert_true, TestSuite

from mojotrees.apple_gpu_policy import (
    API_CUDA,
    API_HIP,
    API_METAL,
    API_UNKNOWN,
    APPLE_GEN_M1,
    APPLE_GEN_M2,
    APPLE_GEN_M3,
    APPLE_GEN_M4,
    APPLE_GEN_M5,
    APPLE_GEN_UNKNOWN,
    BYTES_PER_PARTIAL_CELL,
    CROSSOVER_DISABLED,
    FALLBACK_CORE_COUNT,
    FALLBACK_MAX_THREADS_PER_BLOCK,
    FALLBACK_SHARED_MEMORY_PER_BLOCK,
    MAX_GRID_DIM_Y,
    MAX_RESIDENT_BLOCKS_PER_CORE,
    MIN_ROWS_PER_TILE_BIN_FACTOR,
    MIN_ROWS_PER_TILE_THREAD_FACTOR,
    PARTIAL_BUDGET_CEILING_BYTES,
    STRATEGY_ATOMIC,
    STRATEGY_AUTO,
    STRATEGY_TILED,
    TARGET_BLOCK_THREADS,
    WARP_GRANULARITY,
    GpuProfile,
    apple_generation_name,
    apple_m4_observed,
    apple_synthetic,
    api_name,
    derive_policy,
    describe_policy,
    parse_api,
    parse_apple_generation,
    partial_budget_bytes,
    resident_blocks_per_core,
    shape_block_threads,
    shared_bytes_for_bins,
    strategy_name,
    synthetic_apple_core_count,
)

# The source of every re-exported constant. Aliased because both modules
# spell these the same way.
from mojotrees.gpu_tiling import (
    BYTES_PER_PARTIAL_CELL as TILING_BYTES_PER_PARTIAL_CELL,
    FALLBACK_MAX_THREADS_PER_BLOCK as TILING_FALLBACK_MAX_THREADS,
    FALLBACK_SHARED_MEMORY_PER_BLOCK as TILING_FALLBACK_SHARED,
    FALLBACK_SM_COUNT as TILING_FALLBACK_SM_COUNT,
    MAX_GRID_DIM_Y as TILING_MAX_GRID_DIM_Y,
    MIN_ROWS_PER_TILE_BIN_FACTOR as TILING_MIN_ROWS_BIN_FACTOR,
    MIN_ROWS_PER_TILE_THREAD_FACTOR as TILING_MIN_ROWS_THREAD_FACTOR,
    PARTIAL_BUDGET_BYTES as TILING_PARTIAL_BUDGET_BYTES,
    STRATEGY_ATOMIC as TILING_STRATEGY_ATOMIC,
    STRATEGY_AUTO as TILING_STRATEGY_AUTO,
    STRATEGY_TILED as TILING_STRATEGY_TILED,
    TARGET_BLOCKS_PER_SM as TILING_TARGET_BLOCKS_PER_SM,
    TARGET_BLOCK_THREADS as TILING_TARGET_BLOCK_THREADS,
    WARP_GRANULARITY as TILING_WARP_GRANULARITY,
    shared_bytes_for as tiling_shared_bytes_for,
)


def _discrete(core_count: Int, budget_bytes: Int = 0) -> GpuProfile:
    """A discrete-class device: separate memory, large core count."""
    return GpuProfile(
        API_CUDA,
        APPLE_GEN_UNKNOWN,
        core_count,
        1024,
        49152,
        budget_bytes,
        False,
        True,
    )


def _unified(core_count: Int, budget_bytes: Int = 0) -> GpuProfile:
    """The same device with unified memory, so a test can vary that alone."""
    return GpuProfile(
        API_METAL,
        APPLE_GEN_UNKNOWN,
        core_count,
        1024,
        49152,
        budget_bytes,
        True,
        True,
    )


def _assert_covers_rows(n_rows: Int, n_tiles: Int, rows_per_tile: Int) raises:
    """Tiles must cover every row, and the last tile must not be empty."""
    assert_true(n_tiles >= 1)
    assert_true(rows_per_tile >= 1)
    assert_true(n_tiles * rows_per_tile >= n_rows)
    assert_true((n_tiles - 1) * rows_per_tile < n_rows)


def test_mirrored_constants_match_gpu_tiling() raises:
    """These names were once copies pinned equal to their source; they are
    now re-exports of gpu_tiling.mojo, and this asserts the re-export path
    resolves to the same values under this layer's spellings."""
    assert_equal(STRATEGY_AUTO, TILING_STRATEGY_AUTO)
    assert_equal(STRATEGY_ATOMIC, TILING_STRATEGY_ATOMIC)
    assert_equal(STRATEGY_TILED, TILING_STRATEGY_TILED)
    assert_equal(TARGET_BLOCK_THREADS, TILING_TARGET_BLOCK_THREADS)
    assert_equal(WARP_GRANULARITY, TILING_WARP_GRANULARITY)
    assert_equal(BYTES_PER_PARTIAL_CELL, TILING_BYTES_PER_PARTIAL_CELL)
    assert_equal(MAX_GRID_DIM_Y, TILING_MAX_GRID_DIM_Y)
    assert_equal(PARTIAL_BUDGET_CEILING_BYTES, TILING_PARTIAL_BUDGET_BYTES)
    assert_equal(FALLBACK_CORE_COUNT, TILING_FALLBACK_SM_COUNT)
    assert_equal(FALLBACK_MAX_THREADS_PER_BLOCK, TILING_FALLBACK_MAX_THREADS)
    assert_equal(FALLBACK_SHARED_MEMORY_PER_BLOCK, TILING_FALLBACK_SHARED)
    assert_equal(MIN_ROWS_PER_TILE_BIN_FACTOR, TILING_MIN_ROWS_BIN_FACTOR)
    assert_equal(
        MIN_ROWS_PER_TILE_THREAD_FACTOR, TILING_MIN_ROWS_THREAD_FACTOR
    )
    # The residency ceiling is the tiling module's fixed target, so a device
    # whose threadgroup memory fits that many partials plans as it does now.
    assert_equal(MAX_RESIDENT_BLOCKS_PER_CORE, TILING_TARGET_BLOCKS_PER_SM)

    var bins = [1, 16, 63, 255, 1024]
    for i in range(len(bins)):
        assert_equal(
            shared_bytes_for_bins(bins[i]), tiling_shared_bytes_for(bins[i])
        )


def test_api_parsing_tolerates_reported_spellings() raises:
    """What a backend answers is not a normalized identifier."""
    assert_equal(parse_api("metal"), API_METAL)
    assert_equal(parse_api("Metal"), API_METAL)
    assert_equal(parse_api("METAL"), API_METAL)
    assert_equal(parse_api("cuda"), API_CUDA)
    assert_equal(parse_api("CUDA"), API_CUDA)
    assert_equal(parse_api("hip"), API_HIP)
    assert_equal(parse_api("ROCm"), API_HIP)
    # Unrecognized is a portable path, not an error.
    assert_equal(parse_api(""), API_UNKNOWN)
    assert_equal(parse_api("levelzero"), API_UNKNOWN)

    assert_equal(api_name(API_METAL), "metal")
    assert_equal(api_name(API_CUDA), "cuda")
    assert_equal(api_name(API_HIP), "hip")
    assert_equal(api_name(API_UNKNOWN), "unknown")


def test_apple_generation_parsing() raises:
    assert_equal(parse_apple_generation("apple-m1"), APPLE_GEN_M1)
    assert_equal(parse_apple_generation("Apple M2"), APPLE_GEN_M2)
    assert_equal(parse_apple_generation("apple-m3-max"), APPLE_GEN_M3)
    assert_equal(parse_apple_generation("apple-m4"), APPLE_GEN_M4)
    assert_equal(parse_apple_generation("apple-m5"), APPLE_GEN_M5)
    # Nothing to read, and nothing invented from a string naming two.
    assert_equal(parse_apple_generation(""), APPLE_GEN_UNKNOWN)
    assert_equal(parse_apple_generation("apple-gpu"), APPLE_GEN_UNKNOWN)
    assert_equal(parse_apple_generation("m2-and-m3"), APPLE_GEN_UNKNOWN)

    assert_equal(apple_generation_name(APPLE_GEN_M4), "m4")
    assert_equal(apple_generation_name(APPLE_GEN_UNKNOWN), "unknown")
    assert_equal(apple_generation_name(99), "unknown")


def test_non_metal_architectures_are_not_mined_for_a_generation() raises:
    """A CUDA architecture string is not an Apple part number, however much
    of one it happens to contain."""
    var cuda = GpuProfile.from_reported("cuda", "sm_90", 108, 1024, 49152)
    assert_equal(cuda.api, API_CUDA)
    assert_equal(cuda.apple_generation, APPLE_GEN_UNKNOWN)
    assert_false(cuda.unified_memory)
    assert_false(cuda.is_apple())

    var metal = GpuProfile.from_reported("metal", "apple-m3", 10, 1024, 32768)
    assert_equal(metal.apple_generation, APPLE_GEN_M3)
    assert_true(metal.unified_memory)
    assert_true(metal.is_apple())


def test_reported_attributes_are_sanitized() raises:
    """A backend that answers nothing has to degrade the plan, not produce
    one the device cannot run."""
    var blank = GpuProfile.from_reported("", "", 0, 0, 0, 0)
    assert_equal(blank.core_count, FALLBACK_CORE_COUNT)
    assert_equal(blank.max_threads_per_block, FALLBACK_MAX_THREADS_PER_BLOCK)
    assert_equal(
        blank.max_shared_memory_per_block, FALLBACK_SHARED_MEMORY_PER_BLOCK
    )
    assert_equal(blank.memory_budget_bytes, 0)
    assert_false(blank.synthetic)

    var bad = GpuProfile.from_reported("metal", "apple-m4", -4, -1, -1, -1)
    assert_equal(bad.core_count, FALLBACK_CORE_COUNT)
    assert_equal(bad.max_threads_per_block, FALLBACK_MAX_THREADS_PER_BLOCK)
    assert_equal(bad.memory_budget_bytes, 0)


def test_generic_fallback_is_portable_and_not_apple_shaped() raises:
    """The Metal/CUDA/HIP fallback: it has to plan, and it must not be a
    disguised Apple profile."""
    var generic = GpuProfile.generic()
    assert_equal(generic.api, API_UNKNOWN)
    assert_equal(generic.apple_generation, APPLE_GEN_UNKNOWN)
    assert_false(generic.unified_memory)
    assert_false(generic.is_apple())
    assert_true(generic.synthetic)
    assert_equal(generic.core_count, FALLBACK_CORE_COUNT)

    var policy = derive_policy(generic, 1_000_000, 32, 255)
    _assert_covers_rows(1_000_000, policy.n_tiles, policy.rows_per_tile)
    assert_true(policy.block_threads <= generic.max_threads_per_block)
    assert_equal(policy.block_threads % WARP_GRANULARITY, 0)


def test_synthetic_fixtures_are_labeled_and_planable() raises:
    """Every M1-M5 fixture is synthetic, conservative, and produces a plan
    the device it describes could actually run."""
    for generation in range(APPLE_GEN_M1, APPLE_GEN_M5 + 1):
        var profile = apple_synthetic(generation)
        assert_true(profile.synthetic)
        assert_true(profile.is_apple())
        assert_true(profile.unified_memory)
        assert_equal(profile.apple_generation, generation)
        assert_equal(
            profile.core_count, synthetic_apple_core_count(generation)
        )
        # Conservative: the portable Apple floor, never the M4's 32 KiB.
        assert_equal(
            profile.max_shared_memory_per_block,
            FALLBACK_SHARED_MEMORY_PER_BLOCK,
        )
        # No budget has been read on any of them.
        assert_equal(profile.memory_budget_bytes, 0)

        var policy = derive_policy(profile, 500_000, 16, 255)
        _assert_covers_rows(500_000, policy.n_tiles, policy.rows_per_tile)
        assert_true(policy.block_threads <= profile.max_threads_per_block)
        assert_true(
            shared_bytes_for_bins(255) <= profile.max_shared_memory_per_block
        )

    var raised = False
    try:
        _ = apple_synthetic(APPLE_GEN_M5 + 1)
    except:
        raised = True
    assert_true(raised)

    raised = False
    try:
        _ = apple_synthetic(APPLE_GEN_UNKNOWN)
    except:
        raised = True
    assert_true(raised)


def test_the_observed_m4_is_a_fixture_not_a_default() raises:
    """The one Apple reading in the repository is named, flagged as a
    reading, and inherited by nothing."""
    var m4 = apple_m4_observed()
    assert_false(m4.synthetic)
    assert_equal(m4.apple_generation, APPLE_GEN_M4)
    assert_equal(m4.core_count, 10)
    assert_equal(m4.max_shared_memory_per_block, 32768)

    # Nothing else carries its numbers: not the portable fallback, and not
    # the synthetic fixture for its own generation.
    assert_true(GpuProfile.generic().core_count != m4.core_count)
    assert_true(
        GpuProfile.generic().max_shared_memory_per_block
        != m4.max_shared_memory_per_block
    )
    var synthetic_m4 = apple_synthetic(APPLE_GEN_M4)
    assert_true(
        synthetic_m4.max_shared_memory_per_block
        != m4.max_shared_memory_per_block
    )

    # And a device that reports other numbers plans from those, not from it.
    var other = GpuProfile.from_reported("metal", "apple-m1", 7, 1024, 16384)
    var m4_policy = derive_policy(m4, 2_000_000, 4, 255)
    var other_policy = derive_policy(other, 2_000_000, 4, 255)
    assert_true(other_policy.n_tiles != m4_policy.n_tiles)


def test_plans_track_the_reported_core_count() raises:
    """The whole point of deriving from what the device says: a wider device
    gets more threadgroups for the same shape."""
    var small = derive_policy(_discrete(10), 4_000_000, 4, 255)
    var large = derive_policy(_discrete(108), 4_000_000, 4, 255)
    assert_true(large.n_tiles > small.n_tiles)
    assert_true(large.rows_per_tile < small.rows_per_tile)
    _assert_covers_rows(4_000_000, small.n_tiles, small.rows_per_tile)
    _assert_covers_rows(4_000_000, large.n_tiles, large.rows_per_tile)


def test_residency_comes_from_reported_threadgroup_memory() raises:
    """The Apple-shaped part of the occupancy target: how many partial
    histograms the advertised threadgroup memory actually holds."""
    # 255 bins need 3060 bytes, so 16 KiB holds five and 48 KiB holds the
    # capped maximum.
    var narrow = GpuProfile.from_reported("metal", "apple-m2", 8, 1024, 16384)
    var wide = GpuProfile.from_reported("cuda", "sm_80", 8, 1024, 49152)
    assert_equal(resident_blocks_per_core(narrow, 255), 5)
    assert_equal(
        resident_blocks_per_core(wide, 255), MAX_RESIDENT_BLOCKS_PER_CORE
    )

    # A wider histogram lowers residency on the same device; one that only
    # just fits still gets a block rather than zero.
    assert_true(
        resident_blocks_per_core(narrow, 1024)
        < resident_blocks_per_core(narrow, 255)
    )
    assert_equal(resident_blocks_per_core(narrow, 1365), 1)

    # And residency reaches the plan: same cores, same shape, more room.
    var narrow_policy = derive_policy(narrow, 8_000_000, 4, 255)
    var wide_policy = derive_policy(wide, 8_000_000, 4, 255)
    assert_equal(narrow_policy.resident_blocks_per_core, 5)
    assert_equal(
        wide_policy.resident_blocks_per_core, MAX_RESIDENT_BLOCKS_PER_CORE
    )
    assert_true(wide_policy.n_tiles > narrow_policy.n_tiles)


def test_block_threads_are_launchable_and_shape_aware() raises:
    """A warp multiple, at least one warp, never above the device maximum,
    and never wider than the rows can feed."""
    var profiles = [
        GpuProfile.generic(),
        apple_m4_observed(),
        _discrete(108),
        GpuProfile.from_reported("metal", "apple-m1", 7, 128, 16384),
    ]
    for i in range(len(profiles)):
        var threads = shape_block_threads(profiles[i], 1_000_000)
        assert_equal(threads % WARP_GRANULARITY, 0)
        assert_true(threads >= WARP_GRANULARITY)
        assert_true(threads <= profiles[i].max_threads_per_block)

    # Narrow datasets do not launch a block whose lanes have no rows: 100
    # rows rounds down to one warp, not to the 256-thread target.
    var m4 = apple_m4_observed()
    assert_equal(shape_block_threads(m4, 1_000_000), TARGET_BLOCK_THREADS)
    assert_equal(shape_block_threads(m4, 100), WARP_GRANULARITY)
    assert_equal(shape_block_threads(m4, 1), WARP_GRANULARITY)
    assert_equal(shape_block_threads(m4, 200), 192)


def test_unified_memory_gets_a_tighter_partial_budget() raises:
    """The budget on Apple is system RAM, shared with the dataset the host
    is holding, so the same reported number buys a smaller buffer."""
    var budget = 4 << 20
    assert_true(
        partial_budget_bytes(_unified(108, budget))
        < partial_budget_bytes(_discrete(108, budget))
    )
    # Unreported means the portable ceiling, which is what the tiling module
    # uses unconditionally today.
    assert_equal(
        partial_budget_bytes(_discrete(108, 0)), PARTIAL_BUDGET_CEILING_BYTES
    )
    assert_equal(
        partial_budget_bytes(_unified(108, 0)), PARTIAL_BUDGET_CEILING_BYTES
    )
    # A budget large enough to exceed the ceiling is still capped by it.
    assert_equal(
        partial_budget_bytes(_discrete(108, 64 << 30)),
        PARTIAL_BUDGET_CEILING_BYTES,
    )

    # And the difference reaches the plan for a shape the budget binds.
    var one = derive_policy(_unified(108, budget), 10_000_000, 4, 255)
    var two = derive_policy(_discrete(108, budget), 10_000_000, 4, 255)
    assert_true(one.partial_cell_limit < two.partial_cell_limit)
    assert_true(one.n_tiles < two.n_tiles)
    _assert_covers_rows(10_000_000, one.n_tiles, one.rows_per_tile)
    _assert_covers_rows(10_000_000, two.n_tiles, two.rows_per_tile)


def test_partial_buffer_stays_inside_its_limit() raises:
    """Whatever the shape, an auto-resolved plan never asks for more partial
    cells than the budget allowed."""
    var profiles = [
        GpuProfile.generic(),
        apple_m4_observed(),
        _unified(10, 8 << 20),
        _discrete(108, 256 << 20),
    ]
    var rows = [1, 1000, 250_000, 20_000_000]
    var features = [1, 8, 300]
    for p in range(len(profiles)):
        for r in range(len(rows)):
            for f in range(len(features)):
                var policy = derive_policy(
                    profiles[p], rows[r], features[f], 255
                )
                _assert_covers_rows(
                    rows[r], policy.n_tiles, policy.rows_per_tile
                )
                assert_true(policy.n_tiles <= MAX_GRID_DIM_Y)
                assert_true(
                    policy.partial_cells <= policy.partial_cell_limit
                )
                if policy.strategy == STRATEGY_TILED:
                    assert_equal(
                        policy.partial_cells,
                        policy.n_tiles * features[f] * 255,
                    )
                else:
                    assert_equal(policy.partial_cells, 0)


def test_strategy_choice_matches_the_tiling_rule() raises:
    """Deliberately no Apple divergence here: nothing has measured atomic
    throughput against the tiled reduction on any Apple part."""
    # Nothing to reduce at one tile, so the preserved atomic path wins.
    var tiny = derive_policy(apple_m4_observed(), 512, 4, 255)
    assert_equal(tiny.n_tiles, 1)
    assert_equal(tiny.rows_per_tile, 512)
    assert_equal(tiny.strategy, STRATEGY_ATOMIC)
    assert_equal(tiny.partial_cells, 0)

    var big = derive_policy(apple_m4_observed(), 5_000_000, 4, 255)
    assert_true(big.n_tiles > 1)
    assert_equal(big.strategy, STRATEGY_TILED)
    assert_true(big.partial_cells > 0)

    # A histogram too wide for the budget to hold two tiles falls back to
    # atomics, which allocate nothing.
    var wide = derive_policy(_discrete(108), 5_000_000, 8_000_000, 255)
    assert_equal(wide.strategy, STRATEGY_ATOMIC)
    assert_equal(wide.partial_cells, 0)
    _assert_covers_rows(5_000_000, wide.n_tiles, wide.rows_per_tile)


def test_requested_strategy_is_honored() raises:
    var profile = apple_m4_observed()
    var atomic = derive_policy(profile, 2_000_000, 10, 255, STRATEGY_ATOMIC)
    assert_equal(atomic.strategy, STRATEGY_ATOMIC)
    assert_equal(atomic.partial_cells, 0)

    var tiled = derive_policy(profile, 2_000_000, 10, 255, STRATEGY_TILED)
    assert_equal(tiled.strategy, STRATEGY_TILED)
    assert_equal(tiled.partial_cells, tiled.n_tiles * 10 * 255)

    assert_equal(strategy_name(STRATEGY_ATOMIC), "atomic")
    assert_equal(strategy_name(STRATEGY_TILED), "tiled")
    assert_equal(strategy_name(STRATEGY_AUTO), "auto")


def test_shape_and_shared_memory_are_validated() raises:
    var profile = apple_m4_observed()
    var shapes_rows = [0, 100, 100, -1]
    var shapes_features = [4, 0, 4, 4]
    var shapes_bins = [255, 255, 0, 255]
    for i in range(len(shapes_rows)):
        var raised = False
        try:
            _ = derive_policy(
                profile, shapes_rows[i], shapes_features[i], shapes_bins[i]
            )
        except:
            raised = True
        assert_true(raised)

    # 255 bins need more than 1 KiB of threadgroup memory, so a device
    # advertising that cannot run the kernel at all.
    assert_true(shared_bytes_for_bins(255) > 1024)
    var raised = False
    try:
        _ = derive_policy(
            GpuProfile.from_reported("metal", "apple-m1", 7, 1024, 1024),
            100_000,
            4,
            255,
        )
    except:
        raised = True
    assert_true(raised)


def test_crossover_inputs_are_reported_but_no_threshold_is_invented() raises:
    """`auto` resolves to the CPU in device.mojo because the one end-to-end
    measurement taken is slower on the GPU. This layer reports what a
    crossover rule would key on and refuses to supply the rule."""
    var profiles = [
        GpuProfile.generic(),
        apple_m4_observed(),
        apple_synthetic(APPLE_GEN_M1),
        _discrete(108, 256 << 20),
    ]
    for i in range(len(profiles)):
        var policy = derive_policy(profiles[i], 1_000_000, 20, 255)
        assert_equal(policy.crossover.min_cells, CROSSOVER_DISABLED)
        assert_true(CROSSOVER_DISABLED < 0)
        assert_equal(policy.crossover.cells, 1_000_000 * 20)
        assert_equal(policy.crossover.n_bins, 255)
        assert_equal(policy.crossover.api, profiles[i].api)
        assert_equal(
            policy.crossover.apple_generation, profiles[i].apple_generation
        )
        assert_equal(policy.crossover.core_count, profiles[i].core_count)
        assert_equal(
            policy.crossover.device_parallel_width,
            profiles[i].core_count * policy.block_threads,
        )

    # The width is the quantity a bare cell count cannot stand in for: the
    # same shape means different things across a 7-core and a 108-core part.
    var m1 = apple_synthetic(APPLE_GEN_M1)
    var narrow = derive_policy(m1, 1_000_000, 20, 255)
    var wide = derive_policy(_discrete(108), 1_000_000, 20, 255)
    assert_equal(narrow.crossover.cells, wide.crossover.cells)
    assert_true(
        wide.crossover.device_parallel_width
        > narrow.crossover.device_parallel_width
    )


def test_description_distinguishes_a_reading_from_a_fixture() raises:
    var observed = apple_m4_observed()
    var line = describe_policy(
        observed, derive_policy(observed, 1_000_000, 8, 255)
    )
    assert_true(line.find("api=metal") >= 0)
    assert_true(line.find("apple_gen=m4") >= 0)
    assert_true(line.find("(reported)") >= 0)

    var fixture = apple_synthetic(APPLE_GEN_M2)
    var fixture_line = describe_policy(
        fixture, derive_policy(fixture, 1_000_000, 8, 255)
    )
    assert_true(fixture_line.find("(synthetic)") >= 0)
    assert_true(fixture_line.find("apple_gen=m2") >= 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
