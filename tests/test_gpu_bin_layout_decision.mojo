"""The arithmetic behind "the GPU histogram path reads feature-major".

Pure host arithmetic over `gpu_binned_layout`, so it runs on every machine
with or without an accelerator, and it opens no `DeviceContext`. Nothing here
is a measurement and nothing here should be read as one: every number is a
DERIVED BOUND out of a sector model, and the module docstring says which
device constant would have to be measured before any of it becomes a
threshold.

What this file pins is the *shape* of the conclusion rather than any
particular ratio, because the ratios move with a sector size nobody has
measured:

- at `feature_group = 1` a blocked layout and a feature-major one touch the
  identical number of sectors, exactly, at every depth and every sector size;
- the row-side term divides by `feature_group` on both layouts, so it is not
  an argument for blocking (this is the correction that closed the lane);
- the blocked layout's bin advantage is bounded by the block width and is
  1.0 at the root;
- feature subsampling inverts the sign.
"""

from std.testing import assert_equal, assert_true, TestSuite

from mojotrees.gpu_bin_packing import BIN_WIDTH_FALLBACK
from mojotrees.gpu_binned_layout import (
    SECTOR_MODEL_BOUND,
    SECTOR_MODEL_OCCUPANCY,
    layout_node_cost,
    layout_tree_cost,
    plan_feature_blocked,
    plan_feature_major,
    sectors_touched,
    sectors_touched_occupancy,
    sectors_under,
)

# A power of two, so every sector count divides exactly and the identities
# below are equalities rather than near-equalities. Nothing in the model
# depends on the row count being round; the assertions are just cleaner to
# read when the arithmetic does not carry a rounding term.
comptime N_ROWS = 131_072
comptime N_FEATURES = 64
comptime N_BINS = 255


def _widths() -> List[Int]:
    var w = List[Int](capacity=N_FEATURES)
    w.resize(N_FEATURES, BIN_WIDTH_FALLBACK)
    return w^


def _all_features() -> List[Int]:
    var a = List[Int](capacity=N_FEATURES)
    for f in range(N_FEATURES):
        a.append(f)
    return a^


def _every_other() -> List[Int]:
    """Half the features, spread across every block: what a random
    `feature_fraction` of 0.5 looks like to a blocked layout."""
    var a = List[Int]()
    for f in range(0, N_FEATURES, 2):
        a.append(f)
    return a^


def test_group_one_never_helps_and_usually_hurts() raises:
    """The load-bearing negative result. With one feature slot per
    threadgroup, which is what `free_feature_group` returns on every backend
    that is not Metal at 255 bins, a launch fetches a `G`-wide row and spends
    one byte of it, so the block is walked again for every other lane. The
    blocked layout can never touch fewer sectors than the feature-major one
    there, at any depth, under either model, at any sector size, and in the
    streaming regime it touches strictly more. The row side and the launch
    count are identical, because they belong to the launch shape and not to
    the layout."""
    var major = plan_feature_major(N_ROWS, N_FEATURES, N_BINS, _widths())
    var blocked = plan_feature_blocked(
        N_ROWS, N_FEATURES, N_BINS, _widths(), 8, True
    )
    var active = _all_features()
    for s in [32, 64, 128]:
        for m in [SECTOR_MODEL_BOUND, SECTOR_MODEL_OCCUPANCY]:
            var a = layout_tree_cost(major, 6, active, s, 1, m)
            var b = layout_tree_cost(blocked, 6, active, s, 1, m)
            assert_equal(a.row_sectors, b.row_sectors)
            assert_equal(a.launches, b.launches)
            assert_true(b.bin_sectors >= a.bin_sectors)
    # At the root, where every layout streams, the regression is the full
    # block width: the blocked buffer is fetched once per lane.
    var root_major = layout_node_cost(major, N_ROWS, active, 128, 1)
    var root_blocked = layout_node_cost(blocked, N_ROWS, active, 128, 1)
    assert_equal(root_blocked.bin_sectors, 8 * root_major.bin_sectors)


def test_row_side_divides_by_the_launch_group_on_both_layouts() raises:
    """The correction that killed the old argument for blocking. The row side
    is charged once per launch block, and a launch block is `feature_group`
    feature slots on whatever layout is underneath, so doubling the group
    halves the row side for the feature-major layout too."""
    var major = plan_feature_major(N_ROWS, N_FEATURES, N_BINS, _widths())
    var active = _all_features()
    var g1 = layout_node_cost(major, N_ROWS, active, 128, 1)
    var g2 = layout_node_cost(major, N_ROWS, active, 128, 2)
    var g8 = layout_node_cost(major, N_ROWS, active, 128, 8)
    assert_equal(g1.launches, N_FEATURES)
    assert_equal(g2.launches, N_FEATURES // 2)
    assert_equal(g8.launches, N_FEATURES // 8)
    assert_equal(g1.row_sectors, 2 * g2.row_sectors)
    assert_equal(g2.row_sectors, 4 * g8.row_sectors)
    # And the bin term is untouched by the launch shape, on this layout.
    assert_equal(g1.bin_sectors, g8.bin_sectors)


def test_no_bin_advantage_at_the_root() raises:
    """At the root the two layouts move the identical bytes in the identical
    order, so they touch the identical sectors. Any advantage a blocked
    layout has is a deep-node advantage by construction."""
    var major = plan_feature_major(N_ROWS, N_FEATURES, N_BINS, _widths())
    var blocked = plan_feature_blocked(
        N_ROWS, N_FEATURES, N_BINS, _widths(), 8, True
    )
    var active = _all_features()
    for s in [32, 64, 128]:
        var a = layout_node_cost(major, N_ROWS, active, s, 8)
        var b = layout_node_cost(blocked, N_ROWS, active, s, 8)
        assert_equal(a.bin_sectors, b.bin_sectors)


def test_deep_node_advantage_is_bounded_by_the_block_width() raises:
    """`clamp(G / (rho * S), 1, G)`: never worse than a tie, never better
    than the block width. A test of the bound, not of any ratio inside it."""
    var major = plan_feature_major(N_ROWS, N_FEATURES, N_BINS, _widths())
    var blocked = plan_feature_blocked(
        N_ROWS, N_FEATURES, N_BINS, _widths(), 8, True
    )
    var active = _all_features()
    for s in [32, 64, 128]:
        for m in [SECTOR_MODEL_BOUND, SECTOR_MODEL_OCCUPANCY]:
            var rows = N_ROWS
            for _ in range(8):
                var a = layout_node_cost(major, rows, active, s, 8, m)
                var b = layout_node_cost(blocked, rows, active, s, 8, m)
                assert_true(b.bin_sectors <= a.bin_sectors)
                assert_true(8 * b.bin_sectors >= a.bin_sectors)
                rows //= 2


def test_feature_subsampling_inverts_the_sign() raises:
    """A random half of the features leaves every storage block live while
    halving feature-major's touched columns, so the blocked layout moves
    strictly more bin traffic. This is the configuration in which the change
    would have been a regression."""
    var major = plan_feature_major(N_ROWS, N_FEATURES, N_BINS, _widths())
    var blocked = plan_feature_blocked(
        N_ROWS, N_FEATURES, N_BINS, _widths(), 8, True
    )
    var half = _every_other()
    var a = layout_tree_cost(major, 5, half, 128, 8, SECTOR_MODEL_OCCUPANCY)
    var b = layout_tree_cost(blocked, 5, half, 128, 8, SECTOR_MODEL_OCCUPANCY)
    assert_true(b.bin_sectors > a.bin_sectors)


def test_the_two_sector_models_agree_at_the_ends() raises:
    """They are two readings of one gather, so they have to coincide where
    the gather is unambiguous: a node of every row streams the whole extent
    under both, and a single access touches one sector under both."""
    for s in [32, 64, 128]:
        var span = N_ROWS
        assert_equal(
            sectors_touched(N_ROWS, span, 1, s),
            sectors_touched_occupancy(N_ROWS, span, 1, s),
        )
        assert_equal(sectors_touched(1, span, 1, s), 1)
        assert_equal(sectors_touched_occupancy(1, span, 1, s), 1)
        # Between the ends the occupancy model never exceeds the bound.
        assert_true(
            sectors_touched_occupancy(N_ROWS // 32, span, 1, s)
            <= sectors_touched(N_ROWS // 32, span, 1, s)
        )


def test_unmeasured_sector_size_stays_unmeasured() raises:
    """`SECTOR_BYTES_UNKNOWN` propagates as zero rather than as a guess,
    through both models and through the dispatcher."""
    assert_equal(sectors_touched(1000, 100_000, 1, 0), 0)
    assert_equal(sectors_touched_occupancy(1000, 100_000, 1, 0), 0)
    assert_equal(sectors_under(SECTOR_MODEL_BOUND, 1000, 100_000, 1, 0), 0)
    assert_equal(
        sectors_under(SECTOR_MODEL_OCCUPANCY, 1000, 100_000, 1, 0), 0
    )
    var major = plan_feature_major(N_ROWS, N_FEATURES, N_BINS, _widths())
    var cost = layout_node_cost(major, N_ROWS, _all_features(), 0, 8)
    assert_true(not cost.is_priced())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
