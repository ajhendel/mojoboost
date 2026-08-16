"""The row partition's launch geometry cannot move a single row.

`gpu_active_rows` used to partition a leaf's range in four launches: a
flag-and-scan pass, a launch that scanned the per-block left counts into
per-block offsets, a scatter, and a copy back. The block-sums scan is now
done by every scatter threadgroup at its own head, which removes that
launch and leaves three. Doing it there is only affordable if the block
count is bounded, because each block re-reads the whole block-sums array, so
the partition now caps its threadgroup count (`GpuActiveRows.partition_block_cap`,
`_partition_grid`) and gives each threadgroup several `block_threads`-wide
tiles to walk when the range is longer than the cap can cover in one tile
each.

That is a change of launch *shape*, and the property this file exists to
hold it to is that the shape is invisible. A row's destination inside its
range is `P(j)` if it goes left and `T + (j - P(j))` if it goes right, where
`P(j)` counts the left-going rows at positions before `j` and `T` counts all
of them; neither depends on how the range was cut into chunks, how many
chunks there were, how many tiles each walked, or which scan arm computed
the prefix. So the permutation must be identical element for element, and
not merely equivalent, across every cap. It has to be identical because
`tests/test_host_replica.mojo` verifies a host replica bit for bit against the
device and `readback_range` hands a node's compacted rows straight to a host
histogram build: an equivalent permutation would silently fail both.

What is asserted, in order:

- `_partition_grid` covers the range, never exceeds the cap, never returns a
  threadgroup that owns nothing, and degenerates to exactly the old
  one-tile-per-block geometry whenever the range fits under the cap that
  way. This half is a host predicate and runs with no accelerator.
- the whole row buffer and every split's left count are identical across
  block caps of 1, 2, 3, and the default, over a scripted three-level tree,
  and identical to `partition_range_host`, the serial reference model, which
  is also what asserts that the rows outside the partitioned window are
  untouched.
- the same across caps on the fallback (hand-rolled) scan arm, and the two
  arms agree with each other at a cap that forces many tiles per block, so
  the folded head scan is exercised on both arms.
- a bagged root, a categorical routing, and a missing-bin routing partition
  identically under a forced multi-tile cap, since those change which rows
  are flagged and a tiling bug could hide behind a uniform flag pattern.
- a range that goes entirely one way, and a single-row range, under a cap of
  one, which is the extreme where a single threadgroup walks every tile.
- a cap above the threadgroup width, which is the only shape that gives the
  scatter's head scan more than one chunk of block sums to walk. Production
  never asks for it, since the default cap is one threadgroup width; the
  setter will, so it is held to the same reference model as everything else.
- `set_partition_block_cap` refuses a cap below one.

Nothing here times anything. This file cannot measure and does not claim
that three launches are faster than four; it claims only that they compute
the same thing.

The device half skips (passing) with no accelerator present, so the file
stays green on CPU-only machines.
"""

from std.sys import has_accelerator
from std.testing import assert_equal, assert_raises, assert_true, TestSuite
from max.gpu.host import DeviceBuffer, DeviceContext

from mojotrees.binning import BinnedMatrix
from mojotrees.categorical import CategoricalSpec, cat_add, cat_empty
from mojotrees.gpu_active_rows import (
    GpuActiveRows,
    LeafRange,
    RowRouting,
    _partition_grid,
    partition_range_host,
)
from mojotrees.gpu_tiling import DeviceCaps, query_device_caps
from support import _splitmix64


def _make_data(
    n_rows: Int, n_features: Int, n_bins: Int, missing_bin: Int = -1
) raises -> BinnedMatrix:
    """A column-major binned matrix of pseudorandom bins, with every feature
    reserving `missing_bin` when one is asked for. The same shape the other
    active-row test files build, repeated here rather than imported so that
    this file does not have to reach into another test module."""
    var bins = List[UInt8](capacity=n_rows * n_features)
    for f in range(n_features):
        for r in range(n_rows):
            var v = _splitmix64(UInt64(f * n_rows + r) + 0x2C1B3A5D)
            bins.append(UInt8(Int(v % UInt64(n_bins))))
    if missing_bin < 0:
        return BinnedMatrix(bins^, n_rows, n_features, n_bins)
    var table = List[Int](capacity=n_features)
    for _ in range(n_features):
        table.append(missing_bin)
    return BinnedMatrix(
        bins^, n_rows, n_features, n_bins, CategoricalSpec.none(), table^
    )


def _zeros(n: Int) -> List[Int32]:
    var out = List[Int32](capacity=n)
    for _ in range(n):
        out.append(Int32(0))
    return out^


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


def _spread_bag(n_rows: Int, size: Int) -> List[Int]:
    """`size` rows spread across the dataset, so the bin gathers are not one
    contiguous stripe and the bag order is not the row order."""
    var bag = List[Int]()
    var stride = n_rows // size
    for i in range(size):
        bag.append(i * stride)
    return bag^


def _all_right_routing(feature: Int) -> RowRouting:
    """The empty categorical set, so no row is a member and every row goes
    right."""
    return RowRouting.categorical(feature, cat_empty())


def _all_left_routing(feature: Int) -> RowRouting:
    """Every bin in the set, so every row goes left. Bin 0 is deliberately a
    member here, unlike a real categorical split, because this routing exists
    to drive a whole range one way and not to model a split."""
    var bitset = cat_empty()
    for b in range(256):
        cat_add(bitset, b)
    return RowRouting.categorical(feature, bitset)


def _run_script(
    mut ctx: DeviceContext,
    caps: DeviceCaps,
    data: BinnedMatrix,
    mut bins: DeviceBuffer[DType.uint8],
    bag: List[Int],
    parents: List[Int],
    routings: List[RowRouting],
    primitives: Bool,
    block_cap: Int,
    mut out_rows: List[Int32],
    mut out_lefts: List[Int],
) raises:
    """Grow one scripted tree at one block cap on one scan arm, and hand back
    the whole row buffer and every split's left count.

    Split `i` takes `parents[i]` and numbers its children `2i + 1` and
    `2i + 2`, the same convention the scan-primitives file uses, so the caller
    writes only the parent it wants split next. The left count is downloaded
    rather than supplied, which is what puts the device's own count under
    test rather than the caller's. `block_cap` of zero means leave the
    constructor's default alone, which is how the default geometry gets into
    the comparison as one arm among the others.
    """
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var rows = GpuActiveRows(
            ctx, data.n_rows, data.n_features, data.n_bins, caps
        )
        rows.set_scan_primitives(primitives)
        if block_cap > 0:
            rows.set_partition_block_cap(block_cap)
            assert_equal(rows.partition_block_cap, block_cap)
        else:
            assert_equal(rows.partition_block_cap, rows.block_threads)
        rows.begin_tree(bag)
        out_lefts.clear()
        for i in range(len(parents)):
            out_lefts.append(
                rows.partition(
                    bins.unsafe_ptr(),
                    parents[i],
                    2 * i + 1,
                    2 * i + 2,
                    routings[i],
                )
            )
        out_rows = rows.download_rows()
        rows.ranges.check_invariants()


def _reference_rows(
    data: BinnedMatrix,
    bag: List[Int],
    parents: List[Int],
    routings: List[RowRouting],
    mut out_rows: List[Int32],
    mut out_lefts: List[Int],
) raises:
    """The same script on the host, through `partition_range_host`.

    Seeded exactly as `begin_tree` seeds the device: the identity permutation
    unbagged, the bag in the caller's order otherwise, with the slots past the
    active prefix left at zero, which is what the constructor writes them to.
    The child windows are tracked here rather than read back from the device
    so that a wrong window on the device cannot make itself look right.
    """
    out_rows = _zeros(data.n_rows)
    if len(bag) == 0:
        for r in range(data.n_rows):
            out_rows[r] = Int32(r)
    else:
        for i in range(len(bag)):
            out_rows[i] = Int32(bag[i])
    var scratch = _zeros(data.n_rows)
    var n_active = data.n_rows if len(bag) == 0 else len(bag)

    var begins = List[Int]()
    var ends = List[Int]()
    begins.append(0)
    ends.append(n_active)
    out_lefts.clear()
    for i in range(len(parents)):
        var window = LeafRange(begins[parents[i]], ends[parents[i]])
        var want = partition_range_host(
            out_rows, scratch, data, window, routings[i]
        )
        out_lefts.append(want)
        while len(begins) < 2 * i + 3:
            begins.append(0)
            ends.append(0)
        begins[2 * i + 1] = window.begin
        ends[2 * i + 1] = window.begin + want
        begins[2 * i + 2] = window.begin + want
        ends[2 * i + 2] = window.end
        begins[parents[i]] = window.begin
        ends[parents[i]] = window.begin


def _assert_caps_agree(
    mut ctx: DeviceContext,
    caps: DeviceCaps,
    data: BinnedMatrix,
    mut bins: DeviceBuffer[DType.uint8],
    bag: List[Int],
    parents: List[Int],
    routings: List[RowRouting],
    primitives: Bool,
    block_caps: List[Int],
) raises:
    """Run one script at several block caps and require every one of them to
    produce the identical row buffer and the identical left counts, and to
    match the serial reference model over the whole buffer."""
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var want_rows = List[Int32]()
        var want_lefts = List[Int]()
        _reference_rows(data, bag, parents, routings, want_rows, want_lefts)

        for c in range(len(block_caps)):
            var got_rows = List[Int32]()
            var got_lefts = List[Int]()
            _run_script(
                ctx,
                caps,
                data,
                bins,
                bag,
                parents,
                routings,
                primitives,
                block_caps[c],
                got_rows,
                got_lefts,
            )
            assert_equal(len(got_lefts), len(want_lefts))
            for i in range(len(want_lefts)):
                assert_equal(got_lefts[i], want_lefts[i])
            assert_equal(len(got_rows), len(want_rows))
            for j in range(len(want_rows)):
                assert_equal(Int(got_rows[j]), Int(want_rows[j]))


def _three_level_script(
    mut parents: List[Int], mut routings: List[RowRouting]
) raises:
    """Root, then both of its children, so two of the three splits act on a
    range that does not begin at zero and whose original order is the
    parent's compacted order rather than the identity."""
    parents.clear()
    routings.clear()
    parents.append(0)
    parents.append(1)
    parents.append(2)
    routings.append(
        RowRouting.numerical(0, 7, missing_bin=0, default_left=True)
    )
    routings.append(
        RowRouting.numerical(2, 5, missing_bin=0, default_left=False)
    )
    routings.append(
        RowRouting.numerical(1, 9, missing_bin=0, default_left=True)
    )


def _caps_1_2_3_and_default() -> List[Int]:
    """Caps that force one, two, and three threadgroups, plus the default
    (0 means leave it alone). One threadgroup is the extreme where a single
    block walks every tile and its head scan has exactly one block sum to
    scan; two and three are where the head scan has something to add up and
    the chunk boundaries are not the tile boundaries."""
    var out = List[Int]()
    out.append(1)
    out.append(2)
    out.append(3)
    out.append(0)
    return out^


def test_partition_grid_covers_bounds_and_degenerates() raises:
    """The launch geometry's four properties, as a host predicate.

    Coverage and the cap are what correctness rests on; "no empty
    threadgroup" is what keeps a range of `cap + 1` tiles from launching `cap`
    blocks half of which own nothing; and the degeneracy is the claim that a
    range short enough to fit one tile per block gets exactly the geometry
    this module launched before the head scan was folded into the scatter,
    which is most ranges of most trees.
    """
    var ns = List[Int]()
    ns.append(1)
    ns.append(255)
    ns.append(256)
    ns.append(257)
    ns.append(511)
    ns.append(1024)
    ns.append(65535)
    ns.append(65536)
    ns.append(65537)
    ns.append(1000000)

    var widths = List[Int]()
    widths.append(128)
    widths.append(256)
    widths.append(1024)

    var caps = List[Int]()
    caps.append(1)
    caps.append(2)
    caps.append(3)
    caps.append(128)
    caps.append(256)
    caps.append(1024)

    for a in range(len(ns)):
        for b in range(len(widths)):
            for c in range(len(caps)):
                var n = ns[a]
                var threads = widths[b]
                var cap = caps[c]
                var grid = _partition_grid(n, threads, cap)
                var blocks = grid[0]
                var tiles = grid[1]

                assert_true(blocks >= 1)
                assert_true(tiles >= 1)
                assert_true(blocks <= cap)
                # Every element of the range is inside some block's chunk.
                assert_true(blocks * tiles * threads >= n)
                # And no block's chunk starts past the end of the range, so
                # no threadgroup is launched to own nothing.
                assert_true((blocks - 1) * tiles * threads < n)

                # Below the cap the geometry is the old one, exactly.
                var tiles_total = (n + threads - 1) // threads
                if tiles_total <= cap:
                    assert_equal(tiles, 1)
                    assert_equal(blocks, tiles_total)


def test_partition_grid_rejects_degenerate_arguments() raises:
    """A zero-length range, a zero-width threadgroup, and a zero cap are all
    wiring mistakes rather than shapes with an answer, so they raise here
    instead of producing a grid that would launch nothing or divide by zero.
    `enqueue_partition` never reaches this with a zero-length range, because
    an empty window returns before the grid is derived; the check is here
    because a helper that can be called from a test should not have a silent
    degenerate case. Runs without an accelerator."""
    with assert_raises():
        _ = _partition_grid(0, 256, 256)
    with assert_raises():
        _ = _partition_grid(-1, 256, 256)
    with assert_raises():
        _ = _partition_grid(1000, 0, 256)
    with assert_raises():
        _ = _partition_grid(1000, 256, 0)


def test_the_block_cap_cannot_move_a_row_on_the_primitive_arm() raises:
    """The main assertion of the lane, on the default scan arm.

    A three-level tree over an unbagged root of 4096 rows, run at caps of
    one, two, three, and the default, compared element for element against
    each other through the serial reference model. At a cap of one, 4096 rows
    of a 256-wide threadgroup is a single block walking sixteen tiles; at the
    default it is sixteen blocks walking one tile each, which is the geometry
    the module launched before this lane. Those two are the extremes of the
    change and they have to produce the same buffer.
    """
    comptime if not has_accelerator():
        return
    else:
        var n_rows = 4096
        var data = _make_data(n_rows, 3, 16, missing_bin=0)
        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var bins = _upload_bins(ctx, data)

        var parents = List[Int]()
        var routings = List[RowRouting]()
        _three_level_script(parents, routings)

        _assert_caps_agree(
            ctx,
            caps,
            data,
            bins,
            List[Int](),
            parents,
            routings,
            True,
            _caps_1_2_3_and_default(),
        )


def test_the_block_cap_cannot_move_a_row_on_the_fallback_arm() raises:
    """The same, on the hand-rolled Hillis-Steele arm.

    The head scan the scatter now performs exists twice, once per arm, so a
    tiling bug could live in one of them alone. This runs the identical script
    with `set_scan_primitives(False)` and holds it to the identical reference,
    which also means the two arms are being compared with each other by
    transitivity at every cap.
    """
    comptime if not has_accelerator():
        return
    else:
        var n_rows = 4096
        var data = _make_data(n_rows, 3, 16, missing_bin=0)
        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var bins = _upload_bins(ctx, data)

        var parents = List[Int]()
        var routings = List[RowRouting]()
        _three_level_script(parents, routings)

        _assert_caps_agree(
            ctx,
            caps,
            data,
            bins,
            List[Int](),
            parents,
            routings,
            False,
            _caps_1_2_3_and_default(),
        )


def test_a_forced_multi_tile_cap_survives_bags_and_categoricals() raises:
    """A bagged root, a categorical set-membership routing, and a
    missing-bin routing, all at a cap of two.

    Two threadgroups over a bagged root of 3000 rows is six tiles each at a
    256-wide threadgroup, so every one of these runs the multi-tile loop and
    the folded head scan. Bagging matters because the root is then not the
    identity permutation and stability is stability in *buffer* order;
    categoricals and the missing bin matter because they change which rows
    are flagged, and a tiling bug that happened to be invisible under one
    flag pattern would not be under another.
    """
    comptime if not has_accelerator():
        return
    else:
        var n_rows = 6000
        var data = _make_data(n_rows, 3, 16, missing_bin=0)
        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var bins = _upload_bins(ctx, data)

        var bitset = cat_empty()
        cat_add(bitset, 3)
        cat_add(bitset, 4)
        cat_add(bitset, 11)

        var parents = List[Int]()
        parents.append(0)
        parents.append(1)
        parents.append(2)
        var routings = List[RowRouting]()
        routings.append(RowRouting.categorical(1, bitset))
        routings.append(
            RowRouting.numerical(0, 6, missing_bin=0, default_left=True)
        )
        routings.append(
            RowRouting.numerical(2, 8, missing_bin=0, default_left=False)
        )

        var block_caps = List[Int]()
        block_caps.append(2)
        block_caps.append(0)

        _assert_caps_agree(
            ctx,
            caps,
            data,
            bins,
            _spread_bag(n_rows, 3000),
            parents,
            routings,
            True,
            block_caps,
        )
        _assert_caps_agree(
            ctx,
            caps,
            data,
            bins,
            _spread_bag(n_rows, 3000),
            parents,
            routings,
            False,
            block_caps,
        )


def test_one_directional_and_tiny_ranges_at_a_cap_of_one() raises:
    """Ranges that go entirely one way, and a range of a single row, with
    every tile walked by one threadgroup.

    An all-left range makes the running count across tiles equal the tile
    index times the tile width, and an all-right range leaves it at zero for
    every tile; both are the cases where a carry that is dropped or applied
    twice between tiles produces a buffer that is still a permutation and
    still looks plausible, which is why they are asserted against the
    reference model and not against a count. The single-row range is the
    other end: one tile, one live lane, and `blocks == tiles == 1`.
    """
    comptime if not has_accelerator():
        return
    else:
        var n_rows = 3000
        var data = _make_data(n_rows, 3, 16, missing_bin=0)
        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var bins = _upload_bins(ctx, data)

        var parents = List[Int]()
        parents.append(0)
        parents.append(1)
        var routings = List[RowRouting]()
        routings.append(_all_left_routing(1))
        routings.append(_all_right_routing(1))

        var block_caps = List[Int]()
        block_caps.append(1)
        block_caps.append(0)

        _assert_caps_agree(
            ctx,
            caps,
            data,
            bins,
            List[Int](),
            parents,
            routings,
            True,
            block_caps,
        )

        var tiny = List[Int]()
        tiny.append(17)
        var one_split = List[Int]()
        one_split.append(0)
        var one_routing = List[RowRouting]()
        one_routing.append(
            RowRouting.numerical(0, 7, missing_bin=0, default_left=True)
        )
        _assert_caps_agree(
            ctx,
            caps,
            data,
            bins,
            tiny,
            one_split,
            one_routing,
            True,
            block_caps,
        )


def test_a_cap_above_the_threadgroup_width_scans_in_several_chunks() raises:
    """The scatter's head scan walks the block sums in chunks of a
    threadgroup width, and this is the only test that gives it more than one
    chunk to walk.

    The default cap is one threadgroup width, so in production the head scan
    is a single chunk and the loop runs once; the chunked form is there
    because `set_partition_block_cap` will take a larger cap and a benchmark
    exploring the cap on a wide device should not fall off a cliff into
    silently wrong rows. Reaching two chunks needs more than `threads` blocks,
    and `_partition_grid` trims the block count back down whenever the tile
    count can absorb it, so the shape that actually gets there is a range of
    exactly `2 * threads` tiles at a cap of `2 * threads`: one tile per block
    and twice as many blocks as a chunk of the head scan holds.

    That is `2 * threads * threads` rows, which is 131,072 at the usual
    256-wide threadgroup and is two splits over one small binned matrix, not
    a benchmark. On a device whose `block_threads` is wider than 256 the same
    shape would need a genuinely large row count, so the test skips there
    rather than allocating it; the chunked path is width-independent and the
    narrow device exercises it.
    """
    comptime if not has_accelerator():
        return
    else:
        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var probe = GpuActiveRows(ctx, 16, 2, 16, caps)
        var threads = probe.block_threads
        if 2 * threads * threads > 200000:
            return

        var n_rows = 2 * threads * threads
        # The shape the docstring argues for, asserted rather than assumed:
        # more blocks than one chunk of the head scan holds.
        var grid = _partition_grid(n_rows, threads, 2 * threads)
        assert_equal(grid[1], 1)
        assert_true(grid[0] > threads)

        var data = _make_data(n_rows, 2, 16, missing_bin=0)
        var bins = _upload_bins(ctx, data)

        var parents = List[Int]()
        parents.append(0)
        parents.append(1)
        var routings = List[RowRouting]()
        routings.append(
            RowRouting.numerical(0, 7, missing_bin=0, default_left=True)
        )
        routings.append(
            RowRouting.numerical(1, 4, missing_bin=0, default_left=False)
        )

        var block_caps = List[Int]()
        block_caps.append(2 * threads)
        block_caps.append(0)

        _assert_caps_agree(
            ctx,
            caps,
            data,
            bins,
            List[Int](),
            parents,
            routings,
            True,
            block_caps,
        )
        _assert_caps_agree(
            ctx,
            caps,
            data,
            bins,
            List[Int](),
            parents,
            routings,
            False,
            block_caps,
        )


def test_the_block_cap_setter_refuses_a_cap_below_one() raises:
    """A cap of zero would derive a grid of no threadgroups, which would
    enqueue a partition that wrote nothing and report a stale left count, so
    it is refused where it is asked for rather than at the launch. Needs a
    device only because `GpuActiveRows` allocates on one."""
    comptime if not has_accelerator():
        return
    else:
        var n_rows = 512
        var data = _make_data(n_rows, 2, 16)
        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var rows = GpuActiveRows(
            ctx, n_rows, data.n_features, data.n_bins, caps
        )
        with assert_raises():
            rows.set_partition_block_cap(0)
        with assert_raises():
            rows.set_partition_block_cap(-4)
        # And the refusal left the default in place rather than half applied.
        assert_equal(rows.partition_block_cap, rows.block_threads)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
