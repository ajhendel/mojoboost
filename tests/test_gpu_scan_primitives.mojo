"""The two scan arms of the row partition compute the same permutation.

`gpu_active_rows` scans a threadgroup's routing flags two ways. The default
arm calls `max.gpu.primitives.block.prefix_sum`, warp shuffles plus a single
shared round; the fallback arm, selected by `set_scan_primitives(False)` or
`MOJOTREES_GPU_SCAN_PRIMITIVES=0`, keeps the hand-rolled Hillis-Steele scan
with two barriers per doubling step. The whole point of the lane is that the
choice is invisible: the packed offset word, the block sums, the scatter, and
therefore the permutation and the left count are meant to be byte for byte
identical, so that the row order a node's compacted range holds is still the
row order the CPU grower's row list holds, which the hybrid leaf scheduler
verifies bit for bit and `readback_range` hands to a host histogram build.

An equivalent permutation would not do. These tests therefore assert element
for element and not by count or by set:

- both arms produce the identical row buffer and the identical left count
  over a block-aligned range, a range one row past a block boundary, one row
  short of it, a multi-block range, a single-row range, an empty range, and a
  range every row of which goes the same way;
- the permutation is *stable* on both arms, so within each child the rows
  come out in ascending original order, which for an identity-seeded root is
  ascending row id, and both arms match `partition_range_host`, the serial
  reference model, over the whole buffer;
- the categorical set-membership path and the missing-bin default-left and
  default-right paths partition identically on both arms;
- the left count the device reports is the number of rows the routing rule
  sends left, counted independently on the host.

Nothing here times either arm. This file cannot measure and does not claim
that one arm is faster than the other; it claims only that they agree.

The device half skips (passing) with no accelerator present, so the file
stays green on CPU-only machines; the width-menu test runs everywhere.
"""

from std.sys import has_accelerator
from std.testing import assert_equal, assert_false, assert_true, TestSuite
from max.gpu.host import DeviceBuffer, DeviceContext

from mojotrees.binning import BinnedMatrix
from mojotrees.categorical import CategoricalSpec, cat_add, cat_empty
from mojotrees.gpu_active_rows import (
    GpuActiveRows,
    LeafRange,
    RowRouting,
    _scan_primitive_width_supported,
    partition_range_host,
)
from mojotrees.gpu_tiling import DeviceCaps, query_device_caps
from support import _splitmix64


def _make_data(
    n_rows: Int, n_features: Int, n_bins: Int, missing_bin: Int = -1
) raises -> BinnedMatrix:
    """A column-major binned matrix of pseudorandom bins, with every feature
    reserving `missing_bin` when one is asked for. The same shape the
    active-row tests use, repeated here rather than imported so that this
    file does not have to reach into another test module."""
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


def _all_left_routing(feature: Int) -> RowRouting:
    """Every bin in the set, so every row goes left. Bin 0 is deliberately a
    member here, unlike a real categorical split, because this routing exists
    to drive a whole range one way and not to model a split."""
    var bitset = cat_empty()
    for b in range(256):
        cat_add(bitset, b)
    return RowRouting.categorical(feature, bitset)


def _all_right_routing(feature: Int) -> RowRouting:
    """The empty set, so no row is a member and every row goes right."""
    return RowRouting.categorical(feature, cat_empty())


def _run_arm(
    mut ctx: DeviceContext,
    caps: DeviceCaps,
    data: BinnedMatrix,
    mut bins: DeviceBuffer[DType.uint8],
    bag: List[Int],
    parents: List[Int],
    routings: List[RowRouting],
    primitives: Bool,
    mut out_rows: List[Int32],
    mut out_lefts: List[Int],
) raises:
    """Grow one scripted tree on one arm and hand back the whole row buffer
    and every split's left count.

    Split `i` takes `parents[i]` and numbers its children `2i + 1` and
    `2i + 2`, so the caller writes only the parent it wants split next. The
    left count is downloaded rather than supplied, which is what puts the
    device's own count under test rather than the caller's.
    """
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var rows = GpuActiveRows(
            ctx, data.n_rows, data.n_features, data.n_bins, caps
        )
        rows.set_scan_primitives(primitives)
        assert_equal(rows.scan_primitives, primitives)
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


def _assert_arms_agree(
    mut ctx: DeviceContext,
    caps: DeviceCaps,
    data: BinnedMatrix,
    mut bins: DeviceBuffer[DType.uint8],
    bag: List[Int],
    parents: List[Int],
    routings: List[RowRouting],
) raises:
    """Run the same script on both arms and require the whole row buffer and
    every left count to agree element for element."""
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var prim_rows = List[Int32]()
        var prim_lefts = List[Int]()
        _run_arm(
            ctx,
            caps,
            data,
            bins,
            bag,
            parents,
            routings,
            True,
            prim_rows,
            prim_lefts,
        )
        var hand_rows = List[Int32]()
        var hand_lefts = List[Int]()
        _run_arm(
            ctx,
            caps,
            data,
            bins,
            bag,
            parents,
            routings,
            False,
            hand_rows,
            hand_lefts,
        )

        assert_equal(len(prim_rows), len(hand_rows))
        assert_equal(len(prim_lefts), len(hand_lefts))
        for i in range(len(prim_lefts)):
            assert_equal(prim_lefts[i], hand_lefts[i])
        for j in range(len(prim_rows)):
            assert_equal(Int(prim_rows[j]), Int(hand_rows[j]))

        # And both against the serial reference model, over the whole
        # buffer: agreeing with each other but not with the host would mean
        # two identically wrong arms.
        var host_rows = _zeros(data.n_rows)
        if len(bag) == 0:
            for r in range(data.n_rows):
                host_rows[r] = Int32(r)
        else:
            for i in range(len(bag)):
                host_rows[i] = Int32(bag[i])
        var scratch = _zeros(data.n_rows)
        var n_active = data.n_rows if len(bag) == 0 else len(bag)
        var begins = List[Int]()
        var ends = List[Int]()
        begins.append(0)
        ends.append(n_active)
        for i in range(len(parents)):
            var window = LeafRange(begins[parents[i]], ends[parents[i]])
            var want = partition_range_host(
                host_rows, scratch, data, window, routings[i]
            )
            assert_equal(prim_lefts[i], want)
            # Children take contiguous halves, mirroring the range table.
            while len(begins) < 2 * i + 3:
                begins.append(0)
                ends.append(0)
            begins[2 * i + 1] = window.begin
            ends[2 * i + 1] = window.begin + want
            begins[2 * i + 2] = window.begin + want
            ends[2 * i + 2] = window.end
            begins[parents[i]] = window.begin
            ends[parents[i]] = window.begin
        for j in range(data.n_rows):
            assert_equal(Int(prim_rows[j]), Int(host_rows[j]))


def _spread_bag(n_rows: Int, size: Int) -> List[Int]:
    """`size` rows spread across the dataset, so the bin gathers are not one
    contiguous stripe and the bag order is not the row order."""
    var bag = List[Int]()
    var stride = n_rows // size
    for i in range(size):
        bag.append(i * stride)
    return bag^


def test_scan_primitive_width_menu_is_the_powers_of_two() raises:
    """`block.prefix_sum` takes its block size as a compile-time parameter,
    so only the instantiated widths can run the primitive arm. This asserts
    which those are, and that the legal-but-uninstantiated multiples of 64 a
    `MOJOTREES_GPU_BLOCK_THREADS` override can produce are reported as
    unsupported rather than silently rounded to a neighbour. Runs without an
    accelerator: it is a host predicate."""
    assert_true(_scan_primitive_width_supported(128))
    assert_true(_scan_primitive_width_supported(256))
    assert_true(_scan_primitive_width_supported(512))
    assert_true(_scan_primitive_width_supported(1024))
    assert_false(_scan_primitive_width_supported(64))
    assert_false(_scan_primitive_width_supported(192))
    assert_false(_scan_primitive_width_supported(320))
    assert_false(_scan_primitive_width_supported(2048))


def test_both_arms_agree_at_the_block_boundary_shapes() raises:
    """A block-aligned range, one row past a block boundary, one row short of
    it, several blocks, and a single row. The tail block is where a scan
    differs from a scan: the primitive one has to give the same answer for a
    threadgroup in which only some lanes carry a live flag."""
    comptime if not has_accelerator():
        return
    else:
        var n_rows = 3000
        var data = _make_data(n_rows, 3, 16, missing_bin=0)
        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var bins = _upload_bins(ctx, data)
        var threads = GpuActiveRows(
            ctx, n_rows, data.n_features, data.n_bins, caps
        ).block_threads
        assert_true(_scan_primitive_width_supported(threads))
        assert_true(threads * 2 <= n_rows)

        var sizes = List[Int]()
        sizes.append(threads)
        sizes.append(threads + 1)
        sizes.append(threads - 1)
        sizes.append(2 * threads)
        sizes.append(2 * threads + 3)
        sizes.append(1)

        var parents = List[Int]()
        parents.append(0)
        parents.append(1)
        parents.append(2)
        var routings = List[RowRouting]()
        routings.append(
            RowRouting.numerical(0, 7, missing_bin=0, default_left=True)
        )
        routings.append(
            RowRouting.numerical(2, 5, missing_bin=0, default_left=False)
        )
        routings.append(
            RowRouting.numerical(1, 9, missing_bin=0, default_left=True)
        )

        for s in range(len(sizes)):
            _assert_arms_agree(
                ctx,
                caps,
                data,
                bins,
                _spread_bag(n_rows, sizes[s]),
                parents,
                routings,
            )


def test_both_arms_agree_when_the_block_sums_need_more_than_one_chunk() raises:
    """A range wide enough that stage two cannot scan the block sums in a
    single chunk.

    Stage two walks the per-block counts in chunks of its own threadgroup
    width and carries a running total between chunks. On the primitive arm
    that carry comes out of `block.broadcast` from the last thread rather
    than out of a shared slot, and the loop is the one place the primitive
    collectives are called more than once per launch, so it needs a range
    with more blocks than the threadgroup is wide. At the 256-wide default
    that is any range past 65536 rows, which no other shape in this file
    reaches; the count below clears it on any width in the menu except 1024,
    where the chunk still holds every block a range this size produces and
    the test degrades to another single-chunk case rather than failing.
    """
    comptime if not has_accelerator():
        return
    else:
        var n_rows = 200000
        var data = _make_data(n_rows, 2, 16, missing_bin=0)
        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var bins = _upload_bins(ctx, data)

        var parents = List[Int]()
        parents.append(0)
        parents.append(2)
        var routings = List[RowRouting]()
        routings.append(
            RowRouting.numerical(0, 7, missing_bin=0, default_left=True)
        )
        routings.append(
            RowRouting.numerical(1, 4, missing_bin=0, default_left=False)
        )
        _assert_arms_agree(
            ctx, caps, data, bins, List[Int](), parents, routings
        )


def test_both_arms_agree_on_empty_and_one_directional_ranges() raises:
    """A range every row of which goes left, then one every row of which goes
    right, then a split of the empty child each of those leaves behind. An
    empty window enqueues nothing on either arm, so what is under test is
    that neither arm disturbs the buffer and both report zero."""
    comptime if not has_accelerator():
        return
    else:
        var n_rows = 700
        var data = _make_data(n_rows, 2, 16)
        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var bins = _upload_bins(ctx, data)

        # Everything left, then split the empty right child, then split the
        # full left child every row of which goes right.
        var parents = List[Int]()
        parents.append(0)
        parents.append(2)
        parents.append(1)
        var routings = List[RowRouting]()
        routings.append(_all_left_routing(0))
        routings.append(_all_left_routing(1))
        routings.append(_all_right_routing(1))
        _assert_arms_agree(
            ctx, caps, data, bins, List[Int](), parents, routings
        )

        # The mirror image, and a one-row range driven both ways.
        var flipped = List[RowRouting]()
        flipped.append(_all_right_routing(0))
        flipped.append(_all_right_routing(1))
        flipped.append(_all_left_routing(1))
        _assert_arms_agree(
            ctx, caps, data, bins, List[Int](), parents, flipped
        )
        _assert_arms_agree(
            ctx, caps, data, bins, _spread_bag(n_rows, 1), parents, flipped
        )


def test_both_arms_agree_on_categorical_and_missing_routing() raises:
    """The two routing paths that are not the plain inclusive threshold: set
    membership, where bin 0 is never a member so missing and unseen
    categories go right, and the missing bin taken both ways. The routing
    rule is shared between the arms, so what this pins is that neither scan
    disturbs a range whose flags are patterned rather than random."""
    comptime if not has_accelerator():
        return
    else:
        var n_rows = 2200
        var data = _make_data(n_rows, 3, 16, missing_bin=0)
        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var bins = _upload_bins(ctx, data)

        var bitset = cat_empty()
        cat_add(bitset, 3)
        cat_add(bitset, 5)
        cat_add(bitset, 11)

        var parents = List[Int]()
        parents.append(0)
        parents.append(1)
        parents.append(4)
        var routings = List[RowRouting]()
        routings.append(RowRouting.categorical(0, bitset))
        # The same missing bin, taken left and then right, so a rule that
        # ignored `default_left` would move rows in exactly one of these.
        routings.append(
            RowRouting.numerical(1, 6, missing_bin=0, default_left=True)
        )
        routings.append(
            RowRouting.numerical(1, 6, missing_bin=0, default_left=False)
        )
        _assert_arms_agree(
            ctx, caps, data, bins, List[Int](), parents, routings
        )


def test_the_primitive_arm_is_stable_and_counts_left_correctly() raises:
    """Stability stated directly, on the arm that ships. With an
    identity-seeded root the original order is ascending row id, so each
    child's rows must come out strictly ascending, and the left count the
    device reports must be the number of rows the routing rule sends left,
    counted independently here rather than taken from the reference
    partition."""
    comptime if not has_accelerator():
        return
    else:
        var n_rows = 2600
        var data = _make_data(n_rows, 3, 16, missing_bin=0)
        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var bins = _upload_bins(ctx, data)
        var rows = GpuActiveRows(
            ctx, n_rows, data.n_features, data.n_bins, caps
        )
        rows.set_scan_primitives(True)
        rows.begin_tree()

        var routing = RowRouting.numerical(
            0, 7, missing_bin=0, default_left=True
        )
        var want_left = 0
        for r in range(n_rows):
            if routing.goes_left(data.bin_at(r, routing.feature)):
                want_left += 1
        var got_left = rows.partition(bins.unsafe_ptr(), 0, 1, 2, routing)
        assert_equal(got_left, want_left)
        assert_equal(rows.range_of(1).count(), want_left)
        assert_equal(rows.range_of(2).count(), n_rows - want_left)

        var left = rows.download_range(1)
        var right = rows.download_range(2)
        assert_equal(len(left), want_left)
        assert_equal(len(right), n_rows - want_left)
        for i in range(1, len(left)):
            assert_true(left[i] > left[i - 1])
        for i in range(1, len(right)):
            assert_true(right[i] > right[i - 1])
        # And every row is on the side its own bin says, which is what makes
        # the ascending order a *stable partition* rather than a sort.
        for i in range(len(left)):
            assert_true(
                routing.goes_left(data.bin_at(left[i], routing.feature))
            )
        for i in range(len(right)):
            assert_false(
                routing.goes_left(data.bin_at(right[i], routing.feature))
            )

        # A second level, so stability is asserted for a range that does not
        # start at 0 and whose original order is the parent's compacted
        # order rather than the identity.
        var deeper = RowRouting.numerical(
            2, 11, missing_bin=0, default_left=False
        )
        _ = rows.partition(bins.unsafe_ptr(), 2, 3, 4, deeper)
        var deep_left = rows.download_range(3)
        var deep_right = rows.download_range(4)
        for i in range(1, len(deep_left)):
            assert_true(deep_left[i] > deep_left[i - 1])
        for i in range(1, len(deep_right)):
            assert_true(deep_right[i] > deep_right[i - 1])
        rows.ranges.check_invariants()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
