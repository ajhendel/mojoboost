"""The bit-packed bin layout, read by the real histogram kernels.

`test_gpu_packed_bins.mojo` checks the addresses on the host and opens no
device. This file is the other half: it puts a bit-packed matrix on an
accelerator, builds a node's histogram through
`GpuActiveRows.enqueue_range_histogram` with the packed arm on and with it
off, and asserts the two are equal **bit for bit**.

Bit for bit and not to a tolerance, because the claim is not that the two are
close. Packing does not renumber a bin: `gpu_bin_packing` stores an integer
and returns the same integer, so for every `(row, feature)` the decoded id is
the id the dense matrix held. The same id then selects the same shared cell,
the same three quantized values are added to it by the same atomics, and
nothing about the visit set, the tiling, the slot assignment or the flush
moves. Accumulation is fixed-point Int32 throughout, so there is not even a
reordering for associativity to have to excuse. A single differing word means
the reader decoded a cell belonging to a different row -- a legal bin id in a
well-formed histogram, which is why nothing downstream would have caught it.

**Both arms run in one process against one `GpuActiveRows`.** That is the
protocol this machine's device timings force (`bench/README.md`), and it is
also the stronger correctness statement: the same buffers, the same
permutation, the same tiling, one call to `set_packed_bins` between them. A
two-process comparison could not tell a layout bug from a fixture difference.

The fixture is mixed-cardinality on purpose, at widths 4, 3, 2, 1 and 2 with
the dataset's own `n_bins` at 16. A matrix whose features all share one width
spells a stride error and a shift error identically; these five do not. One
feature is deliberately *not* the widest, so the reader cannot pass by using a
single hoisted width, and one is a boolean, which is the narrowest stream the
layout can produce.

Both strategies are exercised. The atomic kernel and the partial kernel each
carry their own copy of the pointer resolution and each inlines the row loop
twice, so a change made in one and missed in the other is exactly the kind of
half-application this repository has warned itself about.
"""

from std.sys import has_accelerator
from std.testing import assert_equal, assert_raises, assert_true, TestSuite
from max.gpu.host import DeviceBuffer, DeviceContext

from mojotrees.binning import BinnedMatrix
from mojotrees.gpu_active_rows import GpuActiveRows, RowRouting
from mojotrees.gpu_packed_bins import (
    packed_bin_bytes_ratio,
    packed_bytes,
    packed_is_identity,
    packed_roundtrips,
    packed_widths_from_matrix,
)
from mojotrees.gpu_tiling import (
    STRATEGY_ATOMIC,
    STRATEGY_TILED,
    query_device_caps,
)
from support import _splitmix64


def _used_bins() -> List[Int]:
    """Used-bin counts per feature: 16, 8, 4, 2 and 3, so the widths are 4, 3,
    2, 1 and 2.

    Three of the five do not divide a byte evenly, which is where the
    two-byte decode window earns its place, and the widths are not monotone in
    the feature index, so nothing can pass by assuming a ladder.
    """
    return [16, 8, 4, 2, 3]


def _mixed_data(n_rows: Int) raises -> BinnedMatrix:
    """A matrix whose columns have genuinely different cardinalities.

    Row 0 of every column is that feature's top bin, so the observed extent is
    the intended one whatever the row count, and every other cell is drawn
    pseudorandomly inside the feature's own range. The draw is seeded on
    `(f, r)` so no two columns are the same sequence: a reader that fetched
    the wrong *column* would otherwise find the right value there.
    """
    var used_bins = _used_bins()
    var n_features = len(used_bins)
    var bins = List[UInt8](capacity=n_rows * n_features)
    bins.resize(n_rows * n_features, 0)
    for f in range(n_features):
        var used = used_bins[f]
        for r in range(n_rows):
            var v: Int
            if r == 0:
                v = used - 1
            else:
                v = Int(
                    _splitmix64(UInt64(f * n_rows + r) + 0x51ED2701)
                    % UInt64(used)
                )
            bins[f * n_rows + r] = UInt8(v)
    return BinnedMatrix(bins^, n_rows, n_features, 16)


def _upload_bins(
    mut ctx: DeviceContext, data: BinnedMatrix
) raises -> DeviceBuffer[DType.uint8]:
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var buf = ctx.enqueue_create_buffer[DType.uint8](len(data.bins))
        ctx.enqueue_copy(dst_buf=buf, src_ptr=data.bins.unsafe_ptr())
        ctx.synchronize()
        return buf^


def _fixed_scale(values: List[Float64]) -> Float32:
    var total = 0.0
    for i in range(len(values)):
        total += abs(values[i])
    if total < 1e-12:
        total = 1e-12
    return Float32(Float64(1 << 30) / total)


def _packed_arm_case(strategy: Int) raises:
    """One node's histogram with the packed arm off, then on, compared word
    for word."""
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var n_rows = 2048
        var n_bins = 16
        var data = _mixed_data(n_rows)
        var n_features = data.n_features

        var widths = packed_widths_from_matrix(data)
        assert_equal(len(widths), n_features)
        assert_equal(widths[0], 4)
        assert_equal(widths[1], 3)
        assert_equal(widths[2], 2)
        assert_equal(widths[3], 1)
        assert_equal(widths[4], 2)
        # The layout is not the identity here, so the arm is a real change
        # rather than a second copy of the same bytes.
        assert_true(not packed_is_identity(widths))
        # And the host round trip holds before a device ever sees it, so a
        # failure below is the *reader's* and not the encoding's.
        assert_true(packed_roundtrips(data, widths))

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

        # The slots are not the identity, so the reader has to take a width
        # and a base from `feat_ids[slot]` rather than from the slot index.
        var feat_dev = ctx.enqueue_create_buffer[DType.int32](n_features)
        with feat_dev.map_to_host() as host:
            host.unsafe_ptr().unsafe_store(0, Int32(3))
            host.unsafe_ptr().unsafe_store(1, Int32(0))
            host.unsafe_ptr().unsafe_store(2, Int32(4))
            host.unsafe_ptr().unsafe_store(3, Int32(1))
            host.unsafe_ptr().unsafe_store(4, Int32(2))

        var hist_size = n_features * n_bins
        var cells = 3 * hist_size
        var plain_dev = ctx.enqueue_create_buffer[DType.int32](cells)
        var pack_dev = ctx.enqueue_create_buffer[DType.int32](cells)
        var host_plain = ctx.enqueue_create_host_buffer[DType.int32](cells)
        var host_pack = ctx.enqueue_create_host_buffer[DType.int32](cells)

        # A node whose range does not start at 0, so the packed reader has to
        # follow the permutation rather than the row index.
        var routing = RowRouting.numerical(0, 6)
        _ = rows.partition(bins.unsafe_ptr(), 0, 1, 2, routing)
        var node = 2
        assert_true(len(rows.download_range(node)) > 0)

        var tiling = rows.range_tiling(
            caps, node, n_features, strategy, 1 << 20
        )
        assert_equal(tiling.strategy, strategy)
        var part_cells = tiling.partial_cells
        if part_cells < 1:
            part_cells = 1
        var part_dev = ctx.enqueue_create_buffer[DType.int32](3 * part_cells)

        # Arm one: the byte layout every other reader of `bins` still uses.
        assert_true(not rows.packed_bins_active())
        rows.enqueue_range_histogram(
            tiling,
            node,
            bins.unsafe_ptr(),
            grad_dev.unsafe_ptr(),
            hess_dev.unsafe_ptr(),
            feat_dev.unsafe_ptr(),
            plain_dev.unsafe_ptr(),
            part_dev.unsafe_ptr(),
            n_features,
            g_scale,
            h_scale,
        )

        # Arm two: the same instance, one setter apart.
        rows.set_packed_bins(widths.copy())
        rows.enqueue_range_histogram(
            tiling,
            node,
            bins.unsafe_ptr(),
            grad_dev.unsafe_ptr(),
            hess_dev.unsafe_ptr(),
            feat_dev.unsafe_ptr(),
            pack_dev.unsafe_ptr(),
            part_dev.unsafe_ptr(),
            n_features,
            g_scale,
            h_scale,
        )
        ctx.enqueue_copy(dst_ptr=host_plain.unsafe_ptr(), src_buf=plain_dev)
        ctx.enqueue_copy(dst_ptr=host_pack.unsafe_ptr(), src_buf=pack_dev)
        ctx.synchronize()
        # The layout only becomes active once `_ensure_packed` has filled the
        # buffer, which the launch above is what triggers.
        assert_true(rows.packed_bins_active())

        var a = host_plain.unsafe_ptr()
        var b = host_pack.unsafe_ptr()
        var populated = 0
        for i in range(cells):
            assert_equal(Int(a.unsafe_load(i)), Int(b.unsafe_load(i)))
            if a.unsafe_load(i) != 0:
                populated += 1
        # A comparison of two all-zero buffers would pass for the wrong
        # reason, and an unbuilt packed buffer decodes to zeros.
        assert_true(populated > 0)

        # Turning it back off inside the same process restores the byte
        # reader, which is what makes the two arms interleavable rather than
        # a build-time choice.
        rows.set_packed_bins(List[Int]())
        assert_true(not rows.packed_bins_active())


def test_packed_bins_build_the_identical_histogram_atomic() raises:
    comptime if not has_accelerator():
        return
    else:
        _packed_arm_case(STRATEGY_ATOMIC)


def test_packed_bins_build_the_identical_histogram_tiled() raises:
    comptime if not has_accelerator():
        return
    else:
        _packed_arm_case(STRATEGY_TILED)


def test_packed_and_blocked_layouts_refuse_each_other() raises:
    """The two layouts are orthogonal in principle and are not composed here.

    Refused rather than left to the reader, because a reader handed both a
    block stride and a bit width would form an address that is wrong in a way
    that produces a legal bin id. Neither arm has been measured yet; the cross
    product would be a third unmeasured thing.
    """
    comptime if not has_accelerator():
        return
    else:
        var n_rows = 512
        var data = _mixed_data(n_rows)
        var widths = packed_widths_from_matrix(data)
        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var rows = GpuActiveRows(
            ctx, n_rows, data.n_features, 16, caps
        )
        rows.set_packed_bins(widths.copy())
        rows.set_feature_group(2)
        with assert_raises(contains="cannot both be on"):
            rows.set_blocked_layout(2)
        rows.set_packed_bins(List[Int]())
        rows.set_blocked_layout(2)
        with assert_raises(contains="cannot both be on"):
            rows.set_packed_bins(widths.copy())
        rows.set_blocked_layout(0)
        rows.set_feature_group(1)


def test_a_matrix_with_nothing_to_pack_is_refused() raises:
    """An all-width-8 table is the passthrough plan, whose buffer is
    `BinnedMatrix.bins` itself. Uploading a second copy of it would cost
    residency and buy nothing, so the setter says so."""
    comptime if not has_accelerator():
        return
    else:
        var n_rows = 256
        var data = _mixed_data(n_rows)
        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var rows = GpuActiveRows(
            ctx, n_rows, data.n_features, 16, caps
        )
        var eight = List[Int]()
        for _ in range(data.n_features):
            eight.append(8)
        with assert_raises(contains="nothing to pack"):
            rows.set_packed_bins(eight^)
        var short = List[Int]()
        short.append(4)
        with assert_raises(contains="one width per feature"):
            rows.set_packed_bins(short^)
        assert_true(not rows.packed_bins_active())


def test_the_packed_buffer_is_smaller_than_the_matrix_it_shadows() raises:
    """The residency claim, which is the one thing this layout has that the
    blocked layout does not: its second buffer is a fraction of the first."""
    var n_rows = 2048
    var data = _mixed_data(n_rows)
    var widths = packed_widths_from_matrix(data)
    var total = packed_bytes(n_rows, widths)
    assert_true(total < n_rows * data.n_features)
    # (4 + 3 + 2 + 1 + 2) / 40 = 0.3 of the bin bytes per visit.
    var ratio = packed_bin_bytes_ratio(widths)
    assert_true(ratio > 0.29 and ratio < 0.31)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
