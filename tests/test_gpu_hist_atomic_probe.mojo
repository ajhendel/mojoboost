"""The histogram atomics probe, which builds a WRONG histogram on purpose.

`GpuActiveRows.set_histogram_atomic_probe` turns off the three shared
`Atomic.fetch_add` calls the histogram row loop issues per (row, feature) and
accumulates a per-thread sink instead. It performs the whole gather and the
whole address computation and skips only the atomics, so the ratio of its
wall clock to the shipping arm's is what share of the histogram phase those
three atomics are. That ratio is the arm's only product and
`set_histogram_atomic_probe` is where what it licenses is written down.

**This file does not time anything.** Timing is
`tools/probe_hist_atomic_fraction.mojo`, which interleaves the two arms in
one process because this machine's device timings drift several-fold between
time windows. What this file establishes is the four properties a probe has
to have before anyone is allowed to run it:

1. It is off by default, on a freshly constructed `GpuActiveRows`.
2. It cannot be turned on without an acknowledgment written into the call.
3. It refuses the two entry points through which a histogram becomes a tree:
   the device-resident growth plane (`enqueue_desc_histogram`) and any
   host-plane build that folds a sibling subtraction.
4. It really is the wrong-histogram arm, and it leaves no residue: a build
   with the probe set differs from the true histogram, and the next build
   with it cleared is bit-identical to the true one.

Property 4 is the one that would catch a probe that had quietly become a
no-op -- a flag threaded through the launch but dropped before the row loop
would leave every histogram correct and every timing meaningless, and nothing
else in this repository would notice.

What this file deliberately does NOT assert is that the probe's own gather
survived optimization. That is not an executable property: a probe whose
loads had been deleted would produce exactly the same all-but-empty output as
one whose loads had not. It was established instead by compiling a reduced
replica of the row loop to optimized target assembly and reading it, with a
negative control that shows the check can fail; `_hist_rows_step` records
what was read and what the control was.

The device half skips (passing) with no accelerator.
"""

from std.sys import has_accelerator
from std.testing import assert_equal, assert_raises, assert_true, TestSuite
from max.gpu.host import DeviceBuffer, DeviceContext

from mojotrees.binning import BinnedMatrix
from mojotrees.gpu_active_rows import GpuActiveRows, RowRouting
from mojotrees.gpu_tiling import STRATEGY_ATOMIC, query_device_caps
from support import _splitmix64


# --- Fixtures, built the way test_gpu_hist_row_unroll.mojo builds them ---


def _make_data(
    n_rows: Int, n_features: Int, n_bins: Int
) raises -> BinnedMatrix:
    var bins = List[UInt8](capacity=n_rows * n_features)
    for f in range(n_features):
        for r in range(n_rows):
            var v = _splitmix64(UInt64(f * n_rows + r) + 0x51ED2701)
            bins.append(UInt8(Int(v % UInt64(n_bins))))
    return BinnedMatrix(bins^, n_rows, n_features, n_bins)


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


def _identity_slots(
    mut ctx: DeviceContext, n_features: Int
) raises -> DeviceBuffer[DType.int32]:
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var buf = ctx.enqueue_create_buffer[DType.int32](n_features)
        with buf.map_to_host() as host:
            for s in range(n_features):
                host.unsafe_ptr().unsafe_store(s, Int32(s))
        return buf^


# --- The gates ---


def test_the_probe_is_off_on_a_fresh_instance() raises:
    """A default that can be wrong is not a default anyone reads; this asserts
    it rather than trusting the constructor's comment."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var rows = GpuActiveRows(ctx, 64, 2, 32, caps)
        assert_true(not rows.hist_atomic_probe)


def test_the_probe_refuses_without_the_acknowledgment() raises:
    """The acknowledgment argument has no default, so this is the only way a
    call can ask for the probe and be told no. Turning it *off* without the
    acknowledgment is allowed, because a caller putting a wrong histogram
    back into a right one should never be argued with."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var rows = GpuActiveRows(ctx, 64, 2, 32, caps)
        with assert_raises(contains="WRONG histogram"):
            rows.set_histogram_atomic_probe(True, False)
        assert_true(not rows.hist_atomic_probe)
        rows.set_histogram_atomic_probe(False, False)
        assert_true(not rows.hist_atomic_probe)
        rows.set_histogram_atomic_probe(True, True)
        assert_true(rows.hist_atomic_probe)
        rows.set_histogram_atomic_probe(False, True)
        assert_true(not rows.hist_atomic_probe)


def test_the_probe_refuses_the_device_resident_growth_plane() raises:
    """Every histogram of every non-root node of every device-owned tree comes
    through `enqueue_desc_histogram`. The refusal is at the top of it, before
    anything is enqueued, so this needs only well-typed buffers and never a
    live descriptor."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 256
        var n_features = 2
        var n_bins = 32
        var data = _make_data(n_rows, n_features, n_bins)
        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var bins = _upload_bins(ctx, data)
        var feat_dev = _identity_slots(ctx, n_features)
        var rows = GpuActiveRows(ctx, n_rows, n_features, n_bins, caps)
        rows.begin_tree()
        var grad_dev = ctx.enqueue_create_buffer[DType.float32](n_rows)
        var hess_dev = ctx.enqueue_create_buffer[DType.float32](n_rows)
        var cells = 3 * n_features * n_bins
        var pool = ctx.enqueue_create_buffer[DType.int32](2 * cells)

        rows.set_histogram_atomic_probe(True, True)
        with assert_raises(contains="device-resident growth plane"):
            rows.enqueue_desc_histogram(
                2,
                n_rows,
                bins.unsafe_ptr(),
                grad_dev.unsafe_ptr(),
                hess_dev.unsafe_ptr(),
                feat_dev.unsafe_ptr(),
                pool.unsafe_ptr(),
                n_features,
                Float32(1.0),
                Float32(1.0),
                caps,
            )
        rows.set_histogram_atomic_probe(False, True)


def test_the_probe_refuses_to_feed_a_sibling_subtraction() raises:
    """A fused subtraction means a tree is being grown: this node is a child
    and its sibling is about to be derived from it. Corrupting a sibling that
    never ran the probe is worse than corrupting the node that did, because
    the sibling looks correct."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 1024
        var n_features = 2
        var n_bins = 32
        var data = _make_data(n_rows, n_features, n_bins)
        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var bins = _upload_bins(ctx, data)
        var feat_dev = _identity_slots(ctx, n_features)
        var rows = GpuActiveRows(ctx, n_rows, n_features, n_bins, caps)
        rows.begin_tree()
        var grad_dev = ctx.enqueue_create_buffer[DType.float32](n_rows)
        var hess_dev = ctx.enqueue_create_buffer[DType.float32](n_rows)
        var hist_size = n_features * n_bins
        var cells = 3 * hist_size
        var out_dev = ctx.enqueue_create_buffer[DType.int32](2 * cells)
        var part_dev = ctx.enqueue_create_buffer[DType.int32](1)

        _ = rows.partition(
            bins.unsafe_ptr(), 0, 1, 2, RowRouting.numerical(0, n_bins // 2)
        )
        var tiling = rows.range_tiling(
            caps, 2, n_features, STRATEGY_ATOMIC, 1
        )
        rows.set_histogram_atomic_probe(True, True)
        with assert_raises(contains="sibling subtraction"):
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
                subtract=True,
                sub_offset=cells,
            )
        rows.set_histogram_atomic_probe(False, True)


# --- The arm itself ---


def test_the_probe_is_wrong_and_leaves_no_residue() raises:
    """Three builds of the same node, in one process, on one instance.

    The first, with the probe clear, is the truth. The second, with the probe
    set, must differ from it -- specifically its count plane must be short,
    because the counts are exactly what the skipped atomics were accumulating.
    The third, with the probe cleared again, must be bit-identical to the
    first.

    The second assertion is the one worth having. A flag that reached the
    launch and was dropped before the row loop would leave build two equal to
    build one, every timing the tool takes would be a comparison of an arm
    against itself, and the fraction reported would be zero for a reason that
    has nothing to do with atomics. The third catches the opposite failure: a
    probe that left the instance, the shared planes, or the quantized buffer
    in a state the next build inherits.

    The three builds are written as one loop rather than three calls to a
    helper because the helper would have to take both the output buffer and a
    mutable pointer into it, which is the aliasing this language refuses.

    Small integer gradients under unit scales, for the same reason
    test_gpu_hist_row_unroll.mojo uses them: `Int32(round(x * 1.0))` is `x` on
    both sides, so nothing here can be a rounding disagreement.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 4096
        var n_features = 3
        var n_bins = 32
        var data = _make_data(n_rows, n_features, n_bins)
        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var bins = _upload_bins(ctx, data)
        var feat_dev = _identity_slots(ctx, n_features)
        var rows = GpuActiveRows(ctx, n_rows, n_features, n_bins, caps)
        rows.begin_tree()

        var grad_dev = ctx.enqueue_create_buffer[DType.float32](n_rows)
        var hess_dev = ctx.enqueue_create_buffer[DType.float32](n_rows)
        with grad_dev.map_to_host() as host:
            for r in range(n_rows):
                host.unsafe_ptr().unsafe_store(
                    r, Float32(Int(_splitmix64(UInt64(r) + 3) % 401) - 200)
                )
        with hess_dev.map_to_host() as host:
            for r in range(n_rows):
                host.unsafe_ptr().unsafe_store(r, Float32(1.0))

        var hist_size = n_features * n_bins
        var cells = 3 * hist_size
        var out_dev = ctx.enqueue_create_buffer[DType.int32](cells)
        var host_out = ctx.enqueue_create_host_buffer[DType.int32](cells)
        var part_dev = ctx.enqueue_create_buffer[DType.int32](1)
        var tiling = rows.range_tiling(caps, 0, n_features, STRATEGY_ATOMIC, 1)
        assert_equal(tiling.strategy, STRATEGY_ATOMIC)

        var truth = List[Int](capacity=cells)
        var probed = List[Int](capacity=cells)
        var again = List[Int](capacity=cells)
        for arm in range(3):
            rows.set_histogram_atomic_probe(arm == 1, True)
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
            )
            ctx.enqueue_copy(dst_ptr=host_out.unsafe_ptr(), src_buf=out_dev)
            ctx.synchronize()
            for i in range(cells):
                var v = Int(host_out.unsafe_ptr().unsafe_load(i))
                if arm == 0:
                    truth.append(v)
                elif arm == 1:
                    probed.append(v)
                else:
                    again.append(v)
        assert_true(not rows.hist_atomic_probe)

        var true_count = 0
        for b in range(hist_size):
            true_count += truth[2 * hist_size + b]
        # The node is the root, so every active row is counted once per
        # feature. If this is not so, the rest of the test is comparing
        # against something that is not the truth.
        assert_equal(true_count, n_features * n_rows)

        var probe_count = 0
        for b in range(hist_size):
            probe_count += probed[2 * hist_size + b]
        # The skipped atomics are exactly the counts, so the probe's count
        # plane carries almost none of them. "Almost" and not "none": the
        # sink lands in one cell per owned slot, and it is an arbitrary
        # integer. Asserting the plane is short by most of its mass is
        # therefore the strongest thing that is true whatever the race left
        # behind, and it is far more than enough to separate a live probe
        # from a dropped flag.
        assert_true(probe_count * 2 < true_count)

        for i in range(cells):
            assert_equal(again[i], truth[i])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
