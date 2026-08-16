"""The two decomposition arms this lane added, which build WRONG histograms.

`GpuActiveRows.set_histogram_probe_mode` selects one of two instruments that
cut the histogram phase where the atomics probe could not:

- `HIST_PROBE_NO_GATHER` drops the row load, the quantized gradient load and
  the bin gather and keeps the three shared atomics, the trip count, the
  shared zeroing, the barrier and the whole flush.
- `HIST_PROBE_EMPTY` returns from every kernel of the build as soon as its
  threadgroup planes exist.

**This file does not time anything.** Timing is
`tools/probe_hist_atomic_fraction.mojo`, which interleaves all four arms in
one process because this machine's device timings drift several-fold between
time windows. What this file establishes is the properties an instrument has
to have before anyone is allowed to run it, and it is the sibling of
`tests/test_gpu_hist_atomic_probe.mojo`, which establishes the same
properties for the arm that lane built:

1. Both are off on a freshly constructed `GpuActiveRows`, and so is the
   predicate the refusals are written against.
2. Neither can be turned on without an acknowledgment written into the call,
   an unknown mode is refused, and the atomics arm cannot be reached through
   this door.
3. The arms are mutually exclusive in both directions, so there is no state
   in which two of them are live and the launch has to pick.
4. Both refuse the two entry points through which a histogram becomes a tree:
   the device-resident growth plane and any host-plane build that folds a
   sibling subtraction.
5. Each really is its own wrong-histogram arm, in a way that is specific to
   what that arm removes, and each leaves no residue.

Property 5 is the one worth having and it is written differently for each
arm, because a generic "the output differs" would pass for a probe that had
quietly become the wrong instrument:

- The gather-free arm keeps every atomic, so its count plane must still hold
  exactly `n_features * n_rows` -- the same total mass as the truth -- while
  being distributed over the wrong bins. A flag that reached the launch and
  was dropped would leave the histogram bit-identical to the truth; an arm
  that had lost its atomics as well would leave the mass short. The assertion
  catches both.
- The empty arm runs no accumulation and no reduction at all, so on the
  atomic strategy, where the output is zeroed before the build, its histogram
  must be **entirely zero**. That is the strongest statement available and it
  is exactly what "the kernel returned before doing anything" means.

What this file deliberately does NOT assert is that either arm's kept work
survived optimization. That is not an executable property: an arm whose work
had been deleted would produce the same wrong output as one whose work had
not. It was established instead by compiling reduced replicas to optimized
target assembly and reading them, each with a negative control that shows the
check can fail. `_hist_rows_step` and `_hist_probe_empty_mark` record what was
read and what each control was; the empty arm's first version failed its own
control, which is why its shape is what it is.

The device half skips (passing) with no accelerator.
"""

from std.sys import has_accelerator
from std.testing import assert_equal, assert_raises, assert_true, TestSuite
from max.gpu.host import DeviceBuffer, DeviceContext

from mojotrees.binning import BinnedMatrix
from mojotrees.gpu_active_rows import (
    HIST_PROBE_EMPTY,
    HIST_PROBE_NO_ATOMICS,
    HIST_PROBE_NO_GATHER,
    HIST_PROBE_OFF,
    GpuActiveRows,
    RowRouting,
)
from mojotrees.gpu_tiling import STRATEGY_ATOMIC, query_device_caps
from support import _splitmix64


# --- Fixtures, built the way test_gpu_hist_atomic_probe.mojo builds them ---


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


def test_every_arm_is_off_on_a_fresh_instance() raises:
    """A default that can be wrong is not a default anyone reads. This
    asserts the mode field and the predicate the refusals are written
    against, because a predicate that answered False while an arm was live
    would disable every refusal at once."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var rows = GpuActiveRows(ctx, 64, 2, 32, caps)
        assert_equal(rows.hist_probe_mode, HIST_PROBE_OFF)
        assert_true(not rows.hist_atomic_probe)
        assert_true(not rows.histogram_probe_active())


def test_the_modes_refuse_without_the_acknowledgment() raises:
    """The acknowledgment argument has no default, so this is the only way a
    call can ask for an arm and be told no. Selecting `HIST_PROBE_OFF`
    without it is allowed, because a caller putting a wrong histogram back
    into a right one should never be argued with."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var rows = GpuActiveRows(ctx, 64, 2, 32, caps)
        with assert_raises(contains="WRONG histogram"):
            rows.set_histogram_probe_mode(HIST_PROBE_NO_GATHER, False)
        assert_equal(rows.hist_probe_mode, HIST_PROBE_OFF)
        with assert_raises(contains="WRONG histogram"):
            rows.set_histogram_probe_mode(HIST_PROBE_EMPTY, False)
        assert_equal(rows.hist_probe_mode, HIST_PROBE_OFF)
        rows.set_histogram_probe_mode(HIST_PROBE_OFF, False)
        assert_equal(rows.hist_probe_mode, HIST_PROBE_OFF)
        rows.set_histogram_probe_mode(HIST_PROBE_NO_GATHER, True)
        assert_equal(rows.hist_probe_mode, HIST_PROBE_NO_GATHER)
        assert_true(rows.histogram_probe_active())
        rows.set_histogram_probe_mode(HIST_PROBE_EMPTY, True)
        assert_equal(rows.hist_probe_mode, HIST_PROBE_EMPTY)
        rows.set_histogram_probe_mode(HIST_PROBE_OFF, True)
        assert_true(not rows.histogram_probe_active())


def test_an_unknown_mode_is_refused_and_so_is_the_atomics_arm() raises:
    """Two refusals with one reason. An out-of-range mode would otherwise be
    stored, passed to the kernel, matched by no branch, and run the shipping
    arm under an instrument's label -- which is the failure this repository
    has already had once with an environment-variable arm. And the atomics
    arm has its own field and its own setter, so admitting it here would
    leave the two fields able to disagree about which arm is live."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var rows = GpuActiveRows(ctx, 64, 2, 32, caps)
        with assert_raises(contains="unknown histogram probe mode"):
            rows.set_histogram_probe_mode(7, True)
        assert_equal(rows.hist_probe_mode, HIST_PROBE_OFF)
        with assert_raises(contains="unknown histogram probe mode"):
            rows.set_histogram_probe_mode(-1, True)
        assert_equal(rows.hist_probe_mode, HIST_PROBE_OFF)
        with assert_raises(contains="unknown histogram probe mode"):
            rows.set_histogram_probe_mode(HIST_PROBE_NO_ATOMICS, True)
        assert_equal(rows.hist_probe_mode, HIST_PROBE_OFF)
        assert_true(not rows.hist_atomic_probe)


def test_the_arms_are_exclusive_in_both_directions() raises:
    """Two arms at once is a kernel nobody described, and the launch would
    have to pick one silently. Both setters refuse instead, and the refusal
    is symmetric so neither ordering can sneak past it."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var rows = GpuActiveRows(ctx, 64, 2, 32, caps)

        rows.set_histogram_atomic_probe(True, True)
        with assert_raises(contains="atomics probe is already selected"):
            rows.set_histogram_probe_mode(HIST_PROBE_NO_GATHER, True)
        assert_equal(rows.hist_probe_mode, HIST_PROBE_OFF)
        # Turning an arm OFF while another is live is always allowed: it can
        # only move the instance toward the correct kernel.
        rows.set_histogram_probe_mode(HIST_PROBE_OFF, True)
        rows.set_histogram_atomic_probe(False, True)

        rows.set_histogram_probe_mode(HIST_PROBE_EMPTY, True)
        with assert_raises(contains="another histogram probe arm"):
            rows.set_histogram_atomic_probe(True, True)
        assert_true(not rows.hist_atomic_probe)
        rows.set_histogram_atomic_probe(False, True)
        rows.set_histogram_probe_mode(HIST_PROBE_OFF, True)
        assert_true(not rows.histogram_probe_active())


def test_the_modes_refuse_the_device_resident_growth_plane() raises:
    """Every histogram of every non-root node of every device-owned tree comes
    through `enqueue_desc_histogram`. The refusal is at the top of it, before
    anything is enqueued, so this needs only well-typed buffers and never a
    live descriptor. Both arms are checked, because the refusal is written
    against a predicate and a predicate that missed one arm would let that
    arm grow a whole tree out of garbage."""
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

        for arm in range(2):
            var mode = HIST_PROBE_NO_GATHER if arm == 0 else HIST_PROBE_EMPTY
            rows.set_histogram_probe_mode(mode, True)
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
            rows.set_histogram_probe_mode(HIST_PROBE_OFF, True)


def test_the_modes_refuse_to_feed_a_sibling_subtraction() raises:
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
        var tiling = rows.range_tiling(caps, 2, n_features, STRATEGY_ATOMIC, 1)
        for arm in range(2):
            var mode = HIST_PROBE_NO_GATHER if arm == 0 else HIST_PROBE_EMPTY
            rows.set_histogram_probe_mode(mode, True)
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
            rows.set_histogram_probe_mode(HIST_PROBE_OFF, True)


# --- The arms themselves ---


def test_the_arms_are_wrong_in_their_own_way_and_leave_no_residue() raises:
    """Four builds of the same node, in one process, on one instance.

    Build one, with every arm clear, is the truth. Build two is the
    gather-free arm; build three is the empty arm; build four clears
    everything again and must be bit-identical to the first.

    Each wrong build is asserted against the specific thing its own arm
    removes, because a generic inequality would pass for the wrong
    instrument:

    - The gather-free arm keeps every shared atomic and the whole flush, so
      its count plane must carry the SAME total mass as the truth,
      `n_features * n_rows`, spread over the wrong bins. Equal mass and an
      unequal vector is a much narrower claim than "it differs" and it is the
      claim that separates a live arm from three failures at once: a flag
      dropped before the row loop (mass equal, vector equal), an arm that
      lost its atomics too (mass short), and an arm whose trip count changed
      (mass wrong in either direction).
    - The empty arm runs no accumulation and no reduction, and the atomic
      strategy zeroes the output before the build, so its histogram must be
      entirely zero -- every cell of every plane.

    The final build catches the opposite failure from both: an arm that left
    the instance, the shared planes, or the quantized buffer in a state the
    next build inherits.

    The four builds are written as one loop rather than four calls to a
    helper because the helper would have to take both the output buffer and a
    mutable pointer into it, which is the aliasing this language refuses.

    Small integer gradients under unit scales, for the reason
    test_gpu_hist_atomic_probe.mojo uses them: `Int32(round(x * 1.0))` is `x`
    on both sides, so nothing here can be a rounding disagreement.
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
        # The empty arm's all-zero assertion is a statement about the atomic
        # strategy, where the output is zeroed before the build and nothing
        # else writes it. If this ever resolves to the tiled strategy the
        # assertion below is about a different kernel pair and the reduction
        # would be reading partials nothing wrote this build.
        assert_equal(tiling.strategy, STRATEGY_ATOMIC)

        var truth = List[Int](capacity=cells)
        var gathered = List[Int](capacity=cells)
        var emptied = List[Int](capacity=cells)
        var again = List[Int](capacity=cells)
        for arm in range(4):
            var mode = HIST_PROBE_OFF
            if arm == 1:
                mode = HIST_PROBE_NO_GATHER
            elif arm == 2:
                mode = HIST_PROBE_EMPTY
            rows.set_histogram_probe_mode(mode, True)
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
                    gathered.append(v)
                elif arm == 2:
                    emptied.append(v)
                else:
                    again.append(v)
        assert_true(not rows.histogram_probe_active())

        var true_count = 0
        for b in range(hist_size):
            true_count += truth[2 * hist_size + b]
        # The node is the root, so every active row is counted once per
        # feature. If this is not so, the rest of the test is comparing
        # against something that is not the truth.
        assert_equal(true_count, n_features * n_rows)

        # The gather-free arm: same mass, wrong distribution.
        var gathered_count = 0
        for b in range(hist_size):
            gathered_count += gathered[2 * hist_size + b]
        assert_equal(gathered_count, true_count)
        var differs = False
        for i in range(cells):
            if gathered[i] != truth[i]:
                differs = True
        assert_true(differs)

        # The empty arm: nothing ran, so nothing was written.
        for i in range(cells):
            assert_equal(emptied[i], 0)

        # And neither left anything behind.
        for i in range(cells):
            assert_equal(again[i], truth[i])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
