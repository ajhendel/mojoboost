"""The fused per-round gradient upload, checked at the bytes and at the
histogram.

`GpuHistogramBuilder` now holds its two derivative planes in one
`2 * n_rows` Float32 device allocation laid out as `[grad | hess]`, with
`grad_dev` and `hess_dev` as `create_sub_buffer` windows onto it, and
`upload_staged` moves both planes with a single `enqueue_copy` of the whole
allocation. `set_fused_gradient_upload(False)` puts the two per-plane copies
back at run time, so both arms are reachable in one process.

Every failure this file is looking for is silent. A window that ignored its
offset, a fused copy that landed the hessian plane where the gradient plane
belongs, a window whose parent was freed when the builder moved -- none of
them raise. Each one produces a quietly wrong histogram, a quietly worse
model, and no error. So the checks here are on exact values and not on
tolerances, and the first of them looks at the device's bytes rather than at
anything derived from them.

Three kinds of check, because they fail in different ways.

1. **The bytes the device holds.** The staged values are small integers, so
   their `Float32` images are exact and a `Float64` comparison against the
   original list is an equality and not an approximation. The gradient and
   hessian lists are deliberately disjoint in value (gradients negative,
   hessians positive), so a copy that wrote one plane over the other, or
   wrote a plane at the wrong offset, cannot pass by coincidence. Read back
   three ways: through the parent allocation, which is what the fused arm
   writes, and through each window, which is what every kernel and every
   consumer in `gpu_objectives_native` and `gpu_multiclass_batch` reads.

2. **The two arms against each other, at the histogram.** Accumulation is
   fixed-point `Int32` and the scales are derived from the same values in
   both arms, so the two arms must agree on every raw cell, not to a
   tolerance. This is the check that a cross-arm difference in the *upload*
   would show up in the thing the trainer actually consumes.

3. **A builder moved into a `List`.** The trainer's builders move, and a
   window whose parent allocation did not survive the move would read freed
   memory. Uploading and reading back after the move is what establishes
   that the parent's lifetime is the builder's.

The whole file skips (passing) with no accelerator.
"""

from std.sys import has_accelerator
from std.testing import assert_equal, assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojotrees.binning import BinnedMatrix
from mojotrees.histogram_gpu import GpuHistogramBuilder

from support import _splitmix64


# --- Fixtures ---


def _make_data(
    n_rows: Int, n_features: Int, n_bins: Int
) raises -> BinnedMatrix:
    """A column-major binned matrix of pseudorandom bins, built the way the
    other GPU histogram tests build one."""
    var bins = List[UInt8](capacity=n_rows * n_features)
    for f in range(n_features):
        for r in range(n_rows):
            var v = _splitmix64(UInt64(f * n_rows + r) + 0x0DF1A5E3)
            bins.append(UInt8(Int(v % UInt64(n_bins))))
    return BinnedMatrix(bins^, n_rows, n_features, n_bins)


def _grad_of(n_rows: Int, salt: Int) raises -> List[Float64]:
    """Gradients: small negative integers, every one exactly representable
    as a `Float32`. Disjoint in sign from the hessians below, so a plane
    written over the other plane is visible in the value itself."""
    var out = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        out.append(Float64(-1 - ((r + salt) % 97)))
    return out^


def _hess_of(n_rows: Int, salt: Int) raises -> List[Float64]:
    """Hessians: small positive integers, exactly representable, and never
    equal to any gradient."""
    var out = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        out.append(Float64(1 + ((r * 3 + salt) % 61)))
    return out^


def _readback(
    mut builder: GpuHistogramBuilder, window: Int
) raises -> List[Float32]:
    """The device's own Float32, read back with a copy this file issues.

    `window` picks what is read: 0 the whole `[grad | hess]` allocation, 1
    the gradient window, 2 the hessian window. Reading the parent and reading
    the two windows are different questions -- the first is what the fused
    copy wrote, the second is what a kernel sees -- and a mistake in the
    window offsets shows only in the second.
    """
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var n = builder.n_rows if window > 0 else 2 * builder.n_rows
        var out = List[Float32](capacity=n)
        for _ in range(n):
            out.append(Float32(0))
        if window == 0:
            builder.ctx.enqueue_copy(
                dst_ptr=out.unsafe_ptr(), src_buf=builder.gh_dev
            )
        elif window == 1:
            builder.ctx.enqueue_copy(
                dst_ptr=out.unsafe_ptr(), src_buf=builder.grad_dev
            )
        else:
            builder.ctx.enqueue_copy(
                dst_ptr=out.unsafe_ptr(), src_buf=builder.hess_dev
            )
        builder.ctx.synchronize()
        return out^


def _assert_planes_landed(
    mut builder: GpuHistogramBuilder,
    grad: List[Float64],
    hess: List[Float64],
) raises:
    """Both planes are on the device, at their own offsets, with their own
    values, read through the parent and through each window."""
    var n = len(grad)
    var whole = _readback(builder, 0)
    assert_equal(len(whole), 2 * n)
    for r in range(n):
        assert_equal(whole[r], Float32(grad[r]))
        assert_equal(whole[n + r], Float32(hess[r]))

    var g_view = _readback(builder, 1)
    var h_view = _readback(builder, 2)
    assert_equal(len(g_view), n)
    assert_equal(len(h_view), n)
    for r in range(n):
        # The gradient window is the first half and the hessian window is the
        # second, which is the property an offset counted in bytes rather
        # than elements would break.
        assert_equal(g_view[r], Float32(grad[r]))
        assert_equal(h_view[r], Float32(hess[r]))

    # The windows as *pointers*, which is a second question and the one the
    # histogram kernels ask: they are handed `grad_dev.unsafe_ptr()` and
    # `hess_dev.unsafe_ptr()`, never the buffers. A `unsafe_ptr()` that
    # answered the parent's base for both windows would leave every check
    # above passing and would feed the hessian plane the gradients.
    assert_true(
        builder.grad_dev.unsafe_ptr().unsafe_offset(n)
        == builder.hess_dev.unsafe_ptr()
    )


# --- 1. The bytes the device holds ---


def test_each_upload_arm_lands_both_planes_at_their_own_offsets() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 96
        var data = _make_data(n_rows, 3, 16)
        var builder = GpuHistogramBuilder(data)

        # The default arm: one copy of the whole allocation.
        assert_true(builder.fused_gradient_upload())
        var g0 = _grad_of(n_rows, 0)
        var h0 = _hess_of(n_rows, 0)
        builder.upload_gradients(g0, h0)
        _assert_planes_landed(builder, g0, h0)

        # The arm that shipped: one copy per plane, into the two windows.
        # Different values, so a stale device buffer would fail rather than
        # pass on the previous round's bytes.
        builder.set_fused_gradient_upload(False)
        assert_true(not builder.fused_gradient_upload())
        var g1 = _grad_of(n_rows, 41)
        var h1 = _hess_of(n_rows, 41)
        builder.upload_gradients(g1, h1)
        _assert_planes_landed(builder, g1, h1)

        # And back, so the switch is a run-time arm in both directions and
        # not a one-way latch.
        builder.set_fused_gradient_upload(True)
        var g2 = _grad_of(n_rows, 7)
        var h2 = _hess_of(n_rows, 7)
        builder.upload_gradients(g2, h2)
        _assert_planes_landed(builder, g2, h2)


# --- 2. The two arms against each other, at the histogram ---


def _root_histogram(
    mut builder: GpuHistogramBuilder,
    grad: List[Float64],
    hess: List[Float64],
    fused: Bool,
) raises -> List[Float64]:
    """The root node's gradient, hessian and count planes, flattened, built
    under the requested upload arm."""
    builder.set_fused_gradient_upload(fused)
    builder.upload_gradients(grad, hess)
    builder.begin_tree()
    var hist = builder.build_leaf(0)
    var out = List[Float64](capacity=3 * len(hist.grad))
    for i in range(len(hist.grad)):
        out.append(hist.grad[i])
    for i in range(len(hist.hess)):
        out.append(hist.hess[i])
    for i in range(len(hist.count)):
        out.append(Float64(hist.count[i]))
    return out^


def test_both_upload_arms_build_the_same_histogram() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 1100
        var data = _make_data(n_rows, 5, 32)
        var builder = GpuHistogramBuilder(data)
        var grad = _grad_of(n_rows, 13)
        var hess = _hess_of(n_rows, 13)

        # The shipped arm is the reference, and the fused arm is compared to
        # it, in that order, so a regression is reported against what used to
        # run rather than against the new thing.
        var split_arm = _root_histogram(builder, grad, hess, False)
        var fused_arm = _root_histogram(builder, grad, hess, True)
        assert_equal(len(split_arm), len(fused_arm))
        for i in range(len(split_arm)):
            # Exact: the same fixed-point Int32 accumulation of the same
            # Float32 values under the same scales, dequantized the same way.
            assert_equal(split_arm[i], fused_arm[i])

        # A histogram that summed nothing would compare equal to another
        # histogram that summed nothing, so establish that these are not
        # empty: the count plane totals `n_rows` per feature.
        var cells = len(split_arm) // 3
        var counted = 0.0
        for i in range(2 * cells, 3 * cells):
            counted += split_arm[i]
        assert_equal(counted, Float64(n_rows * data.n_features))

        # Which plane the kernel read through which window, end to end. The
        # gradients are all negative and the hessians all positive, so a
        # kernel that had been handed the parent's base for both windows
        # would produce a non-negative gradient plane and a negative hessian
        # plane. Both arms share the windows, so a cross-arm comparison
        # cannot see this and only the signs can.
        for i in range(cells):
            assert_true(split_arm[i] <= 0.0)
            assert_true(split_arm[cells + i] >= 0.0)


# --- 3. A builder moved into a `List` ---


def test_the_windows_survive_the_builder_moving() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 96
        var data = _make_data(n_rows, 3, 16)
        var grad = _grad_of(n_rows, 5)
        var hess = _hess_of(n_rows, 5)

        var held = List[GpuHistogramBuilder]()
        held.append(GpuHistogramBuilder(data))
        # Uploading only after the move is the point: the windows were taken
        # in a constructor whose frame is gone, and the parent allocation
        # they view is a field of the value that moved.
        held[0].upload_gradients(grad, hess)
        _assert_planes_landed(held[0], grad, hess)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
