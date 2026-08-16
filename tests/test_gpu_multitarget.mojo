"""Multi-target (CatBoost MultiRMSE) derivatives on the device.

Covers the multi-output half of `src/mojotrees/gpu_objectives_native.mojo`
-- the `multi_output` objective state, `_multi_grad_hess_kernel`,
`fill_multi_grad_hess` -- and the one hook it needs in
`gpu_multiclass_batch.mojo`, `GpuClassBatch.fill_multi_output_gradients`.

Why the fixture puts the outputs on different scales
----------------------------------------------------
Every test here fits three outputs whose residuals live three orders of
magnitude apart: output 0 in units of 1, output 1 in units of 1000, output 2
in units of 0.001. That is not decoration, and equal-scale outputs would be
the trap. Two whole classes of defect are invisible on equal-scale outputs
and unmissable here:

  one shared scale     A fixed-point scale is `2^k <= 2^30 / sum|g|`, so
                       outputs whose magnitude sums differ by 1e3 must get
                       scales differing by about 2^10. An implementation that
                       sized every output's lattice from one shared magnitude
                       would agree with a per-output one to the bit on
                       equal-scale outputs, and would quantize the small
                       output's gradients toward zero here -- silently, since
                       the fit would still converge on the large output.
                       `test_multi_output_scales_are_per_output` measures the
                       ratio and would fail at 1.
  an output-index bug  Swapping two identically-scaled planes changes nothing
                       observable, so an equal-scale fixture cannot see a
                       `j_begin + slot` that is off by one or a read of
                       `r * n_outputs + slot` where it meant
                       `r * n_outputs + j`. Here every plane's own magnitude
                       identifies it, and
                       `test_multi_output_planes_are_not_interchangeable`
                       asserts that identification directly.

How the references are built
----------------------------
The reference for output `j` is not a hand-written formula: it is the
**scalar** device path run on output `j`'s column alone -- an ordinary
single-output `GpuObjectiveState` under `SQUARED_ERROR`, its own
`fill_grad_hess`, its own `magnitude_sums`. So each assertion is an equality
between two device paths that must agree bit for bit by construction (same
expression, same operand order, same reduction grid), and a mismatch is a
real defect rather than a tolerance argument. `test_multi_output_matches_
the_cpu` pins the whole thing to `boosting.fill_grad_hess` in Float64 as
well, so the two device paths cannot agree on a wrong answer.

Skips (passing) when no accelerator is present, so the suite stays green on
CPU-only machines.
"""

from std.math import isfinite
from std.sys import has_accelerator
from std.testing import assert_equal, assert_raises, assert_true, TestSuite
from max.gpu.host import DeviceBuffer, DeviceContext

from mojotrees.boosting import (
    CUSTOM,
    POISSON,
    QUANTILE,
    SQUARED_ERROR,
    fill_grad_hess,
)
from mojotrees.gpu_multiclass_batch import GpuClassBatch
from mojotrees.gpu_objectives_native import (
    GpuObjectiveState,
    device_fixed_scale,
    supports_multi_output_objective,
)
from support import _uniform


comptime N_ROWS = 96
comptime N_OUTPUTS = 3


def _f32(x: Float64) -> Float64:
    """`x` rounded to what Float32 can hold, so the host reference and the
    device see bit-identical inputs and a difference is the kernel's."""
    return Float64(Float32(x))


def _unit(j: Int) -> Float64:
    """Output `j`'s unit: the three orders of magnitude the whole file turns
    on. Powers of ten rather than arbitrary factors so the expected scale
    separation is easy to state -- a 1e3 ratio in the magnitude sum is about
    2^10 in the power-of-two fixed-point scale. Every value built from a
    unit passes through `_f32`, so whether the unit itself is exact in
    Float32 does not matter: the host reference and the device see the same
    bits either way.

    The one place the scales are chosen. Setting all three to 1.0 is the
    equal-scale fixture this file exists to argue against, and doing so
    should make `test_multi_output_scales_are_per_output` and
    `test_multi_output_planes_are_not_interchangeable` vacuous rather than
    failing, which is the point."""
    if j == 1:
        return 1000.0
    if j == 2:
        return 0.001
    return 1.0


def _multi_target(n_rows: Int, n_outputs: Int) -> List[Float64]:
    """Row-major target matrix `y[r * n_outputs + j]`, output `j` scaled to
    its own units."""
    var y = List[Float64](capacity=n_rows * n_outputs)
    for r in range(n_rows):
        for j in range(n_outputs):
            var u = _uniform(UInt64(r) * 31 + UInt64(j) * 7 + 11)
            y.append(_f32(_unit(j) * (-2.0 + 4.0 * u)))
    return y^


def _multi_raw(target: List[Float64], n_outputs: Int) -> List[Float64]:
    """Row-major raw scores a residual away from the target, the residual in
    the same units as its output. Never near zero, so a sign error cannot
    hide in a residual that rounds to nothing."""
    var raw = List[Float64](capacity=len(target))
    var n_rows = len(target) // n_outputs
    for r in range(n_rows):
        for j in range(n_outputs):
            var i = r * n_outputs + j
            var u = _uniform(UInt64(r) * 13 + UInt64(j) * 5 + 3)
            var sign = 1.0 if (r + j) % 2 == 0 else -1.0
            raw.append(_f32(target[i] + sign * _unit(j) * (0.25 + 1.5 * u)))
    return raw^


def _column(flat: List[Float64], j: Int, n_outputs: Int) -> List[Float64]:
    """Output `j`'s column of a row-major matrix, as a standalone vector."""
    var n_rows = len(flat) // n_outputs
    var out = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        out.append(flat[r * n_outputs + j])
    return out^


def _weights(n_rows: Int, with_zeros: Bool) -> List[Float64]:
    var w = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        if with_zeros and r % 7 == 0:
            w.append(0.0)
        else:
            w.append(_f32(0.25 + 1.5 * _uniform(UInt64(r) * 17 + 3)))
    return w^


def _assert_close(
    got: Float64, want: Float64, rel: Float64, floor: Float64, what: String
) raises:
    var diff = abs(got - want)
    var bound = floor + rel * abs(want)
    assert_true(
        diff <= bound,
        String(what, ": got ", got, ", want ", want, ", diff ", diff),
    )


def _download_planes(
    ctx: DeviceContext,
    buf: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_planes: Int,
) raises -> List[Float64]:
    """A class-major plane buffer, `plane[c * n_rows + r]`, as Float64.

    Local to the tests rather than a method on the state: nothing in a
    training round ever reads a whole multi-output plane back to the host,
    and adding a downloader to the source would be a device-to-host transfer
    with no caller.
    """
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var n = n_rows * n_planes
        var host = ctx.enqueue_create_host_buffer[DType.float32](n)
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=buf)
        ctx.synchronize()
        var out = List[Float64](capacity=n)
        var src = host.unsafe_ptr()
        for i in range(n):
            out.append(Float64(src.unsafe_load(i)))
        return out^


def _multi_state(
    ctx: DeviceContext,
    target: List[Float64],
    raw: List[Float64],
    weights: List[Float64],
) raises -> GpuObjectiveState:
    """A multi-output state parked at `raw`. The comptime guard keeps the
    device instantiation out of CPU-only builds; only guarded tests reach
    here."""
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var state = GpuObjectiveState(
            ctx, target, weights, N_OUTPUTS, 2048, True
        )
        state.set_raw(ctx, raw)
        return state^


def test_supports_multi_output_objective() raises:
    """Squared error alone is separable over a vector target. Everything
    else is refused by name, which is what keeps a caller who asks for a
    multi-target poisson from silently getting a per-output squared
    error."""
    assert_true(supports_multi_output_objective(SQUARED_ERROR))
    assert_true(not supports_multi_output_objective(POISSON))
    assert_true(not supports_multi_output_objective(QUANTILE))
    assert_true(not supports_multi_output_objective(CUSTOM))
    assert_true(not supports_multi_output_objective(-1))
    assert_true(not supports_multi_output_objective(99))


def test_multi_output_matches_the_cpu() raises:
    """Every output's device plane against `boosting.fill_grad_hess` on that
    output's column, in Float64.

    The outer bound on the whole path: Float32 device arithmetic against a
    Float64 host reference, relative with an absolute floor sized *per
    output* -- an absolute floor good for the unit-1000 output would accept
    any answer at all on the unit-0.001 one, which is the second way an
    equal-scale fixture flatters an implementation.
    """
    comptime if not has_accelerator():
        return
    else:
        var ctx = DeviceContext()
        var target = _multi_target(N_ROWS, N_OUTPUTS)
        var raw = _multi_raw(target, N_OUTPUTS)
        var empty = List[Float64]()
        var state = _multi_state(ctx, target, raw, empty)

        var grad_dev = ctx.enqueue_create_buffer[DType.float32](
            N_ROWS * N_OUTPUTS
        )
        var hess_dev = ctx.enqueue_create_buffer[DType.float32](
            N_ROWS * N_OUTPUTS
        )
        state.fill_multi_grad_hess(
            ctx, SQUARED_ERROR, 0, N_OUTPUTS, grad_dev, hess_dev
        )
        var g = _download_planes(ctx, grad_dev, N_ROWS, N_OUTPUTS)
        var h = _download_planes(ctx, hess_dev, N_ROWS, N_OUTPUTS)

        for j in range(N_OUTPUTS):
            var y_j = _column(target, j, N_OUTPUTS)
            var raw_j = _column(raw, j, N_OUTPUTS)
            var cpu_g = List[Float64]()
            var cpu_h = List[Float64]()
            fill_grad_hess(
                raw_j, y_j, SQUARED_ERROR, empty, 0.9, cpu_g, cpu_h
            )
            # The floor scales with the output's units; the relative bound
            # does not.
            var floor = 1e-6 * _unit(j)
            for r in range(N_ROWS):
                _assert_close(
                    g[j * N_ROWS + r],
                    cpu_g[r],
                    1e-6,
                    floor,
                    String("multi gradient out ", j, " row ", r),
                )
                _assert_close(
                    h[j * N_ROWS + r],
                    cpu_h[r],
                    1e-6,
                    1e-9,
                    String("multi hessian out ", j, " row ", r),
                )


def test_multi_output_plane_equals_the_scalar_device_path() raises:
    """Output `j`'s plane, bit for bit, against a scalar device fit on
    output `j`'s column alone.

    The tightest statement available: both planes come from the same
    expression in the same operand order over the same Float32 inputs, so
    `assert_equal` is the right comparison and any tolerance here would be
    hiding something. This is also what pins the addressing -- a
    `r * n_outputs + j` read that is off by one produces a plausible plane
    that no tolerance test would reject.
    """
    comptime if not has_accelerator():
        return
    else:
        var ctx = DeviceContext()
        var target = _multi_target(N_ROWS, N_OUTPUTS)
        var raw = _multi_raw(target, N_OUTPUTS)
        var empty = List[Float64]()
        var state = _multi_state(ctx, target, raw, empty)

        var grad_dev = ctx.enqueue_create_buffer[DType.float32](
            N_ROWS * N_OUTPUTS
        )
        var hess_dev = ctx.enqueue_create_buffer[DType.float32](
            N_ROWS * N_OUTPUTS
        )
        state.fill_multi_grad_hess(
            ctx, SQUARED_ERROR, 0, N_OUTPUTS, grad_dev, hess_dev
        )
        var g = _download_planes(ctx, grad_dev, N_ROWS, N_OUTPUTS)
        var h = _download_planes(ctx, hess_dev, N_ROWS, N_OUTPUTS)

        for j in range(N_OUTPUTS):
            var y_j = _column(target, j, N_OUTPUTS)
            var raw_j = _column(raw, j, N_OUTPUTS)
            var scalar = GpuObjectiveState(ctx, y_j, empty)
            scalar.set_raw(ctx, raw_j)
            var sg = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
            var sh = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
            scalar.fill_grad_hess(ctx, SQUARED_ERROR, 0.9, sg, sh)
            var want = scalar.download_grad_hess(ctx, sg, sh)
            for r in range(N_ROWS):
                assert_equal(g[j * N_ROWS + r], want[r])
                assert_equal(h[j * N_ROWS + r], want[N_ROWS + r])


def test_multi_output_planes_are_not_interchangeable() raises:
    """Plane `j` carries output `j` and no other, demonstrated by magnitude
    rather than asserted.

    On an equal-scale fixture this test is vacuous: every plane would look
    like every other and a swapped `j_begin + slot` would pass. Here each
    plane's mean magnitude sits in its own output's units, three orders of
    magnitude from its neighbours', so the assertion below is a real
    identification.
    """
    comptime if not has_accelerator():
        return
    else:
        var ctx = DeviceContext()
        var target = _multi_target(N_ROWS, N_OUTPUTS)
        var raw = _multi_raw(target, N_OUTPUTS)
        var empty = List[Float64]()
        var state = _multi_state(ctx, target, raw, empty)

        var grad_dev = ctx.enqueue_create_buffer[DType.float32](
            N_ROWS * N_OUTPUTS
        )
        var hess_dev = ctx.enqueue_create_buffer[DType.float32](
            N_ROWS * N_OUTPUTS
        )
        state.fill_multi_grad_hess(
            ctx, SQUARED_ERROR, 0, N_OUTPUTS, grad_dev, hess_dev
        )
        var g = _download_planes(ctx, grad_dev, N_ROWS, N_OUTPUTS)

        for j in range(N_OUTPUTS):
            var total = 0.0
            for r in range(N_ROWS):
                total += abs(g[j * N_ROWS + r])
            var mean = total / Float64(N_ROWS)
            # The residual is `unit * (0.25 .. 1.75)`, so the mean magnitude
            # is inside `unit * [0.25, 1.75]` and a plane holding a
            # neighbour's output would miss this window by 1000x.
            assert_true(
                mean > 0.2 * _unit(j) and mean < 2.0 * _unit(j),
                String(
                    "plane ",
                    j,
                    " mean magnitude ",
                    mean,
                    " is not in output ",
                    j,
                    "'s units (",
                    _unit(j),
                    ")",
                ),
            )

        # And a batch that starts at output 1 puts output 1 in slot 0, not
        # output 0: the slot-to-output map is `j_begin + slot`.
        var tail_g = ctx.enqueue_create_buffer[DType.float32](N_ROWS * 2)
        var tail_h = ctx.enqueue_create_buffer[DType.float32](N_ROWS * 2)
        state.fill_multi_grad_hess(ctx, SQUARED_ERROR, 1, 2, tail_g, tail_h)
        var tail = _download_planes(ctx, tail_g, N_ROWS, 2)
        for r in range(N_ROWS):
            assert_equal(tail[r], g[1 * N_ROWS + r])
            assert_equal(tail[N_ROWS + r], g[2 * N_ROWS + r])


def test_multi_output_scales_are_per_output() raises:
    """One fixed-point scale per output, each derived from that output's own
    magnitude sum, and each equal to what the scalar device path derives for
    that output alone.

    The scale is `2^k <= 2^30 / sum|g|`, so outputs whose magnitude sums
    differ by 1e3 get scales differing by roughly 2^10 and outputs differing
    by 1e6 by roughly 2^20. The ratio assertions below are the shared-scale
    detector: a single scale shared across the batch makes every ratio
    exactly 1.
    """
    comptime if not has_accelerator():
        return
    else:
        var ctx = DeviceContext()
        var target = _multi_target(N_ROWS, N_OUTPUTS)
        var raw = _multi_raw(target, N_OUTPUTS)
        var empty = List[Float64]()
        var state = _multi_state(ctx, target, raw, empty)

        # Two features and 16 bins: this test never enqueues a histogram, so
        # the batch is here for its derivative planes, its per-slot
        # reduction and its per-slot scales.
        var batch = GpuClassBatch(ctx, N_ROWS, 2, 16, N_OUTPUTS, False)
        batch.fill_multi_output_gradients(state, 0, N_OUTPUTS)
        batch.refresh_scales(N_OUTPUTS)

        for j in range(N_OUTPUTS):
            var y_j = _column(target, j, N_OUTPUTS)
            var raw_j = _column(raw, j, N_OUTPUTS)
            var scalar = GpuObjectiveState(ctx, y_j, empty)
            scalar.set_raw(ctx, raw_j)
            var sg = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
            var sh = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
            scalar.fill_grad_hess(ctx, SQUARED_ERROR, 0.9, sg, sh)
            var mags = scalar.magnitude_sums(ctx, sg, sh)
            # Exact: the batched reduction accumulates the same rows in the
            # same grid-stride order as the single-slot one, and both folds
            # run ascending in Float64, so the totals and therefore the
            # power-of-two scales are the same numbers.
            assert_equal(
                batch.scale_of(j), Float64(device_fixed_scale(mags.grad))
            )
            assert_equal(
                batch.hess_scale_of(j), Float64(device_fixed_scale(mags.hess))
            )

        # The separation itself. Output 0 is 1e3 above output 1's units and
        # 1e3 below output 2's, so the gradient scales run the other way by
        # about 2^10 each. A shared scale makes both of these 1.0.
        var r01 = batch.scale_of(0) / batch.scale_of(1)
        var r21 = batch.scale_of(2) / batch.scale_of(1)
        assert_true(
            r01 > 256.0,
            String("outputs 0 and 1 share a scale; ratio ", r01),
        )
        assert_true(
            r21 > 65536.0,
            String("outputs 2 and 1 share a scale; ratio ", r21),
        )


def test_multi_output_weights() raises:
    """A per-row weight multiplies both derivatives of every output, and a
    zero-weight row is exactly zero in every one of them.

    One weight per row, not per (row, output): a sample weight weights the
    observation. The hessian is the bare weight under squared error, which is
    why `round_has_constant_hessian` refuses a constant-hessian declaration
    on a weighted fit and why a weighted multi-target round stages both
    planes per output rather than the gradient alone.
    """
    comptime if not has_accelerator():
        return
    else:
        var ctx = DeviceContext()
        var target = _multi_target(N_ROWS, N_OUTPUTS)
        var raw = _multi_raw(target, N_OUTPUTS)
        var w = _weights(N_ROWS, True)
        var state = _multi_state(ctx, target, raw, w)

        var grad_dev = ctx.enqueue_create_buffer[DType.float32](
            N_ROWS * N_OUTPUTS
        )
        var hess_dev = ctx.enqueue_create_buffer[DType.float32](
            N_ROWS * N_OUTPUTS
        )
        state.fill_multi_grad_hess(
            ctx, SQUARED_ERROR, 0, N_OUTPUTS, grad_dev, hess_dev
        )
        var g = _download_planes(ctx, grad_dev, N_ROWS, N_OUTPUTS)
        var h = _download_planes(ctx, hess_dev, N_ROWS, N_OUTPUTS)

        for j in range(N_OUTPUTS):
            for r in range(N_ROWS):
                var i = j * N_ROWS + r
                if w[r] == 0.0:
                    assert_equal(g[i], 0.0)
                    assert_equal(h[i], 0.0)
                else:
                    var want = w[r] * (
                        raw[r * N_OUTPUTS + j] - target[r * N_OUTPUTS + j]
                    )
                    _assert_close(
                        g[i],
                        want,
                        1e-6,
                        1e-6 * _unit(j),
                        String("weighted gradient out ", j, " row ", r),
                    )
                    # The hessian IS the weight, and carries no output units.
                    _assert_close(
                        h[i],
                        w[r],
                        1e-6,
                        1e-9,
                        String("weighted hessian out ", j, " row ", r),
                    )


def test_multi_output_raw_update_is_per_output() raises:
    """`update_raw(k=j)` advances output `j`'s slot of the row-major raw
    scores and leaves every other output's alone.

    Reused from the multiclass path with nothing added: an output owns slot
    `r * n_outputs + j` exactly as a class owns `r * n_classes + k`, which is
    what makes K trees per round the shape the rest of the codebase can
    already carry.
    """
    comptime if not has_accelerator():
        return
    else:
        var ctx = DeviceContext()
        var target = _multi_target(N_ROWS, N_OUTPUTS)
        var empty = List[Float64]()
        var state = GpuObjectiveState(
            ctx, target, empty, N_OUTPUTS, 2048, True
        )
        state.init_raw(ctx, [0.0, 0.0, 0.0])

        var leaf_dev = ctx.enqueue_create_buffer[DType.int32](N_ROWS)
        ctx.enqueue_memset(leaf_dev, 0)
        state.update_raw(ctx, leaf_dev, [2.0], 0.5, 1)
        var raw = state.download_raw(ctx)
        assert_equal(len(raw), N_ROWS * N_OUTPUTS)
        for r in range(N_ROWS):
            assert_equal(raw[r * N_OUTPUTS + 0], 0.0)
            _assert_close(
                raw[r * N_OUTPUTS + 1],
                1.0,
                1e-6,
                1e-7,
                String("output 1 raw at row ", r),
            )
            assert_equal(raw[r * N_OUTPUTS + 2], 0.0)


def test_single_output_state_is_unchanged() raises:
    """The `multi_output=False` default leaves the scalar path exactly where
    it was.

    A regression guard on the one constructor this lane touched. `n_rows` is
    `len(target)`, the target buffer is `n_rows` long, and the gradients
    still equal `boosting.fill_grad_hess` to Float32; a multiclass state
    still rejects a non-label target. Everything else in the scalar path is
    unreachable from the new parameter, and `git diff -U0` shows every
    scalar kernel untouched.
    """
    comptime if not has_accelerator():
        return
    else:
        var ctx = DeviceContext()
        var matrix = _multi_target(N_ROWS, N_OUTPUTS)
        var matrix_raw = _multi_raw(matrix, N_OUTPUTS)
        var y = _column(matrix, 0, N_OUTPUTS)
        var raw = _column(matrix_raw, 0, N_OUTPUTS)
        var empty = List[Float64]()

        var state = GpuObjectiveState(ctx, y, empty)
        assert_equal(state.n_rows, N_ROWS)
        assert_equal(state.n_classes, 1)
        assert_true(not state.multi_output)
        state.set_raw(ctx, raw)
        var gd = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
        var hd = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
        state.fill_grad_hess(ctx, SQUARED_ERROR, 0.9, gd, hd)
        var got = state.download_grad_hess(ctx, gd, hd)

        var cpu_g = List[Float64]()
        var cpu_h = List[Float64]()
        fill_grad_hess(raw, y, SQUARED_ERROR, empty, 0.9, cpu_g, cpu_h)
        for r in range(N_ROWS):
            _assert_close(
                got[r], cpu_g[r], 1e-6, 1e-7, String("scalar gradient ", r)
            )
            _assert_close(
                got[N_ROWS + r],
                cpu_h[r],
                1e-6,
                1e-9,
                String("scalar hessian ", r),
            )

        # A multiclass state still validates its labels, so the new branch
        # did not widen the class-label check out of the softmax path.
        with assert_raises():
            _ = GpuObjectiveState(ctx, [0.0, 1.0, 7.5], empty, 3)


def test_multi_output_contract_errors() raises:
    """Every refusal, by name. Nothing the device cannot honour is accepted
    and quietly downgraded."""
    comptime if not has_accelerator():
        return
    else:
        var ctx = DeviceContext()
        var empty = List[Float64]()
        var target = _multi_target(N_ROWS, N_OUTPUTS)
        var raw = _multi_raw(target, N_OUTPUTS)

        # A multi-output state needs at least two outputs, and a target
        # length that is a whole number of rows.
        with assert_raises():
            _ = GpuObjectiveState(ctx, target, empty, 1, 2048, True)
        with assert_raises():
            _ = GpuObjectiveState(
                ctx, [0.0, 1.0, 2.0, 3.0, 4.0], empty, 3, 2048, True
            )
        # A per-row weight vector is per ROW, so its length is n_rows and
        # not n_rows * n_outputs.
        with assert_raises():
            _ = GpuObjectiveState(
                ctx, target, _weights(N_ROWS * N_OUTPUTS, False), N_OUTPUTS,
                2048, True
            )

        var state = _multi_state(ctx, target, raw, empty)
        var gd = ctx.enqueue_create_buffer[DType.float32](N_ROWS * N_OUTPUTS)
        var hd = ctx.enqueue_create_buffer[DType.float32](N_ROWS * N_OUTPUTS)

        # The uncoupled state has no softmax and says so, rather than
        # returning a stale or zeroed probability plane.
        with assert_raises():
            state.refresh_softmax(ctx)
        with assert_raises():
            state.fill_softmax_grad_hess(ctx, 0, gd, hd)
        # And the scalar filler refuses a matrix target rather than reading
        # its first n_rows entries as a column.
        with assert_raises():
            state.fill_grad_hess(ctx, SQUARED_ERROR, 0.9, gd, hd)

        # Objectives with no multi-target form are refused by name, not
        # served as a per-output squared error.
        with assert_raises():
            state.fill_multi_grad_hess(ctx, POISSON, 0, N_OUTPUTS, gd, hd)
        with assert_raises():
            state.fill_multi_grad_hess(ctx, QUANTILE, 0, N_OUTPUTS, gd, hd)
        with assert_raises():
            state.fill_multi_grad_hess(ctx, CUSTOM, 0, N_OUTPUTS, gd, hd)

        # Output ranges.
        with assert_raises():
            state.fill_multi_grad_hess(ctx, SQUARED_ERROR, 0, 0, gd, hd)
        with assert_raises():
            state.fill_multi_grad_hess(ctx, SQUARED_ERROR, -1, 2, gd, hd)
        with assert_raises():
            state.fill_multi_grad_hess(
                ctx, SQUARED_ERROR, 1, N_OUTPUTS, gd, hd
            )

        # A scalar state refuses the multi-target filler.
        var y0 = _column(target, 0, N_OUTPUTS)
        var scalar = GpuObjectiveState(ctx, y0, empty)
        scalar.init_raw(ctx, [0.0])
        with assert_raises():
            scalar.fill_multi_grad_hess(ctx, SQUARED_ERROR, 0, 1, gd, hd)

        # And the batch hook refuses a softmax state, so a multiclass round
        # cannot reach the multi-target kernel by passing the wrong state.
        var labels = List[Float64](capacity=N_ROWS)
        for r in range(N_ROWS):
            labels.append(Float64(r % N_OUTPUTS))
        var soft = GpuObjectiveState(ctx, labels, empty, N_OUTPUTS)
        soft.init_raw(ctx, [0.0, 0.0, 0.0])
        var batch = GpuClassBatch(ctx, N_ROWS, 2, 16, N_OUTPUTS, False)
        with assert_raises():
            batch.fill_multi_output_gradients(soft, 0, N_OUTPUTS)

        # The valid call still works afterwards.
        state.fill_multi_grad_hess(
            ctx, SQUARED_ERROR, 0, N_OUTPUTS, gd, hd
        )
        var g = _download_planes(ctx, gd, N_ROWS, N_OUTPUTS)
        assert_true(isfinite(g[0]))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
