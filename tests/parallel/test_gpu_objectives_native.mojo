"""Device-side objective kernels: CPU reference parity, finite differences,
weights, clipping, and the device-resident raw-score update.

`tests/test_gpu_objectives.mojo` covers GPU *training* per objective, end to
end, where the gradients are still computed on the host. This file covers
`src/mojotrees/gpu_objectives_native.mojo`, which moves that computation onto
the device, and so tests the derivatives directly rather than through a
trained model:

  cpu parity      every device objective against `fill_grad_hess`, the
                  Float64 CPU implementation, row by row
  derivatives     each gradient against a central difference of the loss it
                  is supposed to be the derivative of, and each hessian
                  against a second central difference where the objective's
                  hessian *is* the second derivative (huber, quantile, L1 and
                  MAPE use LightGBM's `hess = w` instead, and poisson inflates
                  its hessian by `max_delta_step`, so those are excluded and
                  pinned against the CPU reference instead)
  weights         weighted derivatives, including zero-weight rows, which must
                  produce an exactly zero gradient and hessian
  softmax         multiclass probabilities and per-class derivatives against
                  `_softmax_inplace` + `_fill_softmax_grad_hess`
  raw update      `raw[r] += lr * value[leaf[r]]` from a device leaf array,
                  including rows the device never routed
  scale           the device magnitude reduction against a host sum, and the
                  fixed-point scale derived from it
  stability       raw scores far enough out to overflow Float32, where the
                  exponent clamp is the difference between a finite gradient
                  and an infinity
  contract        the errors: custom objectives stay on the host, unknown
                  codes, wrong lengths, out-of-range classes and node tables

Every input is rounded through Float32 before the CPU reference runs
(`_f32`), so the two backends see bit-identical inputs and the comparison
measures kernel arithmetic rather than input rounding. Tolerances are
relative with an absolute floor, sized for Float32 (see `_assert_close`);
the device carries Float64 nowhere, because Apple GPUs have no Float64.

Skips (passing) when no accelerator is present so the suite stays green on
CPU-only machines.
"""

from std.math import exp, isfinite, log
from std.sys import has_accelerator
from std.testing import assert_equal, assert_raises, assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojotrees.boosting import (
    BINARY_LOGISTIC,
    CROSS_ENTROPY,
    CUSTOM,
    FAIR,
    GAMMA,
    HUBER,
    L1,
    MAPE,
    POISSON,
    QUANTILE,
    SQUARED_ERROR,
    TWEEDIE,
    _fill_softmax_grad_hess,
    _softmax_inplace,
    fill_grad_hess,
)
from mojotrees.gpu_objectives_native import (
    GpuObjectiveState,
    device_fixed_scale,
    supports_device_objective,
)


def _splitmix64(state: UInt64) -> UInt64:
    var z = state + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _uniform(counter: UInt64) -> Float64:
    return Float64(_splitmix64(counter) >> 11) * (1.0 / 9007199254740992.0)


def _f32(x: Float64) -> Float64:
    """`x` rounded to what Float32 can hold. Inputs built through this are
    identical on both backends, so a difference in the output is the
    kernel's, not the upload's."""
    return Float64(Float32(x))


def _assert_close(
    got: Float64, want: Float64, rel: Float64, floor: Float64, what: String
) raises:
    var diff = abs(got - want)
    var bound = floor + rel * abs(want)
    assert_true(
        diff <= bound,
        String(what, ": got ", got, ", want ", want, ", diff ", diff),
    )


def _all_objectives() -> List[Int]:
    """Every objective the device kernels implement, in the order the kernel
    branches on them."""
    return [
        SQUARED_ERROR,
        BINARY_LOGISTIC,
        CROSS_ENTROPY,
        POISSON,
        GAMMA,
        TWEEDIE,
        HUBER,
        QUANTILE,
        L1,
        MAPE,
        FAIR,
    ]


def _alpha_for(objective: Int) -> Float64:
    """The objective's parameter, rounded through Float32 so both backends
    use exactly the same value: quantile/huber `alpha`, tweedie's variance
    power, fair's `c`."""
    if objective == TWEEDIE:
        return _f32(1.5)
    if objective == FAIR:
        return _f32(1.0)
    return _f32(0.9)


def _offset(counter: UInt64) -> Float64:
    """A residual `raw - target` that is never near zero and never near the
    huber transition at 0.9, so the piecewise objectives are tested inside
    their branches rather than on a boundary a Float32 rounding step could
    flip. Both huber regions are covered: half the rows land in [0.2, 0.6],
    half in [1.2, 2.0]."""
    var u = _uniform(counter)
    var sign = 1.0 if (counter & 1) == 0 else -1.0
    if (counter & 2) == 0:
        return sign * (0.2 + 0.4 * u)
    return sign * (1.2 + 0.8 * u)


def _target_for(objective: Int, n: Int) -> List[Float64]:
    var y = List[Float64](capacity=n)
    for r in range(n):
        var u = _uniform(UInt64(r) * 7 + 11)
        if objective == BINARY_LOGISTIC:
            y.append(1.0 if u > 0.5 else 0.0)
        elif objective == CROSS_ENTROPY:
            y.append(_f32(u))
        elif objective == POISSON or objective == TWEEDIE:
            # Nonnegative counts, whole numbers, exact in Float32.
            y.append(Float64(Int(u * 5.0)))
        elif objective == GAMMA:
            y.append(_f32(0.5 + 2.5 * u))
        elif objective == MAPE:
            # Away from zero on both sides, so the label weight
            # `1 / max(1, |y|)` is exercised above and below its floor.
            y.append(_f32(0.4 + 2.0 * u) * (1.0 if u > 0.5 else -1.0))
        else:
            y.append(_f32(-2.0 + 4.0 * u))
    return y^


def _raw_for(objective: Int, target: List[Float64]) -> List[Float64]:
    var raw = List[Float64](capacity=len(target))
    for r in range(len(target)):
        if objective == BINARY_LOGISTIC or objective == CROSS_ENTROPY:
            raw.append(_f32(-3.0 + 6.0 * _uniform(UInt64(r) * 13 + 5)))
        elif objective == POISSON or objective == GAMMA or (
            objective == TWEEDIE
        ):
            # The exp-link objectives, kept in a range where exp(raw) is an
            # ordinary number; the clamp gets its own test.
            raw.append(_f32(-1.0 + 2.0 * _uniform(UInt64(r) * 13 + 5)))
        else:
            raw.append(_f32(target[r] + _offset(UInt64(r) * 3 + 1)))
    return raw^


def _weights(n: Int, with_zeros: Bool) -> List[Float64]:
    var w = List[Float64](capacity=n)
    for r in range(n):
        if with_zeros and r % 7 == 0:
            w.append(0.0)
        else:
            w.append(_f32(0.25 + 1.5 * _uniform(UInt64(r) * 17 + 3)))
    return w^


def _device_grad_hess(
    ctx: DeviceContext,
    target: List[Float64],
    raw: List[Float64],
    weights: List[Float64],
    objective: Int,
    alpha: Float64,
) raises -> List[Float64]:
    """One round of device-side derivatives, downloaded: gradients first,
    then hessians. The comptime guard keeps the device instantiation out of
    CPU-only builds; only guarded tests call this."""
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var n = len(target)
        var state = GpuObjectiveState(ctx, target, weights)
        state.set_raw(ctx, raw)
        var grad_dev = ctx.enqueue_create_buffer[DType.float32](n)
        var hess_dev = ctx.enqueue_create_buffer[DType.float32](n)
        state.fill_grad_hess(ctx, objective, alpha, grad_dev, hess_dev)
        return state.download_grad_hess(ctx, grad_dev, hess_dev)


def _row_loss(
    objective: Int, raw: Float64, y: Float64, alpha: Float64
) -> Float64:
    """The per-row loss whose first derivative in `raw` each objective's
    gradient is. Written out here rather than reused from boosting.mojo so
    the finite-difference check has an independent reference; squared error
    is the half-square LightGBM differentiates, not the plain square its
    reported metric uses."""
    var d = raw - y
    if objective == BINARY_LOGISTIC or objective == CROSS_ENTROPY:
        # log(1 + exp(raw)) - y * raw, in the overflow-safe form.
        var soft: Float64
        if raw >= 0.0:
            soft = raw + log(1.0 + exp(-raw))
        else:
            soft = log(1.0 + exp(raw))
        return soft - y * raw
    if objective == POISSON:
        return exp(raw) - y * raw
    if objective == GAMMA:
        return raw + y * exp(-raw)
    if objective == TWEEDIE:
        return (
            -y * exp((1.0 - alpha) * raw) / (1.0 - alpha)
            + exp((2.0 - alpha) * raw) / (2.0 - alpha)
        )
    if objective == FAIR:
        var t = abs(d) / alpha
        return alpha * alpha * (t - log(1.0 + t))
    if objective == HUBER:
        var ad = abs(d)
        if ad <= alpha:
            return 0.5 * ad * ad
        return alpha * (ad - 0.5 * alpha)
    if objective == QUANTILE:
        return (1.0 - alpha) * d if d >= 0.0 else -alpha * d
    if objective == L1:
        return abs(d)
    if objective == MAPE:
        var m = abs(y)
        var lw = 1.0 / m if m > 1.0 else 1.0
        return abs(d) * lw
    return 0.5 * d * d


def test_supports_device_objective() raises:
    """The split between the device kernels and the host path. Custom
    objectives are the one built-in code that stays on the host: the callback
    is host code over host lists, and there is no device image of it."""
    var objectives = _all_objectives()
    for i in range(len(objectives)):
        assert_true(supports_device_objective(objectives[i]))
    assert_true(not supports_device_objective(CUSTOM))
    assert_true(not supports_device_objective(-1))
    assert_true(not supports_device_objective(99))


def test_device_grad_hess_matches_cpu() raises:
    """Every device objective against `fill_grad_hess`, row by row, on inputs
    both backends see identically. Float32 against Float64, so the bound is
    relative with an absolute floor rather than exact."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n = 1_024
        var empty = List[Float64]()
        var ctx = DeviceContext()
        var objectives = _all_objectives()
        for i in range(len(objectives)):
            var objective = objectives[i]
            var alpha = _alpha_for(objective)
            var target = _target_for(objective, n)
            var raw = _raw_for(objective, target)
            var got = _device_grad_hess(
                ctx, target, raw, empty, objective, alpha
            )

            var grad = List[Float64]()
            var hess = List[Float64]()
            fill_grad_hess(raw, target, objective, empty, alpha, grad, hess)

            assert_equal(len(got), 2 * n)
            for r in range(n):
                _assert_close(
                    got[r], grad[r], 1e-5, 1e-6,
                    String("grad objective ", objective, " row ", r),
                )
                _assert_close(
                    got[n + r], hess[r], 1e-5, 1e-6,
                    String("hess objective ", objective, " row ", r),
                )
                assert_true(got[n + r] >= 0.0)


def test_device_grad_hess_weights() raises:
    """Sample weights, the built-in convention: both derivatives scale by the
    row weight, so a zero-weight row contributes an exactly zero gradient and
    an exactly zero hessian and is invisible to every histogram."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n = 512
        var weights = _weights(n, True)
        var zeros = 0
        for r in range(n):
            if weights[r] == 0.0:
                zeros += 1
        assert_true(zeros > 0)

        var ctx = DeviceContext()
        var objectives = _all_objectives()
        for i in range(len(objectives)):
            var objective = objectives[i]
            var alpha = _alpha_for(objective)
            var target = _target_for(objective, n)
            var raw = _raw_for(objective, target)
            var got = _device_grad_hess(
                ctx, target, raw, weights, objective, alpha
            )

            var grad = List[Float64]()
            var hess = List[Float64]()
            fill_grad_hess(
                raw, target, objective, weights, alpha, grad, hess
            )
            for r in range(n):
                if weights[r] == 0.0:
                    assert_equal(got[r], 0.0)
                    assert_equal(got[n + r], 0.0)
                _assert_close(
                    got[r], grad[r], 1e-5, 1e-6,
                    String("weighted grad objective ", objective, " row ", r),
                )
                _assert_close(
                    got[n + r], hess[r], 1e-5, 1e-6,
                    String("weighted hess objective ", objective, " row ", r),
                )


def test_device_gradient_finite_difference() raises:
    """Each device gradient against a central difference of the loss it is
    the derivative of. This is the check that would catch a transcription
    error the CPU reference shares, which row-by-row parity cannot."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n = 256
        var h = 1e-3
        var empty = List[Float64]()
        var ctx = DeviceContext()
        var objectives = _all_objectives()
        for i in range(len(objectives)):
            var objective = objectives[i]
            var alpha = _alpha_for(objective)
            var target = _target_for(objective, n)
            var raw = _raw_for(objective, target)
            var got = _device_grad_hess(
                ctx, target, raw, empty, objective, alpha
            )
            for r in range(n):
                var up = _row_loss(objective, raw[r] + h, target[r], alpha)
                var down = _row_loss(objective, raw[r] - h, target[r], alpha)
                var fd = (up - down) / (2.0 * h)
                _assert_close(
                    got[r], fd, 1e-3, 1e-4,
                    String("fd grad objective ", objective, " row ", r),
                )


def test_device_hessian_finite_difference() raises:
    """Second central differences, for the objectives whose hessian is the
    true second derivative.

    Huber, quantile, L1 and MAPE are excluded because LightGBM's hessian for
    them is the row weight, not the (zero) curvature of a piecewise-linear
    loss, and poisson is excluded because its hessian is inflated by
    `max_delta_step`. Those four are pinned against the CPU reference in
    `test_device_grad_hess_matches_cpu` instead, which is where the
    convention lives.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var smooth: List[Int] = [
            SQUARED_ERROR, BINARY_LOGISTIC, CROSS_ENTROPY, GAMMA, TWEEDIE,
            FAIR,
        ]
        var n = 256
        var h = 1e-3
        var empty = List[Float64]()
        var ctx = DeviceContext()
        for i in range(len(smooth)):
            var objective = smooth[i]
            var alpha = _alpha_for(objective)
            var target = _target_for(objective, n)
            var raw = _raw_for(objective, target)
            var got = _device_grad_hess(
                ctx, target, raw, empty, objective, alpha
            )
            for r in range(n):
                var up = _row_loss(objective, raw[r] + h, target[r], alpha)
                var mid = _row_loss(objective, raw[r], target[r], alpha)
                var down = _row_loss(objective, raw[r] - h, target[r], alpha)
                var fd = (up - 2.0 * mid + down) / (h * h)
                _assert_close(
                    got[n + r], fd, 5e-3, 1e-3,
                    String("fd hess objective ", objective, " row ", r),
                )


def test_device_softmax_matches_cpu() raises:
    """Multiclass softmax: the device computes the probabilities once per
    round and each class's one-vs-rest derivatives from them, exactly as the
    host trainer does."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n = 512
        var n_classes = 4
        var labels = List[Int](capacity=n)
        var target = List[Float64](capacity=n)
        for r in range(n):
            var k = Int(_uniform(UInt64(r) * 5 + 2) * Float64(n_classes))
            if k >= n_classes:
                k = n_classes - 1
            labels.append(k)
            target.append(Float64(k))
        var raw = List[Float64](capacity=n * n_classes)
        for i in range(n * n_classes):
            raw.append(_f32(-2.0 + 4.0 * _uniform(UInt64(i) * 29 + 7)))
        var weights = _weights(n, True)

        var ctx = DeviceContext()
        var state = GpuObjectiveState(ctx, target, weights, n_classes)
        state.set_raw(ctx, raw)
        state.refresh_softmax(ctx)
        var grad_dev = ctx.enqueue_create_buffer[DType.float32](n)
        var hess_dev = ctx.enqueue_create_buffer[DType.float32](n)

        var prob = raw.copy()
        for r in range(n):
            _softmax_inplace(prob, r * n_classes, n_classes)

        for k in range(n_classes):
            state.fill_softmax_grad_hess(ctx, k, grad_dev, hess_dev)
            var got = state.download_grad_hess(ctx, grad_dev, hess_dev)
            var grad = List[Float64]()
            var hess = List[Float64]()
            _fill_softmax_grad_hess(
                prob, labels, k, n_classes, weights, grad, hess
            )
            for r in range(n):
                _assert_close(
                    got[r], grad[r], 1e-4, 1e-6,
                    String("softmax grad class ", k, " row ", r),
                )
                _assert_close(
                    got[n + r], hess[r], 1e-4, 1e-6,
                    String("softmax hess class ", k, " row ", r),
                )
                if weights[r] == 0.0:
                    assert_equal(got[r], 0.0)
                    assert_equal(got[n + r], 0.0)


def test_device_raw_update_matches_host() raises:
    """The device-resident prediction update against the host loop it
    replaces. Rows the device never routed keep their scores: that is the
    out-of-bag case, and the caller has to handle it."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n = 300
        var n_nodes = 7
        var learning_rate = _f32(0.1)
        var target = _target_for(SQUARED_ERROR, n)
        var base = _f32(0.25)
        var values = List[Float64](capacity=n_nodes)
        for i in range(n_nodes):
            values.append(_f32(-1.0 + 0.4 * Float64(i)))

        var ctx = DeviceContext()
        var state = GpuObjectiveState(ctx, target)
        state.init_raw(ctx, [base])
        var leaf_dev = ctx.enqueue_create_buffer[DType.int32](n)
        var leaves = List[Int](capacity=n)
        with leaf_dev.map_to_host() as host:
            var dst = host.unsafe_ptr()
            for r in range(n):
                var node: Int
                if r % 23 == 0:
                    node = -1  # out of bag: never routed
                elif r % 31 == 0:
                    node = n_nodes + 3  # past the tree's node count
                else:
                    node = 3 + (r % 4)
                leaves.append(node)
                dst.unsafe_store(r, Int32(node))

        state.update_raw(ctx, leaf_dev, values, learning_rate)
        var got = state.download_raw(ctx)
        assert_equal(len(got), n)
        var skipped = 0
        for r in range(n):
            var want = base
            if leaves[r] >= 0 and leaves[r] < n_nodes:
                want += learning_rate * values[leaves[r]]
            else:
                skipped += 1
            _assert_close(
                got[r], want, 1e-6, 1e-7, String("raw update row ", r)
            )
        assert_true(skipped > 0)

        # A second tree accumulates on top of the first, which is what a
        # boosting round actually does.
        state.update_raw(ctx, leaf_dev, values, learning_rate)
        var twice = state.download_raw(ctx)
        for r in range(n):
            var step = got[r] - base
            _assert_close(
                twice[r], base + 2.0 * step, 1e-6, 1e-7,
                String("second raw update row ", r),
            )


def test_device_raw_update_is_per_class() raises:
    """A multiclass round grows one tree per class, so the update has to
    touch that class's scores and no other."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n = 128
        var n_classes = 3
        var target = List[Float64](capacity=n)
        for r in range(n):
            target.append(Float64(r % n_classes))
        var base: List[Float64] = [_f32(-0.5), _f32(0.0), _f32(0.5)]

        var ctx = DeviceContext()
        var state = GpuObjectiveState(ctx, target, List[Float64](), n_classes)
        state.init_raw(ctx, base)
        var leaf_dev = ctx.enqueue_create_buffer[DType.int32](n)
        with leaf_dev.map_to_host() as host:
            var dst = host.unsafe_ptr()
            for r in range(n):
                dst.unsafe_store(r, Int32(1))
        var values: List[Float64] = [_f32(0.0), _f32(2.0)]

        state.update_raw(ctx, leaf_dev, values, _f32(0.5), 1)
        var got = state.download_raw(ctx)
        assert_equal(len(got), n * n_classes)
        for r in range(n):
            _assert_close(
                got[r * n_classes + 0], base[0], 1e-6, 1e-7, "class 0 held"
            )
            _assert_close(
                got[r * n_classes + 1], base[1] + _f32(0.5) * _f32(2.0),
                1e-6, 1e-7, "class 1 advanced",
            )
            _assert_close(
                got[r * n_classes + 2], base[2], 1e-6, 1e-7, "class 2 held"
            )


def test_magnitude_sums_match_host() raises:
    """The device magnitude reduction against a host sum of the same values,
    and the fixed-point scale derived from it. The scale is what every GPU
    histogram quantizes by, so an error here would move every split."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n = 4_099  # not a multiple of the reduction's grid stride
        var alpha = _alpha_for(POISSON)
        var target = _target_for(POISSON, n)
        var raw = _raw_for(POISSON, target)

        var ctx = DeviceContext()
        var state = GpuObjectiveState(ctx, target)
        state.set_raw(ctx, raw)
        var grad_dev = ctx.enqueue_create_buffer[DType.float32](n)
        var hess_dev = ctx.enqueue_create_buffer[DType.float32](n)
        state.fill_grad_hess(ctx, POISSON, alpha, grad_dev, hess_dev)
        var sums = state.magnitude_sums(ctx, grad_dev, hess_dev)
        var values = state.download_grad_hess(ctx, grad_dev, hess_dev)

        var g_total = 0.0
        var h_total = 0.0
        for r in range(n):
            g_total += abs(values[r])
            h_total += abs(values[n + r])
        assert_true(g_total > 0.0)
        _assert_close(sums.grad, g_total, 1e-4, 1e-6, "gradient magnitude")
        _assert_close(sums.hess, h_total, 1e-4, 1e-6, "hessian magnitude")

        # The reduction is a fixed grid stride over a fixed block count, so
        # repeating it is bit-identical, which is what keeps the scale (and
        # every histogram derived from it) deterministic.
        var again = state.magnitude_sums(ctx, grad_dev, hess_dev)
        assert_equal(again.grad, sums.grad)
        assert_equal(again.hess, sums.hess)

        # The scale puts the whole magnitude sum at half the Int32 range, so
        # no partial sum over any subset of rows can overflow.
        var scale = Float64(device_fixed_scale(sums.grad))
        _assert_close(
            scale * sums.grad, Float64(1 << 30), 1e-6, 1e-6, "fixed scale"
        )


def test_device_fixed_scale_edges() raises:
    """A magnitude sum below the floor still yields a finite positive scale;
    a non-finite one is refused rather than quantized into garbage."""
    var floored = device_fixed_scale(0.0)
    assert_true(floored > 0.0)
    assert_equal(floored, device_fixed_scale(1e-13))
    var inf = 1.0 / 0.0
    with assert_raises():
        _ = device_fixed_scale(inf)
    with assert_raises():
        _ = device_fixed_scale(inf - inf)


def test_device_exp_clamp_keeps_gradients_finite() raises:
    """Raw scores far enough out that Float32 `exp` would overflow. The CPU
    path in Float64 has room the device does not, so the kernels clamp the
    exponent argument; the result is a large finite gradient instead of an
    infinity that the fixed-point scale would then reject.

    This is a deliberate divergence from the CPU path and the only one these
    kernels add beyond the Float32 carrier.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n = 64
        var exp_link: List[Int] = [POISSON, GAMMA, TWEEDIE]
        var empty = List[Float64]()
        var ctx = DeviceContext()
        var target = List[Float64](capacity=n)
        for r in range(n):
            target.append(Float64(r % 4))
        var raw = List[Float64](capacity=n)
        for r in range(n):
            raw.append(500.0 if r % 2 == 0 else -500.0)

        for i in range(len(exp_link)):
            var objective = exp_link[i]
            var got = _device_grad_hess(
                ctx, target, raw, empty, objective, _alpha_for(objective)
            )
            for r in range(n):
                assert_true(
                    isfinite(got[r]),
                    String("clamped grad objective ", objective),
                )
                assert_true(
                    isfinite(got[n + r]) and got[n + r] >= 0.0,
                    String("clamped hess objective ", objective),
                )

        # The saturating side of the logistic loss: the gradient tends to
        # the label distance and the hessian to the floor, never to a NaN
        # from 0 * inf.
        var labels = List[Float64](capacity=n)
        for r in range(n):
            labels.append(1.0 if r % 2 == 0 else 0.0)
        var logistic = _device_grad_hess(
            ctx, labels, raw, empty, BINARY_LOGISTIC,
            _alpha_for(BINARY_LOGISTIC),
        )
        for r in range(n):
            assert_true(isfinite(logistic[r]))
            assert_true(logistic[n + r] > 0.0)
            _assert_close(
                logistic[r], 0.0, 0.0, 1e-6, String("saturated grad row ", r)
            )


def test_device_objective_contract_errors() raises:
    """The refusals. Every one of these would otherwise be a silently wrong
    training run rather than a failure."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n = 32
        var target = _target_for(SQUARED_ERROR, n)
        var ctx = DeviceContext()

        var empty = List[Float64]()
        var short: List[Float64] = [1.0, 2.0]

        # Construction validates what it uploads once and can never recheck.
        with assert_raises():
            _ = GpuObjectiveState(ctx, empty, empty)
        with assert_raises():
            _ = GpuObjectiveState(ctx, target, short)
        with assert_raises():
            _ = GpuObjectiveState(ctx, target, empty, 0)
        var negative_w = List[Float64](capacity=n)
        for _ in range(n):
            negative_w.append(-1.0)
        with assert_raises():
            _ = GpuObjectiveState(ctx, target, negative_w)
        # Multiclass labels must be whole numbers inside the class range.
        with assert_raises():
            _ = GpuObjectiveState(ctx, target, empty, 3)

        var state = GpuObjectiveState(ctx, target)
        var grad_dev = ctx.enqueue_create_buffer[DType.float32](n)
        var hess_dev = ctx.enqueue_create_buffer[DType.float32](n)

        # Gradients before the raw scores exist would read an uninitialized
        # buffer.
        with assert_raises():
            state.fill_grad_hess(ctx, SQUARED_ERROR, 0.9, grad_dev, hess_dev)

        state.init_raw(ctx, [0.0])
        # Custom objectives stay on the host path, and unknown codes are not
        # quietly treated as squared error.
        with assert_raises():
            state.fill_grad_hess(ctx, CUSTOM, 0.9, grad_dev, hess_dev)
        with assert_raises():
            state.fill_grad_hess(ctx, 99, 0.9, grad_dev, hess_dev)
        # Softmax needs a multiclass state.
        with assert_raises():
            state.refresh_softmax(ctx)
        with assert_raises():
            state.fill_softmax_grad_hess(ctx, 0, grad_dev, hess_dev)
        with assert_raises():
            state.init_raw(ctx, [0.0, 1.0])
        with assert_raises():
            state.set_raw(ctx, [0.0])

        var leaf_dev = ctx.enqueue_create_buffer[DType.int32](n)
        ctx.enqueue_memset(leaf_dev, 0)
        with assert_raises():
            state.update_raw(ctx, leaf_dev, empty, 0.1)
        with assert_raises():
            state.update_raw(ctx, leaf_dev, [1.0], 0.1, 1)

        # A tree with more nodes than the node-value table holds is refused
        # rather than truncated.
        var small = GpuObjectiveState(ctx, target, empty, 1, 4)
        small.init_raw(ctx, [0.0])
        with assert_raises():
            small.update_raw(ctx, leaf_dev, [0.0, 1.0, 2.0, 3.0, 4.0], 0.1)

        # And the valid calls still work afterwards.
        state.fill_grad_hess(ctx, SQUARED_ERROR, 0.9, grad_dev, hess_dev)
        state.update_raw(ctx, leaf_dev, [0.5], 0.1)
        var raw = state.download_raw(ctx)
        assert_equal(len(raw), n)
        _assert_close(raw[0], 0.05, 1e-6, 1e-7, "raw after valid update")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
