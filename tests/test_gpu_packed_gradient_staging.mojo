"""The packed Int16 gradient staging arm: identical histograms, and a bound
that refuses rather than truncates.

`GpuActiveRows.set_packed_gradients(True)` stages this round's quantized
derivatives as interleaved **Int16** pairs in the same allocation the Int32
pairs occupy, so the histogram's per-row derivative fetch reads half as many
bytes. Nothing downstream narrows: the threadgroup planes, the global planes
and the sibling subtraction are Int32 on both arms.

What is claimed, and it is two different kinds of claim.

**Exact, given a bound.** Sign extension of a 16-bit two's-complement word is
exact, so whenever every row's `Int32(round(x * scale))` lies in
`[-32768, 32767]` the packed arm adds the identical integer to the identical
shared cell and the histogram is bit-for-bit the Int32 arm's. Every comparison
here is `assert_equal` on raw Int32 cells, never a tolerance. This is the same
family of claim as `set_narrow_index`'s and not the same as `set_row_unroll`'s:
the unroll is exact whatever the data, this one is exact *given a checked
bound on the values*, which is why the bound has a test of its own.

**Loud, when the bound fails.** The bound is per row over the whole round --
the buffer is staged once per round and every node of the tree gathers it, so
a bound established on one node's rows would license nothing for the others.
`_quantize_grad_hess_i16_kernel` compares each row's own integer as it stores
it and counts the failures; `_ensure_quantized` reads the counters and raises
before the histogram that would have read a truncated buffer is enqueued. The
refusal tests below are the ones that matter most, because a wrong histogram
for one node on some tree shapes and not others is the failure mode no other
fixture in this repository catches.

Three refusal cases, and the third is not redundant:

- the gradient plane past the bound refuses;
- the hessian plane past the bound refuses on an ordinary round;
- the hessian plane past the bound **does not** refuse on a constant-hessian
  round, because `_hist_rows_step` never gathers the second word there, so a
  hessian that did not fit is a dead byte. A check that refused it anyway
  would reject exactly the rounds this arm serves best.

The host half runs everywhere; it needs no device, because the rule also lives
in `quantized_gradient` where a CPU-only build can reach it.
"""

from std.sys import has_accelerator
from std.testing import assert_equal, assert_raises, assert_true, TestSuite
from max.gpu.host import DeviceBuffer, DeviceContext

from mojotrees.binning import BinnedMatrix
from mojotrees.gpu_active_rows import (
    QUANT_SOURCE_FLOAT,
    QUANT_SOURCE_INT32,
    QUANT_SOURCE_PACKED16,
    STAGE16_FLAG_WORDS,
    GpuActiveRows,
)
from mojotrees.gpu_tiling import (
    STRATEGY_ATOMIC,
    STRATEGY_TILED,
    query_device_caps,
    histogram_bin_capacity,
    histogram_shared_bytes,
)
from mojotrees.quantized_gradient import (
    INT16_LIMIT,
    check_int16_staging,
    int16_staging_fits,
)
from support import _splitmix64


# --- The host half: the bound, without a device -------------------------


def test_the_staging_bound_is_the_int16_range() raises:
    """`max_abs * scale <= 32767`, inclusive at the edge.

    Inclusive because 32,767 is a representable Int16 and the arm stores it
    unchanged; an exclusive comparison would refuse a round that is exactly
    correct. The three points below straddle the edge by one lattice unit on
    each side, which is the smallest distinguishable step and therefore the
    only interesting place to test a threshold.
    """
    assert_true(int16_staging_fits(Float64(INT16_LIMIT), 1.0))
    assert_true(int16_staging_fits(Float64(INT16_LIMIT) - 1.0, 1.0))
    assert_true(not int16_staging_fits(Float64(INT16_LIMIT) + 1.0, 1.0))
    # The scale is the other factor and binds the same way.
    assert_true(int16_staging_fits(1.0, Float64(INT16_LIMIT)))
    assert_true(not int16_staging_fits(2.0, Float64(INT16_LIMIT)))
    assert_true(int16_staging_fits(0.0, 1.0))


def test_the_staging_bound_refuses_and_says_why() raises:
    """`check_int16_staging` is the refusal, and it does not clamp.

    A clamped row is a gradient the fit never had, on one plane, for one
    round, indistinguishable from a data change by any fixture here. The whole
    value of this arm is that it is bit-identical or it is nothing, so the
    only two outcomes are "returns" and "raises".
    """
    check_int16_staging(Float64(INT16_LIMIT), 1.0, "gradient")
    with assert_raises():
        check_int16_staging(Float64(INT16_LIMIT) + 1.0, 1.0, "gradient")
    with assert_raises():
        check_int16_staging(Float64(INT16_LIMIT) + 1.0, 1.0, "hessian")


def test_the_staging_bound_refuses_a_degenerate_input() raises:
    """A non-finite magnitude or a non-positive scale is not a bound that
    failed, it is a round that should never have derived a scale, and the
    predicate refuses rather than answering `False`. Answering `False` would
    route it into the same message as an honest overflow and send the caller
    looking at the row count."""
    with assert_raises():
        _ = int16_staging_fits(-1.0, 1.0)
    with assert_raises():
        _ = int16_staging_fits(1.0, 0.0)
    with assert_raises():
        _ = int16_staging_fits(1.0, -1.0)


def test_the_quant_source_codes_are_distinct() raises:
    """`use_quant` carries three states down eight `enqueue_function`
    argument lists that were not widened to hold a fourth argument. That is
    only sound while the codes are distinct and the Float32 arm keeps zero,
    which every launch site still tests as `!= 0`."""
    assert_equal(QUANT_SOURCE_FLOAT, 0)
    assert_true(QUANT_SOURCE_INT32 != QUANT_SOURCE_FLOAT)
    assert_true(QUANT_SOURCE_PACKED16 != QUANT_SOURCE_FLOAT)
    assert_true(QUANT_SOURCE_PACKED16 != QUANT_SOURCE_INT32)
    assert_equal(STAGE16_FLAG_WORDS, 2)


# --- Shared device fixtures ---------------------------------------------


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


def _plane(
    mut ctx: DeviceContext, values: List[Float64]
) raises -> DeviceBuffer[DType.float32]:
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var buf = ctx.enqueue_create_buffer[DType.float32](len(values))
        with buf.map_to_host() as host:
            for r in range(len(values)):
                host.unsafe_ptr().unsafe_store(r, Float32(values[r]))
        return buf^


# --- The identity half ---------------------------------------------------


def _identity_case(n_bins: Int, n_features: Int, n_rows: Int) raises:
    """The two staged widths, on one shape, over every strategy and group
    rung the device holds, with the Int32 arm as the reference.

    The values are integers in `[-200, 200]` for the gradient and `[1, 97]`
    for the hessian under unit scales, so every quantized word is an exact
    small integer and the per-row bound holds by construction with four orders
    of magnitude of room. That is deliberate: this half is testing that the
    narrowed load reaches the same bytes, not that the bound works, and mixing
    the two would leave a failure ambiguous. The refusal half below is where
    the bound is exercised.

    The reference is the Int32 quantized arm rather than the Float32 planes,
    because those two are already held identical by
    `test_gpu_hist_row_unroll.mojo` and what is new here is only the width.
    The Float32 arm is swept anyway, so a change that broke the three-way
    agreement would still show.
    """
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var data = _make_data(n_rows, n_features, n_bins)
        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var bins = _upload_bins(ctx, data)
        var feat_dev = _identity_slots(ctx, n_features)
        var rows = GpuActiveRows(ctx, n_rows, n_features, n_bins, caps)
        rows.begin_tree()

        var gvals = List[Float64](capacity=n_rows)
        var hvals = List[Float64](capacity=n_rows)
        for r in range(n_rows):
            gvals.append(Float64(Int(_splitmix64(UInt64(r) + 3) % 401) - 200))
            hvals.append(Float64(Int(_splitmix64(UInt64(r) + 77) % 97) + 1))
        var grad_dev = _plane(ctx, gvals)
        var hess_dev = _plane(ctx, hvals)

        var node = 0
        var window = rows.range_of(node)
        assert_true(window.count() > 0)

        var hist_size = n_features * n_bins
        var cells = 3 * hist_size
        var out_dev = ctx.enqueue_create_buffer[DType.int32](cells)
        var host_out = ctx.enqueue_create_host_buffer[DType.int32](cells)
        var cap = histogram_bin_capacity(n_bins)
        var reference = List[Int](capacity=cells)
        var have_reference = False
        var arms = 0

        var strategies = [STRATEGY_ATOMIC, STRATEGY_TILED]
        var groups = [1, 4]
        for si in range(len(strategies)):
            var tiling = rows.range_tiling(
                caps, node, n_features, strategies[si], 1 << 20
            )
            var part_cells = tiling.partial_cells
            if part_cells < 1:
                part_cells = 1
            var part_dev = ctx.enqueue_create_buffer[DType.int32](
                3 * part_cells
            )
            for gi in range(len(groups)):
                var group = groups[gi]
                if (
                    histogram_shared_bytes(cap, group)
                    > caps.max_shared_memory_per_block
                ):
                    continue
                rows.set_feature_group(group)
                # Three sources: Float32 planes, Int32 pairs, Int16 pairs.
                for src in range(3):
                    rows.set_quantized_gradients(src > 0)
                    rows.set_packed_gradients(src == 2)
                    assert_equal(rows.packed_gradients_on(), src == 2)
                    rows.enqueue_range_histogram(
                        tiling,
                        node,
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
                    ctx.enqueue_copy(
                        dst_ptr=host_out.unsafe_ptr(), src_buf=out_dev
                    )
                    ctx.synchronize()
                    var got = host_out.unsafe_ptr()
                    if not have_reference:
                        for i in range(cells):
                            reference.append(Int(got.unsafe_load(i)))
                        have_reference = True
                    else:
                        for i in range(cells):
                            assert_equal(
                                Int(got.unsafe_load(i)), reference[i]
                            )
                    arms += 1
        # Two strategies by at least one group rung by three sources. A
        # silently skipped sweep would otherwise pass on one arm.
        assert_true(arms >= 6)
        rows.set_feature_group(1)
        rows.set_packed_gradients(False)
        rows.set_quantized_gradients(False)


def test_the_packed_width_builds_the_same_histogram() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        _identity_case(32, 3, 1500)
        _identity_case(64, 17, 2600)
        _identity_case(256, 5, 1100)


def test_the_published_width_matches_the_arm() raises:
    """`staged_gradient_bytes_per_row` is the number this lane is judged on,
    so it is checked against the arms rather than trusted. Four cases: both
    widths crossed with the hessian plane elided and not."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var rows = GpuActiveRows(ctx, 256, 2, 32, caps)
        rows.set_quantized_gradients(True)

        rows.set_constant_hessian(False)
        rows.set_packed_gradients(False)
        assert_equal(rows.staged_gradient_bytes_per_row(), 8)
        rows.set_packed_gradients(True)
        assert_equal(rows.staged_gradient_bytes_per_row(), 4)

        rows.set_constant_hessian(True)
        if not rows.const_hessian_allowed:
            print("skipped: MOJOTREES_CONST_HESSIAN=0")
            return
        assert_equal(rows.staged_gradient_bytes_per_row(), 2)
        rows.set_packed_gradients(False)
        assert_equal(rows.staged_gradient_bytes_per_row(), 4)


# --- The refusal half ----------------------------------------------------


def _stage_once(
    mut ctx: DeviceContext,
    g_value: Float64,
    h_value: Float64,
    g_scale: Float32,
    h_scale: Float32,
    const_hess: Bool,
) raises -> Bool:
    """Stage one round through the packed arm and report whether it was
    allowed. Returns `True` when the histogram was enqueued, and raises out of
    `enqueue_range_histogram` when the bound refused it -- which is the whole
    point: the refusal has to happen at the staging pass, before the histogram
    that would have read a truncated buffer is queued behind it.

    Declared as returning a value so a caller cannot mistake "raised" for
    "returned"; the caller wraps it in `assert_raises` or asserts the result.
    """
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var n_rows = 512
        var n_features = 2
        var n_bins = 32
        var data = _make_data(n_rows, n_features, n_bins)
        var caps = query_device_caps(ctx)
        var bins = _upload_bins(ctx, data)
        var feat_dev = _identity_slots(ctx, n_features)
        var rows = GpuActiveRows(ctx, n_rows, n_features, n_bins, caps)
        rows.begin_tree()
        rows.set_quantized_gradients(True)
        rows.set_packed_gradients(True)
        rows.set_constant_hessian(const_hess)

        var gvals = List[Float64](capacity=n_rows)
        var hvals = List[Float64](capacity=n_rows)
        for _ in range(n_rows):
            gvals.append(g_value)
            hvals.append(h_value)
        var grad_dev = _plane(ctx, gvals)
        var hess_dev = _plane(ctx, hvals)

        var hist_size = n_features * n_bins
        var out_dev = ctx.enqueue_create_buffer[DType.int32](3 * hist_size)
        var tiling = rows.range_tiling(
            caps, 0, n_features, STRATEGY_ATOMIC, 1 << 20
        )
        var part_dev = ctx.enqueue_create_buffer[DType.int32](3)
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
            g_scale,
            h_scale,
        )
        ctx.synchronize()
        return True


def test_a_gradient_past_the_bound_ends_the_fit() raises:
    """40,000 at unit scale is one integer the Int16 word cannot hold, and the
    truncation it would suffer is silent: the wrapped value is a perfectly
    ordinary-looking gradient of the wrong sign. This is the case the arm
    exists to refuse."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var ctx = DeviceContext()
        assert_true(
            _stage_once(ctx, 1000.0, 1.0, Float32(1.0), Float32(1.0), False)
        )
        with assert_raises():
            _ = _stage_once(
                ctx, 40000.0, 1.0, Float32(1.0), Float32(1.0), False
            )


def test_a_hessian_past_the_bound_ends_an_ordinary_round() raises:
    """The hessian word is gathered on any round that does not declare a
    constant hessian, so it is bound by the same inequality and refused the
    same way. Reached by moving the hessian scale rather than the value,
    which is the factor `set_scale_refresh`'s headroom moves in practice."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var ctx = DeviceContext()
        with assert_raises():
            _ = _stage_once(
                ctx, 1.0, 1.0, Float32(1.0), Float32(1000000.0), False
            )


def test_a_dead_hessian_word_does_not_end_a_constant_hessian_round() raises:
    """The negative control, and the one that would be missed by a check
    written the obvious way.

    On a constant-hessian round `_hist_rows_step` never gathers the second
    word, so a hessian that did not fit is a dead byte in a buffer and not a
    wrong histogram. A single overflow flag shared by both planes would refuse
    this round, and it would refuse it *silently correctly* -- the arithmetic
    would look right, and the arm would simply stop working on exactly the
    objective class (unweighted squared error) it serves best. Two counters
    are what make this pass, so this test is what shows the second counter is
    load-bearing rather than decoration.

    The same shape with the declaration off is the previous test, which
    raises; the pair is the control.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var probe = GpuActiveRows(ctx, 16, 1, 32, caps)
        probe.set_constant_hessian(True)
        if not probe.const_hessian_allowed:
            print("skipped: MOJOTREES_CONST_HESSIAN=0")
            return
        assert_true(
            _stage_once(
                ctx, 1.0, 1.0, Float32(1.0), Float32(1000000.0), True
            )
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
