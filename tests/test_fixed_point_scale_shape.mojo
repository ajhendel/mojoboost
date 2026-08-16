"""The power-of-two fixed-point scale rule, checked against its claims.

WHAT THIS FILE IS FOR
---------------------
`quantized_gradient.fixed_point_scale_pow2` is the package's one statement of
the magnitude-sum scale rule: take `2^30 / sum|g|` and round it *down* to a
power of two. The GPU histogram builder uses it, the device objective path
uses it, the distributed exchange uses it, and a CPU fixed-point histogram is
meant to call or copy it. That is a lot of weight on one function, and the
function's docstring makes four claims that are checkable without a device
and without a benchmark. This file checks them:

  1. The returned scale is *exactly* a power of two, at Float32 and at
     Float64, with no mantissa bits set.
  2. It never exceeds `2^30 / T` (the safe direction) and never falls below
     half of it (the bound on what flooring costs). Together these are the
     overflow proof and the one-bit precision cost, asserted rather than
     argued.
  3. The two roundings it claims to delete are actually deleted: the Float64
     reciprocal used by the download path is exact, and the Float32 product
     inside the quantization kernel is exact.
  4. Sibling subtraction is exact under both scale shapes, because it is a
     property of the integer accumulator and not of the scale.

Every arithmetic claim here is checked with `==` on floats or on their bit
patterns. Nothing in this file uses a tolerance, because none of the claims
is approximate; a tolerance would turn "exact" into "close" and there would
be nothing left to test.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
No device is opened and no kernel is instantiated, so this runs in the CPU
set. The rule is host arithmetic over two Float64s; putting its test behind
an accelerator would mean a CPU-only runner could not enforce the rule that a
CPU implementation is supposed to build to.

No timing, and no fit. The accuracy argument for the power-of-two shape is
complete without a measurement, and the speed effect is unmeasured in either
direction (`histogram_gpu.set_scale_shape` says so). This file measures
nothing.

THE ORACLE
----------
`_pow2_floor_by_loop` recomputes the floor by repeated exact halving and
doubling, so the assertions compare the shipped bit-field extraction against
an independent computation rather than against a restatement of itself.
Multiplying and dividing a float by two is exact for every value in the range
these tests use, so the oracle is exact too.
"""

from std.math import round
from std.testing import assert_equal, assert_raises, assert_true, TestSuite

from mojotrees.quantized_gradient import (
    DEFAULT_SCALE_SHAPE,
    FIXED_ONE,
    MAGNITUDE_FLOOR,
    SCALE_SHAPE_ARBITRARY,
    SCALE_SHAPE_POW2,
    describe_scale_shape,
    fixed_point_scale,
    fixed_point_scale_arbitrary,
    fixed_point_scale_pow2,
    fixed_point_scale_shaped,
    floored_magnitude,
    largest_power_of_two_at_most,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pow2_floor_by_loop(x: Float64) -> Float64:
    """The largest power of two at or below `x`, by exact halving/doubling.

    The independent oracle. Every step multiplies or divides by two, which is
    exact in binary floating point away from the subnormal range, so this
    computes the true answer rather than an approximation of it. It is
    deliberately not the bit-field spelling under test.
    """
    var p = 1.0
    if x >= 1.0:
        while p * 2.0 <= x:
            p = p * 2.0
    else:
        while p > x:
            p = p / 2.0
    return p


def _is_power_of_two_f64(x: Float64) -> Bool:
    """Whether `x` is a positive normal Float64 with no mantissa bits set,
    which for a positive normal is exactly "is a power of two"."""
    if x <= 0.0:
        return False
    var bits = x.to_bits().cast[DType.uint64]()
    var biased = (bits >> 52) & 0x7FF
    var mantissa = bits & 0xF_FFFF_FFFF_FFFF
    return biased != 0 and biased != 0x7FF and mantissa == 0


def _is_power_of_two_f32(x: Float32) -> Bool:
    """The same question at Float32 width. Asked separately because the
    device kernels receive the scale at this width, so 'exactly a power of
    two' has to be true of the Float32 the kernel multiplies by and not only
    of the Float64 it was derived from."""
    if x <= 0.0:
        return False
    var bits = x.to_bits().cast[DType.uint32]()
    var biased = (bits >> 23) & 0xFF
    var mantissa = bits & 0x7F_FFFF
    return biased != 0 and biased != 0xFF and mantissa == 0


def _totals() -> List[Float64]:
    """A spread of magnitude sums, chosen to land in different binades and to
    include the two cases where the flooring does nothing and where it costs
    the full bit.

    `FIXED_ONE` and `FIXED_ONE / 4.0` make `2^30 / T` land exactly on a power
    of two, so a rule that floored one binade too far would show up. The
    values just above them are the worst case, where the quotient is barely
    below a power of two and the floor gives up almost the whole bit.
    """
    return [
        1.0,
        2.0,
        3.0,
        7.0,
        1e-6,
        1e-3,
        0.5,
        123.456,
        1e3,
        1e6,
        1e9,
        FIXED_ONE,
        FIXED_ONE / 4.0,
        FIXED_ONE * 1.0000001,
        FIXED_ONE / 4.0 * 1.0000001,
        1.7976931348623157e10,
    ]


# ---------------------------------------------------------------------------
# 1. The scale is exactly a power of two
# ---------------------------------------------------------------------------


def test_scale_is_exactly_a_power_of_two_at_both_widths() raises:
    """No mantissa bits, at Float32 and at Float64.

    This is the claim every other claim in the file rests on. If the returned
    value carried a single mantissa bit, the reciprocal would stop being
    exact, the Float32 product inside the kernel would stop being exact, and
    the host and device factors would stop being the same number at two
    widths.
    """
    var totals = _totals()
    for i in range(len(totals)):
        var s32 = fixed_point_scale_pow2(totals[i])
        assert_true(
            _is_power_of_two_f32(s32),
            String("scale is not a power of two at Float32 for T=", totals[i]),
        )
        assert_true(
            _is_power_of_two_f64(Float64(s32)),
            String("scale is not a power of two at Float64 for T=", totals[i]),
        )


def test_float32_narrowing_is_lossless() raises:
    """The Float32 the kernel receives is bit-for-bit the Float64 the rule
    derived, so 'the host and device are on the same lattice' stops being a
    convention two files have to maintain and becomes an identity."""
    var totals = _totals()
    for i in range(len(totals)):
        var s32 = fixed_point_scale_pow2(totals[i])
        var expected = _pow2_floor_by_loop(FIXED_ONE / totals[i])
        assert_equal(Float64(s32), expected)


def test_default_shape_is_the_power_of_two_one() raises:
    """`fixed_point_scale` is what six call sites import, so what it returns
    with no shape named is the package's actual behavior."""
    assert_equal(DEFAULT_SCALE_SHAPE, SCALE_SHAPE_POW2)
    var totals = _totals()
    for i in range(len(totals)):
        assert_equal(
            fixed_point_scale(totals[i]), fixed_point_scale_pow2(totals[i])
        )
        assert_equal(
            fixed_point_scale_shaped(totals[i], DEFAULT_SCALE_SHAPE),
            fixed_point_scale_pow2(totals[i]),
        )


# ---------------------------------------------------------------------------
# 2. The overflow bound, and what the flooring costs
# ---------------------------------------------------------------------------


def test_scaled_total_never_exceeds_half_the_int32_range() raises:
    """The safe direction, which is the whole reason the rule floors rather
    than rounds to nearest.

    `T * s <= 2^30` is the first term of the repository's overflow argument
    (`test_gpu_portability.test_fixed_point_accumulation_cannot_overflow_int32`,
    `distributed_gpu.check_fixed_point_headroom`). Asserted with `<=` on the
    exact product, not with a tolerance: a rule that rounded up would put the
    product above 2^30 and the whole argument would need re-deriving in the
    loosening direction.
    """
    var totals = _totals()
    for i in range(len(totals)):
        var s = Float64(fixed_point_scale_pow2(totals[i]))
        assert_true(
            totals[i] * s <= FIXED_ONE,
            String("scaled total exceeds 2^30 at T=", totals[i]),
        )


def test_flooring_costs_at_most_one_bit() raises:
    """The other side of the same coin: the scaled total stays above 2^29, so
    the exactness above is bought with at most one bit of lattice resolution
    and never with an unbounded amount.

    Strictly greater than 2^29, because `2^k > (2^30 / T) / 2` holds strictly
    for the floor of a positive quotient.
    """
    var totals = _totals()
    for i in range(len(totals)):
        var s = Float64(fixed_point_scale_pow2(totals[i]))
        assert_true(
            totals[i] * s > FIXED_ONE / 2.0,
            String("scaled total fell below 2^29 at T=", totals[i]),
        )


def test_scale_never_exceeds_the_arbitrary_one() raises:
    """The flooring is a floor of the *same* quotient the shipped arm used,
    so the power-of-two factor is never the larger of the two. That is the
    inequality the overflow proof is written against, and it is the one thing
    a 'round to nearest power of two' variant would break."""
    var totals = _totals()
    for i in range(len(totals)):
        var pow2 = Float64(fixed_point_scale_pow2(totals[i]))
        var arb = Float64(fixed_point_scale_arbitrary(totals[i]))
        assert_true(
            pow2 <= arb,
            String("power-of-two scale exceeds the arbitrary one at T=",
                   totals[i]),
        )


# ---------------------------------------------------------------------------
# 3. The two roundings the rule claims to delete
# ---------------------------------------------------------------------------


def test_dequantization_reciprocal_is_exact() raises:
    """`histogram_gpu.histogram_from_host` computes `1.0 / g_scale` once and
    multiplies every Int32 cell by it. Under this rule the reciprocal is
    itself a power of two and the round trip is the identity, so a downloaded
    cell converts to Float64 with no error at all rather than with two
    roundings.
    """
    var totals = _totals()
    for i in range(len(totals)):
        var s = Float64(fixed_point_scale_pow2(totals[i]))
        var inv = 1.0 / s
        assert_true(
            _is_power_of_two_f64(inv),
            String("reciprocal is not a power of two at T=", totals[i]),
        )
        assert_equal(inv * s, 1.0)
        # And the conversion a downloaded cell actually goes through.
        var cells = [1, -1, 7, -12345, 1 << 20, -(1 << 29)]
        for c in range(len(cells)):
            var q = Float64(cells[c])
            assert_equal(q * inv * s, q)


def test_quantization_product_is_exact_at_float32() raises:
    """Every histogram kernel computes `Int32(round(x * scale))` with both
    operands Float32. Multiplying a Float32 by a power of two shifts its
    exponent and touches no mantissa bit, so the product handed to `round` is
    the true product; the shipped arm handed `round` a value already off by a
    relative 2^-24, which can move a quantized value by a whole unit.

    Checked by widening: the Float32 product must equal the Float64 product
    of the same two numbers exactly.
    """
    var totals = _totals()
    var values = [
        Float32(1.0),
        Float32(-1.0),
        Float32(0.125),
        Float32(3.0),
        Float32(-7.75),
        Float32(1.2345679e-3),
        Float32(-9.87654e2),
    ]
    for i in range(len(totals)):
        var s = fixed_point_scale_pow2(totals[i])
        for v in range(len(values)):
            var x = values[v]
            var narrow = Float64(x * s)
            var wide = Float64(x) * Float64(s)
            # Skip the pairs whose product leaves the Float32 normal range;
            # exactness is a statement about representable products and the
            # rule's own range guard is what keeps the real ones inside it.
            if wide == 0.0 or (wide < 0.0) != (narrow < 0.0):
                continue
            if abs(wide) < 1.2e-38 or abs(wide) > 3.4e38:
                continue
            assert_equal(narrow, wide)


# ---------------------------------------------------------------------------
# 4. Sibling subtraction does not care about the scale
# ---------------------------------------------------------------------------


def test_sibling_subtraction_is_exact_under_both_shapes() raises:
    """`parent - left == right`, cell for cell, whichever scale shape
    produced the cells.

    The exactness is a property of Int32 addition being associative and
    commutative and of a parent cell being the exact integer sum of its
    children's, and a scale is one fixed multiplier applied to every row
    before any integer reaches a bin. So this test is not really about the
    power of two; it is the record that the power of two cannot have broken
    it, which is worth having because every other exactness claim in the
    package rests on this one.

    Written directly over integers rather than through a histogram builder,
    because a builder would need a device and the property being checked is
    the arithmetic.
    """
    var shapes = [SCALE_SHAPE_POW2, SCALE_SHAPE_ARBITRARY]
    var grads = [
        0.5, -1.25, 3.0, -0.125, 7.5, -2.0, 0.0625, -11.0, 4.25, -0.75
    ]
    var total = 0.0
    for i in range(len(grads)):
        total += abs(grads[i])

    for s in range(len(shapes)):
        var scale = Float64(fixed_point_scale_shaped(total, shapes[s]))
        # Rows 0, 2, 4, 6, 8 go left; the rest go right.
        var left = Int64(0)
        var right = Int64(0)
        var parent = Int64(0)
        for i in range(len(grads)):
            var q = Int64(round(grads[i] * scale))
            parent += q
            if i % 2 == 0:
                left += q
            else:
                right += q
        assert_equal(parent - left, right)
        assert_equal(parent - right, left)


# ---------------------------------------------------------------------------
# 5. The extremes the rule's docstring commits to
# ---------------------------------------------------------------------------


def test_all_zero_gradients_floor_rather_than_divide_by_zero() raises:
    """An all-zero round is the degenerate case, and the answer is a finite
    scale from `MAGNITUDE_FLOOR` rather than an infinity. Every value
    quantizes to zero either way; what matters is that nothing raises and
    nothing returns a non-finite factor."""
    var zero = fixed_point_scale_pow2(0.0)
    assert_true(zero > 0.0)
    assert_true(_is_power_of_two_f32(zero))
    # Anything below the floor is the same round as an all-zero one.
    assert_equal(zero, fixed_point_scale_pow2(MAGNITUDE_FLOOR / 10.0))
    assert_equal(zero, fixed_point_scale_pow2(1e-30))
    assert_equal(floored_magnitude(0.0), MAGNITUDE_FLOOR)
    assert_equal(floored_magnitude(-1.0), MAGNITUDE_FLOOR)


def test_a_single_enormous_outlier_stays_inside_the_bound() raises:
    """One row carrying essentially the whole magnitude sum. It is the case
    that pays the full flooring cost, because it is also the case the
    arbitrary scale placed exactly on 2^30, and it is the case an unclamped
    total-bound rule has to be checked on."""
    var outlier = 1e8
    var rest = 1e-4
    var total = outlier + rest * 9.0
    var s = Float64(fixed_point_scale_pow2(total))
    var quantized = round(outlier * s)
    assert_true(quantized <= FIXED_ONE)
    assert_true(quantized > FIXED_ONE / 2.0 - 1.0)
    assert_true(total * s <= FIXED_ONE)


def test_non_finite_and_out_of_range_totals_are_refused() raises:
    """A non-finite magnitude sum is a bug upstream and is refused rather
    than quantized into garbage, and a sum so large that the scale is not a
    representable Float32 power of two is refused rather than rounded *up* to
    the smallest subnormal, which is the direction the bound forbids."""
    var inf = 1.0 / 0.0
    with assert_raises():
        _ = fixed_point_scale_pow2(inf)
    with assert_raises():
        _ = fixed_point_scale_pow2(inf - inf)
    with assert_raises():
        _ = fixed_point_scale_pow2(1e300)


def test_power_of_two_floor_matches_the_loop_oracle() raises:
    """The bit-field extraction against repeated exact halving, including at
    exact powers of two, where a `floor(log2(x))` spelling is most likely to
    come back one binade off."""
    var xs = [
        1.0, 2.0, 4.0, 1024.0, 1.5, 3.9999, 0.5, 0.25, 0.3, 1e-9, 1e9,
        FIXED_ONE, FIXED_ONE - 1.0, FIXED_ONE + 1.0,
    ]
    for i in range(len(xs)):
        assert_equal(
            largest_power_of_two_at_most(xs[i]), _pow2_floor_by_loop(xs[i])
        )
    with assert_raises():
        _ = largest_power_of_two_at_most(0.0)
    with assert_raises():
        _ = largest_power_of_two_at_most(-1.0)
    with assert_raises():
        _ = largest_power_of_two_at_most(1.0 / 0.0)


# ---------------------------------------------------------------------------
# 6. Both arms stay reachable
# ---------------------------------------------------------------------------


def test_both_shapes_are_reachable_and_differ() raises:
    """The A/B the change is supposed to be measurable through. The two arms
    must actually be two arms: on a total whose quotient is not already a
    power of two they return different factors, and on one where it is they
    agree, which is what says the flooring is a floor and not an offset."""
    var arbitrary = fixed_point_scale_shaped(3.0, SCALE_SHAPE_ARBITRARY)
    var pow2 = fixed_point_scale_shaped(3.0, SCALE_SHAPE_POW2)
    assert_true(Float64(pow2) < Float64(arbitrary))
    assert_equal(arbitrary, fixed_point_scale_arbitrary(3.0))

    # `2^30 / 1.0` is already a power of two, so the two arms coincide.
    assert_equal(
        fixed_point_scale_shaped(1.0, SCALE_SHAPE_ARBITRARY),
        fixed_point_scale_shaped(1.0, SCALE_SHAPE_POW2),
    )

    assert_equal(describe_scale_shape(SCALE_SHAPE_POW2), "power-of-two")
    assert_equal(describe_scale_shape(SCALE_SHAPE_ARBITRARY), "arbitrary")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
