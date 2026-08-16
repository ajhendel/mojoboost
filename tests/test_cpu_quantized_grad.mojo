"""The CPU quantized-gradient path: LightGBM's parameters, integer cells.

`quantized_gradient.build_histogram_subset_quantized_into_scratch` is the
row-blocked subset builder of `histogram.mojo` over LightGBM's **packed**
integer cell -- a `(gradient, hessian)` pair sharing one integer, an int16
pair in an int32 or an int32 pair in an int64 -- and `quantize_round_cpu` is
the once-per-round map that feeds it. This file establishes five things and
refuses to establish anything else:

1. **Off is the untouched Float64 builder**, proved with `to_bits()` on every
   cell of every plane. Not "close", not "within a tolerance": the same bytes,
   because the off arm of `build_histogram_subset_maybe_quantized` calls
   `histogram.build_histogram_subset_into_scratch` and nothing else.
2. **On, the integer kernel actually ran.** Every quantized fixture asserts
   `QuantBuildReport.row_accumulations == n_active * row_count`, which is zero
   on the float path and cannot be reached by accident. This project has
   shipped three tests that compared two arms which were equal whether or not
   the optimization fired; the report exists so this is not a fourth.
3. **On, the result is exact and not merely deterministic.** Integer addition
   is associative, so the block fold is an exact sum: the same bytes at every
   `MOJOTREES_NUM_WORKERS`, at every `MOJOTREES_CPU_ROW_BLOCKS`, and at every
   task count. `MOJOTREES_CPU_ROW_BLOCKS` is the sharp version of this claim,
   because on the Float64 path it is the one environment variable documented
   to move an answer, and here it provably cannot.
4. **Stochastic rounding is seeded and reproducible.** LightGBM's is not: it
   fills its dither array from one `std::mt19937(seed + thread_id)` per OpenMP
   block and then rotates it by a fresh draw every round
   (`src/treelearner/gradient_discretizer.cpp`), so a row's dither depends on
   the thread count. mojotrees keys the draw on `(seed, round, class, plane,
   row)`, and the tests below assert the arrays are identical at one worker
   and at eight, and that stochastic rounding *fired* rather than silently
   falling back to round-to-nearest.
5. **The packing is LightGBM's and so is its overflow rule.** The cell is
   packed and unpacked the way `DenseBin::ConstructHistogramIntInner` packs
   and unpacks it, the width comes from
   `GradientDiscretizer::SetNumBitsInHistogramBin`, and both arms of that rule
   are exercised rather than one. The rule is asserted against LightGBM's own
   formula rather than against this module's restatement of it.

Nothing here measures anything, and nothing here makes an accuracy claim. The
quantized histogram is not compared against the float one for closeness at any
point, because "how much accuracy does B = 4 cost" is a question for a
`bench/real_data` run and not for a unit test with six fixtures in it. What is
compared for exactness is the two places where quantization is *more* accurate
than Float64 and can be held to it: the dequantization under a power-of-two
lattice, and sibling subtraction.

The LightGBM facts asserted here were read from LightGBM 4.7.0.99:
`include/LightGBM/config.h` for the four names and defaults, and
`src/treelearner/gradient_discretizer.cpp`
(`GradientDiscretizer::DiscretizeGradients`) for the scale rule, the
constant-hessian special case, and the magnitude-symmetric rounding;
`GradientDiscretizer::SetNumBitsInHistogramBin` for the histogram bit width;
`src/io/dense_bin.hpp` (`ConstructHistogramIntInner`) for the packing itself;
and `src/treelearner/serial_tree_learner.cpp` for the `hist_bits <= 16` branch
that makes the two arms the two this file tests.
"""

from std.os import setenv
from std.testing import assert_equal, assert_not_equal, assert_raises, assert_true, TestSuite

from mojotrees.binning import BinnedMatrix, bin_equal_width
from mojotrees.histogram import (
    CONSTANT_HESSIAN,
    Histogram,
    build_histogram_subset_into_scratch,
)
from mojotrees.params import parse_params
from mojotrees.quantized_gradient import (
    DEFAULT_NUM_GRAD_QUANT_BINS,
    FIXED_ONE,
    MODE_FLOAT,
    MODE_QUANTIZED,
    QUANT_REASON_BACKEND,
    QUANT_REASON_NOT_REQUESTED,
    QUANT_REASON_OK,
    HIST_BITS_16,
    HIST_BITS_32,
    HIST_BITS_NONE,
    QuantBuildReport,
    QuantDecision,
    QuantGradParams,
    QuantRoundKey,
    ROUND_NEAREST,
    ROUND_STOCHASTIC,
    SCALE_MAGNITUDE_SUM,
    SCALE_MAX_ABS,
    build_histogram_subset_maybe_quantized,
    build_histogram_subset_quantized_into_scratch,
    cpu_quant_grad_allowed,
    cpu_quant_params,
    decide_cpu_histogram,
    env_cpu_quant_scale_rule,
    describe_histogram_bits,
    fixed_point_scale_pow2,
    gradient_stats,
    histogram_bits_for_node,
    largest_power_of_two_at_most,
    quant_uniform,
    quantize_scalar,
    quantize_round_cpu,
)
from mojotrees.tree_parameters_extra import (
    DEFAULT_NUM_GRAD_QUANT_BINS as EXTRA_DEFAULT_BINS,
    ExtraTreeParams,
)

from support import _uniform


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _reset_env():
    _ = setenv("MOJOTREES_NUM_WORKERS", "")
    _ = setenv("MOJOTREES_CPU_ROW_BLOCKS", "")
    _ = setenv("MOJOTREES_CPU_ROW_BLOCK_AMORTIZE", "")
    _ = setenv("MOJOTREES_CPU_FEATURE_GROUP", "")
    _ = setenv("MOJOTREES_CPU_COMPACT_MIN_ROWS", "")
    _ = setenv("MOJOTREES_CPU_QUANT_GRAD", "")
    _ = setenv("MOJOTREES_CPU_QUANT_SCALE", "")
    _ = setenv("MOJOTREES_CPU_CORE_POOL", "")


def _make_data(
    n_rows: Int, n_features: Int, n_bins: Int, seed: UInt64
) raises -> BinnedMatrix:
    var values = List[Float64](capacity=n_rows * n_features)
    for k in range(n_rows * n_features):
        values.append(_uniform(seed + UInt64(k)))
    return bin_equal_width(values, n_rows, n_features, n_bins)


def _grads(n_rows: Int, seed: UInt64) -> List[Float64]:
    """Signed gradients spanning sixteen decades, alternating sign and scale.

    The wide range is deliberate on both paths. In Float64 it makes a
    reassociation visible in the low bits; on the integer path it is what
    drives most rows onto the low end of the lattice, so a fixture that only
    exercised well-conditioned gradients would not notice a scale that was one
    binade wrong.
    """
    var g = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        var u = 2.0 * _uniform(seed + UInt64(r)) - 1.0
        var scale = 1.0e6 if r % 3 == 0 else 1.0e-6
        g.append(u * scale)
    return g^


def _hessians(n_rows: Int, seed: UInt64) -> List[Float64]:
    var h = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        h.append(_uniform(seed + UInt64(r)) + 0.01)
    return h^


def _ones(n_rows: Int) -> List[Float64]:
    var h = List[Float64](capacity=n_rows)
    for _ in range(n_rows):
        h.append(CONSTANT_HESSIAN)
    return h^


def _row_ids(n_rows: Int) -> List[Int]:
    var rows = List[Int](capacity=n_rows)
    for r in range(n_rows):
        rows.append(r)
    return rows^


def _assert_same_bits(a: Histogram, b: Histogram, label: String) raises:
    """Exact equality on all three planes, through `to_bits()`.

    `to_bits()` rather than `==` because the contract is bytes: `-0.0 == 0.0`
    is true and they are different histograms, and a NaN would compare unequal
    to itself and pass a `!=` guard silently. No tolerance appears anywhere in
    this file.
    """
    assert_equal(a.n_features, b.n_features)
    assert_equal(a.n_bins, b.n_bins)
    for i in range(a.n_features * a.n_bins):
        if a.grad_at(i).to_bits() != b.grad_at(i).to_bits():
            print("  gradient mismatch at cell", i, "for", label)
        assert_equal(a.grad_at(i).to_bits(), b.grad_at(i).to_bits())
        assert_equal(a.hess_at(i).to_bits(), b.hess_at(i).to_bits())
        assert_equal(a.count_at(i), b.count_at(i))


def _quantized_build(
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    rows: List[Int],
    const_hessian: Bool = False,
    features: List[Int] = [],
    num_grad_quant_bins: Int = DEFAULT_NUM_GRAD_QUANT_BINS,
    stochastic_rounding: Bool = True,
) raises -> Tuple[Histogram, QuantBuildReport]:
    """One round, one node, quantized, in the order the module's contract
    fixes: measure, decide, quantize, accumulate.

    The decision and the quantization derive their lattices independently from
    the same `(stats, params, const_hessian)`, and this helper asserts the two
    came out equal. That is not a formality: they are two call sites of
    `derive_scales`, and a round whose rows were quantized on one lattice and
    whose histograms were dequantized on another is silently wrong rather than
    an error.
    """
    var params = cpu_quant_params(
        True, num_grad_quant_bins, False, stochastic_rounding
    )
    var stats = gradient_stats(grad, hess, rows)
    var decision = decide_cpu_histogram(
        stats, params, len(rows), const_hessian
    )
    var qg = List[Int64]()
    var qh = List[Int64]()
    var scales = quantize_round_cpu(
        grad, hess, params, QuantRoundKey.single(params.seed, 0),
        qg, qh, rows, const_hessian,
    )
    assert_equal(scales.grad_units, decision.scales.grad_units)
    assert_equal(scales.hess_units, decision.scales.hess_units)

    var out = Histogram.zeroed(data.n_features, data.n_bins)
    var scratch = List[Int64]()
    var report = build_histogram_subset_quantized_into_scratch(
        out, scratch, data, qg, qh, rows, 0, len(rows), decision.scales,
        features, const_hessian, params.rounding_mode(),
    )
    return (out^, report^)


# ---------------------------------------------------------------------------
# The parameter surface: LightGBM's names, LightGBM's defaults
# ---------------------------------------------------------------------------


def test_lightgbm_defaults_are_lightgbms() raises:
    """The four defaults, read off `include/LightGBM/config.h` 4.7.0.99.

        bool use_quantized_grad     = false;
        int  num_grad_quant_bins    = 4;
        bool quant_train_renew_leaf = false;
        bool stochastic_rounding    = true;

    Spelled out as literals rather than compared against a constant, so that
    changing the constant does not also change the assertion. This is the
    whole of the public surface: no scale rule, no seed, no accumulator width,
    because LightGBM has none of those and this surface is exactly LightGBM's.
    """
    _reset_env()
    var extra = ExtraTreeParams()
    assert_equal(extra.use_quantized_grad, False)
    assert_equal(extra.num_grad_quant_bins, 4)
    assert_equal(extra.quant_train_renew_leaf, False)
    assert_equal(extra.stochastic_rounding, True)


def test_the_bin_count_default_does_not_drift() raises:
    """`tree_parameters_extra` restates the default because it must not import
    `quantized_gradient` (that module imports `raw_leaf_output` from it, and a
    second edge would be a cycle). This is the mechanism that keeps the two
    copies equal."""
    _reset_env()
    assert_equal(EXTRA_DEFAULT_BINS, DEFAULT_NUM_GRAD_QUANT_BINS)
    assert_equal(EXTRA_DEFAULT_BINS, 4)


def test_parameter_string_carries_the_four_names() raises:
    """The parameter string accepts LightGBM's spellings and nothing else in
    the family. `use_quantized_grad` is deliberately left at its default here;
    the next test is about what happens when it is set."""
    _reset_env()
    var config = parse_params(
        "objective=regression num_grad_quant_bins=16"
        " quant_train_renew_leaf=true stochastic_rounding=false"
    )
    assert_equal(config.booster.tree.extra.num_grad_quant_bins, 16)
    assert_equal(config.booster.tree.extra.quant_train_renew_leaf, True)
    assert_equal(config.booster.tree.extra.stochastic_rounding, False)
    assert_equal(config.booster.tree.extra.use_quantized_grad, False)


def test_enabling_it_from_a_parameter_string_is_refused_by_name() raises:
    """No trainer is wired to the quantized histogram, so accepting the key
    would train a float model that silently ignored it. The package's rule is
    to say so where the parameter was set."""
    _reset_env()
    with assert_raises(contains="use_quantized_grad"):
        _ = parse_params("objective=regression use_quantized_grad=true")


def test_odd_and_out_of_range_bin_counts_are_refused() raises:
    """LightGBM computes `num_grad_quant_bins_ / 2` in integer arithmetic, so
    an odd count truncates and the two halves of the gradient lattice stop
    matching. mojotrees refuses rather than truncating."""
    _reset_env()
    var odd = ExtraTreeParams()
    odd.num_grad_quant_bins = 5
    with assert_raises(contains="even"):
        odd.check_scalars(20)
    var small = ExtraTreeParams()
    small.num_grad_quant_bins = 1
    with assert_raises(contains="between 2"):
        small.check_scalars(20)


def test_defaults_leave_the_bundle_inactive() raises:
    """`is_active` is what a grower tests once per tree to take its existing
    path, so the default bundle must answer False with the four new fields on
    it, or every fit would change shape without a parameter being set."""
    _reset_env()
    var extra = ExtraTreeParams()
    assert_equal(extra.is_active(), False)
    extra.check_scalars(20)
    extra.use_quantized_grad = True
    assert_equal(extra.is_active(), True)


# ---------------------------------------------------------------------------
# The decision, and the two off switches
# ---------------------------------------------------------------------------


def test_off_by_default_reports_not_requested() raises:
    _reset_env()
    var n_rows = 200
    var grad = _grads(n_rows, 1)
    var hess = _hessians(n_rows, 2)
    var rows = _row_ids(n_rows)
    var stats = gradient_stats(grad, hess, rows)
    var decision = decide_cpu_histogram(
        stats, QuantGradParams.default(), n_rows
    )
    assert_equal(decision.is_quantized(), False)
    assert_equal(decision.mode, MODE_FLOAT)
    assert_equal(decision.reason, QUANT_REASON_NOT_REQUESTED)


def test_the_environment_off_switch_forces_the_float_path() raises:
    """`MOJOTREES_CPU_QUANT_GRAD=0` is the bisection switch: it overrides an
    explicit request rather than being overridden by one."""
    _reset_env()
    var n_rows = 200
    var grad = _grads(n_rows, 3)
    var hess = _hessians(n_rows, 4)
    var rows = _row_ids(n_rows)
    var stats = gradient_stats(grad, hess, rows)
    var params = cpu_quant_params(True)

    assert_equal(cpu_quant_grad_allowed(), True)
    var on = decide_cpu_histogram(stats, params, n_rows)
    assert_equal(on.is_quantized(), True)
    assert_equal(on.reason, QUANT_REASON_OK)

    _ = setenv("MOJOTREES_CPU_QUANT_GRAD", "0")
    assert_equal(cpu_quant_grad_allowed(), False)
    var off = decide_cpu_histogram(stats, params, n_rows)
    assert_equal(off.is_quantized(), False)
    assert_equal(off.reason, QUANT_REASON_BACKEND)
    _reset_env()


def test_the_scale_rule_arm_is_reachable_and_defaults_to_the_shared_one()\
        raises:
    """The default is the magnitude-sum power-of-two lattice, which is the one
    the GPU histogram already ships and which `fixed_point_scale_pow2` states
    once for both. `MOJOTREES_CPU_QUANT_SCALE=0` selects LightGBM's max-abs
    lattice instead, and only on that arm does `num_grad_quant_bins` reach the
    scale at all -- which is the honest consequence of defaulting to the
    shared rule and is asserted here rather than left to be discovered."""
    _reset_env()
    assert_equal(env_cpu_quant_scale_rule(), SCALE_MAGNITUDE_SUM)
    _ = setenv("MOJOTREES_CPU_QUANT_SCALE", "0")
    assert_equal(env_cpu_quant_scale_rule(), SCALE_MAX_ABS)
    _reset_env()


def test_max_abs_arm_is_lightgbms_scale_formula() raises:
    """`gradient_scale_ = max|g| / (num_grad_quant_bins / 2)`, inverted:
    `units = (B / 2) / max|g|`. Read off
    `GradientDiscretizer::DiscretizeGradients`. Two bin counts, so a fixture
    cannot pass while ignoring the parameter."""
    _reset_env()
    _ = setenv("MOJOTREES_CPU_QUANT_SCALE", "0")
    var n_rows = 300
    var grad = _grads(n_rows, 5)
    var hess = _hessians(n_rows, 6)
    var rows = _row_ids(n_rows)
    var stats = gradient_stats(grad, hess, rows)

    var four = decide_cpu_histogram(stats, cpu_quant_params(True, 4), n_rows)
    var sixteen = decide_cpu_histogram(
        stats, cpu_quant_params(True, 16), n_rows
    )
    assert_equal(four.scales.rule, SCALE_MAX_ABS)
    assert_equal(four.scales.grad_units, 2.0 / stats.max_abs_grad)
    assert_equal(sixteen.scales.grad_units, 8.0 / stats.max_abs_grad)
    # `hessian_scale_ = max|h| / B` on the general arm, so units = B / max|h|.
    assert_equal(four.scales.hess_units, 4.0 / stats.max_abs_hess)
    assert_equal(sixteen.scales.hess_units, 16.0 / stats.max_abs_hess)
    _reset_env()


def test_max_abs_constant_hessian_lattice_is_one_unit_wide() raises:
    """LightGBM sets `hessian_scale_ = max_hessian_abs_` under
    `is_constant_hessian_`, so the hessian lattice is exactly one unit wide
    and every row quantizes to the integer 1. That is what makes the
    dequantized hessian plane come back as the count."""
    _reset_env()
    _ = setenv("MOJOTREES_CPU_QUANT_SCALE", "0")
    var n_rows = 128
    var grad = _grads(n_rows, 7)
    var hess = _ones(n_rows)
    var rows = _row_ids(n_rows)
    var stats = gradient_stats(grad, hess, rows)
    var decision = decide_cpu_histogram(
        stats, cpu_quant_params(True), n_rows, True
    )
    assert_equal(decision.scales.hess_units, 1.0 / CONSTANT_HESSIAN)
    assert_equal(decision.scales.hess_max_unit, Int64(1))
    _reset_env()


def test_the_shared_pow2_scale_is_what_the_cpu_derives() raises:
    """The CPU calls `fixed_point_scale_pow2`, it does not re-implement it.

    Three assertions, and the third is the one that catches a `floor(log2(x))`
    spelling: the factor equals the shared function's answer exactly, it is
    its own power-of-two floor (so it *is* a power of two), and it is at or
    below `2^30 / sum|g|` rather than above it. Rounding the scale up is the
    one direction the overflow argument forbids, and a `log2` spelling comes
    back one binade high at an exact power of two, which is precisely a scale
    rounded up.
    """
    _reset_env()
    var n_rows = 400
    var grad = _grads(n_rows, 8)
    var hess = _hessians(n_rows, 9)
    var rows = _row_ids(n_rows)
    var stats = gradient_stats(grad, hess, rows)
    var decision = decide_cpu_histogram(stats, cpu_quant_params(True), n_rows)

    assert_equal(decision.scales.rule, SCALE_MAGNITUDE_SUM)
    assert_equal(
        decision.scales.grad_units,
        Float64(fixed_point_scale_pow2(stats.sum_abs_grad)),
    )
    assert_equal(
        decision.scales.grad_units,
        largest_power_of_two_at_most(decision.scales.grad_units),
    )
    assert_true(
        decision.scales.grad_units <= FIXED_ONE / stats.sum_abs_grad
    )


# ---------------------------------------------------------------------------
# Off is the untouched Float64 builder
# ---------------------------------------------------------------------------


def test_off_is_bit_identical_to_the_float_builder() raises:
    """The central claim of "off by default", proved rather than asserted.

    Both arms build the same node the same way; the only difference is that
    one goes through `build_histogram_subset_maybe_quantized` with a float
    decision. Every cell of every plane compared through `to_bits()`, and the
    report is checked to confirm the float arm is what ran -- without that
    check this test would pass just as happily if the dispatcher had
    quantized and happened to agree, which it would not, but a test that only
    works because the wrong answer is obviously wrong is not a test.
    """
    _reset_env()
    var n_rows = 900
    var n_features = 7
    var n_bins = 16
    var data = _make_data(n_rows, n_features, n_bins, 21)
    var grad = _grads(n_rows, 22)
    var hess = _hessians(n_rows, 23)
    var rows = _row_ids(n_rows)

    var direct = Histogram.zeroed(n_features, n_bins)
    var direct_pairs = List[Float64]()
    build_histogram_subset_into_scratch(
        direct, direct_pairs, data, grad, hess, rows, 0, n_rows,
    )

    var routed = Histogram.zeroed(n_features, n_bins)
    var pairs = List[Float64]()
    var scratch = List[Int64]()
    var empty_q = List[Int64]()
    var report = build_histogram_subset_maybe_quantized(
        routed, pairs, scratch, data, grad, hess, empty_q, empty_q,
        rows, 0, n_rows,
        QuantDecision.floating(QUANT_REASON_NOT_REQUESTED),
    )
    assert_equal(report.mode, MODE_FLOAT)
    assert_equal(report.row_accumulations, 0)
    _assert_same_bits(direct, routed, "off arm")


# ---------------------------------------------------------------------------
# The marker
# ---------------------------------------------------------------------------


def test_the_report_proves_the_integer_kernel_ran() raises:
    """`row_accumulations` is the marker. It is `n_active * row_count` on the
    quantized path and exactly zero on the float one, so no fixture below can
    pass while silently running Float64."""
    _reset_env()
    var n_rows = 900
    var n_features = 7
    var n_bins = 16
    var data = _make_data(n_rows, n_features, n_bins, 31)
    var grad = _grads(n_rows, 32)
    var hess = _hessians(n_rows, 33)
    var rows = _row_ids(n_rows)

    var built = _quantized_build(data, grad, hess, rows)
    var report = built[1].copy()
    assert_equal(report.mode, MODE_QUANTIZED)
    assert_equal(report.reason, QUANT_REASON_OK)
    assert_equal(report.scale_rule, SCALE_MAGNITUDE_SUM)
    assert_equal(report.row_accumulations, n_features * n_rows)
    assert_true(report.blocks >= 1)
    assert_true(report.group_width >= 1)
    # The magnitude-sum lattice bounds a field at 2^30, which is above 2^16,
    # so it always lands on the wide arm. Asserted rather than assumed,
    # because it is the honest cost of defaulting to that lattice.
    assert_equal(report.hist_bits, HIST_BITS_32)
    assert_equal(describe_histogram_bits(report.hist_bits), "int32 pair in int64")


def test_counts_are_right_and_excluded_features_come_back_zero() raises:
    """Feature subsampling: the excluded slices must be zeroed even though the
    accumulation never visits them, and every active feature's counts must sum
    to the node's row count."""
    _reset_env()
    var n_rows = 600
    var n_features = 9
    var n_bins = 8
    var data = _make_data(n_rows, n_features, n_bins, 41)
    var grad = _grads(n_rows, 42)
    var hess = _hessians(n_rows, 43)
    var rows = _row_ids(n_rows)
    var features: List[Int] = [1, 4, 7]

    var built = _quantized_build(data, grad, hess, rows, False, features)
    var hist = built[0].copy()
    var report = built[1].copy()
    assert_equal(report.row_accumulations, 3 * n_rows)

    for f in range(n_features):
        var total = 0
        var base = f * n_bins
        for b in range(n_bins):
            total += hist.count_at(base + b)
        if f == 1 or f == 4 or f == 7:
            assert_equal(total, n_rows)
        else:
            assert_equal(total, 0)
            for b in range(n_bins):
                assert_equal(hist.grad_at(base + b).to_bits(), Float64(0.0).to_bits())
                assert_equal(hist.hess_at(base + b).to_bits(), Float64(0.0).to_bits())


# ---------------------------------------------------------------------------
# Exactness: the fold, the workers, the blocks
# ---------------------------------------------------------------------------


def test_identical_at_every_worker_count() raises:
    """Determinism across `MOJOTREES_NUM_WORKERS`, which is the round's
    non-negotiable. Integer addition is associative and commutative, so this
    holds by construction rather than by a summation-order argument; the test
    is the guard on the construction."""
    _reset_env()
    var n_rows = 1500
    var n_features = 11
    var n_bins = 12
    var data = _make_data(n_rows, n_features, n_bins, 51)
    var grad = _grads(n_rows, 52)
    var hess = _hessians(n_rows, 53)
    var rows = _row_ids(n_rows)

    _ = setenv("MOJOTREES_NUM_WORKERS", "1")
    var one = _quantized_build(data, grad, hess, rows)
    assert_equal(one[1].row_accumulations, n_features * n_rows)

    var counts: List[String] = ["3", "8"]
    for i in range(len(counts)):
        _ = setenv("MOJOTREES_NUM_WORKERS", counts[i])
        var other = _quantized_build(data, grad, hess, rows)
        assert_equal(other[1].row_accumulations, n_features * n_rows)
        _assert_same_bits(one[0], other[0], String("workers=", counts[i]))
    _reset_env()


def test_identical_at_every_block_count() raises:
    """The claim integers buy, in its sharp form.

    `MOJOTREES_CPU_ROW_BLOCKS` is documented in `apple_cpu_policy` as the one
    variable in that file that moves bits, because a fold of `B` Float64
    partial sums is a different Float64 from one ascending sum. On the integer
    path it cannot move a bit at any setting, because the fold is an exact
    integer sum. So this test asserts what the Float64 path's own test cannot:
    not "deterministic at a fixed block count" but *identical across block
    counts*.

    The report is checked to have actually taken the requested block count, so
    a fixture cannot pass by having every arm collapse to one block.
    """
    _reset_env()
    var n_rows = 2400
    var n_features = 6
    var n_bins = 10
    var data = _make_data(n_rows, n_features, n_bins, 61)
    var grad = _grads(n_rows, 62)
    var hess = _hessians(n_rows, 63)
    var rows = _row_ids(n_rows)

    _ = setenv("MOJOTREES_CPU_ROW_BLOCKS", "1")
    var flat = _quantized_build(data, grad, hess, rows)
    assert_equal(flat[1].blocks, 1)

    var requests: List[String] = ["2", "3", "7", "16"]
    var saw_blocked = False
    for i in range(len(requests)):
        _ = setenv("MOJOTREES_CPU_ROW_BLOCKS", requests[i])
        var blocked = _quantized_build(data, grad, hess, rows)
        assert_equal(blocked[1].blocks, Int(requests[i]))
        if blocked[1].blocks > 1:
            saw_blocked = True
        _assert_same_bits(flat[0], blocked[0], String("blocks=", requests[i]))
    assert_true(saw_blocked)
    _reset_env()


def test_identical_at_every_feature_group_width() raises:
    """The interleave ladder is a traffic decision and not a value decision on
    this path either. Five rungs, one histogram."""
    _reset_env()
    var n_rows = 800
    var n_features = 17
    var n_bins = 9
    var data = _make_data(n_rows, n_features, n_bins, 71)
    var grad = _grads(n_rows, 72)
    var hess = _hessians(n_rows, 73)
    var rows = _row_ids(n_rows)

    _ = setenv("MOJOTREES_CPU_FEATURE_GROUP", "1")
    var base = _quantized_build(data, grad, hess, rows)
    assert_equal(base[1].group_width, 1)

    var rungs: List[String] = ["2", "4", "8", "16"]
    for i in range(len(rungs)):
        _ = setenv("MOJOTREES_CPU_FEATURE_GROUP", rungs[i])
        var other = _quantized_build(data, grad, hess, rows)
        assert_equal(other[1].group_width, Int(rungs[i]))
        _assert_same_bits(base[0], other[0], String("group=", rungs[i]))
    _reset_env()


# ---------------------------------------------------------------------------
# The two places quantization is MORE exact than Float64
# ---------------------------------------------------------------------------


def test_dequantization_is_exact_under_the_pow2_lattice() raises:
    """A cell is `Float64(integer) * 2^-k`, and both factors are exact, so the
    dequantized value is the exact real value of the cell rather than a
    rounding of it. Checked by multiplying back: `cell * units` must land on
    an exact integer, with no tolerance."""
    _reset_env()
    var n_rows = 700
    var n_features = 5
    var n_bins = 12
    var data = _make_data(n_rows, n_features, n_bins, 81)
    var grad = _grads(n_rows, 82)
    var hess = _hessians(n_rows, 83)
    var rows = _row_ids(n_rows)

    var built = _quantized_build(data, grad, hess, rows)
    var hist = built[0].copy()
    var report = built[1].copy()
    assert_equal(report.row_accumulations, n_features * n_rows)
    var g_units = report.scales.grad_units
    var h_units = report.scales.hess_units
    for i in range(n_features * n_bins):
        var gq = hist.grad_at(i) * g_units
        var hq = hist.hess_at(i) * h_units
        assert_equal(gq, Float64(Int64(gq)))
        assert_equal(hq, Float64(Int64(hq)))


def test_sibling_subtraction_is_exact_on_the_lattice() raises:
    """`parent - left == right`, exactly, in Float64.

    In floating point the subtraction trick is exact only up to cancellation
    and this assertion would be false. Here all three cells are exact integers
    times one shared power of two, so the identity survives the
    dequantization. This is the one place quantization makes a result *more*
    accurate than the float path rather than less, and it is asserted with no
    tolerance because it is exact or it is nothing.
    """
    _reset_env()
    var n_rows = 640
    var n_features = 5
    var n_bins = 8
    var data = _make_data(n_rows, n_features, n_bins, 91)
    var grad = _grads(n_rows, 92)
    var hess = _hessians(n_rows, 93)
    var all_rows = _row_ids(n_rows)

    var left_rows = List[Int](capacity=n_rows)
    var right_rows = List[Int](capacity=n_rows)
    for r in range(n_rows):
        if r % 3 == 0:
            left_rows.append(r)
        else:
            right_rows.append(r)

    # One lattice for all three builds: the scale is derived from the whole
    # node once, exactly as `decide` requires, and the children are windows
    # into it. Deriving a scale per child would put them on two lattices and
    # the identity would be meaningless.
    var params = cpu_quant_params(True)
    var stats = gradient_stats(grad, hess, all_rows)
    var decision = decide_cpu_histogram(stats, params, n_rows)
    var qg = List[Int64]()
    var qh = List[Int64]()
    _ = quantize_round_cpu(
        grad, hess, params, QuantRoundKey.single(params.seed, 0),
        qg, qh, all_rows,
    )

    var parent = Histogram.zeroed(n_features, n_bins)
    var left = Histogram.zeroed(n_features, n_bins)
    var right = Histogram.zeroed(n_features, n_bins)
    var scratch = List[Int64]()
    var rp = build_histogram_subset_quantized_into_scratch(
        parent, scratch, data, qg, qh, all_rows, 0, n_rows, decision.scales,
    )
    var lp = build_histogram_subset_quantized_into_scratch(
        left, scratch, data, qg, qh, left_rows, 0, len(left_rows),
        decision.scales,
    )
    var rr = build_histogram_subset_quantized_into_scratch(
        right, scratch, data, qg, qh, right_rows, 0, len(right_rows),
        decision.scales,
    )
    assert_equal(rp.row_accumulations, n_features * n_rows)
    assert_equal(lp.row_accumulations, n_features * len(left_rows))
    assert_equal(rr.row_accumulations, n_features * len(right_rows))

    for i in range(n_features * n_bins):
        assert_equal(
            (parent.grad_at(i) - left.grad_at(i)).to_bits(),
            right.grad_at(i).to_bits(),
        )
        assert_equal(
            (parent.hess_at(i) - left.hess_at(i)).to_bits(),
            right.hess_at(i).to_bits(),
        )
        assert_equal(
            parent.count_at(i) - left.count_at(i), right.count_at(i)
        )


def test_constant_hessian_plane_is_exactly_the_float_builders() raises:
    """With a declared constant hessian the quantized hessian plane comes back
    as `Float64(count)` -- bit for bit what the Float64 builder produces.

    Two steps, both exact. Every row's hessian quantizes to the same integer
    `hq_const` (the lattice puts `1.0 * hess_units` on an exact integer, which
    `_const_hessian_unit` checks rather than assumes), so a bin's hessian sum
    is `hq_const * count` in integers; and dequantizing that by a power of two
    gives `Float64(count)` back. The gradient plane is *not* asserted equal to
    the float builder's and must not be -- quantization moves gradients, that
    is what it is for.
    """
    _reset_env()
    var n_rows = 800
    var n_features = 6
    var n_bins = 10
    var data = _make_data(n_rows, n_features, n_bins, 101)
    var grad = _grads(n_rows, 102)
    var hess = _ones(n_rows)
    var rows = _row_ids(n_rows)

    var built = _quantized_build(data, grad, hess, rows, True)
    var quantized = built[0].copy()
    var report = built[1].copy()
    assert_equal(report.row_accumulations, n_features * n_rows)
    assert_equal(report.const_hessian_elided, True)

    var floated = Histogram.zeroed(n_features, n_bins)
    var pairs = List[Float64]()
    build_histogram_subset_into_scratch(
        floated, pairs, data, grad, hess, rows, 0, n_rows, [], True,
    )
    for i in range(n_features * n_bins):
        assert_equal(quantized.count_at(i), floated.count_at(i))
        assert_equal(
            quantized.hess_at(i).to_bits(), floated.hess_at(i).to_bits()
        )
        assert_equal(
            quantized.hess_at(i).to_bits(),
            Float64(quantized.count_at(i)).to_bits(),
        )


# ---------------------------------------------------------------------------
# Stochastic rounding: seeded, reproducible, and it actually fires
# ---------------------------------------------------------------------------


def test_stochastic_rounding_is_magnitude_symmetric_like_lightgbms() raises:
    """LightGBM's rule, read off `GradientDiscretizer::DiscretizeGradients`.

        gradient >= 0 ? int8(g * inv + u) : int8(g * inv - u)

    with the cast truncating toward zero, so `|q| = floor(|y| + u)`. That is
    not `floor(y + u)` for a negative `y`, and this test picks a value where
    the two disagree rather than one where they happen to coincide -- a
    fixture that only used positive gradients would pass under either rule and
    establish nothing.
    """
    _reset_env()
    # A counter whose draw is large enough that the two spellings differ at
    # y = -1.3: symmetric gives -floor(1.3 + u) = -2 for u >= 0.7, while
    # floor(y + u) gives -1. Search for such a counter rather than hardcoding
    # one, so the assertion does not depend on the mixer's internals.
    var counter = UInt64(0)
    var found = False
    for c in range(64):
        if quant_uniform(UInt64(c)) >= 0.7:
            counter = UInt64(c)
            found = True
            break
    assert_true(found)

    var q = quantize_scalar(-1.3, 1.0, Int64(1 << 30), ROUND_STOCHASTIC, counter)
    assert_equal(q, Int64(-2))
    # The positive twin of the same magnitude lands on +2, which is what
    # "magnitude-symmetric" means and what `floor(y + u)` would break.
    var p = quantize_scalar(1.3, 1.0, Int64(1 << 30), ROUND_STOCHASTIC, counter)
    assert_equal(p, Int64(2))


def test_stochastic_rounding_is_reproducible_across_workers() raises:
    """The draw is keyed on `(seed, round, class, plane, row)` and on nothing
    else, so the quantized arrays are identical at one worker and at eight.

    LightGBM's own stochastic rounding is not reproducible this way -- its
    dither comes from one `std::mt19937(seed + thread_id)` per OpenMP block
    and is then rotated by a fresh draw each round -- so this is a deliberate
    divergence and this test is what makes it worth having.
    """
    _reset_env()
    var n_rows = 3000
    var grad = _grads(n_rows, 111)
    var hess = _hessians(n_rows, 112)
    var rows = _row_ids(n_rows)
    var params = cpu_quant_params(True, 4, False, True)
    assert_equal(params.rounding_mode(), ROUND_STOCHASTIC)
    var key = QuantRoundKey.single(params.seed, 3)

    _ = setenv("MOJOTREES_NUM_WORKERS", "1")
    var g1 = List[Int64]()
    var h1 = List[Int64]()
    _ = quantize_round_cpu(grad, hess, params, key, g1, h1, rows)

    var counts: List[String] = ["3", "8"]
    for i in range(len(counts)):
        _ = setenv("MOJOTREES_NUM_WORKERS", counts[i])
        var g2 = List[Int64]()
        var h2 = List[Int64]()
        _ = quantize_round_cpu(grad, hess, params, key, g2, h2, rows)
        assert_equal(len(g1), len(g2))
        for r in range(n_rows):
            assert_equal(g1[r], g2[r])
            assert_equal(h1[r], h2[r])
    _reset_env()


def test_stochastic_rounding_actually_fires() raises:
    """The marker for the rounding mode itself.

    A reproducibility test passes trivially if `stochastic_rounding=true`
    quietly ran round-to-nearest, so this asserts the two modes produce
    *different* arrays, and that they differ on many rows rather than on one
    unlucky tie.
    """
    _reset_env()
    var n_rows = 3000
    var grad = _grads(n_rows, 121)
    var hess = _hessians(n_rows, 122)
    var rows = _row_ids(n_rows)
    var key = QuantRoundKey.single(11, 0)

    var stoch = cpu_quant_params(True, 4, False, True)
    var nearest = cpu_quant_params(True, 4, False, False)
    assert_equal(stoch.rounding_mode(), ROUND_STOCHASTIC)
    assert_equal(nearest.rounding_mode(), ROUND_NEAREST)

    var gs = List[Int64]()
    var hs = List[Int64]()
    _ = quantize_round_cpu(grad, hess, stoch, key, gs, hs, rows)
    var gn = List[Int64]()
    var hn = List[Int64]()
    _ = quantize_round_cpu(grad, hess, nearest, key, gn, hn, rows)

    var differing = 0
    for r in range(n_rows):
        if gs[r] != gn[r] or hs[r] != hn[r]:
            differing += 1
    assert_true(differing > n_rows // 10)


def test_the_round_index_moves_the_dither() raises:
    """A row's draw depends on the round, so two rounds do not share a dither
    sequence. Without this, "seeded" could mean "the same numbers forever"."""
    _reset_env()
    var n_rows = 2000
    var grad = _grads(n_rows, 131)
    var hess = _hessians(n_rows, 132)
    var rows = _row_ids(n_rows)
    var params = cpu_quant_params(True, 4, False, True)

    var g0 = List[Int64]()
    var h0 = List[Int64]()
    _ = quantize_round_cpu(
        grad, hess, params, QuantRoundKey.single(params.seed, 0), g0, h0, rows
    )
    var g1 = List[Int64]()
    var h1 = List[Int64]()
    _ = quantize_round_cpu(
        grad, hess, params, QuantRoundKey.single(params.seed, 1), g1, h1, rows
    )
    var differing = 0
    for r in range(n_rows):
        if g0[r] != g1[r]:
            differing += 1
    assert_true(differing > 0)


def test_stochastic_rounding_leaves_the_histogram_deterministic() raises:
    """The end-to-end version of the two tests above: a stochastic round
    builds the same histogram bytes at every worker count. This is the one
    that would fail if the dither were drawn inside the accumulation rather
    than once per round."""
    _reset_env()
    var n_rows = 1800
    var n_features = 8
    var n_bins = 11
    var data = _make_data(n_rows, n_features, n_bins, 141)
    var grad = _grads(n_rows, 142)
    var hess = _hessians(n_rows, 143)
    var rows = _row_ids(n_rows)

    _ = setenv("MOJOTREES_NUM_WORKERS", "1")
    var one = _quantized_build(data, grad, hess, rows, False, [], 4, True)
    assert_equal(one[1].row_accumulations, n_features * n_rows)

    var counts: List[String] = ["3", "8"]
    for i in range(len(counts)):
        _ = setenv("MOJOTREES_NUM_WORKERS", counts[i])
        var other = _quantized_build(data, grad, hess, rows, False, [], 4, True)
        _assert_same_bits(
            one[0], other[0], String("stochastic workers=", counts[i])
        )
    _reset_env()


# ---------------------------------------------------------------------------
# LightGBM's packing, and LightGBM's overflow rule
# ---------------------------------------------------------------------------


def test_bit_rule_matches_lightgbms_formula_on_lightgbms_lattice() raises:
    """`SetNumBitsInHistogramBin`, spelled out below.

        max_stat_per_bin = num_data_in_leaf * num_grad_quant_bins
        < 256   -> 8 bits, < 65536 -> 16 bits, else 32

    Asserted against that formula written out here rather than against this
    module's restatement of it, so the two have to agree independently. The
    8-bit case routes to the 16-bit arm: a bound that fits 8 bits fits 16, and
    LightGBM's own tree learner branches on `hist_bits <= 16` rather than on
    three widths, so this loses no arm it uses.
    """
    _reset_env()
    _ = setenv("MOJOTREES_CPU_QUANT_SCALE", "0")
    var hess = _hessians(64, 201)
    var grad = _grads(64, 202)
    var rows = _row_ids(64)
    var stats = gradient_stats(grad, hess, rows)

    var bins: List[Int] = [4, 16]
    var node_rows: List[Int] = [1, 63, 4095, 16383, 16384, 100000]
    for bi in range(len(bins)):
        var params = cpu_quant_params(True, bins[bi])
        var scales = decide_cpu_histogram(stats, params, 1).scales.copy()
        for ni in range(len(node_rows)):
            var n = node_rows[ni]
            var max_stat_per_bin = n * bins[bi]
            var expected = HIST_BITS_16 if max_stat_per_bin < 65536 else (
                HIST_BITS_32 if max_stat_per_bin < 4294967296 else HIST_BITS_NONE
            )
            assert_equal(
                histogram_bits_for_node(n, scales, ROUND_NEAREST), expected
            )
    _reset_env()


def test_the_wide_arm_is_where_the_magnitude_sum_lattice_lives() raises:
    """A 2^30 field bound is above 2^16 at every node size, so the shared
    power-of-two lattice never reaches LightGBM's narrow cell. Stated as a
    test because it is the concrete price of that default and a reader should
    not have to derive it."""
    _reset_env()
    var grad = _grads(500, 211)
    var hess = _hessians(500, 212)
    var rows = _row_ids(500)
    var stats = gradient_stats(grad, hess, rows)
    var scales = decide_cpu_histogram(
        stats, cpu_quant_params(True), 500
    ).scales.copy()
    var sizes: List[Int] = [1, 100, 100000]
    for i in range(len(sizes)):
        assert_equal(
            histogram_bits_for_node(sizes[i], scales, ROUND_NEAREST),
            HIST_BITS_32,
        )


def test_the_narrow_arm_is_reached_and_agrees_with_the_wide_one() raises:
    """LightGBM's lattice on a small node packs an int16 pair into an int32.

    The two arms are separate kernel instantiations over different cell types,
    so this asserts the narrow one *runs* (through the report) and that it
    produces the identical histogram to the wide one on a node where both are
    valid. If the packing, the mask, or the arithmetic shift were wrong on
    either width, the two would disagree.
    """
    _reset_env()
    _ = setenv("MOJOTREES_CPU_QUANT_SCALE", "0")
    var n_rows = 900
    var n_features = 6
    var n_bins = 10
    var data = _make_data(n_rows, n_features, n_bins, 221)
    var grad = _grads(n_rows, 222)
    var hess = _hessians(n_rows, 223)
    var rows = _row_ids(n_rows)

    # n * B = 900 * 4 = 3600 < 65536, so the narrow arm.
    var narrow = _quantized_build(data, grad, hess, rows, False, [], 4, False)
    assert_equal(narrow[1].hist_bits, HIST_BITS_16)
    assert_equal(narrow[1].row_accumulations, n_features * n_rows)

    # n * B = 900 * 128 = 115200 >= 65536, so the wide arm, on a lattice 32
    # times finer. The histograms are not comparable across bin counts, so
    # this only asserts the arm; the agreement test is below.
    var wide = _quantized_build(data, grad, hess, rows, False, [], 128, False)
    assert_equal(wide[1].hist_bits, HIST_BITS_32)
    assert_equal(wide[1].row_accumulations, n_features * n_rows)
    _reset_env()


def test_both_cell_widths_produce_the_same_histogram() raises:
    """The same lattice, the same rows, two cell widths, one histogram.

    The width is chosen by the row count, so the two arms are forced apart by
    building the same node twice at bin counts that straddle the 65536
    threshold *with the same lattice*. That is not possible through the public
    path -- the bin count is the lattice -- so instead the wide arm is forced
    by asking for a node bound the narrow one cannot hold, and the comparison
    is made against the narrow arm's own histogram at the same scale, which
    `decide_cpu_histogram` hands to both.
    """
    _reset_env()
    _ = setenv("MOJOTREES_CPU_QUANT_SCALE", "0")
    var n_rows = 800
    var n_features = 5
    var n_bins = 8
    var data = _make_data(n_rows, n_features, n_bins, 231)
    var grad = _grads(n_rows, 232)
    var hess = _hessians(n_rows, 233)
    var rows = _row_ids(n_rows)

    var params = cpu_quant_params(True, 4, False, False)
    var stats = gradient_stats(grad, hess, rows)
    var decision = decide_cpu_histogram(stats, params, n_rows)
    var qg = List[Int64]()
    var qh = List[Int64]()
    _ = quantize_round_cpu(
        grad, hess, params, QuantRoundKey.single(params.seed, 0), qg, qh, rows
    )

    var narrow = Histogram.zeroed(n_features, n_bins)
    var wide = Histogram.zeroed(n_features, n_bins)
    var s1 = List[Int64]()
    var s2 = List[Int64]()
    # 800 * 4 = 3200 -> narrow.
    var rn = build_histogram_subset_quantized_into_scratch(
        narrow, s1, data, qg, qh, rows, 0, n_rows, decision.scales,
        [], False, ROUND_NEAREST,
    )
    assert_equal(rn.hist_bits, HIST_BITS_16)
    # A widened lattice with the identical units, so the same integers are
    # accumulated but the bound forces the wide cell. `grad_max_unit` and
    # `hess_max_unit` are the bound, not the scale; raising them changes the
    # arm and nothing else.
    var widened = decision.scales.copy()
    widened.grad_max_unit = Int64(1 << 20)
    widened.hess_max_unit = Int64(1 << 20)
    var rw = build_histogram_subset_quantized_into_scratch(
        wide, s2, data, qg, qh, rows, 0, n_rows, widened,
        [], False, ROUND_NEAREST,
    )
    assert_equal(rw.hist_bits, HIST_BITS_32)
    _assert_same_bits(narrow, wide, "cell widths")
    _reset_env()


def test_an_unrepresentable_bound_falls_back_rather_than_wrapping() raises:
    """LightGBM's ladder ends in an unconditional `else 32` and wraps silently
    above `n * B >= 2^32`. This refuses instead, with the overflow reason, and
    the refusal is the divergence worth having: a wrapped histogram is wrong
    in a way nothing downstream can detect."""
    _reset_env()
    _ = setenv("MOJOTREES_CPU_QUANT_SCALE", "0")
    var grad = _grads(64, 241)
    var hess = _hessians(64, 242)
    var rows = _row_ids(64)
    var stats = gradient_stats(grad, hess, rows)
    var scales = decide_cpu_histogram(
        stats, cpu_quant_params(True, 4), 64
    ).scales.copy()
    assert_equal(
        histogram_bits_for_node(2000000000, scales, ROUND_NEAREST),
        HIST_BITS_NONE,
    )
    assert_equal(describe_histogram_bits(HIST_BITS_NONE), "none")
    _reset_env()


def test_the_packed_low_field_is_the_count_under_a_constant_hessian() raises:
    """LightGBM packs the literal 1 into the low field when the hessian is
    constant, so the low field accumulates the row count and the histogram
    needs no count plane at all.

    The assertion is that the dequantized hessian plane and the count plane
    say the same thing, which they can only do if the low field carried the
    count; `const_hessian_elided` on the report is what says the arm ran.
    """
    _reset_env()
    var n_rows = 700
    var n_features = 6
    var n_bins = 9
    var data = _make_data(n_rows, n_features, n_bins, 251)
    var grad = _grads(n_rows, 252)
    var hess = _ones(n_rows)
    var rows = _row_ids(n_rows)

    var built = _quantized_build(data, grad, hess, rows, True)
    var hist = built[0].copy()
    var report = built[1].copy()
    assert_equal(report.const_hessian_elided, True)
    assert_equal(report.row_accumulations, n_features * n_rows)
    for i in range(n_features * n_bins):
        assert_equal(
            hist.hess_at(i).to_bits(), Float64(hist.count_at(i)).to_bits()
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
