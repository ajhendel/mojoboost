"""`derivative_precision`, and the proof that setting it does something.

Three claims, and none of them is allowed a tolerance.

**The default moves nothing.** Every assertion about the `float32` arm is
against `to_bits()` or against the builder called exactly as it was called
before this switch existed. If any of them fails, the default path changed
and `tests/test_golden_bits.mojo` is about to say so louder.

**The setting is observable, and the fixture proves the gate opened rather
than assuming it.** This repository has already shipped one test whose six
fixtures all ran below the gate and verified nothing, and a second comparing
two arms that were equal whether or not the optimization fired. So the
fixtures here assert the *shape* they depend on -- one row per bin, so a
cell holds exactly one derivative and its value is independent of summation
order, of the row-block count, and of the worker count -- and then assert
the exact Float64 each arm must hold. A cell that held the sum of several
derivatives would be order-dependent and could only be checked loosely,
which is the check this file refuses to write.

**Determinism holds in both settings**, at `MOJOTREES_NUM_WORKERS` 1, 3 and
8, bit for bit against the one-worker baseline.

Deliberately not named `test_gpu_*`: `tools/run_tests.sh` selects the
accelerator subset by name, and a CPU test wearing that prefix is silently
dropped from the CPU suite.
"""

from std.os import setenv
from std.testing import assert_equal, assert_false, assert_raises, assert_true, TestSuite

from mojotrees.binning import BinnedMatrix, bin_equal_width
from mojotrees.boosting import SQUARED_ERROR, BINARY_LOGISTIC, fill_grad_hess
from mojotrees.histogram import (
    ConstHessianSettings,
    Histogram,
    build_histogram,
    build_histogram_subset,
    check_derivative_precision,
    derivative,
    derivative_precision_narrows,
    score_t,
)
from mojotrees.histogram_sparse import (
    build_histogram_sparse,
    sum_all,
    sum_rows,
)
from mojotrees.params import parse_params
from mojotrees.parallel import DispatchSettings
from mojotrees.sparse import CscMatrix, fit_bins_csc, transform_csc
from mojotrees.tree_parameters_extra import (
    DERIV_PRECISION_FLOAT32,
    DERIV_PRECISION_FLOAT64,
    ExtraTreeParams,
    derivative_precision_name,
    parse_derivative_precision,
)


comptime ENV = String("MOJOTREES_DERIVATIVE_PRECISION")


def plain_settings() -> DispatchSettings:
    """The dispatch sentinel, spelled once so the calls below stay short."""
    return DispatchSettings.unresolved()


def _float32(*, resolved: Bool) -> ConstHessianSettings:
    """A snapshot that narrows. `resolved=False` is the sentinel every
    defaulted parameter constructs, which also narrows."""
    if resolved:
        return ConstHessianSettings(True, False, True, True)
    return ConstHessianSettings.unresolved()


def _float64() -> ConstHessianSettings:
    """A resolved snapshot at `derivative_precision = "float64"`.

    Built fieldwise rather than through `resolve()` so the test does not
    depend on the process environment, which another test in the same
    process may be holding.
    """
    return ConstHessianSettings(True, False, True, False)


def _wide_derivatives(n: Int) -> List[Float64]:
    """Values chosen so that **every one of them** loses bits at Float32.

    A derivative that happened to be Float32-exact would make the two arms
    agree and the test vacuous, so the fixture is asserted rather than
    assumed: `test_the_fixture_actually_loses_bits` checks each one.
    """
    var out = List[Float64](capacity=n)
    for i in range(n):
        # `(i + 1/3) / 7`, alternating in sign. A third is not a dyadic
        # rational, so the significand is full at every `i` in this file's
        # range and the value rounds when it is narrowed. The first draft of
        # this fixture was `(3 + 7i)/10 + i/3` and landed exactly on 6.5 and
        # 22.0 at two of its 32 values, which is exactly the vacuous-fixture
        # failure `test_the_fixture_actually_loses_bits` exists to catch --
        # and did.
        var v = (Float64(i) + 1.0 / 3.0) / 7.0
        out.append(v if i % 2 == 0 else -v)
    return out^


def _one_row_per_bin(n_bins: Int) raises -> BinnedMatrix:
    """One feature, `n_bins` rows, row `r` alone in bin `r`.

    Equal-width binning of `0, 1, ..., n_bins - 1` into `n_bins` bins puts
    one row in each. Nothing here trusts that; the callers assert every
    count is exactly 1 before reading a gradient.
    """
    var values = List[Float64](capacity=n_bins)
    for r in range(n_bins):
        values.append(Float64(r))
    return bin_equal_width(values, n_bins, 1, n_bins)


def _assert_one_row_per_bin(hist: Histogram, n_bins: Int) raises:
    """The gate proof for the fixtures below: a cell holds exactly one
    derivative, so its value is a single Float64 and not a sum whose order
    the schedule could move."""
    for b in range(n_bins):
        assert_equal(hist.count_at(b), 1)


# ---------------------------------------------------------------------------
# The narrowing itself
# ---------------------------------------------------------------------------


def test_derivative_at_true_is_exactly_score_t() raises:
    """`float32` is the name `score_t` already had, and this is the whole of
    the default-path claim at the leaf: the two produce the same bits for
    every value, so no read site changed by being rewritten."""
    var values = _wide_derivatives(32)
    for i in range(len(values)):
        var v = values[i]
        assert_equal(derivative[True](v).to_bits(), score_t(v).to_bits())
        assert_equal(
            derivative[True](-v).to_bits(), score_t(-v).to_bits()
        )
    # And on the values a derivative actually takes at the edges.
    var edges: List[Float64] = [0.0, -0.0, 1.0, -1.0, 1e-16, 1e300]
    for i in range(len(edges)):
        assert_equal(
            derivative[True](edges[i]).to_bits(), score_t(edges[i]).to_bits()
        )


def test_derivative_at_false_is_the_identity() raises:
    """`float64` returns the argument, bit for bit. Not "close to": the same
    Float64, including the low 29 significand bits the narrowing zeroes."""
    var values = _wide_derivatives(32)
    for i in range(len(values)):
        assert_equal(derivative[False](values[i]).to_bits(), values[i].to_bits())
    var edges: List[Float64] = [0.0, -0.0, 1.0, -1.0, 1e-16, 1e300]
    for i in range(len(edges)):
        assert_equal(
            derivative[False](edges[i]).to_bits(), edges[i].to_bits()
        )


def test_the_fixture_actually_loses_bits() raises:
    """The gate proof for every comparison in this file.

    If these values were Float32-exact the two arms would agree and every
    "the setting is observable" test below would pass while establishing
    nothing. So each one is checked to differ.
    """
    var values = _wide_derivatives(32)
    for i in range(len(values)):
        assert_true(
            derivative[True](values[i]).to_bits()
            != derivative[False](values[i]).to_bits()
        )


def test_narrowing_is_idempotent_which_is_what_the_gather_relies_on(
) raises:
    """`Float64(Float32(Float64(Float32(x))))` is `Float64(Float32(x))`.

    The property the read sites depend on: a value the objective already
    narrowed is unchanged by the read site narrowing it again, which is why
    the gathered and un-gathered paths add the identical Float64.
    """
    var values = _wide_derivatives(32)
    for i in range(len(values)):
        var once = derivative[True](values[i])
        assert_equal(derivative[True](once).to_bits(), once.to_bits())


# ---------------------------------------------------------------------------
# The snapshot, and the environment
# ---------------------------------------------------------------------------


def test_the_sentinel_narrows() raises:
    """`unresolved()` means `float32`, which is the documented default and
    the reason no builder pays an environment read per node."""
    assert_true(ConstHessianSettings.unresolved().narrow)


def test_the_environment_reaches_the_snapshot() raises:
    """`resolve()` observes `MOJOTREES_DERIVATIVE_PRECISION`, in both
    directions, and unset means `float32`."""
    _ = setenv(ENV, "")
    assert_true(ConstHessianSettings.resolve().narrow)
    assert_true(derivative_precision_narrows())

    _ = setenv(ENV, "float64")
    assert_false(ConstHessianSettings.resolve().narrow)
    assert_false(derivative_precision_narrows())

    _ = setenv(ENV, "float32")
    assert_true(ConstHessianSettings.resolve().narrow)
    assert_true(derivative_precision_narrows())

    _ = setenv(ENV, "")


def test_a_mistyped_precision_is_refused_not_defaulted() raises:
    """A typo that quietly selected the default is how an A/B runs one arm
    under the other's label. It raises instead."""
    _ = setenv(ENV, "double")
    with assert_raises():
        check_derivative_precision()
    with assert_raises():
        _ = ConstHessianSettings.resolve()
    _ = setenv(ENV, "")
    check_derivative_precision()


# ---------------------------------------------------------------------------
# The dense histogram
# ---------------------------------------------------------------------------


def test_the_default_histogram_is_bit_identical_to_the_sentinel() raises:
    """The default-moves-nothing proof at the builder.

    The two-argument form is the call every caller in the package makes and
    the four-argument snapshot is what a wired trainer will pass. They must
    produce the same cells, bit for bit, or the switch changed the default
    while claiming not to.
    """
    var n_bins = 12
    var data = _one_row_per_bin(n_bins)
    var grad = _wide_derivatives(n_bins)
    var hess = _wide_derivatives(n_bins)

    var plain = build_histogram(data, grad, hess)
    var snapshot = build_histogram(
        data, grad, hess, [], False,
        settings=plain_settings(),
        const_hessian_env=_float32(resolved=True),
    )
    _assert_one_row_per_bin(plain, n_bins)
    for i in range(plain.n_cells()):
        assert_equal(plain.grad_at(i).to_bits(), snapshot.grad_at(i).to_bits())
        assert_equal(plain.hess_at(i).to_bits(), snapshot.hess_at(i).to_bits())
        assert_equal(plain.count_at(i), snapshot.count_at(i))


def test_the_full_builder_carries_the_setting_into_the_cell() raises:
    """The observability proof, exactly rather than approximately.

    One row per bin, so cell `b` holds row `b`'s derivative and nothing
    else. The `float32` arm must hold `derivative[True](g[b])` and the
    `float64` arm must hold `g[b]`, both to the bit, and
    `test_the_fixture_actually_loses_bits` has already established that
    those two are different numbers.
    """
    var n_bins = 12
    var data = _one_row_per_bin(n_bins)
    var grad = _wide_derivatives(n_bins)
    var hess = _wide_derivatives(n_bins)

    var narrow = build_histogram(
        data, grad, hess, [], False,
        settings=plain_settings(), const_hessian_env=_float32(resolved=True),
    )
    var wide = build_histogram(
        data, grad, hess, [], False,
        settings=plain_settings(), const_hessian_env=_float64(),
    )
    _assert_one_row_per_bin(narrow, n_bins)
    _assert_one_row_per_bin(wide, n_bins)

    for b in range(n_bins):
        assert_equal(
            narrow.grad_at(b).to_bits(), derivative[True](grad[b]).to_bits()
        )
        assert_equal(
            narrow.hess_at(b).to_bits(), derivative[True](hess[b]).to_bits()
        )
        assert_equal(wide.grad_at(b).to_bits(), grad[b].to_bits())
        assert_equal(wide.hess_at(b).to_bits(), hess[b].to_bits())


def test_the_subset_builder_carries_the_setting_into_the_cell() raises:
    """The same claim for the row-subset builder, which is the one a grower
    reaches once per node and the one that owns the gathered pair buffer."""
    var n_bins = 12
    var data = _one_row_per_bin(n_bins)
    var grad = _wide_derivatives(n_bins)
    var hess = _wide_derivatives(n_bins)
    var rows = List[Int](capacity=n_bins)
    for r in range(n_bins):
        rows.append(r)

    var narrow = build_histogram_subset(
        data, grad, hess, rows, [], False,
        settings=plain_settings(), const_hessian_env=_float32(resolved=True),
    )
    var wide = build_histogram_subset(
        data, grad, hess, rows, [], False,
        settings=plain_settings(), const_hessian_env=_float64(),
    )
    _assert_one_row_per_bin(narrow, n_bins)
    _assert_one_row_per_bin(wide, n_bins)

    for b in range(n_bins):
        assert_equal(
            narrow.grad_at(b).to_bits(), derivative[True](grad[b]).to_bits()
        )
        assert_equal(wide.grad_at(b).to_bits(), grad[b].to_bits())

    # And the two builders agree with each other at each setting, which is
    # the property `compact_rows` and the row blocks are not allowed to move.
    var full_narrow = build_histogram(
        data, grad, hess, [], False,
        settings=plain_settings(), const_hessian_env=_float32(resolved=True),
    )
    var full_wide = build_histogram(
        data, grad, hess, [], False,
        settings=plain_settings(), const_hessian_env=_float64(),
    )
    for i in range(narrow.n_cells()):
        assert_equal(narrow.grad_at(i).to_bits(), full_narrow.grad_at(i).to_bits())
        assert_equal(wide.grad_at(i).to_bits(), full_wide.grad_at(i).to_bits())


# ---------------------------------------------------------------------------
# Determinism, in both settings
# ---------------------------------------------------------------------------


def _fill_hist_at_workers(
    workers: String,
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    env: ConstHessianSettings,
) raises -> Histogram:
    _ = setenv("MOJOTREES_NUM_WORKERS", workers)
    return build_histogram(
        data, grad, hess, [], False,
        settings=plain_settings(), const_hessian_env=env,
    )


def test_determinism_across_worker_counts_in_both_settings() raises:
    """1, 3 and 8 workers, both settings, bit for bit against one worker.

    A wide-enough shape that the dispatch actually fans out; the point is
    that fanning out cannot move a cell at either precision. The
    `float64` arm runs the unblocked ladder by construction (the pair
    buffer is a Float32 shape), which is a different kernel from the
    `float32` arm and so is checked separately rather than assumed to
    inherit the property.
    """
    var n_rows = 4096
    var n_features = 9
    var n_bins = 16
    var values = List[Float64](capacity=n_rows * n_features)
    for k in range(n_rows * n_features):
        values.append(Float64((k * 7919) % 1000) / 7.0)
    var data = bin_equal_width(values, n_rows, n_features, n_bins)
    var grad = _wide_derivatives(n_rows)
    var hess = _wide_derivatives(n_rows)

    var counts: List[String] = ["1", "3", "8"]
    var settings: List[ConstHessianSettings] = [
        _float32(resolved=True), _float64()
    ]
    for s in range(len(settings)):
        var baseline = _fill_hist_at_workers(
            "1", data, grad, hess, settings[s]
        )
        for i in range(len(counts)):
            var got = _fill_hist_at_workers(
                counts[i], data, grad, hess, settings[s]
            )
            for c in range(baseline.n_cells()):
                assert_equal(
                    got.grad_at(c).to_bits(), baseline.grad_at(c).to_bits()
                )
                assert_equal(
                    got.hess_at(c).to_bits(), baseline.hess_at(c).to_bits()
                )
                assert_equal(got.count_at(c), baseline.count_at(c))
    _ = setenv("MOJOTREES_NUM_WORKERS", "")


# ---------------------------------------------------------------------------
# The sparse builders
# ---------------------------------------------------------------------------


def test_the_sparse_node_totals_carry_the_setting() raises:
    """`sum_rows` and `sum_all` are read sites too: they feed the leftover
    assigned to a sparse default bin, so a setting that did not reach them
    would put the whole node's rounding difference into one cell."""
    var grad = _wide_derivatives(8)
    var hess = _wide_derivatives(8)
    var rows: List[Int] = [0, 2, 4, 6]

    var narrow_all = sum_all(grad, hess, True)
    var wide_all = sum_all(grad, hess, False)
    assert_true(narrow_all.grad.to_bits() != wide_all.grad.to_bits())
    assert_equal(sum_all(grad, hess).grad.to_bits(), narrow_all.grad.to_bits())

    var narrow_rows = sum_rows(grad, hess, rows, True)
    var wide_rows = sum_rows(grad, hess, rows, False)
    assert_true(narrow_rows.grad.to_bits() != wide_rows.grad.to_bits())
    assert_equal(
        sum_rows(grad, hess, rows).grad.to_bits(), narrow_rows.grad.to_bits()
    )

    # The `float64` total is the exact Float64 sum in row order; the
    # `float32` total is the sum of the narrowed values. Both are stated
    # rather than compared to each other, so a change to either side fails.
    var want_wide = 0.0
    var want_narrow = 0.0
    for i in range(len(rows)):
        want_wide += grad[rows[i]]
        want_narrow += derivative[True](grad[rows[i]])
    assert_equal(wide_rows.grad.to_bits(), want_wide.to_bits())
    assert_equal(narrow_rows.grad.to_bits(), want_narrow.to_bits())


def test_the_sparse_builder_carries_the_setting() raises:
    """One stored entry per row, one feature, so the accumulation is
    observable the same way the dense fixtures are."""
    var n_rows = 6
    var row_index: List[Int] = [0, 1, 2, 3, 4, 5]
    var values: List[Float64] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    var offsets: List[Int] = [0, 6]
    var csc = CscMatrix(row_index^, values^, offsets^, n_rows, 1)
    var mapper = fit_bins_csc(csc, 8)
    var data = transform_csc(mapper, csc)

    var grad = _wide_derivatives(n_rows)
    var hess = _wide_derivatives(n_rows)

    var narrow = build_histogram_sparse(data, grad, hess, [], True)
    var wide = build_histogram_sparse(data, grad, hess, [], False)
    var default_arm = build_histogram_sparse(data, grad, hess)

    var moved = False
    for i in range(narrow.n_cells()):
        # The default is the narrowing arm, bit for bit.
        assert_equal(
            default_arm.grad_at(i).to_bits(), narrow.grad_at(i).to_bits()
        )
        if narrow.grad_at(i).to_bits() != wide.grad_at(i).to_bits():
            moved = True
    # And the setting is observable rather than merely accepted.
    assert_true(moved)


# ---------------------------------------------------------------------------
# The objective
# ---------------------------------------------------------------------------


def test_fill_grad_hess_carries_the_setting() raises:
    """The objective side of the switch, which is where LightGBM's
    `score_t` narrowing is applied and where `float64` stops applying it.

    Checked against the exact expression each arm must store: the `float32`
    arm stores `derivative[True](g)` of the value the `float64` arm stores,
    which is the whole relationship between the two.
    """
    var n = 64
    var raw = List[Float64](capacity=n)
    var target = List[Float64](capacity=n)
    for r in range(n):
        raw.append(Float64(r) / 11.0 - 3.0)
        target.append(Float64((r * 13) % 17) / 7.0)
    var weights = List[Float64]()

    var g_wide = List[Float64]()
    var h_wide = List[Float64]()
    _ = setenv(ENV, "float64")
    fill_grad_hess(
        raw, target, BINARY_LOGISTIC, weights, 0.0, g_wide, h_wide
    )

    var g_narrow = List[Float64]()
    var h_narrow = List[Float64]()
    _ = setenv(ENV, "float32")
    fill_grad_hess(
        raw, target, BINARY_LOGISTIC, weights, 0.0, g_narrow, h_narrow
    )

    var g_default = List[Float64]()
    var h_default = List[Float64]()
    _ = setenv(ENV, "")
    fill_grad_hess(
        raw, target, BINARY_LOGISTIC, weights, 0.0, g_default, h_default
    )

    var moved = False
    for r in range(n):
        # Unset is the narrowing default, bit for bit.
        assert_equal(g_default[r].to_bits(), g_narrow[r].to_bits())
        assert_equal(h_default[r].to_bits(), h_narrow[r].to_bits())
        # And the narrowing arm is the wide arm put through `score_t`.
        assert_equal(
            g_narrow[r].to_bits(), derivative[True](g_wide[r]).to_bits()
        )
        assert_equal(
            h_narrow[r].to_bits(), derivative[True](h_wide[r]).to_bits()
        )
        if g_narrow[r].to_bits() != g_wide[r].to_bits():
            moved = True
    assert_true(moved)


def test_squared_error_keeps_a_constant_hessian_at_both_settings() raises:
    """`CONSTANT_HESSIAN` is 1.0, exactly representable at both precisions,
    so the specialization's guarantee is untouched by the switch. Asserted
    rather than argued, because the declaration is what the whole
    three-plane elision rests on."""
    var n = 32
    var raw = List[Float64](capacity=n)
    var target = List[Float64](capacity=n)
    for r in range(n):
        raw.append(Float64(r) / 3.0)
        target.append(Float64(r) / 7.0)
    var weights = List[Float64]()

    var names: List[String] = ["float32", "float64"]
    for i in range(len(names)):
        _ = setenv(ENV, names[i])
        var g = List[Float64]()
        var h = List[Float64]()
        fill_grad_hess(raw, target, SQUARED_ERROR, weights, 0.0, g, h)
        for r in range(n):
            assert_equal(h[r].to_bits(), Float64(1.0).to_bits())
    _ = setenv(ENV, "")


# ---------------------------------------------------------------------------
# The parameter
# ---------------------------------------------------------------------------


def test_the_parameter_parses_both_names_and_refuses_the_rest() raises:
    assert_equal(parse_derivative_precision("float32"), DERIV_PRECISION_FLOAT32)
    assert_equal(parse_derivative_precision("float64"), DERIV_PRECISION_FLOAT64)
    assert_equal(derivative_precision_name(DERIV_PRECISION_FLOAT32), "float32")
    assert_equal(derivative_precision_name(DERIV_PRECISION_FLOAT64), "float64")
    with assert_raises():
        _ = parse_derivative_precision("double")
    with assert_raises():
        _ = parse_derivative_precision("64")


def test_the_default_parameter_is_float32_and_inert() raises:
    var extra = ExtraTreeParams()
    assert_equal(extra.derivative_precision, DERIV_PRECISION_FLOAT32)
    assert_false(extra.is_active())
    extra.check_derivative_precision()


def test_float64_is_accepted_now_that_a_trainer_carries_it() raises:
    """The refusal is GONE, and this test is the record of why it existed.

    It was called `test_float64_is_refused_by_name_until_a_trainer_carries_it`
    and it asserted that `float64` raised. That was correct for as long as
    the parameter reached the histogram and not the objective: a fit wired
    only halfway would store `Float64(Float32(g))` and read it
    un-re-narrowed, which is not float32 (accumulation order has moved) and
    not float64 (the low 29 significand bits are already gone). A third
    numerical configuration is worse than either arm, so refusing beat
    ignoring.

    Both halves are now wired -- the histogram snapshot in `fc223da`, the
    objective across thirty call sites after it -- so the refusal has nothing
    left to protect and asserting it would pin a bug in place.

    The claim it used to make is not dropped, it MOVED, and to a stronger
    form:
    `test_derivative_precision_wiring.test_the_parameter_and_the_environment_train_the_same_model`
    compares full ensemble bits between the two entries, gated on the default
    arm differing first, which is the assertion the halfway state could not
    have passed.

    Twice now a docstring in this file predicted its own deletion and was
    wrong about the reason. Keeping the history here is cheaper than
    rediscovering it: a test that predicts its own removal is making a claim
    about the code, and that claim can be false.
    """
    var extra = ExtraTreeParams()
    extra.derivative_precision = DERIV_PRECISION_FLOAT64
    # Accepted by name now. `check_scalars` still has to pass it through,
    # because a range check that only ran on the default value would let a
    # typo reach the fit -- which is the failure the refusal was a blunt
    # instrument against and is now the only thing standing there.
    extra.check_derivative_precision()
    extra.check_scalars(20)

    # And the range check itself still bites. This is the assertion that
    # matters after the lift: `float64` is a value, not an escape hatch, and
    # anything outside the two named settings is still refused.
    var bogus = ExtraTreeParams()
    bogus.derivative_precision = 99
    with assert_raises():
        bogus.check_derivative_precision()


def test_the_parameter_string_reaches_the_field() raises:
    """Both settings parse, and only the two of them do.

    `float64` used to raise here, for the same reason the refusal above used
    to exist and with the same expiry. Now it has to reach the field, because
    a string form that refused the value the trainers carry would make the
    parameter unreachable from `parse_params` -- which is the surface a
    config file and the CLI both go through.

    The typo arm is the half that does not expire. `double` is what a user
    coming from another library writes, and it must fail loudly rather than
    resolving to a default and training the wrong precision in silence.
    """
    var config = parse_params("derivative_precision=float32")
    assert_equal(
        config.booster.tree.extra.derivative_precision,
        DERIV_PRECISION_FLOAT32,
    )
    var wide = parse_params("derivative_precision=float64")
    assert_equal(
        wide.booster.tree.extra.derivative_precision,
        DERIV_PRECISION_FLOAT64,
    )
    with assert_raises():
        _ = parse_params("derivative_precision=double")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
