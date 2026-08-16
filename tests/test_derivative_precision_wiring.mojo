"""The wiring from the `derivative_precision` **parameter** to the fit, and
the exact boundary of what that wiring reaches.

`tests/test_derivative_precision.mojo` proves the switch itself: that the two
arms are different numbers and that every read site honors the snapshot it is
handed. This file proves the hop *before* that one -- that
`ExtraTreeParams.derivative_precision` reaches the snapshot at all -- and,
just as importantly, marks where it stops.

Three claims, none of them allowed a tolerance.

**The default does not move.** `widened(False)` is the identity on all four
fields, and a fit that sets nothing resolves the snapshot it always resolved.

**The parameter entry and the environment entry produce the same snapshot,
and therefore the same cells.** That is the assertion that proves the wiring
rather than assuming it, and it is taken at the histogram cell, bit for bit,
not at the flag. Both arms are also asserted *different* from the default, so
the test cannot pass by both arms being vacuously equal -- the failure this
repository has shipped twice.

**The refusal that is left is the objective, and it is asserted rather than
described.** `check_derivative_precision` still raises on `float64` set
through the parameters, because `boosting.fill_grad_hess` selects its row
loop from the environment alone. A fit configured through the parameter would
narrow at the objective and not re-narrow at the histogram, which is neither
arm of the switch. When that raise stops firing, `test_the_objective_half_is
_still_missing` fails, which is deliberate: it is the tripwire on the day the
other half lands.

Determinism across `MOJOTREES_NUM_WORKERS` is inherited rather than
re-measured, and `test_the_parameter_snapshot_is_field_identical_to_the
_environment_snapshot` is what makes that legitimate: the parameter produces
a snapshot whose four fields equal the environment's, so it is the same input
to the same builder that
`test_derivative_precision.test_determinism_across_worker_counts_in_both
_settings` already checked at 1, 3 and 8 workers. Re-running a 4096-row
dispatch here would re-measure a proven property and cost the orchestrator a
compile for it.

Deliberately not named `test_gpu_*`: `tools/run_tests.sh` selects the
accelerator subset by name, and a CPU test wearing that prefix is silently
dropped from the CPU suite.
"""

from std.os import setenv
from std.testing import (
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
    TestSuite,
)

from mojotrees.binning import BinnedMatrix, bin_equal_width
from mojotrees.histogram import (
    ConstHessianSettings,
    Histogram,
    build_histogram,
    derivative,
)
from mojotrees.parallel import DispatchSettings
from mojotrees.tree_parameters_extra import (
    DERIV_PRECISION_FLOAT32,
    DERIV_PRECISION_FLOAT64,
    ExtraTreeParams,
)


comptime ENV = String("MOJOTREES_DERIVATIVE_PRECISION")


def _extra_at(precision: Int) -> ExtraTreeParams:
    """An otherwise-default bundle at one `derivative_precision` code.

    Built by assignment rather than through `parse_params`, because the
    parameter string is refused for `float64` (see
    `test_the_objective_half_is_still_missing`) and this file has to be able
    to construct the value the refusal is about in order to test the wiring
    underneath it.
    """
    var extra = ExtraTreeParams()
    extra.derivative_precision = precision
    return extra^


def _assert_same_snapshot(
    got: ConstHessianSettings, want: ConstHessianSettings
) raises:
    """All four fields, not just `narrow`. A widening that also moved
    `allowed`, `verify` or `resolved` would change the constant-hessian
    decision or the sentinel semantics as a side effect of a precision
    setting, which is exactly the kind of coupling this asserts against.
    """
    assert_equal(got.allowed, want.allowed)
    assert_equal(got.verify, want.verify)
    assert_equal(got.resolved, want.resolved)
    assert_equal(got.narrow, want.narrow)


def _wide_derivatives(n: Int) -> List[Float64]:
    """Values chosen so that every one of them loses bits at Float32.

    `(i + 1/3) / 7`, alternating in sign. A third is not a dyadic rational,
    so the significand is full at every `i` in this file's range.
    `test_the_fixture_actually_loses_bits` asserts it rather than trusting
    it: a Float32-exact fixture would make both arms agree and every
    comparison below vacuous.
    """
    var out = List[Float64](capacity=n)
    for i in range(n):
        var v = (Float64(i) + 1.0 / 3.0) / 7.0
        out.append(v if i % 2 == 0 else -v)
    return out^


def _one_row_per_bin(n_bins: Int) raises -> BinnedMatrix:
    """One feature, `n_bins` rows, row `r` alone in bin `r`.

    A cell then holds exactly one derivative, so its value is independent of
    summation order, of the row-block count and of the worker count, and can
    be asserted as an exact Float64. `_assert_one_row_per_bin` checks the
    shape rather than assuming the binning delivered it.
    """
    var values = List[Float64](capacity=n_bins)
    for r in range(n_bins):
        values.append(Float64(r))
    return bin_equal_width(values, n_bins, 1, n_bins)


def _assert_one_row_per_bin(hist: Histogram, n_bins: Int) raises:
    for b in range(n_bins):
        assert_equal(hist.count_at(b), 1)


def _assert_cells_equal(a: Histogram, b: Histogram) raises:
    assert_equal(a.n_cells(), b.n_cells())
    for i in range(a.n_cells()):
        assert_equal(a.grad_at(i).to_bits(), b.grad_at(i).to_bits())
        assert_equal(a.hess_at(i).to_bits(), b.hess_at(i).to_bits())
        assert_equal(a.count_at(i), b.count_at(i))


def _cells_differ(a: Histogram, b: Histogram) raises -> Bool:
    for i in range(a.n_cells()):
        if a.grad_at(i).to_bits() != b.grad_at(i).to_bits():
            return True
    return False


# ---------------------------------------------------------------------------
# The predicate, and the field it reads
# ---------------------------------------------------------------------------


def test_wants_float64_derivatives_reads_the_field() raises:
    """The one predicate a trainer calls. False at the default, which is
    what keeps every unconfigured fit on the path it was already on."""
    assert_false(ExtraTreeParams().wants_float64_derivatives())
    assert_false(
        _extra_at(DERIV_PRECISION_FLOAT32).wants_float64_derivatives()
    )
    assert_true(
        _extra_at(DERIV_PRECISION_FLOAT64).wants_float64_derivatives()
    )


# ---------------------------------------------------------------------------
# The fold, and what it is not allowed to move
# ---------------------------------------------------------------------------


def test_widening_by_false_is_the_identity() raises:
    """The default-moves-nothing proof at the fold itself.

    Every fit in the package that does not set `derivative_precision` calls
    `widened(False)` once per tree, so if this is not the identity on all
    four fields, every fit moved.
    """
    var sentinel = ConstHessianSettings.unresolved()
    _assert_same_snapshot(sentinel.widened(False), sentinel)

    var resolved_narrow = ConstHessianSettings(True, False, True, True)
    _assert_same_snapshot(resolved_narrow.widened(False), resolved_narrow)

    var resolved_wide = ConstHessianSettings(True, False, True, False)
    _assert_same_snapshot(resolved_wide.widened(False), resolved_wide)

    # Including the const-hessian fields at their non-default values, so a
    # fold that clobbered them with `resolve()`'s answers would fail here
    # rather than in somebody's hessian plane.
    var odd = ConstHessianSettings(False, True, True, True)
    _assert_same_snapshot(odd.widened(False), odd)


def test_widening_by_true_clears_narrow_and_nothing_else() raises:
    """`float64` from the parameter, from every starting snapshot.

    The sentinel case is the one that matters most: `narrow` is the one
    field every reader consults even under the sentinel, so widening a
    sentinel is how an unwired grower still honors the parameter.
    """
    var sentinel = ConstHessianSettings.unresolved()
    var widened_sentinel = sentinel.widened(True)
    assert_false(widened_sentinel.narrow)
    assert_equal(widened_sentinel.allowed, sentinel.allowed)
    assert_equal(widened_sentinel.verify, sentinel.verify)
    # Widening does not resolve. A snapshot that arrived as the sentinel is
    # still the sentinel for the two const-hessian readers, which is what
    # keeps them on their live-read path rather than trusting fields that
    # were never read.
    assert_false(widened_sentinel.resolved)

    var odd = ConstHessianSettings(False, True, True, True)
    var widened_odd = odd.widened(True)
    assert_false(widened_odd.narrow)
    assert_equal(widened_odd.allowed, False)
    assert_equal(widened_odd.verify, True)
    assert_equal(widened_odd.resolved, True)


def test_the_fold_is_monotone_and_idempotent() raises:
    """Neither property is decoration: `boosting._boost_rounds` folds the
    parameter in at the fit and `tree.grow_tree_leaves_profiled` folds the
    same parameter in again at every tree. If the fold were not idempotent
    those two would disagree, and if it were not monotone the order they run
    in would be observable in a fit's numbers.
    """
    var wide = ConstHessianSettings(True, False, True, False)
    # Idempotent: widening an already-wide snapshot changes nothing.
    _assert_same_snapshot(wide.widened(True), wide)
    _assert_same_snapshot(wide.widened(True).widened(True), wide)
    # Monotone: `False` never narrows what `True` widened.
    _assert_same_snapshot(wide.widened(False), wide)

    var narrow = ConstHessianSettings(True, False, True, True)
    _assert_same_snapshot(narrow.widened(True).widened(False), wide)


# ---------------------------------------------------------------------------
# Precedence, decided explicitly: float64 wins from either entry
# ---------------------------------------------------------------------------


def test_precedence_all_four_combinations() raises:
    """The whole precedence rule, as a truth table.

    `float64` from the parameter **or** from the environment wins. The row
    that decides the rule is (environment float64, parameter float32): the
    parameter's default cannot be told apart from "the caller said nothing",
    so a parameter-beats-environment rule would silently narrow the
    documented environment entry that this project's Float32 accuracy
    decision was taken through.
    """
    _ = setenv(ENV, "")
    assert_true(ConstHessianSettings.resolve_with(False).narrow)
    assert_false(ConstHessianSettings.resolve_with(True).narrow)

    _ = setenv(ENV, "float32")
    assert_true(ConstHessianSettings.resolve_with(False).narrow)
    assert_false(ConstHessianSettings.resolve_with(True).narrow)

    _ = setenv(ENV, "float64")
    # The row that decides the rule.
    assert_false(ConstHessianSettings.resolve_with(False).narrow)
    assert_false(ConstHessianSettings.resolve_with(True).narrow)

    _ = setenv(ENV, "")


def test_resolve_with_false_is_exactly_resolve() raises:
    """An unconfigured fit takes the snapshot it always took, at every
    setting of the environment, on all four fields."""
    var envs: List[String] = ["", "float32", "float64"]
    for i in range(len(envs)):
        _ = setenv(ENV, envs[i])
        _assert_same_snapshot(
            ConstHessianSettings.resolve_with(False),
            ConstHessianSettings.resolve(),
        )
    _ = setenv(ENV, "")


def test_a_mistyped_environment_is_still_refused_through_resolve_with(
) raises:
    """`resolve_with` must not become a way around the typo refusal. A value
    that is neither name raises whatever the parameter says, in both
    directions, because a typo that quietly selected the default is how an
    A/B runs one arm under the other's label.
    """
    _ = setenv(ENV, "double")
    with assert_raises():
        _ = ConstHessianSettings.resolve_with(False)
    with assert_raises():
        _ = ConstHessianSettings.resolve_with(True)
    _ = setenv(ENV, "")
    _ = ConstHessianSettings.resolve_with(True)


# ---------------------------------------------------------------------------
# The wiring, proven at the cell
# ---------------------------------------------------------------------------


def test_the_fixture_actually_loses_bits() raises:
    """The gate proof for every cell comparison below. If these values were
    Float32-exact the two arms would agree and the wiring test would pass
    while establishing nothing."""
    var values = _wide_derivatives(12)
    for i in range(len(values)):
        assert_true(
            derivative[True](values[i]).to_bits()
            != derivative[False](values[i]).to_bits()
        )


def test_the_parameter_snapshot_is_field_identical_to_the_environment_snapshot(
) raises:
    """The wiring claim at the snapshot: the parameter entry and the
    environment entry produce the *same object*, not merely two objects that
    agree about `narrow`.

    This is what lets the worker-count determinism already proven for the
    environment arm transfer to the parameter arm without re-measuring it:
    the same four fields go into the same builder.
    """
    _ = setenv(ENV, "float64")
    var from_env = ConstHessianSettings.resolve()
    _ = setenv(ENV, "")
    var from_param = ConstHessianSettings.resolve_with(
        _extra_at(DERIV_PRECISION_FLOAT64).wants_float64_derivatives()
    )
    _assert_same_snapshot(from_param, from_env)
    # And the gate: both differ from the default, so the equality above is
    # not the equality of two default snapshots.
    var from_default = ConstHessianSettings.resolve()
    assert_true(from_default.narrow)
    assert_false(from_param.narrow)


def test_the_parameter_reaches_the_histogram_cell() raises:
    """The wiring claim where it is worth having: at the cells.

    Three arms of the same builder on the same data. The parameter arm and
    the environment arm must be bit-identical, and both must differ from the
    default arm. The second half is the gate proof -- without it, a wiring
    that did nothing at all would make all three arms equal and the test
    would pass.

    One row per bin, so a cell holds exactly one derivative and its bits are
    a property of the setting rather than of the schedule.
    """
    var n_bins = 12
    var data = _one_row_per_bin(n_bins)
    var grad = _wide_derivatives(n_bins)
    var hess = _wide_derivatives(n_bins)

    _ = setenv(ENV, "float64")
    var env_arm = build_histogram(
        data, grad, hess, [], False,
        settings=DispatchSettings.unresolved(),
        const_hessian_env=ConstHessianSettings.resolve(),
    )

    _ = setenv(ENV, "")
    var param_arm = build_histogram(
        data, grad, hess, [], False,
        settings=DispatchSettings.unresolved(),
        const_hessian_env=ConstHessianSettings.resolve_with(
            _extra_at(DERIV_PRECISION_FLOAT64).wants_float64_derivatives()
        ),
    )
    var default_arm = build_histogram(
        data, grad, hess, [], False,
        settings=DispatchSettings.unresolved(),
        const_hessian_env=ConstHessianSettings.resolve_with(
            ExtraTreeParams().wants_float64_derivatives()
        ),
    )

    _assert_one_row_per_bin(env_arm, n_bins)
    _assert_one_row_per_bin(param_arm, n_bins)
    _assert_one_row_per_bin(default_arm, n_bins)

    # The claim.
    _assert_cells_equal(param_arm, env_arm)
    # The gate: the arms are not all the same histogram.
    assert_true(_cells_differ(param_arm, default_arm))

    # Stated absolutely rather than only relatively, so a change to either
    # side fails here instead of cancelling.
    for b in range(n_bins):
        assert_equal(param_arm.grad_at(b).to_bits(), grad[b].to_bits())
        assert_equal(param_arm.hess_at(b).to_bits(), hess[b].to_bits())
        assert_equal(
            default_arm.grad_at(b).to_bits(),
            derivative[True](grad[b]).to_bits(),
        )


def test_the_sentinel_path_honors_the_parameter_too() raises:
    """A grower handed no snapshot still honors the parameter, because
    `tree.grow_tree_leaves_profiled` widens whatever it ended up with --
    the passed snapshot or the per-tree fallback.

    Asserted at the builder through a widened sentinel, which is the exact
    value that path constructs when the environment is unset.
    """
    var n_bins = 12
    var data = _one_row_per_bin(n_bins)
    var grad = _wide_derivatives(n_bins)
    var hess = _wide_derivatives(n_bins)

    _ = setenv(ENV, "")
    var widened_sentinel = ConstHessianSettings.unresolved().widened(True)
    var wide = build_histogram(
        data, grad, hess, [], False,
        settings=DispatchSettings.unresolved(),
        const_hessian_env=widened_sentinel,
    )
    var plain = build_histogram(data, grad, hess)
    _assert_one_row_per_bin(wide, n_bins)
    assert_true(_cells_differ(wide, plain))
    for b in range(n_bins):
        assert_equal(wide.grad_at(b).to_bits(), grad[b].to_bits())


# ---------------------------------------------------------------------------
# Where the wiring stops, asserted rather than described
# ---------------------------------------------------------------------------


def test_the_objective_half_is_still_missing() raises:
    """`derivative_precision = "float64"` set through the parameters is
    still refused, and this test is the tripwire on the day it stops being.

    The histogram half is wired -- every test above proves it -- but
    `boosting.fill_grad_hess` and `boosting._fill_softmax_grad_hess` still
    select their row loop from `MOJOTREES_DERIVATIVE_PRECISION` alone and
    take no parameter. A fit configured through the field alone would narrow
    at the objective and *not* re-narrow at the histogram, which is neither
    arm: not `float32`, because the gathered pair buffer and the row-blocked
    histograms are off and a GOSS or weighted round accumulates
    `w * Float32(g)` un-re-narrowed; and not `float64`, because the
    objective already discarded the low 29 significand bits the setting
    exists to keep.

    When the objective carries the field, delete this test and assert the
    end-to-end equality it is standing in for: a fit at
    `derivative_precision = "float64"` predicting bit-identically to the same
    fit under `MOJOTREES_DERIVATIVE_PRECISION=float64`, and differently from
    the default.
    """
    var extra = _extra_at(DERIV_PRECISION_FLOAT64)
    with assert_raises():
        extra.check_derivative_precision()
    with assert_raises():
        extra.check_scalars(20)

    # The default is not refused, at either setting of the environment, so
    # the refusal above is about the parameter and not about the switch.
    _ = setenv(ENV, "float64")
    ExtraTreeParams().check_derivative_precision()
    _ = setenv(ENV, "")
    ExtraTreeParams().check_derivative_precision()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
