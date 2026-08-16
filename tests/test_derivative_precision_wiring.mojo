"""The `derivative_precision` switch, end to end, through both of its
entries.

`tests/test_derivative_precision.mojo` proves the switch itself: that the two
arms are different numbers and that every read site honors the snapshot it is
handed. This file proves that the **parameter** reaches those read sites and
the objective alike, that it reaches them identically to
`MOJOTREES_DERIVATIVE_PRECISION`, and that the one backend which cannot carry
the setting refuses it instead of ignoring it.

Four claims, none of them allowed a tolerance.

**The default does not move.** `widened(False)` is the identity on all four
snapshot fields, and a fit that sets nothing trains the model this package
has always trained.

**The two entries produce the same model, bit for bit.** Asserted at the
ensemble -- every split feature, every threshold bin, every leaf value --
rather than at a flag or at a prediction, because a prediction can hide a
leaf that moved behind a split that moved the other way. Taken plain and
again under a weighted round, which is the case where honoring the setting at
only one of the two halves produces a third answer that is neither arm.

**Every comparison is gated.** Each equality is preceded by an assertion that
the default arm *differs*, so a wiring that did nothing would fail here
rather than pass by making all three arms equal. This repository has shipped
that failure twice.

**Determinism holds at the parameter entry**, at `MOJOTREES_NUM_WORKERS` 1, 3
and 8, bit for bit. Not inherited: the `float64` arm runs the unblocked
accumulation ladder, and the parameter is a route into it that no other test
drives.

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
from mojotrees.boosting import (
    Booster,
    BoosterParams,
    SQUARED_ERROR,
    train,
)
from mojotrees.histogram import (
    ConstHessianSettings,
    Histogram,
    build_histogram,
    check_device_derivative_precision,
    derivative,
)
from mojotrees.parallel import DispatchSettings
from mojotrees.params import parse_params
from mojotrees.tree import TreeParams
from mojotrees.tree_parameters_extra import (
    DERIV_PRECISION_FLOAT32,
    DERIV_PRECISION_FLOAT64,
    ExtraTreeParams,
)


comptime ENV = String("MOJOTREES_DERIVATIVE_PRECISION")


def _extra_at(precision: Int) -> ExtraTreeParams:
    """An otherwise-default bundle at one `derivative_precision` code.

    Built by assignment rather than through `parse_params` so that a nonsense
    code can be constructed too; `test_the_parameter_is_no_longer_refused`
    checks the string route separately.
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
# Fit-level fixtures
# ---------------------------------------------------------------------------


def _booster_params(precision: Int) raises -> BoosterParams:
    """An otherwise-default booster at one `derivative_precision`.

    Everything else is the package default, which is the point: the only
    thing that may differ between the arms compared below is this field.
    """
    var extra = ExtraTreeParams()
    extra.derivative_precision = precision
    return BoosterParams(
        6, 0.1, TreeParams(8, 5, 1.0, 1e-3, 0.0, extra=extra^)
    )


def _fit_features(n_rows: Int, n_features: Int) -> List[Float64]:
    """Column-major features with no exact ties inside a column, so a split
    decision is decided by the gains rather than by a tie-break."""
    var out = List[Float64](capacity=n_rows * n_features)
    for f in range(n_features):
        for r in range(n_rows):
            out.append(Float64(((r * 37 + f * 11) % n_rows)) / 7.0)
    return out^


def _fit_target(features: List[Float64], n_rows: Int) -> List[Float64]:
    """A target whose derivatives are not Float32-exact.

    The `/ 3.0` is what does it: a third is not a dyadic rational, so the
    residual the objective differentiates has a full significand and the two
    precisions genuinely disagree. That the disagreement actually reaches
    the model is not assumed: every fit-level test below asserts the default
    arm differs from the float64 arm before it asserts anything else.
    """
    var out = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        out.append(
            2.0 * features[r]
            - features[n_rows + r]
            + Float64(r % 13) / 3.0
        )
    return out^


def _bits(v: Float64) -> UInt64:
    return UInt64(v.to_bits())


def _ensemble_bits(booster: Booster) -> List[UInt64]:
    """Every split feature, every threshold bin and every leaf value in the
    ensemble, as integers.

    Compared instead of a prediction because a prediction can hide a leaf
    that moved behind a split that moved the other way. Same shape as
    `tests/test_leaf_estimation._ensemble_bits`.
    """
    var out = List[UInt64]()
    out.append(UInt64(len(booster.trees)))
    out.append(_bits(booster.base_score))
    for t in range(len(booster.trees)):
        ref tree = booster.trees[t]
        out.append(UInt64(len(tree.feature)))
        for node in range(len(tree.feature)):
            out.append(UInt64(tree.feature[node]))
            out.append(UInt64(tree.threshold_bin[node]))
            out.append(_bits(tree.value[node]))
    return out^


def _assert_same_bits(
    a: List[UInt64], b: List[UInt64], what: String
) raises:
    assert_equal(len(a), len(b), what)
    for i in range(len(a)):
        assert_equal(a[i], b[i], what)


def _bits_differ(a: List[UInt64], b: List[UInt64]) -> Bool:
    if len(a) != len(b):
        return True
    for i in range(len(a)):
        if a[i] != b[i]:
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


def test_the_parameter_and_the_environment_train_the_same_model() raises:
    """The end-to-end claim, and the one this whole lane exists to make
    true: a fit configured with `derivative_precision = "float64"` is the
    same model, bit for bit, as the same fit under
    `MOJOTREES_DERIVATIVE_PRECISION=float64` -- and a different model from
    the default.

    This test replaces `test_the_objective_half_is_still_missing`, which
    asserted the refusal that stood while only the histogram half was wired.
    Its docstring specified exactly this assertion as its successor.

    Compared as `_ensemble_bits`: every split feature, every threshold bin
    and every leaf value in every tree. A prediction could hide a leaf that
    moved behind a split that moved the other way; this cannot.

    **The gate is the third arm.** Without `assert_true` on the default arm
    differing, a wiring that did nothing would make all three ensembles
    equal and this test would pass having established nothing -- the exact
    failure this repository has shipped twice.
    """
    var n_rows = 240
    var n_features = 4
    var n_bins = 16
    var features = _fit_features(n_rows, n_features)
    var data = bin_equal_width(features, n_rows, n_features, n_bins)
    var target = _fit_target(features, n_rows)

    var default_params = _booster_params(DERIV_PRECISION_FLOAT32)
    var param_params = _booster_params(DERIV_PRECISION_FLOAT64)

    _ = setenv(ENV, "")
    var default_arm = _ensemble_bits(
        train(data, target, SQUARED_ERROR, default_params)
    )
    var param_arm = _ensemble_bits(
        train(data, target, SQUARED_ERROR, param_params)
    )

    _ = setenv(ENV, "float64")
    var env_arm = _ensemble_bits(
        train(data, target, SQUARED_ERROR, default_params)
    )
    # Both entries at once: monotone, so this must equal each of them.
    var both_arm = _ensemble_bits(
        train(data, target, SQUARED_ERROR, param_params)
    )
    _ = setenv(ENV, "")

    # The gate, asserted before the claim so a vacuous pass is impossible.
    assert_true(_bits_differ(default_arm, env_arm))

    # The claim.
    _assert_same_bits(param_arm, env_arm, "parameter vs environment")
    _assert_same_bits(both_arm, env_arm, "both entries vs environment")

    # And the default is still the default: a fit that set nothing and had
    # nothing set is the fit this package has always produced.
    var again = _ensemble_bits(
        train(data, target, SQUARED_ERROR, default_params)
    )
    _assert_same_bits(again, default_arm, "default is stable")


def test_the_two_entries_agree_under_a_weighted_round() raises:
    """The same claim where the two halves are hardest to keep in step.

    A weighted round multiplies a stored derivative by `w` after the
    objective wrote it, so the product is not Float32-representable and the
    histogram read site's narrowing is no longer idempotent. That is exactly
    the case where honoring the setting at only one of the two halves
    produces a third answer, and it is the case the sparse and distributed
    growers were silently getting wrong.
    """
    var n_rows = 240
    var n_features = 4
    var n_bins = 16
    var features = _fit_features(n_rows, n_features)
    var data = bin_equal_width(features, n_rows, n_features, n_bins)
    var target = _fit_target(features, n_rows)
    var weights = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        # Never 1.0 and never Float32-exact, so the multiply always moves
        # bits the read site would otherwise have to reproduce.
        weights.append(0.5 + Float64((r * 7) % 11) / 3.0)

    var default_params = _booster_params(DERIV_PRECISION_FLOAT32)
    var param_params = _booster_params(DERIV_PRECISION_FLOAT64)

    _ = setenv(ENV, "")
    var default_arm = _ensemble_bits(
        train(data, target, SQUARED_ERROR, default_params, weights)
    )
    var param_arm = _ensemble_bits(
        train(data, target, SQUARED_ERROR, param_params, weights)
    )
    _ = setenv(ENV, "float64")
    var env_arm = _ensemble_bits(
        train(data, target, SQUARED_ERROR, default_params, weights)
    )
    _ = setenv(ENV, "")

    assert_true(_bits_differ(default_arm, env_arm))
    _assert_same_bits(param_arm, env_arm, "weighted parameter vs environment")


def test_determinism_across_worker_counts_at_the_parameter_entry() raises:
    """1, 3 and 8 workers, bit for bit, on a fit configured through the
    parameter.

    Required by the round's correctness contract and not inherited from
    anywhere: the `float64` arm runs the unblocked accumulation ladder,
    which is a different kernel from the default arm's, and it now reaches
    that kernel by a route (the parameter) that no existing test drives.
    """
    var n_rows = 240
    var n_features = 4
    var n_bins = 16
    var features = _fit_features(n_rows, n_features)
    var data = bin_equal_width(features, n_rows, n_features, n_bins)
    var target = _fit_target(features, n_rows)
    var params = _booster_params(DERIV_PRECISION_FLOAT64)

    _ = setenv(ENV, "")
    _ = setenv("MOJOTREES_NUM_WORKERS", "1")
    var baseline = _ensemble_bits(
        train(data, target, SQUARED_ERROR, params)
    )
    var counts: List[String] = ["1", "3", "8"]
    for i in range(len(counts)):
        _ = setenv("MOJOTREES_NUM_WORKERS", counts[i])
        var got = _ensemble_bits(train(data, target, SQUARED_ERROR, params))
        _assert_same_bits(got, baseline, "workers=" + counts[i])
    _ = setenv("MOJOTREES_NUM_WORKERS", "")


def test_the_parameter_is_no_longer_refused() raises:
    """The refusal is gone, and the range check is not.

    `float64` is accepted because both halves carry it now. A code that is
    neither name is still refused, because a value that is not a value is
    still a mistake.
    """
    _extra_at(DERIV_PRECISION_FLOAT64).check_derivative_precision()
    _extra_at(DERIV_PRECISION_FLOAT64).check_scalars(20)
    _extra_at(DERIV_PRECISION_FLOAT32).check_derivative_precision()

    var nonsense = ExtraTreeParams()
    nonsense.derivative_precision = 7
    with assert_raises():
        nonsense.check_derivative_precision()

    # And the parameter string is accepted end to end now.
    var config = parse_params("derivative_precision=float64")
    assert_equal(
        config.booster.tree.extra.derivative_precision,
        DERIV_PRECISION_FLOAT64,
    )
    with assert_raises():
        _ = parse_params("derivative_precision=double")


def test_derivative_precision_does_not_move_is_active() raises:
    """The exclusion that makes the end-to-end equality above true rather
    than merely likely.

    `is_active()` means "the gain needs `split._feature_gain`'s adjustment
    pass and the device kernel cannot score this". `derivative_precision`
    needs neither, and while it was in this predicate the parameter entry
    took `_feature_gain`'s active path while the environment entry took the
    inactive one -- the two entries of one switch on two code paths.

    **That divergence was measured, and it is inert**: with
    `derivative_precision` put back into the predicate, the two end-to-end
    arms above are still bit-identical, because the active path's only effect
    at these defaults is to clamp a non-positive gain to 0.0 and the fold
    accepts only a gain strictly greater than a best that starts at 0.0. So
    this assertion guards no live bug. It guards the *mechanism*: while that
    predicate could see the parameter and not the environment, "the two
    entries produce the same model" was a coincidence of the gain arithmetic
    rather than a property, and a future change to `_feature_gain` could have
    ended it silently.
    """
    assert_false(_extra_at(DERIV_PRECISION_FLOAT64).is_active())
    assert_false(_extra_at(DERIV_PRECISION_FLOAT32).is_active())
    # The predicate still answers True for the things it is about, so this
    # exclusion did not empty it.
    var gain_floor = ExtraTreeParams()
    gain_floor.min_gain_to_split = 0.5
    assert_true(gain_floor.is_active())


def test_the_device_refuses_float64_by_name() raises:
    """A backend that cannot carry the setting refuses it rather than
    ignoring it, from either entry.

    The device stores every derivative as Float32 on upload
    (`gpu_gradient_stream.stage_gradients`), so there is no threading that
    could make `float64` mean anything there. Asserted at the check itself
    rather than through a GPU fit, because this is a CPU test file and the
    check is the whole of the behavior.
    """
    _ = setenv(ENV, "")
    # The default is allowed through, or every GPU fit would fail.
    check_device_derivative_precision(False)
    # The parameter entry is refused.
    with assert_raises():
        check_device_derivative_precision(True)
    # And so is the environment entry, which is the one likelier to be left
    # set by accident after a CPU comparison.
    _ = setenv(ENV, "float64")
    with assert_raises():
        check_device_derivative_precision(False)
    with assert_raises():
        check_device_derivative_precision(True)
    _ = setenv(ENV, "")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
