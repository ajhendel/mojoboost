"""The refusal sweep: what the accelerator is handed and does not do.

This file exists because three parameters and one environment knob were
accepted by the GPU trainers, applied by none of them, and reported as
success. They were found by enumerating the parameter and environment
surface against the code that would have to read each item, rather than
against any aggregate predicate; the enumeration is in
`docs/DEVICE_SELECTION.md` under "What a GPU fit honours, refuses, and used
to ignore".

Every test here asserts one of two things.

**That the refusal fires.** `device='gpu'` is a named backend, and a caller
who named it is entitled to be told the backend cannot honour their request.
The trainer-level halves reach `train_gpu` itself, so the check is proved to
be on the trainer's path and not only in the policy.

**That `auto` routes to the CPU for the right reason.** This is the half
that is easy to get wrong and it is why every `auto` test in this file uses
5,000,000 x 50, which is twenty times `AUTO_GPU_MIN_ROWS` and five times the
largest shape any crossover record covers. At a small shape `auto` selects
the CPU anyway and an assertion on the *device* would pass whatever the
block did. So each `auto` test asserts three things together: the selected
device, `DECISION_AUTO_CPU_BLOCKED` rather than
`DECISION_AUTO_CPU_BELOW_EVIDENCE` (which is what says *which* of the two
reasons applied), and the block code itself. And each is paired with a
control at the identical shape with the flag off, which must reach the GPU
on evidence. Without the control the whole file would pass on a build where
`auto` never reaches the accelerator at all.

Capabilities are injected rather than detected, exactly as
`tests/test_device.mojo` injects them and for the same reason:
`DeviceCapabilities.detect` opens no device and cannot produce a
`PROFILE_REPORTED` Apple M4, and the one installed crossover rule can only
match one. So the policy half of this file runs identically on a laptop with
an accelerator and on a CI runner without one, and only the trainer half
skips.

BITS. Nothing here can move a fit that was already honoured. Every check
added by this sweep returns immediately at its parameter's own default
(`EfbSettings.enabled` False, `LinearParams.enabled` False,
`ForcedSplits.none()`, `MOJOTREES_CONST_HESSIAN_VERIFY` unset,
`MOJOTREES_GPU_VERIFY_ROWS` unset), and the non-default values it now
refuses are exactly the values that previously produced a fit which ignored
them. There is no configuration that trained one way before this lane and
trains a different way after it: a configuration either trains identically
or raises.
"""

from std.os import setenv
from std.sys import has_accelerator
from std.testing import (
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
    TestSuite,
)

from mojotrees.apple_gpu_policy import GpuProfile, apple_m4_observed
from mojotrees.binning import fit_bins, map_forced_splits
from mojotrees.boosting import BoosterParams, SQUARED_ERROR
from mojotrees.device_policy import (
    AUTO_DEVICE,
    AUTO_GPU_MIN_ROWS,
    AUTO_MIN_CELLS,
    BLOCK_CONST_HESSIAN_VERIFY,
    BLOCK_FEATURE_BUNDLING,
    BLOCK_FORCED_SPLITS,
    BLOCK_LINEAR_TREE,
    CPU_DEVICE,
    DECISION_AUTO_CPU_BLOCKED,
    DECISION_AUTO_GPU_EVIDENCE,
    DECISION_GPU_REFUSED,
    DeviceCapabilities,
    DeviceRequest,
    GPU_DEVICE,
    M4_TRAINING_MIN_FEATURES,
    NO_DEVICE,
    PROFILE_REPORTED,
    decide_device,
)
from mojotrees.efb import EfbSettings
from mojotrees.initialization import SessionState
from mojotrees.linear_tree import LinearParams
from mojotrees.train_gpu import (
    _check_verify_rows_reachable,
    train_gpu,
    verify_rows_requested,
)
from mojotrees.tree import TreeParams
from mojotrees.tree_parameters_extra import ExtraTreeParams, parse_forced_splits
from mojotrees.unified_memory_policy import SessionMemoryPlan
from support import _make_features as _features

# The bin count both Apple M4 records in the installed crossover rule were
# taken at, so that only the field a test varies is doing any work.
comptime _MEASURED_BINS = 255

# Twenty times `AUTO_GPU_MIN_ROWS` and five times the largest shape any
# record covers. A shape this far above the floor is the only way to tell
# "the CPU because the GPU cannot do this" apart from "the CPU because the
# workload is small", which is the distinction this whole file is about.
comptime _BIG_ROWS = 5_000_000

comptime _CONST_HESSIAN_VERIFY = "MOJOTREES_CONST_HESSIAN_VERIFY"
comptime _VERIFY_ROWS = "MOJOTREES_GPU_VERIFY_ROWS"
comptime _SPLIT_STRATEGY = "MOJOTREES_GPU_SPLIT_STRATEGY"

# A small dense fixture for the trainer-level tests. They all raise before
# any histogram is accumulated, so the shape only has to be legal.
comptime _N_ROWS = 200
comptime _N_FEATURES = 4


def _caps(
    const_hessian_verify: Bool = False,
) raises -> DeviceCapabilities:
    """A reported Apple M4 with a working accelerator.

    `gpu_available` is True regardless of what this build was compiled on,
    which is what lets the policy half of this file run on a machine with no
    accelerator.
    """
    return DeviceCapabilities(
        True,
        True,
        False,
        apple_m4_observed(),
        PROFILE_REPORTED,
        AUTO_MIN_CELLS,
        SessionState.cold(),
        SessionMemoryPlan.staged(),
        const_hessian_verify=const_hessian_verify,
    )


def _big(
    device: Int,
    bundling: Bool = False,
    linear_tree: Bool = False,
    forced_splits: Bool = False,
) raises -> DeviceRequest:
    """A dense, complete, single-output squared-error request at a shape the
    installed crossover rule covers with room to spare."""
    return DeviceRequest(
        device,
        _BIG_ROWS,
        M4_TRAINING_MIN_FEATURES,
        1,
        _MEASURED_BINS,
        SQUARED_ERROR,
        False,
        False,
        False,
        False,
        bundling,
        linear_tree,
        forced_splits,
    )


def _control_reaches_the_gpu() raises:
    """The same shape with nothing set must select the GPU on evidence.

    Called by every `auto` test below. It is not decoration: without it an
    assertion that `auto` chose the CPU proves nothing, because `auto` would
    choose the CPU at this shape on any build where the crossover rule
    cannot fire, and every test in this file would pass for a reason that has
    nothing to do with the block it is testing.
    """
    var permitted = decide_device(_big(AUTO_DEVICE), _caps())
    assert_equal(permitted.selected_device, GPU_DEVICE)
    assert_equal(permitted.decision_code, DECISION_AUTO_GPU_EVIDENCE)


# --- enable_bundle ----------------------------------------------------


def test_bundling_routes_auto_to_the_cpu() raises:
    """`enable_bundle` sends `auto` to the CPU, for a reason that is not the
    shape.

    Bundling is applied by the dense CPU trainers in boosting.mojo, which fit
    a plan once per training call and grow every tree on the bundled matrix.
    `train_gpu` accumulates from the unbundled binned matrix and read the
    setting nowhere at all, so a bundled fit was an unbundled fit reported as
    success. `train_gpu_sparse._refuse_bundling` had refused it since it
    shipped, which is what made the dense gap visible when the two were
    compared.
    """
    var decision = decide_device(_big(AUTO_DEVICE, bundling=True), _caps())
    assert_equal(decision.selected_device, CPU_DEVICE)
    # The whole test. `DECISION_AUTO_CPU_BLOCKED` is what says the answer
    # came from a block; `DECISION_AUTO_CPU_BELOW_EVIDENCE` would say it came
    # from the shape, and at 5,000,000 x 50 that would be false.
    assert_equal(decision.decision_code, DECISION_AUTO_CPU_BLOCKED)
    assert_equal(decision.blocking_reasons.codes[0], BLOCK_FEATURE_BUNDLING)
    assert_false(decision.validated())
    _control_reaches_the_gpu()


def test_bundling_refuses_an_explicit_gpu() raises:
    var decision = decide_device(_big(GPU_DEVICE, bundling=True), _caps())
    assert_true(decision.blocked)
    assert_equal(decision.selected_device, NO_DEVICE)
    assert_equal(decision.decision_code, DECISION_GPU_REFUSED)
    assert_equal(decision.blocking_reasons.codes[0], BLOCK_FEATURE_BUNDLING)
    with assert_raises():
        decision.raise_if_blocked()


def test_train_gpu_refuses_bundling() raises:
    """The trainer half, so the check is proved to be on `train_gpu`'s own
    path and not only in the policy that routes to it.

    This is the protection a caller who reaches `train_gpu` directly has, and
    it is the only one: `resolve_device` describes the shape and the
    objective, so `model.fit` cannot see `enable_bundle` and cannot route
    around it.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var features = _features(_N_ROWS, _N_FEATURES)
        var mapper = fit_bins(features, _N_ROWS, _N_FEATURES, 16)
        var data = mapper.transform(features, _N_ROWS)
        var params = BoosterParams(
            2, 0.1, TreeParams.default(), EfbSettings(True)
        )
        with assert_raises(contains="enable_bundle"):
            _ = train_gpu(data, _target(features), SQUARED_ERROR, params)


# --- linear_tree ------------------------------------------------------


def test_linear_tree_routes_auto_to_the_cpu() raises:
    """`linear_tree` sends `auto` to the CPU, for a reason that is not the
    shape.

    Linear leaves are fitted from the raw feature matrix. Every GPU trainer
    takes a binned matrix, exactly as `boosting.train` does, and
    `boosting.train` refuses it for that reason while no GPU trainer did.
    `model.fit` happens to refuse it before dispatching, which is why this
    was invisible from the Python surface; `model.fit_multiclass`,
    `external_memory.train_external*`, and any direct trainer call went
    straight past it.
    """
    var decision = decide_device(_big(AUTO_DEVICE, linear_tree=True), _caps())
    assert_equal(decision.selected_device, CPU_DEVICE)
    assert_equal(decision.decision_code, DECISION_AUTO_CPU_BLOCKED)
    assert_equal(decision.blocking_reasons.codes[0], BLOCK_LINEAR_TREE)
    _control_reaches_the_gpu()


def test_linear_tree_refuses_an_explicit_gpu() raises:
    var decision = decide_device(_big(GPU_DEVICE, linear_tree=True), _caps())
    assert_true(decision.blocked)
    assert_equal(decision.selected_device, NO_DEVICE)
    assert_equal(decision.blocking_reasons.codes[0], BLOCK_LINEAR_TREE)
    with assert_raises():
        decision.raise_if_blocked()


def test_train_gpu_refuses_linear_trees() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var features = _features(_N_ROWS, _N_FEATURES)
        var mapper = fit_bins(features, _N_ROWS, _N_FEATURES, 16)
        var data = mapper.transform(features, _N_ROWS)
        var params = BoosterParams(
            2,
            0.1,
            TreeParams.default(),
            EfbSettings.disabled(),
            LinearParams(True),
        )
        with assert_raises(contains="linear trees are not supported"):
            _ = train_gpu(data, _target(features), SQUARED_ERROR, params)


# --- forced splits ----------------------------------------------------


def test_forced_splits_route_auto_to_the_cpu() raises:
    """A forced-split document sends `auto` to the CPU, for a reason that is
    not the shape.

    This is the one the aggregate guard appeared to cover.
    `ExtraTreeParams.is_active()` names `forced`, so the parameter reads as
    handled; it is handled on exactly one path, the non-default
    `MOJOTREES_GPU_SPLIT_STRATEGY=device`, whose
    `_check_device_search_supported` refuses the whole bundle. The shipping
    host split scan goes through `tree._search`, which refuses
    `needs_grower_support()` -- `max_delta_step`, `path_smooth`,
    `extra_trees`, `random_strength` -- and that set does not contain
    `forced`.

    It was worse than a gap, because AUTO steered into it: `is_active()`
    being True is exactly what makes the split-search decision decline the
    device arm, so a forced-split fit was routed onto the arm that drops
    them by the same predicate that would have refused it on the other.
    """
    var decision = decide_device(
        _big(AUTO_DEVICE, forced_splits=True), _caps()
    )
    assert_equal(decision.selected_device, CPU_DEVICE)
    assert_equal(decision.decision_code, DECISION_AUTO_CPU_BLOCKED)
    assert_equal(decision.blocking_reasons.codes[0], BLOCK_FORCED_SPLITS)
    _control_reaches_the_gpu()


def test_forced_splits_refuse_an_explicit_gpu() raises:
    var decision = decide_device(
        _big(GPU_DEVICE, forced_splits=True), _caps()
    )
    assert_true(decision.blocked)
    assert_equal(decision.selected_device, NO_DEVICE)
    assert_equal(decision.blocking_reasons.codes[0], BLOCK_FORCED_SPLITS)
    with assert_raises():
        decision.raise_if_blocked()


def test_train_gpu_refuses_a_mapped_forced_split_document() raises:
    """The document is *mapped* here, and that is the point.

    `ExtraTreeParams.check_scalars` already refused an unmapped document,
    because a raw threshold is not a bin. A document that has been through
    `binning.map_forced_splits` passes that check -- it is exactly the case
    the CPU grower applies -- and it is exactly the case the GPU grower used
    to accept and never read. A test on an unmapped document would have
    passed before this lane and proved nothing.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var features = _features(_N_ROWS, _N_FEATURES)
        var mapper = fit_bins(features, _N_ROWS, _N_FEATURES, 16)
        var data = mapper.transform(features, _N_ROWS)
        var forced = map_forced_splits(
            mapper, parse_forced_splits(
                String("{\"feature\": 0, \"threshold\": 0.5}")
            )
        )
        var extra = ExtraTreeParams()
        extra.forced = forced^
        var tree = TreeParams.default()
        tree.extra = extra^
        var params = BoosterParams(2, 0.1, tree^)
        with assert_raises(contains="forced splits are applied by"):
            _ = train_gpu(data, _target(features), SQUARED_ERROR, params)


# --- MOJOTREES_CONST_HESSIAN_VERIFY -----------------------------------


def test_const_hessian_verify_routes_auto_to_the_cpu() raises:
    """The audit sends `auto` to the CPU, for a reason that is not the shape.

    The two halves of one diagnostic pair were in different states.
    `MOJOTREES_CONST_HESSIAN` enables a histogram shortcut and the device
    honours it through its own read; `MOJOTREES_CONST_HESSIAN_VERIFY` audits
    the declaration by walking the host hessian array, and no device builder
    can. So a GPU fit under the audit took the shortcut and reported an
    audited result. That is the worst shape a silent ignore takes: a knob
    that quietly fails to change a fit costs a wrong number, and a knob that
    quietly fails to *check* costs a wrong number the user has stopped
    looking for.

    Injected as a capability rather than set in the environment, which is how
    `derivative_precision_float64` is tested beside it: the capability is
    what `decide_device` reads, and injecting it keeps this test from
    depending on process-wide state that another test in the same file could
    disturb.
    """
    var decision = decide_device(
        _big(AUTO_DEVICE), _caps(const_hessian_verify=True)
    )
    assert_equal(decision.selected_device, CPU_DEVICE)
    assert_equal(decision.decision_code, DECISION_AUTO_CPU_BLOCKED)
    assert_equal(
        decision.blocking_reasons.codes[0], BLOCK_CONST_HESSIAN_VERIFY
    )
    _control_reaches_the_gpu()


def test_const_hessian_verify_refuses_an_explicit_gpu() raises:
    var decision = decide_device(
        _big(GPU_DEVICE), _caps(const_hessian_verify=True)
    )
    assert_true(decision.blocked)
    assert_equal(decision.selected_device, NO_DEVICE)
    assert_equal(decision.decision_code, DECISION_GPU_REFUSED)
    assert_equal(
        decision.blocking_reasons.codes[0], BLOCK_CONST_HESSIAN_VERIFY
    )
    with assert_raises():
        decision.raise_if_blocked()

    # An explicit `cpu` is unaffected: it was always going to run the audit.
    var pinned = decide_device(
        _big(CPU_DEVICE), _caps(const_hessian_verify=True)
    )
    assert_equal(pinned.selected_device, CPU_DEVICE)
    assert_false(pinned.blocked)


def test_train_gpu_refuses_the_const_hessian_audit() raises:
    """The trainer half, and the one test in this file that has to touch the
    environment, because the knob is an environment variable and the trainer
    reads it directly rather than through an injected capability. It is
    cleared again whichever way the assertion goes."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var features = _features(_N_ROWS, _N_FEATURES)
        var mapper = fit_bins(features, _N_ROWS, _N_FEATURES, 16)
        var data = mapper.transform(features, _N_ROWS)
        var params = BoosterParams(2, 0.1, TreeParams.default())
        _ = setenv(_CONST_HESSIAN_VERIFY, "1", True)
        var raised = False
        try:
            _ = train_gpu(data, _target(features), SQUARED_ERROR, params)
        except:
            raised = True
        _ = setenv(_CONST_HESSIAN_VERIFY, "", True)
        assert_true(raised)

        # And the default is untouched: with the knob unset the same fit
        # trains. This is the line that says the refusal costs nothing to a
        # configuration that never asked for the audit.
        _ = train_gpu(data, _target(features), SQUARED_ERROR, params)


# --- MOJOTREES_GPU_VERIFY_ROWS ----------------------------------------


def test_verify_rows_is_refused_on_the_plane_that_cannot_run_it() raises:
    """The row-count cross-check, refused where it would not have run.

    `MOJOTREES_GPU_VERIFY_ROWS=1` asks that after each split the device's
    left-count is downloaded and compared against the histogram's.
    `GpuActiveRows.partition` does exactly that, and it is the arm the
    incremental `apply_split` / `finish_split` loops take. The device-owned
    growth plane still *writes* the count -- its own docstring says the value
    "stays written so that `MOJOTREES_GPU_VERIFY_ROWS` remains meaningful for
    anyone who wants to check" -- and nothing on that plane ever reads it
    back, because reading it per step is the host wait the plane exists to
    remove. So the flag was accepted and did nothing, on what is now the
    default path.

    Both variables are set for this test and both are cleared afterwards
    whichever way the assertion goes. `MOJOTREES_GPU_SPLIT_STRATEGY=device`
    is what puts the fit on the device split search, which is where the
    resident plane is elected.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var features = _features(_N_ROWS, _N_FEATURES)
        var mapper = fit_bins(features, _N_ROWS, _N_FEATURES, 16)
        var data = mapper.transform(features, _N_ROWS)
        var params = BoosterParams(2, 0.1, TreeParams.default())
        _ = setenv(_SPLIT_STRATEGY, "device", True)
        _ = setenv(_VERIFY_ROWS, "1", True)
        var raised = False
        var message = String("")
        try:
            _ = train_gpu(data, _target(features), SQUARED_ERROR, params)
        except e:
            raised = True
            message = String(e)
        _ = setenv(_VERIFY_ROWS, "", True)
        _ = setenv(_SPLIT_STRATEGY, "", True)
        assert_true(raised)
        # Named, so that this cannot pass on some unrelated failure of the
        # device split search.
        assert_true(message.find("MOJOTREES_GPU_VERIFY_ROWS") >= 0)
        assert_true(message.find("MOJOTREES_GPU_TREE_RESIDENT=0") >= 0)


def test_verify_rows_costs_the_default_path_nothing() raises:
    """The other half, and the one that says the refusal above is free.

    With the flag unset, `verify_rows_requested()` is False and
    `_check_verify_rows_reachable()` returns without raising, so the resident
    plane is elected exactly as it was. A fit that never asked for the
    cross-check trains bit for bit as it did before this lane.
    """
    _ = setenv(_VERIFY_ROWS, "", True)
    assert_false(verify_rows_requested())
    _check_verify_rows_reachable()

    _ = setenv(_VERIFY_ROWS, "1", True)
    assert_true(verify_rows_requested())
    var raised = False
    try:
        _check_verify_rows_reachable()
    except:
        raised = True
    _ = setenv(_VERIFY_ROWS, "", True)
    assert_true(raised)


# --- The defaults are unmoved -----------------------------------------


def test_every_new_block_is_silent_at_its_default() raises:
    """The bits claim, stated as an assertion rather than as prose.

    A request with all three parameters at their own defaults and no audit
    requested must record none of the four new blocking reasons, at a shape
    where the GPU is selected. If any of them fired here it would be
    changing a fit that was already honoured, which is the one thing this
    lane may not do.
    """
    var decision = decide_device(_big(AUTO_DEVICE), _caps())
    assert_equal(decision.selected_device, GPU_DEVICE)
    assert_true(decision.blocking_reasons.is_empty())
    assert_true(decision.validated())

    # And explicitly, so that a future block landing in the list does not
    # quietly satisfy the assertion above by some other route.
    for i in range(decision.blocking_reasons.count()):
        var code = decision.blocking_reasons.codes[i]
        assert_true(code != BLOCK_FEATURE_BUNDLING)
        assert_true(code != BLOCK_LINEAR_TREE)
        assert_true(code != BLOCK_FORCED_SPLITS)
        assert_true(code != BLOCK_CONST_HESSIAN_VERIFY)


def _target(features: List[Float64]) -> List[Float64]:
    """A plain linear target. Nothing here trains far enough for its shape to
    matter; every trainer test raises before the first histogram."""
    var y = List[Float64](capacity=_N_ROWS)
    for r in range(_N_ROWS):
        y.append(
            3.0 * features[0 * _N_ROWS + r] - 2.0 * features[1 * _N_ROWS + r]
        )
    return y^


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
