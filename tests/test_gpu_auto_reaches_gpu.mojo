"""`device='auto'` reaches the accelerator: proved by what the fit produced.

WHY THIS FILE EXISTS SEPARATELY FROM `test_device_auto_crossover.mojo`. That
file proves the *policy* answers GPU: `resolve_device` returns `GPU_DEVICE`,
the decision code is `DECISION_AUTO_GPU_EVIDENCE`, `validated()` is True. Its
own docstring says what it cannot prove, in as many words: "It does not prove
that a fit runs on the accelerator, because that is a training run". A
returned enum is not a backend. This repository has twice shipped a device
test that asserted a value rather than an effect, and
`bench/results/session3_2026-08-16/RESULTS.md` records both.

So every assertion here is on a **model**, not on a device code, and the
observable is one the host path cannot produce. The GPU trainer accumulates
its histograms in Float32 fixed point where the CPU trainer is Float64
throughout, so at this scale the two backends produce leaf values that differ
in their bits while routing rows to the same leaves. One dataset, four fits,
and the comparison is bit-for-bit:

    auto  == gpu   ->  `auto` took the accelerator path
    auto  != cpu   ->  and the equality above is not both arms agreeing

Neither half is enough alone. The first would hold vacuously if the CPU and
GPU trainers happened to agree; the second would hold if `auto` had gone
somewhere else entirely. Together they say `auto` ran the GPU trainer.

The float64 pair is the same construction for the second thing this lane
changed. `MOJOTREES_DERIVATIVE_PRECISION=float64` is a request the device
cannot honor (`gpu_gradient_stream.stage_gradients` narrows every derivative
to Float32 on upload), so `auto` must route to the CPU **at a shape well
above the crossover floor**, and the proof that it routed rather than merely
resolved is that its model is the host's model bit for bit and is not the
model the same call produced with the flag off.

NAMING. `test_gpu_*` and it carries no cpu-safe marker comment, so
`tools/run_tests.sh` classifies it GPU-only and a CPU-only runner never
compiles it. That is correct here and is the difference from the crossover
file: this one opens a device and trains on it.

COST. One 250,000 x 50 dense matrix, binned once and reused by every fit,
with three boosting rounds and eight leaves. Nothing here is timed and
nothing here is a benchmark; the shape is 250,000 rows because that is the
threshold under test and no smaller shape exercises it.

CHECKED AGAINST VACUITY, by breaking it. `_ROWS` was set to
`AUTO_GPU_MIN_ROWS - 1` and re-run on 2026-08-16:
`test_auto_actually_trains_on_the_accelerator` failed, with "device='auto'
produced the CPU trainer's model bit for bit at 249999 x 50, so it did not
reach the accelerator". So the passing result is a statement about the
threshold and not about the fixture. Anyone changing the shape here should
repeat that, because a fixture that sits on the wrong side of the gate it
tests is this repository's most-repeated test defect.
"""

from std.os import setenv
from std.sys import has_accelerator
from std.testing import assert_equal, assert_false, assert_true, TestSuite

from mojotrees.boosting import BoosterParams, SQUARED_ERROR
from mojotrees.device import AUTO_DEVICE, CPU_DEVICE, GPU_DEVICE
from mojotrees.device_policy import (
    AUTO_GPU_MIN_ROWS,
    M4_TRAINING_MIN_FEATURES,
    build_accelerator_target,
)
from mojotrees.model import Model
from mojotrees.tree import TreeParams
from mojotrees.trainset import Dataset, train_dataset

from support import _uniform

comptime _PRECISION = "MOJOTREES_DERIVATIVE_PRECISION"

# The shape under test. Exactly the floor, because a shape above it would
# pass whatever the floor were set to, and this file's whole subject is
# whether `auto` crosses at the number the policy says it does.
comptime _ROWS = AUTO_GPU_MIN_ROWS
comptime _FEATURES = M4_TRAINING_MIN_FEATURES

# Small enough that four fits are quick, large enough that the trees have
# real structure to disagree about.
comptime _ROUNDS = 3
comptime _LEAVES = 8

comptime _OBSERVED_M4_TARGET = String("metal:4-metal4")


def _on_the_measured_build() -> Bool:
    """Whether this binary targets the accelerator the crossover rule is
    scoped to. The rule is Metal-on-M4 only, so on any other accelerator
    `auto` correctly keeps the CPU and there is nothing here to prove."""
    return build_accelerator_target() == _OBSERVED_M4_TARGET


def _set(name: String, value: String):
    _ = setenv(name, value, True)


def _features() -> List[Float64]:
    """Column-major deterministic features, the same generator
    `tests/support.mojo` uses."""
    var out = List[Float64](capacity=_ROWS * _FEATURES)
    for k in range(_ROWS * _FEATURES):
        out.append(_uniform(UInt64(k)))
    return out^


def _target(features: List[Float64]) -> List[Float64]:
    """Strong, distinct per-feature effects, so the two backends are not
    deciding splits on knife-edge gain ties. A tie broken differently would
    change the tree *shape*, and this file is about leaf value bits."""
    var y = List[Float64](capacity=_ROWS)
    for r in range(_ROWS):
        var x0 = features[0 * _ROWS + r]
        var x1 = features[1 * _ROWS + r]
        var x2 = features[2 * _ROWS + r]
        var x3 = features[3 * _ROWS + r]
        y.append(4.0 * x0 - 3.0 * x1 + 2.0 * (x2 - 0.5) * (x3 - 0.5))
    return y^


def _identical(a: Model, b: Model) -> Bool:
    """Whether two fitted models are the same model, bit for bit.

    Structure and leaf values both, and the values compared as `Float64`
    equality, which for two finite doubles is bit equality. This is the
    observable the whole file rests on, so it is deliberately strict:
    nothing here is a tolerance.
    """
    if len(a.booster.trees) != len(b.booster.trees):
        return False
    if a.booster.base_score != b.booster.base_score:
        return False
    for t in range(len(a.booster.trees)):
        ref ta = a.booster.trees[t]
        ref tb = b.booster.trees[t]
        if len(ta.value) != len(tb.value):
            return False
        for i in range(len(ta.value)):
            if ta.feature[i] != tb.feature[i]:
                return False
            if ta.threshold_bin[i] != tb.threshold_bin[i]:
                return False
            if ta.value[i] != tb.value[i]:
                return False
    return True


def _fit(dataset: Dataset, device: Int) raises -> Model:
    return train_dataset(
        dataset,
        SQUARED_ERROR,
        BoosterParams(
            _ROUNDS, 0.1, TreeParams(_LEAVES, 20, 0.0, 1e-3, 0.0)
        ),
        device=device,
    )


def test_auto_actually_trains_on_the_accelerator() raises:
    """THE PROOF, and it is a model rather than a device code.

    `train_dataset` is the entry point a real caller uses and it resolves
    the device itself, so nothing here injects a decision. The three fits
    differ only in the `device` argument.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        if not _on_the_measured_build():
            print("skipped: not the accelerator the crossover rule covers")
            return
        _set(_PRECISION, "")
        var features = _features()
        var dataset = Dataset(features, _ROWS, _FEATURES, _target(features))

        var auto = _fit(dataset, AUTO_DEVICE)
        var gpu = _fit(dataset, GPU_DEVICE)
        var cpu = _fit(dataset, CPU_DEVICE)

        # The two halves, and neither is sufficient alone. If the CPU and
        # GPU trainers ever did agree bit for bit at this scale the second
        # assertion fails and this test stops being able to prove anything,
        # which is the right way for it to break: loudly, rather than by
        # passing on a comparison that no longer separates the backends.
        assert_false(
            _identical(auto, cpu),
            String(
                "device='auto' produced the CPU trainer's model bit for"
                " bit at ",
                _ROWS,
                " x ",
                _FEATURES,
                ", so it did not reach the accelerator",
            ),
        )
        assert_true(
            _identical(auto, gpu),
            String(
                "device='auto' produced neither the CPU model nor the GPU"
                " model at ",
                _ROWS,
                " x ",
                _FEATURES,
            ),
        )

        # And the fits are not degenerate: a model with no splits would
        # compare equal to anything and prove nothing.
        assert_equal(len(auto.booster.trees), _ROUNDS)
        var splits = 0
        for t in range(len(auto.booster.trees)):
            for i in range(len(auto.booster.trees[t].feature)):
                if auto.booster.trees[t].feature[i] >= 0:
                    splits += 1
        assert_true(splits > 0)


def test_float64_derivatives_route_auto_to_the_host_and_it_runs_there() raises:
    """The precision route, proved the same way and at a shape that cannot
    be confused with the threshold.

    250,000 rows is exactly `AUTO_GPU_MIN_ROWS`, so with the flag off this
    identical call reaches the accelerator (the test above). With
    `MOJOTREES_DERIVATIVE_PRECISION=float64` it must reach the host
    instead, and it must do so by *routing* rather than by raising: a
    caller who wrote `auto` asked us to pick a backend, and picking the one
    that can honor the request is the answer. Precision is a capability, so
    it outranks the crossover and no shape comparison happens at all.

    The three assertions are what separate that from the alternatives. It
    equals the host's float64 model, so it ran on the host. The host's
    float64 model is not the host's float32 model, so the flag reached the
    arithmetic and the first assertion is not comparing two identical
    things. And it is not the float32 `auto` model, which is the
    accelerator's.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        if not _on_the_measured_build():
            print("skipped: not the accelerator the crossover rule covers")
            return
        _set(_PRECISION, "")
        var features = _features()
        var dataset = Dataset(features, _ROWS, _FEATURES, _target(features))
        var auto_float32 = _fit(dataset, AUTO_DEVICE)
        var cpu_float32 = _fit(dataset, CPU_DEVICE)

        _set(_PRECISION, "float64")
        var auto_float64 = _fit(dataset, AUTO_DEVICE)
        var cpu_float64 = _fit(dataset, CPU_DEVICE)
        _set(_PRECISION, "")

        assert_true(
            _identical(auto_float64, cpu_float64),
            String(
                "device='auto' under float64 did not produce the host"
                " trainer's model, so it did not route to the host"
            ),
        )
        assert_false(
            _identical(cpu_float64, cpu_float32),
            String(
                "float64 and float32 host fits agree bit for bit, so the"
                " assertion above cannot distinguish the backends"
            ),
        )
        assert_false(
            _identical(auto_float64, auto_float32),
            String(
                "device='auto' produced the same model with the precision"
                " flag on and off, which is the defect this route fixes"
            ),
        )


def test_explicit_gpu_under_float64_refuses_rather_than_routing() raises:
    """The asymmetry, at the entry point rather than in the policy.

    `tests/test_device.mojo` pins the decision; this pins what a caller
    gets. An explicit `device='gpu'` must raise, because a silent downgrade
    from a named backend is the defect the refusal exists for. It is
    asserted here, in the GPU file, because on a build with no accelerator
    the request would be refused for a different reason and the test would
    pass without meaning it.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        _set(_PRECISION, "")
        var n_rows = 2_000
        var n_features = 4
        var features = List[Float64](capacity=n_rows * n_features)
        for k in range(n_rows * n_features):
            features.append(_uniform(UInt64(k)))
        var label = List[Float64](capacity=n_rows)
        for r in range(n_rows):
            label.append(features[r] * 2.0 - features[n_rows + r])
        var dataset = Dataset(features, n_rows, n_features, label^)

        # With the flag off, an explicit gpu request at this small shape is
        # honored: no crossover rule is consulted for an explicit request.
        var forced = _fit(dataset, GPU_DEVICE)
        assert_equal(len(forced.booster.trees), _ROUNDS)

        _set(_PRECISION, "float64")
        var raised = False
        var message = String("")
        try:
            _ = _fit(dataset, GPU_DEVICE)
        except e:
            raised = True
            message = String(e)
        _set(_PRECISION, "")
        assert_true(raised, "device='gpu' under float64 must raise")
        assert_true(message.find("float64") >= 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
