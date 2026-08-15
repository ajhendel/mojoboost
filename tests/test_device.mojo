"""Device selection: parsing, resolution policy, and trainer equivalence.

Resolution is exercised on any machine: `MOJOTREES_DISABLE_GPU` simulates a
CPU-only build so the unavailable-GPU path is covered on accelerators too,
and `MOJOTREES_AUTO_MIN_CELLS` enables the otherwise-disabled `auto` size
heuristic so both sides of it are covered. Tests that actually train on the
GPU skip (passing) when no accelerator is present.
"""

from std.os import remove, setenv
from std.sys import has_accelerator
from std.testing import (
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
    TestSuite,
)

from mojotrees.boosting import BoosterParams, SQUARED_ERROR
from mojotrees.device import (
    AUTO_DEVICE,
    CPU_DEVICE,
    GPU_DEVICE,
    device_name,
    env_auto_min_cells,
    gpu_available,
    parse_device,
    resolve_device,
)
from mojotrees.model import fit, fit_multiclass
from mojotrees.serialize import load_model, save_model
from mojotrees.tree import TreeParams
from support import _make_features as _features

comptime _DISABLE_GPU = "MOJOTREES_DISABLE_GPU"
comptime _AUTO_MIN_CELLS = "MOJOTREES_AUTO_MIN_CELLS"
comptime _TMP_PATH = "./.test_device_roundtrip.tmp"


def _set_env(name: String, value: String):
    """Set an override; the empty string restores the default (both knobs
    treat unset and empty alike)."""
    _ = setenv(name, value, True)


def _target(features: List[Float64], n_rows: Int) -> List[Float64]:
    """Distinct per-feature effects, so CPU and GPU split decisions do not
    sit on knife-edge gain ties."""
    var y = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        y.append(
            4.0 * features[0 * n_rows + r]
            - 3.0 * features[1 * n_rows + r]
            + 2.0
            * (features[2 * n_rows + r] - 0.5)
            * (features[3 * n_rows + r] - 0.5)
        )
    return y^


def _labels(features: List[Float64], n_rows: Int, n_classes: Int) -> List[Int]:
    var labels = List[Int](capacity=n_rows)
    for r in range(n_rows):
        var k = Int(features[0 * n_rows + r] * Float64(n_classes))
        if k >= n_classes:
            k = n_classes - 1
        labels.append(k)
    return labels^


def test_device_names_round_trip() raises:
    assert_equal(parse_device("cpu"), CPU_DEVICE)
    assert_equal(parse_device("gpu"), GPU_DEVICE)
    assert_equal(parse_device("auto"), AUTO_DEVICE)
    assert_equal(device_name(CPU_DEVICE), "cpu")
    assert_equal(device_name(GPU_DEVICE), "gpu")
    assert_equal(device_name(AUTO_DEVICE), "auto")


def test_invalid_device_raises() raises:
    with assert_raises():
        _ = parse_device("cuda")
    with assert_raises():
        _ = parse_device("CPU")
    with assert_raises():
        _ = parse_device("")
    with assert_raises():
        _ = device_name(7)
    with assert_raises():
        _ = resolve_device(7, 100, 10, 1)


def test_cpu_always_resolves_to_cpu() raises:
    # Large workload, no accelerator, either way: cpu means cpu.
    assert_equal(resolve_device(CPU_DEVICE, 10_000_000, 100, 1), CPU_DEVICE)
    _set_env(_DISABLE_GPU, "1")
    assert_equal(resolve_device(CPU_DEVICE, 10_000_000, 100, 1), CPU_DEVICE)
    _set_env(_DISABLE_GPU, "")
    assert_equal(resolve_device(CPU_DEVICE, 10, 2, 1), CPU_DEVICE)


def test_explicit_gpu_raises_when_unavailable() raises:
    _set_env(_DISABLE_GPU, "1")
    assert_false(gpu_available())
    # No silent fallback: the request fails instead of returning CPU.
    with assert_raises():
        _ = resolve_device(GPU_DEVICE, 1_000_000, 100, 1)
    _set_env(_DISABLE_GPU, "")

    if gpu_available():
        assert_equal(resolve_device(GPU_DEVICE, 10, 2, 1), GPU_DEVICE)
    else:
        with assert_raises():
            _ = resolve_device(GPU_DEVICE, 10, 2, 1)


def test_gpu_accepts_multiclass() raises:
    # Multiclass is no longer an unsupported workload: it grows one tree per
    # class per round through train_multiclass_gpu. Resolution therefore
    # turns only on whether an accelerator is present, exactly as it does
    # for single-output training.
    comptime if not has_accelerator():
        with assert_raises():
            _ = resolve_device(GPU_DEVICE, 1_000, 10, 3)
    else:
        assert_equal(resolve_device(GPU_DEVICE, 1_000, 10, 3), GPU_DEVICE)


def test_auto_chooses_cpu_by_default() raises:
    assert_equal(env_auto_min_cells(), -1)
    assert_equal(resolve_device(AUTO_DEVICE, 10, 2, 1), CPU_DEVICE)
    assert_equal(
        resolve_device(AUTO_DEVICE, 10_000_000, 100, 1), CPU_DEVICE
    )
    _set_env(_DISABLE_GPU, "1")
    assert_equal(
        resolve_device(AUTO_DEVICE, 10_000_000, 100, 1), CPU_DEVICE
    )
    _set_env(_DISABLE_GPU, "")


def test_auto_honors_the_size_threshold() raises:
    _set_env(_AUTO_MIN_CELLS, "10000")
    assert_equal(env_auto_min_cells(), 10_000)

    var expected_big = GPU_DEVICE if gpu_available() else CPU_DEVICE
    # Below the threshold: CPU regardless of hardware.
    assert_equal(resolve_device(AUTO_DEVICE, 50, 10, 1), CPU_DEVICE)
    # At and above it: GPU when one is available.
    assert_equal(resolve_device(AUTO_DEVICE, 1_000, 10, 1), expected_big)
    assert_equal(resolve_device(AUTO_DEVICE, 100_000, 10, 1), expected_big)
    # Multiclass is inside the GPU path now, so it obeys the same threshold
    # as single-output training rather than being forced to the CPU.
    assert_equal(resolve_device(AUTO_DEVICE, 50, 10, 3), CPU_DEVICE)
    assert_equal(resolve_device(AUTO_DEVICE, 100_000, 10, 3), expected_big)

    # A disabled accelerator sends everything back to the CPU.
    _set_env(_DISABLE_GPU, "1")
    assert_equal(resolve_device(AUTO_DEVICE, 100_000, 10, 1), CPU_DEVICE)
    _set_env(_DISABLE_GPU, "")

    # Unparsable and negative overrides fall back to disabled.
    _set_env(_AUTO_MIN_CELLS, "many")
    assert_equal(env_auto_min_cells(), -1)
    _set_env(_AUTO_MIN_CELLS, "-5")
    assert_equal(resolve_device(AUTO_DEVICE, 100_000, 10, 1), CPU_DEVICE)
    _set_env(_AUTO_MIN_CELLS, "")
    assert_equal(env_auto_min_cells(), -1)


def test_fit_cpu_and_auto_agree() raises:
    var n_rows = 400
    var n_features = 5
    var features = _features(n_rows, n_features)
    var target = _target(features, n_rows)
    var params = BoosterParams(10, 0.1, TreeParams.default())

    var cpu = fit(
        features, n_rows, n_features, target, SQUARED_ERROR, params,
        device=CPU_DEVICE,
    )
    var auto = fit(
        features, n_rows, n_features, target, SQUARED_ERROR, params,
        device=AUTO_DEVICE,
    )
    # auto resolves to the CPU trainer here, so this is bit-exact.
    assert_equal(len(cpu.booster.trees), len(auto.booster.trees))
    for r in range(n_rows):
        var row = List[Float64](capacity=n_features)
        for f in range(n_features):
            row.append(features[f * n_rows + r])
        assert_equal(cpu.predict(row), auto.predict(row))


def test_fit_rejects_unknown_device() raises:
    var features = _features(50, 4)
    var target = _target(features, 50)
    var params = BoosterParams(2, 0.1, TreeParams.default())
    with assert_raises():
        _ = fit(
            features, 50, 4, target, SQUARED_ERROR, params, device=42
        )


def test_fit_multiclass_device_selection() raises:
    var n_rows = 300
    var n_features = 4
    var features = _features(n_rows, n_features)
    var labels = _labels(features, n_rows, 3)
    var params = BoosterParams(5, 0.1, TreeParams.default())

    # auto resolves to the CPU while the size heuristic ships disabled.
    var model = fit_multiclass(
        features, n_rows, n_features, labels, 3, params, device=AUTO_DEVICE
    )
    assert_equal(model.booster.n_classes, 3)

    # Explicit gpu trains on the device when there is one, and raises only
    # for the absent accelerator, not for the workload.
    comptime if not has_accelerator():
        with assert_raises():
            _ = fit_multiclass(
                features, n_rows, n_features, labels, 3, params,
                device=GPU_DEVICE,
            )
    else:
        var on_gpu = fit_multiclass(
            features, n_rows, n_features, labels, 3, params,
            device=GPU_DEVICE,
        )
        assert_equal(on_gpu.booster.n_classes, 3)
        assert_equal(
            len(on_gpu.booster.trees), len(model.booster.trees)
        )


def test_fit_gpu_matches_cpu() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 2_000
        var n_features = 6
        var features = _features(n_rows, n_features)
        var target = _target(features, n_rows)
        var params = BoosterParams(10, 0.1, TreeParams(15, 20, 1.0, 1e-3))

        var cpu = fit(
            features, n_rows, n_features, target, SQUARED_ERROR, params,
            device=CPU_DEVICE,
        )
        var gpu = fit(
            features, n_rows, n_features, target, SQUARED_ERROR, params,
            device=GPU_DEVICE,
        )
        assert_equal(len(cpu.booster.trees), len(gpu.booster.trees))
        # GPU histograms are Float32 fixed-point, so agreement is
        # tolerance-based (see train_gpu.mojo).
        for r in range(n_rows):
            var row = List[Float64](capacity=n_features)
            for f in range(n_features):
                row.append(features[f * n_rows + r])
            assert_true(abs(cpu.predict(row) - gpu.predict(row)) <= 1e-3)


def test_gpu_trained_model_round_trips() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 500
        var n_features = 4
        var features = _features(n_rows, n_features)
        var target = _target(features, n_rows)
        var params = BoosterParams(8, 0.1, TreeParams.default())
        var model = fit(
            features, n_rows, n_features, target, SQUARED_ERROR, params,
            device=GPU_DEVICE,
        )

        # The device is a training choice, not part of the model, so the
        # file format is unchanged and the round trip stays bit-exact.
        save_model(model, _TMP_PATH)
        var loaded = load_model(_TMP_PATH)
        remove(_TMP_PATH)
        assert_equal(
            len(model.booster.trees), len(loaded.booster.trees)
        )
        for r in range(n_rows):
            var row = List[Float64](capacity=n_features)
            for f in range(n_features):
                row.append(features[f * n_rows + r])
            assert_equal(model.predict(row), loaded.predict(row))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
