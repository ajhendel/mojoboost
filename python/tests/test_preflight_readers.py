"""The Python readers of the pre-flight, prediction-capability, distributed
GPU status, chunk-count, and device-validation bindings.

Every check compares a Python-side answer with the extension's own or
exercises a binding Python did not call before the integration round:
`efb_check`, `efb_defaults`, `extra_params_check`, `extra_option_supported`,
`forced_splits_check`, `gpu_predict_capability`, `distributed_gpu_status`,
`dataset_chunks_num_data`, and the `gpu_validation_*` surface (the last on
an accelerator only; on a machine without one it checks the refusal).

Runs under pytest and as a script (`python test_preflight_readers.py`).
"""

import numpy as np

import mojotrees
from mojotrees import MojoTreesRegressor, _sequence, dask, device_selection
from mojotrees import _mojotrees as ext
from mojotrees import gpu_validation, preflight
from mojotrees.basic import Dataset


def _data(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = X[:, 0] * 2.0 + rng.normal(scale=0.1, size=n)
    return X, y


def test_bundling_defaults_match_the_estimator():
    native = preflight.bundling_defaults()
    est = MojoTreesRegressor()
    for key, value in native.items():
        assert getattr(est, key) == value, key
    knobs = preflight.bundling_knobs(max_conflict_rate=None)
    assert knobs["max_conflict_rate"] == native["max_conflict_rate"]
    assert isinstance(knobs["bundle_missing"], int)


def test_native_preflight_names_a_bad_knob_before_fitting():
    X, y = _data()
    try:
        MojoTreesRegressor(n_estimators=2, max_conflict_rate=2.0).fit(X, y)
    except ValueError as exc:
        assert "max_conflict_rate" in str(exc)
    else:
        raise AssertionError("max_conflict_rate=2.0 was accepted")


def test_forced_splits_check_reports_shape_and_refuses_bad_features():
    shape = preflight.check_forced_splits(
        {"feature": 0, "threshold": 0.5}, n_features=4
    )
    assert shape == {"n_nodes": 1, "depth": 0}
    try:
        preflight.check_forced_splits({"feature": 9, "threshold": 0.5}, 4)
    except Exception as exc:
        assert "feature" in str(exc)
    else:
        raise AssertionError("feature 9 of 4 was accepted")


def test_unimplemented_option_gets_the_native_message():
    # `forcedsplits_filename` is the probe now. It names a real LightGBM
    # feature reachable from the Mojo API but not carryable in a parameter
    # string, which is what "unimplemented option" is for.
    message = preflight.unimplemented_option_message("forcedsplits_filename")
    assert message is not None and "not implemented" in message
    assert preflight.unimplemented_option_message("num_leaves") is None
    # `feature_pre_filter` is no longer one of them. `false` is the behavior
    # mojotrees has, so it is accepted rather than refused; `true` is still
    # refused because prefiltering can change the trees and not only their
    # cost.
    assert preflight.unimplemented_option_message("feature_pre_filter") is None
    X, y = _data()
    mojotrees.train(
        {"feature_pre_filter": False}, Dataset(X, label=y), num_boost_round=1
    )
    try:
        mojotrees.train(
            {"feature_pre_filter": True}, Dataset(X, label=y), num_boost_round=1
        )
    except ValueError as exc:
        assert "feature_pre_filter" in str(exc)
    else:
        raise AssertionError("feature_pre_filter=true was accepted")


def test_explain_predict_device_reads_the_capability_record():
    X, _ = _data()
    report = device_selection.explain_predict_device(X)
    native = ext.gpu_predict_capability(
        {"n_rows": 120, "n_features": 4, "n_outputs": 1, "n_bins": 0, "sparse": 0}
    )
    assert report.supported == bool(native[0])
    assert report.block_code == int(native[1])
    assert report.to_dict()["shape"]["n_features"] == 4


def test_distributed_gpu_status_is_the_native_record():
    status = dask.distributed_gpu_status()
    native = ext.distributed_gpu_status()
    assert status["available"] == bool(native["available"])
    assert status["gates"] == [str(g) for g in native["gates"]]


def test_batches_train_and_the_chunk_count_agrees():
    X, y = _data()
    ds = Dataset(_sequence.Batches([X[:50], X[50:]]), label=y)
    booster = mojotrees.train(
        {"objective": "regression", "num_leaves": 4}, ds, num_boost_round=2
    )
    assert booster.num_trees() == 2
    assert ds.num_data() == 120


def test_device_metric_support_and_device_eval():
    X, y = _data()
    has_l2, l2_matches = gpu_validation.device_metric_support("l2")
    has_mape, _ = gpu_validation.device_metric_support("mape")
    assert not has_mape
    ds = Dataset(X, label=y, free_raw_data=False)
    booster = mojotrees.train(
        {"objective": "regression", "num_leaves": 4}, ds, num_boost_round=3
    )
    host = booster.eval(ds, "v", "l2")[0][2]
    if not mojotrees.gpu_available():
        try:
            booster.eval(ds, "v", "l2", device="gpu")
        except Exception:
            return
        raise AssertionError("device eval ran with no accelerator")
    if has_l2 and l2_matches:
        device = booster.eval(ds, "v", "l2", device="gpu")[0][2]
        # Float32 device scores; the definition is the host's term for term.
        assert abs(device - host) <= 1e-6 * max(1.0, abs(host))
    resident = gpu_validation.GpuValidation.open(booster, X, y)
    resident.accumulate(booster)
    raw = np.asarray(resident.raw())
    assert np.allclose(raw, booster.predict(X, raw_score=True), atol=1e-6)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
