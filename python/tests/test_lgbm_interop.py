"""LightGBM model-file interop through `mojotrees.lgbm_model_io`.

Every check goes through the four extension entry points the module binds
(`lgbm_interop_status`, `lgbm_file_unsupported_reason`, `lgbm_import_file`,
`lgbm_export_file`), which the integration round connected. The converter
itself is native and experimental; what is checked here is that the door
is open and that a model written by mojotrees crosses to LightGBM's text
and back predicting the same values.

Runs under pytest and as a script (`python test_lgbm_interop.py`).
"""

import os
import tempfile
import warnings

import numpy as np

import mojotrees
from mojotrees import Dataset, train
from mojotrees import _mojotrees as ext


def _booster(learning_rate):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 4))
    y = X[:, 0] * 2.0 - X[:, 2] + rng.normal(scale=0.1, size=200)
    params = {
        "objective": "regression",
        "learning_rate": learning_rate,
        "num_leaves": 7,
    }
    return train(params, Dataset(X, label=y), num_boost_round=5), X


def test_submodule_is_reachable_lazily():
    assert "lgbm_model_io" in mojotrees._LAZY_SUBMODULES
    module = mojotrees.lgbm_model_io
    assert module is mojotrees.lgbm_model_io
    for name in (
        "lgbm_interop_status",
        "lgbm_file_unsupported_reason",
        "lgbm_import_file",
        "lgbm_export_file",
    ):
        assert callable(getattr(ext, name)), name


def test_status_is_the_native_sentence_verbatim():
    from mojotrees import lgbm_model_io

    status = lgbm_model_io.interop_status()
    assert status == ext.lgbm_interop_status()
    assert status.startswith("experimental")


def test_unsupported_reason_names_a_missing_file():
    from mojotrees import lgbm_model_io

    with tempfile.TemporaryDirectory() as tmp:
        reason = lgbm_model_io.unsupported_reason(os.path.join(tmp, "no.txt"))
    assert reason != ""


def test_round_trip_predicts_the_same():
    from mojotrees import lgbm_model_io

    booster, X = _booster(learning_rate=1.0)
    want = booster.predict(X)
    with tempfile.TemporaryDirectory() as tmp:
        lgbm_path = os.path.join(tmp, "model.lgbm.txt")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report = lgbm_model_io.save_lightgbm_model(booster, lgbm_path)
        assert any("experimental" in str(w.message).lower() for w in caught)
        assert report["n_trees"] == 5
        assert report["n_classes"] == 1
        assert report["shrinkage_folded"] is False
        # A regression model's base score is the label mean, which the
        # LightGBM format keeps inside the first tree's leaves; folding it
        # there is a rounding, so the round trip is close, not bit-exact.
        assert report["base_score_folded"] is True
        assert lgbm_model_io.unsupported_reason(lgbm_path) == ""
        back = lgbm_model_io.load_lightgbm_model(lgbm_path)
    got = back.predict(X)
    np.testing.assert_allclose(got, want, rtol=1e-12)


def test_import_report_describes_the_file():
    from mojotrees import lgbm_model_io

    booster, X = _booster(learning_rate=0.3)
    with tempfile.TemporaryDirectory() as tmp:
        lgbm_path = os.path.join(tmp, "model.lgbm.txt")
        native = os.path.join(tmp, "back.mbst")
        out = lgbm_model_io.save_lightgbm_model(booster, lgbm_path)
        assert out["shrinkage_folded"] is True
        report = lgbm_model_io.convert_to_mojotrees(lgbm_path, native)
        assert report["n_features"] == 4
        assert report["n_trees"] == 5
        assert report["n_classes"] == 1
        assert report["n_edges"] > 0
        assert os.path.exists(native)
        back = mojotrees.Booster(model_file=native)
    np.testing.assert_allclose(back.predict(X), booster.predict(X), rtol=1e-12)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
