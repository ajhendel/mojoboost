"""Sparse input on the accelerator, through the estimators and `train`.

An explicit `device="gpu"` on a SciPy matrix reaches the native sparse GPU
trainer (`fit_csc` -> `train_gpu_sparse`), never densifying; `auto` keeps
the CPU. The GPU tests skip without an accelerator. The categorical
forwarding test runs everywhere: the sparse binding used to drop the
categorical indices the estimator sent, so a `categorical_feature` on
sparse input was binned as numeric in silence.
"""

import numpy as np
import pytest

from mojotrees import MojoTreesClassifier, MojoTreesRegressor, inspection

scipy_sparse = pytest.importorskip("scipy.sparse")


def _sparse_regression(n_rows=600, n_features=8, density=0.2, seed=3):
    gen = np.random.default_rng(seed)
    X = gen.random((n_rows, n_features))
    X = np.where(gen.random(X.shape) < density, X * 4.0 - 2.0, 0.0)
    y = X @ (1.0 + 0.37 * np.arange(n_features)) / 4.0
    y += 0.05 * (gen.random(n_rows) - 0.5)
    return X, y


def _walk(node):
    yield node
    for key in ("left_child", "right_child"):
        if key in node:
            yield from _walk(node[key])


def _gpu_or_skip():
    from mojotrees import gpu_available

    if not gpu_available():
        pytest.skip("no accelerator available for training")


def test_sparse_categorical_feature_reaches_the_sparse_trainer():
    gen = np.random.default_rng(5)
    codes = gen.integers(0, 6, size=500).astype(float)
    other = np.where(gen.random(500) < 0.3, gen.random(500), 0.0)
    X = scipy_sparse.csc_matrix(np.column_stack([codes, other]))
    y = np.where(codes % 2 == 0, 1.0, -1.0) + 0.1 * other
    est = MojoTreesRegressor(
        n_estimators=5, num_leaves=5, categorical_feature=[0]
    ).fit(X, y)
    assert est.device_ == "cpu"
    dump = inspection.dump_model(est)
    assert dump["feature_infos"][0]["type"] == "categorical"
    splits = [
        node
        for tree in dump["tree_info"]
        for node in _walk(tree["tree_structure"])
        if "split_index" in node and node["split_feature"] == 0
    ]
    assert splits, "the sparse fit never split on the categorical feature"
    for node in splits:
        assert node["decision_type"] == "=="


def test_sparse_auto_keeps_the_cpu():
    X, y = _sparse_regression()
    est = MojoTreesRegressor(n_estimators=3, device="auto").fit(
        scipy_sparse.csc_matrix(X), y
    )
    assert est.device_ == "cpu"


def test_sparse_gpu_regressor_matches_the_cpu_fit():
    _gpu_or_skip()
    X, y = _sparse_regression()
    Xs = scipy_sparse.csc_matrix(X)
    cpu = MojoTreesRegressor(n_estimators=8, device="cpu").fit(Xs, y)
    gpu = MojoTreesRegressor(n_estimators=8, device="gpu").fit(Xs, y)
    assert cpu.device_ == "cpu"
    assert gpu.device_ == "gpu"
    pc = np.asarray(cpu.predict(Xs))
    pg = np.asarray(gpu.predict(Xs))
    # The tolerance the dense GPU trainer is held to against the CPU one:
    # Float32 histograms on the device, Float64 on the host.
    assert np.abs(pc - pg).mean() <= 1e-3
    assert np.allclose(pg, gpu.predict(X))


def test_sparse_gpu_classifier_binary_and_multiclass():
    _gpu_or_skip()
    X, y = _sparse_regression()
    Xs = scipy_sparse.csc_matrix(X)
    binary = (y > 0).astype(int)
    est = MojoTreesClassifier(n_estimators=6, device="gpu").fit(Xs, binary)
    assert est.device_ == "gpu"
    proba = np.asarray(est.predict_proba(Xs))
    assert proba.shape == (len(y), 2)
    assert (est.predict(Xs) == binary).mean() > 0.7

    three = np.digitize(y, [-0.2, 0.2])
    multi = MojoTreesClassifier(n_estimators=4, device="gpu").fit(Xs, three)
    assert multi.device_ == "gpu"
    assert np.asarray(multi.predict_proba(Xs)).shape == (len(y), 3)


def test_sparse_gpu_refuses_bundling():
    _gpu_or_skip()
    X, y = _sparse_regression()
    est = MojoTreesRegressor(n_estimators=3, device="gpu", enable_bundle=True)
    with pytest.raises((RuntimeError, ValueError), match="bundl"):
        est.fit(scipy_sparse.csc_matrix(X), y)
