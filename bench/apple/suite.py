"""Apple silicon benchmark suite for mojoboost.

This is the runner for the protocol in docs/APPLE_GPU_BENCHMARK_PROTOCOL.md.
It fits eight workloads with up to four engines (mojoboost on CPU, mojoboost
on GPU, LightGBM, XGBoost), records every phase, quality, memory, thermal,
and power quantity the protocol defines, and writes one JSON record per run
validated against bench/apple/schema.json.

    python bench/apple/suite.py --list                 # the catalog
    python bench/apple/suite.py --plan                 # what a run would do
    python bench/apple/suite.py --self-check           # static validation
    python bench/apple/suite.py --run --scale smoke    # a real run

Three rules this file exists to enforce.

1. Nothing is estimated. A quantity that could not be measured is null and
   carries a reason. Energy in particular is measured with powermetrics or
   it is absent; it is never derived from CPU time or from a TDP number.
2. Nothing is dropped. An engine that cannot express a workload produces a
   record with status `unsupported`, not a gap in the table.
3. Every measurement runs in its own process. That is what makes peak
   resident memory attributable, thread environment clean, and first-fit
   cost (library import, kernel compilation, device context creation)
   visible instead of amortized into whichever engine happened to go first.

The suite measures. It does not conclude. Reading a run means reading
`conditions` first: a record whose idle gate failed or whose thermal samples
show throttling is a record of a machine under load, whatever its numbers
say.

No results are checked into this repository yet, and none should be quoted
from a run this file did not produce.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import plistlib
import re
import shutil
import statistics
import subprocess
import sys
import time

SUITE_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0.0"
PROTOCOL_VERSION = "1.0.0"

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
SCHEMA_PATH = os.path.join(HERE, "schema.json")
DEFAULT_RESULTS_DIR = os.path.join(HERE, "results")

#: Load average above which the idle gate refuses to run. One core busy on a
#: ten core machine is already enough to move a histogram timing.
MAX_LOAD_AVERAGE = 0.75

#: Process names whose presence means the machine is building something.
BUSY_PROCESS_PATTERNS = ("mojo", "pixi", "pytest", "cmake", "clang", "Xcode")

#: A non-warmup repetition set whose slowest run is more than this multiple
#: of its fastest is unstable. The protocol reruns rather than reports.
MAX_SPREAD_RATIO = 1.25

MASK64 = (1 << 64) - 1


# --------------------------------------------------------------------------
# Workload catalog
# --------------------------------------------------------------------------

def _workload(**kw):
    """One catalog entry with the shared fitting parameters filled in.

    The parameters are mojoboost's defaults, which are LightGBM's defaults,
    which is what makes a cross-engine comparison a comparison of
    implementations rather than of configurations. `docs/LIGHTGBM_PARITY.md`
    is the statement of where the two libraries' defaults still differ, and
    `_engine_params` below is where those differences are forced to agree.
    """
    base = {
        "kind": "dense",
        "task": "regression",
        "valid_rows": 20_000,
        "rounds": 100,
        "num_leaves": 31,
        "max_bin": 255,
        "learning_rate": 0.1,
        "min_data_in_leaf": 20,
        "lambda_l2": 1.0,
        "n_classes": None,
        "nonzeros_per_row": None,
        "missing_fraction": None,
        "categorical_features": None,
        "categorical_cardinality": None,
        "fits": None,
        "predict_batches": None,
        "predict_batch_rows": None,
        "seed": 20260814,
    }
    base.update(kw)
    return base


#: The eight workloads. Shapes are chosen to straddle the region where a
#: GPU stops being launch-overhead bound, not to flatter either backend, and
#: the protocol requires all eight or none: a suite run that reports only the
#: large dense shape is a marketing exercise.
WORKLOADS = {
    # Small enough that per-launch and per-round fixed costs dominate. This
    # is the shape a GPU is expected to lose on, and reporting it is the
    # point.
    "w1_small_dense": _workload(rows=20_000, features=20, valid_rows=5_000),

    # The middle of the range, where the crossover is expected to sit.
    "w2_medium_dense": _workload(rows=200_000, features=100),

    # Large enough to matter and small enough to hold in the memory of a
    # base configuration Mac. Anything larger belongs in a separate
    # memory-limit study, not here.
    "w3_large_dense": _workload(rows=1_000_000, features=50, valid_rows=50_000),

    # Genuinely sparse, generated the way bench/bench_sparse.mojo does.
    # mojoboost has no sparse GPU kernel, so this workload's job is to
    # record that as `unsupported` rather than to hide it.
    "w4_sparse": _workload(
        kind="sparse", rows=200_000, features=500, nonzeros_per_row=10,
        valid_rows=20_000,
    ),

    # Multiclass multiplies the per-round work by the class count and the
    # gradient traffic with it, which is the shape most sensitive to
    # transfer cost.
    "w5_multiclass": _workload(
        task="multiclass", rows=200_000, features=50, n_classes=5,
    ),

    # Missing values and unordered categories, the two paths whose split
    # search differs from the plain numerical scan.
    "w6_missing_categorical": _workload(
        kind="missing_categorical", rows=200_000, features=50,
        missing_fraction=0.1, categorical_features=5,
        categorical_cardinality=40,
    ),

    # The same data fitted several times in one process. The gap between
    # the first fit and the rest is the compilation, allocation, and device
    # setup a user pays once, and it is the number a persistent GPU session
    # is meant to move.
    "w7_repeated_fit": _workload(rows=200_000, features=50, rounds=50, fits=5),

    # Prediction only, on a model fitted inside the same process, in
    # batches, because inference latency is a different question from
    # training throughput and is answered by a different part of the code.
    "w8_prediction": _workload(
        rows=200_000, features=50, rounds=100, valid_rows=200_000,
        predict_batches=10, predict_batch_rows=20_000,
    ),
}

#: Reduced shapes for a smoke run. A smoke run proves the harness works. It
#: is not a result and the protocol forbids quoting one.
SMOKE_SCALE = {
    "rows": 0.02, "valid_rows": 0.1, "rounds": 0.1, "fits": 1.0,
    "predict_batches": 0.2,
}


def scaled_workloads(scale):
    """The catalog at `full` or `smoke` scale, ids intact."""
    if scale == "full":
        return {k: dict(v) for k, v in WORKLOADS.items()}
    out = {}
    for wid, wl in WORKLOADS.items():
        small = dict(wl)
        for key, factor in SMOKE_SCALE.items():
            if small.get(key):
                small[key] = max(1, int(small[key] * factor))
        small["_smoke"] = True
        out[wid] = small
    return out


# --------------------------------------------------------------------------
# Deterministic data generation
# --------------------------------------------------------------------------
#
# The same counter-based splitmix64 stream bench/bench_train.mojo and
# bench/bench_lightgbm.py use, so a dataset here is the dataset those
# drivers produce and no file is exchanged between engines. Every engine in
# a measurement gets arrays generated from the same seed and reports the
# same `data_digest`; a digest mismatch is the check that they did.

def _splitmix64(x):
    import numpy as np

    with np.errstate(over="ignore"):
        z = (x + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(MASK64)
        z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & np.uint64(MASK64)
        z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & np.uint64(MASK64)
        return z ^ (z >> np.uint64(31))


def _uniform(counter):
    import numpy as np

    return (_splitmix64(counter) >> np.uint64(11)).astype(np.float64) * (
        1.0 / 9007199254740992.0
    )


def _counters(start, count):
    import numpy as np

    return np.arange(start, start + count, dtype=np.uint64)


def _signal(X):
    """The target bench_train.mojo uses: linear, interaction, and quadratic
    terms of the first four features, everything else noise."""
    x0, x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    return 5.0 * x0 + 4.0 * x1 * x2 + 3.0 * (x3 - 0.5) * (x3 - 0.5)


def _sigmoid(x):
    import numpy as np

    return np.where(x >= 0.0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _add_repo_python_path():
    """Put the source tree's `python/` directory ahead of site-packages, the
    way python/test_python_api.py and python/tests/conftest.py do, so a run
    measures the working copy. Build the extension first with
    bindings/build.sh; an unbuilt tree reports the import error as
    `unsupported` rather than quietly measuring nothing."""
    src = os.path.join(REPO_ROOT, "python")
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)


def _digest(*arrays):
    h = hashlib.blake2b(digest_size=16)
    for a in arrays:
        if a is None:
            continue
        h.update(bytes(str(a.dtype), "ascii"))
        h.update(bytes(str(a.shape), "ascii"))
        h.update(a.tobytes())
    return h.hexdigest()


def generate(wl):
    """Training and validation data for one workload.

    Returns a dict with `X`, `y`, `X_valid`, `y_valid`, `categorical` (the
    column indices declared categorical, possibly empty), `digest`, and
    `data_bytes`, the exact size of the arrays computed from the shapes.
    """
    import numpy as np

    n_total = int(wl["rows"]) + int(wl["valid_rows"])
    n_feat = int(wl["features"])
    seed = np.uint64(wl["seed"]) * np.uint64(0x100000000)

    if wl["kind"] == "sparse":
        data = _sparse_matrix(wl, n_total, n_feat, seed)
        X_all, dense_probe = data["X"], data["probe"]
        y_all = _labels(wl, dense_probe, n_total, n_feat, seed)
        digest = _digest(X_all.data, X_all.indices.astype(np.int64), y_all)
        data_bytes = int(
            X_all.data.nbytes + X_all.indices.nbytes + X_all.indptr.nbytes
            + y_all.nbytes
        )
        split = _split_sparse(X_all, y_all, int(wl["rows"]))
    else:
        X_all = _uniform(_counters(seed, n_total * n_feat)).reshape(
            n_feat, n_total
        ).T.copy()
        if wl["kind"] == "missing_categorical":
            X_all = _add_missing_and_categories(wl, X_all, n_total, n_feat, seed)
        y_all = _labels(wl, X_all, n_total, n_feat, seed)
        digest = _digest(X_all, y_all)
        data_bytes = int(X_all.nbytes + y_all.nbytes)
        split = _split_dense(X_all, y_all, int(wl["rows"]))

    split["digest"] = digest
    split["data_bytes"] = data_bytes
    split["categorical"] = (
        list(range(n_feat - int(wl["categorical_features"] or 0), n_feat))
        if wl["kind"] == "missing_categorical" and wl["categorical_features"]
        else []
    )
    return split


def _labels(wl, X, n_total, n_feat, seed):
    import numpy as np

    signal = _signal(np.nan_to_num(X[:, :4], nan=0.5))
    noise = _uniform(_counters(seed + np.uint64(n_total * n_feat), n_total))
    if wl["task"] == "binary":
        return (noise < _sigmoid(2.0 * (signal - 3.0))).astype(np.float64)
    if wl["task"] == "multiclass":
        k = int(wl["n_classes"])
        # Class boundaries at equal quantiles of the signal plus noise, so
        # every class is populated and the problem is genuinely k-way.
        z = signal + 0.5 * (noise - 0.5)
        edges = np.quantile(z, np.linspace(0.0, 1.0, k + 1)[1:-1])
        return np.searchsorted(edges, z).astype(np.float64)
    return signal + 0.1 * (noise - 0.5)


def _add_missing_and_categories(wl, X, n_total, n_feat, seed):
    import numpy as np

    n_cat = int(wl["categorical_features"] or 0)
    card = int(wl["categorical_cardinality"] or 0)
    if n_cat:
        codes = _uniform(
            _counters(seed + np.uint64(2 * n_total * n_feat), n_total * n_cat)
        ).reshape(n_total, n_cat)
        X[:, n_feat - n_cat:] = np.floor(codes * card)
    frac = float(wl["missing_fraction"] or 0.0)
    if frac > 0.0:
        n_num = n_feat - n_cat
        mask = _uniform(
            _counters(seed + np.uint64(3 * n_total * n_feat), n_total * n_num)
        ).reshape(n_total, n_num) < frac
        block = X[:, :n_num]
        block[mask] = np.nan
        X[:, :n_num] = block
    return X


def _sparse_matrix(wl, n_total, n_feat, seed):
    """CSC matrix with a fixed number of nonzeros per row, plus a small
    dense probe of the first four columns for label generation."""
    import numpy as np
    from scipy import sparse

    nnz_row = int(wl["nonzeros_per_row"])
    total = n_total * nnz_row
    cols = (_splitmix64(_counters(seed, total)) % np.uint64(n_feat)).astype(np.int64)
    vals = _uniform(_counters(seed + np.uint64(total), total))
    rows = np.repeat(np.arange(n_total, dtype=np.int64), nnz_row)
    coo = sparse.coo_matrix((vals, (rows, cols)), shape=(n_total, n_feat))
    csc = coo.tocsc()
    csc.sum_duplicates()
    probe = np.asarray(csc[:, :4].todense(), dtype=np.float64)
    return {"X": csc, "probe": probe}


def _split_dense(X, y, n_train):
    return {
        "X": X[:n_train], "y": y[:n_train],
        "X_valid": X[n_train:], "y_valid": y[n_train:],
    }


def _split_sparse(X, y, n_train):
    return {
        "X": X[:n_train].tocsc(), "y": y[:n_train],
        "X_valid": X[n_train:].tocsr(), "y_valid": y[n_train:],
    }


# --------------------------------------------------------------------------
# Quality, scored by this file for every engine
# --------------------------------------------------------------------------

def score(wl, pred, y):
    """Held-out quality from raw predictions. One scorer for every engine,
    so a quality difference is a model difference and not a metric
    difference."""
    import numpy as np

    out = {"available": True, "reason": None}
    pred = np.asarray(pred, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if wl["task"] == "regression":
        out["valid_rmse"] = float(np.sqrt(np.mean((pred - y) ** 2)))
    elif wl["task"] == "binary":
        p = np.clip(pred.reshape(-1), 1e-15, 1.0 - 1e-15)
        out["valid_logloss"] = float(
            -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        )
        out["valid_accuracy"] = float(np.mean((p >= 0.5) == (y >= 0.5)))
    else:
        if pred.ndim != 2 or pred.shape[1] != int(wl["n_classes"]):
            raise ValueError(
                f"multiclass scoring needs one probability row per row, got "
                f"shape {pred.shape} for {wl['n_classes']} classes; the "
                f"engine returned something this scorer would misread"
            )
        p = np.clip(pred, 1e-15, 1.0)
        p = p / p.sum(axis=1, keepdims=True)
        idx = y.astype(np.int64)
        out["valid_multi_logloss"] = float(
            -np.mean(np.log(p[np.arange(len(idx)), idx]))
        )
        out["valid_accuracy"] = float(np.mean(p.argmax(axis=1) == idx))
    out["prediction_digest"] = _digest(pred)
    return out


# --------------------------------------------------------------------------
# Engines
# --------------------------------------------------------------------------

class Engine:
    """One library on one device.

    `supports` answers whether the engine can express a workload at all,
    with the reason recorded when it cannot. `fit` returns the timings of
    one repetition plus the held-out predictions the suite scores. Neither
    ever falls back to another device or another algorithm: an engine that
    cannot run a workload says so.
    """

    name = "engine"
    device = "cpu"
    module = None

    def version(self):
        mod = self._import()
        return getattr(mod, "__version__", None)

    def _import(self):
        raise NotImplementedError

    def available(self):
        try:
            self._import()
        except Exception as exc:
            return False, f"{self.name} unavailable: {exc}"
        return True, None

    def supports(self, wl):
        return True, None

    def env(self, threads):
        """Thread environment for the worker process. Set before the library
        is imported, which is why every measurement is its own process."""
        return {
            "OMP_NUM_THREADS": str(threads),
            "MKL_NUM_THREADS": str(threads),
            "VECLIB_MAXIMUM_THREADS": str(threads),
        }

    def fit(self, wl, data, threads):
        raise NotImplementedError


class MojoBoostEngine(Engine):
    name = "mojoboost_cpu"
    device = "cpu"

    def _import(self):
        _add_repo_python_path()
        import mojoboost

        return mojoboost

    def env(self, threads):
        e = super().env(threads)
        # The CPU backend takes its thread count from the environment; there
        # is no num_threads parameter. `1` forces the serial path.
        e["MOJOBOOST_NUM_WORKERS"] = str(threads)
        return e

    def supports(self, wl):
        if self.device == "gpu":
            if wl["kind"] == "sparse":
                return False, "no sparse GPU kernel; a sparse fit on device would have to densify"
            mb = self._import()
            if not mb.gpu_available():
                return False, "mojoboost.gpu_available() is False on this build or machine"
        return True, None

    def fit(self, wl, data, threads):
        mb = self._import()
        t = {}
        params = _engine_params(wl, self.name, threads)
        t0 = time.perf_counter()

        if wl["kind"] == "sparse":
            # Sparse input has no Dataset path, so binning is inside fit and
            # is recorded as such rather than guessed at.
            est = _mojoboost_estimator(mb, wl, params)
            est.fit(data["X"], data["y"])
            t["binning_s"] = None
            t["train_s"] = None
            t["fit_s"] = time.perf_counter() - t0
            model = est
            device_resolved = getattr(est, "device_", None)
        else:
            ds_params = {"max_bin": int(wl["max_bin"])}
            ds = mb.Dataset(
                data["X"], label=data["y"], params=ds_params,
                categorical_feature=data["categorical"] or None,
            )
            ds.construct()
            t["binning_s"] = time.perf_counter() - t0
            t1 = time.perf_counter()
            model = mb.train(params, ds, int(wl["rounds"]))
            t["train_s"] = time.perf_counter() - t1
            t["fit_s"] = time.perf_counter() - t0
            device_resolved = self.device

        t2 = time.perf_counter()
        pred = model.predict(data["X_valid"])
        t["predict_s"] = time.perf_counter() - t2
        t["total_s"] = time.perf_counter() - t0
        return {"timings": t, "pred": pred, "device_resolved": device_resolved,
                "model": model}


class MojoBoostGPUEngine(MojoBoostEngine):
    name = "mojoboost_gpu"
    device = "gpu"


class LightGBMEngine(Engine):
    name = "lightgbm_cpu"
    device = "cpu"

    def _import(self):
        import lightgbm

        return lightgbm

    def fit(self, wl, data, threads):
        lgb = self._import()
        params = _engine_params(wl, self.name, threads)
        t = {}
        t0 = time.perf_counter()
        ds = lgb.Dataset(
            data["X"], label=data["y"],
            params={"max_bin": int(wl["max_bin"]), "verbose": -1},
            categorical_feature=data["categorical"] or "auto",
            free_raw_data=False,
        )
        ds.construct()
        t["binning_s"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        model = lgb.train(params, ds, num_boost_round=int(wl["rounds"]))
        t["train_s"] = time.perf_counter() - t1
        t["fit_s"] = time.perf_counter() - t0
        t2 = time.perf_counter()
        pred = model.predict(data["X_valid"])
        t["predict_s"] = time.perf_counter() - t2
        t["total_s"] = time.perf_counter() - t0
        return {"timings": t, "pred": pred, "device_resolved": "cpu",
                "model": model}


class XGBoostEngine(Engine):
    name = "xgboost_cpu"
    device = "cpu"

    def _import(self):
        import xgboost

        return xgboost

    def supports(self, wl):
        if wl["kind"] == "missing_categorical":
            # XGBoost can split categoricals, but by a different rule
            # (`max_cat_to_onehot` partitioning against LightGBM's sorted
            # category statistic). The timing is comparable, the fitted
            # model is not, and `engine.comparability` says so on the record.
            return True, None
        return True, None

    def fit(self, wl, data, threads):
        xgb = self._import()
        params = _engine_params(wl, self.name, threads)
        t = {}
        t0 = time.perf_counter()
        feature_types = None
        if data["categorical"]:
            feature_types = [
                "c" if i in set(data["categorical"]) else "q"
                for i in range(int(wl["features"]))
            ]
        dtrain = xgb.QuantileDMatrix(
            data["X"], label=data["y"], max_bin=int(wl["max_bin"]),
            feature_types=feature_types, enable_categorical=bool(feature_types),
        )
        t["binning_s"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        model = xgb.train(params, dtrain, num_boost_round=int(wl["rounds"]))
        t["train_s"] = time.perf_counter() - t1
        t["fit_s"] = time.perf_counter() - t0
        t2 = time.perf_counter()
        dvalid = xgb.DMatrix(
            data["X_valid"], feature_types=feature_types,
            enable_categorical=bool(feature_types),
        )
        pred = model.predict(dvalid)
        t["predict_s"] = time.perf_counter() - t2
        t["total_s"] = time.perf_counter() - t0
        return {"timings": t, "pred": pred, "device_resolved": "cpu",
                "model": model}


class XGBoostGPUEngine(XGBoostEngine):
    name = "xgboost_gpu"
    device = "gpu"

    def supports(self, wl):
        return False, (
            "XGBoost's GPU tree method targets CUDA; there is no Metal "
            "backend, so no Apple silicon GPU comparison exists for it"
        )


ENGINES = {
    e.name: e for e in (
        MojoBoostEngine(), MojoBoostGPUEngine(), LightGBMEngine(),
        XGBoostEngine(), XGBoostGPUEngine(),
    )
}


def _mojoboost_estimator(mb, wl, params):
    kwargs = dict(params)
    kwargs.pop("objective", None)
    kwargs.pop("num_class", None)
    rounds = int(wl["rounds"])
    if wl["task"] == "multiclass":
        return mb.MojoBoostClassifier(n_estimators=rounds, **kwargs)
    if wl["task"] == "binary":
        return mb.MojoBoostClassifier(n_estimators=rounds, **kwargs)
    return mb.MojoBoostRegressor(n_estimators=rounds, **kwargs)


def _engine_params(wl, engine_name, threads):
    """Parameters for one engine, matched across engines wherever the
    libraries admit a common meaning.

    Where they do not, the difference is forced onto the engine rather than
    left to its default, and named in `comparability` on the record. The two
    that matter here are LightGBM's feature bundling (off, mojoboost has no
    EFB) and XGBoost's regularization defaults (`lambda` 1.0 matches
    mojoboost, `min_child_weight` is a different quantity from
    `min_data_in_leaf` and is set to the hessian floor instead).
    """
    common_leaves = int(wl["num_leaves"])
    if engine_name.startswith("mojoboost"):
        p = {
            "objective": {
                "regression": "regression", "binary": "binary",
                "multiclass": "multiclass",
            }[wl["task"]],
            "num_leaves": common_leaves,
            "learning_rate": float(wl["learning_rate"]),
            "min_data_in_leaf": int(wl["min_data_in_leaf"]),
            "lambda_l2": float(wl["lambda_l2"]),
            "device": "gpu" if engine_name.endswith("_gpu") else "cpu",
        }
        if wl["task"] == "multiclass":
            p["num_class"] = int(wl["n_classes"])
        return p
    if engine_name.startswith("lightgbm"):
        p = {
            "objective": {
                "regression": "regression", "binary": "binary",
                "multiclass": "multiclass",
            }[wl["task"]],
            "num_leaves": common_leaves,
            "learning_rate": float(wl["learning_rate"]),
            "min_data_in_leaf": int(wl["min_data_in_leaf"]),
            "lambda_l2": float(wl["lambda_l2"]),
            "min_sum_hessian_in_leaf": 1e-3,
            "num_threads": int(threads),
            "enable_bundle": False,
            "force_row_wise": True,
            "verbose": -1,
            "deterministic": True,
        }
        if wl["task"] == "multiclass":
            p["num_class"] = int(wl["n_classes"])
        return p
    p = {
        "objective": {
            "regression": "reg:squarederror", "binary": "binary:logistic",
            "multiclass": "multi:softprob",
        }[wl["task"]],
        "tree_method": "hist",
        "grow_policy": "lossguide",
        "max_leaves": common_leaves,
        "max_depth": 0,
        "eta": float(wl["learning_rate"]),
        "min_child_weight": 1e-3,
        "reg_lambda": float(wl["lambda_l2"]),
        "nthread": int(threads),
        "device": "cuda" if engine_name.endswith("_gpu") else "cpu",
    }
    if wl["task"] == "multiclass":
        p["num_class"] = int(wl["n_classes"])
    return p


#: Semantic differences that survive parameter matching. These are printed
#: with every table and stored on every record; a reader who does not see
#: them is reading a comparison that has been laundered.
COMPARABILITY = {
    "lightgbm_cpu": [
        "min_data_in_bin=3 has no mojoboost equivalent, so bin edges can differ",
        "enable_bundle is forced off because mojoboost has no exclusive feature bundling",
    ],
    "xgboost_cpu": [
        "leaf-wise growth is approximated with grow_policy=lossguide and max_depth=0",
        "min_child_weight is a hessian sum, not a row count, so min_data_in_leaf has no exact counterpart",
        "categorical splits use XGBoost's partitioning rule, which is not LightGBM's",
    ],
}


# --------------------------------------------------------------------------
# Machine, software, conditions
# --------------------------------------------------------------------------

def _run(cmd, timeout=30):
    """Command output, or None if the command is missing or fails. Never
    raises: a missing tool is a null field, not a crashed run."""
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _sysctl(key, cast=str):
    val = _run(["sysctl", "-n", key])
    if val is None:
        return None
    val = val.strip()
    try:
        return cast(val)
    except (TypeError, ValueError):
        return None


def _chip_family(brand):
    if not brand:
        return "unknown"
    m = re.search(r"\bM([1-9])\b", brand)
    if m and 1 <= int(m.group(1)) <= 5:
        return f"m{m.group(1)}"
    return "other"


def _chip_label(brand):
    if not brand:
        return None
    m = re.search(r"\bM[1-9](\s+(Pro|Max|Ultra))?\b", brand)
    return m.group(0) if m else None


def machine_info():
    brand = _sysctl("machdep.cpu.brand_string")
    gpu_cores, gpu_name = _gpu_info()
    return {
        "chip_family": _chip_family(brand),
        "chip_label": _chip_label(brand),
        "cpu_brand_string": brand,
        "model_identifier": _sysctl("hw.model"),
        "arch": platform.machine(),
        "physical_cores": _sysctl("hw.physicalcpu", int),
        "performance_cores": _sysctl("hw.perflevel0.physicalcpu", int),
        "efficiency_cores": _sysctl("hw.perflevel1.physicalcpu", int),
        "logical_cores": _sysctl("hw.logicalcpu", int),
        "gpu_cores": gpu_cores,
        "gpu_name": gpu_name,
        "memory_bytes": _sysctl("hw.memsize", int),
        "unified_memory": platform.machine() == "arm64" and sys.platform == "darwin",
    }


def _gpu_info():
    """GPU core count and name from system_profiler. The core-count key has
    moved between macOS releases, so a miss returns None rather than a
    guess from the chip name."""
    raw = _run(["system_profiler", "-json", "SPDisplaysDataType"], timeout=60)
    if not raw:
        return None, None
    try:
        data = json.loads(raw)["SPDisplaysDataType"][0]
    except (ValueError, KeyError, IndexError):
        return None, None
    cores = data.get("sppci_cores") or data.get("spdisplays_gpu_cores")
    try:
        cores = int(str(cores).split()[0])
    except (TypeError, ValueError):
        cores = None
    return cores, data.get("sppci_model") or data.get("_name")


def _module_version(name):
    try:
        mod = __import__(name)
    except Exception:
        return None
    return getattr(mod, "__version__", None) or "unknown"


def software_info():
    mojo = _run(["mojo", "--version"]) or _run(["pixi", "run", "mojo", "--version"])
    max_line = None
    listing = _run(["pixi", "list", "--environment", "default"], timeout=120)
    if listing:
        for line in listing.splitlines():
            if line.split()[:1] == ["max"]:
                max_line = " ".join(line.split()[:2])
                break
    gpu_build = None
    try:
        _add_repo_python_path()
        import mojoboost

        gpu_build = bool(mojoboost.gpu_available())
    except Exception:
        pass
    return {
        "os_name": platform.system(),
        "os_version": platform.mac_ver()[0] or platform.release(),
        "os_build": (_run(["sw_vers", "-buildVersion"]) or "").strip() or None,
        "kernel": platform.release(),
        "python": platform.python_version(),
        "mojo": (mojo or "").strip() or None,
        "max": max_line,
        "mojoboost": _module_version("mojoboost"),
        "mojoboost_gpu_build": gpu_build,
        "lightgbm": _module_version("lightgbm"),
        "xgboost": _module_version("xgboost"),
        "numpy": _module_version("numpy"),
        "scipy": _module_version("scipy"),
    }


def thermal_sample():
    """Throttle state from pmset, which needs no privileges. The powermetrics
    thermal sampler is richer and needs sudo; when the energy sampler is
    running it fills `thermal_pressure` and this stays the fallback."""
    raw = _run(["pmset", "-g", "therm"])
    if raw is None:
        return {
            "available": False,
            "reason": "pmset -g therm produced no output",
            "source": "none",
            "cpu_speed_limit_percent": None,
            "scheduler_limit_percent": None,
            "available_cpus": None,
            "thermal_pressure": None,
            "raw": None,
        }
    def _num(key):
        m = re.search(rf"{key}\s*=\s*(\d+)", raw)
        return float(m.group(1)) if m else None

    return {
        "available": True,
        "reason": None,
        "source": "pmset_therm",
        "cpu_speed_limit_percent": _num("CPU_Speed_Limit"),
        "scheduler_limit_percent": _num("CPU_Scheduler_Limit"),
        "available_cpus": int(_num("CPU_Available_CPUs") or 0) or None,
        "thermal_pressure": None,
        "raw": raw.strip()[:2000],
    }


def is_throttled(sample):
    if not sample or not sample.get("available"):
        return False
    limit = sample.get("cpu_speed_limit_percent")
    pressure = sample.get("thermal_pressure")
    if limit is not None and limit < 100:
        return True
    return bool(pressure) and pressure.lower() != "nominal"


def _ancestor_pids():
    """This process and every parent of it.

    A suite launched through `pixi run` has pixi as a parent and a Mojo
    toolchain shim somewhere above it, so a gate that counted its own
    launcher as a competing build would never pass and would teach whoever
    hit it to use --allow-busy, which is worse than no gate.
    """
    pids, pid = set(), os.getpid()
    for _ in range(16):
        pids.add(pid)
        out = (_run(["ps", "-o", "ppid=", "-p", str(pid)]) or "").strip()
        if not out.isdigit():
            break
        pid = int(out)
        if pid <= 1:
            break
    return pids


def idle_gate():
    """The protocol's preconditions, checked rather than assumed. Returns
    (passed, failures, conditions fragment)."""
    failures = []
    try:
        load1 = os.getloadavg()[0]
    except OSError:
        load1 = None
    if load1 is not None and load1 > MAX_LOAD_AVERAGE:
        failures.append(f"1 minute load average {load1:.2f} above {MAX_LOAD_AVERAGE}")

    mine = _ancestor_pids()
    busy = []
    for pattern in BUSY_PROCESS_PATTERNS:
        found = set()
        for args_ in (["-x", pattern], ["-f", f"/{pattern} "]):
            for pid in (_run(["pgrep"] + args_) or "").split():
                if pid.isdigit() and int(pid) not in mine:
                    found.add(int(pid))
        if found:
            busy.append(f"{pattern} ({len(found)})")
    if busy:
        failures.append("competing processes: " + ", ".join(busy))

    batt = _run(["pmset", "-g", "batt"]) or ""
    if "AC Power" in batt:
        power = "ac"
    elif "Battery Power" in batt:
        power = "battery"
        failures.append("running on battery; Apple silicon changes clocks and power limits")
    else:
        power = "unknown"

    lpm_raw = _run(["pmset", "-g"]) or ""
    m = re.search(r"lowpowermode\s+(\d)", lpm_raw)
    low_power = bool(int(m.group(1))) if m else None
    if low_power:
        failures.append("Low Power Mode is on")

    thermal = thermal_sample()
    if is_throttled(thermal):
        failures.append("machine is already thermally limited before the first run")

    return (not failures), failures, {
        "load_average_before": load1,
        "competing_processes": busy,
        "power_source": power,
        "low_power_mode": low_power,
        "thermal_before": thermal,
    }


# --------------------------------------------------------------------------
# Energy
# --------------------------------------------------------------------------

def energy_unavailable(reason):
    return {"available": False, "reason": reason, "method": "none",
            "requires_sudo": True, "sample_interval_ms": None,
            "sample_count": None, "window_seconds": None, "keys_seen": []}


class PowerSampler:
    """powermetrics wrapped around a measured region.

    powermetrics reports the whole machine, not a process, and needs root.
    Both facts are recorded on every energy block it produces. The parser
    keeps the key names it actually matched, because those names differ
    between macOS releases and chips, and a run whose keys list is empty is
    a parser failure to fix rather than a machine that used no power.
    """

    #: Power series the record has fields for, and which statistics exist
    #: for each. Anything powermetrics reports that is not in this table is
    #: still listed in `keys_seen` and is not silently invented a field for,
    #: because a record that does not validate against the schema is a
    #: record nobody can read.
    SERIES = {
        "cpu_power_w": ("mean", "max", "energy"),
        "gpu_power_w": ("mean", "max", "energy"),
        "ane_power_w": ("mean",),
        "package_power_w": ("mean", "energy"),
        "combined_power_w": ("mean", "energy"),
    }

    def __init__(self, interval_ms=200, enabled=False):
        self.interval_ms = int(interval_ms)
        self.enabled = enabled
        self.proc = None
        self.started = None
        self.reason = None
        if enabled and shutil.which("powermetrics") is None:
            self.reason = "powermetrics is not on PATH"
            self.enabled = False
        elif enabled and os.geteuid() != 0:
            self.reason = (
                "powermetrics needs root; rerun the suite under sudo -E or "
                "accept a record with no energy fields"
            )
            self.enabled = False

    def __enter__(self):
        if not self.enabled:
            return self
        self.proc = subprocess.Popen(
            ["powermetrics", "--samplers", "cpu_power,gpu_power,thermal",
             "-i", str(self.interval_ms), "-f", "plist"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        self.started = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.proc is not None:
            self.proc.terminate()
        return False

    def result(self):
        if not self.enabled:
            return energy_unavailable(self.reason or "energy sampling not requested")
        window = time.perf_counter() - (self.started or time.perf_counter())
        try:
            raw = self.proc.communicate(timeout=30)[0] or b""
        except subprocess.SubprocessError:
            self.proc.kill()
            return energy_unavailable("powermetrics did not exit cleanly")
        return self.parse(raw, window, self.interval_ms)

    @classmethod
    def parse(cls, raw, window_seconds, interval_ms):
        """Mean and max power per key over the sampled window, and energy as
        mean power times the window. Both plist and plain text output are
        accepted, since the flag names have moved across releases."""
        samples = cls._plists(raw) or cls._text(raw)
        if not samples:
            return energy_unavailable(
                "powermetrics produced no parseable samples; the key names "
                "on this macOS version may differ from the parser's"
            )
        keys, series = set(), {}
        for sample in samples:
            for key, value in sample.items():
                keys.add(key)
                series.setdefault(key, []).append(value)

        out = {
            "available": True, "reason": None, "method": "powermetrics",
            "requires_sudo": True, "sample_interval_ms": interval_ms,
            "sample_count": len(samples),
            "window_seconds": round(window_seconds, 6),
            "keys_seen": sorted(keys),
        }
        for key, values in series.items():
            stats = cls.SERIES.get(key)
            if not stats:
                continue
            mean = sum(values) / len(values)
            if "mean" in stats:
                out[f"{key}_mean"] = mean
            if "max" in stats:
                out[f"{key}_max"] = max(values)
            if "energy" in stats:
                out[f"{key[:-len('_power_w')]}_energy_j"] = mean * window_seconds
        return out

    @classmethod
    def _plists(cls, raw):
        out = []
        for chunk in raw.split(b"\x00"):
            chunk = chunk.strip()
            if not chunk.startswith(b"<?xml"):
                continue
            try:
                doc = plistlib.loads(chunk)
            except Exception:
                continue
            sample = {}
            proc = doc.get("processor", {})
            for src, dst in (
                ("cpu_power", "cpu_power_w"), ("gpu_power", "gpu_power_w"),
                ("ane_power", "ane_power_w"),
                ("combined_power", "combined_power_w"),
                ("package_watts", "package_power_w"),
            ):
                if src in proc:
                    value = float(proc[src])
                    # powermetrics reports milliwatts for these keys and
                    # watts for package_watts.
                    sample[dst] = value if src == "package_watts" else value / 1000.0
            if sample:
                out.append(sample)
        return out

    @classmethod
    def _text(cls, raw):
        text = raw.decode("utf-8", "replace")
        pattern = re.compile(
            r"^(CPU|GPU|ANE|Combined|Package)\s+Power[^:]*:\s*([0-9.]+)\s*(m?W)",
            re.IGNORECASE | re.MULTILINE,
        )
        samples, current = [], {}
        for label, value, unit in pattern.findall(text):
            key = f"{label.lower()}_power_w"
            watts = float(value) / (1000.0 if unit.lower() == "mw" else 1.0)
            if key in current:
                samples.append(current)
                current = {}
            current[key] = watts
        if current:
            samples.append(current)
        return samples


# --------------------------------------------------------------------------
# Worker: one measurement in one process
# --------------------------------------------------------------------------

def run_worker(spec):
    """Run one (workload, engine) pair. Called in a fresh child process so
    peak resident memory is attributable and the thread environment is the
    one the plan asked for."""
    import resource

    wl = spec["workload"]
    engine = ENGINES[spec["engine"]]
    threads = int(spec["threads"])
    reps = int(spec["repetitions"])

    t_import = time.perf_counter()
    ok, reason = engine.available()
    if not ok:
        return {"status": "unsupported", "message": reason, "repetitions": []}
    import_s = time.perf_counter() - t_import

    ok, reason = engine.supports(wl)
    if not ok:
        return {"status": "unsupported", "message": reason, "repetitions": []}

    t_gen = time.perf_counter()
    data = generate(wl)
    data_gen_s = time.perf_counter() - t_gen

    repetitions, quality, last_pred, device_resolved = [], None, None, None
    for i in range(reps):
        try:
            if wl["id"] == "w7_repeated_fit":
                rep = _repeated_fit(engine, wl, data, threads)
            elif wl["id"] == "w8_prediction":
                rep = _prediction(engine, wl, data, threads)
            else:
                rep = engine.fit(wl, data, threads)
        except Exception as exc:  # a failed engine is a recorded fact
            return {
                "status": "error",
                "message": f"{type(exc).__name__}: {exc}",
                "repetitions": repetitions,
            }
        timings = rep["timings"]
        timings["import_s"] = import_s if i == 0 else None
        timings["data_gen_s"] = data_gen_s if i == 0 else None
        timings.setdefault("warmup_compile_s", None)
        repetitions.append({
            "index": i,
            "warmup": i == 0,
            "timings": timings,
            "device_phases": {
                "source": "unavailable",
                "reason": (
                    "no phase counters are exported to Python; see "
                    "handoffs/apple_a8_benchmarks.md for the instrumentation "
                    "hook this field is waiting on"
                ),
            },
        })
        last_pred = rep.get("pred")
        device_resolved = rep.get("device_resolved") or device_resolved

    if last_pred is not None:
        quality = score(wl, last_pred, data["y_valid"])

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024
    return {
        "status": "ok",
        "message": None,
        "repetitions": repetitions,
        "quality": quality or {"available": False, "reason": "no predictions produced"},
        "memory": {
            "available": True, "reason": None,
            "peak_rss_bytes": int(peak_bytes),
            "data_bytes": int(data["data_bytes"]),
            "footprint_bytes": None, "swap_ins": None,
            "compressed_pages": None,
        },
        "data_digest": data["digest"],
        "device_resolved": device_resolved,
    }


def _repeated_fit(engine, wl, data, threads):
    """Fit the same data several times in one process. The first fit carries
    library initialization, kernel compilation, and device setup; the rest
    are steady state, and the gap between them is the number a persistent
    device session is meant to close."""
    fits = int(wl["fits"] or 1)
    per_fit = []
    for _ in range(fits):
        rep = engine.fit(wl, data, threads)
        per_fit.append(rep["timings"]["fit_s"])
    t = dict(rep["timings"])
    t["first_fit_s"] = per_fit[0]
    t["steady_fit_s"] = (
        statistics.median(per_fit[1:]) if len(per_fit) > 1 else None
    )
    t["total_s"] = sum(per_fit)
    return {"timings": t, "pred": rep.get("pred"),
            "device_resolved": rep.get("device_resolved")}


def _prediction(engine, wl, data, threads):
    """Fit once, then score in batches. Training time is still recorded, but
    the reported metric for this workload is `predict_s`."""
    import numpy as np

    rep = engine.fit(wl, data, threads)
    model = rep["model"]
    batch = int(wl["predict_batch_rows"] or 0) or len(data["y_valid"])
    n_batches = int(wl["predict_batches"] or 1)
    X = data["X_valid"]
    t0 = time.perf_counter()
    rows = 0
    for b in range(n_batches):
        lo = (b * batch) % max(1, X.shape[0])
        hi = min(lo + batch, X.shape[0])
        if hi <= lo:
            continue
        model.predict(X[lo:hi])
        rows += hi - lo
    elapsed = time.perf_counter() - t0
    t = dict(rep["timings"])
    t["predict_s"] = elapsed
    t["predict_rows_per_s"] = rows / elapsed if elapsed > 0 else None
    t["total_s"] = t.get("fit_s", 0.0) + elapsed
    return {"timings": t, "pred": rep.get("pred"),
            "device_resolved": rep.get("device_resolved")}


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def build_plan(args):
    """The (workload, engine, threads) triples a run would execute, in the
    order it would execute them."""
    catalog = scaled_workloads(args.scale)
    wanted_workloads = args.workloads or list(catalog)
    wanted_engines = args.engines or [
        "mojoboost_cpu", "mojoboost_gpu", "lightgbm_cpu", "xgboost_cpu",
    ]
    plan = []
    for wid in wanted_workloads:
        if wid not in catalog:
            raise SystemExit(f"unknown workload {wid!r}; see --list")
        wl = dict(catalog[wid])
        wl["id"] = wid
        for ename in wanted_engines:
            if ename not in ENGINES:
                raise SystemExit(f"unknown engine {ename!r}; see --list")
            for threads in args.threads:
                plan.append({
                    "workload": wl, "engine": ename, "threads": int(threads),
                    "repetitions": args.repetitions,
                })
    return plan


def _measurement_record(item, worker, engine_version):
    wl = item["workload"]
    engine = ENGINES[item["engine"]]
    record_wl = {k: v for k, v in wl.items() if not k.startswith("_")}
    record_wl["data_digest"] = worker.get("data_digest")
    summary = _summarize(wl, worker.get("repetitions") or [])
    return {
        "workload": record_wl,
        "engine": {
            "name": engine.name,
            "version": engine_version,
            "device": engine.device,
            "requested_threads": item["threads"],
            "resolved_threads": None,
            "device_resolved": worker.get("device_resolved"),
            "env": engine.env(item["threads"]),
            "params": _engine_params(wl, engine.name, item["threads"]),
            "comparability": COMPARABILITY.get(engine.name, []),
        },
        "status": worker["status"],
        "message": worker.get("message"),
        "repetitions": worker.get("repetitions") or [],
        "summary": summary,
        "quality": worker.get("quality") or {
            "available": False, "reason": "measurement did not run",
        },
        "memory": worker.get("memory") or {
            "available": False, "reason": "measurement did not run",
        },
        "energy": worker.get("energy") or energy_unavailable(
            "energy sampling not requested"
        ),
        "thermal_after": worker.get("thermal_after") or {
            "available": False, "reason": "not sampled", "source": "none",
        },
    }


def _headline_metric(wl):
    if wl["id"] == "w8_prediction":
        return "predict_s"
    if wl["id"] == "w7_repeated_fit":
        return "steady_fit_s"
    return "fit_s"


def _summarize(wl, repetitions):
    metric = _headline_metric(wl)
    values = [
        r["timings"].get(metric) for r in repetitions
        if not r["warmup"] and r["timings"].get(metric) is not None
    ]
    if not values:
        return {"n": 0, "metric": metric, "median_s": None, "min_s": None,
                "max_s": None, "spread_ratio": None}
    lo, hi = min(values), max(values)
    return {
        "n": len(values), "metric": metric,
        "median_s": statistics.median(values), "min_s": lo, "max_s": hi,
        "spread_ratio": (hi / lo) if lo > 0 else None,
    }


def execute(args):
    """Run the plan. Every measurement is a child process; the parent owns
    the power sampler, the thermal samples, and the cooldown."""
    started = time.gmtime()
    passed, failures, cond = idle_gate()
    if not passed and not args.allow_busy:
        sys.stderr.write(
            "idle gate failed:\n  " + "\n  ".join(failures)
            + "\nFix the machine or pass --allow-busy to record a run that "
              "the protocol does not allow to be quoted.\n"
        )
        return 2

    plan = build_plan(args)
    # The baseline is taken before the workloads, on the idle machine the
    # gate has just checked, which is the only moment it means anything.
    baseline = _idle_baseline(args)
    measurements = []
    for i, item in enumerate(plan):
        if i:
            time.sleep(args.cooldown)
        sys.stderr.write(
            f"[{i + 1}/{len(plan)}] {item['workload']['id']} "
            f"{item['engine']} threads={item['threads']}\n"
        )
        with PowerSampler(args.energy_interval_ms, args.energy) as sampler:
            worker = _spawn(item, ENGINES[item["engine"]].env(item["threads"]))
        worker["energy"] = _net_of_idle(sampler.result(), baseline)
        worker["thermal_after"] = thermal_sample()
        version = None
        try:
            version = ENGINES[item["engine"]].version()
        except Exception:
            pass
        measurements.append(_measurement_record(item, worker, version))

    thermal_after = thermal_sample()
    conditions = dict(cond)
    conditions.update({
        "idle_gate_passed": passed,
        "idle_gate_failures": failures,
        "load_average_after": (os.getloadavg()[0] if hasattr(os, "getloadavg") else None),
        "thermal_after": thermal_after,
        "throttle_detected": (
            is_throttled(cond["thermal_before"]) or is_throttled(thermal_after)
            or any(is_throttled(m.get("thermal_after")) for m in measurements)
        ),
        "display_sleep_prevented": None,
        "idle_power_baseline": baseline,
    })

    record = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "suite_version": SUITE_VERSION,
        "run_id": args.run_id or time.strftime("%Y%m%dT%H%M%SZ", started),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", started),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "invocation": {
            "argv": sys.argv, "cwd": os.getcwd(),
            "repetitions": args.repetitions,
            "cooldown_seconds": args.cooldown,
            "energy_requested": bool(args.energy),
            "git_commit": (_run(["git", "-C", REPO_ROOT, "rev-parse", "HEAD"]) or "").strip() or None,
            "git_dirty": bool((_run(["git", "-C", REPO_ROOT, "status", "--porcelain"]) or "").strip()),
        },
        "machine": machine_info(),
        "software": software_info(),
        "conditions": conditions,
        "measurements": measurements,
        "notes": list(args.note or []),
    }
    if args.scale != "full":
        record["notes"].append(
            f"scale={args.scale}: reduced shapes, harness check only, not a result"
        )

    errors = validate(record, load_schema())
    for err in errors:
        sys.stderr.write(f"schema violation: {err}\n")

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"{record['run_id']}.json")
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2, sort_keys=False)
        fh.write("\n")
    sys.stderr.write(f"wrote {path}\n")
    _report(record)
    return 1 if errors else 0


def _machine_power(energy):
    """The whole-machine power a block implies, preferring the combined key
    and falling back to the sum of the parts it did report. None when it
    reported neither, which is the difference between an idle baseline and
    an assumption."""
    if not energy.get("available"):
        return None
    combined = energy.get("combined_power_w_mean")
    if combined is not None:
        return combined
    parts = [
        energy.get(k) for k in
        ("cpu_power_w_mean", "gpu_power_w_mean", "ane_power_w_mean")
    ]
    present = [p for p in parts if p is not None]
    return sum(present) if present else None


def _net_of_idle(energy, baseline):
    """Fill `energy_above_idle_j`, the workload's energy net of the machine
    simply being switched on. Absent a baseline the field stays null rather
    than being quietly equated to the gross figure."""
    idle_w = _machine_power(baseline)
    window = energy.get("window_seconds")
    gross = _machine_power(energy)
    if idle_w is None or window is None or gross is None:
        energy["energy_above_idle_j"] = None
    else:
        energy["energy_above_idle_j"] = (gross - idle_w) * window
    return energy


def _idle_baseline(args):
    """Machine power with nothing running, taken under the same sampler, so
    an energy number can be read net of the machine simply being on."""
    if not args.energy:
        return energy_unavailable("energy sampling not requested")
    with PowerSampler(args.energy_interval_ms, True) as sampler:
        if not sampler.enabled:
            return sampler.result()
        time.sleep(args.idle_baseline_seconds)
        return sampler.result()


def _spawn(item, env_overrides):
    env = dict(os.environ)
    env.update(env_overrides)
    env["MOJOBOOST_APPLE_SUITE_WORKER"] = "1"
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--worker"],
        input=json.dumps(item), capture_output=True, text=True, env=env,
        check=False,
    )
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_SENTINEL):
            return json.loads(line[len(RESULT_SENTINEL):])
    return {
        "status": "error",
        "message": (
            f"worker exited {proc.returncode} without a result line; "
            f"stderr tail: {proc.stderr.strip()[-2000:]}"
        ),
        "repetitions": [],
    }


RESULT_SENTINEL = "MOJOBOOST_APPLE_SUITE_RESULT "


def _report(record):
    """A short text table for the operator. The JSON is the record; this is
    a reading aid, and it prints the caveats with the numbers rather than
    under them."""
    lines = ["", "workload                engine          thr   metric        median_s  quality"]
    for m in record["measurements"]:
        s, q = m["summary"], m["quality"]
        qv = next(
            (f"{k}={v:.6g}" for k, v in q.items()
             if k.startswith("valid_") and isinstance(v, float)),
            "-",
        )
        median = f"{s['median_s']:.4f}" if s.get("median_s") is not None else "-"
        lines.append(
            f"{m['workload']['id']:<23} {m['engine']['name']:<15} "
            f"{m['engine']['requested_threads']:<5} {s['metric']:<13} "
            f"{median:>8}  {qv}"
        )
        if m["status"] != "ok":
            lines.append(f"    {m['status']}: {m['message']}")
    cond = record["conditions"]
    if not cond["idle_gate_passed"]:
        lines.append("\nIDLE GATE FAILED. These timings are not quotable.")
    if cond["throttle_detected"]:
        lines.append("\nTHERMAL LIMIT DETECTED. These timings are not quotable.")
    if not cond["idle_power_baseline"]["available"]:
        lines.append(
            "\nNo energy was measured: "
            + str(cond["idle_power_baseline"]["reason"])
        )
    sys.stderr.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# A small JSON Schema validator
# --------------------------------------------------------------------------
#
# The suite validates its own output before writing it, and must do so with
# no third-party dependency, because the machine running a benchmark should
# be carrying the benchmark's dependencies and nothing else. `jsonschema` is
# used when it happens to be installed; the subset below covers every
# construct schema.json uses.

def load_schema():
    with open(SCHEMA_PATH) as fh:
        return json.load(fh)


def validate(instance, schema, path="$"):
    try:
        import jsonschema
    except ImportError:
        return _validate(instance, schema, schema, path)
    validator = jsonschema.Draft202012Validator(schema)
    return [f"{'.'.join(str(p) for p in e.path) or '$'}: {e.message}"
            for e in validator.iter_errors(instance)]


def _validate(node, schema, root, path):
    errors = []
    if "$ref" in schema:
        target = root
        for part in schema["$ref"].lstrip("#/").split("/"):
            target = target[part]
        return _validate(node, target, root, path)

    types = schema.get("type")
    if types is not None:
        if isinstance(types, str):
            types = [types]
        if not any(_is_type(node, t) for t in types):
            return [f"{path}: expected {'/'.join(types)}, got {type(node).__name__}"]

    if "enum" in schema and node not in schema["enum"]:
        errors.append(f"{path}: {node!r} not in {schema['enum']}")
    if "pattern" in schema and isinstance(node, str):
        if not re.search(schema["pattern"], node):
            errors.append(f"{path}: {node!r} does not match {schema['pattern']}")
    if "minimum" in schema and isinstance(node, (int, float)) and node < schema["minimum"]:
        errors.append(f"{path}: {node} below minimum {schema['minimum']}")
    if "minLength" in schema and isinstance(node, str) and len(node) < schema["minLength"]:
        errors.append(f"{path}: shorter than {schema['minLength']}")

    if isinstance(node, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in node:
                errors.append(f"{path}: missing required {key!r}")
        for key, value in node.items():
            if key in props:
                errors.extend(_validate(value, props[key], root, f"{path}.{key}"))
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: unexpected property {key!r}")
            elif isinstance(schema.get("additionalProperties"), dict):
                errors.extend(_validate(
                    value, schema["additionalProperties"], root, f"{path}.{key}"
                ))
    elif isinstance(node, list) and "items" in schema:
        for i, item in enumerate(node):
            errors.extend(_validate(item, schema["items"], root, f"{path}[{i}]"))
    return errors


def _is_type(node, name):
    if name == "null":
        return node is None
    if name == "boolean":
        return isinstance(node, bool)
    if name == "integer":
        return isinstance(node, int) and not isinstance(node, bool)
    if name == "number":
        return isinstance(node, (int, float)) and not isinstance(node, bool)
    if name == "string":
        return isinstance(node, str)
    if name == "array":
        return isinstance(node, list)
    if name == "object":
        return isinstance(node, dict)
    return True


# --------------------------------------------------------------------------
# Static self-check
# --------------------------------------------------------------------------

def self_check():
    """Structural validation with nothing measured and nothing installed.

    It checks that the schema parses and its `$ref`s resolve, that every
    catalog entry validates as a workload, that every engine name and
    workload id in this file is in the schema's enums, and that a template
    record, all nulls and no measurements, validates. The template is marked
    `is_template` so it can never be mistaken for a run.
    """
    problems = []
    schema = load_schema()

    for ref in re.findall(r'"\$ref":\s*"#/\$defs/([a-z_]+)"', json.dumps(schema)):
        if ref not in schema["$defs"]:
            problems.append(f"dangling $ref to {ref}")

    wl_enum = set(schema["$defs"]["workload"]["properties"]["id"]["enum"])
    if wl_enum != set(WORKLOADS):
        problems.append(
            f"workload ids disagree with the schema: "
            f"{sorted(wl_enum ^ set(WORKLOADS))}"
        )
    engine_enum = set(schema["$defs"]["engine"]["properties"]["name"]["enum"])
    if engine_enum != set(ENGINES):
        problems.append(
            f"engine names disagree with the schema: "
            f"{sorted(engine_enum ^ set(ENGINES))}"
        )

    for wid, wl in WORKLOADS.items():
        entry = dict(wl)
        entry["id"] = wid
        entry["data_digest"] = None
        problems.extend(
            _validate(entry, schema["$defs"]["workload"], schema, f"workload[{wid}]")
        )
        for engine in ENGINES:
            params = _engine_params(entry, engine, 1)
            if not params.get("objective"):
                problems.append(f"{engine} has no objective for {wid}")

    # The energy block is the part of a record most likely to grow a field
    # the schema has no room for, because powermetrics key names move
    # between macOS releases. Parse a synthetic two-sample text output and
    # validate what comes out. The input is obviously not a machine and
    # nothing derived from it is ever written to a record.
    synthetic = b"CPU Power: 1234 mW\nGPU Power: 567 mW\nANE Power: 0 mW\n" * 2
    block = PowerSampler.parse(synthetic, 2.0, 200)
    if not block.get("available"):
        problems.append("the energy parser rejected its own synthetic sample")
    problems.extend(_validate(block, schema["$defs"]["energy"], schema, "energy"))
    netted = _net_of_idle(dict(block), block)
    if netted.get("energy_above_idle_j") is None:
        problems.append("energy_above_idle_j stayed null with a baseline present")
    problems.extend(_validate(netted, schema["$defs"]["energy"], schema, "energy_net"))

    template = _template_record()
    problems.extend(validate(template, schema))

    for line in problems:
        sys.stderr.write(f"self-check: {line}\n")
    if problems:
        sys.stderr.write(f"self-check FAILED with {len(problems)} problems\n")
        return 1
    sys.stderr.write(
        f"self-check ok: {len(WORKLOADS)} workloads, {len(ENGINES)} engines, "
        f"schema {SCHEMA_VERSION}, protocol {PROTOCOL_VERSION}. "
        "No measurement was taken.\n"
    )
    return 0


def _template_record():
    """An empty record, structurally complete and numerically empty."""
    wl = dict(WORKLOADS["w1_small_dense"])
    wl["id"] = "w1_small_dense"
    wl["data_digest"] = None
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "suite_version": SUITE_VERSION,
        "run_id": "template",
        "started_utc": "1970-01-01T00:00:00Z",
        "finished_utc": "1970-01-01T00:00:00Z",
        "is_template": True,
        "invocation": {
            "argv": ["--self-check"], "cwd": ".", "repetitions": 1,
            "cooldown_seconds": 0, "energy_requested": False,
            "git_commit": None, "git_dirty": None,
        },
        "machine": {
            "chip_family": "unknown", "chip_label": None,
            "cpu_brand_string": None, "model_identifier": None, "arch": None,
            "physical_cores": None, "performance_cores": None,
            "efficiency_cores": None, "logical_cores": None,
            "gpu_cores": None, "gpu_name": None, "memory_bytes": None,
            "unified_memory": None,
        },
        "software": {
            "os_name": None, "os_version": None, "os_build": None,
            "kernel": None, "python": None, "mojo": None, "max": None,
            "mojoboost": None, "mojoboost_gpu_build": None,
            "lightgbm": None, "xgboost": None, "numpy": None, "scipy": None,
        },
        "conditions": {
            "idle_gate_passed": False, "idle_gate_failures": ["template"],
            "load_average_before": None, "load_average_after": None,
            "competing_processes": [], "power_source": "unknown",
            "low_power_mode": None, "display_sleep_prevented": None,
            "thermal_before": {"available": False, "reason": "template", "source": "none"},
            "thermal_after": {"available": False, "reason": "template", "source": "none"},
            "throttle_detected": False,
            "idle_power_baseline": energy_unavailable("template"),
        },
        "measurements": [{
            "workload": wl,
            "engine": {
                "name": "mojoboost_cpu", "version": None, "device": "cpu",
                "requested_threads": 1, "resolved_threads": None,
                "device_resolved": None, "env": {}, "params": {},
                "comparability": [],
            },
            "status": "skipped",
            "message": "template record; no measurement was taken",
            "repetitions": [],
            "summary": {"n": 0, "metric": "fit_s", "median_s": None,
                        "min_s": None, "max_s": None, "spread_ratio": None},
            "quality": {"available": False, "reason": "template"},
            "memory": {"available": False, "reason": "template"},
            "energy": energy_unavailable("template"),
            "thermal_after": {"available": False, "reason": "template", "source": "none"},
        }],
        "notes": ["structural template emitted by --self-check; not a result"],
    }


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def default_threads():
    """The two matched-thread points the protocol requires: one, and the
    performance-core count. Efficiency cores are excluded by default because
    including them changes the ratio between engines rather than scaling it
    (see the protocol's thread-matching section)."""
    p_cores = _sysctl("hw.perflevel0.physicalcpu", int) or os.cpu_count() or 1
    return [1, p_cores]


def parse_args(argv):
    ap = argparse.ArgumentParser(
        description="Apple silicon benchmark suite for mojoboost",
        epilog="Read docs/APPLE_GPU_BENCHMARK_PROTOCOL.md before running.",
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="print the catalog")
    mode.add_argument("--plan", action="store_true", help="print the run plan as JSON")
    mode.add_argument("--self-check", action="store_true",
                      help="static validation; measures nothing")
    mode.add_argument("--run", action="store_true", help="take measurements")
    mode.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)

    ap.add_argument("--scale", choices=["full", "smoke"], default="full")
    ap.add_argument("--workloads", nargs="+", metavar="ID")
    ap.add_argument("--engines", nargs="+", metavar="NAME")
    ap.add_argument("--threads", nargs="+", type=int, default=None)
    ap.add_argument("--repetitions", type=int, default=5,
                    help="repetitions per measurement, the first discarded as warmup")
    ap.add_argument("--cooldown", type=float, default=60.0,
                    help="seconds of idle between measurements")
    ap.add_argument("--energy", action="store_true",
                    help="sample power with powermetrics; needs root")
    ap.add_argument("--energy-interval-ms", type=int, default=200)
    ap.add_argument("--idle-baseline-seconds", type=float, default=30.0)
    ap.add_argument("--allow-busy", action="store_true",
                    help="record a run that fails the idle gate, marked unquotable")
    ap.add_argument("--out-dir", default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--note", action="append", metavar="TEXT")
    args = ap.parse_args(argv)
    if args.threads is None:
        args.threads = default_threads()
    # On a machine with one performance core the two required points
    # coincide; running it twice would not make it two measurements.
    args.threads = sorted(dict.fromkeys(int(t) for t in args.threads))
    if any(t < 1 for t in args.threads):
        ap.error("--threads values must be at least 1")
    if args.repetitions < 2:
        ap.error("--repetitions must be at least 2; the first is the warmup")
    return args


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    if args.worker:
        spec = json.loads(sys.stdin.read())
        result = run_worker(spec)
        sys.stdout.write(RESULT_SENTINEL + json.dumps(result) + "\n")
        return 0
    if args.self_check:
        return self_check()
    if args.list:
        for wid, wl in scaled_workloads(args.scale).items():
            print(f"{wid}: {json.dumps({k: v for k, v in wl.items() if v is not None})}")
        print("\nengines: " + ", ".join(ENGINES))
        print("default threads: " + ", ".join(str(t) for t in args.threads))
        return 0
    if args.plan:
        print(json.dumps(build_plan(args), indent=2))
        return 0
    if args.run:
        return execute(args)
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
