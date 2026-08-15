"""Training benchmark for mojotrees.

Generates a deterministic synthetic dataset with counter-based splitmix64
(bit-identical to bench_lightgbm.py), then times quantile binning and
boosted training with the library defaults (LightGBM-matched).

Usage: mojo run -I src bench/bench_train.mojo \
    [n_rows] [n_features] [reg|binary] [seed]
"""

from std.math import exp, log
from std.sys import argv
from std.time import perf_counter_ns

from mojotrees.binning import fit_bins
from mojotrees.boosting import (
    BINARY_LOGISTIC,
    SQUARED_ERROR,
    BoosterParams,
    train,
)


def _splitmix64(state: UInt64) -> UInt64:
    var z = state + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _uniform(counter: UInt64) -> Float64:
    return Float64(_splitmix64(counter) >> 11) * (1.0 / 9007199254740992.0)


def _sigmoid(x: Float64) -> Float64:
    if x >= 0.0:
        return 1.0 / (1.0 + exp(-x))
    var e = exp(x)
    return e / (1.0 + e)


def main() raises:
    var n_rows = 100_000
    var n_features = 100
    var objective = SQUARED_ERROR
    var obj_name = String("reg")
    var seed = 0
    var args = argv()
    if len(args) > 1:
        n_rows = Int(String(args[1]))
    if len(args) > 2:
        n_features = Int(String(args[2]))
    if len(args) > 3:
        obj_name = String(args[3])
        if obj_name == "binary":
            objective = BINARY_LOGISTIC
        elif obj_name != "reg":
            raise Error("objective must be 'reg' or 'binary'")
    if len(args) > 4:
        seed = Int(String(args[4]))
    if n_features < 4:
        raise Error("need at least 4 features")

    # Column-major features: value at (row r, feature f) is uniform(f * n_rows + r).
    var features = List[Float64](capacity=n_rows * n_features)
    var seed_offset = UInt64(seed) * 0x9E3779B97F4A7C15
    for k in range(n_rows * n_features):
        features.append(_uniform(seed_offset + UInt64(k)))

    # Target uses features 0..3 plus a noise stream at counters >= n_rows * n_features.
    var noise_base = seed_offset + UInt64(n_rows * n_features)
    var target = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        var x0 = features[0 * n_rows + r]
        var x1 = features[1 * n_rows + r]
        var x2 = features[2 * n_rows + r]
        var x3 = features[3 * n_rows + r]
        var signal = (
            5.0 * x0 + 4.0 * x1 * x2 + 3.0 * (x3 - 0.5) * (x3 - 0.5)
        )
        var u = _uniform(noise_base + UInt64(r))
        if objective == BINARY_LOGISTIC:
            var p = _sigmoid(2.0 * (signal - 3.0))
            target.append(1.0 if u < p else 0.0)
        else:
            target.append(signal + 0.1 * (u - 0.5))

    print(
        "mojotrees bench:", n_rows, "rows x", n_features, "features,",
        obj_name, "seed", seed,
    )

    var t0 = perf_counter_ns()
    var mapper = fit_bins(features, n_rows, n_features, 255)
    var data = mapper.transform(features, n_rows)
    var t1 = perf_counter_ns()
    var booster = train(data, target, objective, BoosterParams.default())
    var t2 = perf_counter_ns()

    var loss = 0.0
    for r in range(n_rows):
        if objective == BINARY_LOGISTIC:
            var p = booster.predict_row(data, r)
            if p < 1e-15:
                p = 1e-15
            if p > 1.0 - 1e-15:
                p = 1.0 - 1e-15
            var y = target[r]
            if y > 0.5:
                loss -= log(p)
            else:
                loss -= log(1.0 - p)
        else:
            var d = booster.predict_row(data, r) - target[r]
            loss += d * d
    loss /= Float64(n_rows)

    print("binning_s:", Float64(t1 - t0) / 1e9)
    print("train_s:", Float64(t2 - t1) / 1e9)
    print("total_s:", Float64(t2 - t0) / 1e9)
    print("n_trees:", len(booster.trees))
    if objective == BINARY_LOGISTIC:
        print("train_logloss:", loss)
    else:
        print("train_mse:", loss)
