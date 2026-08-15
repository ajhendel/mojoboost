"""GOSS vs full-data training benchmark.

Trains the same synthetic dataset twice with identical parameters, once on
every row and once with Gradient-based One-Side Sampling, and reports wall
time and training loss for both. The dataset is the counter-based splitmix64
stream `bench_train.mojo` and `bench_lightgbm.py` use, so numbers are
comparable across all three drivers.

GOSS trades accuracy for speed, so both halves of the trade are printed:
a speedup line and a loss line. A speedup with a materially worse loss is
not a win, and neither number means anything without the other.

Usage:
    mojo run -I src bench/bench_goss.mojo [n_rows] [n_features] [reg|binary]
                                          [top_rate] [other_rate]
"""

from std.math import exp, log
from std.sys import argv
from std.time import perf_counter_ns

from mojotrees.bagging import BaggingParams
from mojotrees.binning import fit_bins
from mojotrees.boosting import (
    BINARY_LOGISTIC,
    SQUARED_ERROR,
    Booster,
    BoosterParams,
    train,
)
from mojotrees.binning import BinnedMatrix
from mojotrees.goss import GossParams


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


def _train_loss(
    booster: Booster,
    data: BinnedMatrix,
    target: List[Float64],
    objective: Int,
) raises -> Float64:
    """Mean squared error, or log loss for the binary objective."""
    var loss = 0.0
    for r in range(data.n_rows):
        if objective == BINARY_LOGISTIC:
            var p = booster.predict_row(data, r)
            if p < 1e-15:
                p = 1e-15
            if p > 1.0 - 1e-15:
                p = 1.0 - 1e-15
            if target[r] > 0.5:
                loss -= log(p)
            else:
                loss -= log(1.0 - p)
        else:
            var d = booster.predict_row(data, r) - target[r]
            loss += d * d
    return loss / Float64(data.n_rows)


def main() raises:
    var n_rows = 100_000
    var n_features = 100
    var objective = SQUARED_ERROR
    var obj_name = String("reg")
    var top_rate = 0.2
    var other_rate = 0.1
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
        top_rate = Float64(String(args[4]))
    if len(args) > 5:
        other_rate = Float64(String(args[5]))
    if n_features < 4:
        raise Error("need at least 4 features")

    # Column-major features: (row r, feature f) is uniform(f * n_rows + r).
    var features = List[Float64](capacity=n_rows * n_features)
    for k in range(n_rows * n_features):
        features.append(_uniform(UInt64(k)))

    var noise_base = UInt64(n_rows * n_features)
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
        "mojotrees goss bench:",
        n_rows,
        "rows x",
        n_features,
        "features,",
        obj_name,
    )
    print("top_rate:", top_rate, "other_rate:", other_rate)

    var mapper = fit_bins(features, n_rows, n_features, 255)
    var data = mapper.transform(features, n_rows)
    var params = BoosterParams.default()

    var t0 = perf_counter_ns()
    var full = train(data, target, objective, params)
    var t1 = perf_counter_ns()
    var sampled = train(
        data,
        target,
        objective,
        params,
        [],
        0.9,
        BaggingParams.disabled(),
        GossParams.enable(top_rate, other_rate),
    )
    var t2 = perf_counter_ns()

    var full_s = Float64(t1 - t0) / 1e9
    var goss_s = Float64(t2 - t1) / 1e9
    print("full_train_s:", full_s)
    print("goss_train_s:", goss_s)
    print("speedup:", full_s / goss_s)
    print("full_n_trees:", len(full.trees))
    print("goss_n_trees:", len(sampled.trees))

    var full_loss = _train_loss(full, data, target, objective)
    var goss_loss = _train_loss(sampled, data, target, objective)
    if objective == BINARY_LOGISTIC:
        print("full_train_logloss:", full_loss)
        print("goss_train_logloss:", goss_loss)
    else:
        print("full_train_mse:", full_loss)
        print("goss_train_mse:", goss_loss)
    print("loss_ratio:", goss_loss / full_loss)
    # LightGBM skips sampling for the first int(1 / learning_rate) rounds, so
    # a short run spends a large share of its rounds on full data.
    print(
        "warmup_rounds:",
        GossParams.enable(top_rate, other_rate).warmup(params.learning_rate),
        "of",
        params.n_estimators,
    )
