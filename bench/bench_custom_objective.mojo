"""Custom-objective overhead benchmark (native Mojo interface).

Trains the same dataset three ways and times only the boosting loop:

  builtin   the built-in SQUARED_ERROR objective (`train`)
  custom    the same derivatives through the custom-objective interface
            (`train_custom` with `squared_error_grad_hess`)
  closure   the same, through a closure that captures a scale factor, to
            show that capturing state does not reintroduce dispatch

All three grow identical trees (the custom runs start from the label mean,
as the built-in one does), so the only difference measured is the objective
plumbing: a comptime-specialized call per round for the custom path plus its
per-round validation pass, against the built-in branch on the objective code.

Usage:
    mojo run -I src bench/bench_custom_objective.mojo \
        [n_rows] [n_features] [n_rounds]
"""

from std.sys import argv
from std.time import perf_counter_ns

from mojotrees.binning import fit_bins
from mojotrees.boosting import SQUARED_ERROR, BoosterParams, train
from mojotrees.objective import (
    mean_label,
    squared_error_grad_hess,
    train_custom,
)
from mojotrees.tree import TreeParams


def _splitmix64(state: UInt64) -> UInt64:
    var z = state + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _uniform(counter: UInt64) -> Float64:
    return Float64(_splitmix64(counter) >> 11) * (1.0 / 9007199254740992.0)


def main() raises:
    var n_rows = 100_000
    var n_features = 20
    var n_rounds = 100
    var args = argv()
    if len(args) > 1:
        n_rows = Int(String(args[1]))
    if len(args) > 2:
        n_features = Int(String(args[2]))
    if len(args) > 3:
        n_rounds = Int(String(args[3]))
    if n_features < 4:
        raise Error("need at least 4 features")

    var features = List[Float64](capacity=n_rows * n_features)
    for k in range(n_rows * n_features):
        features.append(_uniform(UInt64(k)))
    var target = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        var x0 = features[0 * n_rows + r]
        var x1 = features[1 * n_rows + r]
        var x2 = features[2 * n_rows + r]
        var x3 = features[3 * n_rows + r]
        target.append(5.0 * x0 + 4.0 * x1 * x2 + 3.0 * (x3 - 0.5) * (x3 - 0.5))

    var mapper = fit_bins(features, n_rows, n_features, 255)
    var data = mapper.transform(features, n_rows)
    var params = BoosterParams(
        n_rounds, 0.1, TreeParams(31, 20, 1.0, 1e-3)
    )
    var base = mean_label(target, [])
    var scale = 1.0

    def scaled_grad_hess(
        raw: List[Float64],
        labels: List[Float64],
        mut grad: List[Float64],
        mut hess: List[Float64],
    ) raises {imm scale}:
        grad.clear()
        hess.clear()
        for r in range(len(labels)):
            grad.append(scale * (raw[r] - labels[r]))
            hess.append(scale)

    print(
        "mojotrees custom-objective bench:",
        n_rows,
        "rows x",
        n_features,
        "features,",
        n_rounds,
        "rounds",
    )

    var t0 = perf_counter_ns()
    var builtin = train(data, target, SQUARED_ERROR, params)
    var t1 = perf_counter_ns()
    var custom = train_custom(
        data, target, squared_error_grad_hess, params, base_score=base
    )
    var t2 = perf_counter_ns()
    var closure = train_custom(
        data, target, scaled_grad_hess, params, base_score=base
    )
    var t3 = perf_counter_ns()

    # Guard the comparison: the three must be the same model, or the timings
    # are of different amounts of work.
    var worst = 0.0
    for r in range(n_rows):
        var b = builtin.predict_row(data, r)
        var d1 = abs(custom.predict_row(data, r) - b)
        var d2 = abs(closure.predict_row(data, r) - b)
        if d1 > worst:
            worst = d1
        if d2 > worst:
            worst = d2

    var builtin_s = Float64(t1 - t0) / 1e9
    var custom_s = Float64(t2 - t1) / 1e9
    var closure_s = Float64(t3 - t2) / 1e9
    print("builtin_s:", builtin_s)
    print("custom_s:", custom_s)
    print("closure_s:", closure_s)
    print("custom_overhead_pct:", 100.0 * (custom_s - builtin_s) / builtin_s)
    print("closure_overhead_pct:", 100.0 * (closure_s - builtin_s) / builtin_s)
    print("max_abs_prediction_diff:", worst)
    print("n_trees:", len(builtin.trees))
