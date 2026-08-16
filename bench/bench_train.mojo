"""Training benchmark for mojotrees.

Generates a deterministic synthetic dataset with counter-based splitmix64
(bit-identical to bench_lightgbm.py), then times ingestion, quantile binning
and boosted training with the library defaults (LightGBM-matched).

Three phases, and the first one is new
--------------------------------------
`ingest_s` is the row-major to column-major transpose, and it is here so that
this file's total and `bench_lightgbm.py`'s total span the same work.

LightGBM's `binning_s` in that file is `lgb.Dataset(X, ...).construct()` over
`make_data`'s array, which is C-ordered, so their figure has always contained
their ingestion. This one's did not: the matrix was generated column-major,
in the layout `fit_bins` wants, so a benchmark comparing the two totals was
charging LightGBM for a transpose and charging mojotrees for nothing. The
generator now writes the same values in the row-major order a caller's NumPy
array actually arrives in, and the transpose that follows is timed.

The values are unchanged: `raw[r * n_features + f]` holds the counter that
used to be written straight into `features[f * n_rows + r]`, and the
transpose puts it back in that slot. The binned matrix, the trees, and the
loss are the same bytes they were.

`total_s` still means binning plus training, so a figure recorded before this
change still means what it meant. `e2e_s` is the one to read against
LightGBM's `total_s`.

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
from mojotrees.trainset import to_column_major


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

    # The caller's matrix, in the layout a caller's matrix arrives in: C
    # order, `raw[r * n_features + f]`. Value at (row r, feature f) is still
    # uniform(f * n_rows + r), the same counter bench_lightgbm.py uses, so
    # the two harnesses train on identical numbers.
    var seed_offset = UInt64(seed) * 0x9E3779B97F4A7C15
    var raw = List[Float64](unsafe_uninit_length=n_rows * n_features)
    for r in range(n_rows):
        for f in range(n_features):
            raw[r * n_features + f] = _uniform(
                seed_offset + UInt64(f * n_rows + r)
            )

    # Ingestion: the transpose into `features[f * n_rows + r]`, which is what
    # the binner reads and what the Python wrapper's `_arrays.column_major`
    # produces for a NumPy caller. Timed, because LightGBM's Dataset
    # construction pays it and this harness used to skip it.
    var t_ingest = perf_counter_ns()
    var features = to_column_major(raw, n_rows, n_features)
    var t_ingest_end = perf_counter_ns()
    # The source matrix is dead here and a million by fifty of it is 400 MB.
    # A benchmark that holds it through training measures the memory pressure
    # (the same reason bench_lightgbm.py's `make_data` generates column by
    # column rather than as one whole-array expression).
    _ = raw^

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

    print("ingest_s:", Float64(t_ingest_end - t_ingest) / 1e9)
    print("binning_s:", Float64(t1 - t0) / 1e9)
    print("train_s:", Float64(t2 - t1) / 1e9)
    # `total_s` is binning plus training, unchanged, so that figures recorded
    # before ingestion was timed still compare. `e2e_s` adds the transpose and
    # is the line that spans the same work as bench_lightgbm.py's `total_s`.
    print("total_s:", Float64(t2 - t0) / 1e9)
    print("e2e_s:", Float64(t2 - t_ingest) / 1e9)
    print("n_trees:", len(booster.trees))
    if objective == BINARY_LOGISTIC:
        print("train_logloss:", loss)
    else:
        print("train_mse:", loss)
