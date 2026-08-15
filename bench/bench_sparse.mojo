"""Sparse vs dense benchmark on a genuinely sparse dataset.

Generates a deterministic synthetic matrix in which each row holds a fixed
small number of nonzeros drawn from a counter-based splitmix64 stream, then
times and measures both paths end to end:

- dense: `fit_bins` + `transform` + `train`, over the full n_rows x
  n_features matrix;
- sparse: `fit_bins_csc` + `transform_csc` + `train_sparse`, over CSC.

Both fit the same model on the same data (implicit zeros are numerical
zeros), so the reported training loss is a correctness check as much as a
benchmark line: the two should agree to rounding.

The dense matrix is materialized only so the dense path has something to run
on. A real sparse workload never builds it, which is the point: the memory
lines below report what each path has to hold.

Usage:
    mojo run -I src bench/bench_sparse.mojo [n_rows] [n_features] [nnz_per_row]

Reported memory is the exact size of the arrays each path allocates, not a
process-level measurement. Reported times are single runs of a single
configuration; rerun on your own hardware before quoting any of it.
"""

from std.sys import argv
from std.time import perf_counter_ns

from mojotrees.binning import fit_bins
from mojotrees.boosting import SQUARED_ERROR, BoosterParams, train
from mojotrees.boosting_sparse import train_sparse
from mojotrees.sparse import CscMatrix, fit_bins_csc, transform_csc
from mojotrees.tree import TreeParams


def _splitmix64(state: UInt64) -> UInt64:
    var z = state + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _uniform(counter: UInt64) -> Float64:
    return Float64(_splitmix64(counter) >> 11) * (1.0 / 9007199254740992.0)


def _mb(bytes: Int) -> Float64:
    return Float64(bytes) / (1024.0 * 1024.0)


def main() raises:
    var n_rows = 200_000
    var n_features = 500
    var nnz_per_row = 10
    var args = argv()
    if len(args) > 1:
        n_rows = Int(String(args[1]))
    if len(args) > 2:
        n_features = Int(String(args[2]))
    if len(args) > 3:
        nnz_per_row = Int(String(args[3]))
    if nnz_per_row < 1 or nnz_per_row > n_features:
        raise Error("nnz_per_row must be in [1, n_features]")
    if n_features < 8:
        raise Error("need at least 8 features")

    # Column-major dense matrix, mostly zeros. Each row picks `nnz_per_row`
    # features by stepping through the feature space with a row-dependent
    # stride, so the nonzeros are spread over every column.
    var dense = List[Float64](capacity=n_rows * n_features)
    dense.resize(n_rows * n_features, 0.0)
    var stride = n_features // nnz_per_row
    for r in range(n_rows):
        var offset = Int(_splitmix64(UInt64(r)) % UInt64(stride))
        for j in range(nnz_per_row):
            var f = offset + j * stride
            if f >= n_features:
                continue
            dense[f * n_rows + r] = (
                4.0 * _uniform(UInt64(r) * UInt64(n_features) + UInt64(f))
                - 2.0
            )

    var target = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        var total = 0.0
        for f in range(8):
            total += (1.0 + 0.37 * Float64(f)) * dense[f * n_rows + r]
        target.append(
            total / 4.0 + 0.05 * (_uniform(UInt64(7_000_000) + UInt64(r)) - 0.5)
        )

    # CSC built directly from the generator, the way a real caller would.
    var row_index = List[Int]()
    var values = List[Float64]()
    var col_offsets = List[Int](capacity=n_features + 1)
    col_offsets.append(0)
    for f in range(n_features):
        for r in range(n_rows):
            var v = dense[f * n_rows + r]
            if v != 0.0:
                row_index.append(r)
                values.append(v)
        col_offsets.append(len(values))
    var nnz = len(values)
    var csc = CscMatrix(row_index^, values^, col_offsets^, n_rows, n_features)

    print(
        "mojotrees sparse bench:",
        n_rows,
        "rows x",
        n_features,
        "features,",
        nnz,
        "nonzeros, density",
        csc.density(),
    )

    var params = BoosterParams(50, 0.1, TreeParams(31, 20, 1.0, 1e-3))
    var max_bins = 255

    var t0 = perf_counter_ns()
    var dmapper = fit_bins(dense, n_rows, n_features, max_bins)
    var t1 = perf_counter_ns()
    var dbinned = dmapper.transform(dense, n_rows)
    var t2 = perf_counter_ns()
    var dbooster = train(dbinned, target, SQUARED_ERROR, params)
    var t3 = perf_counter_ns()

    var t4 = perf_counter_ns()
    var smapper = fit_bins_csc(csc, max_bins)
    var t5 = perf_counter_ns()
    var sbinned = transform_csc(smapper, csc)
    var t6 = perf_counter_ns()
    var sbooster = train_sparse(sbinned, target, SQUARED_ERROR, params)
    var t7 = perf_counter_ns()

    # Sparse binning is bit-identical to dense binning, so both boosters can
    # be scored against the same binned matrix.
    var dmse = 0.0
    var smse = 0.0
    for r in range(n_rows):
        var d = dbooster.predict_row(dbinned, r) - target[r]
        dmse += d * d
        var s = sbooster.predict_row(dbinned, r) - target[r]
        smse += s * s

    # Exact array sizes, in bytes, of what each path has to hold. Int and
    # Float64 are 8 bytes; a bin is 1.
    var dense_raw = n_rows * n_features * 8
    var dense_bins = n_rows * n_features * 1
    var sparse_raw = nnz * 8 + nnz * 8 + (n_features + 1) * 8
    var sparse_bins = (
        nnz * 8 + nnz * 1 + (n_features + 1) * 8 + n_features * 9
    )
    # Tree growth also keeps the entry permutation and its scratch buffer,
    # which the dense path does not need.
    var sparse_index = nnz * 8 * 2

    print("--- memory (MB) ---")
    print("dense raw matrix:      ", _mb(dense_raw))
    print("dense binned matrix:   ", _mb(dense_bins))
    print("dense total:           ", _mb(dense_raw + dense_bins))
    print("sparse raw CSC:        ", _mb(sparse_raw))
    print("sparse binned CSC:     ", _mb(sparse_bins))
    print("sparse growth index:   ", _mb(sparse_index))
    print("sparse total:          ", _mb(sparse_raw + sparse_bins + sparse_index))
    print(
        "sparse/dense:          ",
        Float64(sparse_raw + sparse_bins + sparse_index)
        / Float64(dense_raw + dense_bins),
    )

    print("--- time (s) ---")
    print("dense  fit_bins:       ", Float64(t1 - t0) / 1e9)
    print("dense  transform:      ", Float64(t2 - t1) / 1e9)
    print("dense  train:          ", Float64(t3 - t2) / 1e9)
    print("dense  total:          ", Float64(t3 - t0) / 1e9)
    print("sparse fit_bins_csc:   ", Float64(t5 - t4) / 1e9)
    print("sparse transform_csc:  ", Float64(t6 - t5) / 1e9)
    print("sparse train_sparse:   ", Float64(t7 - t6) / 1e9)
    print("sparse total:          ", Float64(t7 - t4) / 1e9)
    print(
        "sparse speedup:        ",
        Float64(t3 - t0) / Float64(t7 - t4),
    )

    print("--- fit (must agree) ---")
    print("dense  trees:", len(dbooster.trees), "train MSE:", dmse / Float64(n_rows))
    print("sparse trees:", len(sbooster.trees), "train MSE:", smse / Float64(n_rows))
