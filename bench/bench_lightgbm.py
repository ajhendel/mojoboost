"""LightGBM reference benchmark.

Generates the same synthetic dataset as bench_train.mojo (counter-based
splitmix64, bit-identical values) and trains LightGBM with parameters
matching mojoboost's defaults, so wall time and train loss are directly
comparable.

Usage: python bench/bench_lightgbm.py [--rows N] [--features N]
       [--objective reg|binary] [--threads N]
"""

import argparse
import time

import lightgbm as lgb
import numpy as np

MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
INV_2_53 = 1.0 / 9007199254740992.0


def splitmix64(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = (x + np.uint64(0x9E3779B97F4A7C15)) & MASK
        z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & MASK
        z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & MASK
        return z ^ (z >> np.uint64(31))


def uniform(counter: np.ndarray) -> np.ndarray:
    return (splitmix64(counter) >> np.uint64(11)).astype(np.float64) * INV_2_53


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0.0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def make_data(n_rows: int, n_features: int, objective: str):
    # Value at (row r, feature f) is uniform(f * n_rows + r), matching the
    # column-major counters in bench_train.mojo.
    k = np.arange(n_rows * n_features, dtype=np.uint64)
    X = uniform(k).reshape(n_features, n_rows).T.copy()

    x0, x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    signal = 5.0 * x0 + 4.0 * x1 * x2 + 3.0 * (x3 - 0.5) * (x3 - 0.5)
    u = uniform(np.arange(n_rows, dtype=np.uint64) + np.uint64(n_rows * n_features))
    if objective == "binary":
        p = sigmoid(2.0 * (signal - 3.0))
        y = (u < p).astype(np.float64)
    else:
        y = signal + 0.1 * (u - 0.5)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=100_000)
    ap.add_argument("--features", type=int, default=100)
    ap.add_argument("--objective", choices=["reg", "binary"], default="reg")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    X, y = make_data(args.rows, args.features, args.objective)
    print(
        f"lightgbm bench: {args.rows} rows x {args.features} features, "
        f"{args.objective}, {args.threads} thread(s)"
    )

    params = {
        "objective": "binary" if args.objective == "binary" else "regression",
        "num_leaves": 31,
        "learning_rate": 0.1,
        "min_data_in_leaf": 20,
        "min_sum_hessian_in_leaf": 1e-3,
        "lambda_l2": 1.0,
        "max_bin": 255,
        "num_threads": args.threads,
        "verbose": -1,
        # mojoboost has no feature bundling; keep the comparison honest.
        "enable_bundle": False,
        "force_row_wise": True,
    }

    t0 = time.perf_counter()
    dataset = lgb.Dataset(X, label=y, params=params, free_raw_data=False)
    dataset.construct()
    t1 = time.perf_counter()
    booster = lgb.train(params, dataset, num_boost_round=100)
    t2 = time.perf_counter()

    pred = booster.predict(X)
    if args.objective == "binary":
        p = np.clip(pred, 1e-15, 1.0 - 1e-15)
        loss = -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        loss_name = "train_logloss"
    else:
        loss = float(np.mean((pred - y) ** 2))
        loss_name = "train_mse"

    print(f"binning_s: {t1 - t0:.6f}")
    print(f"train_s: {t2 - t1:.6f}")
    print(f"total_s: {t2 - t0:.6f}")
    print(f"n_trees: {booster.num_trees()}")
    print(f"{loss_name}: {loss}")


if __name__ == "__main__":
    main()
