"""LightGBM comparison for LambdaRank.

Two independent checks on the same synthetic ranking data, with matched
parameters on both sides:

1. The NDCG metric. mojotrees's `ndcg_score` is applied to LightGBM's own
   validation predictions and compared against the ndcg@k LightGBM reports
   for that same iteration. Same scores, same labels, same query
   boundaries, so any difference is a difference in the metric itself. The
   two agree to about 1e-9 rather than to machine precision because Mojo's
   `log2` carries around 1e-10 of relative error in the position discounts.

2. Ranking quality. Both libraries train a LambdaRank model and are scored
   on held-out queries at several cutoffs. These are not expected to match
   exactly: tree growth diverges after the first floating-point tie, the
   binners break ties differently, and LightGBM reads its pairwise sigmoid
   from a lookup table where mojotrees evaluates it. Comparable NDCG is the
   claim, not identical models.

Queries are split into train and validation by query, never by row.

Usage:
    python bench/compare_ranking.py [--queries N] [--valid-queries N]
                                    [--features N] [--rounds N] [--seed N]

Needs lightgbm, numpy, and a built extension (`pixi run build-python`).
"""

import argparse
import os
import sys

import lightgbm as lgb
import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python")
)

from mojotrees import MojoTreesRanker, ndcg_score  # noqa: E402

MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
INV_2_53 = 1.0 / 9007199254740992.0

# LightGBM's defaults for the parameters mojotrees also has.
TRUNCATION_LEVEL = 30
SIGMOID = 1.0
EVAL_AT = (1, 3, 5, 10)


def splitmix64(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = (x + np.uint64(0x9E3779B97F4A7C15)) & MASK
        z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & MASK
        z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & MASK
        return z ^ (z >> np.uint64(31))


def uniform(counter: np.ndarray) -> np.ndarray:
    return (splitmix64(counter) >> np.uint64(11)).astype(np.float64) * INV_2_53


def make_ranking_data(n_queries, n_features, seed, docs_lo=4, docs_hi=16):
    """Queries of varying length whose relevance is a noisy monotone
    function of a latent utility, discretized to LightGBM's 0..4 grades.

    Only the first three features carry signal; the rest are noise, so a
    ranker has something to overfit if it is going to."""
    base = np.uint64(seed) * np.uint64(1_000_003)
    sizes = docs_lo + (
        splitmix64(base + np.arange(n_queries, dtype=np.uint64))
        % np.uint64(docs_hi - docs_lo + 1)
    ).astype(np.int64)
    n_rows = int(sizes.sum())

    k = np.arange(n_rows * n_features, dtype=np.uint64) + base + np.uint64(7)
    X = uniform(k).reshape(n_features, n_rows).T.copy()

    noise = uniform(
        np.arange(n_rows, dtype=np.uint64) + base + np.uint64(9_000_017)
    )
    utility = (
        3.0 * X[:, 0]
        + 2.0 * X[:, 1] * X[:, 2]
        + 0.3 * (noise - 0.5)
    )

    # Grade within each query, so every query has a spread of labels: the
    # top document scores 4 and the bottom 0, by rank inside the query.
    y = np.zeros(n_rows, dtype=np.int32)
    start = 0
    for size in sizes:
        end = start + int(size)
        order = np.argsort(-utility[start:end], kind="stable")
        grades = np.linspace(4.0, 0.0, int(size))
        y[start + order] = np.rint(grades).astype(np.int32)
        start = end
    return X, y, sizes.astype(np.int64)


def split_by_query(X, y, group, n_valid_queries):
    """Hold out the last `n_valid_queries` queries whole. No query ever has
    rows on both sides of the split."""
    n_valid_rows = int(group[-n_valid_queries:].sum())
    cut = len(y) - n_valid_rows
    return (
        (X[:cut], y[:cut], group[:-n_valid_queries]),
        (X[cut:], y[cut:], group[-n_valid_queries:]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=800)
    ap.add_argument("--valid-queries", type=int, default=200)
    ap.add_argument("--features", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    if args.valid_queries >= args.queries:
        raise SystemExit("--valid-queries must be smaller than --queries")

    X, y, group = make_ranking_data(args.queries, args.features, args.seed)
    (Xt, yt, gt), (Xv, yv, gv) = split_by_query(X, y, group, args.valid_queries)
    print(
        f"ranking comparison: {args.queries} queries "
        f"({len(yt)} train rows / {len(yv)} valid rows), "
        f"{args.features} features, {args.rounds} rounds"
    )

    shared = dict(
        num_leaves=31,
        learning_rate=0.1,
        n_estimators=args.rounds,
        min_data_in_leaf=20,
        max_bin=255,
    )

    # LightGBM's native API, not its scikit-learn one: this benchmark
    # environment does not install scikit-learn.
    lgb_params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": list(EVAL_AT),
        "num_leaves": shared["num_leaves"],
        "learning_rate": shared["learning_rate"],
        "min_data_in_leaf": shared["min_data_in_leaf"],
        "min_sum_hessian_in_leaf": 1e-3,
        "lambda_l2": 0.0,
        "max_bin": shared["max_bin"],
        "lambdarank_truncation_level": TRUNCATION_LEVEL,
        "sigmoid": SIGMOID,
        "lambdarank_norm": True,
        # mojotrees has no feature bundling; keep the comparison honest.
        "enable_bundle": False,
        "force_row_wise": True,
        "num_threads": 1,
        "verbose": -1,
    }
    train_set = lgb.Dataset(
        Xt, label=yt, group=gt, params=lgb_params, free_raw_data=False
    )
    valid_set = lgb.Dataset(
        Xv,
        label=yv,
        group=gv,
        reference=train_set,
        params=lgb_params,
        free_raw_data=False,
    )
    evals = {}
    lgb_booster = lgb.train(
        lgb_params,
        train_set,
        num_boost_round=args.rounds,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[lgb.record_evaluation(evals)],
    )
    lgb_pred = lgb_booster.predict(Xv)

    # 1. Metric cross-check on LightGBM's own predictions.
    reported = evals["valid"]
    print("\nmetric cross-check (same scores, both NDCG implementations)")
    print(f"{'cutoff':>8} {'lightgbm':>12} {'mojotrees':>12} {'abs diff':>10}")
    worst = 0.0
    for k in EVAL_AT:
        theirs = reported[f"ndcg@{k}"][-1]
        ours = ndcg_score(lgb_pred, yv, gv, at=k)
        worst = max(worst, abs(theirs - ours))
        print(f"{k:>8} {theirs:>12.9f} {ours:>12.9f} {abs(theirs - ours):>10.2e}")
    print(f"largest disagreement: {worst:.2e}")

    # 2. Quality comparison.
    model = MojoTreesRanker(
        lambdarank_truncation_level=TRUNCATION_LEVEL,
        sigmoid=SIGMOID,
        lambdarank_norm=True,
        min_child_hess=1e-3,
        lambda_l2=0.0,
        **shared,
    ).fit(Xt, yt, group=gt)
    our_pred = model.predict(Xv)

    print("\nvalidation NDCG (independently trained models)")
    print(f"{'cutoff':>8} {'lightgbm':>12} {'mojotrees':>12} {'delta':>10}")
    for k in EVAL_AT:
        theirs = ndcg_score(lgb_pred, yv, gv, at=k)
        ours = ndcg_score(our_pred, yv, gv, at=k)
        print(f"{k:>8} {theirs:>12.6f} {ours:>12.6f} {ours - theirs:>+10.6f}")
    print(f"\nlightgbm trees: {lgb_booster.num_trees()}")
    print(f"mojotrees trees: {model.best_iteration_}")


if __name__ == "__main__":
    main()
