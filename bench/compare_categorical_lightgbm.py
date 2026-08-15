"""Categorical-feature comparison against LightGBM.

Builds a synthetic dataset whose signal lives entirely in unordered category
groupings, then fits four models on the identical matrix:

- mojotrees with the columns marked categorical
- LightGBM with the same columns marked categorical
- mojotrees treating the columns as ordinary numerical features
- LightGBM treating the columns as ordinary numerical features

The numerical fits are the control. Category codes are assigned so that no
`code <= t` threshold separates the groups, so a model that treats the codes
as ordinal has to carve out one code at a time; the gap between the two rows
of each library is what native categorical support buys, and the gap between
the two libraries is how closely mojotrees tracks LightGBM.

Both libraries get matched hyperparameters, including LightGBM's categorical
ones (`max_cat_to_onehot`, `max_cat_threshold`, `cat_smooth`, `cat_l2`,
`min_data_per_group`). Predictions are not expected to match bit for bit:
mojotrees keeps exact row counts where LightGBM estimates them from Hessian
sums, and their category-dropping rules for very high cardinality differ (see
`src/mojotrees/categorical.mojo`). Held-out error is the comparison.

Usage: python bench/compare_categorical_lightgbm.py
       [--rows N] [--low-cardinality K] [--high-cardinality K]
       [--num-leaves N] [--n-estimators N] [--seed N]

Run it in the bench environment, which has LightGBM:
    pixi run -e bench python bench/compare_categorical_lightgbm.py
"""

import argparse
import sys
import time

import numpy as np

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover - environment guard
    sys.exit(
        "lightgbm is not installed; run this in the bench environment:\n"
        "    pixi run -e bench python bench/compare_categorical_lightgbm.py"
    )

try:
    from mojotrees import MojoTreesRegressor
except ImportError:  # pragma: no cover - environment guard
    sys.exit(
        "mojotrees is not importable; build the extension first:\n"
        "    pixi run build-python"
    )


def make_data(rng, n_rows, low_k, high_k):
    """Two categorical columns plus one numerical column.

    Each category is assigned an effect drawn independently of its code, so
    the mapping from code to effect is arbitrary: the only way to fit it is
    to group categories, not to threshold their codes.
    """
    low = rng.integers(0, low_k, size=n_rows)
    high = rng.integers(0, high_k, size=n_rows)
    num = rng.random(n_rows)

    # Arbitrary per-category effects, uncorrelated with the code order.
    low_effect = rng.normal(0.0, 1.0, size=low_k)
    high_effect = rng.normal(0.0, 1.0, size=high_k)

    y = (
        low_effect[low]
        + high_effect[high]
        + 2.0 * num
        + 0.1 * rng.normal(0.0, 1.0, size=n_rows)
    )
    X = np.column_stack(
        [low.astype(np.float64), high.astype(np.float64), num]
    )
    return X, y


def rmse(pred, y):
    return float(np.sqrt(np.mean((np.asarray(pred) - y) ** 2)))


def fit_mojotrees(X_tr, y_tr, X_te, args, categorical):
    model = MojoTreesRegressor(
        objective="regression",
        num_leaves=args.num_leaves,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        min_data_in_leaf=20,
        lambda_l2=1.0,
        min_child_hess=1e-3,
        max_bin=255,
        categorical_feature=categorical,
        max_cat_to_onehot=args.max_cat_to_onehot,
        max_cat_threshold=args.max_cat_threshold,
        cat_smooth=args.cat_smooth,
        cat_l2=args.cat_l2,
        min_data_per_group=args.min_data_per_group,
    )
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t0
    return model.predict(X_te), fit_s


def fit_lightgbm(X_tr, y_tr, X_te, args, categorical):
    params = {
        "objective": "regression",
        "num_leaves": args.num_leaves,
        "learning_rate": args.learning_rate,
        "min_data_in_leaf": 20,
        "min_sum_hessian_in_leaf": 1e-3,
        "lambda_l2": 1.0,
        "max_bin": 255,
        "num_threads": 1,
        "verbose": -1,
        # mojotrees has no feature bundling; keep the comparison honest.
        "enable_bundle": False,
        "force_row_wise": True,
        "max_cat_to_onehot": args.max_cat_to_onehot,
        "max_cat_threshold": args.max_cat_threshold,
        "cat_smooth": args.cat_smooth,
        "cat_l2": args.cat_l2,
        "min_data_per_group": args.min_data_per_group,
    }
    cat = categorical if categorical else "auto"
    dataset = lgb.Dataset(
        X_tr,
        label=y_tr,
        params=params,
        categorical_feature=cat if categorical else [],
        free_raw_data=False,
    )
    t0 = time.perf_counter()
    booster = lgb.train(params, dataset, num_boost_round=args.n_estimators)
    fit_s = time.perf_counter() - t0
    return booster.predict(X_te), fit_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=20_000)
    ap.add_argument("--low-cardinality", type=int, default=4)
    ap.add_argument("--high-cardinality", type=int, default=60)
    ap.add_argument("--num-leaves", type=int, default=31)
    ap.add_argument("--n-estimators", type=int, default=100)
    ap.add_argument("--learning-rate", type=float, default=0.1)
    ap.add_argument("--max-cat-to-onehot", type=int, default=4)
    ap.add_argument("--max-cat-threshold", type=int, default=32)
    ap.add_argument("--cat-smooth", type=float, default=10.0)
    ap.add_argument("--cat-l2", type=float, default=10.0)
    ap.add_argument("--min-data-per-group", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    X, y = make_data(
        rng, args.rows, args.low_cardinality, args.high_cardinality
    )
    n_train = int(args.rows * 0.8)
    X_tr, y_tr = X[:n_train], y[:n_train]
    X_te, y_te = X[n_train:], y[n_train:]

    print(
        f"categorical comparison: {args.rows} rows, "
        f"feature 0 = {args.low_cardinality} categories, "
        f"feature 1 = {args.high_cardinality} categories, "
        f"feature 2 numerical"
    )
    print(
        f"num_leaves={args.num_leaves} n_estimators={args.n_estimators} "
        f"learning_rate={args.learning_rate}"
    )
    print()

    categorical = [0, 1]
    rows = []
    for label, categorical_arg in (
        ("categorical", categorical),
        ("numerical (control)", []),
    ):
        mb_pred, mb_s = fit_mojotrees(X_tr, y_tr, X_te, args, categorical_arg)
        lgb_pred, lgb_s = fit_lightgbm(X_tr, y_tr, X_te, args, categorical_arg)
        rows.append(
            (
                label,
                rmse(mb_pred, y_te),
                mb_s,
                rmse(lgb_pred, y_te),
                lgb_s,
                float(np.mean(np.abs(np.asarray(mb_pred) - lgb_pred))),
            )
        )

    header = (
        f"{'treatment':<22}{'mojotrees':>12}{'fit_s':>9}"
        f"{'lightgbm':>12}{'fit_s':>9}{'mean|diff|':>12}"
    )
    print(header)
    print("-" * len(header))
    for label, mb, mb_s, lg, lg_s, diff in rows:
        print(
            f"{label:<22}{mb:>12.5f}{mb_s:>9.2f}"
            f"{lg:>12.5f}{lg_s:>9.2f}{diff:>12.5f}"
        )
    print()
    print("test RMSE, lower is better; mean|diff| is between the two")
    print("libraries' predictions on the same treatment.")

    cat_row, num_row = rows
    if cat_row[1] < num_row[1]:
        gain = 100.0 * (1.0 - cat_row[1] / num_row[1])
        print(
            f"\nmojotrees: native categorical splits cut test RMSE by"
            f" {gain:.1f}% against the ordinal control."
        )
    else:
        print(
            "\nmojotrees: native categorical splits did not beat the ordinal"
            " control on this configuration."
        )


if __name__ == "__main__":
    main()
