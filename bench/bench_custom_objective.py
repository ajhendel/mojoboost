"""Python-callback objective overhead benchmark.

Fits the same dataset twice with the sklearn-style wrapper:

  builtin   objective="regression" (the native SQUARED_ERROR objective)
  callback  objective=<python callable> computing the same derivatives,
            started from the same base score

Both fits produce identical models, so the difference is exactly the cost of
routing the objective through Python: one call per boosting round, plus the
buffer copies on either side of it. The result is reported per round, which
is the number that matters when deciding whether a Python objective is
affordable for a given dataset size.

The native Mojo interface has its own benchmark,
bench/bench_custom_objective.mojo; that one has no Python in the loop at all.

Usage:
    python bench/bench_custom_objective.py [--rows N] [--features N]
                                           [--rounds N] [--repeat N]

Needs numpy and a built extension (`pixi run build-python`).
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python")
)

from mojotrees import MojoTreesRegressor  # noqa: E402

_MASK = (1 << 64) - 1


def _splitmix64(state):
    z = (state + 0x9E3779B97F4A7C15) & _MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
    return z ^ (z >> 31)


def make_dataset(n_rows, n_features):
    """The same counter-based stream as bench_train.mojo, so datasets match
    across drivers."""
    values = np.fromiter(
        (_splitmix64(k) >> 11 for k in range(n_rows * n_features)),
        dtype=np.float64,
        count=n_rows * n_features,
    ) * (1.0 / 9007199254740992.0)
    X = values.reshape(n_features, n_rows).T.copy()
    y = (
        5.0 * X[:, 0]
        + 4.0 * X[:, 1] * X[:, 2]
        + 3.0 * (X[:, 3] - 0.5) ** 2
    )
    return X, y


def squared_error(raw, labels):
    """The custom-objective form of the built-in squared-error objective."""
    return raw - labels, np.ones_like(raw)


def _time_fit(make_estimator, X, y, repeat):
    best = float("inf")
    model = None
    for _ in range(repeat):
        est = make_estimator()
        start = time.perf_counter()
        model = est.fit(X, y)
        best = min(best, time.perf_counter() - start)
    return best, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--features", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    X, y = make_dataset(args.rows, args.features)
    print(
        f"mojotrees python-callback bench: {args.rows} rows x "
        f"{args.features} features, {args.rounds} rounds, "
        f"best of {args.repeat}"
    )

    builtin_s, builtin_model = _time_fit(
        lambda: MojoTreesRegressor(n_estimators=args.rounds), X, y, args.repeat
    )
    callback_s, callback_model = _time_fit(
        lambda: MojoTreesRegressor(
            objective=squared_error,
            base_score="mean",
            n_estimators=args.rounds,
        ),
        X,
        y,
        args.repeat,
    )

    # The comparison is only meaningful if both fits produced the same model.
    worst = float(
        np.abs(builtin_model.predict(X) - callback_model.predict(X)).max()
    )
    if worst != 0.0:
        raise SystemExit(
            f"the two fits disagree (max |diff| {worst}); the timings are "
            "not comparable"
        )

    overhead_s = callback_s - builtin_s
    print(f"builtin_s: {builtin_s:.4f}")
    print(f"callback_s: {callback_s:.4f}")
    print(f"overhead_total_s: {overhead_s:.4f}")
    print(f"overhead_per_round_ms: {1000.0 * overhead_s / args.rounds:.4f}")
    print(f"overhead_pct: {100.0 * overhead_s / builtin_s:.2f}")
    print(f"max_abs_prediction_diff: {worst}")


if __name__ == "__main__":
    main()
