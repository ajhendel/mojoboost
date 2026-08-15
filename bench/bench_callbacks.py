"""What a per-iteration callback costs.

    pixi run -e pytest python bench/bench_callbacks.py

The claim this measures is the one in python/mojotrees/callback.py: a
callback costs one crossing of the Python boundary per phase per round, and
nothing per row. Two things are reported, because a timing alone would not
establish it:

- the crossing count, asserted against `2 * rounds` (one before-iteration
  call and one after-iteration call), so a regression that started calling
  per row would fail here rather than merely look slow
- the wall clock of four runs, so the per-iteration overhead is a number
  rather than an adjective

The baseline is a fit with `callbacks=None`, which does not cross the
boundary at all: the Mojo bridge checks a captured flag and returns without
touching Python (see `py_callback` in bindings/_mojotrees.mojo). The gap
between that and an inert callback is the cost of the boundary itself; the
gap to `log_evaluation(period=0)` and `record_evaluation` is what a real
callback adds on top.

Rows and features are deliberately large enough that a per-row callback
would be obvious: at 20000 rows a per-row crossing would cost thousands of
times what a per-iteration one does.

Read the two kinds of evidence differently. The crossing count is exact and
reproducible, and it is what rules out per-row work. The timings are not:
the whole fit takes a fraction of a second, the per-iteration cost is a few
tens of microseconds, and on a busy machine the run-to-run spread is the
same size as the thing being measured. Observed here across runs, the
overhead of an inert callback ranged from below the noise floor to about 3%
of a 60-round fit. Treat that as an upper bound of the right order, not as a
figure to quote; for a tighter number, run it on an idle machine and raise
REPEATS.
"""

import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python")
)

from mojotrees import MojoTreesRegressor  # noqa: E402
from mojotrees.callback import (  # noqa: E402
    log_evaluation,
    record_evaluation,
)

N_ROWS = 20000
N_FEATURES = 20
N_VALID = 4000
ROUNDS = 60
REPEATS = 3


def make_data(seed=0):
    gen = np.random.default_rng(seed)
    X = gen.random((N_ROWS + N_VALID, N_FEATURES))
    y = (
        3.0 * X[:, 0]
        + 2.0 * X[:, 1] * X[:, 2]
        - 1.5 * X[:, 3]
        + 0.05 * gen.standard_normal(N_ROWS + N_VALID)
    )
    return X[:N_ROWS], y[:N_ROWS], X[N_ROWS:], y[N_ROWS:]


def mse(y_true, y_pred):
    return float(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2))


def run(X, y, Xv, yv, callbacks):
    model = MojoTreesRegressor(n_estimators=ROUNDS)
    start = time.perf_counter()
    model.fit(
        X,
        y,
        eval_set=[(Xv, yv)],
        eval_metric=("mse", mse),
        callbacks=callbacks,
    )
    return time.perf_counter() - start, model


def timed(X, y, Xv, yv, build):
    """Best of REPEATS, which is the honest summary for a latency: the
    minimum is the run least disturbed by everything else on the machine."""
    times = []
    for _ in range(REPEATS):
        elapsed, model = run(X, y, Xv, yv, build())
        times.append(elapsed)
    return min(times), statistics.median(times), model


def main():
    X, y, Xv, yv = make_data()
    print(
        f"{N_ROWS} rows x {N_FEATURES} features, {N_VALID} validation rows, "
        f"{ROUNDS} rounds, best of {REPEATS}\n"
    )

    # A counting callback doubles as the crossing-count evidence.
    counter = {"before": 0, "after": 0}

    def count_before(env):
        counter["before"] += 1

    count_before.before_iteration = True

    def count_after(env):
        counter["after"] += 1

    history = {}
    cases = [
        ("no callbacks (baseline)", lambda: None),
        ("inert callback", lambda: [count_before, count_after]),
        ("log_evaluation(period=0)", lambda: [log_evaluation(period=0)]),
        ("record_evaluation", lambda: [record_evaluation(history)]),
    ]

    baseline = None
    noise = 0.0
    for name, build in cases:
        if name.startswith("inert"):
            counter["before"] = counter["after"] = 0
        best, median, model = timed(X, y, Xv, yv, build)
        rounds = model.n_iter_
        if baseline is None:
            baseline = best
            # The spread between the best and median baseline run is the
            # resolution of this measurement. A delta smaller than it is
            # noise, not a cost, and is reported as such below.
            noise = median - best
            overhead = f"  (noise floor +/-{noise * 1e3:.1f} ms)"
        else:
            delta = best - baseline
            if abs(delta) <= noise:
                overhead = "  (within the noise floor)"
            else:
                per_iter = delta / max(rounds, 1)
                overhead = (
                    f"  (+{delta * 1e3:6.1f} ms total, "
                    f"{per_iter * 1e6:6.1f} us/iteration)"
                )
        print(f"{name:28s} best {best:6.3f}s  median {median:6.3f}s{overhead}")

    # The crossing count is the part that has to be exact. The inert case
    # ran REPEATS times, and one call of each phase per round.
    rounds_run = ROUNDS
    expected = REPEATS * rounds_run
    print(
        f"\ncrossings: {counter['before']} before-iteration, "
        f"{counter['after']} after-iteration, "
        f"expected {expected} of each "
        f"({REPEATS} runs x {rounds_run} rounds)"
    )
    if counter["before"] != expected or counter["after"] != expected:
        print(
            "FAIL: the callback is not being called exactly once per phase "
            "per round",
            file=sys.stderr,
        )
        return 1
    print("one crossing per phase per round, and none per row: confirmed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
