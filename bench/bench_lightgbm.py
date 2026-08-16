"""LightGBM reference benchmark, standalone and as an interleaved arm.

Generates the same synthetic dataset as bench_train.mojo (counter-based
splitmix64, bit-identical values) and trains LightGBM with parameters
matching mojotrees's defaults, so wall time and train loss are directly
comparable.

This module has two callers and the second one is why it is shaped the way
it is.

Run directly, it is the standalone reference: it now takes a repeat count
and reports minimum, median, maximum and spread, in the same reduction the
Mojo harness uses, so the comparator has a *measured* noise floor instead of
an assumed one. Before this it trained once and printed one number, and a
single sample has a spread of zero by construction rather than a noise floor
of zero. On a machine whose measured drift across time windows runs to
factors of two and three, a single-sample comparator cannot settle a
few-percent question no matter how many decimal places it carries.

Imported, it is the `lightgbm` arm of bench/bench_train_gpu.mojo, reached
through Mojo's Python interop so that a LightGBM measurement is taken in the
same process, in the same time window, alternating with the mojotrees arms,
under the same repeat count and the same median-with-spread reduction and
the same resolved-versus-indistinguishable verdict every other arm gets.
`InterleavedArm` is the whole of that interface. Keep `main()` behind the
`__name__` guard: importing this module must generate no data and train
nothing.

The parameter alignment is not restated here. It is imported from
bench/real_data/scenarios.py, which is where every entry is justified next
to the reason it exists, so the two comparisons in this repository cannot
drift apart by one of them being edited. One entry is dropped rather than
inherited and `SPEED_EXCLUDED_ALIGNMENT` says which and why.

Usage: python bench/bench_lightgbm.py [--rows N] [--features N]
       [--objective reg|binary] [--threads N] [--repeats N] [--rounds N]
"""

import argparse
import importlib.util
import os
import statistics
import time

import lightgbm as lgb
import numpy as np


def _load_scenarios():
    """`bench/real_data/scenarios.py`, loaded by path rather than by name.

    Resolved from this file's own location, not the working directory,
    because the Mojo harness imports this module from wherever the benchmark
    was launched. Loaded through importlib rather than by putting
    `bench/real_data` on sys.path, because this module is imported into the
    harness's process and a directory of a dozen short module names on the
    front of the search path is a change to somebody else's imports.
    """
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "real_data", "scenarios.py"
    )
    spec = importlib.util.spec_from_file_location("bench_scenarios", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scenarios = _load_scenarios()

MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
INV_2_53 = 1.0 / 9007199254740992.0

#: Alignment keys deliberately not inherited from scenarios.LIGHTGBM_ALIGNMENT.
#:
#: `deterministic` is a reproducibility setting, and LightGBM's own
#: documentation records that it can slow training down. The real_data
#: differential wants it because a repeat run there has to be a repeat run of
#: the same fit. Here the measured quantity *is* the wall time, so switching
#: on a setting whose cost lands entirely on the comparator's side of the
#: comparison would hand mojotrees an advantage that came from the harness
#: rather than from the code. The fit is still pinned by `seed`, which is
#: inherited and costs nothing.
SPEED_EXCLUDED_ALIGNMENT = ("deterministic",)

#: The name each mojotrees default arrives under from the Mojo harness,
#: against the canonical `scenarios.BASE_PARAMS` key it has to equal.
#:
#: The alignment table says what the two engines were *meant* to agree on.
#: This says whether they still do. The harness reads its own
#: `BoosterParams.default()` and `TreeParams.default()` and hands the values
#: over, and `check_alignment` refuses to train if any of them has moved,
#: because a defaults change on the mojotrees side would otherwise turn a
#: like-for-like comparison into an unlabelled one that still prints two
#: numbers and a percentage.
MOJO_DEFAULT_ROUTING = {
    "rounds": "n_estimators",
    "learning_rate": "learning_rate",
    "num_leaves": "num_leaves",
    "min_data_in_leaf": "min_data_in_leaf",
    "lambda_l2": "lambda_l2",
    "lambda_l1": "lambda_l1",
    "min_child_hess": "min_child_hess",
    "max_depth": "max_depth",
    "max_bin": "max_bin",
}


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


def make_data(n_rows: int, n_features: int, objective: str, seed: int = 0):
    """The bench_train.mojo dataset, bit-identical, one feature at a time.

    Value at (row r, feature f) is uniform(f * n_rows + r), matching the
    column-major counters in bench_train.mojo and in bench_train_gpu.mojo.

    The generation is column by column into a preallocated array rather than
    one whole-array expression. The values are identical either way, since
    splitmix64 is elementwise, but the whole-array form holds several
    intermediates of n_rows * n_features uint64 at once, which at a million
    rows by fifty features is gigabytes of transient peak. That peak matters
    now that this runs inside the Mojo harness's own process, alongside the
    harness's feature array and binned matrix: a benchmark that pushes the
    machine into memory pressure measures the memory pressure.
    """
    seed_offset = np.uint64((seed * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF)
    rows = np.arange(n_rows, dtype=np.uint64)
    X = np.empty((n_rows, n_features), dtype=np.float64)
    for f in range(n_features):
        X[:, f] = uniform(rows + np.uint64(f * n_rows) + seed_offset)

    x0, x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    signal = 5.0 * x0 + 4.0 * x1 * x2 + 3.0 * (x3 - 0.5) * (x3 - 0.5)
    u = uniform(rows + seed_offset + np.uint64(n_rows * n_features))
    if objective == "binary":
        p = sigmoid(2.0 * (signal - 3.0))
        y = (u < p).astype(np.float64)
    else:
        y = signal + 0.1 * (u - 0.5)
    return X, y


def lgbm_params(objective: str, threads: int, n_rows: int) -> dict:
    """The parameter dict, aligned to mojotrees's defaults.

    The shared values come from `scenarios.BASE_PARAMS` and the settings
    that exist to remove a difference come from
    `scenarios.LIGHTGBM_ALIGNMENT`, minus `SPEED_EXCLUDED_ALIGNMENT`.

    `threads` of 0 leaves `num_threads` unset, which is LightGBM's own
    default of one thread per physical core. Any other value pins it. The
    caller is responsible for making that number mean the same thing as
    MOJOTREES_NUM_WORKERS does on the mojotrees side; nothing here can
    check it, which is why the resolved number is printed.
    """
    base = scenarios.BASE_PARAMS
    params = {
        "objective": "binary" if objective == "binary" else "regression",
        "num_leaves": base["num_leaves"],
        "max_depth": base["max_depth"],
        "learning_rate": base["learning_rate"],
        "min_data_in_leaf": base["min_data_in_leaf"],
        "min_sum_hessian_in_leaf": base["min_child_hess"],
        "lambda_l1": base["lambda_l1"],
        "lambda_l2": base["lambda_l2"],
        "max_bin": base["max_bin"],
        "use_missing": base["use_missing"],
        # mojotrees has no feature bundling; keep the comparison honest.
        # This one is in the alignment too, restated because it is the
        # single largest structural difference between the two engines.
        "enable_bundle": False,
        "verbose": -1,
    }
    for key, value in scenarios.LIGHTGBM_ALIGNMENT.items():
        if key in SPEED_EXCLUDED_ALIGNMENT:
            continue
        params[key] = value
    # `bin_construct_sample_cnt` is deliberately NOT set. It used to be
    # raised to the row count because LightGBM sampled 200000 rows and
    # mojotrees binned every one of them; mojotrees's default is LightGBM's
    # 200000 now, so both engines sample and setting it would pin LightGBM
    # to a fit mojotrees is not doing. Every binning number recorded under
    # the old pin was measured against a constraint this repository imposed
    # and does not describe the stock comparison.
    if threads > 0:
        params["num_threads"] = int(threads)
    return params


def params_summary(params: dict) -> str:
    """The resolved parameters on one line, sorted, for the result record.

    Printed rather than assumed. Every LightGBM number this repository has
    recorded was taken under a parameter set that was written down in one
    file and read from another, and the only defense against that is the
    run stating what it actually used.
    """
    return " ".join(f"{k}={params[k]}" for k in sorted(params))


def check_alignment(mojo_defaults: dict) -> None:
    """Refuse to run if mojotrees's own defaults have moved off the table.

    Raises on the first disagreement rather than reporting it in the result,
    because a mislabelled comparison is worse than a missing one: the number
    would still print, still carry a percentage, and still look like a
    result.
    """
    problems = []
    for name in sorted(mojo_defaults):
        value = mojo_defaults[name]
        canonical = MOJO_DEFAULT_ROUTING.get(name)
        if canonical is None:
            problems.append(f"{name}: no canonical counterpart in BASE_PARAMS")
            continue
        want = scenarios.BASE_PARAMS[canonical]
        if isinstance(want, bool) or isinstance(value, bool):
            same = bool(value) == bool(want)
        else:
            same = float(value) == float(want)
        if not same:
            problems.append(
                f"{name}: mojotrees default {value!r},"
                f" BASE_PARAMS[{canonical!r}] {want!r}"
            )
    if problems:
        raise ValueError(
            "mojotrees defaults no longer match the alignment table in"
            " bench/real_data/scenarios.py, so the two engines would be asked"
            " for different models: " + "; ".join(problems)
        )


def build_dataset(X, y, params: dict):
    """Construct LightGBM's binned Dataset. Returns (dataset, seconds).

    This is LightGBM's binning and it is *measured separately from training*
    on purpose: the mojotrees arms are handed an already binned matrix and
    time only the boosting, so a comparison that folded LightGBM's binning
    into its training number would be comparing two different spans of work.
    """
    t0 = time.perf_counter()
    dataset = lgb.Dataset(X, label=y, params=params, free_raw_data=False)
    dataset.construct()
    return dataset, time.perf_counter() - t0


def train_once(params: dict, dataset, rounds: int):
    """One complete boosting run over an already constructed Dataset.

    Returns (booster, seconds). The Dataset is reused across repeats rather
    than rebuilt, which is what makes the repeats measure training. LightGBM
    only drops the booster's Python reference to it when a run finishes, so
    the constructed handle survives every repeat.

    `keep_training_booster=True` is not a tuning choice. Left at its default,
    `lgb.train` finishes by serializing the fitted model to a string and
    parsing it back, and that round trip lands inside the wall time even
    though it is not training. mojotrees's arms pay no such cost, so leaving
    it in would put a serialization charge on the comparator's side of a
    comparison that is meant to be about boosting. Removing it makes
    LightGBM's number smaller and the margin against it harder to claim,
    which is the only direction a correction like this is allowed to move.
    Every LightGBM figure recorded before this change carries the round trip;
    the two are not interchangeable to better than about a percent.
    """
    t0 = time.perf_counter()
    booster = lgb.train(
        params,
        dataset,
        num_boost_round=int(rounds),
        keep_training_booster=True,
    )
    return booster, time.perf_counter() - t0


def train_loss(booster, X, y, objective: str):
    """Mean training loss, in the same definition the Mojo harness uses.

    Returns (name, value). Squared error is the mean of the squared
    residual and binary is the mean log loss with the same 1e-15 clip, so
    the number sits beside `_train_loss` in bench_train_gpu.mojo and is read
    against it directly. A speed comparison with no loss column beside it is
    how a worse model gets recorded as a win.
    """
    pred = booster.predict(X)
    if objective == "binary":
        p = np.clip(pred, 1e-15, 1.0 - 1e-15)
        return "train_logloss", float(
            -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        )
    return "train_mse", float(np.mean((pred - y) ** 2))


def reduce_samples(values):
    """(minimum, median, maximum, spread) over one arm's repeats.

    The same reduction bench_train_gpu.mojo applies to its own arms: the
    minimum leads because it is the sample least contaminated by thermal
    drift and by one-time setup, and the spread beside it, taken as
    (max - min) / min, is what says whether the minimum can be trusted at
    all.
    """
    lo = min(values)
    hi = max(values)
    return lo, statistics.median(values), hi, (hi - lo) / lo


class InterleavedArm:
    """The `lightgbm` arm of bench/bench_train_gpu.mojo, held across repeats.

    Everything that is not training happens in the constructor: data
    generation, parameter resolution and Dataset construction. `train` then
    does one boosting run and nothing else, so the Mojo harness can put its
    own clock around a call that contains only the work the mojotrees arms
    are also timing.

    The first `train` call additionally pays LightGBM's one-time thread pool
    creation, which is the same shape of cost as the device setup the first
    GPU repeat pays. It is handled the same way, by the summary leading with
    the minimum rather than the mean, and it is one more reason to pass a
    repeat count above one.

    The data is regenerated here rather than handed over from Mojo. The two
    generators are the same counter-based splitmix64 sequence and produce
    bit-identical values, and copying fifty million doubles across the
    interop boundary one PythonObject at a time would cost more than the
    benchmark does.

    `mojo_defaults` is the harness's own resolved `BoosterParams.default()`
    and `TreeParams.default()`, checked against the alignment table before
    anything is generated. It is a guard, not an input: the parameters
    LightGBM trains under still come from `scenarios.BASE_PARAMS`.
    """

    def __init__(
        self,
        n_rows: int,
        n_features: int,
        objective: str,
        threads: int,
        seed: int,
        **mojo_defaults,
    ):
        if objective not in ("reg", "binary"):
            raise ValueError(
                "the lightgbm arm handles 'reg' and 'binary'; multiclass would"
                " need the harness's quantile bucketing replicated here and it"
                " has not been"
            )
        check_alignment(mojo_defaults)
        self.objective = objective
        self.rounds = int(mojo_defaults.get("rounds", 100))
        self.threads = int(threads)
        self.X, self.y = make_data(n_rows, n_features, objective, seed)
        self.params = lgbm_params(objective, threads, n_rows)
        self.dataset, self.binning_s = build_dataset(self.X, self.y, self.params)
        self.booster = None

    def summary(self) -> str:
        return params_summary(self.params)

    def resolved_threads(self) -> int:
        """The pinned thread count, or 0 when LightGBM chose for itself."""
        return int(self.params.get("num_threads", 0))

    def train(self) -> int:
        """One boosting run. Returns the tree count; the caller holds the clock."""
        self.booster, _ = train_once(self.params, self.dataset, self.rounds)
        return int(self.booster.num_trees())

    def loss(self) -> float:
        """The training loss of the most recent `train`, outside any timed region."""
        if self.booster is None:
            raise RuntimeError("loss requested before any training run")
        return train_loss(self.booster, self.X, self.y, self.objective)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=100_000)
    ap.add_argument("--features", type=int, default=100)
    ap.add_argument("--objective", choices=["reg", "binary"], default="reg")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument(
        "--repeats",
        type=int,
        default=1,
        help=(
            "training runs over one constructed Dataset. Three or more before"
            " quoting a number against anything."
        ),
    )
    args = ap.parse_args()
    if args.repeats < 1:
        ap.error("repeats must be at least 1")

    X, y = make_data(args.rows, args.features, args.objective, args.seed)
    params = lgbm_params(args.objective, args.threads, args.rows)
    print(
        f"lightgbm bench: {args.rows} rows x {args.features} features, "
        f"{args.objective}, {args.threads} thread(s), seed {args.seed}"
    )
    print(f"params: {params_summary(params)}")
    print(f"repeats: {args.repeats}")
    if args.repeats < 3:
        print(
            "warning: fewer than 3 repeats cannot separate a real difference"
            " from machine drift; pass a repeat count of 3 or more before"
            " reporting a delta"
        )

    dataset, binning_s = build_dataset(X, y, params)

    samples = []
    booster = None
    for rep in range(args.repeats):
        booster, seconds = train_once(params, dataset, args.rounds)
        samples.append(seconds)
        print(f"run {rep + 1} lightgbm train_s: {seconds:.6f}")

    lo, med, hi, spread = reduce_samples(samples)
    loss_name, loss = train_loss(booster, X, y, args.objective)

    print(f"binning_s: {binning_s:.6f}")
    # The repeats again, on one line, in the order they ran. The `run N`
    # lines above already carry them, and this one exists so that a single
    # copied line still carries the dispersion; the Mojo harness prints the
    # same `_train_s_samples` line per arm for the same reason.
    print("train_s_samples: " + " ".join(f"{s:.6f}" for s in samples))
    print(f"train_s: {lo:.6f}")
    print(f"train_s_median: {med:.6f}")
    print(f"train_s_max: {hi:.6f}")
    print(f"spread_pct: {round(spread * 100.0, 1)}")
    # (max - min) / median rather than / min: the arm's dispersion rather
    # than its excursion above its own best sample, and the one to quote
    # beside a median. `spread_pct` above is unchanged so that figures
    # recorded before this line still mean what they meant.
    print(f"spread_pct_of_median: {round((hi - lo) / med * 100.0, 1)}")
    print(f"total_s: {binning_s + lo:.6f}")
    print(f"n_trees: {booster.num_trees()}")
    print(f"{loss_name}: {loss}")


if __name__ == "__main__":
    main()
