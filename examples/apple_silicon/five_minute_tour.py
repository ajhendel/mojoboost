"""Five minute tour of mojotrees on an Apple Silicon Mac.

Run it with the package importable, either from a source checkout

    pixi run build-python
    PYTHONPATH=python python examples/apple_silicon/five_minute_tour.py

or from an installed wheel

    python examples/apple_silicon/five_minute_tour.py

Everything here uses only the standard library, so it runs in the default
pixi environment where numpy is not installed. numpy arrays and pandas
frames work the same way wherever this passes lists.

Nothing in this script measures a speedup. The optional `--time` section
prints wall clock numbers from the machine it runs on, labeled as such, and
draws no conclusion from them. Read examples/apple_silicon/TIMINGS.md for
the table those numbers belong in and the rules for filling it.
"""

import argparse
import math
import os
import tempfile
import time

import mojotrees
from mojotrees import MojoTreesRegressor, gpu_available


# ----------------------------------------------------------------------
# Deterministic sample data, standard library only.
# ----------------------------------------------------------------------

_MASK = (1 << 64) - 1


def _splitmix64(state):
    """One splitmix64 draw. Returns (uniform in [0, 1), next state)."""
    state = (state + 0x9E3779B97F4A7C15) & _MASK
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
    z = z ^ (z >> 31)
    return (z >> 11) * (1.0 / (1 << 53)), state


def make_data(n_rows, n_features, seed=7):
    """A smooth nonlinear regression problem with a little noise.

    The same seed gives the same data on every machine and every run, so
    two runs of this script are comparable to each other.
    """
    state = seed & _MASK
    X = []
    y = []
    for _ in range(n_rows):
        row = []
        for _ in range(n_features):
            value, state = _splitmix64(state)
            row.append(value)
        noise, state = _splitmix64(state)
        target = (
            3.0 * row[0]
            + 2.0 * row[1] * row[1]
            - 1.5 * math.sin(6.28318530718 * row[2])
            + 0.1 * (noise - 0.5)
        )
        X.append(row)
        y.append(target)
    return X, y


def split(X, y, holdout):
    """Last `holdout` rows become the validation set."""
    cut = len(X) - holdout
    return X[:cut], y[:cut], X[cut:], y[cut:]


def rmse(y_true, y_pred):
    total = 0.0
    for a, b in zip(y_true, y_pred):
        d = float(a) - float(b)
        total += d * d
    return math.sqrt(total / len(y_true))


def rule(title):
    print()
    print(title)
    print("-" * len(title))


# ----------------------------------------------------------------------
# 1. What this build can do
# ----------------------------------------------------------------------


def report_build():
    rule("1. What this build can do")
    available = gpu_available()
    print(f"mojotrees version        {mojotrees.__version__}")
    print(f"gpu_available()          {available}")
    print(f"MOJOTREES_DISABLE_GPU    {os.environ.get('MOJOTREES_DISABLE_GPU', '<unset>')}")
    print(f"MOJOTREES_AUTO_MIN_CELLS {os.environ.get('MOJOTREES_AUTO_MIN_CELLS', '<unset>')}")
    print()
    if available:
        print(
            "An accelerator was present when this extension was compiled, so\n"
            "the GPU training path can be requested with device='gpu'. On an\n"
            "Apple Silicon Mac that accelerator is the integrated GPU and the\n"
            "backend is Metal."
        )
    else:
        print(
            "This build reports no accelerator. Either it was compiled on a\n"
            "machine without one, or MOJOTREES_DISABLE_GPU=1 is set. Every\n"
            "step below still runs, on the CPU."
        )
    print(
        "\nAvailability is a property of the build and not of this machine.\n"
        "Mojo resolves it at compile time, so a wheel built where a GPU was\n"
        "present reports one here as well. See src/mojotrees/device.mojo."
    )
    return available


# ----------------------------------------------------------------------
# 2. device="auto"
# ----------------------------------------------------------------------


def show_auto(X, y):
    rule("2. device='auto' picks a backend and records what ran")
    model = MojoTreesRegressor(n_estimators=10, num_leaves=15, device="auto")
    model.fit(X, y)
    print(f"requested device  auto")
    print(f"model.device_     {model.device_}")
    print(
        "\n'auto' resolves to the CPU unless MOJOTREES_AUTO_MIN_CELLS enables\n"
        "the size heuristic. The threshold is disabled by default because no\n"
        "benchmark on any device has established a workload size where the\n"
        "GPU trainer wins, and shipping a crossover number would be a\n"
        "performance claim with nothing behind it."
    )
    return model


# ----------------------------------------------------------------------
# 3. Validation and early stopping
# ----------------------------------------------------------------------


def train_with_validation(X_train, y_train, X_valid, y_valid):
    rule("3. Validation set and early stopping")
    print(
        "Validation metrics are scored on the CPU, so an eval_set with\n"
        "device='gpu' raises rather than falling back. This step trains on\n"
        "the CPU deliberately.\n"
    )
    model = MojoTreesRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        device="cpu",
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_names=["holdout"],
        eval_metric=["l2", "l1"],
        early_stopping_rounds=20,
        min_delta=1e-6,
    )
    history = model.evals_result_["holdout"]["l2"]
    print(f"rounds trained     n_iter_          {model.n_iter_}")
    print(f"best round         best_iteration_  {model.best_iteration_}")
    print(f"best primary l2    best_score_      {model.best_score_:.6f}")
    print(f"patience ran out   stopped_early_   {model.stopped_early_}")
    print(f"first l2 (base score only)          {history[0]:.6f}")
    print(f"last recorded l2                    {history[-1]:.6f}")
    print(
        "\nevals_result_[set][metric][i] is the score after i trees, so index\n"
        "0 is the base score alone."
    )
    return model


# ----------------------------------------------------------------------
# 4. Prediction
# ----------------------------------------------------------------------


def predict(model, X_valid, y_valid):
    rule("4. Prediction")
    preds = list(model.predict(X_valid))
    print(f"rows predicted     {len(preds)}")
    print(f"holdout RMSE       {rmse(y_valid, preds):.6f}")
    print(f"first five         {[round(float(p), 4) for p in preds[:5]]}")
    print(
        "\npredict takes LightGBM's keywords, raw_score, start_iteration,\n"
        "num_iteration, pred_leaf, pred_contrib, and validate_features.\n"
        "num_iteration=None uses best_iteration_, which is what early\n"
        "stopping left behind."
    )
    print(
        "\nNOT AVAILABLE YET. Prediction always runs on the CPU. There is no\n"
        "device argument on predict, and the GPU prediction kernels are a\n"
        "separate piece of work. The call below is a placeholder for what\n"
        "that would look like and is deliberately not executed.\n"
        "\n    # model.predict(X_valid, device='gpu')   # not implemented"
    )
    return preds


# ----------------------------------------------------------------------
# 5. Why that device
# ----------------------------------------------------------------------


def explain_device(available, n_rows, n_features):
    rule("5. Why that device was chosen")
    cells = n_rows * n_features
    threshold = os.environ.get("MOJOTREES_AUTO_MIN_CELLS")
    print(f"workload           {n_rows} rows x {n_features} features = {cells} cells")
    print(f"accelerator        {'present in this build' if available else 'not available'}")
    print(f"auto threshold     {threshold if threshold else 'disabled (default)'}")
    print(f"objective          squared error, single output, covered by the GPU path")
    print(
        "\nThe rules, in the words of src/mojotrees/device.mojo.\n"
        "  cpu   the dependable path, float64 throughout, every objective.\n"
        "  gpu   device resident tree growth. Raises when no accelerator is\n"
        "        present or when the workload is outside what the GPU path\n"
        "        covers. It never silently falls back to the CPU.\n"
        "  auto  the GPU when the complete GPU path covers the workload and\n"
        "        a conservative size heuristic selects it, the CPU otherwise.\n"
        "        The heuristic is off by default, so auto means cpu today."
    )
    print(
        "\nWhat the GPU path covers is single output training with squared\n"
        "error, binary logistic, poisson, huber, quantile, and L1. Multiclass\n"
        "grows one tree per class per round on the CPU only, so device='gpu'\n"
        "raises for it and auto chooses the CPU."
    )
    print(
        "\nNOT AVAILABLE YET. A structured, machine readable version of this\n"
        "explanation is a separate piece of work. The call below is a\n"
        "placeholder and is deliberately not executed.\n"
        "\n    # from mojotrees import explain_device_choice   # not implemented\n"
        "    # report = explain_device_choice(X)              # not implemented"
    )


# ----------------------------------------------------------------------
# 6. Saving and loading
# ----------------------------------------------------------------------


def save_and_load(model, X_valid, preds):
    rule("6. Saving and loading")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "apple_tour.mbst")
        model.save(path)
        size = os.path.getsize(path)
        restored = MojoTreesRegressor.load(path)
        again = list(restored.predict(X_valid))
        identical = all(float(a) == float(b) for a, b in zip(preds, again))
        print(f"written to         {os.path.basename(path)} ({size} bytes)")
        print(f"n_features_in_     {restored.n_features_in_}")
        print(f"best_iteration_    {restored.best_iteration_}")
        print(f"predictions match the original bit for bit   {identical}")
    print(
        "\nThe file holds the model and not the estimator. Hyperparameters,\n"
        "feature names, split gains, and the training device do not travel\n"
        "with it, and a loaded estimator has no device_ for that reason. The\n"
        "ensemble is the same whichever backend trained it. Pickle the whole\n"
        "estimator when you want the rest."
    )


# ----------------------------------------------------------------------
# 7. Optional timing, this machine only
# ----------------------------------------------------------------------


def time_backends(X, y, n_estimators):
    rule("7. Timing on this machine")
    print(
        "These are wall clock seconds measured right now, on this Mac, on\n"
        "this dataset, once each. They are not a benchmark. One run of one\n"
        "shape on one machine supports no claim about either backend.\n"
    )

    fit_kwargs = dict(n_estimators=n_estimators, num_leaves=31, device="cpu")
    start = time.perf_counter()
    MojoTreesRegressor(**fit_kwargs).fit(X, y)
    cpu_seconds = time.perf_counter() - start
    print(f"device='cpu'   {cpu_seconds:8.3f} s")

    if not gpu_available():
        print("device='gpu'        skipped, this build reports no accelerator")
        return
    try:
        start = time.perf_counter()
        gpu_model = MojoTreesRegressor(
            n_estimators=n_estimators, num_leaves=31, device="gpu"
        ).fit(X, y)
        gpu_seconds = time.perf_counter() - start
    except Exception as exc:  # the GPU path refuses rather than falling back
        print(f"device='gpu'        refused, {type(exc).__name__}, {exc}")
        print(
            "\nThat refusal is the design. An explicit device='gpu' either\n"
            "runs on the GPU or raises. Pass device='cpu' or device='auto'\n"
            "to train on the CPU instead."
        )
        return
    print(f"device='gpu'   {gpu_seconds:8.3f} s   (model.device_ = {gpu_model.device_})")
    print(
        f"\nRow for examples/apple_silicon/TIMINGS.md, fill in the machine\n"
        f"and the rest of the protocol yourself.\n"
        f"\n| <chip> | {len(X)} | {len(X[0])} | {n_estimators} | "
        f"{cpu_seconds:.3f} | {gpu_seconds:.3f} | <ratio> | <notes> |"
    )


# ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--features", type=int, default=20)
    parser.add_argument("--holdout", type=int, default=4000)
    parser.add_argument(
        "--time",
        action="store_true",
        help="also fit once per available backend and print the seconds",
    )
    parser.add_argument(
        "--time-rounds",
        type=int,
        default=100,
        help="boosting rounds for the optional timing section",
    )
    args = parser.parse_args()

    print("Native gradient-boosted trees accelerated by the GPU already")
    print("inside every Apple Silicon Mac.")
    print()
    print("Read examples/apple_silicon/README.md for where that headline is")
    print("true today and where it is still a goal.")

    X, y = make_data(args.rows, args.features)
    X_train, y_train, X_valid, y_valid = split(X, y, args.holdout)

    available = report_build()
    show_auto(X_train, y_train)
    model = train_with_validation(X_train, y_train, X_valid, y_valid)
    preds = predict(model, X_valid, y_valid)
    explain_device(available, len(X_train), args.features)
    save_and_load(model, X_valid, preds)
    if args.time:
        time_backends(X_train, y_train, args.time_rounds)

    rule("Done")
    print("Next stops.")
    print("  examples/apple_silicon/README.md   the walkthrough and troubleshooting")
    print("  examples/apple_silicon/TIMINGS.md  the timing table, empty until measured")
    print("  docs/LIGHTGBM_PARITY.md            what matches LightGBM and what does not")


if __name__ == "__main__":
    main()
