"""First five minutes with mojoboost, and a check that the install works.

Run it against an installed package

    python examples/install_smoke.py

or against a source checkout, which needs `pixi run build-python` first

    PYTHONPATH=python python examples/install_smoke.py

It covers the same ground as "The first five minutes" in
docs/INSTALLATION.md, printing what each step produced: diagnostics, a
tiny regression, a validation set with early stopping, a bit-exact save
and load, and what each of the three device values does on this machine.
Nothing here is a benchmark and no step prints a speed.

Standard library only, so it runs in the default pixi environment where
numpy is not installed. numpy arrays, pandas frames, polars frames, and
SciPy sparse matrices work the same way wherever this passes lists.

Exit status is 0 when every step it could run succeeded. A refused
`device="gpu"` is not a failure; on most machines it is the expected
result, and the script says which.
"""

import math
import os
import platform
import sys
import tempfile


# ----------------------------------------------------------------------
# 0. Import, with the most common installation failures named
# ----------------------------------------------------------------------

try:
    import mojoboost
    from mojoboost import MojoBoostRegressor, gpu_available
except ModuleNotFoundError as exc:
    if exc.name == "mojoboost":
        sys.exit(
            "mojoboost is not importable at all. Either install a wheel,\n"
            "or from a source checkout run `pixi run build-python` and\n"
            "rerun this with PYTHONPATH=python. See docs/INSTALLATION.md."
        )
    sys.exit(f"a module mojoboost needs is missing:\n\n    {exc}")
except ImportError as exc:
    # Two very different failures arrive here as the same exception type.
    # A missing `_mojoboost.so` is swallowed as a ModuleNotFoundError by
    # the import machinery's fromlist handling and comes back out of
    # `from . import ... _mojoboost ...` as "cannot import name"; a .so
    # that exists but cannot resolve its MAX runtime libraries fails in
    # dlopen and keeps the loader's own message.
    if "cannot import name '_mojoboost'" in str(exc):
        sys.exit(
            "the mojoboost package imported but its compiled extension is\n"
            "missing. In a source checkout, build it with\n"
            "\n    pixi run build-python\n\n"
            "and rerun this script with PYTHONPATH=python. See\n"
            "docs/INSTALLATION.md, state 3."
        )
    sys.exit(
        "the mojoboost extension is present but failed to load, which\n"
        "usually means its bundled MAX runtime libraries were not found:\n"
        f"\n    {exc}\n\n"
        "From a source checkout, run through `pixi run` so the toolchain\n"
        "environment is active, or build a self-contained wheel with\n"
        "`pixi run build-wheel`. From an installed wheel this is a bug;\n"
        "please report it with the full message. See docs/INSTALLATION.md,\n"
        "'Missing runtime library'."
    )


def rule(title):
    print()
    print(title)
    print("-" * len(title))


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


def make_data(n_rows, n_features, seed=11):
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
            2.5 * row[0]
            + 1.5 * row[1] * row[1]
            - math.sin(6.28318530718 * row[2])
            + 0.1 * (noise - 0.5)
        )
        X.append(row)
        y.append(target)
    return X, y


def rmse(y_true, y_pred):
    total = 0.0
    for a, b in zip(y_true, y_pred):
        d = float(a) - float(b)
        total += d * d
    return math.sqrt(total / len(y_true))


# ----------------------------------------------------------------------
# 1. Diagnostics. Paste this block into any bug report.
# ----------------------------------------------------------------------


def diagnostics():
    rule("1. Diagnostics")
    if hasattr(mojoboost, "show_versions"):
        mojoboost.show_versions()
    else:
        # Installs from before show_versions() existed. Same facts, by
        # hand, so this script stays useful against an older wheel.
        print(f"mojoboost                {mojoboost.__version__}")
        print(f"package path             {mojoboost.__file__}")
        print(f"extension                {mojoboost._mojoboost.__file__}")
        print(f"python                   {sys.version.split()[0]}")
        print(f"executable               {sys.executable}")
        print(f"platform                 {platform.platform()}")
        print(f"machine                  {platform.machine()}")
        print(f"gpu_available()          {gpu_available()}")
        for name in sorted(os.environ):
            if name.startswith(("MOJOBOOST_", "MODULAR_")):
                print(f"{name:<24} {os.environ[name]}")
        print(
            "\nThis install predates mojoboost.show_versions(), so it cannot\n"
            "say whether a GPU path was compiled into it. Upgrade to answer\n"
            "that question."
        )
    print(
        "\nWhether an accelerator is available is a property of the build\n"
        "and not of this machine: Mojo resolves has_accelerator() at\n"
        "compile time, so one wheel carries one answer to everybody who\n"
        "installs it. See docs/DEVICE_SELECTION.md."
    )


# ----------------------------------------------------------------------
# 2. A tiny regression, exactly as small as the README's
# ----------------------------------------------------------------------


def tiny_regression():
    rule("2. A tiny regression")
    X = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]
    y = [0.0, 1.0, 4.0, 9.0, 16.0, 25.0]

    model = MojoBoostRegressor(
        n_estimators=20, num_leaves=7, min_data_in_leaf=1
    )
    model.fit(X, y)
    preds = [round(float(p), 4) for p in model.predict([[1.5], [4.5]])]
    print("six rows, one feature, y = x**2")
    print(f"predict([[1.5], [4.5]])  {preds}")
    print(f"model.device_            {model.device_}")
    print(f"n_features_in_           {model.n_features_in_}")
    print(
        "\nmin_data_in_leaf=1 because the default of 20 is larger than this\n"
        "toy dataset, and a model that cannot split predicts one constant."
    )
    return model


# ----------------------------------------------------------------------
# 3. Validation set and early stopping
# ----------------------------------------------------------------------


def validation_and_early_stopping():
    rule("3. Validation set and early stopping")
    X, y = make_data(4000, 8)
    cut = 3200
    X_train, y_train = X[:cut], y[:cut]
    X_valid, y_valid = X[cut:], y[cut:]

    print(
        "Validation metrics are scored on the CPU, so an eval_set with\n"
        "device='gpu' raises rather than falling back. This step asks for\n"
        "the CPU deliberately.\n"
    )
    model = MojoBoostRegressor(
        n_estimators=400,
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
    )
    history = model.evals_result_["holdout"]["l2"]
    preds = list(model.predict(X_valid))
    print(f"rounds trained   n_iter_          {model.n_iter_}")
    print(f"best round       best_iteration_  {model.best_iteration_}")
    print(f"best primary l2  best_score_      {float(model.best_score_):.6f}")
    print(f"patience ran out stopped_early_   {model.stopped_early_}")
    print(f"l2 before any tree                {history[0]:.6f}")
    print(f"l2 at the last recorded round     {history[-1]:.6f}")
    print(f"holdout RMSE                      {rmse(y_valid, preds):.6f}")
    print(
        "\nThe ensemble is rolled back to best_iteration_, so the model you\n"
        "predict with is the one that scored best rather than the one that\n"
        "trained last."
    )
    return model, X_valid, preds


# ----------------------------------------------------------------------
# 4. Save and load, bit for bit
# ----------------------------------------------------------------------


def save_and_load(model, X_valid, preds):
    rule("4. Saving and loading")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "install_smoke.mbst")
        model.save(path)
        restored = MojoBoostRegressor.load(path)
        again = list(restored.predict(X_valid))
        identical = all(float(a) == float(b) for a, b in zip(preds, again))
        print(f"written          {os.path.basename(path)}"
              f" ({os.path.getsize(path)} bytes)")
        print(f"n_features_in_   {restored.n_features_in_}")
        print(f"best_iteration_  {restored.best_iteration_}")
        print(f"predictions identical bit for bit   {identical}")
    print(
        "\nThe file holds the model and not the estimator. Hyperparameters,\n"
        "feature names, and the training device do not travel with it, so a\n"
        "loaded estimator carries no device_. Pickle the estimator when you\n"
        "want the rest."
    )
    return identical


# ----------------------------------------------------------------------
# 5. The three device values, on this machine
# ----------------------------------------------------------------------


def device_selection():
    rule("5. Device selection")
    X, y = make_data(2000, 6)
    fixed = dict(n_estimators=20, num_leaves=15)

    cpu_model = MojoBoostRegressor(device="cpu", **fixed).fit(X, y)
    print(f"device='cpu'     ran on {cpu_model.device_}")

    auto_model = MojoBoostRegressor(device="auto", **fixed).fit(X, y)
    print(f"device='auto'    ran on {auto_model.device_}")

    if not gpu_available():
        print(
            "device='gpu'     not attempted, this build reports no"
            " accelerator"
        )
    else:
        try:
            gpu_model = MojoBoostRegressor(device="gpu", **fixed).fit(X, y)
        except Exception as exc:
            print(f"device='gpu'     refused, {type(exc).__name__}: {exc}")
            print(
                "\nThat refusal is the design. An explicit device='gpu'\n"
                "either runs on the accelerator or raises. It never falls\n"
                "back to the CPU while you believe you used the GPU."
            )
        else:
            print(f"device='gpu'     ran on {gpu_model.device_}")

    threshold = os.environ.get("MOJOBOOST_AUTO_MIN_CELLS")
    shown = threshold if threshold else "disabled (the default)"
    print(f"\nauto threshold   {shown}")
    print(
        "\ndevice='auto' resolves to the CPU on every machine and every\n"
        "workload unless MOJOBOOST_AUTO_MIN_CELLS is set. The crossover\n"
        "table is empty because no benchmark has established a workload\n"
        "size where end-to-end GPU training beats the multicore CPU\n"
        "trainer, and shipping a threshold before that measurement would\n"
        "be a performance claim with nothing behind it. See\n"
        "docs/DEVICE_SELECTION.md and docs/GPU_VALIDATION.md."
    )


# ----------------------------------------------------------------------


def main():
    print("mojoboost installation smoke test")
    print("=" * 33)
    print(
        "\nThe first five minutes from docs/INSTALLATION.md, in five steps.\n"
        "Nothing here is a benchmark and no step prints a speed."
    )

    diagnostics()
    tiny_regression()
    model, X_valid, preds = validation_and_early_stopping()
    identical = save_and_load(model, X_valid, preds)
    device_selection()

    rule("6. Result")
    if not identical:
        print(
            "FAILED. A loaded model did not reproduce its predictions bit\n"
            "for bit, which the serialization format guarantees. Please\n"
            "report this with the diagnostics block above."
        )
        return 1
    print(
        "Every step succeeded. mojoboost is installed and working on this\n"
        "machine, on the CPU at least.\n"
        "\nNext: docs/LIGHTGBM_PARITY.md for what the library promises,\n"
        "docs/GPU_VALIDATION.md before believing anything about an\n"
        "accelerator, and examples/apple_silicon/ for the longer tour."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
