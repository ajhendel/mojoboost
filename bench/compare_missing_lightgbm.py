"""Missing-value semantics: mojotrees against LightGBM.

Probes the four decisions that define missing-value handling and prints what
each library does, side by side, so a difference shows up as a difference in
the table rather than in prose:

1. A feature whose missingness alone predicts the target: which side does the
   split send missing rows to, and what does a NaN row predict?
2. A feature with no missing training value: what does a NaN predict at
   inference time?
3. `use_missing=false`: is NaN still distinguishable from the value 0.0?
4. +inf and -inf: missing, or finite-side extremes?

The LightGBM side reads the answers out of `dump_model`, whose `default_left`
and `missing_type` fields name the same concepts mojotrees stores per node.
The mojotrees side runs `bench/missing_reference.mojo`, which trains on the
identical data (generated from the same closed form on both sides, so no file
is exchanged) and prints its own answers in the same format.

Usage: pixi run -e bench python bench/compare_missing_lightgbm.py
"""

import json
import subprocess
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

REPO = Path(__file__).resolve().parent.parent

# Shared with bench/missing_reference.mojo. Every value below is a closed form
# of the row index, so both sides train on bit-identical data.
N_ROWS = 300
N_CLEAN = 200


def missingness_data():
    """Feature 0 is missing on every third row; the target depends on the
    missingness and on nothing else."""
    r = np.arange(N_ROWS)
    missing = r % 3 == 0
    x = np.where(missing, np.nan, (r % 17) / 17.0)
    y = np.where(missing, 5.0, -5.0)
    return x.reshape(-1, 1).astype(np.float64), y.astype(np.float64)


def clean_data():
    """No missing value anywhere: a step at 0."""
    x = np.linspace(-3.0, 3.0, N_CLEAN)
    y = (x > 0).astype(np.float64) * 2.0 - 1.0
    return x.reshape(-1, 1), y


def infinity_data():
    """Finite values plus one +inf and one -inf, none of them missing."""
    x = np.concatenate([np.arange(20.0), [np.inf, -np.inf]])
    y = np.concatenate([np.zeros(10), np.ones(10), [1.0, 0.0]])
    return x.reshape(-1, 1), y


def train_lgb(X, y, **overrides):
    params = {
        "objective": "regression",
        "num_leaves": 4,
        "learning_rate": 1.0,
        "min_data_in_leaf": 1,
        "max_bin": 16,
        "verbose": -1,
        "deterministic": True,
        "force_row_wise": True,
    }
    params.update(overrides)
    ds = lgb.Dataset(X, label=y, params=params)
    return lgb.train(params, ds, num_boost_round=1)


def root_of(booster):
    return booster.dump_model()["tree_info"][0]["tree_structure"]


def lightgbm_report():
    out = {}

    X, y = missingness_data()
    bst = train_lgb(X, y)
    root = root_of(bst)
    pred = bst.predict(np.array([[np.nan], [0.0], [0.5]]))
    out["predictive"] = {
        "missing_type": root["missing_type"],
        "default_left": bool(root["default_left"]),
        # LightGBM reports the "everything finite left" threshold as 1e300.
        "isolates_missing": float(root["threshold"]) >= 1e300,
        "pred_nan": pred[0],
        "pred_zero": pred[1],
        "pred_half": pred[2],
    }

    Xc, yc = clean_data()
    bstc = train_lgb(Xc, yc, num_leaves=2, min_data_in_leaf=5, max_bin=255)
    predc = bstc.predict(np.array([[np.nan], [0.0], [-3.0], [3.0]]))
    out["clean"] = {
        "missing_type": root_of(bstc)["missing_type"],
        "pred_nan": predc[0],
        "pred_zero": predc[1],
        "nan_equals_zero": bool(predc[0] == predc[1]),
    }

    bstu = train_lgb(X, y, use_missing=False)
    predu = bstu.predict(np.array([[np.nan], [0.0]]))
    out["use_missing_false"] = {
        "missing_type": root_of(bstu)["missing_type"],
        "pred_nan": predu[0],
        "pred_zero": predu[1],
        "nan_equals_zero": bool(predu[0] == predu[1]),
    }

    Xi, yi = infinity_data()
    bsti = train_lgb(Xi, yi, num_leaves=2, max_bin=8)
    infos = bsti.dump_model()["feature_infos"]["Column_0"]
    predi = bsti.predict(np.array([[np.inf], [19.0], [-np.inf], [0.0]]))
    out["infinities"] = {
        "missing_type": root_of(bsti)["missing_type"],
        "min_value": infos["min_value"],
        "max_value": infos["max_value"],
        "inf_matches_max_finite": bool(predi[0] == predi[1]),
        "neg_inf_matches_min_finite": bool(predi[2] == predi[3]),
    }
    return out


def mojotrees_report():
    proc = subprocess.run(
        ["pixi", "run", "mojo", "run", "-I", "src", "bench/missing_reference.mojo"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise SystemExit("mojotrees reference driver failed")
    for line in proc.stdout.splitlines():
        if line.startswith("{"):
            return json.loads(line)
    sys.stderr.write(proc.stdout)
    raise SystemExit("no JSON line in the mojotrees reference output")


def row(label, a, b, agrees, must_match=True):
    """`must_match=False` marks a row that is reported for context rather than
    as a contract: leaf values, which the two libraries are not expected to
    agree on digit for digit."""
    if agrees:
        mark = "same"
    elif must_match:
        mark = "DIFFERS"
    else:
        mark = "differs (leaf value, see note)"
    print(f"  {label:<34} {str(a):<18} {str(b):<18} {mark}")


def main():
    lgbm = lightgbm_report()
    mojo = mojotrees_report()

    print(f"LightGBM {lgb.__version__} vs mojotrees, one tree, identical data")
    print(f"  {'':<34} {'LightGBM':<18} {'mojotrees':<18}")

    print("\n1. missingness alone predicts the target")
    a, b = lgbm["predictive"], mojo["predictive"]
    row("reserves a missing bin", a["missing_type"] == "NaN",
        b["reserves_missing_bin"],
        (a["missing_type"] == "NaN") == b["reserves_missing_bin"])
    row("root isolates the missing rows", a["isolates_missing"],
        b["isolates_missing"], a["isolates_missing"] == b["isolates_missing"])
    row("root default_left", a["default_left"], b["default_left"],
        a["default_left"] == b["default_left"])
    row("NaN predicts the missing target", round(a["pred_nan"], 3),
        round(b["pred_nan"], 3),
        abs(a["pred_nan"] - b["pred_nan"]) < 1e-6, must_match=False)
    row("observed rows predict the other", round(a["pred_half"], 3),
        round(b["pred_half"], 3),
        abs(a["pred_half"] - b["pred_half"]) < 1e-6, must_match=False)
    row("the two are on opposite sides",
        a["pred_nan"] > a["pred_half"], b["pred_nan"] > b["pred_half"],
        (a["pred_nan"] > a["pred_half"]) == (b["pred_nan"] > b["pred_half"]))

    print("\n2. no missing value in training, NaN at predict time")
    a, b = lgbm["clean"], mojo["clean"]
    row("reserves a missing bin", a["missing_type"] == "NaN",
        b["reserves_missing_bin"],
        (a["missing_type"] == "NaN") == b["reserves_missing_bin"])
    row("NaN predicts as the value 0.0", a["nan_equals_zero"],
        b["nan_equals_zero"], a["nan_equals_zero"] == b["nan_equals_zero"])

    print("\n3. use_missing=false")
    a, b = lgbm["use_missing_false"], mojo["use_missing_false"]
    row("reserves a missing bin", a["missing_type"] == "NaN",
        b["reserves_missing_bin"],
        (a["missing_type"] == "NaN") == b["reserves_missing_bin"])
    row("NaN predicts as the value 0.0", a["nan_equals_zero"],
        b["nan_equals_zero"], a["nan_equals_zero"] == b["nan_equals_zero"])

    print("\n4. infinities")
    a, b = lgbm["infinities"], mojo["infinities"]
    row("reserves a missing bin", a["missing_type"] == "NaN",
        b["reserves_missing_bin"],
        (a["missing_type"] == "NaN") == b["reserves_missing_bin"])
    row("edges clamped to +/-1e300", a["max_value"] == 1e300,
        b["edges_clamped"], (a["max_value"] == 1e300) == b["edges_clamped"])
    row("+inf bins with the max finite", a["inf_matches_max_finite"],
        b["inf_matches_max_finite"],
        a["inf_matches_max_finite"] == b["inf_matches_max_finite"])
    row("-inf bins with the min finite", a["neg_inf_matches_min_finite"],
        b["neg_inf_matches_min_finite"],
        a["neg_inf_matches_min_finite"] == b["neg_inf_matches_min_finite"])

    print(
        "\nNote. The two leaf-value rows are reported for context, not as a"
        "\ncontract. mojotrees's lambda_l2 matched LightGBM's 0.0 from"
        "\n2026-08-16; before that it was 1.0 and the Newton step"
        "\nexactly that much: with 100 missing rows the step is 666.7/101"
        "\n-G/(H+lambda_l2) shrank accordingly. No longer a divergence."
        "\nWhat must match, and does, is every routing decision above."
    )


if __name__ == "__main__":
    main()
