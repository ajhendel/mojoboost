#!/usr/bin/env python3
"""Check the LightGBM parity contract against the repository.

`docs/LIGHTGBM_PARITY.md` claims things about this repository. This script
checks the claims a script can check, so that a supported row cannot quietly
become false:

1. every status cell in the contract uses the documented vocabulary
2. every repository path the contract cites exists
3. a fixed inventory of rows still says `supported` (a deleted or downgraded
   row fails here, which is the point)
4. the public Python symbols those rows depend on still exist
5. the public Mojo symbols those rows depend on are still exported
6. the Mojo test suites the contract cites are wired into a pixi task,
   except for the ones the contract itself lists as known gaps

Standard library only, and nothing here builds or imports the extension
module, so it runs on a bare checkout and in CPU-only CI. When the extension
happens to be importable, check 4 also runs against the live package.

    python3 tools/check_parity.py

Exit status is 0 when every check passes and 1 otherwise.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTRACT = ROOT / "docs" / "LIGHTGBM_PARITY.md"
PY_API = ROOT / "python" / "mojoboost" / "__init__.py"
MOJO_INIT = ROOT / "src" / "mojoboost" / "__init__.mojo"
PIXI = ROOT / "pixi.toml"

STATUSES = {"supported", "partial", "different", "deferred", "unsupported"}

# Paths named in the contract that belong to LightGBM rather than to this
# repository, so their absence here means nothing.
FOREIGN_PATHS = {"c_api.h"}

# Rows that must keep saying `supported`. Deleting one, renaming it, or
# downgrading it fails this check. Names are the row's first cell with
# backticks stripped.
REQUIRED_SUPPORTED = [
    # estimators
    "LGBMRegressor",
    "LGBMClassifier",
    "LGBMRanker",
    # constructor parameters
    "num_leaves",
    "max_depth",
    "learning_rate",
    "n_estimators",
    "min_child_weight",
    "min_child_samples",
    "subsample",
    "subsample_freq",
    "reg_alpha",
    "reg_lambda",
    "importance_type",
    # fit and predict
    "fit(X, y)",
    "fit(sample_weight=)",
    "fit(group=)",
    "fit(eval_set=)",
    "fit(eval_names=)",
    "fit(categorical_feature=)",
    "predict(X)",
    "predict(raw_score=)",
    "predict(start_iteration=) / predict(num_iteration=)",
    "predict(pred_leaf=)",
    "predict(validate_features=)",
    "score(X, y)",
    # fitted attributes
    "n_features_in_",
    "feature_names_in_",
    "classes_ / n_classes_",
    "feature_importances_",
    "evals_result_",
    # data inputs
    "2-D numpy array",
    "pandas DataFrame",
    "Python lists / sequences",
    "NaN as missing",
    # core parameters
    "num_iterations",
    "device_type",
    "num_class",
    "min_data_in_leaf",
    "min_sum_hessian_in_leaf",
    "bagging_fraction",
    "bagging_freq",
    "bagging_seed",
    "feature_fraction",
    "feature_fraction_bynode",
    "feature_fraction_seed",
    "early_stopping_round",
    "early_stopping_min_delta",
    "lambda_l1",
    "lambda_l2",
    "top_rate / other_rate",
    "min_data_per_group",
    "max_cat_threshold",
    "cat_l2",
    "cat_smooth",
    "max_cat_to_onehot",
    "monotone_constraints",
    "interaction_constraints",
    "max_bin",
    "use_missing",
    "categorical_feature",
    "alpha",
    "lambdarank_truncation_level",
    "lambdarank_norm",
    # objectives
    "regression (l2)",
    "regression_l1 / mae",
    "huber",
    "quantile",
    "poisson",
    "binary",
    "multiclass (softmax)",
    "lambdarank",
    # metrics
    "l2",
    "rmse",
    "binary_logloss",
    "multi_logloss",
    "auc",
    "ndcg",
    "custom metrics (feval)",
    # backends and packaging
    "GPU histogram construction",
    "End-to-end GPU training",
    "macOS arm64 wheel",
    "source build from a clean checkout",
]

# Public Python names the supported rows depend on.
REQUIRED_PY_ALL = [
    "MojoBoostRegressor",
    "MojoBoostClassifier",
    "MojoBoostRanker",
    "NotFittedError",
    "gpu_available",
    "group_from_query_ids",
    "ndcg_score",
]

REQUIRED_PY_METHODS = {
    "MojoBoostRegressor": ["fit", "predict", "score", "save", "load"],
    "MojoBoostClassifier": [
        "fit",
        "predict",
        "predict_proba",
        "score",
        "save",
        "load",
    ],
    "MojoBoostRanker": ["fit", "predict", "score", "save", "load"],
}

# Hyperparameters the contract lists as supported or as accepted aliases.
REQUIRED_BASE_PARAMS = [
    "num_leaves",
    "max_depth",
    "learning_rate",
    "n_estimators",
    "min_data_in_leaf",
    "min_child_samples",
    "lambda_l1",
    "lambda_l2",
    "reg_alpha",
    "reg_lambda",
    "min_child_hess",
    "min_child_weight",
    "max_bin",
    "device",
    "device_type",
    "interaction_constraints",
    "monotone_constraints",
    "bagging_fraction",
    "subsample",
    "bagging_freq",
    "subsample_freq",
    "bagging_seed",
    "boosting",
    "boosting_type",
    "top_rate",
    "other_rate",
    "goss_seed",
    "feature_fraction",
    "feature_fraction_bynode",
    "feature_fraction_seed",
    "use_missing",
    "categorical_feature",
    "max_cat_to_onehot",
    "max_cat_threshold",
    "cat_smooth",
    "cat_l2",
    "min_data_per_group",
    "importance_type",
]

# Validation-set arguments the contract lists as supported.
REQUIRED_FIT_ARGS = {
    "MojoBoostRegressor": [
        "sample_weight",
        "eval_set",
        "eval_names",
        "eval_metric",
        "early_stopping_rounds",
        "min_delta",
    ],
    "MojoBoostClassifier": [
        "sample_weight",
        "eval_set",
        "eval_names",
        "eval_metric",
        "early_stopping_rounds",
        "min_delta",
    ],
    "MojoBoostRanker": ["group", "sample_weight"],
}

# Prediction options the contract lists as supported.
REQUIRED_PREDICT_ARGS = {
    "MojoBoostRegressor": [
        "raw_score",
        "start_iteration",
        "num_iteration",
        "pred_leaf",
        "validate_features",
    ],
    "MojoBoostClassifier": [
        "raw_score",
        "start_iteration",
        "num_iteration",
        "pred_leaf",
        "validate_features",
    ],
}

REQUIRED_FITTED_ATTRS = [
    "n_features_in_",
    "feature_names_in_",
    "device_",
    "classes_",
    "n_classes_",
    "best_iteration_",
    "evals_result_",
    "best_score_",
]

# Public Mojo names the supported rows depend on, as exported from
# src/mojoboost/__init__.mojo.
REQUIRED_MOJO_EXPORTS = [
    "Model",
    "MulticlassModel",
    "Booster",
    "BoosterParams",
    "TreeParams",
    "fit",
    "fit_custom",
    "fit_multiclass",
    "fit_ranker",
    "fit_with_metrics",
    "train",
    "train_with_valid",
    "train_multiclass",
    "train_multiclass_with_valid",
    "train_custom",
    "train_with_metrics",
    "train_ranker",
    "train_gpu",
    "train_custom_gpu",
    "train_multiclass_gpu",
    "save_model",
    "load_model",
    "save_multiclass_model",
    "load_multiclass_model",
    "rmse",
    "binary_auc",
    "binary_log_loss",
    "multiclass_log_loss",
    "ndcg",
    "ndcg_at_cutoffs",
    "gain_importance",
    "split_importance",
    "gpu_available",
    "resolve_device",
    "fit_categorical_spec",
    "find_best_categorical_split",
    "MonotoneConstraints",
    "InteractionConstraints",
    "BaggingParams",
    "GossParams",
    "CustomMetric",
    "MetricSuite",
]

# Mojo suites the contract cites that no pixi task runs. Empty is the
# healthy state: every suite the contract offers as evidence is also run.
# When a suite has to be added here, it belongs in the "Known gaps" section
# of the contract too, and this check enforces that in both directions.
KNOWN_UNWIRED_TESTS: set[str] = set()

PATH_RE = re.compile(
    r"`([A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:mojo|py|md|sh|toml|yml|yaml|h|txt))`"
)
PIPE = "\x00"


def fail(problems, message):
    problems.append(message)


def strip_ticks(cell):
    return cell.replace("`", "").replace("**", "").strip()


def tables(text):
    """Every markdown table in `text` as (header cells, [row cells])."""
    lines = text.splitlines()
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("|") and i + 1 < len(lines):
            sep = lines[i + 1]
            if set(sep.replace("|", "").replace(" ", "")) <= {"-", ":"} and (
                "-" in sep
            ):
                header = split_row(line)
                rows = []
                i += 2
                while i < len(lines) and lines[i].startswith("|"):
                    rows.append(split_row(lines[i]))
                    i += 1
                out.append((header, rows))
                continue
        i += 1
    return out


def split_row(line):
    """Cells of a markdown table row, honoring escaped pipes."""
    cells = line.replace("\\|", PIPE).strip().strip("|").split("|")
    return [c.replace(PIPE, "|").strip() for c in cells]


def check_contract(text, problems):
    """Statuses, cited paths, and the required-supported inventory."""
    supported = set()
    seen = {}
    for header, rows in tables(text):
        if len(header) < 2 or header[1].strip().lower() != "status":
            continue
        for row in rows:
            if len(row) < 2:
                fail(problems, f"contract: short table row {row!r}")
                continue
            item = strip_ticks(row[0])
            status = strip_ticks(row[1])
            if status not in STATUSES:
                fail(
                    problems,
                    f"contract: row {item!r} has status {status!r}; expected "
                    "one of " + ", ".join(sorted(STATUSES)),
                )
                continue
            seen.setdefault(item, set()).add(status)
            if status == "supported":
                supported.add(item)

    for item in REQUIRED_SUPPORTED:
        if item in supported:
            continue
        if item in seen:
            fail(
                problems,
                f"contract: {item!r} must stay 'supported' but is now "
                + "/".join(sorted(seen[item]))
                + ". Downgrading it means removing it from "
                "REQUIRED_SUPPORTED in tools/check_parity.py and saying why "
                "in the contract.",
            )
        else:
            fail(
                problems,
                f"contract: the row for {item!r} is gone. A supported row "
                "cannot disappear silently: restore it, or remove it from "
                "REQUIRED_SUPPORTED in tools/check_parity.py.",
            )

    for match in sorted(set(PATH_RE.findall(text))):
        if match in FOREIGN_PATHS:
            continue
        if not (ROOT / match).exists():
            fail(problems, f"contract cites {match}, which does not exist")

    return supported


def python_api(problems):
    """Public Python names, parsed rather than imported."""
    tree = ast.parse(PY_API.read_text())
    classes = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }
    functions = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }

    exported = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            target = node.targets[0]
            if getattr(target, "id", None) == "__all__":
                exported = [
                    elt.value
                    for elt in node.value.elts
                    if isinstance(elt, ast.Constant)
                ]
    for name in REQUIRED_PY_ALL:
        if name not in exported:
            fail(problems, f"python: {name} is missing from __all__")

    for cls, methods in REQUIRED_PY_METHODS.items():
        node = classes.get(cls)
        if node is None:
            fail(problems, f"python: class {cls} is gone")
            continue
        defined = {
            m.name for m in node.body if isinstance(m, ast.FunctionDef)
        }
        for method in methods:
            if method not in defined:
                fail(problems, f"python: {cls}.{method} is gone")

    for name in ("gpu_available", "group_from_query_ids", "ndcg_score"):
        if name not in functions:
            fail(problems, f"python: {name}() is gone")

    base = classes.get("_Base")
    if base is None:
        fail(problems, "python: _Base is gone")
        return
    init = next(
        (
            m
            for m in base.body
            if isinstance(m, ast.FunctionDef) and m.name == "__init__"
        ),
        None,
    )
    if init is None:
        fail(problems, "python: _Base.__init__ is gone")
    else:
        params = {a.arg for a in init.args.args} | {
            a.arg for a in init.args.kwonlyargs
        }
        for name in REQUIRED_BASE_PARAMS:
            if name not in params:
                fail(
                    problems,
                    f"python: hyperparameter {name} is gone from "
                    "_Base.__init__",
                )

    attrs = next(
        (
            m
            for m in base.body
            if isinstance(m, ast.Assign)
            and getattr(m.targets[0], "id", None) == "_FITTED_ATTRS"
        ),
        None,
    )
    if attrs is None:
        fail(problems, "python: _Base._FITTED_ATTRS is gone")
    else:
        recorded = {
            elt.value
            for elt in attrs.value.elts
            if isinstance(elt, ast.Constant)
        }
        for name in REQUIRED_FITTED_ATTRS:
            if name not in recorded:
                fail(
                    problems,
                    f"python: fitted attribute {name} is gone from "
                    "_FITTED_ATTRS",
                )

    for method, required in (
        ("fit", REQUIRED_FIT_ARGS),
        ("predict", REQUIRED_PREDICT_ARGS),
    ):
        for cls, args in required.items():
            node = classes.get(cls)
            if node is None:
                continue
            func = next(
                (
                    m
                    for m in node.body
                    if isinstance(m, ast.FunctionDef) and m.name == method
                ),
                None,
            )
            if func is None:
                continue
            params = {a.arg for a in func.args.args} | {
                a.arg for a in func.args.kwonlyargs
            }
            for name in args:
                if name not in params:
                    fail(
                        problems,
                        f"python: {cls}.{method} lost the {name} argument",
                    )


def python_runtime(problems):
    """The same names again from the built package, when it imports."""
    sys.path.insert(0, str(ROOT / "python"))
    try:
        import mojoboost  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - depends on the build
        print(f"  (skipped live import: {type(exc).__name__}: {exc})")
        return
    finally:
        sys.path.pop(0)
    for name in REQUIRED_PY_ALL:
        if not hasattr(mojoboost, name):
            fail(problems, f"python: the built package has no {name}")
    print("  (live import checked)")


def mojo_exports(problems):
    """Names re-exported from src/mojoboost/__init__.mojo."""
    text = MOJO_INIT.read_text()
    names = set()
    for block in re.findall(r"import\s*\(([^)]*)\)", text):
        names.update(n.strip() for n in block.replace("\n", "").split(","))
    for line in text.splitlines():
        m = re.match(r"from\s+\.\w+\s+import\s+([^(].*)$", line.strip())
        if m:
            names.update(n.strip() for n in m.group(1).split(","))
    names.discard("")
    for name in REQUIRED_MOJO_EXPORTS:
        if name not in names:
            fail(
                problems,
                f"mojo: {name} is no longer exported from "
                "src/mojoboost/__init__.mojo",
            )


def unwired_tests(text, problems):
    """Cited Mojo suites that no pixi task runs."""
    cited = {
        path
        for path in set(PATH_RE.findall(text))
        if path.startswith("tests/") and path.endswith(".mojo")
    }
    pixi = PIXI.read_text()
    wired = {
        path
        for path in cited
        if re.search(r"\b" + re.escape(path) + r"\b", pixi)
    }
    unwired = cited - wired
    if unwired != KNOWN_UNWIRED_TESTS:
        newly = sorted(unwired - KNOWN_UNWIRED_TESTS)
        fixed = sorted(KNOWN_UNWIRED_TESTS - unwired)
        if newly:
            fail(
                problems,
                "pixi: these cited suites are not run by any pixi task and "
                "are not listed as known gaps: " + ", ".join(newly),
            )
        if fixed:
            fail(
                problems,
                "pixi: these suites are wired in now, so drop them from "
                "KNOWN_UNWIRED_TESTS in tools/check_parity.py and from the "
                "'Known gaps' section of the contract: " + ", ".join(fixed),
            )


def main():
    problems = []
    if not CONTRACT.exists():
        print(f"missing {CONTRACT.relative_to(ROOT)}")
        return 1
    text = CONTRACT.read_text()

    print("checking the LightGBM parity contract")
    supported = check_contract(text, problems)
    python_api(problems)
    python_runtime(problems)
    mojo_exports(problems)
    unwired_tests(text, problems)

    print(f"  {len(supported)} rows marked supported")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("  ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
