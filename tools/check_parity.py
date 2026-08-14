#!/usr/bin/env python3
"""Check the LightGBM parity contract against the repository.

`docs/LIGHTGBM_PARITY.md` claims things about this repository. This script
checks the claims a script can check, so that a supported row cannot quietly
become false and a deferred row cannot quietly stay false:

1. every status cell in the contract uses the documented vocabulary
2. every repository path the contract cites exists
3. a fixed inventory of rows still says `supported` (a deleted or downgraded
   row fails here, which is the point)
4. the public Python symbols those rows depend on still exist
5. the public Mojo symbols those rows depend on are still exported
6. the Mojo test suites the contract cites are wired into a pixi task,
   except for the ones the contract itself lists as known gaps
7. no `deferred` or `unsupported` row has gone stale, judged by resolving
   the *public symbols* behind it
8. the capability-level table in section 0 uses exactly the seven levels
   `docs/CAPABILITY_LEVELS.md` defines, and does not contradict itself

Check 7 deserves a word, because it is the one that can be got wrong in a
way that produces a false claim. A watch is a list of public names: an entry
in `mojoboost.__all__` or in a submodule's `__all__`, a method on a public
class, an argument of a public method, a fitted attribute, or a name
re-exported from `src/mojoboost/__init__.mojo`. **A file path is never a
probe.** A module can sit in `src/mojoboost/` fully implemented, fully
tested, and reachable by nobody, which is exactly the state several modules
are in today; upgrading a row because a file appeared would turn this script
into a generator of false claims rather than a defense against them. When
every name behind a watched row resolves, the script does not decide what
the row should say. It fails and asks a human to re-audit that one row.

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
LEVELS_DOC = ROOT / "docs" / "CAPABILITY_LEVELS.md"
PY_PKG = ROOT / "python" / "mojoboost"
PY_API = PY_PKG / "__init__.py"
PY_BASIC = PY_PKG / "basic.py"
MOJO_INIT = ROOT / "src" / "mojoboost" / "__init__.mojo"
PIXI = ROOT / "pixi.toml"

STATUSES = {"supported", "partial", "different", "deferred", "unsupported"}

# The seven capability levels, in the order section 0 of the contract lists
# them. `docs/CAPABILITY_LEVELS.md` is the normative definition and this
# script checks that both files agree.
LEVEL_NAMES = [
    "implemented",
    "integrated",
    "publicly reachable",
    "focused-tested",
    "differential-tested",
    "hardware-validated",
    "release-packaged",
]
LEVEL_CELLS = {"yes", "no", "n/a"}

# Paths named in the contract that belong to LightGBM rather than to this
# repository, so their absence here means nothing.
FOREIGN_PATHS = {"c_api.h"}

# Rows that must keep saying `supported`. Deleting one, renaming it, or
# downgrading it fails this check. Names are the row's first cell with
# backticks stripped.
REQUIRED_SUPPORTED = [
    # the functional API
    "Booster",
    "Dataset",
    "train",
    "booster_",
    "Booster(model_file=) / Booster(model_str=)",
    "Booster.feature_importance",
    "Booster.num_feature / num_trees / num_model_per_iteration",
    "Booster.eval / eval_train / eval_valid / add_valid",
    "Booster.feature_name",
    "Dataset construction and construct",
    "Dataset.get_field and the typed accessors (label, weight, group, init_score)",
    "Dataset.get_data",
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
    "fit(eval_sample_weight=)",
    "fit(eval_metric=)",
    "fit(categorical_feature=)",
    "predict(X)",
    "predict(raw_score=)",
    "predict(start_iteration=) / predict(num_iteration=)",
    "predict(pred_leaf=)",
    "predict(pred_contrib=)",
    "predict(validate_features=)",
    "score(X, y)",
    # fitted attributes
    "n_features_in_",
    "feature_names_in_",
    "classes_ / n_classes_",
    "feature_importances_",
    "evals_result_",
    "n_iter_",
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
    "average_precision",
    "ndcg",
    "map",
    "custom metrics (feval)",
    # backends and packaging
    "GPU histogram construction",
    "End-to-end GPU training",
    "Device-resident objectives",
    "GPU split selection",
    "GPU multiclass",
    "Exact TreeSHAP contributions",
    "source build from a clean checkout",
]

# Public Python names the supported rows depend on.
REQUIRED_PY_ALL = [
    "Booster",
    "Dataset",
    "train",
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
        "eval_sample_weight",
        "early_stopping_rounds",
        "min_delta",
        "callbacks",
    ],
    "MojoBoostClassifier": [
        "sample_weight",
        "eval_set",
        "eval_names",
        "eval_metric",
        "eval_sample_weight",
        "early_stopping_rounds",
        "min_delta",
        "callbacks",
    ],
    "MojoBoostRanker": [
        "group",
        "sample_weight",
        "eval_set",
        "eval_group",
        "eval_metric",
    ],
}

# Prediction options the contract lists as supported.
REQUIRED_PREDICT_ARGS = {
    "MojoBoostRegressor": [
        "raw_score",
        "start_iteration",
        "num_iteration",
        "pred_leaf",
        "pred_contrib",
        "validate_features",
    ],
    "MojoBoostClassifier": [
        "raw_score",
        "start_iteration",
        "num_iteration",
        "pred_leaf",
        "pred_contrib",
        "validate_features",
    ],
    "MojoBoostRanker": [
        "raw_score",
        "start_iteration",
        "num_iteration",
        "pred_leaf",
        "pred_contrib",
        "validate_features",
    ],
}

# Public methods of the functional API, which lives in basic.py rather than
# in the package __init__.
REQUIRED_BASIC_METHODS = {
    "Dataset": [
        "construct",
        "num_data",
        "num_feature",
        "num_bin",
        "get_label",
        "get_weight",
        "get_group",
        "get_init_score",
        "get_data",
        "get_field",
        "feature_name",
        "categorical_feature",
    ],
    "Booster": [
        "update",
        "predict",
        "eval",
        "eval_train",
        "eval_valid",
        "add_valid",
        "feature_importance",
        "save_model",
        "model_to_string",
        "model_from_string",
        "current_iteration",
        "num_trees",
        "num_model_per_iteration",
        "num_feature",
        "feature_name",
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
    "stopped_early_",
    "n_iter_",
    "categorical_feature_",
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
    "Dataset",
    "train_dataset",
    "train_dataset_multiclass",
    "train_dataset_ranker",
    "update_dataset",
    "update_dataset_multiclass",
    "train_more",
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
    # the sparse chain, reachable from Python since contract v2
    "CscMatrix",
    "CsrMatrix",
    "fit_csc",
    "fit_multiclass_csc",
    "predict_csr",
    "predict_proba_csr",
    "predict_raw_csr",
    "train_sparse",
    # contributions
    "predict_contrib",
    "predict_contrib_multiclass",
]

# Rows the contract calls `deferred` or `unsupported` whose public symbols
# are worth watching, so that wiring one of them up without editing the row
# fails loudly instead of aging into a false claim.
#
# Probe grammar, all symbol based and never path based:
#   pyall:NAME              NAME in mojoboost.__all__
#   pysub:MODULE:NAME       NAME in python/mojoboost/MODULE.py's __all__
#   pymethod:CLASS.METHOD   METHOD defined on CLASS (__init__.py or basic.py)
#   pyarg:CLASS.METHOD:ARG  ARG is a parameter of CLASS.METHOD
#   pyattr:NAME             NAME in _Base._FITTED_ATTRS
#   mojo:NAME               NAME re-exported from src/mojoboost/__init__.mojo
#
# A watch fires only when *every* probe resolves, and firing means "re-audit
# this row", not "this row is wrong in a particular direction".
STALE_DEFERRED_WATCHES = {
    "Sequence": ["pyall:Sequence"],
    "DaskLGBMRegressor / DaskLGBMClassifier / DaskLGBMRanker": [
        "pyall:DaskMojoBoostRegressor",
    ],
    "register_logger": ["pyall:register_logger"],
    "plot_importance": ["pyall:plot_importance"],
    "plot_metric": ["pyall:plot_metric"],
    "plot_split_value_histogram": ["pyall:plot_split_value_histogram"],
    "plot_tree / create_tree_digraph": ["pyall:plot_tree"],
    "fit(init_score=)": ["pyarg:MojoBoostRegressor.fit:init_score"],
    "fit(eval_init_score=)": ["pyarg:MojoBoostRegressor.fit:eval_init_score"],
    "fit(init_model=)": ["pyarg:MojoBoostRegressor.fit:init_model"],
    "objective_": ["pyattr:objective_"],
    "Booster.rollback_one_iter / reset_parameter": [
        "pymethod:Booster.rollback_one_iter",
    ],
    "Booster.lower_bound / upper_bound": ["pymethod:Booster.lower_bound"],
    "Booster.get_leaf_output / set_leaf_output / shuffle_models / refit": [
        "pymethod:Booster.refit",
    ],
    "Booster.set_network / free_network": ["pymethod:Booster.set_network"],
    "Dataset.subset": ["pymethod:Dataset.subset"],
    "Dataset.position": ["pymethod:Dataset.position"],
    "Dataset.save_binary / add_features_from": ["pymethod:Dataset.save_binary"],
    # the six implemented-but-unintegrated Mojo modules, watched by the name
    # an integrator would have to export to reach them
    "enable_bundle": ["mojo:fit_bundles"],
    "min_gain_to_split": ["mojo:passes_min_gain"],
    "extra_trees / extra_seed": ["mojo:extra_split_stream"],
    "path_smooth": ["mojo:smooth_leaf_output"],
    "feature_contri": ["mojo:FeaturePenalties"],
    "monotone_penalty": ["mojo:FeaturePenalties"],
    "a real transport (MPI, sockets, gRPC)": ["mojo:RankAddress"],
    "num_machines / local_listen_port / time_out / machine_list_filename / machines": [
        "mojo:RankAddress",
    ],
    "Apple-specific tiling policy": ["mojo:derive_block_threads"],
    "Apple GPU tuning policy": ["mojo:derive_block_threads"],
    "Split gains in a dump": ["mojo:split_gains"],
    "LightGBM model file interop": ["mojo:parse_lgbm_model"],
    "Exclusive feature bundling": ["mojo:fit_bundles"],
    "Distributed transport": ["mojo:RankAddress"],
    "Remaining tree-parameter rules": ["mojo:passes_min_gain"],
    "Per-level feature sampling": ["mojo:select_level_features"],
    "Dask adapter (`mojoboost.dask`)": ["pyall:DaskMojoBoostRegressor"],
    "Explainable device selection (`mojoboost.device_selection`)": [
        "pyall:explain_device_choice",
    ],
}

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


def status_rows(text):
    """(item, status) for every row of every status table in the contract."""
    out = []
    for header, rows in tables(text):
        if len(header) < 2 or header[1].strip().lower() != "status":
            continue
        for row in rows:
            if len(row) < 2:
                continue
            out.append((strip_ticks(row[0]), strip_ticks(row[1]), row))
    return out


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


def levels_table(text):
    """The section 0 capability-level table, as (header, rows)."""
    for header, rows in tables(text):
        if len(header) >= 3 and header[0].strip().lower() == "capability":
            if header[1].strip().lower() == "status" and len(header) >= 10:
                return header, rows
    return None, None


def check_levels_doc(problems):
    """docs/CAPABILITY_LEVELS.md defines exactly the seven levels."""
    if not LEVELS_DOC.exists():
        fail(
            problems,
            "docs/CAPABILITY_LEVELS.md is missing; the contract's section 0 "
            "cites it as the definition of its columns",
        )
        return
    defined = []
    for header, rows in tables(LEVELS_DOC.read_text()):
        if header and header[0].strip().lower() == "level":
            defined = [strip_ticks(r[0]).lower() for r in rows if r]
            break
    if defined != LEVEL_NAMES:
        fail(
            problems,
            "docs/CAPABILITY_LEVELS.md defines "
            + (", ".join(defined) if defined else "no levels")
            + "; expected exactly, and in this order: "
            + ", ".join(LEVEL_NAMES),
        )


def check_levels(text, problems):
    """The section 0 table: column names, cell vocabulary, and the two
    implications the levels carry by definition."""
    header, rows = levels_table(text)
    if header is None:
        fail(
            problems,
            "contract: section 0's capability-level table is missing. It is "
            "what keeps 'supported' from collapsing seven facts into one; "
            "see docs/CAPABILITY_LEVELS.md",
        )
        return
    columns = [strip_ticks(c).lower() for c in header[2:9]]
    if columns != LEVEL_NAMES:
        fail(
            problems,
            "contract: section 0's level columns are "
            + ", ".join(columns)
            + "; expected " + ", ".join(LEVEL_NAMES),
        )
        return
    if strip_ticks(header[-1]).lower() != "evidence":
        fail(problems, "contract: section 0's last column must be Evidence")

    for row in rows:
        if len(row) != len(header):
            fail(
                problems,
                f"contract: section 0 row {strip_ticks(row[0])!r} has "
                f"{len(row)} cells, expected {len(header)}",
            )
            continue
        name = strip_ticks(row[0])
        status = strip_ticks(row[1])
        cells = [strip_ticks(c).lower() for c in row[2:9]]
        for level, cell in zip(LEVEL_NAMES, cells):
            if cell not in LEVEL_CELLS:
                fail(
                    problems,
                    f"contract: section 0 row {name!r} says {cell!r} for "
                    f"{level}; expected one of "
                    + ", ".join(sorted(LEVEL_CELLS)),
                )
        if not strip_ticks(row[-1]):
            fail(
                problems,
                f"contract: section 0 row {name!r} has no evidence. A level "
                "with nothing behind it is the thing this table exists to "
                "prevent",
            )
        scored = dict(zip(LEVEL_NAMES, cells))
        if scored.get("integrated") == "yes" and (
            scored.get("implemented") != "yes"
        ):
            fail(
                problems,
                f"contract: section 0 row {name!r} is integrated but not "
                "implemented, which is not a state that exists",
            )
        if scored.get("publicly reachable") == "yes" and (
            scored.get("integrated") != "yes"
        ):
            fail(
                problems,
                f"contract: section 0 row {name!r} is publicly reachable but "
                "not integrated, which is not a state that exists",
            )
        if scored.get("publicly reachable") == "yes" and status in (
            "deferred",
            "unsupported",
        ):
            fail(
                problems,
                f"contract: section 0 row {name!r} is publicly reachable and "
                f"still says {status!r}. A user can call it, so the row owes "
                "them a real status",
            )


def mojo_export_names():
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
    return names


def symbol_index():
    """Every public symbol the watches can probe, parsed rather than
    imported. Keys are the probe kinds documented on
    STALE_DEFERRED_WATCHES."""
    index = {
        "pyall": set(),
        "pysub": {},
        "methods": {},
        "args": {},
        "pyattr": set(),
        "mojo": mojo_export_names(),
    }

    def read_module(path):
        try:
            return ast.parse(path.read_text())
        except (OSError, SyntaxError):
            return None

    def dunder_all(tree):
        for node in tree.body:
            if isinstance(node, ast.Assign):
                target = node.targets[0]
                if getattr(target, "id", None) == "__all__":
                    return {
                        elt.value
                        for elt in node.value.elts
                        if isinstance(elt, ast.Constant)
                    }
        return set()

    for path in (PY_API, PY_BASIC):
        tree = read_module(path)
        if tree is None:
            continue
        if path == PY_API:
            index["pyall"] = dunder_all(tree)
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            methods = index["methods"].setdefault(node.name, set())
            for member in node.body:
                if isinstance(
                    member, (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    methods.add(member.name)
                    params = {a.arg for a in member.args.args} | {
                        a.arg for a in member.args.kwonlyargs
                    }
                    index["args"][(node.name, member.name)] = params
                elif isinstance(member, ast.Assign):
                    for target in member.targets:
                        if isinstance(target, ast.Name):
                            methods.add(target.id)
                            if (
                                node.name == "_Base"
                                and target.id == "_FITTED_ATTRS"
                            ):
                                index["pyattr"] = {
                                    elt.value
                                    for elt in member.value.elts
                                    if isinstance(elt, ast.Constant)
                                }

    for path in sorted(PY_PKG.glob("*.py")):
        if path.name.startswith("_") or path.name == "basic.py":
            continue
        tree = read_module(path)
        if tree is not None:
            index["pysub"][path.stem] = dunder_all(tree)

    return index


def resolves(probe, index):
    """True when a probe's public symbol exists today."""
    kind, _, rest = probe.partition(":")
    if kind == "pyall":
        return rest in index["pyall"]
    if kind == "pysub":
        module, _, name = rest.partition(":")
        return name in index["pysub"].get(module, set())
    if kind == "pymethod":
        cls, _, method = rest.partition(".")
        return method in index["methods"].get(cls, set())
    if kind == "pyarg":
        target, _, arg = rest.rpartition(":")
        cls, _, method = target.partition(".")
        return arg in index["args"].get((cls, method), set())
    if kind == "pyattr":
        return rest in index["pyattr"]
    if kind == "mojo":
        return rest in index["mojo"]
    raise ValueError(f"unknown probe kind {kind!r} in {probe!r}")


def stale_deferred(text, problems):
    """Watched `deferred`/`unsupported` rows whose public symbols now all
    exist. A file appearing is never enough; see the module docstring."""
    index = symbol_index()
    statuses = {}
    for item, status, _ in status_rows(text):
        statuses.setdefault(item, set()).add(status)

    for item, probes in STALE_DEFERRED_WATCHES.items():
        name = strip_ticks(item)
        if name not in statuses:
            fail(
                problems,
                f"check_parity: STALE_DEFERRED_WATCHES watches {name!r}, "
                "which is not a row in the contract. Rename the watch or "
                "drop it; a watch on nothing checks nothing.",
            )
            continue
        if not (statuses[name] & {"deferred", "unsupported"}):
            continue
        try:
            landed = [p for p in probes if resolves(p, index)]
        except ValueError as exc:
            fail(problems, f"check_parity: {exc}")
            continue
        if len(landed) == len(probes):
            fail(
                problems,
                f"contract: {name!r} still says "
                + "/".join(sorted(statuses[name] & {"deferred", "unsupported"}))
                + ", but every public symbol behind it now exists ("
                + ", ".join(landed)
                + "). Re-audit that row: what a user can now reach, what is "
                "tested, and what is still missing. Then either update the "
                "row or drop the watch with a reason.",
            )


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

    basic = ast.parse(PY_BASIC.read_text())
    basic_classes = {
        node.name: node
        for node in basic.body
        if isinstance(node, ast.ClassDef)
    }
    basic_functions = {
        node.name for node in basic.body if isinstance(node, ast.FunctionDef)
    }
    if "train" not in basic_functions:
        fail(problems, "python: basic.train() is gone")
    for cls, methods in REQUIRED_BASIC_METHODS.items():
        node = basic_classes.get(cls)
        if node is None:
            fail(problems, f"python: class {cls} is gone from basic.py")
            continue
        defined = {
            m.name
            for m in node.body
            if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
        } | {
            target.id
            for m in node.body
            if isinstance(m, ast.Assign)
            for target in m.targets
            if isinstance(target, ast.Name)
        }
        for method in methods:
            if method not in defined:
                fail(problems, f"python: {cls}.{method} is gone")

    train = next(
        (n for n in basic.body
         if isinstance(n, ast.FunctionDef) and n.name == "train"),
        None,
    )
    if train is not None:
        params = {a.arg for a in train.args.args} | {
            a.arg for a in train.args.kwonlyargs
        }
        for name in ("valid_sets", "valid_names", "init_model"):
            if name not in params:
                fail(problems, f"python: basic.train lost the {name} argument")

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
    names = mojo_export_names()
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
    check_levels_doc(problems)
    check_levels(text, problems)
    python_api(problems)
    python_runtime(problems)
    mojo_exports(problems)
    stale_deferred(text, problems)
    unwired_tests(text, problems)

    header, level_rows = levels_table(text)
    print(f"  {len(supported)} rows marked supported")
    print(f"  {len(level_rows or [])} capabilities scored against the levels")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("  ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
