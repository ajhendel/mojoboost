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
6. the Mojo test suites the contract cites are where `tools/run_tests.sh`
   looks for them, and so are run by `pixi run test`, except for the ones
   the contract itself lists as known gaps
7. no `deferred` or `unsupported` row has gone stale, judged by resolving
   the *public symbols* behind it
8. the capability-level table in section 0 uses exactly the seven levels
   `docs/CAPABILITY_LEVELS.md` defines, and does not contradict itself
9. section 0's `publicly reachable` cells agree with the public symbols
   behind them, in both directions
10. every trainer that returns a booster passes the monotone constraints it
    trained under into it, rather than letting them fall back to the
    "no constraints" default
11. **every** default this library states is LightGBM's stock value, on
    every surface that states one -- the `DEFAULT_*` constants, the four
    hyperparameter structs, and the Python estimator signature -- with the
    handful of deliberate divergences asserted *as* divergences so that
    "fixing" one without retiring its reason fails too; `fit_bins` still
    defaults to the constants; and `feature_pre_filter=true` is still
    refused rather than accepted without the feature deletion it names

Check 9 exists because check 7 has a blind spot: it only looks at rows that
say `deferred` or `unsupported`, so a `partial` row can keep claiming a
capability is out of reach long after the name landed in
`mojotrees.__all__`. Reachability is not a claim about quality, so it is
the one cell a script can decide outright rather than referring to a human.

Reachability of a *module* is a different question, and this script does not
answer it. `tools/connectivity_audit.py` computes the import graph and
`tools/audit_integration.py` checks `docs/INTEGRATION_INVENTORY.md` against
it. Two import graphs would be the duplication all three exist to find.

Check 7 deserves a word, because it is the one that can be got wrong in a
way that produces a false claim. A watch is a list of public names: an entry
in `mojotrees.__all__` or in a submodule's `__all__`, a method on a public
class, an argument of a public method, a fitted attribute, or a name
re-exported from `src/mojotrees/__init__.mojo`. **A file path is never a
probe.** A module can sit in `src/mojotrees/` fully implemented, fully
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
import fnmatch
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTRACT = ROOT / "docs" / "LIGHTGBM_PARITY.md"
LEVELS_DOC = ROOT / "docs" / "CAPABILITY_LEVELS.md"
PY_PKG = ROOT / "python" / "mojotrees"
PY_API = PY_PKG / "__init__.py"
PY_BASIC = PY_PKG / "basic.py"
MOJO_INIT = ROOT / "src" / "mojotrees" / "__init__.mojo"

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
    "min_split_gain",
    "subsample",
    "subsample_freq",
    "colsample_bytree",
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
    "min_gain_to_split",
    "early_stopping_round",
    "early_stopping_min_delta",
    "lambda_l1",
    "lambda_l2",
    "top_rate / other_rate",
    "min_data_per_group",
    "max_cat_threshold",
    "cat_l2",
    "cat_smooth",
    # `max_cat_to_onehot` was here and is DOWNGRADED to partial, 2026-08-16.
    #
    # It is not missing and it is not broken; it is off by one against
    # LightGBM, verified from both sources. LightGBM's test is
    # `num_bin <= max_cat_to_onehot` (`feature_histogram.cpp:183`) where
    # `num_bin` counts a dummy bin pushed at index 0 (`bin.cpp:456-459`), so
    # LightGBM one-hots `max_cat_to_onehot - 1` real categories where we
    # one-hot `max_cat_to_onehot`. At the shared default of 4 they one-hot 3
    # and we one-hot 4, which is a different model on any categorical fit
    # with exactly 4 levels.
    #
    # It stays out of this list until the boundary moves, because "supported"
    # on a parity contract means a user gets LightGBM's behavior from
    # LightGBM's parameter, and here they do not. Owned by the
    # `wide-categorical-bins` lane. Do NOT put it back by editing the row in
    # docs/LIGHTGBM_PARITY.md; put it back by fixing the comparison, in the
    # same commit that removes these lines.
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
    "MojoTreesRegressor",
    "MojoTreesClassifier",
    "MojoTreesRanker",
    "NotFittedError",
    "gpu_available",
    "group_from_query_ids",
    "ndcg_score",
]

REQUIRED_PY_METHODS = {
    "MojoTreesRegressor": ["fit", "predict", "score", "save", "load"],
    "MojoTreesClassifier": [
        "fit",
        "predict",
        "predict_proba",
        "score",
        "save",
        "load",
    ],
    "MojoTreesRanker": ["fit", "predict", "score", "save", "load"],
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
    "colsample_bytree",
    "feature_fraction_bynode",
    "colsample_bynode",
    "feature_fraction_seed",
    "min_gain_to_split",
    "min_split_gain",
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
    "MojoTreesRegressor": [
        "sample_weight",
        "eval_set",
        "eval_names",
        "eval_metric",
        "eval_sample_weight",
        "early_stopping_rounds",
        "min_delta",
        "callbacks",
    ],
    "MojoTreesClassifier": [
        "sample_weight",
        "eval_set",
        "eval_names",
        "eval_metric",
        "eval_sample_weight",
        "early_stopping_rounds",
        "min_delta",
        "callbacks",
    ],
    "MojoTreesRanker": [
        "group",
        "sample_weight",
        "eval_set",
        "eval_group",
        "eval_metric",
    ],
}

# Prediction options the contract lists as supported.
REQUIRED_PREDICT_ARGS = {
    "MojoTreesRegressor": [
        "raw_score",
        "start_iteration",
        "num_iteration",
        "pred_leaf",
        "pred_contrib",
        "validate_features",
    ],
    "MojoTreesClassifier": [
        "raw_score",
        "start_iteration",
        "num_iteration",
        "pred_leaf",
        "pred_contrib",
        "validate_features",
    ],
    "MojoTreesRanker": [
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
# src/mojotrees/__init__.mojo.
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
#   pyall:NAME              NAME in mojotrees.__all__
#   pysub:MODULE:NAME       NAME in python/mojotrees/MODULE.py's __all__
#   pymethod:CLASS.METHOD   METHOD defined on CLASS (__init__.py or basic.py)
#   pyarg:CLASS.METHOD:ARG  ARG is a parameter of CLASS.METHOD
#   pyattr:NAME             NAME in _Base._FITTED_ATTRS
#   mojo:NAME               NAME re-exported from src/mojotrees/__init__.mojo
#   env:MOJOTREES_NAME      the variable is read by a shipping source file
#
# `env:` is the one probe that is not a name a caller writes, and it is here
# because section 2 of docs/COMPATIBILITY_POLICY.md makes the MOJOTREES_*
# variables public: a capability a user turns on that way is reachable, and
# a row claiming otherwise is as wrong as one that misses an export. See
# env_var_names for what counts as reading one.
#
# A watch fires only when *every* probe resolves, and firing means "re-audit
# this row", not "this row is wrong in a particular direction".
STALE_DEFERRED_WATCHES = {
    "Sequence": ["pyall:Sequence"],
    "DaskLGBMRegressor / DaskLGBMClassifier / DaskLGBMRanker": [
        "pyall:DaskMojoTreesRegressor",
    ],
    "register_logger": ["pyall:register_logger"],
    "plot_importance": ["pyall:plot_importance"],
    "plot_metric": ["pyall:plot_metric"],
    "plot_split_value_histogram": ["pyall:plot_split_value_histogram"],
    "plot_tree / create_tree_digraph": ["pyall:plot_tree"],
    "fit(init_score=)": ["pyarg:MojoTreesRegressor.fit:init_score"],
    "fit(eval_init_score=)": ["pyarg:MojoTreesRegressor.fit:eval_init_score"],
    "fit(init_model=)": ["pyarg:MojoTreesRegressor.fit:init_model"],
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
    "Dask adapter (`mojotrees.dask`)": ["pyall:DaskMojoTreesRegressor"],
    "Explainable device selection (`mojotrees.device_selection`)": [
        "pyall:explain_device_choice",
    ],
}

# The `publicly reachable` cell of a section 0 row, tied to the public name
# that decides it.
#
# STALE_DEFERRED_WATCHES only fires on `deferred` and `unsupported` rows, so
# a row that says `partial` can carry `publicly reachable: no` forever while
# the symbol behind it quietly lands in `mojotrees.__all__`. Three rows rot
# that way before this check existed: `cv`, `inspection`, and
# `device_selection` all became reachable in one afternoon and all three
# tables still said they were not.
#
# The probe grammar is the one STALE_DEFERRED_WATCHES documents. The rule is
# an equivalence rather than an implication: the cell must say `yes` when
# every probe resolves and `no` when any of them does not. Both directions
# are false claims, one overstating reach and one understating it, and
# understating it is how a capability a user already has stays undocumented.
#
# Every probe here is `pyall:`, deliberately. A capability is watchable this
# way only when one public name decides it, and on the Python side that
# holds: a name is in `mojotrees.__all__` or it is not. On the Mojo side it
# does not. Section 2 of `docs/COMPATIBILITY_POLICY.md` makes the parameter
# string `parse_params` accepts public too, so a capability can be reachable
# through `enable_bundle=true` with no exported symbol anywhere, and a probe
# on the symbol would then demand `no` for something the C ABI and the CLI
# can already ask for. Those rows stay under STALE_DEFERRED_WATCHES, which
# asks a human rather than deciding.
PUBLIC_REACHABILITY_PROBES = {
    "Cross-validation (mojotrees.cv)": ["pyall:cv", "pyall:CVBooster"],
    "Model inspection and dump (mojotrees.inspection)": [
        "pyall:dump_model",
        "pyall:trees_to_dataframe",
        "pyall:trees_to_records",
        "pyall:get_split_value_histogram",
    ],
    "Explainable device selection (mojotrees.device_selection)": [
        "pyall:explain_device_choice",
    ],
    "Startup diagnostics (mojotrees.diagnostics)": ["pyall:describe_install"],
    "Dask adapter (mojotrees.dask)": ["pyall:DaskMojoTreesRegressor"],
    "Class-batched GPU multiclass rounds": [
        "env:MOJOTREES_GPU_CLASS_BATCH",
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


def check_reachability_cells(text, problems):
    """Section 0's `publicly reachable` cells against the public symbols.

    See PUBLIC_REACHABILITY_PROBES. A watch on a row the contract does not
    have is itself a failure, the same way a stale-deferred watch is: a
    watch on nothing checks nothing.
    """
    header, rows = levels_table(text)
    if header is None:
        return  # check_levels has already reported the missing table
    index = symbol_index()
    column = LEVEL_NAMES.index("publicly reachable") + 2
    scored = {}
    for row in rows:
        if len(row) == len(header):
            scored[strip_ticks(row[0])] = strip_ticks(row[column]).lower()

    for item, probes in sorted(PUBLIC_REACHABILITY_PROBES.items()):
        cell = scored.get(item)
        if cell is None:
            fail(
                problems,
                f"check_parity: PUBLIC_REACHABILITY_PROBES watches {item!r}, "
                "which is not a row in section 0. Rename the watch or drop "
                "it; a watch on nothing checks nothing.",
            )
            continue
        if cell == "n/a":
            continue
        try:
            missing = [p for p in probes if not resolves(p, index)]
        except ValueError as exc:
            fail(problems, f"check_parity: {exc}")
            continue
        if not missing and cell != "yes":
            fail(
                problems,
                f"contract: section 0 row {item!r} says publicly reachable "
                f"{cell!r}, but every public symbol behind it resolves ("
                + ", ".join(probes)
                + "). A user can reach it today; say so, and re-audit the "
                "row's status and evidence while you are there.",
            )
        elif missing and cell == "yes":
            fail(
                problems,
                f"contract: section 0 row {item!r} says publicly reachable "
                "yes, but "
                + ", ".join(missing)
                + " does not resolve. Either export the name or correct the "
                "cell; a reachability claim with no public name behind it "
                "is the claim this table exists to prevent.",
            )


def mojo_export_names():
    """Names re-exported from src/mojotrees/__init__.mojo."""
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


def env_var_names():
    """`MOJOTREES_*` names a shipping source file reads.

    Section 2 of `docs/COMPATIBILITY_POLICY.md` makes these part of the
    public surface, so a capability whose only route is an environment
    variable is still publicly reachable, and that claim needs the same kind
    of evidence as an exported name. A variable named only in a comment or a
    document is not a route: the name has to appear inside a string literal
    in a file under `src/mojotrees`, `bindings`, or `python/mojotrees`,
    which is where every reader of one lives.
    """
    names = set()
    quoted = re.compile(r"[\"'](MOJOTREES_[A-Z0-9_]+)[\"']")
    for base, pattern in (
        (ROOT / "src" / "mojotrees", "*.mojo"),
        (ROOT / "bindings", "*.mojo"),
        (PY_PKG, "*.py"),
    ):
        if not base.is_dir():
            continue
        for path in sorted(base.glob(pattern)):
            try:
                text = path.read_text()
            except OSError:
                continue
            names.update(quoted.findall(text))
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
        "env": env_var_names(),
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

    for path in (PY_API, PY_BASIC, PY_PKG / "sklearn.py"):
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
    if kind == "env":
        return rest in index["env"]
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
    # The estimators and the helpers the contract names moved out of
    # __init__.py in the consolidation round (mojotrees.sklearn,
    # _environment, _ranking) and are re-exported from it; the contract is
    # about the package surface, so read the definitions where they live.
    bodies = list(tree.body)
    for extra in ("sklearn.py", "_environment.py", "_ranking.py"):
        path = PY_PKG / extra
        if path.exists():
            bodies.extend(ast.parse(path.read_text()).body)
    classes = {
        node.name: node for node in bodies if isinstance(node, ast.ClassDef)
    }
    functions = {
        node.name for node in bodies if isinstance(node, ast.FunctionDef)
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
        import mojotrees  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - depends on the build
        print(f"  (skipped live import: {type(exc).__name__}: {exc})")
        return
    finally:
        sys.path.pop(0)
    for name in REQUIRED_PY_ALL:
        if not hasattr(mojotrees, name):
            fail(problems, f"python: the built package has no {name}")
    print("  (live import checked)")


def mojo_exports(problems):
    """Names re-exported from src/mojotrees/__init__.mojo."""
    names = mojo_export_names()
    for name in REQUIRED_MOJO_EXPORTS:
        if name not in names:
            fail(
                problems,
                f"mojo: {name} is no longer exported from "
                "src/mojotrees/__init__.mojo",
            )


#: Construction sites that may leave `monotone` at its default, each with
#: the reason it is not a dropped constraint. Keyed by (file, struct); the
#: value is the reason, printed when the exemption is unused so a stale
#: entry gets removed rather than accumulating.
MONOTONE_EXEMPT = {
    ("boosting_rf.mojo", "Booster"): (
        "an empty throwaway built only to call `.response()`, which is a "
        "property of the objective and reads no constraint"
    ),
    ("tree.mojo", "TreeParams"): (
        "`TreeParams.default()`, whose documented job is to state "
        "LightGBM's defaults, of which no constraints is one"
    ),
    ("lgbm_model_io.mojo", "Booster"): (
        "a model parsed from LightGBM's own format, which this reader "
        "does not parse constraints out of; see the import row in the "
        "contract"
    ),
    ("lgbm_model_io.mojo", "MulticlassBooster"): (
        "a model parsed from LightGBM's own format, which this reader "
        "does not parse constraints out of; see the import row in the "
        "contract"
    ),
    ("catboost_reach_bindings.mojo", "TreeParams"): (
        "the multi-target entry point, and the constraint is REFUSED by name "
        "one layer up rather than dropped here: `_fit_multi_target` in "
        "sklearn.py raises on `monotone_constraints`, because "
        "`multi_target.train_multi_rmse` has no monotone pass and forwarding "
        "the bundle would move the silence one layer down instead of ending "
        "it. This gate found the drop and the gate was right; the exemption "
        "is the refusal, not a dispensation"
    ),
}


def _struct_blocks(text):
    """(name, source) for each top-level struct, in file order."""
    lines = text.splitlines()
    starts = [
        (i, m.group(1))
        for i, line in enumerate(lines)
        for m in [re.match(r"struct\s+(\w+)", line)]
        if m
    ]
    for n, (i, name) in enumerate(starts):
        end = starts[n + 1][0] if n + 1 < len(starts) else len(lines)
        yield name, "\n".join(lines[i:end])


def _monotone_carriers():
    """Structs that store a `MonotoneConstraints` behind a defaulted
    argument.

    Discovered rather than listed. A struct that starts carrying
    constraints is covered the day it is written, which is the whole
    point: the last set of dropped constraints was found by reading, and
    reading does not scale to the next one.
    """
    carriers = {}
    for path in sorted((ROOT / "src" / "mojotrees").glob("*.mojo")):
        text = path.read_text()
        for name, block in _struct_blocks(text):
            stores = re.search(
                r"var monotone:\s*MonotoneConstraints\s*$", block, re.M
            )
            defaulted = re.search(
                r"var monotone:\s*MonotoneConstraints\s*=", block
            )
            if stores and defaulted:
                carriers[name] = path.name
    return carriers


def _call_args(text, open_paren):
    """The argument text of the call whose `(` is at `open_paren`."""
    depth = 0
    for i in range(open_paren, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[open_paren + 1 : i]
    return None


def monotone_passthrough(problems):
    """Every construction of a constraint-carrying struct passes them on.

    `monotone` defaults to "no constraints" on all of these, so omitting
    the argument compiles, runs, trains, and returns a model that quietly
    reports it was fit without constraints. That failure has already
    happened once across seven trainers, and it was fixed by hand in each
    of them while the shape that produced it was left in place. Reading
    cannot be the guard here, because the mistake is an *absence* and
    there is nothing at the call site to notice.

    Anything genuinely fine to leave defaulted goes in `MONOTONE_EXEMPT`
    with its reason, so the exemption is an argument rather than silence.
    """
    carriers = _monotone_carriers()
    if not carriers:
        problems.append(
            "no MonotoneConstraints-carrying struct found; the monotone "
            "passthrough check has stopped checking anything"
        )
        return
    seen_exempt = set()
    roots = [ROOT / "src" / "mojotrees", ROOT / "bindings"]
    for base in roots:
        if not base.is_dir():
            continue
        for path in sorted(base.glob("*.mojo")):
            text = path.read_text()
            for name in carriers:
                pattern = r"(?<![\w.])" + name + r"\s*\("
                for m in re.finditer(pattern, text):
                    line_start = text.rfind("\n", 0, m.start()) + 1
                    if text[line_start:].lstrip().startswith("struct "):
                        continue
                    args = _call_args(text, m.end() - 1)
                    if args is None or "monotone" in args:
                        continue
                    key = (path.name, name)
                    if key in MONOTONE_EXEMPT:
                        seen_exempt.add(key)
                        continue
                    line = text.count("\n", 0, m.start()) + 1
                    problems.append(
                        f"{path.name}:{line} builds {name} without passing "
                        "monotone constraints, so the model it returns "
                        "reports none; pass params.tree.monotone.copy() or "
                        "add an entry to MONOTONE_EXEMPT saying why not"
                    )
    for key in sorted(set(MONOTONE_EXEMPT) - seen_exempt):
        problems.append(
            f"MONOTONE_EXEMPT has a stale entry for {key[1]} in {key[0]}; "
            "that construction no longer exists or now passes constraints, "
            "so remove the exemption"
        )


#: LightGBM's stock default for every parameter mojotrees also defaults.
#:
#: Read from **microsoft/LightGBM `include/LightGBM/config.h` at tag
#: v4.7.0**, and cross-checked against `docs/Parameters.rst` at the same
#: tag, on 2026-08-16. `config.h` is the authority where the two could
#: disagree: it is what the library compiles. `num_leaves` is the one entry
#: config.h states indirectly, as `kDefaultNumLeaves`; its value of 31 is
#: taken from `docs/Parameters.rst`.
#:
#: **mojotrees's defaults ARE LightGBM's stock defaults.** That is the
#: policy, not an aspiration, and it is what makes "mojotrees against
#: LightGBM, both at their own defaults" a comparison a reader can act on
#: rather than two libraries answering two different questions. A default
#: that drifts off this table is a silent model change on every fit that
#: did not set the parameter, and it is invisible in a diff of results.
#:
#: This table used to hold three binning constants. It was widened on
#: 2026-08-16, when `lambda_l2` was found sitting at 1.0 against LightGBM's
#: 0.0 -- the third comparator-configuration defect this repository found
#: in a week -- and the lesson taken was that checking the defaults anyone
#: had thought to check is not a gate. Anything genuinely allowed to
#: diverge goes in `STOCK_DIVERGENCES` with its reason and its exit
#: condition, so a divergence is an argument on the page rather than an
#: absence from a list.
LIGHTGBM_STOCK = {
    "num_leaves": 31,
    "max_depth": -1,
    "learning_rate": 0.1,
    "num_iterations": 100,
    "min_data_in_leaf": 20,
    "min_sum_hessian_in_leaf": 1e-3,
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
    "min_gain_to_split": 0.0,
    "max_delta_step": 0.0,
    "path_smooth": 0.0,
    "monotone_penalty": 0.0,
    "extra_trees": False,
    "feature_fraction": 1.0,
    "feature_fraction_bynode": 1.0,
    "pos_bagging_fraction": 1.0,
    "neg_bagging_fraction": 1.0,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    "max_bin": 255,
    "min_data_in_bin": 3,
    "bin_construct_sample_cnt": 200_000,
    "data_random_seed": 1,
    "feature_fraction_seed": 2,
    "bagging_seed": 3,
    "drop_seed": 4,
    "objective_seed": 5,
    "extra_seed": 6,
    "top_k": 20,
    "max_cat_to_onehot": 4,
    "max_cat_threshold": 32,
    "cat_smooth": 10.0,
    "cat_l2": 10.0,
    "min_data_per_group": 100,
    "sigmoid": 1.0,
    "lambdarank_truncation_level": 30,
    "lambdarank_position_bias_regularization": 0.0,
    "fair_c": 1.0,
    "tweedie_variance_power": 1.5,
    "drop_rate": 0.1,
    "max_drop": 50,
    "skip_drop": 0.5,
    "uniform_drop": False,
    "num_grad_quant_bins": 4,
    "use_quantized_grad": False,
    "stochastic_rounding": True,
    "quant_train_renew_leaf": False,
    "refit_decay_rate": 0.9,
}

#: Defaults that are deliberately NOT LightGBM's, asserted as divergent so
#: that "fixing" one without retiring its reason fails here too. A silent
#: convergence is as much a surprise as a silent divergence: the value in
#: the second column is what the code says today and what the
#: documentation is written against.
STOCK_DIVERGENCES = {
    "enable_bundle": (
        True,
        False,
        "exclusive feature bundling changes the feature space before "
        "binning, and mojotrees's EFB is not applied by every trainer yet "
        "(the ranking trainer refuses an active bundling switch by name). "
        "Off by default until it is; see the enable_bundle row of "
        "docs/LIGHTGBM_PARITY.md",
    ),
}

#: `(module, comptime constant, the LightGBM parameter it is)`, checked
#: against `LIGHTGBM_STOCK` above. A constant named here that is not a
#: plain literal, or a module that has moved, fails rather than being
#: skipped.
STOCK_COMPTIME_DEFAULTS = (
    ("binning.mojo", "DEFAULT_MIN_DATA_IN_BIN", "min_data_in_bin"),
    (
        "binning.mojo",
        "DEFAULT_BIN_CONSTRUCT_SAMPLE_CNT",
        "bin_construct_sample_cnt",
    ),
    ("binning.mojo", "DEFAULT_DATA_RANDOM_SEED", "data_random_seed"),
    ("binning.mojo", "DEFAULT_MIN_DATA_IN_LEAF", "min_data_in_leaf"),
    ("sampling.mojo", "DEFAULT_FEATURE_FRACTION_SEED", "feature_fraction_seed"),
    (
        "sampling.mojo",
        "DEFAULT_POS_BAGGING_FRACTION",
        "pos_bagging_fraction",
    ),
    (
        "sampling.mojo",
        "DEFAULT_NEG_BAGGING_FRACTION",
        "neg_bagging_fraction",
    ),
    ("bagging.mojo", "DEFAULT_BAGGING_SEED", "bagging_seed"),
    ("boosting_dart.mojo", "DEFAULT_DROP_RATE", "drop_rate"),
    ("boosting_dart.mojo", "DEFAULT_MAX_DROP", "max_drop"),
    ("boosting_dart.mojo", "DEFAULT_SKIP_DROP", "skip_drop"),
    ("boosting_dart.mojo", "DEFAULT_DROP_SEED", "drop_seed"),
    ("boosting_dart.mojo", "DEFAULT_UNIFORM_DROP", "uniform_drop"),
    (
        "ranking.mojo",
        "DEFAULT_TRUNCATION_LEVEL",
        "lambdarank_truncation_level",
    ),
    ("ranking.mojo", "DEFAULT_SIGMOID", "sigmoid"),
    (
        "ranking_advanced.mojo",
        "DEFAULT_POSITION_BIAS_REGULARIZATION",
        "lambdarank_position_bias_regularization",
    ),
    ("ranking_advanced.mojo", "DEFAULT_PAIR_SAMPLING_SEED", "objective_seed"),
    ("distributed_strategies.mojo", "DEFAULT_TOP_K", "top_k"),
    ("objective_registry.mojo", "DEFAULT_FAIR_C", "fair_c"),
    (
        "objective_registry.mojo",
        "DEFAULT_TWEEDIE_VARIANCE_POWER",
        "tweedie_variance_power",
    ),
    ("tree_parameters_extra.mojo", "DEFAULT_EXTRA_SEED", "extra_seed"),
    (
        "tree_parameters_extra.mojo",
        "DEFAULT_NUM_GRAD_QUANT_BINS",
        "num_grad_quant_bins",
    ),
    (
        "quantized_gradient.mojo",
        "DEFAULT_NUM_GRAD_QUANT_BINS",
        "num_grad_quant_bins",
    ),
    ("efb.mojo", "DEFAULT_ENABLE_BUNDLE", "enable_bundle"),
)

#: `(module, struct, "how to read its defaults", {mojotrees field: LightGBM
#: parameter})`. Three readers, because the three places a Mojo default can
#: live are three different pieces of syntax:
#:
#:   "static"  a `@staticmethod def default()` whose body constructs the
#:             struct; the arguments are zipped against the constructor's
#:             parameter order, so an inserted field is caught rather than
#:             shifting every check one place along
#:   "signature"  a defaulted argument on `__init__`
#:   "assign"  `self.x = <literal>` in a no-argument `__init__`
STOCK_STRUCT_DEFAULTS = (
    (
        "tree.mojo",
        "TreeParams",
        "static",
        {
            "num_leaves": "num_leaves",
            "min_data_in_leaf": "min_data_in_leaf",
            "lambda_reg": "lambda_l2",
            "min_child_hess": "min_sum_hessian_in_leaf",
            "lambda_l1": "lambda_l1",
        },
    ),
    (
        "tree.mojo",
        "TreeParams",
        "signature",
        {
            "max_depth": "max_depth",
            "feature_fraction": "feature_fraction",
            "feature_fraction_bynode": "feature_fraction_bynode",
        },
    ),
    (
        "boosting.mojo",
        "BoosterParams",
        "static",
        {
            "n_estimators": "num_iterations",
            "learning_rate": "learning_rate",
        },
    ),
    (
        "categorical.mojo",
        "CategoricalParams",
        "static",
        {
            "max_cat_to_onehot": "max_cat_to_onehot",
            "max_cat_threshold": "max_cat_threshold",
            "cat_smooth": "cat_smooth",
            "cat_l2": "cat_l2",
            "min_data_per_group": "min_data_per_group",
        },
    ),
    (
        "model_editing.mojo",
        "RefitParams",
        "static",
        {"decay_rate": "refit_decay_rate"},
    ),
    (
        "tree_parameters_extra.mojo",
        "ExtraTreeParams",
        "assign",
        {
            "min_gain_to_split": "min_gain_to_split",
            "max_delta_step": "max_delta_step",
            "path_smooth": "path_smooth",
            "monotone_penalty": "monotone_penalty",
            "extra_trees": "extra_trees",
            "extra_seed": "extra_seed",
            "use_quantized_grad": "use_quantized_grad",
            "num_grad_quant_bins": "num_grad_quant_bins",
            "quant_train_renew_leaf": "quant_train_renew_leaf",
            "stochastic_rounding": "stochastic_rounding",
        },
    ),
)

#: The same defaults again on the Python surface, `{python name: LightGBM
#: parameter}` per construct in `python/mojotrees/sklearn.py`.
#:
#: This half exists because the Mojo default and the Python default are two
#: separate literals with nothing joining them, and **the Python one is what
#: `bench/real_data` actually fits**: its mojotrees arm goes through
#: `mojotrees.train`, which builds a `MojoTrees*` estimator, which resolves
#: every unset hyperparameter from this signature and never reads
#: `TreeParams.default()`. Changing the Mojo default alone would leave the
#: headline comparison running our Python default against LightGBM's C++
#: one, which is a worse state than the divergence being fixed. So both are
#: checked, against one table.
STOCK_PYTHON_CONSTANTS = {
    "_LAMBDA_L1": "lambda_l1",
    "_LAMBDA_L2": "lambda_l2",
    # `learning_rate` moved here from `STOCK_PYTHON_SIGNATURE` on 2026-08-16,
    # and the fact it guards is unchanged: LightGBM's stock rate is still
    # read out of `python/mojotrees/sklearn.py` and still compared against
    # the same table. What moved is where the literal sits. The constructor
    # now defaults `learning_rate` to `None`, because CatBoost's automatic
    # learning rate fires only on an UNSET rate (`options_helper.cpp:277`)
    # and a signature default of 0.1 cannot tell an unset rate from a rate a
    # caller typed as 0.1 -- which is exactly the distinction a CatBoost-mode
    # benchmark arm turns on. `_LEARNING_RATE` is the value an unset rate
    # resolves to at fit time, so it is the Python default this gate exists
    # to check. `_LAMBDA_L2` was already read this way for the same shape of
    # reason.
    "_LEARNING_RATE": "learning_rate",
}

#: The THIRD copy of the stock leaf-regularization defaults, in
#: `Booster.refit`'s local `_leaf(name, alias, default)` helper in
#: `python/mojotrees/basic.py`.
#:
#: It is a third copy because `sklearn.py` imports `basic.py` and not the
#: other way round, so `basic.py` cannot reach `_LAMBDA_L2` without an import
#: cycle. Deduplicating means moving the constants down into `basic.py` and
#: having `sklearn.py` import them, which is the right fix and is not this
#: gate's job.
#:
#: Until then the literal is checked where it sits. This is exactly the
#: failure mode item 5 below names: `lambda_l2` sat at 1.0 against LightGBM's
#: 0.0 in three places, was written down somewhere, and was checked nowhere.
#: Two of the three are now read by this gate and this table is the third.
#: A default that is correct today and unguarded is one edit from being
#: wrong again with nothing to catch it.
STOCK_BASIC_LEAF_DEFAULTS = {
    "lambda_l1": "lambda_l1",
    "lambda_l2": "lambda_l2",
}

STOCK_PYTHON_SIGNATURE = {
    "num_leaves": "num_leaves",
    "max_depth": "max_depth",
    # `learning_rate` is checked as `_LEARNING_RATE` in
    # STOCK_PYTHON_CONSTANTS above, not here: its signature default is now
    # `None` so that "unset" survives to CatBoost's gate. Removing it from
    # this table without adding it there would have unchecked LightGBM's
    # stock rate on the surface bench/real_data fits.
    "n_estimators": "num_iterations",
    "min_data_in_leaf": "min_data_in_leaf",
    "min_child_hess": "min_sum_hessian_in_leaf",
    "max_bin": "max_bin",
    "bagging_fraction": "bagging_fraction",
    "bagging_freq": "bagging_freq",
    "feature_fraction": "feature_fraction",
    "feature_fraction_bynode": "feature_fraction_bynode",
    "min_gain_to_split": "min_gain_to_split",
    "max_delta_step": "max_delta_step",
    "path_smooth": "path_smooth",
}


def _mojo_literal(token, constants):
    """A Mojo literal as a Python value, or `None` if this gate cannot read
    it. Names resolve through `constants` first, so a default written as a
    named constant is checked as its value."""
    token = token.strip()
    if not token:
        return None
    if token in constants:
        return constants[token]
    if token == "True":
        return True
    if token == "False":
        return False
    plain = token.replace("_", "")
    if re.fullmatch(r"[+-]?[0-9]+", plain):
        return int(plain)
    float_re = (
        r"[+-]?(?:[0-9]*\.[0-9]*(?:[eE][+-]?[0-9]+)?"
        r"|[0-9]+[eE][+-]?[0-9]+)"
    )
    if re.fullmatch(float_re, plain):
        return float(plain)
    return None


def _mojo_constants(problems):
    """Every `comptime NAME = <literal>` in the package, as Python values.

    Two modules defining the same name with *different* values is reported
    rather than resolved: the table below names constants by module, and a
    reader who sees one name meaning two numbers is looking at a bug.
    """
    values = {}
    for path in sorted((ROOT / "src" / "mojotrees").glob("*.mojo")):
        for m in re.finditer(
            r"^comptime\s+(\w+)\s*=\s*([^\n#]+?)\s*$", path.read_text(), re.M
        ):
            value = _mojo_literal(m.group(2), {})
            if value is None:
                continue
            name = m.group(1)
            if name in values and values[name] != value:
                fail(
                    problems,
                    f"stock defaults: the constant {name} is defined with "
                    f"two different values ({values[name]!r} and {value!r}); "
                    "this gate cannot say which one a default means",
                )
            values[name] = value
    return values


def _top_level_split(args):
    """Split a call's argument text on commas that are not inside brackets."""
    out, depth, start = [], 0, 0
    for i, ch in enumerate(args):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            out.append(args[start:i])
            start = i + 1
    out.append(args[start:])
    return [a for a in out if a.strip()]


def _ctor_param_names(block):
    """The constructor's parameter names, in order.

    An explicit `def __init__` wins; a `@fieldwise_init` struct has no
    `__init__` to read, and its constructor is its field order.
    """
    m = re.search(r"def __init__\s*\(", block)
    if m:
        args = _call_args(block, m.end() - 1)
        if args is None:
            return []
        names = []
        for arg in _top_level_split(args):
            arg = arg.strip()
            if arg in ("self", "out self", "mut self"):
                continue
            arg = re.sub(r"^(?:var|owned|ref|mut|read)\s+", "", arg)
            name = re.match(r"(\w+)", arg)
            if name:
                names.append(name.group(1))
        return names
    return re.findall(r"^\s+var\s+(\w+)\s*:", block, re.M)


def _signature_defaults(block, constants):
    """`{parameter: value}` for every `__init__` argument with a literal
    default this gate can read."""
    m = re.search(r"def __init__\s*\(", block)
    if not m:
        return {}
    args = _call_args(block, m.end() - 1)
    if args is None:
        return {}
    out = {}
    for arg in _top_level_split(args):
        parts = arg.split("=", 1)
        if len(parts) != 2:
            continue
        name = re.match(r"\s*(?:var|owned|ref|mut|read)?\s*(\w+)", parts[0])
        if not name:
            continue
        value = _mojo_literal(parts[1], constants)
        if value is not None:
            out[name.group(1)] = value
    return out


def _static_default_values(block, struct, constants):
    """`{field: value}` from a `@staticmethod def default()` that
    constructs the struct, zipped against the constructor's parameter
    order so that an inserted field fails instead of shifting the check."""
    offset = block.find("def default")
    if offset < 0:
        return None
    m = re.search(r"(?<![\w.])" + struct + r"\s*\(", block[offset:])
    if not m:
        return None
    args = _call_args(block, offset + m.end() - 1)
    if args is None:
        return None
    names = _ctor_param_names(block)
    out = {}
    for i, arg in enumerate(_top_level_split(args)):
        if "=" in arg and not re.match(r"\s*[-+0-9.]", arg):
            key, _, raw = arg.partition("=")
            key = key.strip()
        elif i < len(names):
            key, raw = names[i], arg
        else:
            continue
        value = _mojo_literal(raw, constants)
        if value is not None:
            out[key] = value
    return out


def _assigned_defaults(block, constants):
    """`{field: value}` from `self.x = <literal>` in a no-argument
    `__init__`."""
    out = {}
    for m in re.finditer(r"^\s+self\.(\w+)\s*=\s*([^\n#]+?)\s*$", block, re.M):
        value = _mojo_literal(m.group(2), constants)
        if value is not None:
            out[m.group(1)] = value
    return out


def _same(got, want):
    """Exact equality, with `1e-3` written as a float and `20` as an int
    counting as equal to each other's spelling but not to a different
    number. `True == 1` is refused, because a flag and a count are not the
    same default."""
    if isinstance(got, bool) != isinstance(want, bool):
        return False
    if isinstance(got, bool):
        return got is want
    return float(got) == float(want)


#: Filled in by `_check_stock`, read by `stock_defaults` to prove the table
#: above is a checklist rather than a wish list.
_STOCK_PROBED: set[str] = set()


def _check_stock(problems, where, mojo_name, lgbm_name, got):
    _STOCK_PROBED.add(lgbm_name)
    want = LIGHTGBM_STOCK[lgbm_name]
    if not _same(got, want):
        fail(
            problems,
            f"stock defaults: {where} defaults {mojo_name} to {got!r}, and "
            f"LightGBM's {lgbm_name} is {want!r}. mojotrees's defaults are "
            "LightGBM's stock defaults, which is what makes a both-at-their-"
            "own-defaults comparison mean anything; change this back, or "
            "move the parameter into STOCK_DIVERGENCES here with its reason "
            "and its exit condition and argue it in "
            "docs/LIGHTGBM_PARITY.md, in one commit",
        )


def stock_defaults(problems):
    """**Every** default this library states is LightGBM's stock value, on
    every surface that states one, and `fit_bins` still uses the constants.

    Four readings, because a default can be wrong in four independent
    places and being right in three of them changes nothing:

    1. the `comptime DEFAULT_*` constants across the package
       (`STOCK_COMPTIME_DEFAULTS`)
    2. the hyperparameter structs -- `TreeParams`, `BoosterParams`,
       `CategoricalParams`, `ExtraTreeParams` -- read through their
       `default()` bodies, their `__init__` signatures, and their `__init__`
       assignments (`STOCK_STRUCT_DEFAULTS`)
    3. the Python estimator signature, which is what `bench/real_data`
       actually fits and which shares no literal with the Mojo side
       (`STOCK_PYTHON_*`)
    4. `fit_bins`, checked to default to the *constants* rather than to a
       literal, so a signature that drifted back to `= 1` fails here even
       with the constant left correct

    This gate checked three binning constants until 2026-08-16. It was
    widened the day `lambda_l2` was found defaulting to 1.0 against
    LightGBM's 0.0 -- a divergence that had been *documented* for months,
    in the README and in two rows of the contract, and was therefore not
    hidden at all, merely never gated. Documentation is not a gate. That is
    the whole argument for the size of the table above.

    `feature_pre_filter` is checked separately, by `feature_pre_filter_gate`.
    It is not a binning *default* on this side -- `fit_bins` defaults it to
    `False` and has to, because `False` is the fit that preceded the option.
    """
    constants = _mojo_constants(problems)
    src = ROOT / "src" / "mojotrees"
    _STOCK_PROBED.clear()
    divergences_probed = set()

    # 1. the package's DEFAULT_* constants.
    for module, name, lgbm in STOCK_COMPTIME_DEFAULTS:
        path = src / module
        if not path.is_file():
            fail(
                problems,
                f"stock defaults: src/mojotrees/{module} is missing, so "
                f"{name} cannot be checked against LightGBM's {lgbm}",
            )
            continue
        m = re.search(
            r"^comptime\s+" + name + r"\s*=\s*([^\n#]+?)\s*$",
            path.read_text(),
            re.M,
        )
        got = _mojo_literal(m.group(1), constants) if m else None
        if got is None:
            fail(
                problems,
                f"stock defaults: {name} is not defined in {module} as a "
                "literal comptime, so this gate cannot read it",
            )
            continue
        if lgbm in STOCK_DIVERGENCES:
            divergences_probed.add(lgbm)
            stock, declared, why = STOCK_DIVERGENCES[lgbm]
            if not _same(got, declared):
                fail(
                    problems,
                    f"stock defaults: {name} in {module} is {got!r}, and "
                    f"STOCK_DIVERGENCES declares {lgbm} as {declared!r} "
                    f"against LightGBM's {stock!r} because {why}. Either "
                    "the declaration is stale or the default moved without "
                    "it; settle which, in one commit",
                )
            continue
        _check_stock(problems, module, name, lgbm, got)

    # 2. the hyperparameter structs.
    for module, struct, how, fields in STOCK_STRUCT_DEFAULTS:
        path = src / module
        if not path.is_file():
            fail(
                problems,
                f"stock defaults: src/mojotrees/{module} is missing",
            )
            continue
        block = None
        for name, candidate in _struct_blocks(path.read_text()):
            if name == struct:
                block = candidate
                break
        if block is None:
            fail(
                problems,
                f"stock defaults: struct {struct} is not in {module} any "
                "more, so its defaults are unchecked",
            )
            continue
        if how == "static":
            values = _static_default_values(block, struct, constants)
            where = f"{struct}.default() in {module}"
        elif how == "signature":
            values = _signature_defaults(block, constants)
            where = f"{struct}.__init__ in {module}"
        else:
            values = _assigned_defaults(block, constants)
            where = f"{struct}.__init__ in {module}"
        if values is None:
            fail(
                problems,
                f"stock defaults: {where} is not in a shape this gate can "
                f"read, so {len(fields)} defaults are unchecked",
            )
            continue
        for field, lgbm in fields.items():
            if field not in values:
                fail(
                    problems,
                    f"stock defaults: {where} no longer sets {field} to a "
                    f"literal this gate can read, so LightGBM's {lgbm} is "
                    "unchecked here",
                )
                continue
            _check_stock(problems, where, field, lgbm, values[field])

    # 3. the Python estimator, which is the surface bench/real_data fits.
    sklearn = ROOT / "python" / "mojotrees" / "sklearn.py"
    if not sklearn.is_file():
        fail(
            problems,
            "stock defaults: python/mojotrees/sklearn.py is missing",
        )
    else:
        text = sklearn.read_text()
        py_constants = {}
        for name, lgbm in STOCK_PYTHON_CONSTANTS.items():
            m = re.search(r"^" + name + r"\s*=\s*([^\n#]+?)\s*$", text, re.M)
            got = _mojo_literal(m.group(1), {}) if m else None
            if got is None:
                fail(
                    problems,
                    f"stock defaults: {name} is not a literal at the top of "
                    "python/mojotrees/sklearn.py, so this gate cannot read "
                    f"the Python default for {lgbm}",
                )
                continue
            py_constants[name] = got
            _check_stock(
                problems, "python/mojotrees/sklearn.py", name, lgbm, got
            )
        base = re.search(r"^class _Base\b", text, re.M)
        init = (
            re.search(r"^    def __init__\s*\(", text[base.start() :], re.M)
            if base
            else None
        )
        if not init:
            fail(
                problems,
                "stock defaults: _Base.__init__ is not where this expects it "
                "in python/mojotrees/sklearn.py, so the estimator signature "
                "-- the defaults bench/real_data actually fits -- is "
                "unchecked",
            )
        else:
            args = _call_args(text, base.start() + init.end() - 1)
            found = {}
            for arg in _top_level_split(args or ""):
                key, _, raw = arg.partition("=")
                if not raw:
                    continue
                value = _mojo_literal(raw, py_constants)
                if value is not None:
                    found[key.strip()] = value
            for field, lgbm in STOCK_PYTHON_SIGNATURE.items():
                if field not in found:
                    fail(
                        problems,
                        f"stock defaults: _Base.__init__ no longer defaults "
                        f"{field} to a literal, so LightGBM's {lgbm} is "
                        "unchecked on the Python surface, which is the one "
                        "bench/real_data fits",
                    )
                    continue
                _check_stock(
                    problems,
                    "_Base.__init__ in python/mojotrees/sklearn.py",
                    field,
                    lgbm,
                    found[field],
                )

    # 3b. the functional API's own copy, in Booster.refit. Separate from 3
    # because it is a separate literal in a separate file that no import
    # binds to the estimator's: a fix applied to sklearn.py alone leaves a
    # refit fitting a different regularizer than the fit it refits.
    basic = ROOT / "python" / "mojotrees" / "basic.py"
    if not basic.is_file():
        fail(problems, "stock defaults: python/mojotrees/basic.py is missing")
    else:
        text = basic.read_text()
        for field, lgbm in STOCK_BASIC_LEAF_DEFAULTS.items():
            m = re.search(
                r"\b"
                + field
                + r"\s*=\s*_leaf\(\s*\"[^\"]*\"\s*,\s*\"[^\"]*\"\s*,"
                r"\s*([^)\n]+?)\s*\)",
                text,
            )
            got = _mojo_literal(m.group(1), {}) if m else None
            if got is None:
                fail(
                    problems,
                    f"stock defaults: {field} is no longer set from a literal "
                    "_leaf(...) default in python/mojotrees/basic.py, so "
                    f"LightGBM's {lgbm} is unchecked on the refit path. This "
                    "is the third copy of that default and the one with no "
                    "import binding it to the other two",
                )
                continue
            _check_stock(
                problems,
                "Booster.refit in python/mojotrees/basic.py",
                field,
                lgbm,
                got,
            )

    # 4. fit_bins still reaches the constants.
    binning = ROOT / "src" / "mojotrees" / "binning.mojo"
    if not binning.is_file():
        fail(problems, "stock defaults: src/mojotrees/binning.mojo is missing")
        return
    text = binning.read_text()
    sig = re.search(r"^def fit_bins\[", text, re.M)
    if not sig:
        fail(problems, "stock defaults: fit_bins is not where this expects it")
    else:
        head = text[sig.start() : sig.start() + 1200]
        for arg, name in (
            ("min_data_in_bin", "DEFAULT_MIN_DATA_IN_BIN"),
            ("bin_construct_sample_cnt", "DEFAULT_BIN_CONSTRUCT_SAMPLE_CNT"),
            ("data_random_seed", "DEFAULT_DATA_RANDOM_SEED"),
        ):
            if not re.search(
                r"\b" + arg + r"\s*:\s*Int\s*=\s*" + name + r"\b", head
            ):
                fail(
                    problems,
                    f"stock defaults: fit_bins does not default {arg} to "
                    f"{name}, so the constant above is not what a caller who "
                    "says nothing gets",
                )

    # 5. the tables above are checklists, not wish lists. An entry nothing
    # probes reads as coverage and provides none, which is the failure the
    # `min_data_in_bin` pin and the `lambda_l2` default both had in common:
    # written down somewhere, checked nowhere.
    unprobed = sorted(set(LIGHTGBM_STOCK) - _STOCK_PROBED)
    if unprobed:
        fail(
            problems,
            "stock defaults: nothing in this repository was probed for "
            + ", ".join(unprobed)
            + ". Either give each a row in STOCK_COMPTIME_DEFAULTS, "
            "STOCK_STRUCT_DEFAULTS or STOCK_PYTHON_SIGNATURE, or drop it "
            "from LIGHTGBM_STOCK; an entry nothing reads looks like "
            "coverage and is not",
        )
    stale = sorted(set(STOCK_DIVERGENCES) - divergences_probed)
    if stale:
        fail(
            problems,
            "stock defaults: STOCK_DIVERGENCES declares "
            + ", ".join(stale)
            + " as a deliberate divergence, and no default was read for it, "
            "so the divergence is neither enforced nor retired",
        )


# The growers that draw a tree's feature set. Each calls
# `sampling.select_tree_features`, and each has to hand it the pool for a
# prefiltered fit to reach `feature_fraction` at all.
FEATURE_POOL_GROWERS = (
    "tree.mojo",
    "tree_sparse.mojo",
    "train_gpu.mojo",
    "train_gpu_sparse.mojo",
)


def feature_pre_filter_gate(problems):
    """`feature_pre_filter` is applied, and stays applied.

    This gate used to assert the opposite: that `check_feature_pre_filter`
    still refused `true`, because accepting a flag while keeping every feature
    would leave mojotrees and LightGBM fitting **different feature spaces**,
    which is the same error EFB is held back for. The filter is implemented
    now, so the thing worth guarding has moved. What follows is the chain that
    makes `true` honest, asserted link by link, so that removing any one link
    fails here rather than quietly turning the flag back into a no-op:

    1. `binning.filter_count` -- LightGBM's `filter_cnt`, `min_data_in_leaf`
       scaled to the bin-construction sample (`src/io/dataset_loader.cpp`).
    2. `binning.need_filter` -- LightGBM's `NeedFilter` (`src/io/bin.cpp`).
    3. `fit_bins` takes `feature_pre_filter` and defaults it to `False`,
       because `False` has to remain the fit that preceded the option.
    4. `BinMapper` and `BinnedMatrix` both carry `usable`, LightGBM's
       `used_features`, and `transform` propagates it -- a grower is handed a
       matrix and nothing else.
    5. `sampling.select_tree_features` takes the pool and sizes the draw by
       `len(pool)`, which is LightGBM's
       `GetCnt(valid_feature_indices_.size(), fraction)`.

    The last link, the growers passing the pool, is not asserted as a
    requirement because it is not landed yet; it is asserted as the *condition*
    for `check_feature_pre_filter` to stop refusing `true`. A parameter string
    additionally cannot carry the flag at all while `params.TrainConfig` has no
    field for it, so that is required too. Accepting the flag without both is
    a setting read and ignored.
    """
    binning = ROOT / "src" / "mojotrees" / "binning.mojo"
    sampling = ROOT / "src" / "mojotrees" / "sampling.mojo"
    extra = ROOT / "src" / "mojotrees" / "tree_parameters_extra.mojo"
    for path in (binning, sampling, extra):
        if not path.is_file():
            fail(problems, f"feature_pre_filter: {path.name} is missing")
            return
    text = binning.read_text()
    sampler = sampling.read_text()

    for pattern, what in (
        (r"^def filter_count\(", "binning.filter_count (LightGBM's filter_cnt)"),
        (r"^def need_filter\(", "binning.need_filter (LightGBM's NeedFilter)"),
        (r"^    var usable: List\[Int\]", "a `usable` field in binning.mojo"),
        (r"def usable_features\(", "BinnedMatrix.usable_features"),
    ):
        if not re.search(pattern, text, re.M):
            fail(
                problems,
                f"feature_pre_filter: {what} is gone from binning.mojo. The "
                "filter is what lets the flag be set without the two "
                "libraries fitting different feature spaces; if it is being "
                "removed, restore the refusal in check_feature_pre_filter in "
                "the same commit",
            )
    # Two of them, one per struct, so losing either is caught.
    if len(re.findall(r"^    var usable: List\[Int\]", text, re.M)) < 2:
        fail(
            problems,
            "feature_pre_filter: only one of BinMapper and BinnedMatrix "
            "carries `usable`. A grower is handed the matrix and nothing "
            "else, so the pool has to ride on both",
        )
    sig = re.search(r"^def fit_bins\[", text, re.M)
    if not sig:
        fail(problems, "feature_pre_filter: fit_bins is not where this expects it")
    elif not re.search(
        r"\bfeature_pre_filter\s*:\s*Bool\s*=\s*False\b",
        text[sig.start() : sig.start() + 1600],
    ):
        fail(
            problems,
            "feature_pre_filter: fit_bins does not take "
            "`feature_pre_filter: Bool = False`. Off has to stay the fit that "
            "preceded the option, and on has to be reachable",
        )
    if not re.search(
        r"^def select_tree_features\((?:.|\n)*?usable: List\[Int\] = \[\]",
        sampler,
        re.M,
    ):
        fail(
            problems,
            "feature_pre_filter: sampling.select_tree_features no longer "
            "takes the usable pool, so a prefiltered fit cannot narrow the "
            "set feature_fraction draws from -- which is the half of the "
            "filter that changes the trees rather than only their cost",
        )
    if not re.search(r"selection_count\(len\(pool\)", sampler):
        fail(
            problems,
            "feature_pre_filter: select_tree_features no longer sizes the "
            "draw by the surviving count. LightGBM's ColSampler uses "
            "GetCnt(valid_feature_indices_.size(), fraction), so a filtered "
            "fit draws a fraction of the survivors, not of everything",
        )

    checker = extra.read_text()
    body = re.search(
        r"^def check_feature_pre_filter\(.*?(?=^def |^struct |\Z)",
        checker,
        re.M | re.S,
    )
    if not body:
        fail(
            problems,
            "feature_pre_filter: check_feature_pre_filter is gone. A "
            "parameter string still cannot carry the flag into binning, so "
            "the name still needs its own refusal rather than the "
            "unknown-parameter path",
        )
        return
    if "raise Error(" in body.group(0):
        return

    # The refusal was removed. Everything it was standing in for has to be
    # true now.
    unwired = []
    for name in FEATURE_POOL_GROWERS:
        path = ROOT / "src" / "mojotrees" / name
        if not path.is_file():
            continue
        grower = path.read_text()
        if "select_tree_features(" not in grower:
            continue
        if "usable_features()" not in grower:
            unwired.append(name)
    if unwired:
        fail(
            problems,
            "feature_pre_filter: check_feature_pre_filter no longer refuses "
            "feature_pre_filter=true, but "
            + ", ".join(unwired)
            + " still calls select_tree_features without the pool, so the "
            "filter is computed and ignored there and those growers fit a "
            "different feature space from LightGBM's",
        )
    params = ROOT / "src" / "mojotrees" / "params.mojo"
    if params.is_file() and not re.search(
        r"\.feature_pre_filter\s*=\s*_parse_bool", params.read_text()
    ):
        fail(
            problems,
            "feature_pre_filter: check_feature_pre_filter accepts true but "
            "params.mojo never stores the parsed value on TrainConfig, so a "
            "parameter string that set it would be read and ignored",
        )


def unwired_tests(text, problems):
    """Cited Mojo suites that no pixi task runs.

    "Wired" used to mean "named in pixi.toml", because `pixi run test` was
    sixty `mojo run` commands chained with `&&` and a suite was in the run
    only if someone had typed its path. It now means "matches the glob
    `tools/run_tests.sh` discovers", which is `tests/test_*.mojo`. That is a
    weaker check by design: under a chain a suite could exist, pass, and be
    named nowhere, which is what `tests/test_gpu_split_policy.mojo` did, and
    a glob cannot leave one out. What is still worth catching is a contract
    citing a suite that is not there at all, or sitting somewhere the runner
    does not look.
    """
    cited = {
        path
        for path in set(PATH_RE.findall(text))
        if path.startswith("tests/") and path.endswith(".mojo")
    }
    wired = {
        path
        for path in cited
        if fnmatch.fnmatch(path, "tests/test_*.mojo")
        and (ROOT / path).exists()
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
    check_reachability_cells(text, problems)
    python_api(problems)
    python_runtime(problems)
    mojo_exports(problems)
    stale_deferred(text, problems)
    unwired_tests(text, problems)
    monotone_passthrough(problems)
    stock_defaults(problems)
    feature_pre_filter_gate(problems)

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
