"""The multi-column label contract, at the Python boundary.

`docs/design/CATBOOST_CATALOG.md` A26/A27/A28 states the finding this module
answers: `boosting.train`, `objective.GradHessFn`, `Booster`, and every
Python and C entry point in this repository take the label as one number per
row, and `MultiRMSE` cannot be spelled in that. `target_matrix.TargetMatrix`
was written as the wider contract and nothing reached it, so `multi_target`
and `survival` were both blocked on the same missing edge.

This module is that edge on the Python side. It is deliberately small and it
is deliberately not in `sklearn.py`: `MojoTreesRegressor.fit` gets one test
and one call, so a lane merging this against another lane's edits to that
file has two lines to reconcile rather than two hundred.

**What the multi-output trainer honors, and what it refuses.** It honors the
ensemble shape (`n_estimators`, `learning_rate`), the tree shape
(`num_leaves`, `max_depth`, `min_data_in_leaf`, `lambda_l1`, `lambda_l2`,
`min_child_hess`), `max_bin`, `use_missing`, and `sample_weight`. It refuses
everything else BY NAME rather than accepting it: bagging, GOSS, EFB
bundling, ordered boosting, dart, rf, linear trees, categorical features,
CEGB, monotone and interaction constraints, eval sets and early stopping,
callbacks, custom objectives and metrics, and the GPU. None of those reach
`multi_target.train_multi_rmse`, and a fit that accepted them would be
reporting a run that did not happen.

**The honest statement about the model, restated here because a Python user
will not read the Mojo docstring.** This grows one tree per target per round.
CatBoost's `MultiRMSE` grows ONE tree per round with a vector leaf value and
picks its structure by the summed-over-targets split score. Because the
`MultiRMSE` derivative has no cross-target term, the shared structure is the
entire modeling content of the objective, so what this fits is bit-identical
to `T` independent squared-error boosters. It is a real multi-output
regression API and it is not CatBoost's `MultiRMSE`.
"""

from __future__ import annotations

from . import _arrays

#: Every estimator attribute a multi-output fit cannot honor, and the
#: sentence each one is refused with. The value each is compared against is
#: the estimator's own default, so an untouched estimator passes every test
#: and only a user who asked for something gets an error.
_UNHONORED = (
    # (attribute, the estimator's OWN default, the mechanism it names). The
    # defaults are copied from `_Base.__init__` and must stay copied from it:
    # a default written from memory here would refuse an untouched estimator,
    # which is the worst possible failure for a refusal table.
    ("subsample", None, "row bagging"),
    ("bagging_fraction", 1.0, "row bagging"),
    ("subsample_freq", None, "row bagging"),
    ("bagging_freq", 0, "row bagging"),
    # `boosting` covers GOSS, dart and rf in one test, so `top_rate` and
    # `other_rate` are deliberately NOT here: they are read only when
    # `boosting='goss'`, and refusing their values on a gbdt fit would refuse
    # a setting that does nothing on the single-output path either.
    ("boosting", "gbdt", "GOSS, dart or rf"),
    ("boosting_type", None, "ordered boosting, dart or rf"),
    ("linear_tree", False, "linear leaves"),
    ("enable_bundle", False, "exclusive feature bundling"),
    ("cat_features", None, "categorical features"),
    ("categorical_features", None, "categorical features"),
    ("monotone_constraints", None, "monotonic constraints"),
    ("interaction_constraints", None, "interaction constraints"),
    ("forced_splits", None, "forced splits"),
    ("cegb_tradeoff", 1.0, "CEGB"),
    ("cegb_penalty_split", 0.0, "CEGB"),
    ("extra_trees", False, "extra trees"),
    ("random_strength", None, "random_strength"),
    ("score_function", None, "score_function"),
    ("leaf_estimation_iterations", None, "leaf_estimation_iterations"),
)


def is_multi_target(y, num_targets):
    """Whether `fit` should take the multi-output path.

    True when `y` has a second dimension above 1, or when `num_targets` was
    named above 1. A 2-D `y` of one column is NOT multi-output: it is the
    ordinary single-target case written with a redundant axis, and sending it
    down this path would silently change the model a user gets.
    """
    if num_targets is not None and int(num_targets) > 1:
        return True
    shape = getattr(y, "shape", None)
    return shape is not None and len(shape) == 2 and shape[1] > 1


def target_matrix(y, num_targets, n_rows):
    """`(row-major float64 buffer, n_targets)` for a 2-D target.

    Row-major, `values[r * T + t]`, which is `TargetMatrix`'s documented
    layout and the layout `train_multi_rmse`'s gradient loop reads
    contiguously. `num_targets`, when named, must AGREE with the array; it is
    a check on what the caller believes and not a reshape instruction, so a
    disagreement is an error rather than a silent transpose.
    """
    if not _arrays.have_numpy():
        raise ImportError(
            "a 2-D target needs numpy: there is no stdlib spelling of an "
            "(n_rows, n_targets) float array worth carrying"
        )
    import numpy as np  # noqa: PLC0415

    yb = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
    if yb.ndim != 2:
        raise ValueError(
            "a multi-target fit needs a 2-D y of shape (n_samples, "
            f"n_targets); this one has {yb.ndim} dimension(s)"
        )
    if yb.shape[0] != n_rows:
        raise ValueError(
            f"y has {yb.shape[0]} rows and X has {n_rows}"
        )
    n_targets = int(yb.shape[1])
    if n_targets < 2:
        raise ValueError(
            "a multi-target fit needs at least 2 target columns; one column "
            "is the ordinary single-target fit and takes a 1-D y"
        )
    if num_targets is not None and int(num_targets) != n_targets:
        raise ValueError(
            f"num_targets={int(num_targets)} disagrees with y, which has "
            f"{n_targets} columns"
        )
    return yb, n_targets


def check_honored(estimator, objective_name):
    """Refuse, by name, every parameter the multi-output trainer does not
    reach. The refusal names the mechanism, not the keyword, because the
    keyword is the thing the user already knows."""
    if objective_name not in ("regression", "regression_l2", "l2", "mse",
                             "mean_squared_error", "multirmse"):
        raise ValueError(
            f"a 2-D y is MultiRMSE, and objective={objective_name!r} is not "
            "a squared-error objective. MultiRMSE's derivative is "
            "der[t] = w * (y[t] - p[t]) and multi_target.mojo implements "
            "that one; no other objective has a multi-output trainer."
        )
    for name, default, mechanism in _UNHONORED:
        if not hasattr(estimator, name):
            continue
        value = getattr(estimator, name)
        if value is None or default is None:
            if value is default:
                continue
        elif value == default:
            continue
        raise ValueError(
            f"{name} is not honored by a multi-target fit: "
            f"{mechanism} reaches boosting.train or alternate_boosting, and "
            "multi_target.train_multi_rmse is neither. Refused rather than "
            "accepted and dropped."
        )


def wire_params(estimator):
    """The exact keys `bindings/catboost_reach_bindings._multi_target_params`
    subscripts. Every one required, no defaults on the native side: a missing
    key is a KeyError at the boundary, which is the convention the ordered
    block already established.

    The aliases are resolved through the estimator's own `_resolve_alias`, so
    `iterations=`, `eta=`, `depth=`, `l2_leaf_reg=` and the rest mean here
    exactly what they mean on the single-output path. Repeating the chains
    would be a second answer to a question that already has one.
    """
    alias = estimator._resolve_alias
    n_estimators = alias("n_estimators", "num_iterations", 100)
    n_estimators = alias(
        "n_estimators", "num_boost_round", 100, n_estimators
    )
    n_estimators = alias("n_estimators", "iterations", 100, n_estimators)
    n_estimators = alias("n_estimators", "max_iter", 100, n_estimators)
    learning_rate = alias("learning_rate", "eta", 0.1)
    learning_rate = alias(
        "learning_rate", "shrinkage_rate", 0.1, learning_rate
    )
    num_leaves = alias("num_leaves", "max_leaves", 31)
    num_leaves = alias("num_leaves", "max_leaf_nodes", 31, num_leaves)
    min_data_in_leaf = alias("min_data_in_leaf", "min_child_samples", 20)
    min_data_in_leaf = alias(
        "min_data_in_leaf", "min_samples_leaf", 20, min_data_in_leaf
    )
    min_child_hess = alias("min_child_hess", "min_child_weight", 1e-3)
    min_child_hess = alias(
        "min_child_hess", "min_sum_hessian_in_leaf", 1e-3, min_child_hess
    )
    lambda_l1 = alias("lambda_l1", "reg_alpha", 0.0)
    lambda_l2 = alias("lambda_l2", "reg_lambda", 0.0)
    lambda_l2 = alias("lambda_l2", "l2_leaf_reg", 0.0, lambda_l2)
    max_depth = alias("max_depth", "depth", -1)
    return {
        "n_estimators": int(n_estimators),
        "learning_rate": float(learning_rate),
        "num_leaves": int(num_leaves),
        "min_data_in_leaf": int(min_data_in_leaf),
        "lambda_l2": float(lambda_l2),
        "lambda_l1": float(lambda_l1),
        "min_child_hess": float(min_child_hess),
        "max_depth": int(max_depth),
        "max_bin": int(estimator.max_bin),
        "use_missing": int(bool(estimator.use_missing)),
        "with_missing_values": 0,
        "weight_addr": 0,
    }
