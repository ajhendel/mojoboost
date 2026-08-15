"""Numeric differential against LightGBM for the features that have never had one.

Every row this script covers is `partial` in `docs/LIGHTGBM_PARITY.md` for
exactly one reason, which is that the code is wired up and tested against
itself and no LightGBM comparison has ever run. This is that comparison,
written as a set of named cases so a run produces a number per feature rather
than a verdict per library.

    python tools/lgbm_differential.py --list          # names, tolerances, no imports
    python tools/lgbm_differential.py                 # every case
    python tools/lgbm_differential.py --case dart     # one case (repeatable)

Run it in the bench environment, which has LightGBM and numpy, against a
built extension.

    pixi run build-python
    pixi run -e bench python tools/lgbm_differential.py

What the exit code means. Zero when every case that ran stayed inside its
tolerance. Nonzero when any check exceeded its tolerance. A skipped case
never fails the run, because a skip is a statement that the comparison does
not exist, not that it passed.

The three rules this harness is written to.

1. **Every tolerance is stated with a reason.** `TOLERANCES` below carries
   the number, its unit, and the sentence that justifies it. A tolerance
   with no reason is a number someone tuned until the test went green.

2. **A comparison that cannot be made honestly SKIPs, loudly.** Five cases
   skip, and each prints why and what a real differential would need. The
   skips are the part of this file that took the thought; getting them
   right matters more than raising the count of cases that run.

3. **Where exact agreement is not expected, something else is asserted.**
   Several cases here compare two libraries that grow deliberately different
   trees. Those cases assert what is actually claimed, which is the
   direction of an effect, a rank correlation, a tree count, or membership
   in a band built from LightGBM's own seed-to-seed spread. None of them
   asserts an equality the implementations do not promise.

Reproducibility. One seed (`SEED`), one set of generators, no file read from
disk, no wall-clock or thread-count dependence. LightGBM runs single-threaded
with `deterministic=True` and `force_row_wise=True`, and mojotrees is
deterministic by construction. A run prints both library versions before the
first case so a result can be attached to the pair that produced it.

Precedent. `bench/compare_missing_lightgbm.py`,
`bench/compare_categorical_lightgbm.py`, and `bench/compare_ranking.py` are
the three existing LightGBM comparisons and this file follows their
conventions, which are closed-form synthetic data, matched hyperparameters on
both sides including the ones whose defaults differ (`lambda_l2`,
`min_sum_hessian_in_leaf`, `enable_bundle`), and a printed note wherever a row
is reported for context rather than asserted as a contract.
"""

import argparse
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# Fixed data. Every generator below is a function of SEED alone, so two runs
# on two machines train on the same matrix.
# --------------------------------------------------------------------------

SEED = 20260815

N_ROWS = 4000
N_FEATURES = 8
TRAIN_FRACTION = 0.75

N_QUERIES = 700
VALID_QUERIES = 200
RANK_FEATURES = 8

# Hyperparameters shared by both libraries wherever a case does not override
# them. The three that matter most are the ones whose defaults disagree:
# mojotrees defaults `lambda_l2` to 1.0 and LightGBM to 0.0, mojotrees calls
# LightGBM's `min_sum_hessian_in_leaf` `min_child_hess`, and LightGBM bundles
# features by default where mojotrees has no bundling on this path.
NUM_LEAVES = 31
LEARNING_RATE = 0.1
N_ESTIMATORS = 120
MIN_DATA_IN_LEAF = 20
MIN_SUM_HESSIAN = 1e-3
LAMBDA_L2 = 1.0
MAX_BIN = 255

# Seeds used to measure LightGBM's own spread where a case compares two
# randomized trainers. Five is enough to bracket the spread and cheap.
BAND_SEEDS = (1, 2, 3, 4, 5)


# --------------------------------------------------------------------------
# Tolerances, with reasons.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Tol:
    case: str
    name: str
    value: float
    unit: str
    reason: str


TOLERANCES = (
    Tol(
        "linear_tree",
        "improvement_over_constant",
        0.95,
        "ratio of RMSE, linear over constant",
        "The data is piecewise linear by construction, so linear leaves must "
        "buy a real improvement on both sides. 5% is a floor far below the "
        "improvement either library should show; the check tests the "
        "direction of the effect, not its size.",
    ),
    Tol(
        "linear_tree",
        "cross_library_rmse",
        0.25,
        "relative difference in test RMSE",
        "LightGBM evaluates candidate splits under the per-leaf regression "
        "while mojotrees grows the constant-leaf tree and refits its leaves "
        "afterwards (docs/LINEAR_TREES.md, LightGBM differences). The two "
        "therefore grow different trees on purpose and equality is not "
        "claimable at any tolerance. 25% bounds the claim that is made, "
        "which is that both land in the same accuracy regime.",
    ),
    Tol(
        "linear_lambda",
        "constant_fallback",
        1e-6,
        "relative difference in prediction",
        "mojotrees grows the tree with constant leaves and refits them, so a "
        "huge linear_lambda must drive the coefficients to zero and leave "
        "exactly the constant-leaf model. Coefficients decay like "
        "1/linear_lambda, so at 1e8 their residual contribution is orders "
        "below 1e-6 of the leaf scale. This is mojotrees's own documented "
        "claim (docs/LINEAR_TREES.md, 'The constant fallback is exact'), "
        "checked here because the LightGBM side of the same probe cannot be.",
    ),
    Tol(
        "linear_lambda",
        "ladder_monotonicity",
        0.02,
        "relative slack per rung",
        "Raising linear_lambda must move each library's fit toward its own "
        "constant-leaf fit. LightGBM's tree shape changes with the penalty, "
        "so one rung of the ladder may wobble; 2% of the RMSE allows a "
        "wobble while still failing a ladder that moves the wrong way.",
    ),
    Tol(
        "dart",
        "band_slack",
        0.5,
        "fraction of LightGBM's own seed band width",
        "The drop sets differ at equal seeds by construction. mojotrees "
        "draws from counter-based splitmix64 and LightGBM from a sequential "
        "generator (docs/DART.md 4.1, third bullet). So the tolerance is not "
        "a chosen number at all, it is measured. LightGBM's dart is fit at "
        "five drop seeds, and mojotrees must land inside that band widened "
        "by half its width. A floor of 2% of the band midpoint keeps the "
        "test from becoming impossible when the five seeds happen to agree.",
    ),
    Tol(
        "dart",
        "band_floor",
        0.02,
        "fraction of the band midpoint",
        "The floor under the measured band, for the case where five "
        "LightGBM seeds produce nearly the same number and the band has no "
        "width to widen.",
    ),
    Tol(
        "rf",
        "band_slack",
        0.5,
        "fraction of LightGBM's own seed band width",
        "Same rule and same reason as dart. The row bags differ at equal "
        "seeds because the draws are counter-based splitmix64 rather than "
        "LightGBM's per-block LCG (docs/RANDOM_FOREST_MODE.md section 9, "
        "last bullet), so the comparison is against LightGBM's measured "
        "spread over bagging seeds rather than against one of its runs.",
    ),
    Tol(
        "rf",
        "band_floor",
        0.02,
        "fraction of the band midpoint",
        "As for dart.",
    ),
    Tol(
        "rf",
        "average_not_sum",
        0.10,
        "relative difference from the label mean",
        "A forest averages its trees and each tree carries the base score, "
        "so its mean prediction sits on the label scale. A model that summed "
        "them would be n_estimators times too large, which 10% catches with "
        "an enormous margin. This checks the aggregation rule, not accuracy.",
    ),
    Tol(
        "label_gain",
        "band_slack",
        0.5,
        "fraction of LightGBM's own seed band width",
        "Tree growth diverges after the first floating-point tie and the two "
        "binners break ties differently, which is the reason "
        "bench/compare_ranking.py already gives for not comparing rankers "
        "prediction by prediction. The band is measured over LightGBM's "
        "feature-fraction seeds.",
    ),
    Tol(
        "label_gain",
        "band_floor",
        0.02,
        "fraction of the band midpoint",
        "The floor under the measured band, for the case where the five "
        "feature-fraction seeds produce nearly the same NDCG and the band "
        "has no width to widen.",
    ),
    Tol(
        "label_gain",
        "rank_correlation",
        0.70,
        "Spearman rho, minimum",
        "Two independently grown rankers on the same signal must order "
        "documents similarly even though they will not score them "
        "identically. 0.70 is well above what unrelated models produce on "
        "this data and well below what agreement on the signal produces.",
    ),
    Tol(
        "lambdarank_position_bias",
        "band_slack",
        0.5,
        "fraction of LightGBM's own seed band width",
        "As for label_gain.",
    ),
    Tol(
        "lambdarank_position_bias",
        "band_floor",
        0.02,
        "fraction of the band midpoint",
        "As for label_gain.",
    ),
    Tol(
        "lambdarank_position_bias",
        "rank_correlation",
        0.60,
        "Spearman rho, minimum",
        "Lower than the label_gain case on purpose. The debiasing weights "
        "are estimated during training on both sides, so the two models "
        "diverge for one more reason than tie-breaking. 0.60 still separates "
        "'fitting the same signal' from 'fitting something else'.",
    ),
    Tol(
        "model_edit_rollback",
        "truncation_exact",
        1e-12,
        "relative difference in prediction",
        "Rolling back one iteration is a truncation of the ensemble, and "
        "prediction sums the surviving trees in the same order a fresh fit "
        "of one fewer round would. The two must agree bit for bit on both "
        "sides; 1e-12 allows only for a summation-order difference that "
        "should not exist and would itself be worth knowing about.",
    ),
    Tol(
        "model_edit_leaf_output",
        "delta_exact",
        1e-9,
        "absolute difference in raw score",
        "The edit adds a known delta to one leaf. Every row that reaches "
        "the leaf must move by exactly that delta (LightGBM, which stores "
        "shrunk leaf values) or by delta times the learning rate "
        "(mojotrees, which stores unshrunk values and multiplies at predict "
        "time, docs/MODEL_EDITING.md). Raw scores here are sums of about a "
        "hundred leaf values of order one, so 1e-9 is far above double "
        "rounding of that sum and far below the 0.25 delta being asserted.",
    ),
    Tol(
        "model_edit_shuffle",
        "reorder_stability",
        1e-9,
        "relative difference in prediction",
        "Shuffling permutes whole iterations, and a sum is order independent "
        "in exact arithmetic. Floating-point addition is not associative, so "
        "a prediction may move by a few ulps of the total; 1e-9 relative is "
        "far above a few ulps at double precision and far below any real "
        "change to the model.",
    ),
    Tol(
        "model_edit_refit",
        "no_op_exact",
        1e-12,
        "relative difference in prediction",
        "At decay_rate=1.0 the blend is new = 1*old + 0*fresh on both sides, "
        "which is the identity on every leaf value. Any difference at all is "
        "a bug rather than a rounding matter, so the tolerance exists only "
        "to absorb a summation-order difference.",
    ),
    Tol(
        "model_edit_refit",
        "direction_of_effect",
        1.0,
        "ratio of RMSE on the refit data, after over before",
        "At decay_rate=0.0 every leaf is rebuilt from the new data, so the "
        "refit model must score at least as well as the original on that "
        "data. This is a strict inequality on both sides with no slack, "
        "because the comparison is each library against itself.",
    ),
)


def tolerances_for(case):
    return [t for t in TOLERANCES if t.case == case]


def tol(case, name):
    for t in TOLERANCES:
        if t.case == case and t.name == name:
            return t.value
    raise KeyError(f"no tolerance {name!r} declared for case {case!r}")


# --------------------------------------------------------------------------
# Reporting.
# --------------------------------------------------------------------------


class SkipCase(Exception):
    """Raised by a case that cannot be compared honestly. The message is the
    reason, and it is printed rather than swallowed."""


class Report:
    def __init__(self, name):
        self.name = name
        self.failures = []

    def note(self, text):
        for i, line in enumerate(_wrap(text, 68)):
            print(f"  {'note ' if i == 0 else '     '}  {line}")

    def value(self, label, text):
        print(f"  value  {label:<40} {text}")

    def check(self, label, ok, detail=""):
        print(f"  check  {label:<40} {'PASS' if ok else 'FAIL'}  {detail}")
        if not ok:
            self.failures.append(label)

    def skip(self, reason):
        raise SkipCase(reason)


def _wrap(text, width):
    words = text.split()
    lines, line = [], ""
    for word in words:
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        lines.append(line)
    return lines or [""]


# --------------------------------------------------------------------------
# Lazy imports. Nothing above or below this point imports either library at
# module scope, so --list works on a bare interpreter.
# --------------------------------------------------------------------------


def _lightgbm():
    try:
        import lightgbm as lgb
    except ImportError:
        raise SystemExit(
            "lightgbm is not installed; run this in the bench environment:\n"
            "    pixi run -e bench python tools/lgbm_differential.py"
        )
    return lgb


def _mojotrees():
    if str(REPO / "python") not in sys.path:
        sys.path.insert(0, str(REPO / "python"))
    try:
        import mojotrees
    except ImportError:
        raise SystemExit(
            "mojotrees is not importable; build the extension first:\n"
            "    pixi run build-python"
        )
    return mojotrees


# --------------------------------------------------------------------------
# Data.
# --------------------------------------------------------------------------


def _rng(offset=0):
    import numpy as np

    return np.random.default_rng(SEED + offset)


def piecewise_linear_data():
    """A target that is linear in two features inside each of four regions
    chosen by a third. A constant-leaf tree has to staircase it; a linear
    leaf fits it in one piece. This is the data a linear-tree comparison
    needs, because on a piecewise-constant target linear leaves buy nothing
    and the case would assert an effect that is not there."""
    import numpy as np

    rng = _rng(0)
    X = rng.uniform(-1.0, 1.0, size=(N_ROWS, N_FEATURES))
    region = (X[:, 2] > 0).astype(np.float64) * 2 + (X[:, 3] > 0)
    slope_a = np.array([3.0, -2.0, 1.5, -4.0])[region.astype(int)]
    slope_b = np.array([-1.0, 2.5, -3.0, 0.5])[region.astype(int)]
    y = slope_a * X[:, 0] + slope_b * X[:, 1] + 0.05 * rng.normal(size=N_ROWS)
    return X, y


def offset_regression_data(offset=100.0):
    """The same shape as `piecewise_linear_data` with the target moved far
    from zero. A penalty that reaches the intercept drags predictions toward
    zero here, which is what makes the probe in `case_linear_lambda` able to
    see it."""
    X, y = piecewise_linear_data()
    return X, y + offset


def smooth_regression_data():
    """An ordinary nonlinear regression target for the cases that are about
    the boosting strategy or the model file rather than about leaf shape."""
    import numpy as np

    rng = _rng(1)
    X = rng.uniform(-1.0, 1.0, size=(N_ROWS, N_FEATURES))
    y = (
        np.sin(3.0 * X[:, 0])
        + 2.0 * X[:, 1] * X[:, 2]
        + X[:, 3] ** 2
        - 0.5 * X[:, 4]
        + 0.1 * rng.normal(size=N_ROWS)
    )
    return X, y


def shifted_regression_data():
    """The same features with a different response. This is the `refit`
    case's new data: the structure a model learned on `smooth_regression_data`
    is still usable, and its leaf values are wrong."""
    import numpy as np

    rng = _rng(2)
    X = rng.uniform(-1.0, 1.0, size=(N_ROWS, N_FEATURES))
    y = (
        1.5 * np.sin(3.0 * X[:, 0])
        - 2.0 * X[:, 1] * X[:, 2]
        + 3.0
        + 0.1 * rng.normal(size=N_ROWS)
    )
    return X, y


def split(X, y):
    cut = int(len(y) * TRAIN_FRACTION)
    return X[:cut], y[:cut], X[cut:], y[cut:]


def ranking_data(position_bias=False):
    """Queries of varying length whose grades are a monotone function of a
    latent utility, discretized to LightGBM's 0..4 grades.

    With `position_bias`, the observed grade is the true grade damped by the
    document's display position, and the true grade is returned separately.
    An unbiased ranker is supposed to recover the second from the first."""
    import numpy as np

    rng = _rng(3)
    sizes = rng.integers(6, 17, size=N_QUERIES).astype(np.int64)
    n_rows = int(sizes.sum())
    X = rng.uniform(0.0, 1.0, size=(n_rows, RANK_FEATURES))
    utility = (
        3.0 * X[:, 0]
        + 2.0 * X[:, 1] * X[:, 2]
        - 1.5 * X[:, 3]
        + 0.3 * rng.normal(size=n_rows)
    )

    y_true = np.zeros(n_rows, dtype=np.int32)
    position = np.zeros(n_rows, dtype=np.int32)
    start = 0
    for size in sizes:
        end = start + int(size)
        order = np.argsort(-utility[start:end], kind="stable")
        grades = np.linspace(4.0, 0.0, int(size))
        y_true[start + order] = np.rint(grades).astype(np.int32)
        # The display order is the true order rotated, so position is
        # correlated with relevance without being identical to it.
        position[start:end] = np.arange(int(size), dtype=np.int32)
        start = end

    if not position_bias:
        return X, y_true, sizes, None, y_true

    # Examination probability falls with position; an unexamined document
    # is recorded at a lower grade than it deserves.
    examine = 1.0 / (1.0 + position.astype(np.float64))
    seen = rng.uniform(size=n_rows) < examine
    y_obs = np.where(seen, y_true, np.maximum(y_true - 2, 0)).astype(np.int32)
    return X, y_obs, sizes, position, y_true


def split_by_query(X, y, group, position, y_true, n_valid_queries):
    """Whole queries on each side of the split, never rows."""
    n_valid_rows = int(group[-n_valid_queries:].sum())
    cut = len(y) - n_valid_rows
    train = (X[:cut], y[:cut], group[:-n_valid_queries],
             None if position is None else position[:cut], y_true[:cut])
    valid = (X[cut:], y[cut:], group[-n_valid_queries:],
             None if position is None else position[cut:], y_true[cut:])
    return train, valid


# --------------------------------------------------------------------------
# Matched parameters.
# --------------------------------------------------------------------------


def lgb_params(**overrides):
    params = {
        "objective": "regression",
        "num_leaves": NUM_LEAVES,
        "learning_rate": LEARNING_RATE,
        "min_data_in_leaf": MIN_DATA_IN_LEAF,
        "min_sum_hessian_in_leaf": MIN_SUM_HESSIAN,
        "lambda_l2": LAMBDA_L2,
        "max_bin": MAX_BIN,
        "num_threads": 1,
        "verbose": -1,
        "deterministic": True,
        "force_row_wise": True,
        # mojotrees does no feature bundling on these paths; keep the
        # comparison honest, as the three bench/compare_*.py scripts do.
        "enable_bundle": False,
    }
    params.update(overrides)
    return params


def mt_kwargs(**overrides):
    kwargs = {
        "objective": "regression",
        "num_leaves": NUM_LEAVES,
        "learning_rate": LEARNING_RATE,
        "n_estimators": N_ESTIMATORS,
        "min_data_in_leaf": MIN_DATA_IN_LEAF,
        "min_child_hess": MIN_SUM_HESSIAN,
        "lambda_l2": LAMBDA_L2,
        "max_bin": MAX_BIN,
        "enable_bundle": False,
    }
    kwargs.update(overrides)
    return kwargs


def mt_rank_kwargs(**overrides):
    """`MojoTreesRanker` takes no `objective`: LambdaRank is the one
    objective it trains, so the key has to come out of the shared dict
    rather than be passed as None."""
    kwargs = mt_kwargs(
        lambdarank_truncation_level=30,
        sigmoid=1.0,
        lambdarank_norm=True,
    )
    kwargs.pop("objective")
    kwargs.update(overrides)
    return kwargs


# --------------------------------------------------------------------------
# Small numeric helpers.
# --------------------------------------------------------------------------


def rmse(pred, y):
    import numpy as np

    return float(np.sqrt(np.mean((np.asarray(pred, dtype=np.float64) - y) ** 2)))


def relative(a, b):
    """Relative difference of two numbers against the larger magnitude, so
    it is symmetric and never divides by a near-zero denominator."""
    scale = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / scale


def max_relative(a, b):
    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    scale = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-12)
    return float(np.max(np.abs(a - b) / scale))


def _ranks(x):
    """Ordinal ranks. Ties are not averaged, which is correct here because
    every vector ranked in this file is a continuous model score."""
    import numpy as np

    order = np.argsort(np.asarray(x, dtype=np.float64), kind="stable")
    out = np.empty(len(order), dtype=np.float64)
    out[order] = np.arange(len(order), dtype=np.float64)
    return out


def spearman(a, b):
    import numpy as np

    ra = _ranks(a)
    rb = _ranks(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / denom) if denom > 0.0 else 0.0


def band(values, slack_fraction, floor_fraction):
    """The interval LightGBM's own randomness produced, widened.

    `values` are one metric measured at several LightGBM seeds. The returned
    interval is their range widened by `slack_fraction` of its width, with a
    floor of `floor_fraction` of the midpoint so that a degenerate range does
    not become an impossible test."""
    lo, hi = min(values), max(values)
    mid = 0.5 * (lo + hi)
    slack = max(slack_fraction * (hi - lo), floor_fraction * abs(mid))
    return lo - slack, hi + slack


# --------------------------------------------------------------------------
# Cases that run.
# --------------------------------------------------------------------------


def case_linear_tree(rep):
    lgb = _lightgbm()
    mt = _mojotrees()

    X, y = piecewise_linear_data()
    Xt, yt, Xv, yv = split(X, y)

    def fit_lgb(linear):
        params = lgb_params(linear_tree=linear, linear_lambda=0.0)
        # linear_tree is a dataset parameter in LightGBM as well as a
        # training one, so it has to be present at construction.
        ds = lgb.Dataset(Xt, label=yt, params=params, free_raw_data=False)
        booster = lgb.train(params, ds, num_boost_round=N_ESTIMATORS)
        return rmse(booster.predict(Xv), yv)

    def fit_mt(linear):
        model = mt.MojoTreesRegressor(
            **mt_kwargs(linear_tree=linear, linear_lambda=0.0)
        ).fit(Xt, yt)
        return rmse(model.predict(Xv), yv)

    lgb_const, lgb_linear = fit_lgb(False), fit_lgb(True)
    mt_const, mt_linear = fit_mt(False), fit_mt(True)

    rep.value("LightGBM constant leaves, test RMSE", f"{lgb_const:.6f}")
    rep.value("LightGBM linear leaves, test RMSE", f"{lgb_linear:.6f}")
    rep.value("mojotrees constant leaves, test RMSE", f"{mt_const:.6f}")
    rep.value("mojotrees linear leaves, test RMSE", f"{mt_linear:.6f}")

    ratio = tol("linear_tree", "improvement_over_constant")
    rep.check(
        "LightGBM linear beats its own control",
        lgb_linear <= ratio * lgb_const,
        f"{lgb_linear / lgb_const:.4f} <= {ratio}",
    )
    rep.check(
        "mojotrees linear beats its own control",
        mt_linear <= ratio * mt_const,
        f"{mt_linear / mt_const:.4f} <= {ratio}",
    )

    limit = tol("linear_tree", "cross_library_rmse")
    gap = relative(mt_linear, lgb_linear)
    rep.check(
        "the two linear fits are in one regime",
        gap <= limit,
        f"relative gap {gap:.4f} <= {limit}",
    )
    rep.note(
        "Equality is not asserted and should not be. LightGBM scores "
        "candidate splits under the per-leaf regression, so its trees have a "
        "different shape from mojotrees's, which grows the constant-leaf "
        "tree and refits its leaves afterwards. That is the largest "
        "difference between the two implementations and it is deliberate."
    )


def case_linear_lambda(rep):
    lgb = _lightgbm()
    mt = _mojotrees()

    X, y = offset_regression_data()
    Xt, yt, Xv, yv = split(X, y)
    ladder = (0.0, 1.0, 100.0, 1e8)

    def fit_lgb(linear, lam):
        params = lgb_params(linear_tree=linear, linear_lambda=lam)
        ds = lgb.Dataset(Xt, label=yt, params=params, free_raw_data=False)
        return lgb.train(params, ds, num_boost_round=N_ESTIMATORS).predict(Xv)

    def fit_mt(linear, lam):
        model = mt.MojoTreesRegressor(
            **mt_kwargs(linear_tree=linear, linear_lambda=lam)
        ).fit(Xt, yt)
        return model.predict(Xv)

    lgb_const = fit_lgb(False, 0.0)
    mt_const = fit_mt(False, 0.0)
    lgb_rung = [rmse(fit_lgb(True, lam), yv) for lam in ladder]
    mt_pred = [fit_mt(True, lam) for lam in ladder]
    mt_rung = [rmse(p, yv) for p in mt_pred]

    rep.value("linear_lambda ladder", " ".join(f"{lam:g}" for lam in ladder))
    rep.value("LightGBM RMSE by rung", " ".join(f"{v:.5f}" for v in lgb_rung))
    rep.value("mojotrees RMSE by rung", " ".join(f"{v:.5f}" for v in mt_rung))
    rep.value("LightGBM constant control", f"{rmse(lgb_const, yv):.5f}")
    rep.value("mojotrees constant control", f"{rmse(mt_const, yv):.5f}")

    slack_frac = tol("linear_lambda", "ladder_monotonicity")

    def ladder_ok(rungs):
        slack = slack_frac * max(rungs)
        return all(b >= a - slack for a, b in zip(rungs, rungs[1:]))

    rep.check(
        "LightGBM moves toward its constant fit",
        ladder_ok(lgb_rung),
        f"per-rung slack {slack_frac:g} of the largest RMSE",
    )
    rep.check(
        "mojotrees moves toward its constant fit",
        ladder_ok(mt_rung),
        f"per-rung slack {slack_frac:g} of the largest RMSE",
    )

    exact = tol("linear_lambda", "constant_fallback")
    gap = max_relative(mt_pred[-1], mt_const)
    rep.check(
        "mojotrees at linear_lambda=1e8 is its constant fit",
        gap <= exact,
        f"max relative gap {gap:.2e} <= {exact:g}",
    )

    # The probe that answers docs/LINEAR_TREES.md, "Parity-unverified" item 1:
    # does LightGBM's linear_lambda reach the intercept row? The labels sit
    # around +100, so a penalized intercept collapses predictions toward zero
    # and an unpenalized one leaves them on the label scale.
    import numpy as np

    label_scale = float(np.mean(yv))
    lgb_huge = float(np.mean(fit_lgb(True, 1e8)))
    mt_huge = float(np.mean(mt_pred[-1]))
    rep.value("label mean on the validation rows", f"{label_scale:.4f}")
    rep.value("LightGBM mean prediction at 1e8", f"{lgb_huge:.4f}")
    rep.value("mojotrees mean prediction at 1e8", f"{mt_huge:.4f}")
    penalized = abs(lgb_huge) < 0.5 * abs(label_scale)
    rep.note(
        "Intercept probe, reported and not asserted. LightGBM appears "
        + ("TO penalize" if penalized else "NOT to penalize")
        + " the intercept row of its linear leaf solve. mojotrees does not "
        "penalize it. This is the open question in docs/LINEAR_TREES.md "
        "'Parity-unverified' item 1, and the answer belongs in that document "
        "once a run has produced it. It is not a failure either way. The two "
        "libraries are allowed to place the penalty differently as long as "
        "the placement is written down."
    )


def case_dart(rep):
    lgb = _lightgbm()
    mt = _mojotrees()

    X, y = smooth_regression_data()
    Xt, yt, Xv, yv = split(X, y)

    dart = dict(
        drop_rate=0.1,
        max_drop=50,
        skip_drop=0.5,
        xgboost_dart_mode=False,
        # LightGBM's default is the weight-proportional rule. mojotrees
        # defaults the other way, so the matched configuration has to say so
        # (docs/DART.md 4.1, and docs/LIGHTGBM_PARITY.md notes the default
        # difference under `uniform_drop`).
        uniform_drop=False,
    )

    def fit_lgb_dart(drop_seed):
        params = lgb_params(boosting="dart", drop_seed=drop_seed, **dart)
        ds = lgb.Dataset(Xt, label=yt, params=params, free_raw_data=False)
        booster = lgb.train(params, ds, num_boost_round=N_ESTIMATORS)
        return booster, rmse(booster.predict(Xv), yv)

    lgb_boosters, lgb_scores = [], []
    for seed in BAND_SEEDS:
        booster, score = fit_lgb_dart(seed)
        lgb_boosters.append(booster)
        lgb_scores.append(score)

    lo, hi = band(
        lgb_scores,
        tol("dart", "band_slack"),
        tol("dart", "band_floor"),
    )

    mt_model = mt.MojoTreesRegressor(
        **mt_kwargs(boosting="dart", drop_seed=4, **dart)
    ).fit(Xt, yt)
    mt_pred = mt_model.predict(Xv)
    mt_score = rmse(mt_pred, yv)

    gbdt_params = lgb_params()
    gbdt_ds = lgb.Dataset(Xt, label=yt, params=gbdt_params, free_raw_data=False)
    lgb_gbdt = lgb.train(gbdt_params, gbdt_ds, num_boost_round=N_ESTIMATORS)
    lgb_gbdt_pred = lgb_gbdt.predict(Xv)
    mt_gbdt_pred = mt.MojoTreesRegressor(**mt_kwargs()).fit(Xt, yt).predict(Xv)

    rep.value(
        "LightGBM dart over 5 drop seeds",
        " ".join(f"{v:.5f}" for v in lgb_scores),
    )
    rep.value("widened band", f"[{lo:.5f}, {hi:.5f}]")
    rep.value("mojotrees dart, test RMSE", f"{mt_score:.5f}")

    rep.check(
        "mojotrees dart lands in LightGBM's band",
        lo <= mt_score <= hi,
        f"{mt_score:.5f} in [{lo:.5f}, {hi:.5f}]",
    )

    import numpy as np

    lgb_trees = lgb_boosters[0].num_trees()
    mt_trees = int(getattr(mt_model, "n_iter_", 0)) or mt_model.booster_.num_trees()
    rep.check(
        "both keep every round's tree",
        lgb_trees == N_ESTIMATORS and mt_trees == N_ESTIMATORS,
        f"lightgbm {lgb_trees}, mojotrees {mt_trees}, asked for {N_ESTIMATORS}",
    )

    lgb_moved = float(np.max(np.abs(lgb_boosters[0].predict(Xv) - lgb_gbdt_pred)))
    mt_moved = float(np.max(np.abs(np.asarray(mt_pred) - np.asarray(mt_gbdt_pred))))
    rep.check(
        "dropout changed the model on both sides",
        lgb_moved > 1e-9 and mt_moved > 1e-9,
        f"max |dart - gbdt|: lightgbm {lgb_moved:.3e}, mojotrees {mt_moved:.3e}",
    )
    rep.note(
        "Prediction-level agreement is not asserted. The drop draws come "
        "from counter-based splitmix64 here and from a sequential generator "
        "in LightGBM, so equal seeds select different iterations by "
        "construction and the two ensembles are different models of the same "
        "algorithm. The band is LightGBM's own spread over five drop seeds, "
        "which is the honest yardstick for 'the same algorithm'."
    )


def case_rf(rep):
    lgb = _lightgbm()
    mt = _mojotrees()

    X, y = smooth_regression_data()
    Xt, yt, Xv, yv = split(X, y)

    # LightGBM's RF::Init aborts unless bagging or feature sampling is on,
    # and check_rf_params is the same check on the mojotrees side, so the
    # matched configuration has to supply a randomizer.
    forest = dict(bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8)

    def fit_lgb_rf(seed):
        params = lgb_params(
            boosting="rf",
            learning_rate=1.0,
            bagging_seed=seed,
            feature_fraction_seed=seed,
            **forest,
        )
        ds = lgb.Dataset(Xt, label=yt, params=params, free_raw_data=False)
        booster = lgb.train(params, ds, num_boost_round=N_ESTIMATORS)
        return booster, booster.predict(Xv)

    lgb_boosters, lgb_preds = [], []
    for seed in BAND_SEEDS:
        booster, pred = fit_lgb_rf(seed)
        lgb_boosters.append(booster)
        lgb_preds.append(pred)
    lgb_scores = [rmse(p, yv) for p in lgb_preds]

    lo, hi = band(lgb_scores, tol("rf", "band_slack"), tol("rf", "band_floor"))

    mt_model = mt.MojoTreesRegressor(
        **mt_kwargs(boosting="rf", bagging_seed=3, feature_fraction_seed=2, **forest)
    ).fit(Xt, yt)
    mt_pred = mt_model.predict(Xv)
    mt_score = rmse(mt_pred, yv)

    rep.value(
        "LightGBM rf over 5 seeds",
        " ".join(f"{v:.5f}" for v in lgb_scores),
    )
    rep.value("widened band", f"[{lo:.5f}, {hi:.5f}]")
    rep.value("mojotrees rf, test RMSE", f"{mt_score:.5f}")

    rep.check(
        "mojotrees rf lands in LightGBM's band",
        lo <= mt_score <= hi,
        f"{mt_score:.5f} in [{lo:.5f}, {hi:.5f}]",
    )

    import numpy as np

    label_mean = float(np.mean(yv))
    limit = tol("rf", "average_not_sum")
    for label, pred in (("LightGBM", lgb_preds[0]), ("mojotrees", mt_pred)):
        gap = relative(float(np.mean(pred)), label_mean)
        rep.check(
            f"{label} rf averages rather than sums",
            gap <= limit,
            f"mean prediction {float(np.mean(pred)):.4f} against label mean "
            f"{label_mean:.4f}, relative {gap:.4f} <= {limit}",
        )

    lgb_trees = lgb_boosters[0].num_trees()
    mt_trees = int(getattr(mt_model, "n_iter_", 0)) or mt_model.booster_.num_trees()
    rep.check(
        "both grow exactly n_estimators trees",
        lgb_trees == N_ESTIMATORS and mt_trees == N_ESTIMATORS,
        f"lightgbm {lgb_trees}, mojotrees {mt_trees}",
    )
    rep.note(
        "Four differences from LightGBM are intentional and none of them is "
        "exercised by this configuration. Continued training keeps the base "
        "score, a degenerate tree keeps its grown value rather than being "
        "replaced by a zero leaf, GOSS scales a copy of the shared "
        "gradients, and the GOSS warmup is keyed to the rf shrinkage. They "
        "are listed in docs/RANDOM_FOREST_MODE.md section 9 and each one "
        "deserves its own case before this row claims more than the band."
    )


def case_label_gain(rep):
    lgb = _lightgbm()
    mt = _mojotrees()

    X, y, group, _, _ = ranking_data()
    (Xt, yt, gt, _, _), (Xv, yv, gv, _, _) = split_by_query(
        X, y, group, None, y, VALID_QUERIES
    )

    # A linear gain instead of LightGBM's default 2^i - 1. If the parameter
    # is read at all, the two must disagree with their own default-gain runs.
    custom_gain = [0.0, 1.0, 2.0, 3.0, 4.0]

    def rank_params(seed, gain):
        params = lgb_params(
            objective="lambdarank",
            metric="ndcg",
            eval_at=[5],
            lambdarank_truncation_level=30,
            sigmoid=1.0,
            lambdarank_norm=True,
            feature_fraction=0.8,
            feature_fraction_seed=seed,
        )
        if gain is not None:
            params["label_gain"] = list(gain)
        return params

    def fit_lgb_rank(seed, gain):
        params = rank_params(seed, gain)
        ds = lgb.Dataset(
            Xt, label=yt, group=gt, params=params, free_raw_data=False
        )
        return lgb.train(params, ds, num_boost_round=N_ESTIMATORS).predict(Xv)

    def fit_mt_rank(seed, gain):
        model = mt.MojoTreesRanker(
            **mt_rank_kwargs(
                label_gain=gain,
                feature_fraction=0.8,
                feature_fraction_seed=seed,
            )
        ).fit(Xt, yt, group=gt)
        return model.predict(Xv)

    lgb_band_preds = [fit_lgb_rank(seed, custom_gain) for seed in BAND_SEEDS]
    lgb_scores = [
        float(mt.ndcg_score(p, yv, gv, at=5)) for p in lgb_band_preds
    ]
    lo, hi = band(
        lgb_scores,
        tol("label_gain", "band_slack"),
        tol("label_gain", "band_floor"),
    )

    mt_custom = fit_mt_rank(2, custom_gain)
    mt_score = float(mt.ndcg_score(mt_custom, yv, gv, at=5))

    rep.value(
        "LightGBM NDCG@5 over 5 seeds, custom gain",
        " ".join(f"{v:.5f}" for v in lgb_scores),
    )
    rep.value("widened band", f"[{lo:.5f}, {hi:.5f}]")
    rep.value("mojotrees NDCG@5, custom gain", f"{mt_score:.5f}")

    rep.check(
        "mojotrees lands in LightGBM's band",
        lo <= mt_score <= hi,
        f"{mt_score:.5f} in [{lo:.5f}, {hi:.5f}]",
    )

    rho = spearman(mt_custom, lgb_band_preds[1])
    minimum = tol("label_gain", "rank_correlation")
    rep.check(
        "the two rankers order documents alike",
        rho >= minimum,
        f"Spearman rho {rho:.4f} >= {minimum}",
    )

    import numpy as np

    lgb_default = fit_lgb_rank(2, None)
    mt_default = fit_mt_rank(2, None)
    lgb_moved = float(np.max(np.abs(fit_lgb_rank(2, custom_gain) - lgb_default)))
    mt_moved = float(
        np.max(np.abs(np.asarray(mt_custom) - np.asarray(mt_default)))
    )
    rep.check(
        "label_gain is read on both sides",
        lgb_moved > 1e-9 and mt_moved > 1e-9,
        f"max |custom - default|: lightgbm {lgb_moved:.3e}, "
        f"mojotrees {mt_moved:.3e}",
    )
    rep.note(
        "The gain vector stops at label 4 here. mojotrees refuses labels "
        "above 30 even with a longer vector, which LightGBM also does, and "
        "no case in this file probes that boundary."
    )


def case_lambdarank_position_bias(rep):
    lgb = _lightgbm()
    mt = _mojotrees()

    import inspect

    if "position" not in inspect.signature(lgb.Dataset.__init__).parameters:
        rep.skip(
            "this LightGBM has no Dataset(position=), so unbiased LambdaRank "
            f"cannot be turned on. LightGBM {lgb.__version__} predates the "
            "parameter; 4.1 is the first release that carries it. Install a "
            "newer LightGBM in the bench environment and rerun."
        )

    X, y_obs, group, position, y_true = ranking_data(position_bias=True)
    (Xt, yt, gt, pt, _), (Xv, _, gv, _, yv_true) = split_by_query(
        X, y_obs, group, position, y_true, VALID_QUERIES
    )

    regularization = 0.01

    def fit_lgb_rank(seed, with_position):
        params = lgb_params(
            objective="lambdarank",
            metric="ndcg",
            eval_at=[5],
            lambdarank_truncation_level=30,
            sigmoid=1.0,
            lambdarank_norm=True,
            feature_fraction=0.8,
            feature_fraction_seed=seed,
        )
        if with_position:
            params["lambdarank_position_bias_regularization"] = regularization
        ds = lgb.Dataset(
            Xt,
            label=yt,
            group=gt,
            position=pt if with_position else None,
            params=params,
            free_raw_data=False,
        )
        return lgb.train(params, ds, num_boost_round=N_ESTIMATORS).predict(Xv)

    def fit_mt_rank(seed, with_position):
        model = mt.MojoTreesRanker(
            **mt_rank_kwargs(
                lambdarank_position_bias_regularization=(
                    regularization if with_position else 0.0
                ),
                feature_fraction=0.8,
                feature_fraction_seed=seed,
            )
        ).fit(Xt, yt, group=gt, position=pt if with_position else None)
        return model.predict(Xv)

    lgb_band_preds = [fit_lgb_rank(seed, True) for seed in BAND_SEEDS]
    lgb_scores = [
        float(mt.ndcg_score(p, yv_true, gv, at=5)) for p in lgb_band_preds
    ]
    lo, hi = band(
        lgb_scores,
        tol("lambdarank_position_bias", "band_slack"),
        tol("lambdarank_position_bias", "band_floor"),
    )

    mt_unbiased = fit_mt_rank(2, True)
    mt_score = float(mt.ndcg_score(mt_unbiased, yv_true, gv, at=5))

    rep.value(
        "LightGBM NDCG@5 over 5 seeds, unbiased",
        " ".join(f"{v:.5f}" for v in lgb_scores),
    )
    rep.value("widened band", f"[{lo:.5f}, {hi:.5f}]")
    rep.value("mojotrees NDCG@5, unbiased", f"{mt_score:.5f}")

    rep.check(
        "mojotrees lands in LightGBM's band",
        lo <= mt_score <= hi,
        f"{mt_score:.5f} in [{lo:.5f}, {hi:.5f}]",
    )

    rho = spearman(mt_unbiased, lgb_band_preds[1])
    minimum = tol("lambdarank_position_bias", "rank_correlation")
    rep.check(
        "the two unbiased rankers order documents alike",
        rho >= minimum,
        f"Spearman rho {rho:.4f} >= {minimum}",
    )

    import numpy as np

    lgb_blind = fit_lgb_rank(2, False)
    mt_blind = fit_mt_rank(2, False)
    lgb_moved = float(np.max(np.abs(lgb_band_preds[1] - lgb_blind)))
    mt_moved = float(
        np.max(np.abs(np.asarray(mt_unbiased) - np.asarray(mt_blind)))
    )
    rep.check(
        "the position column is read on both sides",
        lgb_moved > 1e-9 and mt_moved > 1e-9,
        f"max |unbiased - blind|: lightgbm {lgb_moved:.3e}, "
        f"mojotrees {mt_moved:.3e}",
    )

    lgb_gain = float(mt.ndcg_score(lgb_band_preds[1], yv_true, gv, at=5)) - float(
        mt.ndcg_score(lgb_blind, yv_true, gv, at=5)
    )
    mt_gain = mt_score - float(mt.ndcg_score(mt_blind, yv_true, gv, at=5))
    rep.note(
        "Direction of effect, reported and not asserted. Debiasing changed "
        f"NDCG@5 against the true grades by {lgb_gain:+.5f} for LightGBM and "
        f"{mt_gain:+.5f} for mojotrees. Whether debiasing helps on a given "
        "synthetic bias is a property of the data generator, not of either "
        "implementation, so a sign disagreement here is information rather "
        "than a failure. What is asserted is that both read the column and "
        "land in the same accuracy band."
    )
    rep.note(
        "mojotrees refuses eval_set together with position-bias "
        "regularization, so neither side uses a validation set inside the "
        "fit. A case that needs one would have to skip for that reason."
    )


def case_model_edit_rollback(rep):
    lgb = _lightgbm()
    mt = _mojotrees()

    X, y = smooth_regression_data()
    Xt, yt, Xv, _ = split(X, y)
    rounds = 40

    params = lgb_params()
    ds = lgb.Dataset(Xt, label=yt, params=params, free_raw_data=False)
    lgb_full = lgb.train(params, ds, num_boost_round=rounds)
    lgb_short = lgb.train(
        params,
        lgb.Dataset(Xt, label=yt, params=params, free_raw_data=False),
        num_boost_round=rounds - 1,
    )
    lgb_expected = lgb_short.predict(Xv)
    lgb_full.rollback_one_iter()
    lgb_rolled = lgb_full.predict(Xv)

    mt_params = dict(
        objective="regression",
        num_leaves=NUM_LEAVES,
        learning_rate=LEARNING_RATE,
        min_data_in_leaf=MIN_DATA_IN_LEAF,
        # LightGBM's min_sum_hessian_in_leaf, under the name the
        # estimators take it by; mojotrees.train hands params straight to
        # the matching estimator's constructor.
        min_child_hess=MIN_SUM_HESSIAN,
        lambda_l2=LAMBDA_L2,
    )
    dataset_params = {"max_bin": MAX_BIN}
    mt_full = mt.train(
        mt_params,
        mt.Dataset(Xt, label=yt, params=dataset_params),
        num_boost_round=rounds,
    )
    mt_short = mt.train(
        mt_params,
        mt.Dataset(Xt, label=yt, params=dataset_params),
        num_boost_round=rounds - 1,
    )
    mt_expected = mt_short.predict(Xv)
    mt_full.rollback_one_iter()
    mt_rolled = mt_full.predict(Xv)

    limit = tol("model_edit_rollback", "truncation_exact")
    lgb_gap = max_relative(lgb_rolled, lgb_expected)
    mt_gap = max_relative(mt_rolled, mt_expected)

    rep.value("rounds trained, then rolled back", f"{rounds} then 1")
    rep.check(
        "LightGBM rollback equals a fresh short fit",
        lgb_gap <= limit,
        f"max relative gap {lgb_gap:.2e} <= {limit:g}",
    )
    rep.check(
        "mojotrees rollback equals a fresh short fit",
        mt_gap <= limit,
        f"max relative gap {mt_gap:.2e} <= {limit:g}",
    )
    rep.check(
        "both report the same tree count after the rollback",
        lgb_full.num_trees() == mt_full.num_trees() == rounds - 1,
        f"lightgbm {lgb_full.num_trees()}, mojotrees {mt_full.num_trees()}",
    )
    rep.note(
        "Cross-library prediction equality is not asserted here and is not "
        "what the row needs. rollback_one_iter is a statement about "
        "truncation semantics, and the differential is that both libraries "
        "recover their own one-round-shorter model exactly."
    )


def case_model_edit_leaf_output(rep):
    lgb = _lightgbm()
    mt = _mojotrees()
    import numpy as np

    booster_cls = lgb.Booster
    if not hasattr(booster_cls, "set_leaf_output"):
        rep.skip(
            f"LightGBM {lgb.__version__} has no Booster.set_leaf_output, so "
            "there is nothing to compare the write against. The method "
            "arrived in LightGBM 4.x; install a newer one in the bench "
            "environment and rerun."
        )

    X, y = smooth_regression_data()
    Xt, yt, Xv, _ = split(X, y)
    delta = 0.25

    params = lgb_params()
    ds = lgb.Dataset(Xt, label=yt, params=params, free_raw_data=False)
    lgb_booster = lgb.train(params, ds, num_boost_round=N_ESTIMATORS)
    lgb_before = lgb_booster.predict(Xv, raw_score=True)
    lgb_leaves = np.asarray(lgb_booster.predict(Xv, pred_leaf=True))[:, 0]
    lgb_target = int(lgb_leaves[0])
    lgb_old = lgb_booster.get_leaf_output(0, lgb_target)
    lgb_booster.set_leaf_output(0, lgb_target, lgb_old + delta)
    lgb_after = lgb_booster.predict(Xv, raw_score=True)

    mt_model = mt.MojoTreesRegressor(**mt_kwargs()).fit(Xt, yt)
    mt_before = np.asarray(mt_model.predict(Xv, raw_score=True))
    mt_leaves = np.asarray(mt_model.predict(Xv, pred_leaf=True))[:, 0]
    mt_target = int(mt_leaves[0])
    mt_booster = mt_model.booster_
    mt_old = mt_booster.get_leaf_output(0, mt_target)
    stored = mt_booster.set_leaf_output(0, mt_target, mt_old + delta)
    mt_after = np.asarray(mt_model.predict(Xv, raw_score=True))

    limit = tol("model_edit_leaf_output", "delta_exact")

    def moved(before, after, mask, expected):
        moves = (after - before)[mask]
        return float(np.max(np.abs(moves - expected))) if moves.size else float("nan")

    lgb_mask = lgb_leaves == lgb_target
    mt_mask = mt_leaves == mt_target
    rep.value(
        "rows reaching the edited leaf",
        f"lightgbm {int(lgb_mask.sum())}, mojotrees {int(mt_mask.sum())}",
    )
    rep.value("stored leaf value returned by mojotrees", f"{stored:.6f}")

    lgb_err = moved(lgb_before, lgb_after, lgb_mask, delta)
    rep.check(
        "LightGBM moves its leaf's rows by the delta",
        lgb_err <= limit,
        f"max error {lgb_err:.2e} <= {limit:g}, shrinkage folded into the value",
    )

    mt_err = moved(mt_before, mt_after, mt_mask, delta * LEARNING_RATE)
    rep.check(
        "mojotrees moves its leaf's rows by delta * learning_rate",
        mt_err <= limit,
        f"max error {mt_err:.2e} <= {limit:g}, shrinkage applied at predict time",
    )

    lgb_rest = moved(lgb_before, lgb_after, ~lgb_mask, 0.0)
    mt_rest = moved(mt_before, mt_after, ~mt_mask, 0.0)
    rep.check(
        "no other row moves on either side",
        lgb_rest <= limit and mt_rest <= limit,
        f"lightgbm {lgb_rest:.2e}, mojotrees {mt_rest:.2e}",
    )
    rep.note(
        "Leaf values are not compared across libraries and cannot be. "
        "mojotrees addresses leaves by leaf ordinal, the rank among the "
        "tree's leaves in node order that predict(pred_leaf=True) reports, "
        "and LightGBM addresses them by its own leaf id; the two agree only "
        "by coincidence. mojotrees also stores the unshrunk value where "
        "LightGBM folds shrinkage in. Both facts are what the two checks "
        "above assert, each library addressed by its own convention."
    )


def case_model_edit_shuffle(rep):
    lgb = _lightgbm()
    mt = _mojotrees()
    import numpy as np

    X, y = smooth_regression_data()
    Xt, yt, Xv, _ = split(X, y)

    params = lgb_params()
    ds = lgb.Dataset(Xt, label=yt, params=params, free_raw_data=False)
    lgb_booster = lgb.train(params, ds, num_boost_round=N_ESTIMATORS)
    lgb_before = lgb_booster.predict(Xv, raw_score=True)
    lgb_order_before = _tree_signature(lgb_booster.dump_model())
    lgb_booster.shuffle_models(0, N_ESTIMATORS)
    lgb_after = lgb_booster.predict(Xv, raw_score=True)
    lgb_order_after = _tree_signature(lgb_booster.dump_model())

    mt_params = dict(
        objective="regression",
        num_leaves=NUM_LEAVES,
        learning_rate=LEARNING_RATE,
        min_data_in_leaf=MIN_DATA_IN_LEAF,
        # LightGBM's min_sum_hessian_in_leaf, under the name the
        # estimators take it by; mojotrees.train hands params straight to
        # the matching estimator's constructor.
        min_child_hess=MIN_SUM_HESSIAN,
        lambda_l2=LAMBDA_L2,
    )
    mt_booster = mt.train(
        mt_params,
        mt.Dataset(Xt, label=yt, params={"max_bin": MAX_BIN}),
        num_boost_round=N_ESTIMATORS,
    )
    mt_before = np.asarray(mt_booster.predict(Xv))
    mt_order_before = _tree_signature(mt_booster.dump_model())
    mt_booster.shuffle_models(0, N_ESTIMATORS, seed=11)
    mt_after = np.asarray(mt_booster.predict(Xv))
    mt_order_after = _tree_signature(mt_booster.dump_model())

    limit = tol("model_edit_shuffle", "reorder_stability")
    lgb_gap = max_relative(lgb_after, lgb_before)
    mt_gap = max_relative(mt_after, mt_before)

    rep.check(
        "LightGBM predictions survive the shuffle",
        lgb_gap <= limit,
        f"max relative gap {lgb_gap:.2e} <= {limit:g}",
    )
    rep.check(
        "mojotrees predictions survive the shuffle",
        mt_gap <= limit,
        f"max relative gap {mt_gap:.2e} <= {limit:g}",
    )
    rep.check(
        "both keep every tree",
        lgb_booster.num_trees() == mt_booster.num_trees() == N_ESTIMATORS,
        f"lightgbm {lgb_booster.num_trees()}, mojotrees {mt_booster.num_trees()}",
    )
    rep.note(
        "Reordering, reported and not asserted. LightGBM "
        + ("did" if lgb_order_before != lgb_order_after else "did NOT")
        + " change its tree order and mojotrees "
        + ("did" if mt_order_before != mt_order_after else "did NOT")
        + ". LightGBM's shuffle takes no seed and mojotrees's does, so the "
        "two permutations are unrelated by construction and only the "
        "invariance of the prediction is a shared claim."
    )


def _tree_signature(dump):
    """A cheap order-sensitive fingerprint of a dumped ensemble, used only to
    report whether a shuffle actually permuted anything."""
    trees = dump.get("tree_info") or dump.get("trees") or []
    out = []
    for tree in trees:
        structure = tree.get("tree_structure", tree)
        out.append(
            (
                structure.get("split_feature"),
                structure.get("threshold"),
                structure.get("leaf_value"),
            )
        )
    return tuple(out)


def case_model_edit_refit(rep):
    lgb = _lightgbm()
    mt = _mojotrees()

    X, y = smooth_regression_data()
    Xt, yt, Xv, _ = split(X, y)
    Xn, yn = shifted_regression_data()
    Xn, yn = Xn[: len(yt)], yn[: len(yt)]

    params = lgb_params()
    ds = lgb.Dataset(Xt, label=yt, params=params, free_raw_data=False)
    lgb_booster = lgb.train(params, ds, num_boost_round=N_ESTIMATORS)
    lgb_before = lgb_booster.predict(Xv)
    lgb_noop = lgb_booster.refit(Xn, yn, decay_rate=1.0)
    lgb_noop_pred = lgb_noop.predict(Xv)
    lgb_fresh = lgb_booster.refit(Xn, yn, decay_rate=0.0)
    lgb_new_before = rmse(lgb_booster.predict(Xn), yn)
    lgb_new_after = rmse(lgb_fresh.predict(Xn), yn)

    mt_params = dict(
        objective="regression",
        num_leaves=NUM_LEAVES,
        learning_rate=LEARNING_RATE,
        min_data_in_leaf=MIN_DATA_IN_LEAF,
        # LightGBM's min_sum_hessian_in_leaf, under the name the
        # estimators take it by; mojotrees.train hands params straight to
        # the matching estimator's constructor.
        min_child_hess=MIN_SUM_HESSIAN,
        lambda_l2=LAMBDA_L2,
    )
    train_set = mt.Dataset(Xt, label=yt, params={"max_bin": MAX_BIN})
    mt_booster = mt.train(mt_params, train_set, num_boost_round=N_ESTIMATORS)
    mt_before = mt_booster.predict(Xv)
    mt_new_before = rmse(mt_booster.predict(Xn), yn)
    mt_booster.refit(Xn, label=yn, decay_rate=1.0)
    mt_noop_pred = mt_booster.predict(Xv)

    mt_booster2 = mt.train(
        mt_params,
        mt.Dataset(Xt, label=yt, params={"max_bin": MAX_BIN}),
        num_boost_round=N_ESTIMATORS,
    )
    report = mt_booster2.refit(Xn, label=yn, decay_rate=0.0)
    mt_new_after = rmse(mt_booster2.predict(Xn), yn)

    limit = tol("model_edit_refit", "no_op_exact")
    lgb_gap = max_relative(lgb_noop_pred, lgb_before)
    mt_gap = max_relative(mt_noop_pred, mt_before)

    rep.value("mojotrees refit report at decay_rate=0", str(report))
    rep.check(
        "LightGBM refit at decay_rate=1 is a no-op",
        lgb_gap <= limit,
        f"max relative gap {lgb_gap:.2e} <= {limit:g}",
    )
    rep.check(
        "mojotrees refit at decay_rate=1 is a no-op",
        mt_gap <= limit,
        f"max relative gap {mt_gap:.2e} <= {limit:g}",
    )
    ratio = tol("model_edit_refit", "direction_of_effect")
    rep.check(
        "LightGBM refit at decay_rate=0 fits the new data",
        lgb_new_after < ratio * lgb_new_before,
        f"RMSE {lgb_new_before:.5f} -> {lgb_new_after:.5f}",
    )
    rep.check(
        "mojotrees refit at decay_rate=0 fits the new data",
        mt_new_after < ratio * mt_new_before,
        f"RMSE {mt_new_before:.5f} -> {mt_new_after:.5f}",
    )
    rep.note(
        "Two API differences, both reported rather than asserted. LightGBM's "
        "refit returns a new Booster and leaves the original alone; "
        "mojotrees refits in place and returns a dict saying how many trees "
        "and leaves it touched. And mojotrees defaults min_leaf_rows to 1, "
        "so a leaf no refit row reaches keeps its old value, where LightGBM "
        "applies no floor at all. Neither difference is visible in the "
        "checks above, which is why this case asserts the blend rule and the "
        "direction of the effect and nothing else."
    )


# --------------------------------------------------------------------------
# Cases that skip, with the reason each one cannot be compared.
# --------------------------------------------------------------------------


def case_dart_early_stopping(rep):
    rep.skip(
        "There is no shared behavior to compare. LightGBM's "
        "DART::EvalAndCheckEarlyStopping returns false unconditionally "
        "(src/boosting/dart.hpp, read 2026-08-15), so early_stopping_round "
        "is silently inert under boosting='dart' there and a LightGBM dart "
        "run always trains every round it was asked for. mojotrees does "
        "implement dart early stopping, by snapshotting the per-tree weight "
        "vector whenever the validation loss improves and restoring it "
        "(DartBestState, dart_record_best, dart_restore_best, docs/DART.md "
        "section 6), because popping trees off the end recovers the right "
        "tree set and the wrong weights. But that path is unreachable from "
        "Python: the estimators refuse eval_set together with dart by name. "
        "So one library has the feature and does not expose it on this "
        "entry point, and the other does not have the feature. A "
        "differential would be comparing mojotrees against a no-op. What "
        "this row needs instead is a mojotrees-only test that the restored "
        "weights reproduce the best round exactly, plus a Python entry "
        "point for dart with eval_set."
    )


def _tree_learner_skip(rep, mode, extra):
    rep.skip(
        f"LightGBM's tree_learner='{mode}' needs a real multi-process world. "
        "It is configured with num_machines, a machine list file or a "
        "machines string, local_listen_port, and time_out, and the ranks "
        "talk over sockets; there is no in-process form of it. mojotrees "
        "hosts the whole world inside this process over LocalCollective, "
        "which is the only distributed training this build performs "
        "(docs/LIGHTGBM_PARITY.md, tree_learner row). The two configurations "
        "cannot both be constructed in one script, so 'the same tree_learner "
        "on both sides' is not a thing this harness can build. " + extra + " "
        "A real differential for this row needs a launcher that starts N "
        "LightGBM processes with a shared machine list and N mojotrees "
        "ranks, which is a test harness rather than a case in this file, and "
        "it cannot flip the parity row on its own anyway. The row is partial "
        "because no transport ships, not because no number was measured."
    )


def case_tree_learner_feature(rep):
    _tree_learner_skip(
        rep,
        "feature",
        "What the mojotrees side claims, and what tests/"
        "test_distributed_strategies.mojo already checks without LightGBM, "
        "is that feature-parallel training over the in-process world equals "
        "serial training bit for bit. That claim is checkable at 0.0 "
        "tolerance and needs no second library.",
    )


def case_tree_learner_data(rep):
    _tree_learner_skip(
        rep,
        "data",
        "This mode also has no matching algorithm to compare against. "
        "mojotrees all-reduces the full histogram, which is more "
        "communication and exactly reproduces single-node training; "
        "LightGBM reduce-scatters histograms, all-gathers candidate splits, "
        "and fits bin edges from a distributed sample, so LightGBM's data "
        "parallel is not equal to its own serial training either "
        "(docs/distributed.md section 12). Equality across the two would be "
        "the wrong assertion even if both could be launched.",
    )


def case_tree_learner_voting(rep):
    _tree_learner_skip(
        rep,
        "voting",
        "And the vote itself is a different rule on purpose. mojotrees "
        "counts votes and breaks ties by ascending feature id; LightGBM "
        "aggregates local gains as well as counts and runs a second local "
        "pass, which elects a different feature and therefore grows a "
        "different tree (docs/DISTRIBUTED_STRATEGIES.md sections 2.2 and 8). "
        "Voting parallel is inexact by design on both sides, so the honest "
        "assertion would be a quality band and not equality.",
    )


def case_sparse_gpu(rep):
    mt = _mojotrees()
    lgb = _lightgbm()

    where = f"{platform.system()} {platform.machine()}"
    gpu = mt.gpu_available()
    rep.skip(
        "LightGBM has no GPU counterpart to run against on this machine. "
        "Its accelerated learners are device_type='gpu', which is OpenCL, "
        "and device_type='cuda', which is NVIDIA only; neither is built into "
        "the lightgbm wheels this environment installs, and neither targets "
        "Apple silicon at all, where mojotrees's GPU path is Metal. This "
        f"machine is {where}, LightGBM is {lgb.__version__}, and mojotrees "
        f"reports gpu_available()={gpu}. Beyond the platform, the sparse GPU "
        "trainer keeps a CSC binned matrix device-resident and recovers the "
        "implicit zeros by subtraction, which LightGBM's GPU learners have "
        "no analogue of, so even on a CUDA box the comparison would be "
        "between two different algorithms rather than two implementations of "
        "one. What the row actually claims today is a self-check: every "
        "row's device leaf equals the host walk of the grown tree, and the "
        "fit agrees with train_sparse at the dense GPU trainer's tolerance "
        "(tests/test_gpu_sparse.mojo). A LightGBM differential for it would "
        "need a Linux CUDA host with LightGBM built for device_type='cuda', "
        "and it would still be comparing accuracy regimes rather than "
        "numbers."
    )


# --------------------------------------------------------------------------
# Registry.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    name: str
    summary: str
    parity_rows: tuple
    runner: object


CASES = (
    Case(
        "linear_tree",
        "linear_tree=True against LightGBM's, on a piecewise-linear target",
        ("`linear_tree`",),
        case_linear_tree,
    ),
    Case(
        "linear_lambda",
        "the ridge penalty on the leaf coefficients, plus the intercept probe",
        ("`linear_lambda`", "`linear_tree`"),
        case_linear_lambda,
    ),
    Case(
        "dart",
        "boosting='dart' with LightGBM's weight-proportional drop rule",
        (
            "DART and random forest boosting",
            "`drop_rate` / `max_drop` / `skip_drop` / `xgboost_dart_mode` /"
            " `uniform_drop` / `drop_seed`",
        ),
        case_dart,
    ),
    Case(
        "dart_early_stopping",
        "early stopping under dart (SKIP: LightGBM's is inert)",
        ("DART and random forest boosting",),
        case_dart_early_stopping,
    ),
    Case(
        "rf",
        "boosting='rf', averaged trees at shrinkage 1",
        ("DART and random forest boosting", "`boosting`"),
        case_rf,
    ),
    Case(
        "tree_learner_feature",
        "tree_learner='feature' (SKIP: needs a multi-process world)",
        ("`tree_learner`", "feature parallel"),
        case_tree_learner_feature,
    ),
    Case(
        "tree_learner_data",
        "tree_learner='data' (SKIP: needs a multi-process world)",
        ("`tree_learner`",),
        case_tree_learner_data,
    ),
    Case(
        "tree_learner_voting",
        "tree_learner='voting' (SKIP: needs a multi-process world)",
        ("`tree_learner`", "voting parallel"),
        case_tree_learner_voting,
    ),
    Case(
        "label_gain",
        "a custom label_gain vector on both rankers",
        ("`label_gain`",),
        case_label_gain,
    ),
    Case(
        "lambdarank_position_bias",
        "unbiased LambdaRank with a position column",
        ("`lambdarank_position_bias_regularization`", "`Dataset.position`"),
        case_lambdarank_position_bias,
    ),
    Case(
        "model_edit_rollback",
        "rollback_one_iter truncation semantics on both sides",
        ("`Booster.rollback_one_iter` / `reset_parameter`",),
        case_model_edit_rollback,
    ),
    Case(
        "model_edit_leaf_output",
        "get/set_leaf_output, each library addressed by its own convention",
        (
            "`Booster.get_leaf_output` / `set_leaf_output` / `shuffle_models`"
            " / `refit`",
        ),
        case_model_edit_leaf_output,
    ),
    Case(
        "model_edit_shuffle",
        "shuffle_models leaves predictions invariant on both sides",
        (
            "`Booster.get_leaf_output` / `set_leaf_output` / `shuffle_models`"
            " / `refit`",
        ),
        case_model_edit_shuffle,
    ),
    Case(
        "model_edit_refit",
        "refit's decay blend, as a no-op and as a full rebuild",
        (
            "`Booster.get_leaf_output` / `set_leaf_output` / `shuffle_models`"
            " / `refit`",
            "`refit_decay_rate`",
        ),
        case_model_edit_refit,
    ),
    Case(
        "sparse_gpu",
        "sparse GPU training (SKIP: no LightGBM counterpart here)",
        ("Sparse GPU training", "Sparse input on the GPU"),
        case_sparse_gpu,
    ),
)

BY_NAME = {case.name: case for case in CASES}


# --------------------------------------------------------------------------
# Driver.
# --------------------------------------------------------------------------


def print_list():
    print("Cases in tools/lgbm_differential.py")
    print(f"seed {SEED}, {N_ROWS} rows, {N_FEATURES} features, "
          f"{N_QUERIES} queries for the ranking cases")
    for case in CASES:
        print()
        print(f"{case.name}")
        print(f"  {case.summary}")
        print(f"  parity rows: {', '.join(case.parity_rows)}")
        entries = tolerances_for(case.name)
        if not entries:
            print("  tolerance:   none, this case skips")
            continue
        for entry in entries:
            print(f"  tolerance:   {entry.name} = {entry.value:g} "
                  f"({entry.unit})")
            for line in _wrap(entry.reason, 66):
                print(f"               {line}")


def print_versions():
    lgb = _lightgbm()
    mt = _mojotrees()
    print(f"LightGBM  {lgb.__version__}")
    print(f"mojotrees {mt.__version__}")
    try:
        info = mt.build_info()
    except Exception:  # pragma: no cover - build_info is informational
        info = {}
    for key in ("commit", "git_commit", "build_type", "gpu_path_compiled_in"):
        if key in info:
            print(f"  {key}: {info[key]}")
    print(f"platform  {platform.system()} {platform.machine()}, "
          f"python {platform.python_version()}")
    print(f"seed      {SEED}")


def run(case):
    print()
    print(f"=== {case.name} " + "=" * max(0, 66 - len(case.name)))
    print(f"  {case.summary}")
    rep = Report(case.name)
    try:
        case.runner(rep)
    except SkipCase as reason:
        print("  SKIP")
        for line in _wrap(str(reason), 68):
            print(f"    {line}")
        return "skip", []
    return ("fail" if rep.failures else "pass"), rep.failures


def main():
    ap = argparse.ArgumentParser(
        description="LightGBM numeric differential for the parity rows that "
                    "have never had one."
    )
    ap.add_argument("--list", action="store_true",
                    help="print the cases and their tolerances, then exit; "
                         "imports neither library")
    ap.add_argument("--case", action="append", metavar="NAME",
                    help="run one case; repeat for several. Default is all")
    args = ap.parse_args()

    if args.list:
        print_list()
        return 0

    if args.case:
        unknown = [name for name in args.case if name not in BY_NAME]
        if unknown:
            raise SystemExit(
                f"unknown case(s): {', '.join(unknown)}\n"
                "run --list for the names"
            )
        selected = [BY_NAME[name] for name in args.case]
    else:
        selected = list(CASES)

    print_versions()

    results = {}
    failures = {}
    for case in selected:
        status, failed = run(case)
        results[case.name] = status
        if failed:
            failures[case.name] = failed

    print()
    print("=" * 72)
    for case in selected:
        print(f"  {results[case.name].upper():<5} {case.name}")
    n_fail = sum(1 for s in results.values() if s == "fail")
    n_skip = sum(1 for s in results.values() if s == "skip")
    n_pass = sum(1 for s in results.values() if s == "pass")
    print(f"  {n_pass} passed, {n_fail} failed, {n_skip} skipped")
    if failures:
        print()
        print("  checks that exceeded their tolerance:")
        for name, failed in failures.items():
            for label in failed:
                print(f"    {name}: {label}")
    if n_skip:
        print()
        print("  a skip is not a pass. Each skipped case printed why the "
              "comparison does not exist.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
