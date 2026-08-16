"""CatBoost's `MultiRMSE` and `MultiRMSEWithMissingValues`.

Off by default and reachable only from this file; no existing default moves.
`docs/design/CATBOOST_CATALOG.md` A28 carries the source verification.

Read this before using any of it
--------------------------------
**The derivatives are trivial and are not where `MultiRMSE` lives.** For each
of `T` targets, `der[t] = w * (y[t] - p[t])` and `der2[t] = -w`
(`TMultiRMSEError::CalcDers`, hessian type `Diagonal`). There is no
cross-target term anywhere in the derivative. What couples the targets is the
**tree**, and that is verified from three places:

- `approx_dimension.cpp`: a multi-target objective's approx dimension is the
  target dimension `T`.
- `model.h:118`: leaf values are laid out
  `[treeIndex][leafId * ApproxDimension + dimension]` -- **one** tree per
  iteration, with a **vector** leaf value.
- `scoring.cpp:751-766`: `for (int dim ...) UpdateScores(stats + dim*stride,
  ..., scoreCalcer)` into **one** score accumulator whose `AddLeaf` does
  `Scores[splitIdx] += ...`. The split score of a candidate is the **sum over
  targets** of the per-target score.

So CatBoost picks one structure per round by the summed score, then solves a
separate leaf value per target from the diagonal hessian.

**The consequence, stated plainly.** Because the derivative has no
cross-target coupling, the shared structure is the *only* thing that couples
the targets. Grow one tree per target per round -- the shape
`boosting.train_multiclass` uses and the only shape mojotrees can carry today
-- and the result is **bit-identical to `T` independent `SQUARED_ERROR`
boosters**. `train_multi_rmse` below is that shape. It is a useful
multi-output regression API and it is the consumer that gives
`target_matrix.TargetMatrix` a reason to exist, but it is **not CatBoost's
`MultiRMSE`**, and `bench/real_data/scenarios.py`'s
`CATBOOST_UNMATCHABLE["multiclass_tree_count"]` applies here with more force
than it does to multiclass: for multiclass the tree count is a bookkeeping
difference, for `MultiRMSE` the tree shape is the model.

What is owed by the tree lanes, and what is provided here for them
------------------------------------------------------------------
`multi_target_split_gain` and `multi_target_leaf_values` are the two formulas
a shared-structure grower needs, written and tested here so that whoever owns
`tree.mojo` and `histogram.mojo` calls a checked function rather than
rederiving them. The easy wrong answer is worth naming: **the gain of the
summed planes is not the sum of the per-plane gains**, and CatBoost computes
the second. `sum_d T(G_d)^2/(H_d+l2)` and `T(sum_d G_d)^2/(sum_d H_d + l2)`
differ whenever the targets disagree, which is the entire point of a
multi-target fit.

Layout
------
Everything multi is flat row-major with stride `n_targets`:
`grad[r * T + t]`, `hess[r * T + t]`, `raw[r * T + t]`, and
`TargetMatrix.values[r * T + t]`. That is `train_multiclass`'s existing
`raw[r * n_classes + k]` convention, not a new one.

Determinism
-----------
Every loop here is per-row and per-target with no cross-row reads and no
cross-row summation, so splitting it across contiguous row blocks is
bit-identical at every `MOJOTREES_NUM_WORKERS`. The metric's two accumulators
are plain sums in a fixed order (target-major then row, exactly CatBoost's
loop nesting) and are not split.
"""

from std.math import isfinite, isnan, sqrt

from .binning import BinMapper, BinnedMatrix, fit_bins
from .boosting import BoosterParams
from .gain import leaf_score, soft_threshold_l1
from .histogram import (
    check_derivative_precision,
    derivative,
    derivative_precision_narrows,
)
from .monotone import MonotoneConstraints
from .target_matrix import TargetMatrix
from .tree import Tree, grow_tree

# ---------------------------------------------------------------------------
# Proposed objective codes
# ---------------------------------------------------------------------------
# Negative, following `MULTICLASS = -1`'s stated rule: "negative to stay out
# of the single-output code space forever", because a multi-output model is
# not trained by `boosting.train` and does not hold one tree per round. These
# belong in `objective_registry.mojo`, which is glue this lane does not own;
# the lane report carries the literal diff.
comptime MULTI_RMSE = -2
comptime MULTI_RMSE_WITH_MISSING = -3


def multi_target_varies_hessian(
    weighted: Bool, with_missing_values: Bool
) -> Bool:
    """Whether a `MultiRMSE` fit has a per-row hessian, so
    `histogram.CONSTANT_HESSIAN` must not be declared for it.

    Unlike Cox and `SurvivalAft`, this one is **conditional**, and the
    condition is exactly the two ways the hessian stops being the literal 1.0:

    - `weighted`: `der2[t] = -w`, so our hessian is the row weight. That is
      the same argument `sampling.mvs_varies_hessian` makes about a sampling
      weight, and it is the ordinary sample-weight case rather than anything
      new.
    - `with_missing_values`: `TMultiRMSEErrorWithMissingValues` writes 0 into
      both derivatives at a NaN target, so a row's hessian is 0 in the target
      planes it is missing and 1 in the others. A hessian plane rebuilt from
      the row count would count those rows.

    Neither is a knob about a `Float64` value: both are structural facts of
    the fit known before the first tree, which is the property
    `mvs_varies_hessian`'s docstring requires of a declaration held for a
    whole fit and set once on the device.
    """
    return weighted or with_missing_values


def check_multi_target_hessian_declaration(
    weighted: Bool, with_missing_values: Bool, const_hessian: Bool
) raises:
    """Refuse a constant-hessian declaration beside a fit whose hessian
    varies. `sampling.check_mvs_hessian_declaration`'s shape."""
    if multi_target_varies_hessian(weighted, with_missing_values) and (
        const_hessian
    ):
        raise Error(
            "a weighted or missing-value MultiRMSE fit must not declare a"
            " constant hessian: the per-row hessian is the row weight, and is"
            " zero in a target plane the row is missing"
        )


# ---------------------------------------------------------------------------
# Derivatives
# ---------------------------------------------------------------------------


def multi_rmse_grad_hess_into[
    NARROW: Bool, MISSING: Bool
](
    raw: List[Float64],
    targets: TargetMatrix,
    weights: List[Float64],
    mut grad: List[Float64],
    mut hess: List[Float64],
) raises:
    """`TMultiRMSEError::CalcDers`, negated into loss sign, for a whole round.

    CatBoost (`error_functions.h:191-218`):

        for i in dim: der[i]  = weight * (target[i] - approx[i])
                      der2[i] = -weight

    and it stores `Der1 = -dL/da`, `Der2 = -d2L/da2`, so ours are
    `grad = w * (p - y)` and `hess = w`. `MISSING` selects
    `TMultiRMSEErrorWithMissingValues`, which is the same function with both
    derivatives zeroed at a NaN target -- one compile-time branch, so the
    default path carries no test.

    `grad` and `hess` are flat `n_rows * n_targets`, row-major, and are
    resized and fully overwritten.
    """
    var n = targets.n_rows
    var t_count = targets.n_targets
    var total = n * t_count
    if len(raw) < total:
        raise Error(
            "raw scores must hold n_rows * n_targets entries, row-major"
        )
    if len(weights) > 0 and len(weights) < n:
        raise Error("sample_weight must be at least as long as the rows")
    if len(grad) != total:
        grad.resize(total, 0.0)
    if len(hess) != total:
        hess.resize(total, 0.0)

    var weighted = len(weights) > 0
    for r in range(n):
        var w = weights[r] if weighted else 1.0
        var base = r * t_count
        for t in range(t_count):
            var y = targets.values[base + t]

            comptime if MISSING:
                if isnan(y):
                    grad[base + t] = 0.0
                    hess[base + t] = 0.0
                    continue
            var g = w * (raw[base + t] - y)
            grad[base + t] = derivative[NARROW](g)
            hess[base + t] = derivative[NARROW](w)


def multi_rmse_grad_hess(
    raw: List[Float64],
    targets: TargetMatrix,
    weights: List[Float64],
    mut grad: List[Float64],
    mut hess: List[Float64],
    with_missing_values: Bool = False,
    float64_derivatives: Bool = False,
) raises:
    """`multi_rmse_grad_hess_into` with `derivative_precision` resolved on
    `boosting.fill_grad_hess`'s rule and precedence, and the missing-value
    arm selected."""
    check_derivative_precision()
    var narrow = derivative_precision_narrows() and not float64_derivatives
    if narrow:
        if with_missing_values:
            multi_rmse_grad_hess_into[True, True](
                raw, targets, weights, grad, hess
            )
        else:
            multi_rmse_grad_hess_into[True, False](
                raw, targets, weights, grad, hess
            )
    else:
        if with_missing_values:
            multi_rmse_grad_hess_into[False, True](
                raw, targets, weights, grad, hess
            )
        else:
            multi_rmse_grad_hess_into[False, False](
                raw, targets, weights, grad, hess
            )


def multi_rmse(
    raw: List[Float64],
    targets: TargetMatrix,
    weights: List[Float64] = [],
    with_missing_values: Bool = False,
) raises -> Float64:
    """CatBoost's `MultiRMSE` eval metric. **Lower is better**, best 0.

        sqrt( sum over t of sum over r of w_r (p - y)^2  /  sum over r of w_r )

    The denominator is the **row**-weight sum and is deliberately not
    multiplied by `T` (`TMultiRMSEMetric::EvalSingleThread` accumulates
    `Stats[1]` in a separate loop over rows only). So `MultiRMSE` is the
    root-mean-square Euclidean norm of the residual *vector*, not the mean of
    the per-target RMSEs, and it is larger than either by roughly `sqrt(T)`.
    At `T = 1` it is exactly RMSE.

    `with_missing_values` selects `TMultiRMSEWithMissingValues`, whose
    denominator is the count of **present** target cells rather than the row
    weight sum -- a different normalization, which is why it is a separate
    registered metric in CatBoost and a flag rather than a silent NaN skip
    here.

    The loop nesting is CatBoost's, target-major then row, so the summation
    order matches.
    """
    var n = targets.n_rows
    var t_count = targets.n_targets
    if len(raw) < n * t_count:
        raise Error(
            "raw scores must hold n_rows * n_targets entries, row-major"
        )
    if len(weights) > 0 and len(weights) < n:
        raise Error("weight must be at least as long as the rows")

    var weighted = len(weights) > 0
    var total_error = 0.0
    var total_weight = 0.0
    for t in range(t_count):
        for r in range(n):
            var y = targets.values[r * t_count + t]
            if with_missing_values and isnan(y):
                continue
            var w = weights[r] if weighted else 1.0
            var d = raw[r * t_count + t] - y
            total_error += w * d * d
            if with_missing_values:
                total_weight += w
    if not with_missing_values:
        for r in range(n):
            total_weight += weights[r] if weighted else 1.0
    if total_weight == 0.0:
        return 0.0
    return sqrt(total_error / total_weight)


# ---------------------------------------------------------------------------
# What a shared-structure grower needs. Not called by anything here.
# ---------------------------------------------------------------------------


def multi_target_split_gain(
    left_g: List[Float64],
    left_h: List[Float64],
    right_g: List[Float64],
    right_h: List[Float64],
    parent_g: List[Float64],
    parent_h: List[Float64],
    lambda_l1: Float64,
    lambda_l2: Float64,
) raises -> Float64:
    """The split gain of one candidate under a `T`-plane objective.

    **The sum over targets of the per-target gain**, which is CatBoost's
    `for (int dim ...) UpdateScores(..., scoreCalcer)` into a single
    accumulator (`scoring.cpp:751-766`, `score_calcers.h`
    `Scores[splitIdx] += ...`).

    The wrong answer this function exists to prevent: **the gain of the summed
    planes**, `T(sum_d G_d)^2 / (sum_d H_d + l2)`. That is a different number
    whenever the targets disagree about a split, which is the entire content
    of a multi-target fit, and it is what you get for free if you hand a
    single-plane grower the elementwise sum of the planes. The two agree only
    when every target's gradient is a fixed multiple of every other's.

    Each argument is one entry per target. `lambda_l2` is applied per plane,
    as CatBoost's `scaledL2Regularizer` is: it is set once per body-tail and
    used inside the per-dimension loop, not once against a summed hessian.
    """
    var t_count = len(left_g)
    if (
        len(left_h) != t_count
        or len(right_g) != t_count
        or len(right_h) != t_count
        or len(parent_g) != t_count
        or len(parent_h) != t_count
    ):
        raise Error("every plane list must have one entry per target")
    var total = 0.0
    for t in range(t_count):
        total += (
            leaf_score(left_g[t], left_h[t], lambda_l1, lambda_l2)
            + leaf_score(right_g[t], right_h[t], lambda_l1, lambda_l2)
            - leaf_score(parent_g[t], parent_h[t], lambda_l1, lambda_l2)
        )
    return total


def multi_target_leaf_values(
    sum_g: List[Float64],
    sum_h: List[Float64],
    lambda_l1: Float64,
    lambda_l2: Float64,
) raises -> List[Float64]:
    """The vector leaf value: one Newton step per target from the diagonal
    hessian.

    CatBoost's `TDiagonalHessian::SolveNewtonEquation` is
    `res[dim] = negativeDer[dim] / (Data[dim] - l2Regularizer)` where
    `negativeDer = -SumDer` and `Data = SumDer2`; with `SumDer = -sum(grad)`
    and `SumDer2 = -sum(hess)` in our signs that is
    `-sum(grad) / (sum(hess) + l2)`, which is `tree._leaf_value`'s Newton step
    per plane. There is no cross-target term: the hessian is declared
    `Diagonal` by `TMultiRMSEError`'s constructor, so the `T` by `T` system
    never exists.

    `lambda_l1` is ours, not CatBoost's -- CatBoost has no L1 on leaf values
    (`PARAMETER_NAMING.md`: `reg_alpha` has no CatBoost column) -- and at its
    default 0 `soft_threshold_l1` returns its argument unchanged, so the
    CatBoost-matching path costs one compare per target.
    """
    var t_count = len(sum_g)
    if len(sum_h) != t_count:
        raise Error("sum_g and sum_h must have one entry per target")
    var out = List[Float64](capacity=t_count)
    for t in range(t_count):
        out.append(
            -soft_threshold_l1(sum_g[t], lambda_l1) / (sum_h[t] + lambda_l2)
        )
    return out^


# ---------------------------------------------------------------------------
# The shape our machinery can carry today
# ---------------------------------------------------------------------------


struct MultiTargetBooster(Copyable, Movable):
    """A fitted multi-output ensemble, one tree per target per round.

    `trees[i * n_targets + t]` is round `i`'s tree for target `t`, the same
    indexing `MulticlassBooster` uses for its per-class trees. `n_iterations`
    is `len(trees) // n_targets` and is the number a comparison against
    CatBoost should quote, because CatBoost's tree count for the same fit is
    that number and not this one.
    """

    var trees: List[Tree]
    var base_scores: List[Float64]
    var n_targets: Int
    var learning_rate: Float64
    var monotone: MonotoneConstraints

    def __init__(
        out self,
        var trees: List[Tree],
        var base_scores: List[Float64],
        n_targets: Int,
        learning_rate: Float64,
        var monotone: MonotoneConstraints = MonotoneConstraints(),
    ):
        self.trees = trees^
        self.base_scores = base_scores^
        self.n_targets = n_targets
        self.learning_rate = learning_rate
        self.monotone = monotone^

    def n_iterations(self) -> Int:
        """Boosting iterations, which is `len(trees) // n_targets`. **This is
        the number comparable to CatBoost's `tree_count_`**, not
        `len(self.trees)`."""
        return len(self.trees) // self.n_targets

    def predict_row(self, data: BinnedMatrix, row: Int) raises -> List[Float64]:
        """The `T` predictions for one row."""
        var out = List[Float64](capacity=self.n_targets)
        for t in range(self.n_targets):
            out.append(self.base_scores[t])
        for i in range(len(self.trees)):
            out[i % self.n_targets] += (
                self.learning_rate * self.trees[i].predict_row(data, row)
            )
        return out^


def train_multi_rmse(
    data: BinnedMatrix,
    targets: TargetMatrix,
    params: BoosterParams,
    sample_weight: List[Float64] = [],
    with_missing_values: Bool = False,
) raises -> MultiTargetBooster:
    """Train a multi-output regression ensemble, **one tree per target per
    round**.

    **This is not CatBoost's `MultiRMSE`** and the difference is not
    cosmetic. CatBoost grows one tree per round whose structure is chosen by
    the summed-over-targets split score and whose leaf value is a vector;
    this grows `T` independent structures. Because `MultiRMSE`'s derivative
    has no cross-target term, the result here is bit-identical to `T`
    separate `SQUARED_ERROR` boosters, and the shared structure CatBoost uses
    is the whole of what `MultiRMSE` adds over that. See the module docstring
    and catalog A28.

    It is here because it is what mojotrees's grower can carry today, it is a
    real multi-output regression API, and it is the working consumer that
    exercises the `TargetMatrix` input path end to end.

    Base scores are the per-target weighted label means, which is
    `SQUARED_ERROR`'s `INIT_LINK_MEAN` applied per plane.
    `catboost_options.cpp:703-710` lists `MultiRMSE` and
    `MultiRMSEWithMissingValues` among the losses `boost_from_average` is
    allowed for, so a mean start is the CatBoost-shaped choice as well as
    ours; a missing cell is skipped in its own target's mean.
    """
    targets.check_rows(data.n_rows)
    if not with_missing_values:
        targets.check_finite()
    if len(sample_weight) > 0 and len(sample_weight) != data.n_rows:
        raise Error("sample_weight length must equal n_rows")
    params.tree.monotone.check_features(data.n_features)

    var n = data.n_rows
    var t_count = targets.n_targets
    var weighted = len(sample_weight) > 0

    var base_scores = List[Float64](capacity=t_count)
    for t in range(t_count):
        var s = 0.0
        var w_sum = 0.0
        for r in range(n):
            var y = targets.values[r * t_count + t]
            if with_missing_values and isnan(y):
                continue
            var w = sample_weight[r] if weighted else 1.0
            s += w * y
            w_sum += w
        if not (w_sum > 0.0):
            raise Error(
                String(
                    "target ",
                    t,
                    " has no rows with positive weight, so it has no mean to"
                    " boost from",
                )
            )
        base_scores.append(s / w_sum)

    var raw = List[Float64](capacity=n * t_count)
    for _ in range(n):
        for t in range(t_count):
            raw.append(base_scores[t])

    var trees = List[Tree]()
    var grad = List[Float64](capacity=n * t_count)
    var hess = List[Float64](capacity=n * t_count)
    var plane_g = List[Float64](capacity=n)
    var plane_h = List[Float64](capacity=n)
    for _ in range(n):
        plane_g.append(0.0)
        plane_h.append(0.0)

    for i in range(params.n_estimators):
        multi_rmse_grad_hess(
            raw,
            targets,
            sample_weight,
            grad,
            hess,
            with_missing_values,
        )
        for t in range(t_count):
            for r in range(n):
                plane_g[r] = grad[r * t_count + t]
                plane_h[r] = hess[r * t_count + t]
            # The tree index is (round, target) flattened, so two targets in
            # one round draw different feature subsamples and different
            # random-strength noise -- the same rule `train_multiclass` uses
            # for its per-class trees.
            var tree = grow_tree(
                data, plane_g, plane_h, params.tree, [], i * t_count + t
            )
            for r in range(n):
                raw[r * t_count + t] += (
                    params.learning_rate * tree.predict_row(data, r)
                )
            trees.append(tree^)

    return MultiTargetBooster(
        trees^,
        base_scores^,
        t_count,
        params.learning_rate,
        params.tree.monotone.copy(),
    )


# ---------------------------------------------------------------------------
# The reachable entry point: raw matrix in, fitted multi-output model out
# ---------------------------------------------------------------------------
#
# `train_multi_rmse` above takes a `BinnedMatrix`, which is a thing only the
# native API can build. Everything below exists so that a Python caller can
# hand over the same column-major float64 matrix every other fit takes, plus
# a 2-D target, and get a model back that predicts. Without it the objective
# is built, tested, and reachable by nobody; see catalog A31.
#
# The tree shape has NOT changed and the honest statement in this module's
# docstring stands: this is `T` independent squared-error boosters and it is
# not CatBoost's `MultiRMSE`. Making it reachable does not make it CatBoost.


struct MultiTargetModel(Copyable, Movable):
    """A fitted multi-output ensemble plus the bin edges it was fitted on.

    The same pairing `model.Model` is (`mapper` + `booster`), and for the
    same reason: the trees index bins, so a raw row cannot be scored without
    the mapper that produced them. It is a separate struct rather than a
    field on `Model` because `Model.booster` is a single-output `Booster` and
    widening it is `model.mojo`'s decision, not this module's.
    """

    var mapper: BinMapper
    var booster: MultiTargetBooster

    def __init__(
        out self, var mapper: BinMapper, var booster: MultiTargetBooster
    ):
        self.mapper = mapper^
        self.booster = booster^

    def n_targets(self) -> Int:
        return self.booster.n_targets

    def n_features(self) -> Int:
        return self.mapper.n_features

    def n_iterations(self) -> Int:
        """Boosting rounds, which is `len(trees) // n_targets`. The number
        comparable to CatBoost's `tree_count_`."""
        return self.booster.n_iterations()

    def predict(self, row: List[Float64]) raises -> List[Float64]:
        """The `T` raw predictions for one raw example of length
        `n_features`.

        There is no response transform: `MultiRMSE` is a squared-error
        family and its link is the identity, so raw and response are the
        same number and no caller has to choose between them.
        """
        var bins = self.mapper.bin_row(row)
        var t_count = self.booster.n_targets
        var out = List[Float64](capacity=t_count)
        for t in range(t_count):
            out.append(self.booster.base_scores[t])
        for i in range(len(self.booster.trees)):
            out[i % t_count] += (
                self.booster.learning_rate
                * self.booster.trees[i].predict_bins(bins)
            )
        return out^


def fit_multi_rmse[
    features_origin: ImmOrigin, //
](
    features: Span[Float64, features_origin],
    n_rows: Int,
    n_features: Int,
    targets: TargetMatrix,
    params: BoosterParams,
    max_bins: Int = 255,
    sample_weight: List[Float64] = [],
    with_missing_values: Bool = False,
    use_missing: Bool = True,
    categorical_features: List[Int] = [],
) raises -> MultiTargetModel:
    """Fit `MultiRMSE` on a column-major raw feature matrix
    (`features[f * n_rows + r]`) and a `TargetMatrix` of `n_rows * T`.

    The binning half of `model.fit`, verbatim in shape, followed by
    `train_multi_rmse`. It is CPU only and says so rather than resolving a
    device: `train_gpu` has no multi-output round loop, so `device='gpu'` is
    refused at the boundary above instead of being silently downgraded.

    `with_missing_values` selects `MultiRMSEWithMissingValues`: a `NaN` in
    the TARGET is a row that target does not train on, which is a different
    thing from a `NaN` in `features`, where it is the missing-value marker
    that `use_missing` governs.
    """
    if targets.n_targets < 1:
        raise Error("fit_multi_rmse: n_targets must be at least 1")
    targets.check_rows(n_rows)
    var mapper = fit_bins(
        features,
        n_rows,
        n_features,
        max_bins,
        use_missing=use_missing,
        categorical_features=categorical_features,
    )
    var data = mapper.transform(features, n_rows)
    var booster = train_multi_rmse(
        data, targets, params, sample_weight, with_missing_values
    )
    return MultiTargetModel(mapper^, booster^)


def predict_multi_rmse[
    features_origin: ImmOrigin, //
](
    model: MultiTargetModel,
    features: Span[Float64, features_origin],
    n_rows: Int,
    n_features: Int,
) raises -> List[Float64]:
    """Score a column-major raw matrix, returning `n_rows * T` row-major:
    `out[r * T + t]`.

    Row-major out because that is the layout `TargetMatrix` uses on the way
    in, and a caller that hands over a `(n, T)` array should get one back
    without transposing.
    """
    if n_features != model.mapper.n_features:
        raise Error(
            String(
                "predict_multi_rmse: the model was fitted on ",
                model.mapper.n_features,
                " features and was handed ",
                n_features,
            )
        )
    if n_rows < 0:
        raise Error("predict_multi_rmse: n_rows must not be negative")
    var t_count = model.booster.n_targets
    var out = List[Float64](capacity=n_rows * t_count)
    var row = List[Float64](capacity=n_features)
    for _ in range(n_features):
        row.append(0.0)
    for r in range(n_rows):
        for f in range(n_features):
            row[f] = features[f * n_rows + r]
        var scores = model.predict(row)
        for t in range(t_count):
            out.append(scores[t])
    return out^
