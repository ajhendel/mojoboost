"""Custom objectives.

A custom objective is a callable that receives the current raw predictions
and the labels and fills gradient and hessian arrays:

    def my_objective(
        raw: List[Float64],
        target: List[Float64],
        mut grad: List[Float64],
        mut hess: List[Float64],
    ) raises:
        ...

`GradHessFn` is that signature. It is a compile-time callable parameter, so
`train_custom[F: GradHessFn]` specializes on the concrete callable and the
per-round call is a direct, inlinable call rather than an indirect dispatch;
nothing crosses a dynamic boundary inside the boosting loop. Closures may
capture state (`{imm alpha}`) to carry hyperparameters.

The callback is invoked **once per boosting round** over the whole row set,
never per row. That is what makes a Python callback affordable (see
`bindings/_mojoboost.mojo` and `bench/bench_custom_objective.py`): the cost
is one call per round plus one pass over the rows, not one call per row.

Contract, and how it differs from LightGBM:

- Sample weights. The callback returns *unweighted* per-row derivatives and
  never sees the weights; `train_custom` multiplies both arrays by the row
  weight afterwards, exactly as the built-in objectives do
  (`grad[r] = w * grad[r]`). LightGBM instead hands the raw scores to the
  Python callback and expects the callback to fold the weights in itself.
  Ours is the intentional difference: it keeps a custom objective and the
  matching built-in bit-identical under weights, and makes a weight-unaware
  callback correct by construction.
- Validation. Every round the returned arrays are checked for length,
  non-finite values, and negative hessians (`check_custom_grad_hess`).
  LightGBM does not check, and silently trains on NaN. The check is one
  O(n_rows) pass per round, negligible beside histogram accumulation, and
  it is not optional.
- Base score. Custom objectives start from `base_score` (default 0.0)
  rather than from the label mean: the framework does not know the link
  function, so it cannot compute a sensible average start. LightGBM's
  `boost_from_average` likewise does not apply to custom objectives.
- Prediction scale. A model trained with a custom objective carries the
  `CUSTOM` objective code, and `predict`/`predict_row` return the raw
  score unchanged, since the framework does not know the inverse link.
  Apply your own link (e.g. a sigmoid) to the raw output. This matches
  LightGBM, where prediction after a custom objective is the raw score.
- Multiclass. Custom objectives are single-output only. LightGBM allows a
  custom multiclass objective returning an (n_rows * n_classes) gradient
  matrix; mojoboost does not, and `train_multiclass` has no custom-objective
  entry point. Passing `CUSTOM` to the built-in `train`/`train_gpu` raises
  and points at `train_custom`.

Early stopping is supported by `train_custom_with_valid`, which takes a
second callable of type `EvalLossFn` (raw scores and labels in, a scalar
lower-is-better loss out), because a custom objective carries no metric the
framework could infer.

What the registry says about `CUSTOM`
------------------------------------
`objective_registry.mojo` holds one record per objective, `CUSTOM` included,
and every clause of the contract above is one of its fields:

    objective_grad_hess_source(CUSTOM) == GRAD_CALLBACK
    objective_init_kind(CUSTOM)        == INIT_CALLER
    objective_link(CUSTOM)             == LINK_IDENTITY
    objective_gradients_on_device(CUSTOM) is False
    objective_class_weight_kind(CUSTOM) == CLASS_WEIGHT_NONE
    objective_has_default_metric(CUSTOM) is False

`_check_custom_contract` asserts all six once per training run, so the two
files cannot drift into disagreeing about what a custom objective is: if
someone gives `CUSTOM` a link in the registry, the trainer that returns raw
scores stops rather than returning a number that is no longer what the
registry promises. It is six integer compares before the first tree.

`matching_base_score` is the other direction: it hands a custom objective the
starting score a built-in one would have used, so "reproduce the built-in
SQUARED_ERROR exactly" does not require knowing that squared error starts
from the label mean and poisson from its logarithm. The registry says which
rule each objective follows and boosting.mojo computes it.
"""

from std.math import isfinite

from .binning import BinnedMatrix
from .boosting import (
    CUSTOM,
    Booster,
    BoosterParams,
    _base_score,
    _check_sample_weight,
)
from .objective_registry import (
    CLASS_WEIGHT_NONE,
    GRAD_CALLBACK,
    INIT_CALLER,
    LINK_IDENTITY,
    check_objective_param,
    objective_backends,
    objective_canonical_name,
    objective_class_weight_kind,
    objective_grad_hess_source,
    objective_gradients_on_device,
    objective_has_default_metric,
    objective_init_kind,
    objective_is_builtin,
    objective_link,
)
from .tree import Tree, grow_tree

comptime GradHessFn = def (
    List[Float64], List[Float64], mut List[Float64], mut List[Float64]
) raises -> None
"""Predictions and labels in, gradient and hessian arrays out."""

comptime EvalLossFn = def (List[Float64], List[Float64]) raises -> Float64
"""Raw predictions and labels in, a scalar lower-is-better loss out."""


def _check_custom_contract() raises:
    """Assert that the registry still describes `CUSTOM` the way this module
    trains it.

    Six compares, once per training run, before the first tree. Each one
    pairs a registry fact with the line of this file that depends on it, so a
    change to either side fails here instead of silently changing what a
    model means. See the module docstring for why each holds.
    """
    if objective_grad_hess_source(CUSTOM) != GRAD_CALLBACK:
        raise Error(
            "registry drift: objective_grad_hess_source(CUSTOM) is not"
            " GRAD_CALLBACK, but train_custom takes its derivatives from a"
            " caller-supplied callable"
        )
    if objective_init_kind(CUSTOM) != INIT_CALLER:
        raise Error(
            "registry drift: objective_init_kind(CUSTOM) is not INIT_CALLER,"
            " but train_custom starts from the caller's base_score"
        )
    if objective_link(CUSTOM) != LINK_IDENTITY:
        raise Error(
            "registry drift: objective_link(CUSTOM) is not LINK_IDENTITY,"
            " but a model trained here returns raw scores"
        )
    if objective_gradients_on_device(CUSTOM):
        raise Error(
            "registry drift: objective_gradients_on_device(CUSTOM) is True,"
            " but a custom objective's callback runs on the host"
        )
    if objective_class_weight_kind(CUSTOM) != CLASS_WEIGHT_NONE:
        raise Error(
            "registry drift: objective_class_weight_kind(CUSTOM) is not"
            " CLASS_WEIGHT_NONE, but a custom objective's labels are real"
            " numbers with no class to weight"
        )
    if objective_has_default_metric(CUSTOM):
        raise Error(
            "registry drift: objective_has_default_metric(CUSTOM) is True,"
            " but only a custom objective's author knows what it optimizes"
        )


def custom_objective_backends() raises -> Int:
    """Which backends can train a custom objective, as `SUPPORTS_*` flags.

    Both: `train_custom` here and `train_custom_gpu`, which grows the trees
    on the device and calls the gradient callback on the host. The device
    never computes the derivatives, which is a narrower claim and is
    `objective_gradients_on_device(CUSTOM)`, False. The registry is the one
    place that distinction is written down.
    """
    return objective_backends(CUSTOM)


def check_custom_base_score(base_score: Float64) raises:
    """A custom objective's starting raw score must be a finite number.

    `INIT_CALLER` means the framework picks nothing, so this is the only
    check there is: a NaN base score would make every gradient the callback
    ever sees a NaN, and the round-by-round validation would then blame the
    callback for a value the trainer handed it.
    """
    if not isfinite(base_score):
        raise Error("base_score must be finite")


def matching_base_score(
    objective: Int,
    target: List[Float64],
    weights: List[Float64] = [],
    alpha: Float64 = 0.9,
) raises -> Float64:
    """The `base_score` that starts a custom objective where the built-in
    `objective` would have started.

    A custom objective boosts from `base_score` because the framework does
    not know its link (see the module docstring). A caller reimplementing a
    built-in loss does know, and this saves them from restating the rule:
    `train_custom(..., base_score=matching_base_score(SQUARED_ERROR, y))`
    reproduces the built-in squared-error run, and swapping in `POISSON`
    reproduces the poisson one without the caller having to know that one
    starts from the label mean and the other from its logarithm.

    Computed by boosting.mojo's `_base_score`, the same call the built-in
    trainer makes, so the two are the same number and not two roundings of
    it. Only the built-in single-output objectives have one to match: the
    other three either have no framework-chosen start (`CUSTOM` itself),
    start at zero because the scale is arbitrary (`LAMBDARANK`), or need one
    score per class rather than one (`MULTICLASS`), and each is refused by
    name. `mean_label` remains the direct call for the weighted mean itself,
    which is this function under the identity link.
    """
    if not objective_is_builtin(objective):
        raise Error(
            "objective '",
            objective_canonical_name(objective),
            "' has no single base score a custom objective could match:"
            " init kind ",
            objective_init_kind(objective),
            " (see INIT_* in objective_registry.mojo)",
        )
    check_objective_param(objective, alpha)
    _check_sample_weight(weights, len(target))
    return _base_score(target, objective, weights, alpha)


def check_custom_grad_hess(
    grad: List[Float64], hess: List[Float64], n_rows: Int
) raises:
    """Validate one round of custom-objective output: one finite gradient
    and one finite, nonnegative hessian per row.

    A zero hessian is allowed (that is what a zero-weight row produces);
    a negative one is not, because the Newton leaf value and the split
    gain both assume a positive-semidefinite curvature.
    """
    if len(grad) != n_rows:
        raise Error(
            String(
                "custom objective produced ",
                len(grad),
                " gradients for ",
                n_rows,
                " rows",
            )
        )
    if len(hess) != n_rows:
        raise Error(
            String(
                "custom objective produced ",
                len(hess),
                " hessians for ",
                n_rows,
                " rows",
            )
        )
    for r in range(n_rows):
        if not isfinite(grad[r]):
            raise Error(
                String(
                    "custom objective produced a non-finite gradient at row ",
                    r,
                )
            )
        if not isfinite(hess[r]):
            raise Error(
                String(
                    "custom objective produced a non-finite hessian at row ",
                    r,
                )
            )
        if hess[r] < 0.0:
            raise Error(
                String(
                    "custom objective produced a negative hessian at row ", r
                )
            )


def _apply_sample_weight(
    mut grad: List[Float64], mut hess: List[Float64], weights: List[Float64]
):
    """Scale one round of gradients and hessians by the row weights, in the
    same operand order as the built-in objectives so the results are
    bit-identical."""
    if len(weights) == 0:
        return
    for r in range(len(weights)):
        grad[r] = weights[r] * grad[r]
        hess[r] = weights[r] * hess[r]


def squared_error_grad_hess(
    raw: List[Float64],
    target: List[Float64],
    mut grad: List[Float64],
    mut hess: List[Float64],
) raises:
    """Squared error as a custom objective: gradient `raw - target`,
    hessian 1. Written out as the reference example; with
    `base_score` set to the label mean it reproduces the built-in
    SQUARED_ERROR objective exactly (see tests/test_custom_objective.mojo).
    """
    grad.clear()
    hess.clear()
    for r in range(len(target)):
        grad.append(raw[r] - target[r])
        hess.append(1.0)


def squared_error_loss(
    raw: List[Float64], target: List[Float64]
) raises -> Float64:
    """Mean squared error, matching the built-in early-stopping metric for
    SQUARED_ERROR."""
    var total = 0.0
    for r in range(len(target)):
        var d = raw[r] - target[r]
        total += d * d
    return total / Float64(len(target))


def mean_label(
    target: List[Float64], weights: List[Float64]
) raises -> Float64:
    """The (weighted) label mean, the base score the built-in mean-link
    objectives boost from. Useful as an explicit `base_score` when a custom
    objective should match a built-in one."""
    var mean = 0.0
    var total_w = 0.0
    for r in range(len(target)):
        var w = weights[r] if len(weights) > 0 else 1.0
        mean += w * target[r]
        total_w += w
    if total_w <= 0.0:
        raise Error("sample_weight must have a positive sum")
    return mean / total_w


def apply_sample_weight(
    mut grad: List[Float64], mut hess: List[Float64], weights: List[Float64]
):
    """The public name for the one place a custom objective's derivatives
    meet the row weights.

    Called once per round by every custom-objective trainer here, in
    `train_custom_with_metrics`, and in `train_custom_gpu`. It is the only
    application: the callback never sees the weights, so it cannot apply them
    a second time, and a class weight is not a separate step either. Every
    class-level policy mojoboost has (`class_weight`, `balanced`,
    `scale_pos_weight`, `is_unbalance`) is expanded by class_weight.mojo into
    exactly this `sample_weight` vector *before* training starts, which is
    what makes double application impossible rather than merely discouraged.
    A row with `sample_weight` 2.0 in a class weighted 3.0 arrives here as
    one weight of 6.0.
    """
    _apply_sample_weight(grad, hess, weights)


def train_custom[F: GradHessFn](
    data: BinnedMatrix,
    target: List[Float64],
    grad_hess: F,
    params: BoosterParams,
    sample_weight: List[Float64] = [],
    base_score: Float64 = 0.0,
) raises -> Booster:
    """Train a boosted ensemble against a custom objective.

    Feature subsampling, interaction constraints, `max_depth`, and the
    regularization parameters all live in `params.tree` and apply here
    exactly as they do to the built-in objectives. Row-level sampling
    (bagging, GOSS) is a `train` parameter and is not wired through this
    entry point.

    `grad_hess` is called once per boosting round with the current raw
    scores and the labels and must fill `grad` and `hess` with one entry
    per row; see the module docstring for the full contract. Predictions
    from the returned booster are raw scores.
    """
    _check_custom_contract()
    if len(target) != data.n_rows:
        raise Error("target length must equal n_rows")
    _check_sample_weight(sample_weight, data.n_rows)
    check_custom_base_score(base_score)

    var n = data.n_rows
    var raw = List[Float64](capacity=n)
    for _ in range(n):
        raw.append(base_score)

    var trees = List[Tree]()
    var grad = List[Float64](capacity=n)
    var hess = List[Float64](capacity=n)
    for i in range(params.n_estimators):
        grad_hess(raw, target, grad, hess)
        check_custom_grad_hess(grad, hess, n)
        _apply_sample_weight(grad, hess, sample_weight)
        var tree = grow_tree(data, grad, hess, params.tree, [], i)

        # A single-leaf tree with a near-zero value means the objective has
        # converged; further rounds cannot make progress.
        if tree.n_leaves == 1 and abs(tree.value[0]) < 1e-12:
            break

        for r in range(n):
            raw[r] += params.learning_rate * tree.predict_row(data, r)
        trees.append(tree^)

    return Booster(
        trees^,
        base_score,
        params.learning_rate,
        CUSTOM,
        params.tree.monotone.copy(),
    )


def train_custom_with_valid[F: GradHessFn, L: EvalLossFn](
    data: BinnedMatrix,
    target: List[Float64],
    valid_data: BinnedMatrix,
    valid_target: List[Float64],
    grad_hess: F,
    eval_loss: L,
    params: BoosterParams,
    early_stopping_rounds: Int,
    min_delta: Float64 = 0.0,
    sample_weight: List[Float64] = [],
    base_score: Float64 = 0.0,
) raises -> Booster:
    """Train against a custom objective with validation-set early stopping.

    `eval_loss` scores the validation raw predictions against the validation
    labels; lower is better. Training stops when that loss has not improved
    by more than `min_delta` for `early_stopping_rounds` consecutive rounds,
    and the ensemble is truncated to its best round. As with the built-in
    trainer, sample_weight applies to training rows only.
    """
    _check_custom_contract()
    if len(target) != data.n_rows:
        raise Error("target length must equal n_rows")
    if len(valid_target) != valid_data.n_rows:
        raise Error("valid_target length must equal valid n_rows")
    if valid_data.n_features != data.n_features:
        raise Error("valid_data must have the same features")
    if early_stopping_rounds < 1:
        raise Error("early_stopping_rounds must be positive")
    _check_sample_weight(sample_weight, data.n_rows)
    check_custom_base_score(base_score)

    var n = data.n_rows
    var raw = List[Float64](capacity=n)
    for _ in range(n):
        raw.append(base_score)
    var valid_raw = List[Float64](capacity=valid_data.n_rows)
    for _ in range(valid_data.n_rows):
        valid_raw.append(base_score)

    var trees = List[Tree]()
    var grad = List[Float64](capacity=n)
    var hess = List[Float64](capacity=n)
    var best_loss = eval_loss(valid_raw, valid_target)
    var best_n_trees = 0
    for i in range(params.n_estimators):
        grad_hess(raw, target, grad, hess)
        check_custom_grad_hess(grad, hess, n)
        _apply_sample_weight(grad, hess, sample_weight)
        var tree = grow_tree(data, grad, hess, params.tree, [], i)
        if tree.n_leaves == 1 and abs(tree.value[0]) < 1e-12:
            break

        for r in range(n):
            raw[r] += params.learning_rate * tree.predict_row(data, r)
        for r in range(valid_data.n_rows):
            valid_raw[r] += (
                params.learning_rate * tree.predict_row(valid_data, r)
            )
        trees.append(tree^)

        var loss = eval_loss(valid_raw, valid_target)
        if loss < best_loss - min_delta:
            best_loss = loss
            best_n_trees = len(trees)
        elif len(trees) - best_n_trees >= early_stopping_rounds:
            break

    while len(trees) > best_n_trees:
        _ = trees.pop()
    return Booster(
        trees^,
        base_score,
        params.learning_rate,
        CUSTOM,
        params.tree.monotone.copy(),
    )
