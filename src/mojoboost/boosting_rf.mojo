"""Random-forest mode: LightGBM's `boosting='rf'`.

What this is, and what it reuses
--------------------------------
Random-forest mode is not a second trainer. It is a different outer loop
around the same pieces the production trainer uses: `tree.grow_tree` grows
every tree, `boosting._fill_grad_hess` produces the gradients,
`boosting._base_score` produces the offset, `boosting._renew_leaf_values`
renews the leaves of the objectives that renew, `bagging.refresh_bag` draws
the rows, and the result is an ordinary `boosting.Booster` holding ordinary
`tree.Tree` values. Nothing here is a parallel model representation and
nothing here re-derives an objective, a gain, a split, or a leaf value.

The one thing that differs from GBDT is what the trees mean together. In
GBDT tree `i + 1` is fitted to the residual left by trees `0..i`, and the
ensemble is a sum. In a random forest every tree is fitted to the *same*
gradients, the ones taken at the constant offset the model boosts from, and
the ensemble is an average. LightGBM calls that `average_output` and
implements it as a flag on the model; mojoboost's `Booster` already has a
single shrinkage factor applied to every tree, so an average of `K` trees is
exactly the ensemble whose factor is `1 / K`. That is the whole
representation: no new field, no format version, and `serialize.save_model`
writes an RF model today with no change at all.

Consequences of that choice, spelled out
----------------------------------------
- The gradients do not change between rounds, so they are computed once and
  every tree is grown from the same two arrays. What makes the trees differ
  is the row bag and the feature sample, which is why bagging is required
  rather than optional (below).
- The raw training score is never updated. A tree is never told what the
  trees before it did, which is the definition of a forest.
- `learning_rate` takes no part. LightGBM silently overrides it to 1;
  mojoboost refuses a rate other than 1.0 instead, following the rule
  `params._check_alpha_key` states for a parameter that does not apply to
  the configuration the caller asked for: a number the caller set that the
  run does not use hides a real mistake.
- The returned `Booster.learning_rate` is `1 / K` over the trees actually
  kept, not over `n_estimators`. A round whose tree came out degenerate is
  skipped (see below), so those two can differ, and averaging over rounds
  that contributed nothing would shrink the model toward the base score.

INTENTIONAL DIFFERENCES FROM LightGBM
-------------------------------------
- LightGBM requires `bagging_freq > 0` and a `bagging_fraction` below 1 for
  `rf` and checks it at configuration time. So does this, for the same
  reason: with every tree grown on every row from identical gradients,
  every tree in the forest is the same tree, and the average of `K` copies
  of one tree is that tree. The check is here rather than deferred to a
  degenerate result.
- A round that produces a single-leaf tree with a near-zero value is
  skipped rather than ending training. In GBDT such a tree means the
  objective has converged and later rounds cannot help; under a forest it
  only means this bag had nothing to give, and the next bag is independent
  of it. This mirrors what `boosting._boost_rounds` already does whenever a
  row sampler is active.
- `feature_fraction` is not required. LightGBM's `rf` checks only that it is
  at most 1. A forest of bagged-but-not-feature-sampled trees is a
  legitimate (if more correlated) model, and `tree.grow_tree` already
  refuses an out-of-range fraction.

WHAT IS NOT SUPPORTED HERE, AND WHY
-----------------------------------
- Multiclass. `boosting.MulticlassBooster` carries one shrinkage factor for
  the whole model as well, so the same `1 / K` representation would work,
  but the round loop has to grow one tree per class per round and share one
  bag across them. `train_rf` is single-output and says so; the multiclass
  entry point is listed in `handoffs/connect_17_alternate_boosting.md`.
- Ranking. `ranking.train_ranker` computes its gradients from query groups
  once per round against the current scores. A forest never updates its
  scores, so lambdarank's gradients would be identical in every round and
  the "forest" would be one tree repeated. That is not a missing wiring, it
  is a statement about the objective, so ranking is refused by name.
- GOSS. GOSS samples by gradient magnitude, and the gradients here never
  change, so every round would draw the identical sample. `train_rf` takes
  no `GossParams` and `alternate_boosting.train_boosting` refuses the
  combination.
- Custom objectives. `boosting._check_objective` already refuses `CUSTOM`
  for the built-in trainer, and this entry point inherits that check.
"""

from .bagging import (
    BaggingParams,
    bagging_enabled,
    check_bagging,
    refresh_bag,
)
from .binning import BinnedMatrix
from .boosting import (
    Booster,
    BoosterParams,
    _base_score,
    _check_objective,
    _check_sample_weight,
    _fill_grad_hess,
    _renew_leaf_values,
    objective_renews_leaves,
    renewal_alpha,
    renewal_weights,
)
from .ranking import LAMBDARANK
from .tree import Tree, grow_tree


def check_rf_params(params: BoosterParams, bagging: BaggingParams) raises:
    """Reject a configuration random-forest mode cannot honor.

    Called before any data is read, so a caller learns the combination is
    wrong without paying for a training run. The two rules are the ones the
    module docstring justifies: the shrinkage factor is the model's own
    `1 / K` and so cannot also be a user rate, and a forest without row
    bagging is one tree copied `n_estimators` times.
    """
    check_bagging(bagging)
    if params.learning_rate != 1.0:
        raise Error(
            "boosting='rf' averages its trees, so learning_rate takes no"
            " part; pass learning_rate=1.0 rather than leaving a rate set"
            " that the run will not use"
        )
    if not bagging_enabled(bagging):
        raise Error(
            "boosting='rf' needs row bagging (bagging_freq > 0 and"
            " bagging_fraction < 1); without it every tree is grown on every"
            " row from identical gradients, so the forest is one tree"
            " repeated"
        )
    if params.n_estimators < 0:
        raise Error("n_estimators must not be negative")


def _check_rf_objective(objective: Int) raises:
    """Refuse the objectives whose gradients are not a function of the raw
    scores alone, which is what lets a forest compute them once."""
    if objective == LAMBDARANK:
        raise Error(
            "boosting='rf' does not support 'lambdarank'; its gradients come"
            " from the ranking of the current scores, and a forest never"
            " updates its scores, so every tree would see one fixed gradient"
            " vector and the forest would be one tree repeated"
        )


def _forest_rate(n_trees: Int) -> Float64:
    """The shrinkage factor that turns a sum over trees into an average.

    An empty forest keeps 1.0: it predicts the base score alone, and the
    factor multiplies nothing.
    """
    if n_trees <= 0:
        return 1.0
    return 1.0 / Float64(n_trees)


def _grow_forest(
    data: BinnedMatrix,
    target: List[Float64],
    objective: Int,
    params: BoosterParams,
    sample_weight: List[Float64],
    alpha: Float64,
    bagging: BaggingParams,
    raw: List[Float64],
    round_offset: Int,
    mut trees: List[Tree],
) raises:
    """Grow `params.n_estimators` trees, all against the fixed scores `raw`,
    appending the ones that carry information to `trees`.

    `raw` is immutable here, which is the whole difference from
    `boosting._boost_rounds`: gradients are filled once, before the loop, and
    every round reuses them. `round_offset` is the number of trees already in
    the forest, so a continued run draws the bags and the feature samples it
    would have drawn had the whole forest been grown in one call.
    """
    var n = data.n_rows
    var grad = List[Float64](capacity=n)
    var hess = List[Float64](capacity=n)
    _fill_grad_hess(raw, target, objective, sample_weight, alpha, grad, hess)

    var signs = params.tree.monotone.active_signs()
    var renews = objective_renews_leaves(objective)
    var renew_w = renewal_weights(objective, target, sample_weight)
    var renew_a = renewal_alpha(objective, alpha)
    var bag = List[Int]()
    for i in range(params.n_estimators):
        var round = round_offset + i
        refresh_bag(bag, bagging, n, round)
        var tree = grow_tree(data, grad, hess, params.tree, bag, round)
        if renews:
            _renew_leaf_values(
                tree, data, target, raw, renew_w, renew_a, bag, signs,
                params.tree.extra,
            )
        # This bag had nothing to give. The next one is drawn independently
        # of it, so the round is skipped rather than ending the forest.
        if tree.n_leaves == 1 and abs(tree.value[0]) < 1e-12:
            continue
        trees.append(tree^)


def train_rf(
    data: BinnedMatrix,
    target: List[Float64],
    objective: Int,
    params: BoosterParams,
    sample_weight: List[Float64] = [],
    alpha: Float64 = 0.9,
    bagging: BaggingParams = BaggingParams.disabled(),
    init_score: List[Float64] = [],
) raises -> Booster:
    """Train a random forest, LightGBM's `boosting='rf'`.

    Every argument carries the meaning it has in `boosting.train`, and the
    returned `Booster` is an ordinary one: it predicts
    `base_score + (1 / K) * sum(tree)` through the same
    `Booster.predict_raw_row` every other model uses, serializes through
    `serialize.save_model` with no format change, and can be handed to
    `contrib`, `importance`, `inspection`, and `gpu_predict` unchanged.

    `params.learning_rate` must be 1.0 and `bagging` must be enabled; see
    `check_rf_params` for why each is a refusal rather than an override.

    A non-empty `init_score` starts from those raw scores instead of the
    objective's own base score, exactly as in `boosting.train`, and the
    returned ensemble carries a base score of 0.
    """
    if len(target) != data.n_rows:
        raise Error("target length must equal n_rows")
    check_rf_params(params, bagging)
    _check_rf_objective(objective)
    _check_objective(objective, target, alpha)
    _check_sample_weight(sample_weight, data.n_rows)
    params.tree.monotone.check_features(data.n_features)
    if len(init_score) != 0 and len(init_score) != data.n_rows:
        raise Error("init_score length must equal n_rows")

    var n = data.n_rows
    var raw = List[Float64](capacity=n)
    var base_score = 0.0
    if len(init_score) == n:
        for r in range(n):
            raw.append(init_score[r])
    else:
        base_score = _base_score(target, objective, sample_weight, alpha)
        for _ in range(n):
            raw.append(base_score)

    var trees = List[Tree]()
    _grow_forest(
        data,
        target,
        objective,
        params,
        sample_weight,
        alpha,
        bagging,
        raw,
        0,
        trees,
    )
    var rate = _forest_rate(len(trees))
    return Booster(
        trees^,
        base_score,
        rate,
        objective,
        params.tree.monotone.copy(),
    )


def is_forest(booster: Booster) -> Bool:
    """Whether `booster` carries the `1 / K` factor a forest is stored with.

    A `Booster` records no boosting mode, so this is a structural test, not a
    label: it is what `train_rf_more` checks before it treats an ensemble's
    trees as an average rather than a sum. A GBDT model whose learning rate
    happens to equal `1 / K` passes, which is why the check reports a
    mismatch rather than certifying a match.
    """
    if len(booster.trees) == 0:
        return booster.learning_rate == 1.0
    return booster.learning_rate == _forest_rate(len(booster.trees))


def train_rf_more(
    mut booster: Booster,
    data: BinnedMatrix,
    target: List[Float64],
    params: BoosterParams,
    sample_weight: List[Float64] = [],
    alpha: Float64 = 0.9,
    bagging: BaggingParams = BaggingParams.disabled(),
    init_score: List[Float64] = [],
) raises -> Int:
    """Grow `params.n_estimators` more trees into a fitted forest and return
    how many were kept.

    Continued training is exact here in a way it is not for GBDT: a forest's
    gradients are taken at the offset alone, which the ensemble carries as
    `base_score`, so resuming needs no pass over the existing trees and the
    trees already grown are not touched. Only the shared factor changes, from
    `1 / K` to `1 / (K + added)`, which is what re-averaging a forest means.

    `init_score` must be the same offset the first call trained under, for
    the same reason `boosting.train_more` says: it is training state the
    ensemble does not carry.

    The ensemble must actually be a forest. `is_forest` is a structural
    check, so this raises on a GBDT model rather than quietly re-weighting
    every tree in it.
    """
    if len(target) != data.n_rows:
        raise Error("target length must equal n_rows")
    if not is_forest(booster):
        raise Error(
            "train_rf_more expects an ensemble trained by train_rf: its"
            " shrinkage factor must be 1 / number of trees, which is how a"
            " forest stores its average. Continuing a summed (GBDT) ensemble"
            " as a forest would re-weight every tree it already holds"
        )
    check_rf_params(params, bagging)
    _check_rf_objective(booster.objective)
    _check_objective(booster.objective, target, alpha)
    _check_sample_weight(sample_weight, data.n_rows)
    params.tree.monotone.check_features(data.n_features)
    if len(init_score) != 0 and len(init_score) != data.n_rows:
        raise Error("init_score length must equal n_rows")

    var n = data.n_rows
    var raw = List[Float64](capacity=n)
    var has_init = len(init_score) == n
    for r in range(n):
        # The offset the first call boosted from. Unlike GBDT there is no
        # accumulation over the existing trees, so no association question
        # arises and this is bit-identical to what the first call held.
        var s = booster.base_score
        if has_init:
            s += init_score[r]
        raw.append(s)

    var grown = List[Tree]()
    _grow_forest(
        data,
        target,
        booster.objective,
        params,
        sample_weight,
        alpha,
        bagging,
        raw,
        len(booster.trees),
        grown,
    )
    var added = len(grown)
    for i in range(added):
        booster.trees.append(grown[i].copy())
    booster.learning_rate = _forest_rate(len(booster.trees))
    return added
