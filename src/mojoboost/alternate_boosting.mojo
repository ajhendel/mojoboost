"""One entry point for LightGBM's `boosting` parameter, and the round loops
the two alternate values need.

LightGBM's `boosting` (aliased `boosting_type`) selects the training
strategy: `gbdt`, `goss`, `dart`, or `rf`. mojoboost has always implemented
the first two, through `boosting.train` and a `goss.GossParams` handed to
it. This module resolves all four names in one place and dispatches, so a
caller selects a strategy by naming it rather than by picking a function.

What this module owns, and what it does not
-------------------------------------------
It owns the vocabulary (`parse_boosting`, `boosting_name`,
`AlternateBoostingParams`), the dispatch, and the DART round loop. It owns
no algorithm:

- `boosting_dart.mojo` is the DART algorithm: which iterations a round
  drops, what the drop and the new tree weigh afterwards, and how the cached
  raw scores move. It is deliberately written as pure state transitions over
  `(trees, weights, raw)` and grows no tree. `train_dart` below is the loop
  that calls it, and it is here rather than in `boosting.mojo` only because
  this lane does not own `boosting.mojo`. That is a temporary address, not a
  design: see the handoff.
- `boosting_rf.mojo` is the random-forest loop, which is small enough to be
  the whole of that mode.
- `gbdt` and `goss` route into the untouched production `boosting.train`
  with the arguments they always took. Nothing about them changes.

Every mode returns an ordinary `boosting.Booster` of ordinary `tree.Tree`
values, predicts through the same `Booster.predict_raw_row`, and serializes
through `serialize.save_model` with no format change. There is no second
model representation here.

How a per-tree weight reaches a single-scalar Booster
----------------------------------------------------
`boosting_dart` carries one weight per tree, because that is what the
algorithm manipulates: a round rescales the weights of the trees it dropped.
`Booster` carries one shrinkage factor for the whole ensemble. The two are
reconciled at the end of training by `fold_weights_into_trees`, which
multiplies each tree's node values by that tree's weight and lets the
ensemble's factor be 1.0.

This is not an approximation and it is not a workaround. It is what LightGBM
does: `Tree::Shrinkage` scales `leaf_value_` and `internal_value_` in place,
and a LightGBM DART model file records already-shrunk leaf values with a
per-tree `shrinkage` field that is a record of what was applied rather than
a factor still to apply. `Booster.predict_raw_row` computes
`score += learning_rate * leaf`, so with `learning_rate = 1.0` and the
weight already inside `leaf` it evaluates the same product, rounded once,
that the weighted form would have. What the fold gives up is the ability to
read a tree's weight back off a saved model, which matters in exactly one
place and is handled there (see `train_dart_more`).

So the consequences, spelled out:

- `serialize.save_model` writes a DART model today, no version bump.
- `contrib`, `importance`, `inspection`, `gpu_predict`, and
  `lgbm_model_io` see a plain ensemble. Internal node values are folded
  along with the leaves, so attribution stays consistent with prediction.
- `boosting.train_more` must NOT be pointed at a DART or an RF ensemble. It
  would read the folded `1.0` (or the forest's `1 / K`) as the rate to
  shrink new GBDT trees by. `train_boosting_more` is the continuation entry
  point and both alternate modes check the ensemble's shape before touching
  it.

Combinations that are refused rather than resolved
--------------------------------------------------
- `dart` or `rf` with GOSS (`boosting_dart.check_dart_supported`,
  `AlternateBoostingParams.validate`).
- `dart` on a non-CPU device, with ranking, or with `uniform_drop=False`
  (`boosting_dart.check_dart_supported`, `DartParams.validate`).
- `rf` with a learning rate other than 1.0, or without row bagging
  (`boosting_rf.check_rf_params`).
- A mode/`GossParams` disagreement in either direction.
- Multiclass. Both alternate modes are single-output here.
  `boosting_dart` is already written for the round-major multiclass layout
  (`n_classes` runs through every one of its entry points), so the multiclass
  loop is a loop, not a redesign; it is listed in the handoff.

DART validation-set early stopping is NOT connected. `boosting_dart`
provides the pieces it needs (`DartBestState`, `dart_record_best`,
`dart_restore_best`) precisely because truncating a DART ensemble to its
best round is wrong: a later round may have rescaled a tree the best round
contained, so popping trees recovers the right tree set with the wrong
weights. There is no `train_dart_with_valid` here, rather than one that
truncates and quietly gets the weights wrong. The handoff records its shape.
"""

from .bagging import (
    BaggingParams,
    bagging_enabled,
    check_bagging,
    refresh_bag,
)
from .binning import BinnedMatrix, fit_bins
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
    train,
    train_more,
)
from .boosting_dart import (
    DartParams,
    check_dart_supported,
    dart_begin_round,
    dart_commit_round,
    dart_normalization,
    dart_recompute_raw,
    dart_uniform_weights,
    select_drop,
)
from .boosting_rf import train_rf, train_rf_more
from .device import CPU_DEVICE
from .goss import GossParams
from .model import Model
from .tree import Tree, grow_tree

# LightGBM's four `boosting` values, as codes. The numbers are internal and
# are never serialized: a fitted model records no mode, because every mode
# produces the same kind of ensemble (see the module docstring).
comptime BOOSTING_GBDT = 0
comptime BOOSTING_GOSS = 1
comptime BOOSTING_DART = 2
comptime BOOSTING_RF = 3


def parse_boosting(name: String) raises -> Int:
    """A `boosting` code from LightGBM's name for it.

    LightGBM's aliases are accepted. Names are canonical lowercase, as in
    `device.parse_device` and `params.objective_from_name`.
    """
    if (
        name == "gbdt"
        or name == "gbrt"
        or name == "tree"
        or name == "gradient_boosting_decision_tree"
    ):
        return BOOSTING_GBDT
    if name == "goss":
        return BOOSTING_GOSS
    if name == "dart":
        return BOOSTING_DART
    if name == "rf" or name == "random_forest":
        return BOOSTING_RF
    raise Error(
        "unknown boosting '", name, "'; expected gbdt, goss, dart, or rf"
    )


def boosting_name(mode: Int) raises -> String:
    """The LightGBM name for a `boosting` code, for reporting."""
    if mode == BOOSTING_GBDT:
        return "gbdt"
    if mode == BOOSTING_GOSS:
        return "goss"
    if mode == BOOSTING_DART:
        return "dart"
    if mode == BOOSTING_RF:
        return "rf"
    raise Error("unknown boosting code ", mode)


struct AlternateBoostingParams(Copyable, Movable):
    """The mode, and the parameters only one mode reads.

    One bundle so that a caller passes one thing and so that the mode and its
    parameters cannot be separated on the way down. `dart` is
    `DartParams.disabled()` unless the mode is `dart`, and selecting `dart`
    with a disabled bundle is refused rather than silently treated as GBDT.
    """

    var mode: Int
    var dart: DartParams

    def __init__(out self):
        """Plain GBDT, LightGBM's default."""
        self.mode = BOOSTING_GBDT
        self.dart = DartParams.disabled()

    @staticmethod
    def default() -> AlternateBoostingParams:
        return AlternateBoostingParams()

    @staticmethod
    def gbdt() -> AlternateBoostingParams:
        return AlternateBoostingParams()

    @staticmethod
    def goss() -> AlternateBoostingParams:
        """GOSS. The rates and the seed travel in the `GossParams` the
        trainer is handed; this only names the strategy."""
        var out = AlternateBoostingParams()
        out.mode = BOOSTING_GOSS
        return out^

    @staticmethod
    def rf() -> AlternateBoostingParams:
        """Random forest. Its settings are `BoosterParams` and
        `BaggingParams`; see `boosting_rf.check_rf_params`."""
        var out = AlternateBoostingParams()
        out.mode = BOOSTING_RF
        return out^

    @staticmethod
    def dart_with(var dart: DartParams) raises -> AlternateBoostingParams:
        """DART with an explicit parameter bundle, which must be enabled."""
        var out = AlternateBoostingParams()
        out.mode = BOOSTING_DART
        out.dart = dart^
        return out^

    @staticmethod
    def named(name: String) raises -> AlternateBoostingParams:
        """A bundle for LightGBM's spelling of a mode. `dart` comes back with
        `DartParams.enable()`, LightGBM's own defaults."""
        var mode = parse_boosting(name)
        if mode == BOOSTING_DART:
            return AlternateBoostingParams.dart_with(DartParams.enable())
        var out = AlternateBoostingParams()
        out.mode = mode
        return out^

    def validate(self, goss: GossParams) raises:
        """Reject a mode that disagrees with the row sampler it was handed,
        or a DART mode with a disabled bundle.

        `goss` is a separate argument everywhere in the Mojo API (it carries
        rates and a seed no other mode reads), so the mode and the sampler
        can contradict each other, and a contradiction means the caller asked
        for two strategies at once. Neither is silently preferred.
        """
        _ = boosting_name(self.mode)
        if self.mode == BOOSTING_GOSS:
            if not goss.enabled:
                raise Error(
                    "boosting='goss' needs an enabled GossParams; it carries"
                    " top_rate, other_rate, and the seed"
                )
            return
        if goss.enabled:
            raise Error(
                "boosting='",
                boosting_name(self.mode),
                "' was given an enabled GossParams; goss is itself a boosting"
                " value, so the two name different strategies",
            )
        if self.mode == BOOSTING_DART:
            if not self.dart.enabled:
                raise Error(
                    "boosting='dart' needs an enabled DartParams; a disabled"
                    " one drops nothing, which is boosting='gbdt' wearing"
                    " dart's parameter names"
                )
            self.dart.validate()
        elif self.dart.enabled:
            raise Error(
                "boosting='",
                boosting_name(self.mode),
                "' was given an enabled DartParams; dropout is a property of"
                " the dart strategy, not a modifier on the others",
            )


def fold_weights_into_trees(
    mut trees: List[Tree], weights: List[Float64]
) raises:
    """Multiply each tree's node values by that tree's weight, so an
    ensemble with per-tree weights becomes one a single-scalar `Booster`
    represents exactly at a factor of 1.0.

    Internal node values as well as leaves. An internal node's value is what
    it held before it was split, which `contrib.mojo` conditions on and
    `lgbm_model_io` writes as `internal_value`; folding one and not the other
    would make a tree's attributions disagree with its predictions. Split
    gains are not folded, matching LightGBM, which leaves them as the gains
    that were actually measured.

    A weight of exactly 1.0 is skipped rather than multiplied through, so a
    tree no round ever rescaled comes back bit-identical to the tree the
    grower produced.

    This is the one piece of `tree.Tree` behavior this lane implements
    outside `tree.mojo`. The handoff asks for it to move there as
    `Tree.shrinkage`, LightGBM's own name, at which point this function
    becomes a loop over that method.
    """
    if len(weights) != len(trees):
        raise Error("fold_weights_into_trees needs one weight per tree")
    for i in range(len(trees)):
        var w = weights[i]
        if w == 1.0:
            continue
        ref tree = trees[i]
        for node in range(len(tree.value)):
            tree.value[node] = tree.value[node] * w


def _dart_rounds(
    data: BinnedMatrix,
    target: List[Float64],
    objective: Int,
    params: BoosterParams,
    sample_weight: List[Float64],
    alpha: Float64,
    bagging: BaggingParams,
    dart: DartParams,
    learning_rate: Float64,
    round_offset: Int,
    mut raw: List[Float64],
    mut trees: List[Tree],
    mut weights: List[Float64],
) raises:
    """The DART loop, shared by the fit-from-scratch and continue-training
    paths.

    On entry and on exit `raw` is the full weighted ensemble's raw score for
    every training row, and `weights` holds one weight per tree. Inside a
    round `raw` is the dropped-out score, which is what the round's gradients
    and its leaf renewal are taken against; that is the whole of DART.

    `round_offset` is the number of trees already grown, so the drop set, the
    bag, and the per-tree feature sample all read the absolute round index
    and a continued run follows the schedule an uninterrupted one would.

    Single output only: `n_classes` is 1 at every call into `boosting_dart`,
    which is written for the round-major multiclass layout as well.
    """
    var n = data.n_rows
    var signs = params.tree.monotone.active_signs()
    var renews = objective_renews_leaves(objective)
    var renew_w = renewal_weights(objective, target, sample_weight)
    var renew_a = renewal_alpha(objective, alpha)
    var grad = List[Float64](capacity=n)
    var hess = List[Float64](capacity=n)
    var bag = List[Int]()
    # What `dart_begin_round` removed, reused across rounds so a dropout
    # round costs one pass over the dropped trees rather than two.
    var contribution = List[Float64]()

    for i in range(params.n_estimators):
        var round = round_offset + i
        var drop = select_drop(dart, len(trees), round)
        dart_begin_round(data, trees, weights, drop, 1, raw, contribution)

        refresh_bag(bag, bagging, n, round)
        _fill_grad_hess(
            raw, target, objective, sample_weight, alpha, grad, hess
        )
        var tree = grow_tree(data, grad, hess, params.tree, bag, round)
        if renews:
            # Renewal takes residuals against the score the tree was actually
            # fitted to, which under DART is the dropped-out score. That is
            # the requirement `check_dart_supported` documents.
            _renew_leaf_values(
                tree, data, target, raw, renew_w, renew_a, bag, signs,
                params.tree.extra,
            )

        if tree.n_leaves == 1 and abs(tree.value[0]) < 1e-12:
            # The round adds nothing, so it is abandoned whole: the dropped
            # trees go back at the weights they had and no normalization is
            # recorded for a tree that was never kept. Under a row sampler,
            # or when something was dropped, the next round draws differently
            # and may do better; with neither, the objective has converged.
            for j in range(len(raw)):
                raw[j] += contribution[j]
            if bagging_enabled(bagging) or not drop.is_empty():
                continue
            break

        var norm = dart_normalization(
            drop.count(), learning_rate, dart.xgboost_dart_mode
        )
        var grown = List[Tree]()
        grown.append(tree^)
        dart_commit_round(
            data, drop, norm, 1, contribution, grown^, trees, weights, raw
        )


def train_dart(
    data: BinnedMatrix,
    target: List[Float64],
    objective: Int,
    params: BoosterParams,
    dart: DartParams = DartParams.enable(),
    sample_weight: List[Float64] = [],
    alpha: Float64 = 0.9,
    bagging: BaggingParams = BaggingParams.disabled(),
    init_score: List[Float64] = [],
) raises -> Booster:
    """Train a DART ensemble, LightGBM's `boosting='dart'`.

    Every argument other than `dart` carries the meaning it has in
    `boosting.train`. The returned `Booster` has a shrinkage factor of 1.0
    because every tree's weight has been folded into its node values; see the
    module docstring for why, and for what that costs.

    A non-empty `init_score` starts from those raw scores instead of the
    objective's own base score, exactly as in `boosting.train`, and the
    returned ensemble carries a base score of 0.
    """
    if len(target) != data.n_rows:
        raise Error("target length must equal n_rows")
    if params.n_estimators < 0:
        raise Error("n_estimators must not be negative")
    if params.learning_rate <= 0.0:
        raise Error("learning_rate must be positive")
    check_dart_supported(dart, CPU_DEVICE, False, False, 1)
    if not dart.enabled:
        raise Error(
            "train_dart needs an enabled DartParams; use DartParams.enable()"
        )
    check_bagging(bagging)
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
    var weights = List[Float64]()
    _dart_rounds(
        data,
        target,
        objective,
        params,
        sample_weight,
        alpha,
        bagging,
        dart,
        params.learning_rate,
        0,
        raw,
        trees,
        weights,
    )
    fold_weights_into_trees(trees, weights)
    return Booster(
        trees^,
        base_score,
        1.0,
        objective,
        params.tree.monotone.copy(),
    )


def train_dart_more(
    mut booster: Booster,
    data: BinnedMatrix,
    target: List[Float64],
    params: BoosterParams,
    dart: DartParams = DartParams.enable(),
    sample_weight: List[Float64] = [],
    alpha: Float64 = 0.9,
    bagging: BaggingParams = BaggingParams.disabled(),
    init_score: List[Float64] = [],
) raises -> Int:
    """Grow `params.n_estimators` more DART rounds into a fitted ensemble and
    return how many trees were added.

    The folded representation is what makes this work rather than what makes
    it hard. A DART ensemble comes back with every weight already inside its
    trees, so lifting it back into the weighted form is
    `dart_uniform_weights(n_trees, 1.0)`: the trees are exactly the trees,
    each at weight 1. Rounds added here rescale those weights as usual and
    the whole ensemble is folded again on the way out, which is idempotent,
    because folding a weight of 1.0 is skipped.

    Two things differ from `boosting.train_more`:

    - `params.learning_rate` is the rate the CONTINUED rounds use and is not
      checked against the ensemble, because a folded DART ensemble's stored
      factor is 1.0 and no longer records the rate it was trained with. It is
      the caller's job to pass the same one. This is the one place the fold
      gives something up.
    - The existing trees are rewritten, not just appended to. A round that
      drops a tree rescales it, so the ensemble handed back is not the
      ensemble handed in plus new trees.

    Resuming rebuilds the raw scores from the model with
    `dart_recompute_raw`. This is NOT bit-identical to training the same
    number of rounds in one call: the single-call path carries `raw` forward
    incrementally through the rescalings, and recomputing sums the same terms
    in a different association. The trees chosen agree; the last bits of the
    scores need not.
    """
    if len(target) != data.n_rows:
        raise Error("target length must equal n_rows")
    if params.n_estimators < 0:
        raise Error("n_estimators must not be negative")
    if params.learning_rate <= 0.0:
        raise Error("learning_rate must be positive")
    if booster.learning_rate != 1.0:
        raise Error(
            "train_dart_more expects an ensemble trained by train_dart: its"
            " shrinkage factor must be 1.0, because a dart round folds each"
            " tree's weight into that tree's node values. Continuing a gbdt"
            " or rf ensemble here would discard the factor it carries"
        )
    check_dart_supported(dart, CPU_DEVICE, False, False, 1)
    if not dart.enabled:
        raise Error(
            "train_dart_more needs an enabled DartParams; use"
            " DartParams.enable()"
        )
    check_bagging(bagging)
    _check_objective(booster.objective, target, alpha)
    _check_sample_weight(sample_weight, data.n_rows)
    params.tree.monotone.check_features(data.n_features)
    if len(init_score) != 0 and len(init_score) != data.n_rows:
        raise Error("init_score length must equal n_rows")

    var weights = dart_uniform_weights(len(booster.trees), 1.0)
    var base_scores = List[Float64]()
    base_scores.append(booster.base_score)
    var raw = dart_recompute_raw(
        data, booster.trees, weights, base_scores, 1
    )
    if len(init_score) == data.n_rows:
        # The offset the first call trained under. `boosting.train_more`
        # folds it in before the trees rather than after; the difference is
        # an association, and this path already does not promise bit
        # equality with a single call (see the docstring).
        for r in range(data.n_rows):
            raw[r] += init_score[r]

    var before = len(booster.trees)
    _dart_rounds(
        data,
        target,
        booster.objective,
        params,
        sample_weight,
        alpha,
        bagging,
        dart,
        params.learning_rate,
        before,
        raw,
        booster.trees,
        weights,
    )
    fold_weights_into_trees(booster.trees, weights)
    return len(booster.trees) - before


def train_boosting(
    data: BinnedMatrix,
    target: List[Float64],
    objective: Int,
    params: BoosterParams,
    boosting: AlternateBoostingParams = AlternateBoostingParams(),
    sample_weight: List[Float64] = [],
    alpha: Float64 = 0.9,
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
    init_score: List[Float64] = [],
) raises -> Booster:
    """Train under the selected `boosting` mode.

    Every argument other than `boosting` carries the meaning it has in
    `boosting.train`, which is where `gbdt` and `goss` go unchanged. The
    returned `Booster` is an ordinary one whatever the mode; what the mode
    changes is which trees are in it and what shrinkage factor they share.
    """
    boosting.validate(goss)
    if boosting.mode == BOOSTING_DART:
        return train_dart(
            data,
            target,
            objective,
            params,
            boosting.dart,
            sample_weight,
            alpha,
            bagging,
            init_score,
        )
    if boosting.mode == BOOSTING_RF:
        return train_rf(
            data,
            target,
            objective,
            params,
            sample_weight,
            alpha,
            bagging,
            init_score,
        )
    return train(
        data,
        target,
        objective,
        params,
        sample_weight,
        alpha,
        bagging,
        goss,
        init_score,
    )


def train_boosting_more(
    mut booster: Booster,
    data: BinnedMatrix,
    target: List[Float64],
    params: BoosterParams,
    boosting: AlternateBoostingParams = AlternateBoostingParams(),
    sample_weight: List[Float64] = [],
    alpha: Float64 = 0.9,
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
    init_score: List[Float64] = [],
) raises -> Int:
    """Continue training under the selected mode, returning how many trees
    were added.

    A fitted `Booster` records no mode, so the caller supplies the same one
    it trained with. Both alternate continuations check the ensemble's shape
    first (`boosting_rf.is_forest`, and DART's unit factor), so continuing a
    GBDT model as a forest, or a forest as GBDT, is reported rather than
    performed. The check is structural and cannot be conclusive, which is why
    it reports a mismatch rather than certifying a match.
    """
    boosting.validate(goss)
    if boosting.mode == BOOSTING_DART:
        return train_dart_more(
            booster,
            data,
            target,
            params,
            boosting.dart,
            sample_weight,
            alpha,
            bagging,
            init_score,
        )
    if boosting.mode == BOOSTING_RF:
        return train_rf_more(
            booster,
            data,
            target,
            params,
            sample_weight,
            alpha,
            bagging,
            init_score,
        )
    return train_more(
        booster,
        data,
        target,
        params,
        sample_weight,
        alpha,
        bagging,
        goss,
        init_score,
    )


def fit_boosting(
    features: List[Float64],
    n_rows: Int,
    n_features: Int,
    target: List[Float64],
    objective: Int,
    params: BoosterParams,
    boosting: AlternateBoostingParams = AlternateBoostingParams(),
    max_bins: Int = 255,
    sample_weight: List[Float64] = [],
    alpha: Float64 = 0.9,
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
    use_missing: Bool = True,
    categorical_features: List[Int] = [],
) raises -> Model:
    """`model.fit` with a `boosting` mode, on a column-major raw feature
    matrix (`features[f * n_rows + r]`).

    This exists only because `model.fit` has no mode argument yet, and it is
    a wrapper rather than a second fit path: it bins with the same
    `binning.fit_bins`, transforms with the same mapper, and returns the same
    `model.Model` that `serialize.save_model` writes and every predictor
    reads. The handoff asks for the mode to move onto `model.fit`, at which
    point this function goes away.

    CPU only. `train_gpu` and the sparse trainers carry their own round
    loops, so a mode reaches them through their own edits, which this lane
    does not own.
    """
    var mapper = fit_bins(
        features,
        n_rows,
        n_features,
        max_bins,
        use_missing=use_missing,
        categorical_features=categorical_features,
    )
    var data = mapper.transform(features, n_rows)
    var booster = train_boosting(
        data,
        target,
        objective,
        params,
        boosting,
        sample_weight,
        alpha,
        bagging,
        goss,
        init_score=[],
    )
    return Model(mapper^, booster^)
