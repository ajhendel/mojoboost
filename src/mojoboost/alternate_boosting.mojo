"""One entry point for LightGBM's `boosting` parameter.

LightGBM's `boosting` (aliased `boosting_type`) selects the training
strategy: `gbdt`, `goss`, `dart`, or `rf`. mojoboost has always implemented
the first two, through `boosting.train` and a `goss.GossParams` handed to it.
This module resolves all four names in one place and dispatches, so a caller
selects a strategy by naming it rather than by picking a function.

What this module owns, and what it does not
-------------------------------------------
It owns the vocabulary (`parse_boosting`, `boosting_name`,
`AlternateBoostingParams`), the dispatch, and the DART round loop. It owns no
algorithm and defines no model:

- `boosting_dart.mojo` is the DART algorithm: which iterations a round drops,
  what the drop and the new tree weigh afterwards, and how the cached raw
  scores move. It is written as pure state transitions over
  `(trees, weights, raw)` and grows no tree. `train_dart` below is the loop
  that drives it, and it is here rather than in `boosting.mojo` only because
  this lane does not own `boosting.mojo`. That is a temporary address, not a
  design; see `handoffs/connect_17_alternate_boosting.md`.
- `boosting_rf.mojo` is the whole of random-forest mode, single-output and
  multiclass, including its own continuation, early stopping, and the
  `Booster` adapter. This module only routes `rf` into it.
- `gbdt` and `goss` route into the untouched production `boosting.train` with
  the arguments they always took. Nothing about them changes.

DART: per-tree weights in a single-scalar Booster
-------------------------------------------------
`boosting_dart` carries one weight per tree, because that is what the
algorithm manipulates: a round rescales the weights of the trees it dropped.
`Booster` carries one shrinkage factor for the whole ensemble. The two are
reconciled at the end of training by `fold_weights_into_trees`, which
multiplies each tree's node values by that tree's weight and lets the
ensemble's factor be 1.0.

This is not an approximation. It is what LightGBM does: `Tree::Shrinkage`
scales `leaf_value_` and `internal_value_` in place, and a LightGBM DART
model file records already-shrunk leaf values with a per-tree `shrinkage`
field that says what was applied rather than what is still to apply.
`Booster.predict_raw_row` computes `score += learning_rate * leaf`, so with
`learning_rate = 1.0` and the weight already inside `leaf` it evaluates the
same product, rounded once, that the weighted form would have. So
`serialize.save_model` writes a DART model today with no version bump, and
`contrib`, `importance`, `inspection`, `gpu_predict`, and `lgbm_model_io` see
a plain ensemble.

What the fold gives up is the ability to read a tree's weight back off a
saved model, which matters in exactly one place and is handled there (see
`train_dart_more`).

RF: dispatch only, because boosting_rf already carries the adapter
------------------------------------------------------------------
`boosting_rf` keeps its own `RfBooster`, because a forest averages rather
than sums and `Booster` has no `average_output` flag. It also supplies the
whole bridge: `RfBooster.to_booster` (base score 0 and a rate of `1 / T`
make a `Booster`'s sum the forest's mean), `is_forest` for the structural
check, and `train_rf` / `train_rf_more`, which take exactly the uniform
argument set the other modes take and return an ordinary `Booster`. So this
module dispatches to those two and adds nothing: no `RfParams` translation
of its own (that is `RfParams.from_booster_params`), no bridge of its own,
and no continuation of its own.

The uniform argument set is narrower than a forest can be. `train_rf` takes
no `GossParams` and no `ClassBaggingParams`, on `boosting_rf`'s own
reasoning that GOSS is a `boosting` value in its own right, so a caller who
selected a strategy by name has already chosen between them. Both are
therefore REFUSED here under `rf` rather than dropped, with the error naming
`boosting_rf.train_forest` and `RfParams`, which is the entry point for a
forest randomized by gradient sampling or by class-conditional bagging, and
the entry point for iteration ranges (a bridged `Booster` divides by the
whole tree count whatever the range).

`boosting.train_more` must never be pointed at a DART or a bridged RF
ensemble. It would read the folded `1.0`, or the forest's `1 / T`, as the
rate to shrink new GBDT trees by. `train_boosting_more` is the continuation
entry point; the DART path checks the ensemble's unit factor and the RF path
checks `boosting_rf.is_forest` before either touches anything.

Combinations that are refused rather than resolved
--------------------------------------------------
- `dart` with GOSS, on a non-CPU device, with ranking, with
  `uniform_drop=False`, or with a configuration that never drops
  (`boosting_dart.check_dart_supported`, `DartParams.validate`).
- `rf` with a learning rate other than 1.0
  (`boosting_rf.check_rf_learning_rate`), with `init_score`
  (`check_rf_init_score`), with a custom objective or `lambdarank`
  (`check_rf_objective`), or with no source of per-tree randomness at all
  (`check_rf_params`).
- `rf` with GOSS or with balanced bagging, refused **here** with the reason
  above and the entry point that does take them.
- `goss` mode with a disabled `GossParams`, and the converse for the modes
  that cannot take one.
- `dart` mode with a disabled `DartParams`, and an enabled one under any
  other mode.
- Multiclass under `dart`. `boosting_rf` already implements multiclass
  forests (`train_rf_multiclass`), so only DART is single-output here;
  `boosting_dart` is written for the round-major multiclass layout, so that
  gap is a loop rather than a redesign.

DART validation-set early stopping is NOT connected. `boosting_dart` provides
`DartBestState`, `dart_record_best`, and `dart_restore_best` precisely because
truncating a DART ensemble to its best round is wrong: a later round may have
rescaled a tree the best round contained, so popping trees recovers the right
tree set with the wrong weights. There is no `train_dart_with_valid` here
rather than one that truncates and quietly gets the weights wrong.
`boosting_rf.train_rf_with_valid` already exists for forests.
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
from .boosting_rf import is_rf_boosting, train_rf, train_rf_more
from .device import CPU_DEVICE
from .goss import GossParams
from .model import Model
from .sampling import ClassBaggingParams
from .tree import Tree, grow_tree

# LightGBM's four `boosting` values, as codes. The numbers are internal and
# are never serialized: a fitted model records no mode.
comptime BOOSTING_GBDT = 0
comptime BOOSTING_GOSS = 1
comptime BOOSTING_DART = 2
comptime BOOSTING_RF = 3


def parse_boosting(name: String) raises -> Int:
    """A `boosting` code from LightGBM's name for it.

    LightGBM's aliases are accepted. The `rf` spellings are resolved by
    `boosting_rf.is_rf_boosting` rather than by a second list here, so the
    names and the trainer that implements them cannot drift apart. Names are
    canonical lowercase, as in `device.parse_device` and
    `params.objective_from_name`.
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
    if is_rf_boosting(name):
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

    One bundle, so that a caller passes one thing and so that the mode and its
    parameters cannot be separated on the way down. `dart` is
    `DartParams.disabled()` unless the mode is `dart`; selecting `dart` with a
    disabled bundle, or leaving an enabled one under another mode, is refused
    rather than silently resolved.

    Random forest needs no bundle here: `boosting_rf.train_rf` takes the same
    `BoosterParams` and `BaggingParams` every other mode is given, and
    assembles its own `RfParams` with `RfParams.from_booster_params`.
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
        """GOSS. The rates and the seed travel in the `GossParams` the trainer
        is handed; this only names the strategy."""
        var out = AlternateBoostingParams()
        out.mode = BOOSTING_GOSS
        return out^

    @staticmethod
    def rf() -> AlternateBoostingParams:
        """Random forest. Its settings are the ordinary tree and sampler
        parameters; see `boosting_rf.check_rf_params`."""
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

    def allows_goss(self) -> Bool:
        """Whether this mode can be handed an enabled `GossParams` here.

        Only `goss`, which requires one. `gbdt` refuses because GOSS is a
        different `boosting` value; `dart` refuses for the reason
        `check_dart_supported` gives. `rf` refuses **here** even though a
        forest can legitimately be randomized by gradient sampling, because
        `boosting_rf.train_rf` takes no `GossParams`: that combination is
        reached through `boosting_rf.train_forest` with an `RfParams`, and
        `train_boosting` says so in the error.
        """
        return self.mode == BOOSTING_GOSS

    def validate(self, goss: GossParams) raises:
        """Reject a mode that disagrees with the row sampler or the dropout
        bundle it was handed.

        `goss` and `dart` are separate arguments everywhere in the Mojo API
        (each carries settings no other mode reads), so a mode and its
        parameters can contradict each other, and a contradiction means the
        caller asked for two strategies at once. Neither is silently
        preferred.
        """
        _ = boosting_name(self.mode)
        if self.mode == BOOSTING_GOSS and not goss.enabled:
            raise Error(
                "boosting='goss' needs an enabled GossParams; it carries"
                " top_rate, other_rate, and the seed"
            )
        if goss.enabled and self.mode == BOOSTING_RF:
            raise Error(
                "boosting='rf' with goss is a real configuration, but not"
                " through this entry point: boosting_rf.train_rf takes no"
                " GossParams. Build a boosting_rf.RfParams with the goss"
                " field set and call boosting_rf.train_forest"
            )
        if goss.enabled and not self.allows_goss():
            raise Error(
                "boosting='",
                boosting_name(self.mode),
                "' cannot be combined with an enabled GossParams; goss is"
                " itself a boosting value, so the two name different"
                " strategies",
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


def _check_rf_uniform_args(class_bagging: ClassBaggingParams) raises:
    """Refuse the one uniform argument `boosting_rf.train_rf` has no place to
    put.

    GOSS is already refused by `AlternateBoostingParams.validate`, which sees
    it. Balanced bagging is not a `boosting` value and so has no mode to
    disagree with, but `train_rf` takes no `ClassBaggingParams` either, and a
    forest randomized only by the class-conditional fractions is a
    configuration `boosting_rf.rf_randomizer_name` explicitly accepts. So it
    is named rather than dropped, with the entry point that takes it.
    """
    if class_bagging.enabled():
        raise Error(
            "boosting='rf' with pos_bagging_fraction / neg_bagging_fraction"
            " is a real configuration, but not through this entry point:"
            " boosting_rf.train_rf takes no ClassBaggingParams. Build a"
            " boosting_rf.RfParams with the class_bagging field set and call"
            " boosting_rf.train_forest"
        )


def fold_weights_into_trees(
    mut trees: List[Tree], weights: List[Float64]
) raises:
    """Multiply each tree's node values by that tree's weight, so an ensemble
    with per-tree weights becomes one a single-scalar `Booster` represents
    exactly at a factor of 1.0.

    Internal node values as well as leaves. An internal node's value is what
    it held before it was split, which `contrib.mojo` conditions on and
    `lgbm_model_io` writes as `internal_value`; folding one and not the other
    would make a tree's attributions disagree with its predictions. Split
    gains are not folded, matching LightGBM, which leaves them as the gains
    that were actually measured.

    A weight of exactly 1.0 is skipped rather than multiplied through, so a
    tree no round ever rescaled comes back bit-identical to the grower's
    output, and folding is idempotent.

    This is the same operation `boosting_rf._add_bias` performs with an
    addition instead of a multiplication, and both are `tree.Tree` behavior
    implemented outside `tree.mojo`. The handoff asks for
    `Tree.shrinkage(rate)` and `Tree.add_bias(bias)`, LightGBM's own names, at
    which point both become loops over a method.
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
    bag, and the per-tree feature sample all read the absolute round index and
    a continued run follows the schedule an uninterrupted one would.

    Single output only: `n_classes` is 1 at every call into `boosting_dart`,
    which is itself written for the round-major multiclass layout as well.
    """
    var n = data.n_rows
    var signs = params.tree.monotone.active_signs()
    var renews = objective_renews_leaves(objective)
    var renew_w = renewal_weights(objective, target, sample_weight)
    var renew_a = renewal_alpha(objective, alpha)
    var grad = List[Float64](capacity=n)
    var hess = List[Float64](capacity=n)
    var bag = List[Int]()
    # What `dart_begin_round` removed, reused across rounds so a dropout round
    # costs one pass over the dropped trees rather than two.
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
            # recorded for a tree that was never kept. Under a row sampler, or
            # when something was dropped, the next round draws differently and
            # may do better; with neither, the objective has converged.
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
    returned ensemble carries a base score of 0. Unlike random-forest mode,
    DART has a per-row starting point for the offset to move, so it is
    accepted rather than refused.
    """
    if len(target) != data.n_rows:
        raise Error("target length must equal n_rows")
    if params.n_estimators < 0:
        raise Error("num_iterations must be nonnegative")
    if params.learning_rate <= 0.0:
        raise Error("learning_rate must be positive")
    if not dart.enabled:
        raise Error(
            "train_dart needs an enabled DartParams; use DartParams.enable()"
        )
    check_dart_supported(dart, CPU_DEVICE, False, False, 1)
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
    each at weight 1. Rounds added here rescale those weights as usual and the
    whole ensemble is folded again on the way out, which is idempotent because
    folding a weight of 1.0 is skipped.

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
        raise Error("num_iterations must be nonnegative")
    if params.learning_rate <= 0.0:
        raise Error("learning_rate must be positive")
    if booster.learning_rate != 1.0:
        raise Error(
            "train_dart_more expects an ensemble trained by train_dart: its"
            " shrinkage factor must be 1.0, because a dart round folds each"
            " tree's weight into that tree's node values. Continuing a gbdt"
            " ensemble, or a random forest bridged by RfBooster.to_booster,"
            " here would discard the factor it carries"
        )
    if not dart.enabled:
        raise Error(
            "train_dart_more needs an enabled DartParams; use"
            " DartParams.enable()"
        )
    check_dart_supported(dart, CPU_DEVICE, False, False, 1)
    check_bagging(bagging)
    _check_objective(booster.objective, target, alpha)
    _check_sample_weight(sample_weight, data.n_rows)
    params.tree.monotone.check_features(data.n_features)
    if len(init_score) != 0 and len(init_score) != data.n_rows:
        raise Error("init_score length must equal n_rows")

    var weights = dart_uniform_weights(len(booster.trees), 1.0)
    var base_scores = List[Float64]()
    base_scores.append(booster.base_score)
    var raw = dart_recompute_raw(data, booster.trees, weights, base_scores, 1)
    if len(init_score) == data.n_rows:
        # The offset the first call trained under. `boosting.train_more` folds
        # it in before the trees rather than after; the difference is an
        # association, and this path already does not promise bit equality
        # with a single call (see above).
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
    class_bagging: ClassBaggingParams = ClassBaggingParams.disabled(),
) raises -> Booster:
    """Train under the selected `boosting` mode.

    Every argument other than `boosting` carries the meaning it has in
    `boosting.train`, which is where `gbdt` and `goss` go unchanged. The
    returned `Booster` is an ordinary one whatever the mode; what the mode
    changes is which trees are in it and what shrinkage factor they share.

    `rf` goes to `boosting_rf.train_rf`, which bridges its own `RfBooster`
    back with `to_booster`. That bridge is exact for whole predictions and
    lossy for iteration ranges (a forest's `Booster` divides by the whole
    tree count whatever the range), so a caller who needs ranges holds the
    `RfBooster` from `boosting_rf.train_forest` instead. `init_score` is
    refused under `rf` by `boosting_rf.check_rf_init_score` rather than
    ignored.
    """
    boosting.validate(goss)
    if boosting.mode == BOOSTING_DART:
        if class_bagging.enabled():
            raise Error(
                "boosting='dart' does not support balanced bagging;"
                " pos_bagging_fraction / neg_bagging_fraction draw a"
                " class-conditional bag, which no dart round has been written"
                " to take"
            )
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
        _check_rf_uniform_args(class_bagging)
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
        class_bagging,
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
    class_bagging: ClassBaggingParams = ClassBaggingParams.disabled(),
) raises -> Int:
    """Continue training under the selected mode, returning how many trees
    were added.

    A fitted `Booster` records no mode, so the caller supplies the same one it
    trained with. The DART path checks the ensemble's unit shrinkage factor
    before touching it, which is structural: it can catch a mismatch but
    cannot certify a match.

    `rf` goes to `boosting_rf.train_rf_more`, which checks
    `boosting_rf.is_forest` first and rescales every existing tree, because
    the averaging weight is a function of the tree count. Note its own
    caveat: the constant the new trees are fitted at cannot be read off a
    bridged `Booster`, so it is recomputed from `target` and `sample_weight`,
    which is exact only when this is the data the forest was trained on. A
    forest kept as an `RfBooster` carries its own base score and needs no
    such precondition; that is `boosting_rf.train_forest_more`.
    """
    boosting.validate(goss)
    if boosting.mode == BOOSTING_RF:
        _check_rf_uniform_args(class_bagging)
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
    if boosting.mode == BOOSTING_DART:
        if class_bagging.enabled():
            raise Error(
                "boosting='dart' does not support balanced bagging"
            )
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
        class_bagging,
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
    class_bagging: ClassBaggingParams = ClassBaggingParams.disabled(),
) raises -> Model:
    """`model.fit` with a `boosting` mode, on a column-major raw feature
    matrix (`features[f * n_rows + r]`).

    This exists only because `model.fit` has no mode argument yet, and it is a
    wrapper rather than a second fit path: it bins with the same
    `binning.fit_bins`, transforms with the same mapper, and returns the same
    `model.Model` that `serialize.save_model` writes and every predictor
    reads. The handoff asks for the mode to move onto `model.fit`, at which
    point this function goes away.

    A forest arrives here already bridged, so a `Model` holding one predicts
    correctly over the whole ensemble and reports iteration ranges that are
    not the forest's; see `train_boosting`.

    CPU only. `train_gpu` and the sparse trainers carry their own round loops,
    so a mode reaches them through their own edits, which this lane does not
    own.
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
        [],
        class_bagging,
    )
    return Model(mapper^, booster^)
