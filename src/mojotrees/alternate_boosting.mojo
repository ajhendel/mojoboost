"""One entry point for LightGBM's `boosting` parameter.

LightGBM's `boosting` (aliased `boosting_type`) selects the training
strategy: `gbdt`, `goss`, `dart`, or `rf`. mojotrees has always implemented
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
  design; the `alternate_boosting` row in `tools/connectivity_audit.py`
  records it.
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
- `dart` with GOSS, on a non-CPU device, with ranking, or with a
  configuration that never drops (`boosting_dart.check_dart_supported`,
  `DartParams.validate`). Both of LightGBM's drop rules are implemented,
  `uniform_drop` either way.
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
- Continuing a **bridged multiclass forest** (`train_boosting_multiclass_more`
  under `rf`), because the class log priors its trees were fitted at are
  folded away and `boosting_rf` exposes no bridged multiclass continuation to
  dispatch to. `train_forest_multiclass_more` is the entry point.

The four entry points and what each mode does
---------------------------------------------
| | `gbdt` / `goss` | `dart` | `rf` |
| `train_boosting` | `boosting.train` | here | `train_rf` |
| `train_boosting_more` | `train_more` | here | `train_rf_more` |
| `..._with_valid` | `train_with_valid` | here | `train_forest_with_valid` |
| `..._multiclass` | `train_multiclass` | here | `train_forest_multiclass` |

`..._multiclass_more` and `..._multiclass_with_valid` follow the multiclass
row, except that `rf` continuation is refused as above.

DART early stopping does not truncate. `boosting_dart` provides
`DartBestState`, `dart_record_best`, and `dart_restore_best` precisely because
truncating a DART ensemble to its best round is wrong: a later round may have
rescaled a tree the best round contained, so popping trees recovers the right
tree set with the wrong weights. `train_dart_with_valid` therefore snapshots
the weight vector on every improvement and restores it at the end, which is
exact because a tree's node values are not touched until
`fold_weights_into_trees` runs once at the end of the fit. Truncation alone
*is* exact for a forest, whose trees are independent, which is why the same
dispatcher sends `rf` to `boosting_rf.train_forest_with_valid`.
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
    MulticlassBooster,
    _base_score,
    _check_objective,
    _check_sample_weight,
    _fill_grad_hess,
    _fill_softmax_grad_hess,
    _mean_loss,
    _multiclass_mean_loss,
    _renew_leaf_values,
    _softmax_inplace,
    objective_renews_leaves,
    renewal_alpha,
    renewal_weights,
    train,
    train_more,
    train_multiclass,
    train_multiclass_more,
    train_multiclass_with_valid,
    train_with_valid,
)
from .boosting_dart import (
    DartBestState,
    DartParams,
    check_dart_supported,
    dart_advance_scores,
    dart_begin_round,
    dart_commit_round,
    dart_normalization,
    dart_recompute_raw,
    dart_record_best,
    dart_restore_best,
    dart_uniform_weights,
    select_drop,
)
from .boosting_rf import (
    RfParams,
    # The per-class log priors a softmax run starts from. Imported rather
    # than copied for the reason `boosting_rf` gives where it defines them:
    # `train_multiclass` computes the same quantity inline and boosting.mojo
    # does not expose it, so there are two copies already and a third here
    # would be the one that drifts. The dependency is the wrong way round
    # (dart reaching into rf), and it goes away when this is lifted into
    # boosting.mojo, which is where both callers should be reading it from.
    _class_log_priors,
    check_rf_learning_rate,
    is_rf_boosting,
    train_forest_multiclass,
    train_forest_multiclass_with_valid,
    train_forest_with_valid,
    train_rf,
    train_rf_more,
)
from .device import CPU_DEVICE
from .efb import prepare_bundling
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

    `weights` is handed to `select_drop` as well as maintained by it, because
    LightGBM's default drop rule (`uniform_drop=false`) picks an iteration
    with a probability proportional to what it currently weighs.

    `params.bundling` is fitted once here and shared by every round, as in
    `boosting._boost_rounds`. A bundled fit differs from an unbundled one in
    cost and not in result: the trees name original features, so the dropped
    trees' contributions and the leaf renewal below read the original matrix
    either way.

    Single output only: `n_classes` is 1 at every call into `boosting_dart`;
    `_dart_rounds_multiclass` is the round-major counterpart.
    """
    var bundling = prepare_bundling(data, params.bundling)
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
        var drop = select_drop(dart, len(trees), round, weights, 1)
        dart_begin_round(data, trees, weights, drop, 1, raw, contribution)

        refresh_bag(bag, bagging, n, round)
        _fill_grad_hess(
            raw, target, objective, sample_weight, alpha, grad, hess
        )
        var tree = grow_tree(
            data, grad, hess, params.tree, bag, round, bundling
        )
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
      the caller's job to pass the same one.
    - Under `uniform_drop=false`, the drop probabilities of the trees that
      were already there are equalized. That rule reads each iteration's
      weight, the fold has put every weight inside a tree's node values, and
      lifting the ensemble back gives all of them 1.0, so the first continued
      rounds cannot tell an iteration that was rescaled to nothing from one
      that was never dropped. Rounds grown here weigh what they weigh and are
      distinguished normally. LightGBM sidesteps this by never dropping an
      iteration from before the continuation at all: its `tree_weight_` holds
      only the current session's iterations and its drop loop runs over
      `iter_`, this session's count.
    - The existing trees are rewritten, not just appended to. A round that
      drops a tree rescales it, so the ensemble handed back is not the
      ensemble handed in plus new trees.

    The first two are what the fold costs. Both are properties of continuing
    a saved model, not of DART, and both go away with a weight vector on
    `Booster`.

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


def train_dart_with_valid(
    data: BinnedMatrix,
    target: List[Float64],
    valid_data: BinnedMatrix,
    valid_target: List[Float64],
    objective: Int,
    params: BoosterParams,
    early_stopping_rounds: Int,
    dart: DartParams = DartParams.enable(),
    min_delta: Float64 = 0.0,
    sample_weight: List[Float64] = [],
    alpha: Float64 = 0.9,
    bagging: BaggingParams = BaggingParams.disabled(),
) raises -> Booster:
    """Train a DART ensemble with validation-set early stopping.

    The loop is `_dart_rounds` with two additions, and it is written out
    rather than shared because both of them are per-round observations the
    round loop has no reason to make.

    **A second score cache.** The validation rows go through the identical
    round: `dart_begin_round` on the validation matrix removes the same
    dropped iterations from `valid_raw`, and `dart_advance_scores` puts back
    the same fraction and adds the new tree at the same weight. Sharing those
    two functions with the training cache is what makes the two caches
    provably the same arithmetic on different rows. Validation rows are never
    bagged and never weighted, as in `boosting.train_with_valid`.

    **A snapshot instead of a truncation.** A GBDT run records its best round
    and pops trees off the end, which works because the trees it keeps were
    untouched by the rounds it discards. A DART round rescales the trees it
    dropped, so the surviving trees at the end of a run carry weights that
    later rounds gave them, and popping recovers the right tree set with the
    wrong weights. So `dart_record_best` snapshots the weight vector on every
    improvement and `dart_restore_best` truncates and then overwrites. The
    trees themselves need no snapshot: their node values are not touched
    until `fold_weights_into_trees` runs, once, here at the end.

    LightGBM has no counterpart to this function. Its DART overrides
    `EvalAndCheckEarlyStopping` to return false unconditionally, so
    `early_stopping_round` is silently inert under `boosting='dart'` there.
    This trains the same rounds and additionally stops on them.
    """
    if len(target) != data.n_rows:
        raise Error("target length must equal n_rows")
    if len(valid_target) != valid_data.n_rows:
        raise Error("valid_target length must equal valid n_rows")
    if valid_data.n_features != data.n_features:
        raise Error("valid_data must have the same features")
    if params.n_estimators < 0:
        raise Error("num_iterations must be nonnegative")
    if params.learning_rate <= 0.0:
        raise Error("learning_rate must be positive")
    if early_stopping_rounds < 1:
        raise Error("early_stopping_rounds must be positive")
    if not dart.enabled:
        raise Error(
            "train_dart_with_valid needs an enabled DartParams; use"
            " DartParams.enable()"
        )
    check_dart_supported(dart, CPU_DEVICE, False, False, 1)
    check_bagging(bagging)
    _check_objective(objective, target, alpha)
    _check_sample_weight(sample_weight, data.n_rows)
    params.tree.monotone.check_features(data.n_features)

    var bundling = prepare_bundling(data, params.bundling)
    var n = data.n_rows
    var n_valid = valid_data.n_rows
    var base_score = _base_score(target, objective, sample_weight, alpha)
    var raw = List[Float64](capacity=n)
    for _ in range(n):
        raw.append(base_score)
    var valid_raw = List[Float64](capacity=n_valid)
    for _ in range(n_valid):
        valid_raw.append(base_score)

    var signs = params.tree.monotone.active_signs()
    var renews = objective_renews_leaves(objective)
    var renew_w = renewal_weights(objective, target, sample_weight)
    var renew_a = renewal_alpha(objective, alpha)
    var grad = List[Float64](capacity=n)
    var hess = List[Float64](capacity=n)
    var bag = List[Int]()
    var contribution = List[Float64]()
    var valid_contribution = List[Float64]()

    var trees = List[Tree]()
    var weights = List[Float64]()
    var best = DartBestState.initial(
        _mean_loss(valid_raw, valid_target, objective, alpha)
    )

    for i in range(params.n_estimators):
        var drop = select_drop(dart, len(trees), i, weights, 1)
        dart_begin_round(data, trees, weights, drop, 1, raw, contribution)
        dart_begin_round(
            valid_data, trees, weights, drop, 1, valid_raw, valid_contribution
        )

        refresh_bag(bag, bagging, n, i)
        _fill_grad_hess(
            raw, target, objective, sample_weight, alpha, grad, hess
        )
        var tree = grow_tree(data, grad, hess, params.tree, bag, i, bundling)
        if renews:
            _renew_leaf_values(
                tree, data, target, raw, renew_w, renew_a, bag, signs,
                params.tree.extra,
            )

        if tree.n_leaves == 1 and abs(tree.value[0]) < 1e-12:
            # Abandoned whole, as in `_dart_rounds`: both caches go back to
            # what they held before the drop and no weight is recorded for a
            # tree that was never kept.
            for j in range(n):
                raw[j] += contribution[j]
            for j in range(n_valid):
                valid_raw[j] += valid_contribution[j]
            if bagging_enabled(bagging) or not drop.is_empty():
                continue
            break

        var norm = dart_normalization(
            drop.count(), params.learning_rate, dart.xgboost_dart_mode
        )
        var grown = List[Tree]()
        grown.append(tree^)
        # The validation cache first, while `grown` is still readable: the
        # commit below consumes it.
        dart_advance_scores(
            valid_data, drop, norm, 1, valid_contribution, grown, valid_raw
        )
        dart_commit_round(
            data, drop, norm, 1, contribution, grown^, trees, weights, raw
        )

        var loss = _mean_loss(valid_raw, valid_target, objective, alpha)
        if not dart_record_best(best, weights, loss, min_delta):
            if len(trees) - best.n_trees >= early_stopping_rounds:
                break

    dart_restore_best(best, trees, weights)
    fold_weights_into_trees(trees, weights)
    return Booster(
        trees^,
        base_score,
        1.0,
        objective,
        params.tree.monotone.copy(),
    )


def _dart_rounds_multiclass(
    data: BinnedMatrix,
    labels: List[Int],
    n_classes: Int,
    params: BoosterParams,
    sample_weight: List[Float64],
    bagging: BaggingParams,
    dart: DartParams,
    learning_rate: Float64,
    round_offset: Int,
    mut raw: List[Float64],
    mut trees: List[Tree],
    mut weights: List[Float64],
) raises -> Int:
    """The softmax DART loop, and the multiclass counterpart of
    `_dart_rounds`. Returns the number of rounds grown.

    `raw` is row-major (`raw[r * n_classes + k]`) and `trees` is round-major
    (`trees[i * n_classes + k]`), which are the layouts
    `_boost_rounds_multiclass` and `MulticlassBooster` already use, and which
    `boosting_dart` was written for: every entry point there takes
    `n_classes` and walks the same layout.

    **A round drops whole iterations.** All `n_classes` trees of a dropped
    iteration come out together and go back together at one factor, because
    dropping one class's tree alone would tilt the softmax toward the classes
    whose trees survived. That is `DartDrop`'s definition, and it is also
    LightGBM's: `DroppingTrees` selects an iteration index and then loops
    `cur_tree_id` over `num_tree_per_iteration_`.

    The round's probabilities come from the dropped-out `raw`, so every
    class's gradient describes the ensemble the round's trees are actually
    added to. One bag serves the whole round, as in
    `_boost_rounds_multiclass`, and each class's tree still draws its own
    feature set from `round * n_classes + k`.

    Softmax is not a leaf-renewing objective, so a multiclass round has no
    renewal step and needs none of `_dart_rounds`'s renewal state.
    """
    var bundling = prepare_bundling(data, params.bundling)
    var n = data.n_rows
    var prob = List[Float64](capacity=n * n_classes)
    for _ in range(n * n_classes):
        prob.append(0.0)
    var grad = List[Float64](capacity=n)
    var hess = List[Float64](capacity=n)
    var bag = List[Int]()
    var contribution = List[Float64]()
    var grown_rounds = 0

    for i in range(params.n_estimators):
        var round = round_offset + i
        var n_iterations = len(trees) // n_classes
        var drop = select_drop(dart, n_iterations, round, weights, n_classes)
        dart_begin_round(
            data, trees, weights, drop, n_classes, raw, contribution
        )

        refresh_bag(bag, bagging, n, round)
        for r in range(n):
            for k in range(n_classes):
                prob[r * n_classes + k] = raw[r * n_classes + k]
            _softmax_inplace(prob, r * n_classes, n_classes)

        var grown = List[Tree](capacity=n_classes)
        var made_progress = False
        for k in range(n_classes):
            _fill_softmax_grad_hess(
                prob, labels, k, n_classes, sample_weight, grad, hess
            )
            var tree = grow_tree(
                data,
                grad,
                hess,
                params.tree,
                bag,
                round * n_classes + k,
                bundling,
            )
            if tree.n_leaves > 1 or abs(tree.value[0]) >= 1e-12:
                made_progress = True
            grown.append(tree^)

        if not made_progress:
            # No class had anything to add, so the whole round is abandoned
            # and the dropped iterations go back at the weights they had.
            for j in range(len(raw)):
                raw[j] += contribution[j]
            if bagging_enabled(bagging) or not drop.is_empty():
                continue
            break

        var norm = dart_normalization(
            drop.count(), learning_rate, dart.xgboost_dart_mode
        )
        dart_commit_round(
            data,
            drop,
            norm,
            n_classes,
            contribution,
            grown^,
            trees,
            weights,
            raw,
        )
        grown_rounds += 1
    return grown_rounds


def train_dart_multiclass(
    data: BinnedMatrix,
    labels: List[Int],
    n_classes: Int,
    params: BoosterParams,
    dart: DartParams = DartParams.enable(),
    sample_weight: List[Float64] = [],
    bagging: BaggingParams = BaggingParams.disabled(),
) raises -> MulticlassBooster:
    """Train a softmax DART ensemble, LightGBM's `boosting='dart'` with
    `objective='multiclass'`.

    Every argument other than `dart` carries the meaning it has in
    `boosting.train_multiclass`, including the base scores, which are the log
    class priors weighted by `sample_weight`. The returned
    `MulticlassBooster` has a `learning_rate` of 1.0 because every tree's
    weight has been folded into its node values; see the module docstring.
    """
    if len(labels) != data.n_rows:
        raise Error("labels length must equal n_rows")
    if n_classes < 2:
        raise Error("n_classes must be at least 2")
    if params.n_estimators < 0:
        raise Error("num_iterations must be nonnegative")
    if params.learning_rate <= 0.0:
        raise Error("learning_rate must be positive")
    if not dart.enabled:
        raise Error(
            "train_dart_multiclass needs an enabled DartParams; use"
            " DartParams.enable()"
        )
    check_dart_supported(dart, CPU_DEVICE, False, False, n_classes)
    check_bagging(bagging)
    _check_sample_weight(sample_weight, data.n_rows)
    params.tree.monotone.check_features(data.n_features)

    var base_scores = _class_log_priors(labels, n_classes, sample_weight)
    var raw = List[Float64](capacity=data.n_rows * n_classes)
    for _ in range(data.n_rows):
        for k in range(n_classes):
            raw.append(base_scores[k])

    var trees = List[Tree]()
    var weights = List[Float64]()
    _ = _dart_rounds_multiclass(
        data,
        labels,
        n_classes,
        params,
        sample_weight,
        bagging,
        dart,
        params.learning_rate,
        0,
        raw,
        trees,
        weights,
    )
    fold_weights_into_trees(trees, weights)
    return MulticlassBooster(
        trees^,
        base_scores^,
        n_classes,
        1.0,
        params.tree.monotone.copy(),
    )


def train_dart_multiclass_more(
    mut booster: MulticlassBooster,
    data: BinnedMatrix,
    labels: List[Int],
    params: BoosterParams,
    dart: DartParams = DartParams.enable(),
    sample_weight: List[Float64] = [],
    bagging: BaggingParams = BaggingParams.disabled(),
) raises -> Int:
    """Grow `params.n_estimators` more softmax DART rounds and return how many
    rounds were added.

    The single-output continuation's reasoning, unchanged (`train_dart_more`):
    a folded ensemble lifts back into the weighted form as
    `dart_uniform_weights(n_trees, 1.0)`, the rate the continued rounds use is
    the caller's to supply because the folded ensemble no longer records one,
    `uniform_drop=false` cannot tell the pre-existing iterations apart by
    weight for the same reason, the existing trees are rewritten rather than
    only appended to, and the resumed run agrees with a single call on the
    trees it grows but not on the last bits of the scores.

    The base scores are the forest's own, not re-derived from the labels in
    hand, for the reason `boosting.train_multiclass_more` gives: they are the
    priors of the data the ensemble was first fitted on.
    """
    if len(labels) != data.n_rows:
        raise Error("labels length must equal n_rows")
    if params.n_estimators < 0:
        raise Error("num_iterations must be nonnegative")
    if params.learning_rate <= 0.0:
        raise Error("learning_rate must be positive")
    if booster.learning_rate != 1.0:
        raise Error(
            "train_dart_multiclass_more expects an ensemble trained by"
            " train_dart_multiclass: its shrinkage factor must be 1.0,"
            " because a dart round folds each tree's weight into that tree's"
            " node values. Continuing a gbdt ensemble, or a random forest"
            " bridged by RfMulticlassBooster.to_multiclass_booster, here"
            " would discard the factor it carries"
        )
    if not dart.enabled:
        raise Error(
            "train_dart_multiclass_more needs an enabled DartParams; use"
            " DartParams.enable()"
        )
    check_dart_supported(dart, CPU_DEVICE, False, False, booster.n_classes)
    check_bagging(bagging)
    _check_sample_weight(sample_weight, data.n_rows)
    params.tree.monotone.check_features(data.n_features)
    for r in range(len(labels)):
        if labels[r] < 0 or labels[r] >= booster.n_classes:
            raise Error("label out of range")

    var weights = dart_uniform_weights(len(booster.trees), 1.0)
    var raw = dart_recompute_raw(
        data,
        booster.trees,
        weights,
        booster.base_scores,
        booster.n_classes,
    )
    var before = booster.n_iterations()
    var added = _dart_rounds_multiclass(
        data,
        labels,
        booster.n_classes,
        params,
        sample_weight,
        bagging,
        dart,
        params.learning_rate,
        before,
        raw,
        booster.trees,
        weights,
    )
    fold_weights_into_trees(booster.trees, weights)
    return added


def train_dart_multiclass_with_valid(
    data: BinnedMatrix,
    labels: List[Int],
    valid_data: BinnedMatrix,
    valid_labels: List[Int],
    n_classes: Int,
    params: BoosterParams,
    early_stopping_rounds: Int,
    dart: DartParams = DartParams.enable(),
    min_delta: Float64 = 0.0,
    sample_weight: List[Float64] = [],
    bagging: BaggingParams = BaggingParams.disabled(),
) raises -> MulticlassBooster:
    """Train a softmax DART ensemble with validation-set early stopping.

    `train_dart_with_valid` over the round-major layout: the same second
    score cache maintained by the same two functions, and the same weight
    snapshot in place of a truncation. The signal is the multiclass log loss
    of the whole ensemble, and a snapshot is taken only between rounds, so
    the tree count it records is always a whole number of rounds and
    restoring keeps the per-class trees in step.
    """
    if len(labels) != data.n_rows:
        raise Error("labels length must equal n_rows")
    if len(valid_labels) != valid_data.n_rows:
        raise Error("valid_labels length must equal valid n_rows")
    if valid_data.n_features != data.n_features:
        raise Error("valid_data must have the same features")
    if n_classes < 2:
        raise Error("n_classes must be at least 2")
    if params.n_estimators < 0:
        raise Error("num_iterations must be nonnegative")
    if params.learning_rate <= 0.0:
        raise Error("learning_rate must be positive")
    if early_stopping_rounds < 1:
        raise Error("early_stopping_rounds must be positive")
    if not dart.enabled:
        raise Error(
            "train_dart_multiclass_with_valid needs an enabled DartParams;"
            " use DartParams.enable()"
        )
    check_dart_supported(dart, CPU_DEVICE, False, False, n_classes)
    check_bagging(bagging)
    _check_sample_weight(sample_weight, data.n_rows)
    params.tree.monotone.check_features(data.n_features)
    for r in range(len(valid_labels)):
        if valid_labels[r] < 0 or valid_labels[r] >= n_classes:
            raise Error("valid label out of range")

    var bundling = prepare_bundling(data, params.bundling)
    var n = data.n_rows
    var n_valid = valid_data.n_rows
    var base_scores = _class_log_priors(labels, n_classes, sample_weight)
    var raw = List[Float64](capacity=n * n_classes)
    for _ in range(n):
        for k in range(n_classes):
            raw.append(base_scores[k])
    var valid_raw = List[Float64](capacity=n_valid * n_classes)
    for _ in range(n_valid):
        for k in range(n_classes):
            valid_raw.append(base_scores[k])

    var prob = List[Float64](capacity=n * n_classes)
    for _ in range(n * n_classes):
        prob.append(0.0)
    var grad = List[Float64](capacity=n)
    var hess = List[Float64](capacity=n)
    var bag = List[Int]()
    var contribution = List[Float64]()
    var valid_contribution = List[Float64]()

    var trees = List[Tree]()
    var weights = List[Float64]()
    var best = DartBestState.initial(
        _multiclass_mean_loss(valid_raw, valid_labels, n_classes)
    )

    for i in range(params.n_estimators):
        var n_iterations = len(trees) // n_classes
        var drop = select_drop(dart, n_iterations, i, weights, n_classes)
        dart_begin_round(
            data, trees, weights, drop, n_classes, raw, contribution
        )
        dart_begin_round(
            valid_data,
            trees,
            weights,
            drop,
            n_classes,
            valid_raw,
            valid_contribution,
        )

        refresh_bag(bag, bagging, n, i)
        for r in range(n):
            for k in range(n_classes):
                prob[r * n_classes + k] = raw[r * n_classes + k]
            _softmax_inplace(prob, r * n_classes, n_classes)

        var grown = List[Tree](capacity=n_classes)
        var made_progress = False
        for k in range(n_classes):
            _fill_softmax_grad_hess(
                prob, labels, k, n_classes, sample_weight, grad, hess
            )
            var tree = grow_tree(
                data, grad, hess, params.tree, bag, i * n_classes + k, bundling
            )
            if tree.n_leaves > 1 or abs(tree.value[0]) >= 1e-12:
                made_progress = True
            grown.append(tree^)

        if not made_progress:
            for j in range(len(raw)):
                raw[j] += contribution[j]
            for j in range(len(valid_raw)):
                valid_raw[j] += valid_contribution[j]
            if bagging_enabled(bagging) or not drop.is_empty():
                continue
            break

        var norm = dart_normalization(
            drop.count(), params.learning_rate, dart.xgboost_dart_mode
        )
        dart_advance_scores(
            valid_data,
            drop,
            norm,
            n_classes,
            valid_contribution,
            grown,
            valid_raw,
        )
        dart_commit_round(
            data,
            drop,
            norm,
            n_classes,
            contribution,
            grown^,
            trees,
            weights,
            raw,
        )

        var loss = _multiclass_mean_loss(valid_raw, valid_labels, n_classes)
        if not dart_record_best(best, weights, loss, min_delta):
            if len(trees) - best.n_trees >= early_stopping_rounds * n_classes:
                break

    dart_restore_best(best, trees, weights)
    fold_weights_into_trees(trees, weights)
    return MulticlassBooster(
        trees^,
        base_scores^,
        n_classes,
        1.0,
        params.tree.monotone.copy(),
    )


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


def train_boosting_with_valid(
    data: BinnedMatrix,
    target: List[Float64],
    valid_data: BinnedMatrix,
    valid_target: List[Float64],
    objective: Int,
    params: BoosterParams,
    early_stopping_rounds: Int,
    boosting: AlternateBoostingParams = AlternateBoostingParams(),
    min_delta: Float64 = 0.0,
    sample_weight: List[Float64] = [],
    alpha: Float64 = 0.9,
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
    class_bagging: ClassBaggingParams = ClassBaggingParams.disabled(),
) raises -> Booster:
    """Train with validation-set early stopping under the selected mode.

    `gbdt` and `goss` go to `boosting.train_with_valid` unchanged. `rf` goes
    to `boosting_rf.train_forest_with_valid` and is bridged back the way
    `train_rf` bridges an ordinary forest.

    Truncating a forest to its best round count is **exact**, unlike
    truncating a DART ensemble: a forest's trees are independent, so its
    first `k` trees are the forest `n_estimators = k` would have grown, and
    the bridged rate of `1 / k` is read off the truncated model. What early
    stopping measures differs from a boosted run, though, and
    `train_forest_with_valid` says what: a forest's validation loss falls
    because averaging removes variance, so the answer is "how many trees are
    enough", not "has this converged".

    `dart` goes to `train_dart_with_valid`, which does NOT truncate: a round
    after the best one may have rescaled a tree the best round contained, so
    it snapshots the weight vector on every improvement
    (`boosting_dart.dart_record_best`) and restores it at the end
    (`dart_restore_best`).
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
        return train_dart_with_valid(
            data,
            target,
            valid_data,
            valid_target,
            objective,
            params,
            early_stopping_rounds,
            boosting.dart,
            min_delta,
            sample_weight,
            alpha,
            bagging,
        )
    if boosting.mode == BOOSTING_RF:
        _check_rf_uniform_args(class_bagging)
        check_rf_learning_rate(params.learning_rate)
        var forest = train_forest_with_valid(
            data,
            target,
            valid_data,
            valid_target,
            objective,
            RfParams.from_booster_params(params, bagging),
            early_stopping_rounds,
            min_delta,
            sample_weight,
            alpha,
        )
        return forest.to_booster()
    return train_with_valid(
        data,
        target,
        valid_data,
        valid_target,
        objective,
        params,
        early_stopping_rounds,
        min_delta,
        sample_weight,
        alpha,
        bagging,
        goss,
        class_bagging,
    )


def train_boosting_multiclass(
    data: BinnedMatrix,
    labels: List[Int],
    n_classes: Int,
    params: BoosterParams,
    boosting: AlternateBoostingParams = AlternateBoostingParams(),
    sample_weight: List[Float64] = [],
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
) raises -> MulticlassBooster:
    """Train a softmax ensemble under the selected mode.

    `gbdt` and `goss` go to `boosting.train_multiclass` unchanged. `rf` goes
    to `boosting_rf.train_forest_multiclass` and is bridged back by
    `RfMulticlassBooster.to_multiclass_booster`, which is the single-output
    bridge per class: zero base scores, a `learning_rate` of `1 / rounds`,
    and correctness for the whole ensemble rather than for iteration ranges.

    There is no balanced-bagging argument here, as there is none on
    `boosting.train_multiclass`: LightGBM's `pos_bagging_fraction` /
    `neg_bagging_fraction` are binary-classification controls, which is what
    `boosting_rf.check_rf_params` says for a forest.

    `dart` goes to `train_dart_multiclass`, whose rounds drop whole
    iterations so that a round's per-class trees are hidden and restored
    together.
    """
    boosting.validate(goss)
    if boosting.mode == BOOSTING_DART:
        return train_dart_multiclass(
            data,
            labels,
            n_classes,
            params,
            boosting.dart,
            sample_weight,
            bagging,
        )
    if boosting.mode == BOOSTING_RF:
        check_rf_learning_rate(params.learning_rate)
        var forest = train_forest_multiclass(
            data,
            labels,
            n_classes,
            RfParams.from_booster_params(params, bagging),
            sample_weight,
        )
        return forest.to_multiclass_booster()
    return train_multiclass(
        data, labels, n_classes, params, sample_weight, bagging, goss
    )


def train_boosting_multiclass_more(
    mut booster: MulticlassBooster,
    data: BinnedMatrix,
    labels: List[Int],
    params: BoosterParams,
    boosting: AlternateBoostingParams = AlternateBoostingParams(),
    sample_weight: List[Float64] = [],
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
) raises -> Int:
    """Continue a softmax ensemble under the selected mode, returning how
    many rounds were added.

    `rf` is **refused**, unlike the single-output path. `boosting_rf` bridges
    a forest's continuation for single-output runs (`train_rf_more`) by
    recomputing the constant its new trees are fitted at from `target`; the
    multiclass constant is a vector of class log priors and `boosting_rf`
    exposes no bridged multiclass continuation, so there is nothing here to
    dispatch to and recomputing the priors in this module would be a second
    copy of an algorithm that already exists. Continue such a forest as an
    `RfMulticlassBooster` through `train_forest_multiclass_more`, which
    carries its own base scores and needs no precondition at all.

    `dart` goes to `train_dart_multiclass_more`, which checks the ensemble's
    unit shrinkage factor first, exactly as the single-output continuation
    does, and rewrites the existing trees rather than only appending to them.
    """
    boosting.validate(goss)
    if boosting.mode == BOOSTING_RF:
        raise Error(
            "boosting='rf' cannot continue a bridged MulticlassBooster: the"
            " class log priors its trees were fitted at are folded away by"
            " to_multiclass_booster and boosting_rf exposes no bridged"
            " multiclass continuation. Hold the forest as an"
            " RfMulticlassBooster and call"
            " boosting_rf.train_forest_multiclass_more"
        )
    if boosting.mode == BOOSTING_DART:
        return train_dart_multiclass_more(
            booster,
            data,
            labels,
            params,
            boosting.dart,
            sample_weight,
            bagging,
        )
    return train_multiclass_more(
        booster, data, labels, params, sample_weight, bagging, goss
    )


def train_boosting_multiclass_with_valid(
    data: BinnedMatrix,
    labels: List[Int],
    valid_data: BinnedMatrix,
    valid_labels: List[Int],
    n_classes: Int,
    params: BoosterParams,
    early_stopping_rounds: Int,
    boosting: AlternateBoostingParams = AlternateBoostingParams(),
    min_delta: Float64 = 0.0,
    sample_weight: List[Float64] = [],
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
) raises -> MulticlassBooster:
    """Train a softmax ensemble with validation-set early stopping under the
    selected mode.

    `rf` goes to `boosting_rf.train_forest_multiclass_with_valid`, whose
    truncation drops whole rounds so the per-class trees stay in step, and is
    bridged back the same way `train_boosting_multiclass` bridges. `dart`
    goes to `train_dart_multiclass_with_valid`, which snapshots weights
    rather than truncating, for the reason `train_boosting_with_valid` gives.
    """
    boosting.validate(goss)
    if boosting.mode == BOOSTING_DART:
        return train_dart_multiclass_with_valid(
            data,
            labels,
            valid_data,
            valid_labels,
            n_classes,
            params,
            early_stopping_rounds,
            boosting.dart,
            min_delta,
            sample_weight,
            bagging,
        )
    if boosting.mode == BOOSTING_RF:
        check_rf_learning_rate(params.learning_rate)
        var forest = train_forest_multiclass_with_valid(
            data,
            labels,
            valid_data,
            valid_labels,
            n_classes,
            RfParams.from_booster_params(params, bagging),
            early_stopping_rounds,
            min_delta,
            sample_weight,
        )
        return forest.to_multiclass_booster()
    return train_multiclass_with_valid(
        data,
        labels,
        valid_data,
        valid_labels,
        n_classes,
        params,
        early_stopping_rounds,
        min_delta,
        sample_weight,
        bagging,
        goss,
    )


def fit_boosting[
    features_origin: ImmOrigin, //
](
    features: Span[Float64, features_origin],
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
