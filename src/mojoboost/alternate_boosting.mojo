"""One entry point for LightGBM's `boosting` parameter.

LightGBM's `boosting` (aliased `boosting_type`) selects the training
strategy: `gbdt`, `goss`, `dart`, or `rf`. mojoboost has always implemented
the first two, through `boosting.train` and a `goss.GossParams` handed to
it. This module adds the other two by dispatching to `boosting_dart` and
`boosting_rf`, and it is the only place the four names are resolved, so a
caller does not choose a trainer by picking a function.

Why the dispatch lives here rather than in `boosting.train`
-----------------------------------------------------------
It should not. `boosting.train` is the production entry point and the right
long-term home for the mode argument; this module exists because the DART
and RF loops are new and are kept behind their own door until they have been
run. `handoffs/connect_17_alternate_boosting.md` records the exact edit that
folds this into `boosting.train`, `params.parse_params`, and the Python
layer, and what this module then becomes (a re-export, or nothing).

Nothing here is a second trainer, a second registry, or a second model
representation. `gbdt` and `goss` route into the untouched production
`boosting.train` with the arguments they always took; `dart` and `rf` route
into loops that themselves call `tree.grow_tree` and return an ordinary
`boosting.Booster`. Every mode's output serializes through
`serialize.save_model` with no format change and predicts through the same
`Booster.predict_raw_row`.

What each mode means, in one line
---------------------------------
- `gbdt`: every tree fitted to the residual of the trees before it, summed.
- `goss`: `gbdt` with each round's rows sampled by gradient magnitude
  (`goss.mojo`); the ensemble is unchanged.
- `dart`: a random subset of the ensemble is dropped before each round and
  the round rescales what it dropped (`boosting_dart.mojo`).
- `rf`: every tree fitted to the same gradients on its own row bag, and the
  ensemble averaged rather than summed (`boosting_rf.mojo`).

Combinations that are refused rather than resolved
--------------------------------------------------
- `dart` or `rf` with GOSS. Both refuse for reasons their own modules give.
- `rf` with a learning rate, `rf` without bagging: see
  `boosting_rf.check_rf_params`.
- `dart` with `uniform_drop=False`: see `boosting_dart.DartParams.validate`.
- DART parameters set under a non-DART mode are NOT refused here, because
  `DartParams` has no "unset" state to detect and a caller who leaves the
  default bundle in place has set nothing. The parameter-string layer is
  where that check belongs, and the handoff asks for it there.
- Multiclass and ranking under `dart` or `rf`. Both are single-output here;
  `train_multiclass`, `train_ranker`, and the sparse and GPU trainers keep
  the modes they have and are untouched by this module.
"""

from .bagging import BaggingParams
from .binning import BinMapper, BinnedMatrix, fit_bins
from .boosting import Booster, BoosterParams, train, train_more
from .boosting_dart import DartParams, train_dart, train_dart_more
from .boosting_rf import train_rf, train_rf_more
from .goss import GossParams
from .model import Model

# LightGBM's four `boosting` values, as codes. Ordered as LightGBM
# documents them; the numbers are internal and are not serialized, since a
# fitted model records no mode (see the module docstring).
comptime BOOSTING_GBDT = 0
comptime BOOSTING_GOSS = 1
comptime BOOSTING_DART = 2
comptime BOOSTING_RF = 3


def parse_boosting(name: String) raises -> Int:
    """A `boosting` code from LightGBM's name for it.

    LightGBM's aliases are accepted (`gbrt`, `gradient_boosting_decision_tree`
    and friends for `gbdt`, `random_forest` for `rf`). Names are canonical
    lowercase, as in `device.parse_device` and `params.objective_from_name`.
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
    """The mode and the parameters only one mode reads.

    Kept in one bundle so a caller passes one thing rather than choosing an
    entry point, and so the mode and its parameters cannot be separated on
    the way down. `dart` is ignored under any mode but `dart`; see the module
    docstring for why that is not refused here.
    """

    var mode: Int
    var dart: DartParams

    def __init__(out self):
        """Plain GBDT, LightGBM's default."""
        self.mode = BOOSTING_GBDT
        self.dart = DartParams()

    @staticmethod
    def default() -> AlternateBoostingParams:
        return AlternateBoostingParams()

    @staticmethod
    def of(mode: Int) raises -> AlternateBoostingParams:
        """A bundle for `mode`, with LightGBM's DART defaults in place."""
        var out = AlternateBoostingParams()
        _ = boosting_name(mode)
        out.mode = mode
        return out^

    @staticmethod
    def named(name: String) raises -> AlternateBoostingParams:
        """A bundle for LightGBM's spelling of a mode."""
        return AlternateBoostingParams.of(parse_boosting(name))

    def validate(self, goss: GossParams) raises:
        """Reject a mode that disagrees with the row sampler it was handed.

        `goss` is a separate argument everywhere in the Mojo API (it carries
        rates and a seed that no other mode reads), so the mode and the
        sampler can contradict each other, and a contradiction means the
        caller asked for two strategies at once. Neither is silently
        preferred.
        """
        _ = boosting_name(self.mode)
        if self.mode == BOOSTING_GOSS:
            if not goss.enabled:
                raise Error(
                    "boosting='goss' needs an enabled GossParams; it is the"
                    " one that carries top_rate, other_rate, and the seed"
                )
            return
        if goss.enabled:
            raise Error(
                "boosting='",
                boosting_name(self.mode),
                "' was given an enabled GossParams; GOSS is itself a"
                " boosting value, so the two name different strategies",
            )
        if self.mode == BOOSTING_DART:
            self.dart.validate()


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
    changes is which trees are in it and what shrinkage factor they share
    (see the module docstring and `boosting_rf` / `boosting_dart`).
    """
    boosting.validate(goss)
    if boosting.mode == BOOSTING_DART:
        return train_dart(
            data,
            target,
            objective,
            params,
            sample_weight,
            alpha,
            bagging,
            boosting.dart,
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
    before they touch it (`boosting_rf.is_forest`, and DART's unit shrinkage
    factor), so continuing a GBDT model as a forest, or a forest as GBDT, is
    reported rather than performed.

    DART continuation rewrites trees the ensemble already holds, because a
    round that drops a tree rescales it. It is the one mode for which the
    ensemble handed back is not the one handed in plus new trees; see
    `boosting_dart.train_dart_more`.
    """
    boosting.validate(goss)
    if boosting.mode == BOOSTING_DART:
        return train_dart_more(
            booster,
            data,
            target,
            params,
            sample_weight,
            alpha,
            bagging,
            boosting.dart,
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
    `model.Model`, which `serialize.save_model` writes and every predictor
    reads. The handoff asks for the mode to move onto `model.fit`, at which
    point this function goes away.

    CPU only. `train_gpu` and the sparse trainers carry their own round
    loops, so a mode reaches them through their own edits, which this lane
    does not own and which the handoff records.
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
