"""Random-forest boosting mode: LightGBM's `boosting=rf`.

This is not row bagging. Ordinary GBDT with `bagging_fraction` (bagging.mojo)
still boosts: every round recomputes gradients at the *current* raw scores, so
tree `i + 1` corrects what trees `0..i` got wrong, and the ensemble is a sum.
Random-forest mode drops the correction entirely:

- gradients and hessians are computed **once**, at the constant raw score
  `base_score`, and every tree in the run fits those same numbers;
- the trees therefore differ only through the randomization the sampler and
  the feature draw supply, which is why one of them has to be on;
- the ensemble is an **average**, not a sum, and there is no shrinkage.

LightGBM implements this in `src/boosting/rf.hpp` as a `GBDT` subclass that
sets `average_output_ = true`, forces `shrinkage_rate_ = 1.0`, calls
`Boosting()` once from `Init` rather than once per iteration, and keeps the
training score as a running mean (`MultiplyScore` by `iter`, add the tree,
`MultiplyScore` by `1 / (iter + 1)`). Everything below is that model.

Where the bias lives
--------------------
LightGBM folds the base score into the tree: a grown tree's leaf values are
Newton steps taken from `init_scores_`, and `Tree::AddBias(init_score)` shifts
every node so that each tree predicts on the raw scale by itself. Averaging
raw-scale trees is then the whole prediction and there is no separate base
score to add. `RfBooster` stores its trees the same way, so a tree here is the
tree LightGBM would have written to its model file.

`RfBooster.base_score` is kept anyway, because two things need the constant
the trees were fitted from and cannot recover it from them: leaf renewal
(below) and continued training. LightGBM keeps it in `init_scores_`, which is
training state it does not persist -- see INTENTIONAL DIFFERENCES.

Leaf renewal
------------
For the three objectives whose Newton step carries no curvature (QUANTILE,
L1, MAPE -- `objective_renews_leaves`), LightGBM replaces each leaf value
with a percentile of the residuals. In GBDT the residual is
`label - current score`; in RF it is `label - init_score`, a constant offset,
because there is no current score to speak of. `boosting._renew_leaf_values`
takes the raw-score vector as an argument, so passing it the constant vector
is exactly LightGBM's `residual_getter` and no second renewal path exists
here. Renewal runs before `AddBias`, as it does in LightGBM: for a constant
offset `c`, `percentile(y - c) + c` is `percentile(y)`, so an L1 forest's
leaves end up at the median of the labels in the leaf, which is what a random
forest is supposed to produce.

Required randomization
----------------------
Because every tree fits identical gradients, a run with no per-tree
randomization would grow `n_estimators` copies of one tree and average them
back to that tree. LightGBM refuses such a configuration in `RF::Init`:

    if data_sample_strategy == "bagging":
        CHECK((bagging_freq > 0 && 0 < bagging_fraction < 1)
              || (0 < feature_fraction < 1))
    else:
        CHECK_EQ(data_sample_strategy, "goss")

`check_rf_params` is that check. See `rf_randomizer_name` for the accepted
sources and the message a rejected configuration gets.

What RF mode does not accept
----------------------------
- **Custom objectives.** LightGBM: "RF mode do not support custom objective
  function, please use built-in objectives." A custom gradient callback would
  be called once, at a constant score, which is not what any caller means.
- **Initial scores.** LightGBM `CHECK_EQ(train_data->metadata().init_score(),
  nullptr)` for a fresh run. A per-row offset is a boosting concept: it shifts
  the point gradients are taken at, and RF takes them at one shared constant.
- **Ranking.** `lambdarank` gradients at a constant score rank nothing, and
  LightGBM's ranker is not reachable through `boosting=rf` in practice.
  `check_rf_objective` rejects it by name rather than training a forest of
  identical, meaningless trees.
- **learning_rate.** `RfParams` has no such field. `check_rf_learning_rate`
  exists so that a parameter layer can reject an explicitly set rate instead
  of quietly ignoring it, which is what LightGBM does.

Determinism
-----------
Every per-tree decision reads the absolute round index, exactly as
`boosting._boost_rounds` does:

- the bag is bag number `round // bagging_freq` (bagging.mojo), so a
  `bagging_freq` above 1 reuses one bag for that many consecutive trees, as in
  LightGBM;
- the feature set is drawn from `(feature_fraction_seed, tree_index)`
  (sampling.mojo), with `tree_index = round` for a single-output forest and
  `round * n_classes + k` for a multiclass one, so no two trees in a run share
  a feature draw;
- the GOSS sample is drawn from `(goss.seed, round)` (goss.mojo).

None of the three carries state between trees, so tree `i` is the same tree
whether it was grown in one call of 100 rounds or in the second of two calls
of 50, and the CPU draws what any other backend would.

INTENTIONAL DIFFERENCES FROM LightGBM
-------------------------------------
- **Continued training keeps the base score.** LightGBM recomputes
  `init_scores_` in `Boosting()`, and `BoostFromAverage` returns 0 once
  `models_` is non-empty, so trees added to an existing rf model are fitted at
  a raw score of 0 rather than at the base score the first trees used. That
  makes `50 + 50` rounds a different model from `100`. `RfBooster` stores the
  base score and `train_forest_more` reuses it, so the two agree here.
- **Degenerate trees keep their value.** When no split passes the tree
  constraints, LightGBM discards the grown stump and appends a tree whose
  single leaf is 0 (`AsConstantTree(0.0)` on the first iteration, a
  default-constructed tree after that), which pulls the average toward zero
  rather than toward the objective's own constant. Here the grown stump is
  renewed, biased, and kept like any other tree, so a forest that can find no
  split predicts the base score instead of nothing. A round is never skipped
  or dropped either: `n_estimators` trees are always grown, because the
  denominator of the average is the tree count and a silent early stop would
  change it. GBDT's "converged, stop early" rule has no meaning here -- the
  gradients never change, so no tree can be more converged than the first.
- **GOSS does not corrupt the shared gradients.** LightGBM's GOSS scales the
  sampled rows' gradients in place, and since rf never recomputes them, the
  multipliers compound across rounds. Each round here scales a copy.
- **The GOSS warmup is keyed to the rf shrinkage.** LightGBM skips sampling
  for `int(1 / config.learning_rate)` rounds, reading the configured
  `learning_rate` even though rf has forced its shrinkage to 1; at the default
  0.1 that is ten full-data rounds, which under constant gradients are ten
  identical trees. `RF_SHRINKAGE` is passed instead, so the warmup is one
  round.
- The bag, the feature draw, and the GOSS sample come from counter-based
  splitmix64 rather than LightGBM's per-block LCG, so the individual rows and
  features drawn differ at equal seeds. The schedules, the counts, and the
  distributions match. This is the trade bagging.mojo and goss.mojo already
  make, for the reason they give.
- Balanced (class-conditional) bagging counts as randomization here.
  LightGBM's `CHECK` reads `bagging_fraction` alone and would abort on a run
  randomized by `pos_bagging_fraction` / `neg_bagging_fraction`, although it
  is the same per-round row draw. See `rf_randomizer_name`.

Two layers, and which one to call
--------------------------------
`train_forest` and `RfBooster` are the algorithm: an averaged model, with
LightGBM's slice semantics and a base score it can continue from. `train_rf`,
`train_rf_more`, and `is_forest` are the `boosting='rf'` surface: the same
training, handed the arguments every other mode takes and returning an
ordinary `Booster` through `RfBooster.to_booster`. `alternate_boosting.mojo`
dispatches to the second layer, which is why an rf model needs no change to
serialization, prediction, contributions, or importance today.

The bridge is exact for a full-model prediction and lossy for two things
only, both listed on `to_booster`: iteration ranges, and the base score a
continuation needs. Neither is repairable without a flag on the model saying
"average these trees", which is the first patch in
`handoffs/remaining_02_rf.md`.

Reachability
------------
`src/mojoboost/alternate_boosting.mojo` imports `train_rf` and
`train_rf_more`, so a Mojo caller can train a forest today. Nothing above
that reaches it: `boosting=rf` is still rejected by params.mojo and by the
Python estimators, `average_output` is still refused by lgbm_model_io.mojo,
and no parameter string or keyword turns the mode on. This module
deliberately does not half-wire any of that: an accepted `boosting=rf` that
trained a summed ensemble would be worse than one that is rejected.
"""

from std.math import log

from .bagging import (
    DEFAULT_BAGGING_SEED,
    BaggingParams,
    bagging_enabled,
    check_bagging,
    refresh_bag,
)
from .binning import BinnedMatrix
from .boosting import (
    SQUARED_ERROR,
    Booster,
    BoosterParams,
    CUSTOM,
    IterationRange,
    MulticlassBooster,
    _base_score,
    _check_class_bagging,
    _check_goss,
    _check_objective,
    _check_sample_weight,
    _clamp_prob,
    _fill_grad_hess,
    _fill_softmax_grad_hess,
    _mean_loss,
    _multiclass_goss_select,
    _multiclass_mean_loss,
    _renew_leaf_values,
    _softmax_inplace,
    objective_renews_leaves,
    renewal_alpha,
    renewal_weights,
)
from .goss import GossParams, GossSelection, apply_goss_scaling, goss_round
from .monotone import MonotoneConstraints
from .ranking import LAMBDARANK
from .sampling import (
    ClassBaggingParams,
    check_feature_fractions,
    has_positive_rows,
    refresh_class_bag,
)
from .tree import Tree, TreeParams, grow_tree

# The shrinkage LightGBM's rf forces (`RF::ResetConfig`: "not shrinkage rate
# for the RF", `shrinkage_rate_ = 1.0f`). Trees are averaged, not summed and
# scaled, so there is no rate to set; this constant exists so that the places
# that must name one -- `RfParams.booster_params`, the GOSS warmup -- name the
# same one.
comptime RF_SHRINKAGE = 1.0

# LightGBM's accepted spellings of `boosting=rf`. A parameter layer resolves
# the name with `is_rf_boosting` so that the two spellings cannot drift apart
# from the trainers that implement them.
comptime RF_BOOSTING_NAMES = String("rf random_forest")

# What `rf_randomizer_name` reports. A caller that has to explain a run (a
# dump, a log line, an error) picks its wording from these rather than from a
# second vocabulary.
comptime RF_RANDOM_BAGGING = String("bagging_fraction")
comptime RF_RANDOM_CLASS_BAGGING = String("pos_bagging_fraction")
comptime RF_RANDOM_GOSS = String("goss")
comptime RF_RANDOM_FEATURE_FRACTION = String("feature_fraction")

# Softmax multiclass has no objective code in the single-output space that the
# sampler checks read (params.mojo gives it a negative one of its own). The one
# rule those checks apply per objective is that balanced bagging is
# binary-only, so any non-binary code carries it; this names which one is
# passed rather than leaving a bare `SQUARED_ERROR` in a multiclass call.
comptime _NOT_BINARY = SQUARED_ERROR


def is_rf_boosting(value: String) -> Bool:
    """Whether `value` names random-forest mode.

    LightGBM accepts `rf` and `random_forest` for `boosting` (equivalently
    `boosting_type`). Names are canonical lowercase, as everywhere else in the
    parameter layer.
    """
    for name in RF_BOOSTING_NAMES.split():
        if value == name:
            return True
    return False


def check_rf_learning_rate(learning_rate: Float64) raises:
    """Reject a learning rate under random-forest mode.

    LightGBM accepts `learning_rate` alongside `boosting=rf` and overrides it
    (`shrinkage_rate_ = 1.0f`), so the number the user set is not the number
    the model trains with -- and unlike the parameters params.mojo already
    guards, this one silently changes nothing at all. A caller that knows the
    rate was set explicitly calls this; a caller that only has a defaulted
    rate must not, or every default-constructed configuration would fail.
    """
    if learning_rate != RF_SHRINKAGE:
        raise Error(
            "learning_rate does not apply to boosting='rf': a random forest"
            " averages its trees and applies no shrinkage (LightGBM forces"
            " shrinkage to 1 and ignores the value)"
        )


@fieldwise_init
struct RfParams(Copyable, Movable):
    """Random-forest configuration.

    `tree` is the ordinary `TreeParams`: every tree control -- `num_leaves`,
    `min_data_in_leaf`, the regularizers, `max_depth`, monotone and
    interaction constraints, the categorical settings, and the whole `extra`
    bundle -- applies to a forest exactly as it does to a boosted ensemble,
    because the same grower grows it.

    `bagging`, `goss`, and `class_bagging` are the three row samplers, no two
    of which can be on at once (`_check_goss`, `_check_class_bagging`). At
    least one of them, or `tree.feature_fraction < 1`, must randomize the run;
    see `check_rf_params`.

    There is no `learning_rate`: see `RF_SHRINKAGE`.
    """

    var n_estimators: Int
    var tree: TreeParams
    var bagging: BaggingParams
    var goss: GossParams
    var class_bagging: ClassBaggingParams

    @staticmethod
    def default() -> RfParams:
        """100 trees on 80% row bags redrawn every round.

        LightGBM has no default that satisfies its own rf check -- its
        `bagging_fraction` default of 1.0 and `bagging_freq` default of 0 are
        exactly the combination `RF::Init` aborts on -- so a user of
        `boosting=rf` must set the sampler themselves. A default that cannot
        be trained with is worse than useless here, so this one is a working
        forest: `bagging_fraction=0.8`, `bagging_freq=1`, and LightGBM's
        `bagging_seed` of 3.
        """
        return RfParams(
            100,
            TreeParams.default(),
            BaggingParams(0.8, 1, DEFAULT_BAGGING_SEED),
            GossParams.disabled(),
            ClassBaggingParams.disabled(),
        )

    @staticmethod
    def from_booster_params(
        params: BoosterParams, bagging: BaggingParams
    ) -> RfParams:
        """The forest a `(BoosterParams, BaggingParams)` pair describes.

        The bridge for callers that hold the ordinary training arguments and
        select the mode separately, which is how `boosting=rf` arrives from
        `alternate_boosting.train_boosting`. `params.learning_rate` is not
        read: `check_rf_learning_rate` is what says so out loud, and the
        adapters below call it.
        """
        return RfParams(
            params.n_estimators,
            params.tree.copy(),
            bagging.copy(),
            GossParams.disabled(),
            ClassBaggingParams.disabled(),
        )

    def booster_params(self) -> BoosterParams:
        """The `BoosterParams` an rf run is equivalent to, with the shrinkage
        LightGBM forces. Only for handing to code that takes a `BoosterParams`
        and reads the tree half of it; the boosting loops here do not use it,
        since summing at rate 1 is not what a forest does."""
        return BoosterParams(self.n_estimators, RF_SHRINKAGE, self.tree.copy())


def rf_randomizer_name(params: RfParams) -> String:
    """Which source randomizes this forest, or an empty string for none.

    Sources are tested in LightGBM's own order, so the name reported for a run
    with more than one plausible source is the one LightGBM's `CHECK` would
    have been satisfied by first. `RF_RANDOM_CLASS_BAGGING` is the one entry
    with no LightGBM counterpart: its `CHECK` reads `bagging_fraction` and
    would abort on a run randomized by the class-conditional fractions alone,
    even though those draw a fresh bag per round in exactly the way the
    uniform ones do. Accepting it is the module docstring's last intentional
    difference.
    """
    if bagging_enabled(params.bagging):
        return RF_RANDOM_BAGGING
    if params.goss.enabled:
        return RF_RANDOM_GOSS
    if params.tree.feature_fraction < 1.0:
        return RF_RANDOM_FEATURE_FRACTION
    if params.class_bagging.enabled():
        return RF_RANDOM_CLASS_BAGGING
    return String("")


def check_rf_objective(objective: Int) raises:
    """Reject the objectives random-forest mode cannot serve.

    `_check_objective` (boosting.mojo) still owns every other objective rule
    and the label checks; this adds only the two exclusions that are rf's own.
    """
    if objective == CUSTOM:
        raise Error(
            "boosting='rf' does not support custom objectives: every tree"
            " fits the gradients of one constant score, so the callback would"
            " be evaluated once and never again (LightGBM refuses the same"
            " combination)"
        )
    if objective == LAMBDARANK:
        raise Error(
            "boosting='rf' does not support 'lambdarank': lambda gradients at"
            " a constant score carry no ranking, so every tree in the forest"
            " would be fitted to the same rank-free target"
        )


def check_rf_params(params: RfParams, objective: Int) raises:
    """Validate a random-forest configuration before any data is read.

    Everything a boosted run checks is checked the same way and by the same
    functions -- the bagging ranges, the GOSS rates, the sampler exclusivity,
    the feature fractions -- plus the two rules that are rf's own: an
    objective it can serve, and at least one source of per-tree randomization.

    The randomization rule is `RF::Init`'s `CHECK`, and it is not a nicety:
    with the gradients fixed for the whole run, an unrandomized forest is
    `n_estimators` copies of one tree averaged back to that tree, at
    `n_estimators` times the cost of growing it once.
    """
    if params.n_estimators < 0:
        raise Error("num_iterations must be nonnegative")
    check_rf_objective(objective)
    check_bagging(params.bagging)
    _check_goss(params.goss, params.bagging)
    _check_class_bagging(
        params.class_bagging, params.bagging, params.goss, objective
    )
    check_feature_fractions(
        params.tree.feature_fraction,
        params.tree.feature_fraction_bynode,
        params.tree.feature_fraction_bylevel,
    )
    if rf_randomizer_name(params).byte_length() == 0:
        raise Error(
            "boosting='rf' needs a source of per-tree randomness: set"
            " bagging_fraction < 1 with bagging_freq > 0, or"
            " feature_fraction < 1, or data_sample_strategy=goss. Every tree"
            " in a forest fits the same gradients, so an unrandomized run"
            " grows num_iterations copies of one tree"
        )


def check_rf_init_score(init_score: List[Float64]) raises:
    """Reject LightGBM's `init_score` under random-forest mode.

    A per-row offset moves the point each row's gradient is taken at, and a
    forest takes every gradient at one shared constant; there is no per-row
    point for the offset to move. LightGBM checks the same thing
    (`CHECK_EQ(train_data->metadata().init_score(), nullptr)`) for a run that
    is not continuing an existing model.
    """
    if len(init_score) != 0:
        raise Error(
            "init_score does not apply to boosting='rf': every tree fits the"
            " gradients of one shared constant score, so there is no per-row"
            " starting point to offset"
        )


def _add_bias(mut tree: Tree, bias: Float64):
    """LightGBM's `Tree::AddBias`: shift every node's value so the tree
    predicts on the raw scale by itself.

    Internal nodes are shifted with the leaves. Their values are the outputs
    the grower gave them, which `path_smooth` reads as a parent output and
    which feature contributions read as a node's expected value, and both of
    those are raw-scale quantities once the tree is.

    A shift leaves every split, every count, and every ordering untouched, so
    a monotone tree stays monotone and a renewed leaf stays renewed.
    """
    if bias == 0.0:
        return
    for i in range(len(tree.value)):
        tree.value[i] += bias


def _same_signs(a: List[Int], b: List[Int]) -> Bool:
    """Whether two monotonic constraint vectors are the same vector. The test
    `boosting._same_signs` makes, repeated only because that one is private to
    its module; the handoff asks for the single copy."""
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def _constant_scores(n: Int, value: Float64) -> List[Float64]:
    """The raw-score vector every tree in a forest is fitted at."""
    var raw = List[Float64](capacity=n)
    raw.resize(n, value)
    return raw^


def _objective_response(objective: Int, raw: Float64) -> Float64:
    """The objective's inverse link, `Booster.response` exactly.

    Delegated to an empty `Booster` rather than re-cased here: the link is a
    property of the objective, and two copies of it would be two things to
    keep in step. The instance holds three empty lists and is built and
    dropped per call; see the handoff for the request to lift `response` out
    of `Booster` so this indirection can go.
    """
    return Booster(List[Tree](), 0.0, RF_SHRINKAGE, objective).response(raw)


struct RfBooster(Copyable, Movable):
    """A fitted single-output random forest.

    Predictions are the **mean** of the trees, not their sum: `trees` carry
    the base score in their node values (see `_add_bias`), so averaging them
    is the whole raw score and nothing is added afterwards. This is LightGBM's
    `average_output_` model, and it is why an `RfBooster` is not a `Booster`
    -- a consumer that summed these trees would get `n_estimators` times the
    right answer.

    `base_score` is the constant every tree was fitted from. It is not part of
    prediction (the trees already carry it); it is what `train_forest_more` needs
    to add trees that belong to the same forest, and what a reader needs to
    interpret a single tree's values.

    `monotone` records the constraints every tree satisfies, as on `Booster`.
    A mean of monotone trees is monotone, so the claim survives averaging.
    """

    var trees: List[Tree]
    var base_score: Float64
    var objective: Int
    var monotone: MonotoneConstraints

    def __init__(
        out self,
        var trees: List[Tree],
        base_score: Float64,
        objective: Int,
        var monotone: MonotoneConstraints = MonotoneConstraints(),
    ):
        self.trees = trees^
        self.base_score = base_score
        self.objective = objective
        self.monotone = monotone^

    @always_inline
    def n_iterations(self) -> Int:
        """Trees in the forest. One iteration is one tree, as for a
        single-output `Booster`."""
        return len(self.trees)

    def average_weight(self) -> Float64:
        """The weight each tree carries in a full-model prediction, `1 / T`.
        Zero for an empty forest, which predicts its base score instead."""
        if len(self.trees) == 0:
            return 0.0
        return 1.0 / Float64(len(self.trees))

    @always_inline
    def response(self, raw: Float64) -> Float64:
        """The objective's inverse link, as on `Booster`."""
        return _objective_response(self.objective, raw)

    def predict_raw_bins_range(
        self, bins: List[Int], rng: IterationRange
    ) -> Float64:
        """Raw output of the trees in `rng`: their mean.

        LightGBM slices a random forest the same way, dividing by the number
        of iterations in the slice rather than by the forest's size
        (`GBDT::Predict` divides by `num_iteration_for_pred_`). Two
        consequences follow, and both are why an averaged model cannot be
        served by `Booster`'s range arithmetic:

        - ranges do not add. `[0, k)` and `[k, n)` are each a full prediction
          from their own trees, so they average rather than sum to the whole
          model. In a boosted ensemble they sum to it exactly.
        - the base score is not a term of the sum at all, so unlike
          `Booster.predict_raw_bins_range` there is no "does this range
          include iteration 0" question to answer. Every tree carries it.

        An empty range has no trees to average and yields the base score: the
        prediction of a forest of no trees, wherever the empty slice was taken.
        """
        if rng.is_empty():
            return self.base_score
        var total = 0.0
        for i in range(rng.start, rng.stop):
            total += self.trees[i].predict_bins(bins)
        return total / Float64(rng.n_iterations())

    def predict_raw_bins(self, bins: List[Int]) -> Float64:
        """Raw prediction from the whole forest."""
        return self.predict_raw_bins_range(
            bins, IterationRange(0, len(self.trees))
        )

    def predict_bins(self, bins: List[Int]) -> Float64:
        """Response-scale prediction from the whole forest."""
        return self.response(self.predict_raw_bins(bins))

    def predict_bins_range(
        self, bins: List[Int], rng: IterationRange
    ) -> Float64:
        """Response-scale prediction from the trees in `rng` alone."""
        return self.response(self.predict_raw_bins_range(bins, rng))

    def predict_raw_row(self, data: BinnedMatrix, row: Int) -> Float64:
        """Raw prediction for one row of an already binned matrix."""
        if len(self.trees) == 0:
            return self.base_score
        var total = 0.0
        for i in range(len(self.trees)):
            total += self.trees[i].predict_row(data, row)
        return total / Float64(len(self.trees))

    def predict_row(self, data: BinnedMatrix, row: Int) -> Float64:
        """Response-scale prediction for one row of a binned matrix."""
        return self.response(self.predict_raw_row(data, row))

    def leaf_ordinals_range(self, rng: IterationRange) -> List[List[Int]]:
        """The per-node leaf ordinal table of each tree in `rng`, as on
        `Booster`. Leaf indices depend on the splits alone, so averaging
        changes nothing about them."""
        var tables = List[List[Int]](capacity=rng.n_iterations())
        for i in range(rng.start, rng.stop):
            tables.append(self.trees[i].leaf_ordinals())
        return tables^

    def leaf_indices_bins(
        self, bins: List[Int], rng: IterationRange
    ) -> List[Int]:
        """The leaf ordinal this example reaches in each tree of `rng`."""
        var out = List[Int](capacity=rng.n_iterations())
        for i in range(rng.start, rng.stop):
            out.append(self.trees[i].leaf_ordinal_bins(bins))
        return out^

    def to_booster(self) raises -> Booster:
        """This forest as an ordinary `Booster`, for the paths that only know
        how to sum trees.

        A `Booster` computes `base_score + sum_i(learning_rate * tree_i)`.
        With a base score of 0 and a rate of `1 / T` that is the forest's mean,
        so the returned ensemble predicts, serializes (serialize.mojo),
        dumps, and explains (contrib.mojo) as this forest does, with no change
        to any of them. It is the whole of what an rf model can reach today.

        Two things do not survive the bridge, and both are why the handoff
        asks for an `average_output` flag rather than leaving this as the
        answer:

        - **ranges.** `Booster.predict_raw_bins_range` divides by `T`
          whatever the range, and adds a base score of 0 to a range starting
          at 0. Only the full range is the forest's own prediction.
        - **continued training.** The returned `Booster` reports a
          `learning_rate` of `1 / T` and a base score of 0, so `train_more`
          would shrink new trees by the averaging weight of the old ones and
          boost them from zero. `train_forest_more` on this `RfBooster` is the
          continuation path; there is no continuing the bridged model.

        The mean here is accumulated pre-scaled (`sum(tree_i / T)`) where
        `predict_raw_bins` accumulates and then divides (`sum(tree_i) / T`).
        The two are the same arithmetic in a different association and agree
        exactly whenever `T` is a power of two, since scaling by an exact
        power of two commutes with rounding; otherwise they can differ in the
        last place.

        An empty forest bridges to the base score carried as a base score,
        which is what an empty forest predicts and what an empty boosted
        ensemble is. `is_forest` cannot tell the two apart, and nothing could:
        with no trees they are the same object.
        """
        if len(self.trees) == 0:
            return Booster(
                List[Tree](),
                self.base_score,
                RF_SHRINKAGE,
                self.objective,
                self.monotone.copy(),
            )
        return Booster(
            self.trees.copy(),
            0.0,
            self.average_weight(),
            self.objective,
            self.monotone.copy(),
        )


def _rf_rounds(
    data: BinnedMatrix,
    target: List[Float64],
    objective: Int,
    params: RfParams,
    sample_weight: List[Float64],
    alpha: Float64,
    base_score: Float64,
    grad0: List[Float64],
    hess0: List[Float64],
    round_offset: Int,
    n_rounds: Int,
    mut trees: List[Tree],
) raises:
    """Grow `n_rounds` forest trees, appending them to `trees`.

    `grad0` and `hess0` are the gradients of the whole run, taken once at the
    constant `base_score`; they are never recomputed and never mutated. The
    shared boosting loop, `_boost_rounds`, exists for the ensembles that do
    recompute them, which is the one structural difference between the two.

    `round_offset` is the number of trees already grown, so every seeded draw
    reads the absolute round index and a continued run draws what an
    uninterrupted one would have.

    Unlike `_boost_rounds` this never stops early and never skips a round:
    exactly `n_rounds` trees are appended. The tree count is the denominator
    of the forest's average, so dropping a tree would rescale every prediction
    the model makes; and with constant gradients a degenerate tree says
    nothing about convergence, only about the sample it was grown on.
    """
    var n = data.n_rows
    var signs = params.tree.monotone.active_signs()
    var renews = objective_renews_leaves(objective)
    var renew_w = renewal_weights(objective, target, sample_weight)
    var renew_a = renewal_alpha(objective, alpha)
    # LightGBM's rf `residual_getter` is `label[i] - init_score`, so the raw
    # scores renewal measures residuals against are this constant vector, not
    # a running score. Built once: it never changes.
    var init_raw = _constant_scores(n, base_score)

    var bag = List[Int]()
    # As in `_boost_rounds`: balanced bagging needs a positive row to apply
    # at all, and the labels do not change, so the test is hoisted out.
    var balanced = params.class_bagging.enabled() and has_positive_rows(target)
    # GOSS rescales the rows it samples, so it needs a fresh copy of the
    # gradients every round; without one the multipliers would compound over a
    # run that never refills them. Every other sampler leaves grad/hess alone,
    # so this one copy is made once and every tree is grown from it unchanged.
    var resamples = params.goss.enabled
    var grad = grad0.copy()
    var hess = hess0.copy()

    for i in range(n_rounds):
        var round = round_offset + i
        if resamples:
            # The three samplers are mutually exclusive, so under GOSS the bag
            # is whatever the previous round's selection left behind; clearing
            # it restores "every row" for a warmup round. The other two rely
            # on the bag persisting between redraws (`bagging_freq > 1`), so
            # they must not be cleared.
            bag.clear()
            grad = grad0.copy()
            hess = hess0.copy()
            goss_round(bag, grad, hess, params.goss, round, RF_SHRINKAGE)
        elif balanced:
            refresh_class_bag(bag, params.class_bagging, target, round)
        else:
            refresh_bag(bag, params.bagging, n, round)

        var tree = grow_tree(data, grad, hess, params.tree, bag, round)
        if renews:
            _renew_leaf_values(
                tree, data, target, init_raw, renew_w, renew_a, bag, signs,
                params.tree.extra,
            )
        # After renewal and after the grower's own leaf finishing, exactly
        # where LightGBM's `AddBias` sits.
        _add_bias(tree, base_score)
        trees.append(tree^)


def train_forest(
    data: BinnedMatrix,
    target: List[Float64],
    objective: Int,
    params: RfParams,
    sample_weight: List[Float64] = [],
    alpha: Float64 = 0.9,
    init_score: List[Float64] = [],
) raises -> RfBooster:
    """Train a random forest: `params.n_estimators` trees, averaged.

    `target`, `sample_weight`, and `alpha` mean what they mean in `train`, and
    are validated by the same functions, because the objective layer is
    unchanged: a forest fits the same gradients of the same losses, once.

    `init_score` is accepted only so that it can be rejected with a reason;
    see `check_rf_init_score`.

    The returned forest holds exactly `params.n_estimators` trees.
    """
    if len(target) != data.n_rows:
        raise Error("target length must equal n_rows")
    _check_objective(objective, target, alpha)
    _check_sample_weight(sample_weight, data.n_rows)
    check_rf_params(params, objective)
    check_rf_init_score(init_score)
    params.tree.monotone.check_features(data.n_features)

    var n = data.n_rows
    var base_score = _base_score(target, objective, sample_weight, alpha)
    var init_raw = _constant_scores(n, base_score)
    # The whole run's gradients, LightGBM's `RF::Boosting()` called once from
    # `Init` rather than once per iteration.
    var grad0 = List[Float64](capacity=n)
    var hess0 = List[Float64](capacity=n)
    _fill_grad_hess(
        init_raw, target, objective, sample_weight, alpha, grad0, hess0
    )

    var trees = List[Tree]()
    _rf_rounds(
        data,
        target,
        objective,
        params,
        sample_weight,
        alpha,
        base_score,
        grad0,
        hess0,
        0,
        params.n_estimators,
        trees,
    )
    return RfBooster(
        trees^, base_score, objective, params.tree.monotone.copy()
    )


def train_forest_more(
    mut forest: RfBooster,
    data: BinnedMatrix,
    target: List[Float64],
    params: RfParams,
    sample_weight: List[Float64] = [],
    alpha: Float64 = 0.9,
) raises -> Int:
    """Add `params.n_estimators` more trees to a fitted forest and return how
    many were added (always `params.n_estimators`).

    Continuing a forest is cheaper than continuing a boosted ensemble and
    exact in a way it cannot be: there are no raw scores to replay, because
    the new trees fit the same gradients the old ones did. Those gradients are
    re-derived from the forest's own `base_score`, so 40 trees followed by 60
    more are the 100-tree forest, for the same data, objective, and tree
    parameters. LightGBM's rf does not agree with itself here; see the module
    docstring.

    `params.n_estimators` counts NEW trees, not the total. The objective is
    the forest's own, and the monotone constraints must match the ones
    recorded on it, for the reason `train_more` gives: one constraint vector
    describes every tree.

    What does change is the weight of every existing tree: a forest of `T`
    trees weights each `1 / T`, so adding trees rescales the whole model.
    That is what averaging means, and it is why prediction from a partially
    grown forest is not a prefix of prediction from the finished one.
    """
    if len(target) != data.n_rows:
        raise Error("target length must equal n_rows")
    if not _same_signs(params.tree.monotone.signs, forest.monotone.signs):
        raise Error(
            "continued training cannot change monotone_constraints: the"
            " forest records the constraints all of its trees satisfy"
        )
    _check_objective(forest.objective, target, alpha)
    _check_sample_weight(sample_weight, data.n_rows)
    check_rf_params(params, forest.objective)
    params.tree.monotone.check_features(data.n_features)

    var n = data.n_rows
    var init_raw = _constant_scores(n, forest.base_score)
    var grad0 = List[Float64](capacity=n)
    var hess0 = List[Float64](capacity=n)
    _fill_grad_hess(
        init_raw,
        target,
        forest.objective,
        sample_weight,
        alpha,
        grad0,
        hess0,
    )

    var grown = List[Tree]()
    _rf_rounds(
        data,
        target,
        forest.objective,
        params,
        sample_weight,
        alpha,
        forest.base_score,
        grad0,
        hess0,
        len(forest.trees),
        params.n_estimators,
        grown,
    )
    var added = len(grown)
    for i in range(added):
        forest.trees.append(grown[i].copy())
    return added


# --- The `boosting=rf` surface --------------------------------------------
#
# `train_forest` above is the algorithm and `RfBooster` is what it produces.
# The three functions below are the surface a caller that selects a strategy
# by name reaches instead: they take the ordinary training arguments, return
# an ordinary `Booster` through `RfBooster.to_booster`, and are what
# `alternate_boosting.mojo` dispatches `boosting='rf'` to. Everything the
# bridge gives up is listed on `to_booster`; in exchange, an rf model
# serializes, predicts, and explains through paths that need no change.


def is_forest(booster: Booster) -> Bool:
    """Whether this ensemble has the shape `RfBooster.to_booster` produces:
    a zero base score and a shrinkage factor of exactly `1 / T`.

    Structural, and deliberately not conclusive. A fitted `Booster` records
    no boosting mode, so this is the only question that can be asked of one,
    and a boosted ensemble could in principle answer yes to it. What it does
    reliably catch is the mistake worth catching: continuing a forest as GBDT
    (which would shrink new trees by `1 / T`) or a GBDT model as a forest
    (which would rescale every existing tree). An empty ensemble answers no,
    because an empty forest and an empty boosted ensemble are the same object.

    The comparison is exact rather than tolerant on purpose: the rate is
    written by `average_weight` and read back by `serialize.load_model`,
    which stores raw IEEE-754 bit patterns, so the round trip is exact and a
    near miss means a different model.
    """
    if len(booster.trees) == 0:
        return False
    if booster.base_score != 0.0:
        return False
    return booster.learning_rate == 1.0 / Float64(len(booster.trees))


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
    """Train `boosting='rf'` and return an ordinary `Booster`.

    `train_forest` with the arguments the other modes take.
    `params.learning_rate` must be 1.0 (`check_rf_learning_rate`) and
    `init_score` must be empty (`check_rf_init_score`); the sampler must
    randomize the run (`check_rf_params`).

    GOSS is not an argument here. It is a `boosting` value of its own, so a
    caller that selects a strategy by name has already chosen between them;
    `train_forest` is the entry point for a forest that samples by gradient.
    """
    check_rf_learning_rate(params.learning_rate)
    var forest = train_forest(
        data,
        target,
        objective,
        RfParams.from_booster_params(params, bagging),
        sample_weight,
        alpha,
        init_score,
    )
    return forest.to_booster()


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
    """Add `params.n_estimators` trees to a forest held as a `Booster`, and
    rescale it to its new size. Returns how many trees were added.

    The ensemble is checked with `is_forest` before anything is touched: this
    rewrites `booster.learning_rate`, which on a boosted ensemble is the
    shrinkage every one of its trees was fitted under and would be a silent
    corruption of the model.

    Unlike `train_forest_more`, the constant the new trees are fitted at
    cannot be read off the model: `to_booster` folds it into the leaves and a
    `Booster` has nowhere to record it. It is recomputed here from `target`
    and `sample_weight`, so the added trees are the trees one call would have
    grown exactly when this is the data the forest was trained on, which is
    the precondition `boosting.train_more` already states for continuing at
    all. A forest kept as an `RfBooster` needs no such precondition, because
    it carries its own base score; that is what `train_forest_more` is for.
    """
    check_rf_learning_rate(params.learning_rate)
    check_rf_init_score(init_score)
    if not is_forest(booster):
        raise Error(
            "boosting='rf' cannot continue this ensemble: a forest has a base"
            " score of 0 and a shrinkage of 1 / n_trees (see"
            " boosting_rf.is_forest), and adding averaged trees to a boosted"
            " ensemble would rescale every tree already in it"
        )
    if len(target) != data.n_rows:
        raise Error("target length must equal n_rows")
    if not _same_signs(params.tree.monotone.signs, booster.monotone.signs):
        raise Error(
            "continued training cannot change monotone_constraints: the"
            " forest records the constraints all of its trees satisfy"
        )
    _check_objective(booster.objective, target, alpha)
    _check_sample_weight(sample_weight, data.n_rows)
    var rf = RfParams.from_booster_params(params, bagging)
    check_rf_params(rf, booster.objective)
    params.tree.monotone.check_features(data.n_features)

    var n = data.n_rows
    # Recomputed rather than recovered; see the docstring.
    var base_score = _base_score(
        target, booster.objective, sample_weight, alpha
    )
    var init_raw = _constant_scores(n, base_score)
    var grad0 = List[Float64](capacity=n)
    var hess0 = List[Float64](capacity=n)
    _fill_grad_hess(
        init_raw,
        target,
        booster.objective,
        sample_weight,
        alpha,
        grad0,
        hess0,
    )

    var grown = List[Tree]()
    _rf_rounds(
        data,
        target,
        booster.objective,
        rf,
        sample_weight,
        alpha,
        base_score,
        grad0,
        hess0,
        len(booster.trees),
        rf.n_estimators,
        grown,
    )
    var added = len(grown)
    for i in range(added):
        booster.trees.append(grown[i].copy())
    # The averaging weight is a function of the tree count, so growing the
    # forest rescales every tree in it, the old ones included. That is what
    # averaging means, and it is why this takes the ensemble by reference.
    booster.learning_rate = 1.0 / Float64(len(booster.trees))
    return added


def train_forest_with_valid(
    data: BinnedMatrix,
    target: List[Float64],
    valid_data: BinnedMatrix,
    valid_target: List[Float64],
    objective: Int,
    params: RfParams,
    early_stopping_rounds: Int,
    min_delta: Float64 = 0.0,
    sample_weight: List[Float64] = [],
    alpha: Float64 = 0.9,
) raises -> RfBooster:
    """Train a forest with validation-set early stopping, truncated to its
    best size.

    The stopping signal is the loss of the *averaged* model after each tree,
    which is the only prediction a forest makes. That differs from a boosted
    run in what it measures: boosting stops when the correction stops
    correcting, while a forest's loss falls because averaging more trees
    removes variance, so early stopping here answers "how many trees are
    enough", not "has this converged". The loss is not monotone in the tree
    count -- one unlucky bag can raise it -- so `min_delta` and
    `early_stopping_rounds` do the same job they do elsewhere.

    Truncation is exact: the forest of the first `k` trees is the forest
    `train_forest` would have grown with `n_estimators = k`, because tree `i` does
    not depend on tree `i - 1`. Truncating a boosted ensemble only holds
    because the base score and the earlier trees are unchanged; here nothing
    is changed at all except the denominator.

    `sample_weight` applies to training rows only and the validation loss is
    unweighted, as in `train_with_valid`. Validation rows are never sampled.
    """
    if len(target) != data.n_rows:
        raise Error("target length must equal n_rows")
    if len(valid_target) != valid_data.n_rows:
        raise Error("valid_target length must equal valid n_rows")
    if data.n_features != valid_data.n_features:
        raise Error("valid_data must have the same number of features")
    if early_stopping_rounds < 1:
        raise Error("early_stopping_rounds must be positive")
    _check_objective(objective, target, alpha)
    _check_sample_weight(sample_weight, data.n_rows)
    check_rf_params(params, objective)
    params.tree.monotone.check_features(data.n_features)

    var n = data.n_rows
    var n_valid = valid_data.n_rows
    var base_score = _base_score(target, objective, sample_weight, alpha)
    var init_raw = _constant_scores(n, base_score)
    var grad0 = List[Float64](capacity=n)
    var hess0 = List[Float64](capacity=n)
    _fill_grad_hess(
        init_raw, target, objective, sample_weight, alpha, grad0, hess0
    )

    # Running totals of the tree outputs on the validation rows, divided by
    # the tree count to score. Kept apart from the mean itself so that adding
    # a tree is one addition per row and the mean is one division per row,
    # rather than rescaling a running mean and losing a digit each round.
    var valid_total = _constant_scores(n_valid, 0.0)
    var valid_raw = _constant_scores(n_valid, base_score)
    var best_loss = _mean_loss(valid_raw, valid_target, objective, alpha)
    var best_n_trees = 0

    var trees = List[Tree]()
    for i in range(params.n_estimators):
        # One round at a time, so that the loss can be read between trees.
        # `_rf_rounds` is still the only place a forest tree is grown: the
        # bag, the feature draw, and the GOSS sample all key on the absolute
        # round index, so growing 1 + 1 + ... is growing n.
        _rf_rounds(
            data,
            target,
            objective,
            params,
            sample_weight,
            alpha,
            base_score,
            grad0,
            hess0,
            i,
            1,
            trees,
        )
        var tree_index = len(trees) - 1
        for r in range(n_valid):
            valid_total[r] += trees[tree_index].predict_row(valid_data, r)
        var weight = 1.0 / Float64(len(trees))
        for r in range(n_valid):
            valid_raw[r] = valid_total[r] * weight

        var loss = _mean_loss(valid_raw, valid_target, objective, alpha)
        if loss < best_loss - min_delta:
            best_loss = loss
            best_n_trees = len(trees)
        elif len(trees) - best_n_trees >= early_stopping_rounds:
            break

    while len(trees) > best_n_trees:
        _ = trees.pop()
    return RfBooster(
        trees^, base_score, objective, params.tree.monotone.copy()
    )


# --- Multiclass ------------------------------------------------------------
#
# LightGBM's `RF::Init` asserts `num_tree_per_iteration_ == num_class_`, so
# random-forest mode covers softmax multiclass: one tree per class per round,
# every one of them fitted to the gradients of the constant per-class base
# scores. The layout is `boosting.MulticlassBooster`'s -- round-major, so the
# tree for (round i, class k) is `trees[i * n_classes + k]` -- and the
# averaging is per class over rounds, not over trees.


def _class_log_priors(
    labels: List[Int], n_classes: Int, sample_weight: List[Float64]
) raises -> List[Float64]:
    """The per-class base scores: log of the (weighted) class prior.

    The same quantity `train_multiclass` computes inline, including the
    clamping that keeps an absent class off negative infinity. Duplicated here
    only because boosting.mojo does not expose it; the handoff asks for it to
    be lifted so both callers read one definition.
    """
    var class_w = List[Float64](capacity=n_classes)
    class_w.resize(n_classes, 0.0)
    var total_w = 0.0
    for r in range(len(labels)):
        if labels[r] < 0 or labels[r] >= n_classes:
            raise Error("label out of range")
        var w = sample_weight[r] if len(sample_weight) > 0 else 1.0
        class_w[labels[r]] += w
        total_w += w
    if total_w <= 0.0:
        raise Error("sample_weight must have a positive sum")
    var base_scores = List[Float64](capacity=n_classes)
    for k in range(n_classes):
        base_scores.append(log(_clamp_prob(class_w[k] / total_w)))
    return base_scores^


def _constant_softmax(
    base_scores: List[Float64], n_rows: Int
) -> List[Float64]:
    """Row-major softmax probabilities of the constant base scores.

    Every row has the same raw scores in a forest, so every row has the same
    probabilities; they are still materialized per row because
    `_fill_softmax_grad_hess` reads `prob[r * n_classes + k]` and reusing it
    is worth more than the row of memory it costs.
    """
    var n_classes = len(base_scores)
    var prob = List[Float64](capacity=n_rows * n_classes)
    for _ in range(n_rows):
        for k in range(n_classes):
            prob.append(base_scores[k])
    for r in range(n_rows):
        _softmax_inplace(prob, r * n_classes, n_classes)
    return prob^


struct RfMulticlassBooster(Copyable, Movable):
    """A fitted softmax random forest.

    Round-major, as `MulticlassBooster` is: the tree for (round i, class k) is
    `trees[i * n_classes + k]`. Class `k`'s raw score is the mean over rounds
    of its own trees, each of which carries `base_scores[k]` in its node
    values, so the class base scores are recorded for continuation rather than
    for prediction -- exactly the single-output arrangement.
    """

    var trees: List[Tree]
    var base_scores: List[Float64]
    var n_classes: Int
    var monotone: MonotoneConstraints

    def __init__(
        out self,
        var trees: List[Tree],
        var base_scores: List[Float64],
        n_classes: Int,
        var monotone: MonotoneConstraints = MonotoneConstraints(),
    ):
        self.trees = trees^
        self.base_scores = base_scores^
        self.n_classes = n_classes
        self.monotone = monotone^

    @always_inline
    def n_iterations(self) -> Int:
        """Rounds in the forest. One round is one tree per class."""
        return len(self.trees) // self.n_classes

    def average_weight(self) -> Float64:
        """The weight each round carries, `1 / rounds`."""
        var rounds = self.n_iterations()
        if rounds == 0:
            return 0.0
        return 1.0 / Float64(rounds)

    def predict_raw_bins_range(
        self, bins: List[Int], rng: IterationRange
    ) -> List[Float64]:
        """Per-class raw scores from the rounds in `rng`: the mean over those
        rounds of each class's trees. An empty range yields the base scores,
        the prediction of a forest with no rounds."""
        var out = List[Float64](capacity=self.n_classes)
        if rng.is_empty():
            for k in range(self.n_classes):
                out.append(self.base_scores[k])
            return out^
        var weight = 1.0 / Float64(rng.n_iterations())
        for k in range(self.n_classes):
            var total = 0.0
            for i in range(rng.start, rng.stop):
                total += self.trees[i * self.n_classes + k].predict_bins(bins)
            out.append(total * weight)
        return out^

    def predict_raw_bins(self, bins: List[Int]) -> List[Float64]:
        """Per-class raw scores from the whole forest."""
        return self.predict_raw_bins_range(
            bins, IterationRange(0, self.n_iterations())
        )

    def predict_proba_bins_range(
        self, bins: List[Int], rng: IterationRange
    ) -> List[Float64]:
        """Class probabilities from the rounds in `rng` alone."""
        var scores = self.predict_raw_bins_range(bins, rng)
        _softmax_inplace(scores, 0, self.n_classes)
        return scores^

    def predict_proba_bins(self, bins: List[Int]) -> List[Float64]:
        """Class probabilities from the whole forest."""
        return self.predict_proba_bins_range(
            bins, IterationRange(0, self.n_iterations())
        )

    def leaf_indices_bins(
        self, bins: List[Int], rng: IterationRange
    ) -> List[Int]:
        """The leaf ordinal this example reaches in each tree of `rng`, class
        index fastest, as on `MulticlassBooster`."""
        var out = List[Int](capacity=rng.n_iterations() * self.n_classes)
        for i in range(rng.start, rng.stop):
            for k in range(self.n_classes):
                out.append(
                    self.trees[i * self.n_classes + k].leaf_ordinal_bins(bins)
                )
        return out^

    def to_multiclass_booster(self) raises -> MulticlassBooster:
        """This forest as an ordinary `MulticlassBooster`, on the terms
        `RfBooster.to_booster` gives: zero base scores, a `learning_rate` of
        `1 / rounds`, and correctness for the full range only."""
        var rounds = self.n_iterations()
        if rounds == 0:
            return MulticlassBooster(
                List[Tree](),
                self.base_scores.copy(),
                self.n_classes,
                RF_SHRINKAGE,
                self.monotone.copy(),
            )
        var zeros = List[Float64](capacity=self.n_classes)
        zeros.resize(self.n_classes, 0.0)
        return MulticlassBooster(
            self.trees.copy(),
            zeros^,
            self.n_classes,
            self.average_weight(),
            self.monotone.copy(),
        )


def _rf_rounds_multiclass(
    data: BinnedMatrix,
    labels: List[Int],
    n_classes: Int,
    params: RfParams,
    sample_weight: List[Float64],
    base_scores: List[Float64],
    prob: List[Float64],
    grads: List[List[Float64]],
    hesses: List[List[Float64]],
    round_offset: Int,
    n_rounds: Int,
    mut trees: List[Tree],
) raises:
    """Grow `n_rounds` softmax forest rounds, appending `n_classes` trees per
    round.

    `grads[k]` and `hesses[k]` are class `k`'s gradients for the whole run,
    taken once at the constant base scores. `prob` is the matching row-major
    softmax, kept because the GOSS selection is computed from it.

    One bag serves a whole round, as in `_boost_rounds_multiclass`, so the
    per-class trees of a round stay comparable; each class's tree still draws
    its own feature set, from `tree_index = round * n_classes + k`.
    """
    var n = data.n_rows
    var bag = List[Int]()
    var resamples = params.goss.enabled
    for i in range(n_rounds):
        var round = round_offset + i
        var selection = GossSelection.all_rows()
        if resamples:
            bag.clear()
            if params.goss.active(round, RF_SHRINKAGE):
                selection = _multiclass_goss_select(
                    prob, labels, n_classes, sample_weight, params.goss, round
                )
                bag = selection.rows.copy()
        else:
            refresh_bag(bag, params.bagging, n, round)

        for k in range(n_classes):
            var tree_index = round * n_classes + k
            var tree: Tree
            if resamples:
                # A private copy per class per round: the multipliers must not
                # compound over a run whose gradients are never refilled.
                var grad = grads[k].copy()
                var hess = hesses[k].copy()
                apply_goss_scaling(selection, grad, hess)
                tree = grow_tree(
                    data, grad, hess, params.tree, bag, tree_index
                )
            else:
                tree = grow_tree(
                    data, grads[k], hesses[k], params.tree, bag, tree_index
                )
            # Softmax is not one of the renewing objectives, so a multiclass
            # tree goes straight from the grower to its bias, as it does in
            # `_boost_rounds_multiclass`.
            _add_bias(tree, base_scores[k])
            trees.append(tree^)


def _multiclass_rf_gradients(
    prob: List[Float64],
    labels: List[Int],
    n_classes: Int,
    sample_weight: List[Float64],
    mut grads: List[List[Float64]],
    mut hesses: List[List[Float64]],
):
    """The whole run's per-class gradients, taken once at the constant softmax
    probabilities."""
    grads.clear()
    hesses.clear()
    for k in range(n_classes):
        var grad = List[Float64]()
        var hess = List[Float64]()
        _fill_softmax_grad_hess(
            prob, labels, k, n_classes, sample_weight, grad, hess
        )
        grads.append(grad^)
        hesses.append(hess^)


def train_forest_multiclass(
    data: BinnedMatrix,
    labels: List[Int],
    n_classes: Int,
    params: RfParams,
    sample_weight: List[Float64] = [],
) raises -> RfMulticlassBooster:
    """Train a softmax random forest on labels in `0..n_classes-1`.

    The base scores are the log class priors, weighted when `sample_weight` is
    given, as in `train_multiclass`. Every tree of every round fits the
    gradients of those constant scores, so the whole forest's gradients are
    one pass, not `n_estimators` of them.

    `class_bagging` is rejected here as it is for a boosted multiclass run:
    LightGBM's balanced bagging is a binary-classification feature, which is
    what `_NOT_BINARY` makes `check_rf_params` say.
    """
    if len(labels) != data.n_rows:
        raise Error("labels length must equal n_rows")
    if n_classes < 2:
        raise Error("n_classes must be at least 2")
    _check_sample_weight(sample_weight, data.n_rows)
    check_rf_params(params, _NOT_BINARY)
    params.tree.monotone.check_features(data.n_features)

    var base_scores = _class_log_priors(labels, n_classes, sample_weight)
    var prob = _constant_softmax(base_scores, data.n_rows)
    var grads = List[List[Float64]]()
    var hesses = List[List[Float64]]()
    _multiclass_rf_gradients(
        prob, labels, n_classes, sample_weight, grads, hesses
    )

    var trees = List[Tree]()
    _rf_rounds_multiclass(
        data,
        labels,
        n_classes,
        params,
        sample_weight,
        base_scores,
        prob,
        grads,
        hesses,
        0,
        params.n_estimators,
        trees,
    )
    return RfMulticlassBooster(
        trees^, base_scores^, n_classes, params.tree.monotone.copy()
    )


def train_forest_multiclass_more(
    mut forest: RfMulticlassBooster,
    data: BinnedMatrix,
    labels: List[Int],
    params: RfParams,
    sample_weight: List[Float64] = [],
) raises -> Int:
    """Add `params.n_estimators` more rounds to a fitted softmax forest and
    return how many were added.

    The forest's own base scores are reused rather than recomputed from the
    labels in hand, for the reason `train_multiclass_more` gives: they are the
    priors of the data the forest was first fitted on, and re-deriving them
    from a second dataset would make the added trees answer a different
    question from the ones already there. Here it also fixes the gradients,
    so the added rounds are the rounds one call would have grown.
    """
    if len(labels) != data.n_rows:
        raise Error("labels length must equal n_rows")
    if not _same_signs(params.tree.monotone.signs, forest.monotone.signs):
        raise Error(
            "continued training cannot change monotone_constraints: the"
            " forest records the constraints all of its trees satisfy"
        )
    _check_sample_weight(sample_weight, data.n_rows)
    check_rf_params(params, _NOT_BINARY)
    params.tree.monotone.check_features(data.n_features)
    for r in range(len(labels)):
        if labels[r] < 0 or labels[r] >= forest.n_classes:
            raise Error("label out of range")

    var prob = _constant_softmax(forest.base_scores, data.n_rows)
    var grads = List[List[Float64]]()
    var hesses = List[List[Float64]]()
    _multiclass_rf_gradients(
        prob, labels, forest.n_classes, sample_weight, grads, hesses
    )

    var grown = List[Tree]()
    _rf_rounds_multiclass(
        data,
        labels,
        forest.n_classes,
        params,
        sample_weight,
        forest.base_scores,
        prob,
        grads,
        hesses,
        forest.n_iterations(),
        params.n_estimators,
        grown,
    )
    for i in range(len(grown)):
        forest.trees.append(grown[i].copy())
    return len(grown) // forest.n_classes


def train_forest_multiclass_with_valid(
    data: BinnedMatrix,
    labels: List[Int],
    valid_data: BinnedMatrix,
    valid_labels: List[Int],
    n_classes: Int,
    params: RfParams,
    early_stopping_rounds: Int,
    min_delta: Float64 = 0.0,
    sample_weight: List[Float64] = [],
) raises -> RfMulticlassBooster:
    """Train a softmax forest with validation-set early stopping, truncated to
    its best round count.

    The signal is the multiclass log loss of the averaged model, on the terms
    `train_forest_with_valid` describes. Truncation drops whole rounds, so the
    per-class trees stay in step.
    """
    if len(labels) != data.n_rows:
        raise Error("labels length must equal n_rows")
    if len(valid_labels) != valid_data.n_rows:
        raise Error("valid_labels length must equal valid n_rows")
    if data.n_features != valid_data.n_features:
        raise Error("valid_data must have the same number of features")
    if n_classes < 2:
        raise Error("n_classes must be at least 2")
    if early_stopping_rounds < 1:
        raise Error("early_stopping_rounds must be positive")
    _check_sample_weight(sample_weight, data.n_rows)
    check_rf_params(params, _NOT_BINARY)
    params.tree.monotone.check_features(data.n_features)
    for r in range(len(valid_labels)):
        if valid_labels[r] < 0 or valid_labels[r] >= n_classes:
            raise Error("valid label out of range")

    var n_valid = valid_data.n_rows
    var base_scores = _class_log_priors(labels, n_classes, sample_weight)
    var prob = _constant_softmax(base_scores, data.n_rows)
    var grads = List[List[Float64]]()
    var hesses = List[List[Float64]]()
    _multiclass_rf_gradients(
        prob, labels, n_classes, sample_weight, grads, hesses
    )

    var valid_total = _constant_scores(n_valid * n_classes, 0.0)
    var valid_raw = List[Float64](capacity=n_valid * n_classes)
    for _ in range(n_valid):
        for k in range(n_classes):
            valid_raw.append(base_scores[k])
    var best_loss = _multiclass_mean_loss(valid_raw, valid_labels, n_classes)
    var best_rounds = 0

    var trees = List[Tree]()
    for i in range(params.n_estimators):
        _rf_rounds_multiclass(
            data,
            labels,
            n_classes,
            params,
            sample_weight,
            base_scores,
            prob,
            grads,
            hesses,
            i,
            1,
            trees,
        )
        for k in range(n_classes):
            var t = i * n_classes + k
            for r in range(n_valid):
                valid_total[r * n_classes + k] += trees[t].predict_row(
                    valid_data, r
                )
        var weight = 1.0 / Float64(i + 1)
        for j in range(n_valid * n_classes):
            valid_raw[j] = valid_total[j] * weight

        var loss = _multiclass_mean_loss(valid_raw, valid_labels, n_classes)
        if loss < best_loss - min_delta:
            best_loss = loss
            best_rounds = i + 1
        elif (i + 1) - best_rounds >= early_stopping_rounds:
            break

    while len(trees) > best_rounds * n_classes:
        _ = trees.pop()
    return RfMulticlassBooster(
        trees^, base_scores^, n_classes, params.tree.monotone.copy()
    )
