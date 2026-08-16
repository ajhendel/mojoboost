"""The remaining LightGBM tree controls, as isolated primitives.

Everything here is a pure decision rule: a scalar formula, a validated
parameter bundle, or a parsed data structure. Nothing in this module grows a
tree, reads a histogram, or touches a grower. That is deliberate. The four
growers (`tree.grow_tree`, `train_gpu.grow_tree_gpu`,
`tree_sparse.grow_tree_sparse`, `distributed.grow_tree_distributed`) already
share `tree._search` and `tree._leaf_value`; a rule expressed here as a
function of numbers can be dropped into those two shared entry points once and
be live on every backend at once, which is what
`handoffs/task12_tree_parameters.md` specifies.

What is wired up, and where
---------------------------
`ExtraTreeParams` is carried on `tree.TreeParams` as `extra` and reaches the
search through `tree._search` -> `split.find_best_split`. Three tiers, and the
tier is a property of what the rule needs rather than of the backend:

- Live wherever `tree._search` is called (the dense CPU grower, the sparse
  grower, and the GPU grower's host split scan): `min_gain_to_split`,
  `monotone_penalty`, `feature_contri`, and `cegb_tradeoff` x
  `cegb_penalty_split`. Each is a function of the histogram, the node's row
  count, and the node's depth, all of which every caller already passes.
- Live in `tree.grow_tree` alone, and refused by `tree._search` for any other
  caller: `extra_trees`/`extra_seed` (it needs the node id and the tree
  index) and `max_delta_step`/`path_smooth` (they need the leaf's row count
  and its parent's finished output). See `needs_grower_support`.
- Live in `tree.grow_tree` alone and reachable from the Mojo API only:
  `forcedsplits_filename`. The document is parsed here
  (`parse_forced_splits`), mapped to bins by `binning.map_forced_splits` --
  the one place a raw threshold and a fitted mapper meet -- and applied by
  `grow_tree` before leaf-wise growth resumes. A parameter string cannot
  carry a document, so `check_extra_option_supported` names that path rather
  than accepting the key.
- Live in `tree.grow_tree` alone because they need the per-ensemble ledger
  `boosting.fit` owns, and refused per grower rather than by name:
  `cegb_penalty_feature_coupled` and `cegb_penalty_feature_lazy`. See
  `cegb.check_cegb_grower_support`.
- Refused by *value* rather than by name: `feature_pre_filter`.
  `check_feature_pre_filter` accepts `false`, which is the behavior mojotrees
  has, and refuses `true` **from a parameter string only**: the filter is
  implemented (`binning.filter_count`, `binning.need_filter`,
  `fit_bins(feature_pre_filter=...)`, `BinnedMatrix.usable_features`,
  `sampling.select_tree_features(..., usable)`) and reachable from the Mojo
  API, but `params.TrainConfig` has no field to carry the flag into binning,
  so a string that set it would be ignored. That is the
  `forcedsplits_filename` case above, not a missing feature.
  `check_extra_option_supported` no longer knows the name at all, so every
  name that checker still refuses is refused for how it is spelled rather
  than for being missing.
  `linear_tree` and `linear_lambda` are `BoosterParams.linear`
  (linear_tree.mojo), not tree controls, and are live on the metric-path
  trainers.

`distributed.grow_tree_distributed` keeps private copies of `_search` and
`_leaf_value` and so honors none of this; `train_gpu`'s device split search
scores candidates in a kernel and honors none of it either. Both are recorded
in `handoffs/integration_08_algorithms.md` as edits this integration could not
make from the files it owns.

What is here, with LightGBM's name and default
---------------------------------------------
- `min_gain_to_split` (0.0): a floor a candidate's gain must clear.
- `max_delta_step` (0.0): a cap on the magnitude of a leaf's output.
- `path_smooth` (0.0): leaf-value shrinkage toward the parent's output.
- `feature_contri` (empty): per-feature multipliers on split gain, held on
  `FeaturePenalties` beside a `cegb.CegbConfig` carrying all four `cegb_*`
  costs (`cegb_tradeoff` 1.0, `cegb_penalty_split` 0.0, both per-feature
  vectors empty).
- `extra_trees` (false) / `extra_seed` (6): one randomly chosen threshold per
  feature instead of a full scan.
- `monotone_penalty` (0.0) and `monotone_constraints_method` ("basic").
- `forcedsplits_filename` (empty): the file's contents parsed into a
  validated forced-split tree, which `binning.map_forced_splits` turns into
  bin space and `tree.grow_tree` applies. Reading the file is a caller's job;
  this module never touches the filesystem.

What is *not* here, because it already exists
---------------------------------------------
`num_leaves`, `max_depth`, `min_data_in_leaf`, `min_sum_hessian_in_leaf`,
`lambda_l1`, `lambda_l2`, `feature_fraction*`, `monotone_constraints`,
`interaction_constraints`, and every categorical hyperparameter
(`cat_smooth`, `cat_l2`, `max_cat_threshold`, `max_cat_to_onehot`,
`min_data_per_group`) are implemented and owned elsewhere; see
`tree.TreeParams`, `monotone.MonotoneConstraints`, and
`categorical.CategoricalParams`. This module adds no second copy of any of
them. The categorical helpers below are the *arithmetic* that
`categorical.mojo` performs inline today, factored out so the new rules
(gain floor, gain multipliers, output cap, smoothing) can be applied to
category partitions and to ordinal thresholds through one shared set of
formulas. The handoff asks for the inline arithmetic to be replaced by these
calls at integration, so no duplicate survives.

Leaf-wise growth is untouched
-----------------------------
None of these rules changes which leaf is chosen next. They change what a
candidate scores (`min_gain_to_split`, the penalties, `monotone_penalty`),
which candidates exist (`extra_trees`, forced splits), and what a leaf emits
(`max_delta_step`, `path_smooth`). Best-first selection still picks the
highest-gain leaf anywhere in the tree, and a rejected candidate simply makes
its leaf offer no split, exactly as the depth limit does today.

One rule here is CatBoost's, not LightGBM's
-------------------------------------------
`random_strength` (0.0) is CatBoost's score-noise regularizer, which
LightGBM has no equivalent of. At its default it takes no draw and changes
no bit, so the bundle's contract is unchanged; above 0 it adds a seeded
normal to every (feature, bin) candidate's gain before the search folds
them. The section below it in this file carries the CatBoost source it was
read from and the two places that source corrects the usual summary of the
formula.

Randomness stays counter-based
------------------------------
`extra_trees` draws from a counter-based splitmix64 stream keyed by
(seed, tree index, node id, feature), and `random_strength` from one keyed by
(seed, tree index, node id, feature, bin): the same construction
`sampling.mojo` uses for feature subsampling and for the same reason: a draw
must not depend on how many draws happened before it, so a tree is
reproducible whatever the thread count or the training history. This costs
LightGBM-identical streams for a given seed, and CatBoost-identical ones --
CatBoost seeds one RNG per candidate task and walks it in evaluation order,
which is exactly the dependence this construction refuses. `sampling.mojo`
already documents that trade against LightGBM.

Linear trees are a subsystem, not a tree control
------------------------------------------------
`linear_tree` and `linear_lambda` are deliberately absent here. A linear
tree does not add a control to the grower; it changes what a leaf *is*.
That subsystem is `linear_tree.mojo`: `BoosterParams.linear` switches it
on, the metric-path trainers (`custom_metric.mojo`) fit each grown tree's
leaves from the raw matrix, `Booster.linear` carries the sidecar, and
`serialize.mojo` writes it as the v5 `linear` section. Growth itself is
untouched, which is why nothing in this module knows about it.
"""

from std.math import exp, log, sqrt

from .cegb import CegbConfig
from .gain import leaf_score, soft_threshold_l1
from .monotone import MONOTONE_FREE, output_score
from .rng import GOLDEN, splitmix64, uniform

# LightGBM's extra_seed default. The other seeds already live with the
# features that use them (`sampling.DEFAULT_FEATURE_FRACTION_SEED`,
# `bagging.DEFAULT_BAGGING_SEED`).
comptime DEFAULT_EXTRA_SEED = 6

# monotone_constraints_method codes. Only `basic` is implemented; the other
# two are named so they can be rejected as the real features they are.
comptime DEFAULT_NUM_GRAD_QUANT_BINS = 4
"""LightGBM's `num_grad_quant_bins` default, read off
`include/LightGBM/config.h` (LightGBM 4.7.0.99).

Restated here rather than imported from `quantized_gradient.mojo`, which
already imports `raw_leaf_output` from this module: the edge goes one way and
a second one would be a cycle. The two constants are asserted equal in
`tests/test_cpu_quantized_grad.mojo`, which is the mechanism that keeps a
restatement from drifting.
"""

comptime DEFAULT_LEAF_ESTIMATION_ITERATIONS = 1
"""How many Newton steps a leaf's value is the result of.

**This project's default is 1 and stays 1**, which is LightGBM's behavior and
the behavior every fit in this repository has ever had: a leaf emits
`-T(G) / (H + lambda_l2)` evaluated once, at the raw scores the tree was grown
from.

CatBoost defaults some objectives to more than one step. Those numbers are
recorded in `boosting.catboost_leaf_estimation_iterations`, which lives there
rather than here because it is keyed on an objective code and this module must
not import the objective registry. They are a record of what CatBoost does,
for a caller that asks for CatBoost's settings by name; nothing in this
package reads them, and **this default does not move because of them**.
"""

comptime MONOTONE_BASIC = 0
comptime MONOTONE_INTERMEDIATE = 1
comptime MONOTONE_ADVANCED = 2

# LightGBM's kEpsilon, used by its monotone penalty as the "forbidden but not
# exactly zero" factor. Reproduced here because the penalty's whole top case
# is that value; the kEpsilon Hessian nudges elsewhere in LightGBM stay
# omitted (see categorical.mojo).
comptime MONOTONE_PENALTY_EPSILON = 1e-15


@always_inline
def _is_finite(v: Float64) -> Bool:
    """False for NaN and for either infinity."""
    return v == v and abs(v) <= Float64.MAX_FINITE


# ---------------------------------------------------------------------------
# min_gain_to_split
# ---------------------------------------------------------------------------


@always_inline
def passes_min_gain(gain: Float64, min_gain_to_split: Float64) -> Bool:
    """Whether a candidate's gain clears the floor.

    LightGBM compares an absolute gain against `parent_gain +
    min_gain_to_split` and rejects on `<=`. mojotrees's gains are already
    relative to the parent (see gain.mojo), so the same rule is `gain >
    min_gain_to_split`, strict on both sides.

    At the default 0.0 this is exactly today's rule: `find_best_split` starts
    from a best gain of 0.0 and accepts only a strictly larger one, so a
    caller that leaves the floor alone gets a bit-identical fit.
    """
    return gain > min_gain_to_split


# ---------------------------------------------------------------------------
# max_delta_step and path_smooth: what a leaf finally emits
# ---------------------------------------------------------------------------


@always_inline
def raw_leaf_output(
    grad_sum: Float64,
    hess_sum: Float64,
    lambda_l1: Float64,
    lambda_l2: Float64,
) -> Float64:
    """The unconstrained Newton step `-T(G) / (H + lambda_l2)`.

    The scalar form of what `tree._leaf_value` computes from a histogram, so
    the cap and the smoothing below can be composed onto it without a
    histogram in hand.
    """
    return -soft_threshold_l1(grad_sum, lambda_l1) / (hess_sum + lambda_l2)


@always_inline
def cap_leaf_output(value: Float64, max_delta_step: Float64) -> Float64:
    """LightGBM's `max_delta_step`: clamp a leaf's magnitude, keeping its
    sign. A value of 0.0 or less means no cap, as in LightGBM.

    This is the general form of what the Poisson objective already does with
    a fixed 0.7 (`boosting.poisson_max_delta_step`): that one bounds the
    Hessian, this one bounds the output. The two are independent and both may
    be active.
    """
    if max_delta_step <= 0.0:
        return value
    if value > max_delta_step:
        return max_delta_step
    if value < -max_delta_step:
        return -max_delta_step
    return value


@always_inline
def smooth_leaf_output(
    value: Float64,
    n_data: Int,
    path_smooth: Float64,
    parent_output: Float64,
) -> Float64:
    """LightGBM's path smoothing: pull a leaf's output toward its parent's,
    by an amount that fades as the leaf holds more rows.

        w   = n_data / path_smooth
        out = value * w / (w + 1) + parent_output / (w + 1)

    A leaf with `path_smooth` rows is the half-way point. `path_smooth <= 0`
    (the default) is the identity, so the unsmoothed path is untouched.

    The root has no parent; `parent_output` is 0.0 there, which shrinks the
    root's own output toward zero. That is what LightGBM's zero-initialized
    parent output amounts to at the root, and it is recorded in the handoff
    as the one behavior in this module to confirm against LightGBM before the
    parity table calls `path_smooth` supported.
    """
    if path_smooth <= 0.0:
        return value
    var w = Float64(n_data) / path_smooth
    return value * (w / (w + 1.0)) + parent_output / (w + 1.0)


def finish_leaf_output(
    value: Float64,
    max_delta_step: Float64,
    path_smooth: Float64,
    n_data: Int,
    parent_output: Float64,
) -> Float64:
    """A raw Newton output turned into the value a leaf emits: cap first,
    then smooth, which is LightGBM's order (the cap lives inside the base
    output calculation, the smoothing wraps it).

    Order matters: capping after smoothing would let a leaf pulled toward a
    large parent output exceed `max_delta_step`, and capping before it means
    the parent's contribution is itself already a capped value.
    """
    return smooth_leaf_output(
        cap_leaf_output(value, max_delta_step),
        n_data,
        path_smooth,
        parent_output,
    )


@always_inline
def split_gain_from_outputs(
    left_grad: Float64,
    left_hess: Float64,
    left_output: Float64,
    right_grad: Float64,
    right_hess: Float64,
    right_output: Float64,
    lambda_l2: Float64,
    parent_score: Float64,
) -> Float64:
    """A candidate's gain when the children are forced to emit given outputs.

    `left_grad` and `right_grad` are the child gradient sums *after* L1 soft
    thresholding. This is the same `output_score` form the monotone-
    constrained scan already uses in `split._split_gain`; it is what
    `max_delta_step` and `path_smooth` need, because both make a child's
    output something other than the free Newton step, and a gain scored at
    the free step would not be the gain the tree actually realizes.

    At the free Newton output this reduces to `G^2 / (H + lambda_l2)`, so a
    scan with neither cap nor smoothing scores exactly as it does today, up
    to floating-point association.
    """
    return (
        output_score(left_grad, left_hess, lambda_l2, left_output)
        + output_score(right_grad, right_hess, lambda_l2, right_output)
        - parent_score
    )


# ---------------------------------------------------------------------------
# Per-feature split penalties
# ---------------------------------------------------------------------------


struct FeaturePenalties(Copyable, Movable):
    """Per-feature costs charged against a candidate's gain.

    Two LightGBM mechanisms, applied at the same point (once a feature's best
    candidate is known) and both inactive by default:

    - `feature_contri`: a multiplier per feature. A feature with multiplier
      0.5 must find twice the gain to compete. Empty means every multiplier
      is 1.0. This struct owns it, and `contri_of` is its only reader.
    - Cost-effective gradient boosting, all four `cegb_*` parameters, held in
      `cegb` and implemented in `cegb.mojo`. This struct carries them so a
      caller sets every per-feature cost in one place; it does not charge
      them. `cegb.CegbNodeCosts` does, at `split._feature_gain`, immediately
      after the multiplier.

    THE DIVISION IS LOAD-BEARING. The multiplier scales a gain and the CEGB
    costs are absolute amounts subtracted from the scaled value, which is
    LightGBM's composition. Applying either one twice is the one way the two
    mechanisms silently corrupt each other, so each has exactly one owner:
    `penalized_gain` multiplies and never subtracts, and `cegb.mojo` subtracts
    and never multiplies.

    INTENTIONAL DIFFERENCE FROM LightGBM

    - Multipliers must be nonnegative. LightGBM accepts a negative
      `feature_contri`, which flips the sign of a gain and breaks the
      invariant that a chosen split has positive gain. Zero is accepted and
      means the feature can never win, which is the useful end of that range.
      `CegbConfig.check` rejects negative costs for the same reason.
    """

    var contri: List[Float64]
    var cegb: CegbConfig

    def __init__(out self):
        """No penalties: every multiplier 1.0, every cost 0.0."""
        self.contri = List[Float64]()
        self.cegb = CegbConfig()

    @staticmethod
    def from_contri(contri: List[Float64]) raises -> FeaturePenalties:
        """Gain multipliers only, LightGBM's `feature_contri`."""
        var out = FeaturePenalties()
        out.contri = contri.copy()
        return out^

    @staticmethod
    def from_cegb(
        tradeoff: Float64,
        penalty_split: Float64,
        penalty_feature_coupled: List[Float64] = [],
        penalty_feature_lazy: List[Float64] = [],
    ) raises -> FeaturePenalties:
        """The CEGB costs, with no gain multipliers.

        Named `from_cegb` rather than `cegb` because `cegb` is now the field
        this builds; the old name would shadow it.
        """
        var out = FeaturePenalties()
        out.cegb = CegbConfig.of(
            tradeoff,
            penalty_split,
            penalty_feature_coupled,
            penalty_feature_lazy,
        )
        return out^

    def contri_active(self) -> Bool:
        """Whether any gain multiplier would change a gain."""
        for f in range(len(self.contri)):
            if self.contri[f] != 1.0:
                return True
        return False

    def is_active(self) -> Bool:
        """Whether anything here would change a gain. An inactive bundle must
        leave the scan bit-identical, so the growers test this once per node
        rather than multiplying by 1.0 per candidate."""
        return self.cegb.is_active() or self.contri_active()

    def contri_of(self, feature: Int) -> Float64:
        """This feature's gain multiplier, 1.0 when unset."""
        if feature < 0 or feature >= len(self.contri):
            return 1.0
        return self.contri[feature]

    def penalized_gain(self, gain: Float64, feature: Int) -> Float64:
        """A candidate's gain after this feature's `feature_contri`
        multiplier, and nothing else.

        The CEGB terms are subtracted immediately after this, by
        `cegb.CegbNodeCosts.adjusted_gain_at`, which is the only place they
        are charged. Keeping the two apart is what lets a node's costs be
        prepared once and applied in one subtraction per candidate, and it is
        what makes double-charging impossible rather than merely unlikely.
        """
        return gain * self.contri_of(feature)

    def check_features(self, n_features: Int) raises:
        """Raise unless the vectors fit a dataset with `n_features` columns
        and hold usable numbers."""
        if len(self.contri) > 0 and len(self.contri) != n_features:
            raise Error(
                "feature_contri has ",
                len(self.contri),
                " entries but the data has ",
                n_features,
                " features",
            )
        for f in range(len(self.contri)):
            if not _is_finite(self.contri[f]) or self.contri[f] < 0.0:
                raise Error(
                    "feature_contri[",
                    f,
                    "] must be a finite nonnegative number",
                )
        self.cegb.check(n_features)


# ---------------------------------------------------------------------------
# extra_trees: one random threshold per feature
# ---------------------------------------------------------------------------


def extra_split_stream(
    seed: Int, tree_index: Int, node: Int, feature: Int
) -> UInt64:
    """The counter stream for one (tree, node, feature) threshold draw.

    Sign bits are masked off so negative seeds are accepted, as in
    `sampling._stream`. The feature is mixed in last, so two features at one
    node draw independently and a feature's draw does not depend on how many
    features were scanned before it: the same tree comes out whatever the
    scan order, the thread count, or the subsampled feature set.
    """
    var h = splitmix64(UInt64(seed & 0x7FFFFFFFFFFFFFFF))
    h = splitmix64(h ^ UInt64(tree_index & 0x7FFFFFFFFFFFFFFF))
    h = splitmix64(h ^ UInt64((node + 1) & 0x7FFFFFFFFFFFFFFF))
    return splitmix64(h ^ UInt64((feature + 1) & 0x7FFFFFFFFFFFFFFF))


def extra_candidate_index(n_candidates: Int, stream: UInt64) -> Int:
    """A uniform index in `[0, n_candidates)` from a counter stream, or -1
    when there is nothing to choose from.

    The 53-bit uniform is rng.mojo's `uniform`, the one feature sampling
    draws, scaled and truncated; the final guard covers the rounding case
    where the scaled value reaches `n_candidates`.
    """
    if n_candidates <= 0:
        return -1
    if n_candidates == 1:
        return 0
    var u = uniform(stream)
    var index = Int(u * Float64(n_candidates))
    if index >= n_candidates:
        index = n_candidates - 1
    return index


def extra_threshold_index(
    n_candidates: Int, seed: Int, tree_index: Int, node: Int, feature: Int
) -> Int:
    """Which of a feature's `n_candidates` thresholds `extra_trees` evaluates
    at this node, or -1 when the feature offers none.

    LightGBM's extra-trees mode scores exactly one threshold per feature and
    then picks the best feature by gain, so a feature whose single draw fails
    `min_data_in_leaf` or `min_sum_hessian_in_leaf` yields no split at all
    rather than falling back to a full scan. That is the rule; the caller
    keeps every other check unchanged.
    """
    return extra_candidate_index(
        n_candidates, extra_split_stream(seed, tree_index, node, feature)
    )


# ---------------------------------------------------------------------------
# random_strength: seeded noise on a candidate's gain (CatBoost)
# ---------------------------------------------------------------------------
#
# The one CatBoost regularizer LightGBM has no equivalent of. Read from
# CatBoost `master` (fetched 2026-08-16), not from the paper or the docs:
#
#   catboost/private/libs/algo/greedy_tensor_search.cpp
#       CalcScoreStDev = RandomStrength
#                      * derivativesStDevFromZero
#                      * (RandomScoreType == NormalWithModelSizeDecrease
#                         ? CalcDerivativesStDevFromZeroMultiplier(n, L)
#                         : 1.0)
#       CalcDerivativesStDevFromZeroPlainBoosting(fold)
#           = sqrt(sum over dims and rows of d^2 / rows)
#       CalcDerivativesStDevFromZeroMultiplier(n, L)
#           modelLeft = exp(log(n) - L);  return modelLeft / (1 + modelLeft)
#   catboost/private/libs/algo/train.cpp
#       modelLength L = iteration_index * learning_rate
#   catboost/private/libs/algo/rand_score.h
#       TRandomScore::GetInstance = Val + NormalDistribution(rand, 0, StDev)
#   catboost/private/libs/algo/tensor_search_helpers.cpp
#       SetBestScore: every candidate bin is noised, then the argmax over
#       bins is taken on the *noised* values
#
# Two things the source corrects about the short summary of this formula:
#
# 1. `derivativesStDevFromZero` is not a standard deviation. No mean is
#    subtracted: it is the root-mean-square of the per-row weighted
#    derivatives about zero, taken over the whole learn set, once per tree.
#    It is a property of the ensemble's current residuals, not of the node
#    being split, and it cannot be recovered from a node histogram (a
#    histogram carries sums of g, never sums of g squared).
# 2. the "model-size decrease" is not a size. It is
#    n / (n + exp(L)) written as modelLeft / (1 + modelLeft): a factor that
#    starts at n/(n+1), effectively 1, on the first tree and decays toward 0
#    once iteration * learning_rate passes log(n). The noise fades as the
#    model grows, which is the point of the term. It applies only under
#    CatBoost's default `random_score_type=NormalWithModelSizeDecrease`; the
#    `Gumbel` type drops it and changes the distribution, and is not
#    implemented here.
#
# Units, which the summary does not mention and which matter. CatBoost's
# default CPU score function is `Cosine`, whose value is
# sum(leafOut * sumDelta) / sqrt(sum(leafOut^2 * sumWeight)), dimensionally a
# gradient, so noise scaled by a gradient RMS is dimensionally consistent
# with it. mojotrees's gain is the second-order gain, dimensionally
# gradient^2 / hessian, which is CatBoost's `score_function=L2` shape. The
# scaling below is applied to that gain unchanged, which is CatBoost's own
# behavior under `score_function=L2` -- CatBoost applies one scoreStDev to
# whichever score function is configured -- but it is not the pairing
# CatBoost ships by default, and the useful range of `random_strength` on a
# gain is therefore not the useful range CatBoost documents.
#
# Where the draw is keyed, and why it is not CatBoost's keying. CatBoost
# seeds one RNG per candidate *task* (`SetBestScore(randSeed + taskIdx, ...)`)
# and walks it in bin order inside that task, so its draw depends on how the
# candidate list was blocked. mojotrees keys every draw by
# (seed, tree, node, feature, bin) instead, exactly as `extra_trees` above
# keys by (seed, tree, node, feature): a draw must not depend on how many
# draws happened before it, or the answer would move with
# MOJOTREES_NUM_WORKERS. That is a deliberate divergence from CatBoost's
# stream and it costs CatBoost-identical noise for a given seed, which is the
# same trade `sampling.mojo` already documents against LightGBM.

# CatBoost's `random_seed` default, which is what its noise stream is keyed
# from. Not a parameter-string key here: the string surface is LightGBM's,
# and `random_strength` is the only name added to it.
comptime DEFAULT_RANDOM_STRENGTH_SEED = 0

# Domain separator folded into the seed so this stream can never coincide
# with `extra_split_stream`'s even when both seeds are equal. ASCII
# "RANDSCOR".
comptime _RANDOM_SCORE_DOMAIN = UInt64(0x52414E4453434F52)

# Marsaglia's polar method rejects a draw outside the unit disc, which is
# 1 - pi/4 of the plane. Sixty-four rejections in a row has probability about
# 3e-43; the bound exists so the loop is provably finite, not because it is
# expected to be reached.
comptime _POLAR_MAX_TRIES = 64


def derivatives_stdev_from_zero(
    gradients: List[Float64], n_rows: Int
) raises -> Float64:
    """CatBoost's `CalcDerivativesStDevFromZeroPlainBoosting`: the RMS of the
    weighted derivatives about zero.

    `gradients` is the ensemble's current gradient vector and `n_rows` the
    number of rows *per output dimension*, so a multiclass caller passes the
    flat `n_rows * n_class` vector and the row count. That matches CatBoost,
    which sums the squares over every dimension and divides by the row count
    of one dimension rather than by the total length.

    Plain boosting only. CatBoost's ordered-boosting variant takes the same
    RMS over each fold's tail segment instead; mojotrees has no ordered
    boosting, so there is nothing here to select between.
    """
    if n_rows <= 0:
        raise Error(
            "derivatives_stdev_from_zero needs a positive row count, got ",
            n_rows,
        )
    if len(gradients) % n_rows != 0:
        raise Error(
            "the gradient vector must be a whole number of output dimensions"
            " of ",
            n_rows,
            " rows, got ",
            len(gradients),
        )
    var sum2 = 0.0
    for i in range(len(gradients)):
        sum2 += gradients[i] * gradients[i]
    return sqrt(sum2 / Float64(n_rows))


def model_size_decrease(
    learn_sample_count: Int, model_length: Float64
) raises -> Float64:
    """CatBoost's `CalcDerivativesStDevFromZeroMultiplier`.

    `model_length` is CatBoost's `modelLength`, which `train.cpp` computes as
    `iteration_index * learning_rate` -- the boosted length of the model so
    far, not a node count and not a byte count. The factor is
    `exp(log(n) - L) / (1 + exp(log(n) - L))`, written here exactly as
    CatBoost writes it rather than as the algebraically equal
    `n / (n + exp(L))`, so the same two roundings happen in the same order.

    An overflow of the exponential is the L -> -infinity limit of the same
    expression and returns 1.0 rather than a NaN.
    """
    if learn_sample_count <= 0:
        raise Error(
            "model_size_decrease needs a positive learn sample count, got ",
            learn_sample_count,
        )
    if not _is_finite(model_length):
        raise Error("model_size_decrease needs a finite model length")
    var model_left = exp(log(Float64(learn_sample_count)) - model_length)
    if not _is_finite(model_left):
        return 1.0
    return model_left / (1.0 + model_left)


def random_score_scale_from_gradients(
    gradients: List[Float64], n_rows: Int, model_length: Float64
) raises -> Float64:
    """The per-tree scale `ExtraTreeParams.random_score_scale` wants:
    `derivatives_stdev_from_zero * model_size_decrease`.

    This is everything in CatBoost's `CalcScoreStDev` except the
    `random_strength` multiplier itself, which stays on the parameter bundle
    because it is the user's knob and this is the ensemble's state. One call
    per tree, before growth, from whichever trainer owns the gradient vector.
    """
    return derivatives_stdev_from_zero(gradients, n_rows) * (
        model_size_decrease(n_rows, model_length)
    )


def random_score_stream(
    seed: Int, tree_index: Int, node: Int, feature: Int, bin: Int
) -> UInt64:
    """The counter key for one candidate's noise draw.

    Keyed by (seed, tree, node, feature, bin) and by nothing else. It reads
    no counter that advances with evaluation order, so the draw for a
    candidate is the same value whether that candidate was the first scored
    in the node or the last, whether its feature ran on its own task or
    shared one, and at any `MOJOTREES_NUM_WORKERS`. Sign bits are masked off
    so negative seeds are accepted, as in `extra_split_stream`.

    The bin is mixed in last, and `bin + 1` rather than `bin` so that the
    lowest bin is not the identity element of the xor.
    """
    var h = splitmix64(
        UInt64(seed & 0x7FFFFFFFFFFFFFFF) ^ _RANDOM_SCORE_DOMAIN
    )
    h = splitmix64(h ^ UInt64(tree_index & 0x7FFFFFFFFFFFFFFF))
    h = splitmix64(h ^ UInt64((node + 1) & 0x7FFFFFFFFFFFFFFF))
    h = splitmix64(h ^ UInt64((feature + 1) & 0x7FFFFFFFFFFFFFFF))
    return splitmix64(h ^ UInt64((bin + 1) & 0x7FFFFFFFFFFFFFFF))


def standard_normal(stream: UInt64) -> Float64:
    """A standard normal draw from one counter key, by Marsaglia's polar
    method.

    Polar rather than Box-Muller because Box-Muller needs a cosine and this
    needs only `log` and `sqrt`, which the package already depends on
    elsewhere; one fewer libm function on a path whose bits have to reproduce
    across the x86-64 and ARM64 CI legs.

    Rejection does not make this stream-dependent. A rejected pair advances a
    counter derived from `stream` alone, so the accepted pair for a given key
    is a function of that key and the number of rejections it takes is a
    property of the key rather than of the evaluation order.
    """
    var i = 0
    while i < _POLAR_MAX_TRIES:
        var base = stream + GOLDEN * UInt64(2 * i)
        var u = 2.0 * uniform(base) - 1.0
        var v = 2.0 * uniform(base + GOLDEN) - 1.0
        var s = u * u + v * v
        if s > 0.0 and s < 1.0:
            return u * sqrt(-2.0 * log(s) / s)
        i += 1
    return 0.0


@always_inline
def random_score_noise(
    stdev: Float64,
    seed: Int,
    tree_index: Int,
    node: Int,
    feature: Int,
    bin: Int,
) -> Float64:
    """The noise added to one (feature, bin) candidate's gain: a draw from
    N(0, `stdev`), keyed as `random_score_stream` describes.

    `stdev` is `ExtraTreeParams.random_score_stdev()`, which is
    `random_strength * random_score_scale`. At or below zero this is exactly
    0.0 and no stream is touched, which is the default path.
    """
    if not (stdev > 0.0) or not _is_finite(stdev):
        return 0.0
    return stdev * standard_normal(
        random_score_stream(seed, tree_index, node, feature, bin)
    )


# ---------------------------------------------------------------------------
# Categorical regularization and threshold arithmetic
# ---------------------------------------------------------------------------
#
# These add no parameters. `categorical.CategoricalParams` remains the single
# owner of `cat_smooth`, `cat_l2`, `max_cat_threshold`, `max_cat_to_onehot`,
# and `min_data_per_group`; what is factored out here is the arithmetic
# `categorical.mojo` performs inline, so the gain floor, the gain
# multipliers, the output cap, and the smoothing can be applied to category
# partitions through the same formulas the ordinal scan uses.


@always_inline
def cat_effective_l2(lambda_l2: Float64, cat_l2: Float64) -> Float64:
    """The L2 term a many-vs-many category split scores its *children* with.

    The parent's gain shift keeps plain `lambda_l2`, which is what puts
    categorical and numerical gains on one scale; that asymmetry is
    LightGBM's and is preserved.
    """
    return lambda_l2 + cat_l2


@always_inline
def cat_sort_key(
    grad_sum: Float64, hess_sum: Float64, cat_smooth: Float64
) -> Float64:
    """A category's position in the gradient-statistics order,
    `G / (H + cat_smooth)`. Fisher's ordering: prefixes of it contain the
    optimal many-vs-many partition for the second-order objective."""
    return grad_sum / (hess_sum + cat_smooth)


@always_inline
def cat_enters_search(count: Int, cat_smooth: Float64) -> Bool:
    """Whether a category has enough rows to be a candidate at all. Rows of
    an excluded category stay with the right child, which is where bin 0
    already routes."""
    return Float64(count) >= cat_smooth


def cat_side_cap(n_used_categories: Int, max_cat_threshold: Int) -> Int:
    """How many categories one side of a many-vs-many split may hold.

    `max_cat_threshold`, further capped at half the usable categories
    (rounded up) so the two directions of the walk cannot both claim the
    whole set. Never negative, so a caller can loop on it directly.
    """
    var cap = max_cat_threshold
    var half = (n_used_categories + 1) // 2
    if half < cap:
        cap = half
    if cap < 0:
        cap = 0
    return cap


def cat_partition_gain(
    left_grad: Float64,
    left_hess: Float64,
    right_grad: Float64,
    right_hess: Float64,
    lambda_l1: Float64,
    lambda_l2: Float64,
    cat_l2: Float64,
    parent_score: Float64,
) -> Float64:
    """A many-vs-many partition's gain: both children scored with
    `cat_effective_l2`, the parent's shift left at `lambda_l2`."""
    var l2 = cat_effective_l2(lambda_l2, cat_l2)
    return (
        leaf_score(left_grad, left_hess, lambda_l1, l2)
        + leaf_score(right_grad, right_hess, lambda_l1, l2)
        - parent_score
    )


# ---------------------------------------------------------------------------
# Monotone penalty and method selection
# ---------------------------------------------------------------------------


def monotone_penalty_factor(depth: Int, penalization: Float64) -> Float64:
    """LightGBM's `monotone_penalty`, as a multiplier on a constrained
    split's gain.

        penalization >= depth + 1  ->  epsilon      (effectively forbidden)
        penalization <= 1          ->  1 - p / 2^depth
        otherwise                  ->  1 - 2^(p - 1 - depth)

    plus LightGBM's kEpsilon in the two lower cases. Depth is counted in
    edges from the root, the same quantity `max_depth` bounds, so a
    penalization of 2 forbids monotone splits at depths 0 and 1 and lets them
    back in, heavily discounted, at depth 2. A penalization of 0 is the
    identity, which is the default.
    """
    if penalization <= 0.0:
        return 1.0
    if penalization >= Float64(depth) + 1.0:
        return MONOTONE_PENALTY_EPSILON
    if penalization <= 1.0:
        return (
            1.0
            - penalization / (2.0 ** Float64(depth))
            + MONOTONE_PENALTY_EPSILON
        )
    return (
        1.0
        - 2.0 ** (penalization - 1.0 - Float64(depth))
        + MONOTONE_PENALTY_EPSILON
    )


def apply_monotone_penalty(
    gain: Float64, sign: Int, depth: Int, penalization: Float64
) -> Float64:
    """`gain` after the monotone penalty, which applies only to a split on a
    feature that actually carries a constraint. An unconstrained feature is
    never penalized, whatever the depth, so a model with no monotone
    constraints is unaffected by any value of `monotone_penalty`."""
    if sign == MONOTONE_FREE or penalization <= 0.0:
        return gain
    return gain * monotone_penalty_factor(depth, penalization)


def parse_monotone_method(name: String) raises -> Int:
    """`monotone_constraints_method` by LightGBM's name.

    Only `basic` exists here. `intermediate` and `advanced` are real LightGBM
    methods that recover some of the accuracy `basic` gives up by tracking
    bounds across whole subtrees rather than parent to child; they are named
    in the error rather than silently accepted, because a caller who asks for
    one and gets `basic` has a differently regularized model than the one
    they asked for.
    """
    if name == "basic":
        return MONOTONE_BASIC
    if name == "intermediate":
        raise Error(
            "monotone_constraints_method 'intermediate' is not implemented;"
            " mojotrees implements 'basic', which tracks each node's output"
            " interval from its parent alone"
        )
    if name == "advanced":
        raise Error(
            "monotone_constraints_method 'advanced' is not implemented;"
            " mojotrees implements 'basic', which tracks each node's output"
            " interval from its parent alone"
        )
    raise Error(
        "unknown monotone_constraints_method '",
        name,
        "'; expected basic, intermediate, or advanced",
    )


def monotone_method_name(method: Int) raises -> String:
    """The LightGBM name for a method code, for reporting."""
    if method == MONOTONE_BASIC:
        return "basic"
    if method == MONOTONE_INTERMEDIATE:
        return "intermediate"
    if method == MONOTONE_ADVANCED:
        return "advanced"
    raise Error("unknown monotone_constraints_method code ", method)


# ---------------------------------------------------------------------------
# Derivative precision
# ---------------------------------------------------------------------------

comptime DERIV_PRECISION_FLOAT32 = 0
"""`derivative_precision = "float32"`, the default: LightGBM's `score_t`.

A per-row gradient and hessian is a Float32 quantity, narrowed at the
objective and re-narrowed at every histogram read site. The whole argument,
including the measured accuracy trade that made this a switch rather than a
constant, is in `histogram.DERIVATIVE_PRECISION_FLOAT32` and in
`docs/NUMERICS.md`.

The codes are spelled here rather than imported from `histogram` because
`binning` imports this module and `histogram` imports `binning`, so the
import edge only goes one way. They are two integers and a pair of names,
and `histogram.derivative_precision_narrows` is the single decision point
that both sides agree with.
"""

comptime DERIV_PRECISION_FLOAT64 = 1
"""`derivative_precision = "float64"`: per-row derivatives stay Float64.

Opt-in, moves bits by design, and gives up the gathered pair buffer and the
row-blocked histograms with it -- see
`histogram.DERIVATIVE_PRECISION_FLOAT64`.
"""


def parse_derivative_precision(name: String) raises -> Int:
    """The code for a `derivative_precision` value, refusing anything else.

    Two names and no more. `double`, `f64`, `64` and the rest are not
    LightGBM's spelling of anything and are not accepted, on the same rule
    `parse_monotone_method` applies: a name that is not the parameter's name
    is a mistake, and a mistake that resolved to the default would run one
    arm of an A/B under the other's label.
    """
    if name == "float32":
        return DERIV_PRECISION_FLOAT32
    if name == "float64":
        return DERIV_PRECISION_FLOAT64
    raise Error(
        "unknown derivative_precision '",
        name,
        "'; expected float32 or float64",
    )


def derivative_precision_name(precision: Int) raises -> String:
    """The name for a precision code, for reporting."""
    if precision == DERIV_PRECISION_FLOAT32:
        return "float32"
    if precision == DERIV_PRECISION_FLOAT64:
        return "float64"
    raise Error("unknown derivative_precision code ", precision)


# ---------------------------------------------------------------------------
# Forced splits
# ---------------------------------------------------------------------------


@fieldwise_init
struct ForcedSplitNode(Copyable, Movable):
    """One node of a forced-split tree.

    `feature` is a dataset column and `threshold` a raw feature value, not a
    bin id: the forced tree is written against the data, and the grower maps
    the threshold through the bin mapper when it applies it. `left` and
    `right` index into `ForcedSplits.nodes`, or are -1 when that side is
    where ordinary leaf-wise growth resumes.
    """

    var feature: Int
    var threshold: Float64
    var left: Int
    var right: Int


struct ForcedSplits(Copyable, Movable):
    """A validated forced-split tree, LightGBM's `forcedsplits_filename`
    content as a structure.

    `nodes[0]` is the root when there is one; an empty list means no forced
    splits and is the default. A parent always precedes its children in the
    list, so the structure is acyclic by construction and one ascending pass
    covers it.

    `bins` is the same tree in bin space: `bins[i]` is the threshold bin
    `nodes[i].threshold` maps to under a fitted `BinMapper`, and the grower
    applies *that*, because it is handed a `BinnedMatrix` and a binned matrix
    carries no bin edges. It is empty on a freshly parsed document and filled
    by `binning.map_forced_splits`, which is the only place a raw threshold
    and a mapper meet. `is_mapped` is the predicate every consumer tests:
    a document that has been parsed but not mapped is refused rather than
    applied, because applying it would mean guessing a bin.
    """

    var nodes: List[ForcedSplitNode]
    var bins: List[Int]

    def __init__(
        out self,
        var nodes: List[ForcedSplitNode],
        var bins: List[Int] = [],
    ):
        """A `bins` table of the wrong length (the empty default included)
        means the tree has not been mapped to bins yet, which is what a parsed
        document is. Appending the argument keeps every one-argument caller
        working unchanged."""
        var n = len(nodes)
        self.nodes = nodes^
        if len(bins) == n:
            self.bins = bins^
        else:
            self.bins = List[Int]()

    @staticmethod
    def none() -> ForcedSplits:
        return ForcedSplits(List[ForcedSplitNode]())

    def is_empty(self) -> Bool:
        return len(self.nodes) == 0

    def is_mapped(self) -> Bool:
        """Whether every node carries the threshold bin the grower applies.
        False for an empty tree, which has nothing to apply."""
        return len(self.nodes) > 0 and len(self.bins) == len(self.nodes)

    def bin_at(self, i: Int) raises -> Int:
        """Node i's threshold bin. Raises unless the tree has been mapped, so
        a caller cannot read a bin that was never computed."""
        if not self.is_mapped():
            raise Error(
                "forced splits have not been mapped to bins; call"
                " binning.map_forced_splits with the fitted BinMapper"
            )
        if i < 0 or i >= len(self.bins):
            raise Error("forced split node index out of range")
        return self.bins[i]

    def n_nodes(self) -> Int:
        return len(self.nodes)

    def depth(self) -> Int:
        """The depth of the deepest forced node, in edges from the root: 0
        for a single forced split. This is what a caller compares against
        `max_depth`, and `n_nodes() + 1` is the number of leaves the forced
        tree alone would produce, which must not exceed `num_leaves`."""
        if len(self.nodes) == 0:
            return 0
        var best = 0
        var stack: List[Int] = [0]
        var depths: List[Int] = [0]
        while len(stack) > 0:
            var i = stack.pop()
            var d = depths.pop()
            if d > best:
                best = d
            if self.nodes[i].left >= 0:
                stack.append(self.nodes[i].left)
                depths.append(d + 1)
            if self.nodes[i].right >= 0:
                stack.append(self.nodes[i].right)
                depths.append(d + 1)
        return best

    def check_features(self, n_features: Int) raises:
        """Raise unless every forced node names a column this dataset has."""
        for i in range(len(self.nodes)):
            var f = self.nodes[i].feature
            if f < 0 or f >= n_features:
                raise Error(
                    "forced splits: feature ",
                    f,
                    " is out of range for ",
                    n_features,
                    " features",
                )

    def check_budget(self, num_leaves: Int, max_depth: Int) raises:
        """Raise unless the forced tree fits inside the growth budget.

        A forced tree with more leaves than `num_leaves`, or deeper than
        `max_depth`, cannot be grown at all. LightGBM stops forcing when it
        runs out of budget; failing here instead means a caller who wrote a
        forced tree that cannot fit is told so rather than silently getting
        part of it.
        """
        if len(self.nodes) == 0:
            return
        var leaves = len(self.nodes) + 1
        if leaves > num_leaves:
            raise Error(
                "forced splits need ",
                leaves,
                " leaves but num_leaves is ",
                num_leaves,
            )
        if max_depth > 0 and self.depth() + 1 > max_depth:
            raise Error(
                "forced splits reach depth ",
                self.depth() + 1,
                " but max_depth is ",
                max_depth,
            )


def _is_number_char(c: String) -> Bool:
    return (
        c == "0"
        or c == "1"
        or c == "2"
        or c == "3"
        or c == "4"
        or c == "5"
        or c == "6"
        or c == "7"
        or c == "8"
        or c == "9"
        or c == "+"
        or c == "-"
        or c == "."
        or c == "e"
        or c == "E"
    )


struct _Cursor(Movable):
    """A byte cursor over the forced-split text.

    The forced-split file is the one place mojotrees reads a JSON-shaped
    document, and it is a fixed four-key schema, so this parses that schema
    rather than pulling in general JSON. Anything outside the schema is an
    error with a byte offset, not a silently ignored key.
    """

    var text: String
    var pos: Int

    def __init__(out self, var text: String):
        self.text = text^
        self.pos = 0

    def at_end(self) -> Bool:
        return self.pos >= self.text.byte_length()

    def peek(self) raises -> String:
        if self.at_end():
            return String("")
        return String(self.text[byte=self.pos : self.pos + 1])

    def skip_ws(mut self) raises:
        while not self.at_end():
            var c = self.peek()
            if c == " " or c == "\t" or c == "\n" or c == "\r":
                self.pos += 1
            else:
                break

    def expect(mut self, c: String) raises:
        self.skip_ws()
        if self.peek() != c:
            raise Error(
                "forced splits: expected '",
                c,
                "' at byte ",
                self.pos,
            )
        self.pos += 1

    def parse_key(mut self) raises -> String:
        self.skip_ws()
        if self.peek() != "\"":
            raise Error(
                "forced splits: expected a quoted key at byte ", self.pos
            )
        self.pos += 1
        var start = self.pos
        while not self.at_end() and self.peek() != "\"":
            self.pos += 1
        if self.at_end():
            raise Error("forced splits: unterminated key")
        var key = String(self.text[byte=start : self.pos])
        self.pos += 1
        return key^

    def parse_number(mut self) raises -> Float64:
        self.skip_ws()
        var start = self.pos
        while not self.at_end() and _is_number_char(self.peek()):
            self.pos += 1
        if self.pos == start:
            raise Error("forced splits: expected a number at byte ", start)
        var text = String(self.text[byte=start : self.pos])
        try:
            return Float64(text)
        except:
            raise Error("forced splits: '", text, "' is not a number")

    def parse_node(mut self, mut nodes: List[ForcedSplitNode]) raises -> Int:
        """Parse one `{...}` node, append it and its subtree to `nodes`, and
        return its index. The node is appended before its children are
        parsed, so a parent's index is always below its children's."""
        self.expect("{")
        var index = len(nodes)
        nodes.append(ForcedSplitNode(-1, 0.0, -1, -1))
        var feature = -1
        var threshold = 0.0
        var left = -1
        var right = -1
        var saw_feature = False
        var saw_threshold = False
        self.skip_ws()
        if self.peek() == "}":
            raise Error("forced splits: a node needs 'feature'")
        while True:
            var key = self.parse_key()
            self.expect(":")
            if key == "feature":
                if saw_feature:
                    raise Error("forced splits: duplicate 'feature'")
                var v = self.parse_number()
                if v != Float64(Int(v)):
                    raise Error(
                        "forced splits: 'feature' must be a whole number,"
                        " got ",
                        v,
                    )
                feature = Int(v)
                if feature < 0:
                    raise Error(
                        "forced splits: 'feature' must be nonnegative, got ",
                        feature,
                    )
                saw_feature = True
            elif key == "threshold":
                if saw_threshold:
                    raise Error("forced splits: duplicate 'threshold'")
                threshold = self.parse_number()
                if not _is_finite(threshold):
                    raise Error("forced splits: 'threshold' must be finite")
                saw_threshold = True
            elif key == "left":
                if left >= 0:
                    raise Error("forced splits: duplicate 'left'")
                left = self.parse_node(nodes)
            elif key == "right":
                if right >= 0:
                    raise Error("forced splits: duplicate 'right'")
                right = self.parse_node(nodes)
            elif key == "cat_threshold":
                raise Error(
                    "forced splits: 'cat_threshold' is not supported; a"
                    " forced categorical split needs the category set to be"
                    " mapped through the fitted category table, which this"
                    " structure does not carry"
                )
            else:
                raise Error(
                    "forced splits: unknown key '",
                    key,
                    "'; expected feature, threshold, left, or right",
                )
            self.skip_ws()
            if self.peek() == ",":
                self.pos += 1
                continue
            break
        self.expect("}")
        if not saw_feature:
            raise Error("forced splits: a node needs 'feature'")
        if not saw_threshold:
            raise Error("forced splits: a node needs 'threshold'")
        nodes[index] = ForcedSplitNode(feature, threshold, left, right)
        return index


def parse_forced_splits(spec: String) raises -> ForcedSplits:
    """Parse LightGBM's forced-splits document into a validated structure.

    The schema is a nested object with the keys `feature`, `threshold`,
    `left`, and `right`; `left` and `right` hold further nodes and may be
    omitted, in which case ordinary leaf-wise growth resumes on that side.
    An empty or blank document means no forced splits.

    Every deviation is an error rather than an ignored key, matching
    `params.parse_params`: a forced-split file is a statement about the model
    a user wants, and a typo that silently drops a level of the tree is worse
    than a failed run. `cat_threshold`, LightGBM's categorical forced split,
    is rejected by name rather than as an unknown key.
    """
    var cursor = _Cursor(spec.copy())
    cursor.skip_ws()
    if cursor.at_end():
        return ForcedSplits.none()
    var nodes = List[ForcedSplitNode]()
    _ = cursor.parse_node(nodes)
    cursor.skip_ws()
    if not cursor.at_end():
        raise Error(
            "forced splits: trailing text after the root node at byte ",
            cursor.pos,
        )
    return ForcedSplits(nodes^)


# ---------------------------------------------------------------------------
# Options that must be rejected rather than ignored
# ---------------------------------------------------------------------------


def check_extra_option_supported(name: String) raises:
    """Raise for a LightGBM tree option that is real but not implemented.

    The repository's rule (see `params._raise_if_unimplemented_objective`) is
    that a name for a genuine LightGBM feature gets an error saying what it
    would take, never an "unknown parameter" message and never silence.
    Names this module *does* implement are not listed here.
    """
    if (
        name == "forcedsplits_filename"
        or name == "forced_splits_filename"
        or name == "forced_splits"
        or name == "fs"
    ):
        raise Error(
            "'forcedsplits_filename' names a document, which a"
            " whitespace-separated parameter string cannot carry any more"
            " than it can carry 'monotone_constraints'. Forced splits are"
            " implemented and reachable from the Mojo API: read the file,"
            " `parse_forced_splits` it, map it with"
            " `binning.map_forced_splits(mapper, forced)`, and set the result"
            " on `TreeParams.extra.forced`. `tree.grow_tree` applies it"
        )


def check_feature_pre_filter(enabled: Bool) raises:
    """Accept `feature_pre_filter=false`; refuse `true` *in a parameter
    string*, because a parameter string cannot reach the filter.

    LightGBM's prefilter is a Dataset construction step: before training it
    drops the features whose every bin boundary would leave a child under
    `min_data_in_leaf`. mojotrees's split search rejects those candidates as
    it scans (`split.find_best_split`), so **`false` is not an unimplemented
    option, it is the option mojotrees implements**, and it used to be refused
    by name for any value at all. A LightGBM configuration that spells out
    the setting mojotrees matches now ports across unchanged.

    `true` is no longer unimplemented either. `binning.fit_bins` takes
    `feature_pre_filter` and `min_data_in_leaf`, counts each feature's bins on
    the sample it fit the edges from, runs `binning.need_filter` (LightGBM's
    `NeedFilter`, transcribed) against `binning.filter_count` (LightGBM's
    `filter_cnt`, `min_data_in_leaf` scaled to the sample), and hands back a
    `BinMapper` whose `usable` list is LightGBM's `used_features`. The list
    rides onto the `BinnedMatrix` through `transform`, and
    `sampling.select_tree_features` takes it as the pool, which is what makes
    the narrowed `feature_fraction` draw the *same* narrowing LightGBM does.

    What a parameter string still cannot do is carry it. `params.TrainConfig`
    holds neither this flag nor `min_data_in_leaf`'s route into `fit_bins`, so
    a string that said `feature_pre_filter=true` would be a setting this
    library read and then ignored, which is the one outcome this repository
    refuses outright. It is refused here for that reason and reported as such,
    the same way `forcedsplits_filename` is refused for naming a document a
    string cannot carry while being fully implemented and reachable from the
    Mojo API.

    Takes the parsed value rather than the key, because the two values mean
    different things here -- which is exactly what a name-only check could not
    express. `check_extra_option_supported` therefore lets the name through.
    """
    if not enabled:
        return
    raise Error(
        "'feature_pre_filter=true' cannot be set from a parameter string:"
        " TrainConfig carries neither this flag nor min_data_in_leaf into"
        " binning, so the string would be read and ignored. The filter itself"
        " is implemented and reachable from the Mojo API: fit with"
        " binning.fit_bins(..., feature_pre_filter=True,"
        " min_data_in_leaf=<the same number the trees use>), transform with"
        " that mapper, and the resulting BinnedMatrix carries LightGBM's"
        " used_features as `usable` for sampling.select_tree_features to draw"
        " from. Note LightGBM's own rule that a prefiltered Dataset cannot"
        " then be reused with a larger min_data_in_leaf"
    )


# ---------------------------------------------------------------------------
# The parameter bundle
# ---------------------------------------------------------------------------


struct ExtraTreeParams(Copyable, Movable):
    """The tree controls in this module, with LightGBM's defaults.

    Held apart from `tree.TreeParams` so this lane changes no shared struct;
    the handoff specifies the fields to fold into `TreeParams` at
    integration. Defaults are LightGBM's, and `is_active` is False for them,
    which is the contract the growers rely on: an untouched bundle must leave
    every fit bit-identical to today's.
    """

    var min_gain_to_split: Float64
    var max_delta_step: Float64
    var path_smooth: Float64
    var extra_trees: Bool
    var extra_seed: Int
    var monotone_penalty: Float64
    var monotone_method: Int
    var penalties: FeaturePenalties
    var forced: ForcedSplits

    # CatBoost's `random_strength`, the one rule on this bundle that is not
    # LightGBM's. `random_strength` is the user's knob and defaults to 0,
    # which is exactly LightGBM's behavior: no draw is taken, no gain is
    # touched, and `is_active` stays False for it.
    #
    # `random_score_scale` is the rest of CatBoost's `CalcScoreStDev` --
    # `derivatives_stdev_from_zero * model_size_decrease` -- and is a
    # property of the ensemble at this iteration rather than of the tree
    # controls, so it is supplied per tree by whoever owns the gradient
    # vector (`random_score_scale_from_gradients`). A split search is handed
    # a histogram, and no histogram carries the sum of squared gradients this
    # needs, so it cannot be reconstructed there. 0.0 means "not supplied",
    # and `check_scalars` refuses a positive `random_strength` beside it
    # rather than training a model whose regularizer silently scaled to
    # nothing.
    var random_strength: Float64
    var random_score_scale: Float64
    var random_strength_seed: Int

    # LightGBM's quantized-training family, verbatim: the four names and the
    # four defaults of `include/LightGBM/config.h` (LightGBM 4.7.0.99), no
    # more and no fewer. The scale rule, the rounding seed, and the
    # accumulator width are NOT here, because LightGBM has no parameters for
    # them and this surface is exactly LightGBM's; they live on
    # `quantized_gradient.QuantGradParams` and are set from
    # `MOJOTREES_*` environment overrides. `quantized_gradient.cpu_quant_params`
    # is the one function that turns these four into that bundle, and it is
    # in that module rather than here because this module must not import it
    # (it imports `raw_leaf_output` from here, so the edge only goes one way).
    var use_quantized_grad: Bool
    var num_grad_quant_bins: Int
    var quant_train_renew_leaf: Bool
    var stochastic_rounding: Bool

    # CatBoost's `leaf_estimation_iterations`, the second name on this bundle
    # that is CatBoost's rather than LightGBM's. **Default 1, and 1 is
    # LightGBM's and this project's behavior**: one Newton step per leaf, taken
    # at the raw scores the tree was grown from, which is exactly what the
    # grower already wrote into `Tree.value`. Above 1 the trainer re-evaluates
    # the leaf's rows at the value the leaf currently holds and takes another
    # step (`boosting._estimate_leaf_values`); iteration 1 is never recomputed,
    # so 1 and "absent" are the same code path and the same bits.
    var leaf_estimation_iterations: Int
    # `derivative_precision`, the precision per-row gradients and hessians
    # are carried at. `float32` is the default and is LightGBM's profile;
    # `float64` is the opt-in arm of a measured trade. Not a LightGBM
    # parameter name -- LightGBM makes this choice at compile time with
    # `SCORE_T_USE_DOUBLE` -- and it is here rather than as an environment
    # override only because it changes a fit's numbers, which is the line
    # this package draws between a parameter and a `MOJOTREES_*` knob.
    var derivative_precision: Int

    def __init__(out self):
        self.min_gain_to_split = 0.0
        self.max_delta_step = 0.0
        self.path_smooth = 0.0
        self.extra_trees = False
        self.extra_seed = DEFAULT_EXTRA_SEED
        self.monotone_penalty = 0.0
        self.monotone_method = MONOTONE_BASIC
        self.penalties = FeaturePenalties()
        self.forced = ForcedSplits.none()
        self.random_strength = 0.0
        self.random_score_scale = 0.0
        self.random_strength_seed = DEFAULT_RANDOM_STRENGTH_SEED
        self.use_quantized_grad = False
        self.num_grad_quant_bins = DEFAULT_NUM_GRAD_QUANT_BINS
        self.quant_train_renew_leaf = False
        self.stochastic_rounding = True
        self.leaf_estimation_iterations = DEFAULT_LEAF_ESTIMATION_ITERATIONS
        self.derivative_precision = DERIV_PRECISION_FLOAT32

    @staticmethod
    def default() -> ExtraTreeParams:
        return ExtraTreeParams()

    def is_active(self) -> Bool:
        """Whether anything here would change a **split search**. False for the
        defaults, so a grower can test this once per tree and take its existing
        path.

        `leaf_estimation_iterations` is deliberately **not** in this test, and
        that is the one exclusion. Every consumer of this predicate is a split
        search or a search-eligibility gate -- `split.find_best_split` gates its
        per-feature cost pass on it, `train_gpu._check_device_search_supported`
        refuses the device scan on it, `gpu_tree_tables` refuses the resident
        tree on it -- and extra Newton steps happen *after* the structure is
        fixed, from the trainer, touching no candidate and no gain. Folding it
        in here would send a `leaf_estimation_iterations > 1` fit down
        `split._feature_gain`'s active path, where `passes_min_gain` rejects a
        gain of exactly 0.0 that the inactive path lets through: a split
        decision moved for a reason that has nothing to do with leaf values.
        `leaf_estimation_active` below is the test for this one, and
        `boosting._check_leaf_estimation_supported` is what keeps a trainer
        that does not implement it from ignoring it.
        """
        return (
            self.min_gain_to_split > 0.0
            or self.max_delta_step > 0.0
            or self.path_smooth > 0.0
            or self.extra_trees
            or self.monotone_penalty > 0.0
            or self.penalties.is_active()
            or not self.forced.is_empty()
            or self.random_strength > 0.0
            or self.use_quantized_grad
            or self.derivative_precision != DERIV_PRECISION_FLOAT32
        )

    def random_score_stdev(self) -> Float64:
        """CatBoost's `scoreStDev`: the standard deviation of the noise added
        to one candidate's gain. 0.0 at the default, where no draw is taken.
        """
        return self.random_strength * self.random_score_scale

    def leaf_estimation_active(self) -> Bool:
        """Whether any leaf value is the result of more than one Newton step.

        False at the default of 1, where `boosting._estimate_leaf_values`
        returns before it reads a row and the leaf keeps exactly the value the
        grower wrote. That early return is the whole of the "1 moves nothing"
        guarantee: iteration 1 is never recomputed, so there is no second route
        to the first value that could round differently from the histogram's.
        """
        return self.leaf_estimation_iterations > 1

    def needs_leaf_finish(self) -> Bool:
        """Whether `finish_leaf_output` would move a leaf's value.

        This is the half of the bundle a *grower* has to honor rather than a
        split search: it needs the leaf's row count and its parent's finished
        output, neither of which a histogram carries. `tree._search` refuses
        an active value from a grower that has not opted in, so a backend that
        does not apply it reports that instead of quietly emitting unsmoothed
        leaves.
        """
        return self.max_delta_step > 0.0 or self.path_smooth > 0.0

    def needs_node_identity(self) -> Bool:
        """Whether an active rule reads the node id and the tree index.

        Two rules do. `extra_trees` keys its threshold draw by
        (seed, tree index, node id, feature), and `random_strength` keys its
        per-candidate noise by (seed, tree index, node id, feature, bin). A
        grower that does not pass its node ids would draw every node from the
        same stream, so `tree._search` refuses both rather than let a
        caller's default 0 stand in for a node id.
        """
        return self.extra_trees or self.random_strength > 0.0

    def needs_grower_support(self) -> Bool:
        """Whether this bundle can only be honored by a grower that opts in.

        The complement -- `min_gain_to_split`, `monotone_penalty`,
        `feature_contri`, and the per-split CEGB cost -- is a function of the
        histogram, the node's row count, and the node's depth, all of which
        every caller of `tree._search` already passes, so those are live on
        every backend that routes through it.
        """
        return self.needs_leaf_finish() or self.needs_node_identity()

    def check_scalars(self, min_data_in_leaf: Int) raises:
        """The half of `check` that needs neither the feature count nor the
        growth budget, so a parameter string can be rejected before any data
        is read (see `params._validate`).

        This is also where an unmapped forced-split document is refused: it
        is a real LightGBM feature whose missing piece is outside a split
        search, since a raw threshold needs the bin mapper to become a bin.
        Refusing it here is the repository's rule -- say so, never ignore it
        -- and keeps a caller from reading an unforced tree as a forced one.
        The CEGB costs are not refused here: whether the coupled and lazy
        penalties can be honored is a property of the grower, so
        `cegb.check_cegb_grower_support` refuses them at `tree._search`, where
        the answer is known.
        """
        if not _is_finite(self.min_gain_to_split) or (
            self.min_gain_to_split < 0.0
        ):
            raise Error(
                "min_gain_to_split must be a finite nonnegative number"
            )
        if not _is_finite(self.max_delta_step) or self.max_delta_step < 0.0:
            raise Error("max_delta_step must be a finite nonnegative number")
        if not _is_finite(self.path_smooth) or self.path_smooth < 0.0:
            raise Error("path_smooth must be a finite nonnegative number")
        if not _is_finite(self.monotone_penalty) or (
            self.monotone_penalty < 0.0
        ):
            raise Error("monotone_penalty must be a finite nonnegative number")
        self.check_random_strength()
        self.check_quantized_grad()
        self.check_leaf_estimation()
        self.check_derivative_precision()
        if self.monotone_method != MONOTONE_BASIC:
            raise Error(
                "monotone_constraints_method '",
                monotone_method_name(self.monotone_method),
                "' is not implemented; mojotrees implements 'basic'",
            )
        # LightGBM raises min_data_in_leaf to 2 with a warning when path
        # smoothing is on, because smoothing a one-row leaf toward its parent
        # is the parent. Rejecting says the same thing without changing a
        # number the caller set.
        if self.path_smooth > 0.0 and min_data_in_leaf < 2:
            raise Error(
                "path_smooth needs min_data_in_leaf of at least 2, got ",
                min_data_in_leaf,
            )
        # A parsed document is not yet applicable: the grower is handed a
        # BinnedMatrix, which carries no bin edges, so a raw threshold has to
        # be mapped through the fitted BinMapper first. Refusing here is what
        # keeps a caller from reading an unforced tree as a forced one.
        if not self.forced.is_empty() and not self.forced.is_mapped():
            raise Error(
                "forced splits carry raw feature thresholds and the grower is"
                " handed a BinnedMatrix, which has no bin edges. Map them"
                " once with binning.map_forced_splits(mapper, forced) and put"
                " the result on ExtraTreeParams.forced"
            )

    def check_random_strength(self) raises:
        """Range-check `random_strength`, and refuse a positive value whose
        per-tree scale nobody supplied.

        Runs whether or not `random_strength` is set, so a nonsense value is
        reported when it is set rather than when it is first used, which is
        the rule `check_quantized_grad` states and this mirrors.

        The refusal is the same shape as that one and for the same reason.
        CatBoost's noise is `random_strength * derivativesStDevFromZero *
        modelSizeDecrease`, and the last two factors are properties of the
        ensemble at this iteration: the RMS of the current gradients over the
        whole learn set, and `iteration * learning_rate`. A split search sees
        a node histogram, which carries sums of gradients and never sums of
        squared gradients, so it cannot compute them and must be handed the
        product. Nothing in this build hands it over yet -- `boosting.fit`
        and the metric-path trainers are untouched -- so a `random_strength`
        set from a parameter string would scale to zero and train an
        unregularized model that reported success. Accepting it and doing
        nothing is the silent downgrade this package refuses everywhere else.

        A Mojo-API caller that computes the scale itself with
        `random_score_scale_from_gradients(grad, n_rows, iteration *
        learning_rate)` and puts it on `random_score_scale` passes this
        check, and the draw is live for `tree.grow_tree` from there.
        """
        if not _is_finite(self.random_strength) or self.random_strength < 0.0:
            raise Error(
                "random_strength must be a finite nonnegative number"
            )
        if not _is_finite(self.random_score_scale) or (
            self.random_score_scale < 0.0
        ):
            raise Error(
                "random_score_scale must be a finite nonnegative number"
            )
        if self.random_strength > 0.0 and self.random_score_scale <= 0.0:
            raise Error(
                "random_strength is recognized but no trainer computes its"
                " per-tree scale in this build, so setting it alone would"
                " train an unregularized model that ignored it. The scale is"
                " CatBoost's derivativesStDevFromZero * modelSizeDecrease;"
                " compute it with"
                " tree_parameters_extra.random_score_scale_from_gradients("
                "grad, n_rows, iteration * learning_rate) and set it on"
                " ExtraTreeParams.random_score_scale, which makes the draw"
                " live for tree.grow_tree"
            )

    def check_leaf_estimation(self) raises:
        """Range-check `leaf_estimation_iterations`, and refuse the one
        combination whose repetition is not idempotent.

        Runs whether or not the value is above 1, so a nonsense count is
        reported where it was set rather than where it is first used, which is
        the rule the two checks above state and this mirrors.

        **`path_smooth` is refused beside it.** The tree stores the *finished*
        leaf value -- capped, smoothed, and clamped -- and an extra iteration
        has to start from the value the leaf actually holds, because that is
        the value whose derivatives it is about to evaluate. The cap and the
        monotone clamp survive that: both are projections onto a fixed set, so
        applying either to an already-projected value is the identity, and
        re-applying them per iteration keeps every intermediate leaf value
        inside the cap and inside its monotone interval. Path smoothing is not
        a projection. It is the affine contraction
        `v -> v*w/(w+1) + parent/(w+1)` with `w/(w+1) < 1`, so applying it once
        per iteration drives the leaf geometrically toward its parent's output
        and `leaf_estimation_iterations=10` would return a leaf that is mostly
        its parent, for reasons that have nothing to do with the loss. There is
        no correct place to put it either: smoothing only the last iteration
        would evaluate every earlier derivative at a value the model will never
        hold. So the combination is reported instead of picking one of two
        wrong answers.
        """
        if self.leaf_estimation_iterations < 1:
            raise Error(
                "leaf_estimation_iterations must be at least 1 (1 is the"
                " default and is one Newton step per leaf, which is"
                " LightGBM's behavior), got ",
                self.leaf_estimation_iterations,
            )
        if self.leaf_estimation_active() and self.path_smooth > 0.0:
            raise Error(
                "leaf_estimation_iterations > 1 and path_smooth > 0 cannot"
                " both be set. A leaf carries its finished value, so each"
                " extra iteration would smooth an already smoothed value and"
                " the leaf would converge to its parent's output rather than"
                " to the loss minimizer. max_delta_step and the monotone"
                " clamp do compose, because both are projections and applying"
                " one twice is applying it once"
            )

    def check_quantized_grad(self) raises:
        """Range-check LightGBM's quantized-training family, and refuse an
        enabled one by name rather than ignoring it.

        Runs whether or not `use_quantized_grad` is set, so a nonsense bin
        count is reported when it is set and not when it is first used, which
        is the rule `QuantGradParams.validate` states and this mirrors.

        `num_grad_quant_bins` must be even. LightGBM computes
        `num_grad_quant_bins_ / 2` in integer arithmetic
        (`gradient_discretizer.cpp`), so an odd count silently truncates and
        the positive and negative halves of the gradient lattice stop
        matching. That is the difference mojotrees takes: an asymmetric
        gradient lattice is a bug in every reported case, not a
        configuration.

        **`use_quantized_grad=true` is refused, with a sentence.** The CPU
        integer histogram exists (`quantized_gradient.
        build_histogram_subset_quantized_into_scratch`) but no trainer calls
        it: `boosting.mojo` and `tree.mojo` are untouched, and
        `quantized_gradient.CONNECTED` is still False. Accepting the key and
        training a float model would be exactly the silent-downgrade failure
        the package refuses everywhere else, so this says so where the
        parameter was set. The wiring that lifts the refusal is the ordered
        patch set in `handoffs/remaining_06_quantized_gradients.md`.
        """
        if self.num_grad_quant_bins < 2 or self.num_grad_quant_bins > 1048576:
            raise Error(
                "num_grad_quant_bins must be between 2 and 1048576, got ",
                self.num_grad_quant_bins,
            )
        if self.num_grad_quant_bins % 2 != 0:
            raise Error(
                "num_grad_quant_bins must be even so the gradient lattice is"
                " symmetric about zero, got ",
                self.num_grad_quant_bins,
            )
        if self.use_quantized_grad:
            raise Error(
                "use_quantized_grad is recognized but no trainer is wired to"
                " the quantized histogram in this build, so setting it would"
                " train a float model that silently ignored it. The CPU"
                " integer accumulation exists and is reachable directly"
                " through"
                " quantized_gradient.build_histogram_subset_quantized_into_scratch"
            )

    def check_derivative_precision(self) raises:
        """Range-check `derivative_precision`, and refuse `float64` by name
        rather than ignoring it.

        Runs whether or not it is set, so a nonsense code is reported when it
        is set rather than when it is first used, which is the rule
        `check_quantized_grad` states and this mirrors.

        **`float64` is refused here, and the refusal is about wiring rather
        than about the setting.** The switch itself is built and live: every
        read site in `histogram.mojo`, `histogram_sparse.mojo` and the
        row-major kernels honors it, and so does the objective in
        `boosting.fill_grad_hess`. What is missing is the one hop from this
        field to the fit -- `tree.mojo` would have to put
        `params.extra.derivative_precision` on the `ConstHessianSettings` it
        resolves, and `tree.mojo` is not this lane's file. Until that lands,
        a value set here would train a Float32 model that reported success,
        which is the silent downgrade this package refuses everywhere else.

        The working entry is the environment: `MOJOTREES_DERIVATIVE_PRECISION
        =float64` is read once per fit by `ConstHessianSettings.resolve()`
        and once per round by `boosting.fill_grad_hess`, and reaches every
        site the parameter would.
        """
        if (
            self.derivative_precision != DERIV_PRECISION_FLOAT32
            and self.derivative_precision != DERIV_PRECISION_FLOAT64
        ):
            raise Error(
                "derivative_precision must be float32 or float64, got code ",
                self.derivative_precision,
            )
        if self.derivative_precision != DERIV_PRECISION_FLOAT32:
            raise Error(
                "derivative_precision='",
                derivative_precision_name(self.derivative_precision),
                "' is recognized and the histogram and objective paths honor"
                " it, but no trainer carries it from the parameters onto the"
                " per-fit settings snapshot in this build, so setting it here"
                " would train a float32 model that silently ignored it. Set"
                " MOJOTREES_DERIVATIVE_PRECISION=float64 instead, which"
                " reaches every site; the wiring that lifts this refusal is"
                " one field on the ConstHessianSettings that tree.mojo"
                " resolves",
            )

    def check(
        self, n_features: Int, num_leaves: Int, max_depth: Int,
        min_data_in_leaf: Int,
    ) raises:
        """Range checks that need no training data, in the style of
        `params._validate`: values are rejected rather than clamped. The
        grower calls this once per tree, so the vector lengths are checked
        against the dataset actually being fitted."""
        self.check_scalars(min_data_in_leaf)
        self.penalties.check_features(n_features)
        self.forced.check_features(n_features)
        self.forced.check_budget(num_leaves, max_depth)
