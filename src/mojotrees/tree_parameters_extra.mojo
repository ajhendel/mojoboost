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
- Parsed, validated, and refused: `cegb_penalty_feature_coupled`, plus
  `linear_tree`/`linear_lambda`, `cegb_penalty_feature_lazy`, and
  `feature_pre_filter`. Each names what it would take; see
  `check_extra_option_supported`.

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
- `feature_contri` (empty): per-feature multipliers on split gain, plus the
  two computable CEGB split penalties (`cegb_tradeoff` 1.0,
  `cegb_penalty_split` 0.0, `cegb_penalty_feature_coupled` empty).
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

Randomness stays counter-based
------------------------------
`extra_trees` draws from a counter-based splitmix64 stream keyed by
(seed, tree index, node id, feature), the same construction `sampling.mojo`
uses for feature subsampling and for the same reason: a draw must not depend
on how many draws happened before it, so a tree is reproducible whatever the
thread count or the training history. This costs LightGBM-identical streams
for a given seed, which `sampling.mojo` already documents as an intentional
difference.

Linear trees are a deferred subsystem, not a missing parameter
--------------------------------------------------------------
`linear_tree` and `linear_lambda` are deliberately absent, and no flag or
placeholder for them appears here. A linear tree does not add a control to
the grower; it changes what a leaf *is*. Each leaf would hold a ridge
regression over the numerical features on its branch rather than a constant,
which needs: raw (unbinned) feature values kept alongside the binned matrix,
a per-leaf normal-equation solve during growth, coefficient storage in
`Tree` and in the serialized format (a version bump), a different prediction
path in every predictor including the GPU one, and its own answers for
missing values, categorical features, and TreeSHAP. That is a subsystem with
its own task, not a parameter that could be honestly stubbed. Until it
exists, `check_extra_option_supported` rejects both names by name, which is
the repository's rule for a real LightGBM feature that is not implemented:
say so, do not ignore it.
"""

from .gain import leaf_score, soft_threshold_l1
from .monotone import MONOTONE_FREE, output_score
from .rng import splitmix64, uniform

# LightGBM's extra_seed default. The other seeds already live with the
# features that use them (`sampling.DEFAULT_FEATURE_FRACTION_SEED`,
# `bagging.DEFAULT_BAGGING_SEED`).
comptime DEFAULT_EXTRA_SEED = 6

# monotone_constraints_method codes. Only `basic` is implemented; the other
# two are named so they can be rejected as the real features they are.
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

    Two LightGBM mechanisms, kept in one struct because both are applied at
    the same point (once a feature's best candidate is known) and both are
    inactive by default:

    - `feature_contri`: a multiplier per feature. A feature with multiplier
      0.5 must find twice the gain to compete. Empty means every multiplier
      is 1.0.
    - Cost-effective gradient boosting: `cegb_tradeoff * cegb_penalty_split *
      rows_in_leaf` is charged to every split, and
      `cegb_tradeoff * cegb_penalty_feature_coupled[f]` is charged the first
      time feature `f` is split on at all.

    INTENTIONAL DIFFERENCES FROM LightGBM

    - Multipliers must be nonnegative. LightGBM accepts a negative
      `feature_contri`, which flips the sign of a gain and breaks the
      invariant that a chosen split has positive gain. Zero is accepted and
      means the feature can never win, which is the useful end of that range.
    - `cegb_penalty_feature_lazy` is not represented. It charges for the rows
      that have not yet read a feature, which is per-row state carried across
      the whole ensemble, not a function of (feature, leaf size). It is a
      subsystem, and `check_extra_option_supported` rejects it by name.
    """

    var contri: List[Float64]
    var cegb_tradeoff: Float64
    var cegb_penalty_split: Float64
    var cegb_penalty_feature_coupled: List[Float64]

    def __init__(out self):
        """No penalties: every multiplier 1.0, every cost 0.0."""
        self.contri = List[Float64]()
        self.cegb_tradeoff = 1.0
        self.cegb_penalty_split = 0.0
        self.cegb_penalty_feature_coupled = List[Float64]()

    @staticmethod
    def from_contri(contri: List[Float64]) raises -> FeaturePenalties:
        """Gain multipliers only, LightGBM's `feature_contri`."""
        var out = FeaturePenalties()
        out.contri = contri.copy()
        return out^

    @staticmethod
    def cegb(
        tradeoff: Float64,
        penalty_split: Float64,
        penalty_feature_coupled: List[Float64] = [],
    ) raises -> FeaturePenalties:
        """The two computable CEGB penalties, with no gain multipliers."""
        var out = FeaturePenalties()
        out.cegb_tradeoff = tradeoff
        out.cegb_penalty_split = penalty_split
        out.cegb_penalty_feature_coupled = penalty_feature_coupled.copy()
        return out^

    def is_active(self) -> Bool:
        """Whether anything here would change a gain. An inactive bundle must
        leave the scan bit-identical, so the growers test this once per node
        rather than multiplying by 1.0 per candidate."""
        if self.cegb_penalty_split > 0.0 and self.cegb_tradeoff != 0.0:
            return True
        if self.cegb_tradeoff != 0.0:
            for f in range(len(self.cegb_penalty_feature_coupled)):
                if self.cegb_penalty_feature_coupled[f] != 0.0:
                    return True
        for f in range(len(self.contri)):
            if self.contri[f] != 1.0:
                return True
        return False

    def contri_of(self, feature: Int) -> Float64:
        """This feature's gain multiplier, 1.0 when unset."""
        if feature < 0 or feature >= len(self.contri):
            return 1.0
        return self.contri[feature]

    def coupled_of(self, feature: Int) -> Float64:
        """This feature's first-use cost, 0.0 when unset."""
        if feature < 0 or feature >= len(self.cegb_penalty_feature_coupled):
            return 0.0
        return self.cegb_penalty_feature_coupled[feature]

    def coupled_is_active(self) -> Bool:
        """Whether any first-use cost would actually be charged.

        The coupled ledger is the one CEGB half the grower cannot carry on its
        own: "first use" is per-model, so it is state every trainer would have
        to thread through every tree. `ExtraTreeParams.check_scalars` refuses
        an active coupled vector for that reason, and this is the predicate it
        refuses on — the vector's length alone is not enough, since an
        all-zero vector charges nothing.
        """
        if self.cegb_tradeoff == 0.0:
            return False
        for f in range(len(self.cegb_penalty_feature_coupled)):
            if self.cegb_penalty_feature_coupled[f] != 0.0:
                return True
        return False

    def split_costs_active(self) -> Bool:
        """Whether the per-split CEGB cost would change a gain. Unlike the
        coupled cost this one is a function of (tradeoff, penalty, rows in the
        leaf) alone, so the split search charges it with no ledger."""
        return self.cegb_tradeoff != 0.0 and self.cegb_penalty_split != 0.0

    def penalized_gain(
        self,
        gain: Float64,
        feature: Int,
        n_data_in_leaf: Int,
        feature_already_used: Bool,
    ) -> Float64:
        """A candidate's gain after this feature's costs.

        The multiplier scales first and the costs are then subtracted, so a
        cost is an absolute amount of gain rather than something the
        multiplier rescales. The result may be negative; the caller's gain
        floor rejects it.
        """
        var out = gain * self.contri_of(feature)
        if self.cegb_tradeoff != 0.0:
            if self.cegb_penalty_split != 0.0:
                out -= (
                    self.cegb_tradeoff
                    * self.cegb_penalty_split
                    * Float64(n_data_in_leaf)
                )
            if not feature_already_used:
                out -= self.cegb_tradeoff * self.coupled_of(feature)
        return out

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
        var coupled = len(self.cegb_penalty_feature_coupled)
        if coupled > 0 and coupled != n_features:
            raise Error(
                "cegb_penalty_feature_coupled has ",
                coupled,
                " entries but the data has ",
                n_features,
                " features",
            )
        for f in range(coupled):
            var v = self.cegb_penalty_feature_coupled[f]
            if not _is_finite(v) or v < 0.0:
                raise Error(
                    "cegb_penalty_feature_coupled[",
                    f,
                    "] must be a finite nonnegative number",
                )
        if not _is_finite(self.cegb_tradeoff) or self.cegb_tradeoff < 0.0:
            raise Error("cegb_tradeoff must be a finite nonnegative number")
        if (
            not _is_finite(self.cegb_penalty_split)
            or self.cegb_penalty_split < 0.0
        ):
            raise Error(
                "cegb_penalty_split must be a finite nonnegative number"
            )


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
    if name == "linear_tree":
        raise Error(
            "'linear_tree' is not implemented; a linear tree replaces each"
            " leaf's constant with a ridge regression over its branch's"
            " numerical features, which needs raw feature values during"
            " growth, per-leaf coefficients in the model and in the"
            " serialized format, and a different prediction path on every"
            " backend. It is a deferred subsystem, not a tree parameter"
        )
    if name == "linear_lambda":
        raise Error(
            "'linear_lambda' is not implemented; it regularizes the per-leaf"
            " regression of 'linear_tree', which is a deferred subsystem"
        )
    if name == "cegb_penalty_feature_lazy":
        raise Error(
            "'cegb_penalty_feature_lazy' is not implemented; it charges for"
            " the rows that have not yet read a feature, which is per-row"
            " state carried across the whole ensemble. 'cegb_penalty_split'"
            " and 'cegb_tradeoff' are implemented"
        )
    if name == "cegb_penalty_feature_coupled":
        raise Error(
            "'cegb_penalty_feature_coupled' is parsed but not applied; it is"
            " charged the first time a feature is split on anywhere in the"
            " ensemble, so it needs a per-model feature-use ledger threaded"
            " through every trainer and every grower. 'cegb_penalty_split'"
            " and 'cegb_tradeoff' are applied by the split search today"
        )
    if name == "feature_pre_filter":
        raise Error(
            "'feature_pre_filter' is not implemented. It is a Dataset"
            " construction step: LightGBM drops, before training, the"
            " features whose every bin boundary would leave a child under"
            " 'min_data_in_leaf'. mojotrees's split search rejects those"
            " candidates as it scans (split.find_best_split), so the fitted"
            " model is already the one LightGBM produces with"
            " feature_pre_filter=false; what is missing is the speed-up, and"
            " with it LightGBM's rule that a prefiltered Dataset cannot be"
            " reused with a larger min_data_in_leaf"
        )
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

    @staticmethod
    def default() -> ExtraTreeParams:
        return ExtraTreeParams()

    def is_active(self) -> Bool:
        """Whether anything here would change a fit. False for the defaults,
        so a grower can test this once per tree and take its existing path."""
        return (
            self.min_gain_to_split > 0.0
            or self.max_delta_step > 0.0
            or self.path_smooth > 0.0
            or self.extra_trees
            or self.monotone_penalty > 0.0
            or self.penalties.is_active()
            or not self.forced.is_empty()
        )

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

        Only `extra_trees` does: its threshold draw is keyed by
        (seed, tree index, node id, feature). A grower that does not pass its
        node ids would draw every node's threshold from the same stream, so
        `tree._search` refuses this too rather than let a caller's default 0
        stand in for a node id.
        """
        return self.extra_trees

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

        This is also where the two options that are parsed but not applied are
        refused. Both are real LightGBM features whose missing piece is
        outside a split search: the coupled CEGB cost needs a per-model
        feature-use ledger, and a forced-split document needs the bin mapper
        to turn a raw threshold into a bin. Refusing them here is the
        repository's rule -- say so, never ignore it -- and keeps a caller
        from reading an unforced tree as a forced one.
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
        if self.penalties.coupled_is_active():
            check_extra_option_supported("cegb_penalty_feature_coupled")
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
