"""Best-split search over histograms.

Uses the standard second-order gain formula (XGBoost/LightGBM), with L1
regularization applied by soft-thresholding each gradient sum:

    T(G) = sign(G) * max(0, |G| - lambda_l1)
    gain = T(GL)^2 / (HL + lambda_l2)
         + T(GR)^2 / (HR + lambda_l2)
         - T(G)^2  / (H + lambda_l2)

With lambda_l1 = 0 (the default) T is the identity and this is the plain
second-order formula. A numerical split at (feature, bin) sends rows with bin
value <= bin to the left child.

Categorical features
--------------------
Features marked categorical in the binned matrix are never scanned by ordinal
threshold: their candidates are *sets* of categories, searched by
`categorical.mojo` and scored with the same gain formula, so numerical and
categorical candidates compete on gain in this one loop. A categorical
feature's missing, unseen, and dropped rows live in its bin 0, which is never
a member of a candidate set and therefore always routes right; the numerical
missing-bin machinery below does not apply to it.

The search takes an optional per-feature allow mask. Feature interaction
constraints are enforced here and nowhere else: a feature the caller has
masked off is skipped before any of its candidates are scored, so a
disallowed split can never win no matter how large its gain would have been
(see interaction.mojo).

Missing values
--------------
A numerical feature with a reserved missing bin (see binning.mojo) is scanned
over its ordinary bins only, and every threshold is evaluated twice: once
with the missing rows added to the left child, once with them added to the
right. The winner's direction comes back as `default_left`, which the tree
stores per node and applies to missing rows during training, prediction, and
GPU row partitioning. The top threshold, "every ordinary bin left, missing
alone right", is a candidate like any other; it is the split LightGBM reports
as `threshold = 1e300`.

Exact ties keep `default_left = True`, matching LightGBM, whose reverse scan
runs first. A node holding no missing rows therefore records the left
default, which is the direction an unseen missing value follows at prediction
time. A feature with no reserved missing bin is scanned exactly as before and
reports `default_left = False`; prediction never consults it, because no bin
id can equal that feature's missing bin of -1.

The remaining tree controls
---------------------------
`ExtraTreeParams` (tree_parameters_extra.mojo) rides in as one argument and
defaults to inactive, in which case nothing below changes. Active, it adds a
gain floor (`min_gain_to_split`), per-feature gain multipliers and the
computable CEGB split cost, a discount on constrained splits
(`monotone_penalty`), a single drawn threshold per feature (`extra_trees`),
and candidate scoring at capped and smoothed child outputs (`max_delta_step`,
`path_smooth`). The costs and the floor are applied once per feature, after
that feature's best candidate is known, which is LightGBM's placement.

Monotonic constraints
---------------------
The search also takes the node's constraint vector and output bounds. With an
active vector, every candidate is scored from child outputs clamped into the
node's bounds, and a candidate whose outputs run against its feature's
constraint is rejected whatever its gain. The bounds apply to candidates on
every feature, constrained or not, because a node under a constrained split
owes its interval no matter which feature it goes on to split. An empty
vector keeps the original unconstrained scoring path (see monotone.mojo).

Scanning features in parallel
-----------------------------
Features are independent: one feature's candidates are scored entirely from
its own histogram slice, and nothing a feature computes is read by another.
The scan is therefore dispatched across features under the usual contract in
`parallel.mojo`, and each feature's own scan runs start to finish inside one
task. That placement is not incidental. Every floating-point sum in this file
(a feature's bin totals, the running prefix sums that make a threshold's left
child) is accumulated in ascending bin order within a single feature, and no
sum crosses a task boundary, so no addition is reassociated and no gain moves
by an ulp at any task count or on any machine.

What does cross the task boundary is the choice among features, and that is a
maximum rather than a sum. It is still not reduced in parallel here. Each
feature writes its own best candidate into its own slot, and the slots are
folded afterwards in ascending scan order by the same strict `>` comparison
the serial loop used. The fold costs one compare per feature against a scan
that costs bins times features, so nothing is bought by making it clever, and
writing it as the serial fold means the tie-break is the serial tie-break by
construction rather than by reconstruction: among candidates of exactly equal
gain the first in scan order wins, which is the lowest feature id when the
caller's feature list is ascending (`find_best_split` requires that it be). A
parallel max carrying the feature index and resolving ties toward the lower
index would answer the same, and would be a second rule to keep in step with
the first.

Noise on a candidate's gain
---------------------------
`random_strength` (CatBoost's, see tree_parameters_extra.mojo) adds a seeded
normal to each candidate's gain. Its placement is chosen so that none of the
paragraph above changes. The draw belongs to a (feature, bin) candidate and
is keyed by (seed, tree index, node id, feature, bin) and by nothing else, so
it is decided inside the feature's own task, before that feature's best is
written to its slot, and the fold still sees one number per feature and folds
them in ascending order under the same strict `>`. Nothing about the noise
reads a counter that advances with evaluation order, so the same tree comes
out at every task count. At the default of 0 no draw is taken, no arithmetic
is added, and the scan takes exactly the path it took before the parameter
existed.

The two candidates at one threshold -- missing rows left and missing rows
right -- share a single draw, because the noise belongs to the threshold and
not to the routing direction. Sharing it keeps the LightGBM rule above
intact: an exact tie between the two directions still keeps `default_left`,
because equal gains stay equal after the same number is added to both.

Which functional is maximized
-----------------------------
`score_function` selects it, and defaults to `SCORE_L2`, which is the
second-order gain spelled at the top of this docstring and is what every
caller in the package gets. `SCORE_COSINE` is CatBoost's CPU default
(`docs/design/CATBOOST_CATALOG.md` A10, verified from source): a ratio,
`sum(-out * G) / sqrt(sum(out^2 * H))` over the children, minus the same
functional of the unsplit node.

**At `lambda_l2 = 0` the two have the same argmax, provably.** Substituting
the free Newton step makes the numerator and the denominator the same
expression, so Cosine collapses to `sqrt` of the L2 score and `sqrt` is
strictly increasing. mojotrees stock is `lambda_l2 = 0`. Cosine is therefore
a difference only under a positive `lambda_l2`, under a leaf-wise queue that
compares gains from different parents, or beside one of the absolute costs
(`min_gain_to_split`, `feature_contri`, CEGB, `random_strength`) whose units
it has changed. That is a result, not a reason to leave it out; see A10.

Nothing on the default path is conditional on it. The selector is read once
per node into a `Bool`, the branch that reads it is loop-invariant for the
whole scan, and `_split_gain` is not touched. The two extra accumulator
planes the shared search needs are allocated with length 0 when it is off.
"""

from std.math import sqrt

from .apple_cpu_policy import split_scan_ops
from .parallel import DispatchSettings, dispatch_features_with
from .categorical import (
    CAT_BITSET_WORDS,
    CatBitset,
    CategoricalParams,
    CategoricalSpec,
    cat_contains,
    cat_empty,
    find_best_categorical_split,
)

# Re-exported: the L1 soft-threshold lives in gain.mojo so the numerical scan
# here and the category partition search in categorical.mojo share one
# definition, but `split.soft_threshold_l1` remains its public name.
from .gain import soft_threshold_l1
from .growth_policy import SharedSplitAudit
from .histogram import Histogram, SIMD_LANES
from .monotone import (
    MONOTONE_FREE,
    OutputBounds,
    monotone_sign,
    output_score,
    violates,
)
from .cegb import (
    CegbLedger,
    CegbNodeCosts,
    check_cegb_grower_support,
    prepare_cegb_node,
)
from .tree_parameters_extra import (
    ExtraTreeParams,
    apply_monotone_penalty,
    extra_threshold_index,
    finish_leaf_output,
    passes_min_gain,
    random_score_noise,
)


# How a scanned feature's slot records the two booleans a `SplitInfo` needs,
# so that a task writes only scalars through unsafe pointers. Absence of both
# is a numerical split routing missing rows right, which is also what a
# feature that produced nothing leaves behind; the fold never reads the flags
# of a feature whose gain did not win, so the two cases never have to be
# told apart.
comptime _FLAG_DEFAULT_LEFT = 1
comptime _FLAG_CATEGORICAL = 2


# --------------------------------------------------------------------------
# score_function: which functional of the children's sums is maximized
# --------------------------------------------------------------------------
#
# `SCORE_L2` is the default and is what every line below this block already
# did: LightGBM's second-order gain, `_split_gain`. `SCORE_COSINE` is
# CatBoost's CPU default (`catboost/private/libs/options/
# oblivious_tree_options.cpp`, `ScoreFunction("score_function",
# EScoreFunction::Cosine)`), and it is a RATIO rather than a sum. See
# `docs/design/CATBOOST_CATALOG.md` A10 for the full source reading; the
# short form is in `_cosine_pair` below.
#
# Nothing here is reachable from `TreeParams` or `ExtraTreeParams` yet: this
# lane owns `split.mojo` only, and the parameter surface lives in files it
# does not own. The two search entry points take the selector directly and
# default it to `SCORE_L2`.

comptime SCORE_L2 = 0
"""LightGBM's second-order gain, `_split_gain`. The default. Unchanged."""

comptime SCORE_COSINE = 1
"""CatBoost `score_function=Cosine`: `sum(-out * G) / sqrt(sum(out^2 * H))`
over the children, minus the same functional of the unsplit node."""


# CatBoost seeds the denominator accumulator at 1e-100 rather than 0
# (`score_calcers.h`, `Scores.resize(splitsCount, {0, 1e-100})`). It is a
# divide-by-zero guard and not a regularizer: `den -> 0` forces `num -> 0`
# through the same zero-weight guard in `_cosine_out`, so the ratio is 0/0
# without it. `min_child_hess` is what actually keeps H off the floor.
comptime _COSINE_DEN_FLOOR = 1e-100


def check_score_function(score_function: Int) raises:
    """Refuse an unknown selector by name rather than falling through to the
    default, which would silently give an L2 answer to a caller who asked for
    something else."""
    if score_function != SCORE_L2 and score_function != SCORE_COSINE:
        raise Error(
            "score_function must be split.SCORE_L2 (0) or split.SCORE_COSINE"
            " (1); got ",
            score_function,
        )


@fieldwise_init
struct _CosineTerms(Copyable, Movable):
    """One candidate's contribution to the two cross-child accumulators
    CatBoost's `TCosineScoreCalcer` keeps, plus whether the candidate is
    admissible at all.

    `ok` is False only for a candidate rejected by an active monotonic
    constraint, which is the one rejection `_split_gain` expresses by
    returning 0.0. A ratio cannot express a rejection that way, because 0.0 is
    a legitimate numerator, so it is carried out of band and the caller
    decides: `find_best_split` drops the candidate, and
    `find_best_split_shared` substitutes that leaf's unsplit terms, which is
    what "contributes 0.0 to a sum of (child - parent) differences" means once
    the sum has been replaced by a ratio.
    """

    var num: Float64
    var den: Float64
    var ok: Bool


@always_inline
def _cosine_out(g: Float64, h: Float64, lambda_reg: Float64) -> Float64:
    """The value CatBoost calls `leafApprox`, in our sign convention.

    `CalcAverage` (`catboost/private/libs/algo_helpers/online_predictor.h`) is
    `count > 0 ? sumDelta / (count + reg) : 0`, so a child of zero weight
    emits zero and contributes nothing to either accumulator instead of
    dividing by `lambda` alone. That guard is CatBoost's and is kept, so the
    Cosine path has no divide-by-zero at `lambda_l2 = 0` where the L2 path
    would produce an infinity; the difference is confined to the Cosine
    branch and cannot reach the default.
    """
    if not (h > 0.0):
        return 0.0
    return -g / (h + lambda_reg)


@always_inline
def _cosine_unsplit(
    g: Float64, h: Float64, lambda_reg: Float64
) -> _CosineTerms:
    """The two accumulator terms of a node scored WITHOUT a split, which is
    CatBoost's `CalcScoreWithoutSplit` (`leafwise_scoring.cpp`): the same
    calcer run over the node's own totals with an empty second child."""
    var out = _cosine_out(g, h, lambda_reg)
    return _CosineTerms(-out * g, out * out * h, True)


@always_inline
def _cosine_pair(
    left_g: Float64,
    left_h: Float64,
    right_g: Float64,
    right_h: Float64,
    lambda_reg: Float64,
    sign: Int,
    bounds: OutputBounds,
    constrained: Bool,
    finish: Bool,
    max_delta_step: Float64,
    path_smooth: Float64,
    left_c: Int,
    right_c: Int,
    parent_output: Float64,
) -> _CosineTerms:
    """One candidate's contribution to CatBoost's two Cosine accumulators.

    `TCosineScoreCalcer::AddLeaf` (`catboost/private/libs/algo/
    score_calcers.h`) is

        Scores[i][0] += leafApprox * leafStats.SumWeightedDelta
        Scores[i][1] += leafApprox * leafApprox * leafStats.SumWeight

    and the score is `Scores[i][0] / sqrt(Scores[i][1])`. CatBoost's
    `SumWeightedDelta` is a sum of derivatives, which is `-G` in our sign
    convention, and their `SumWeight` sits where our `H` sits, so their
    `leafApprox` is our child output and the two terms are `-out * G` and
    `out^2 * H`.

    Why this is not just a rescaling of `_split_gain`. Substituting the free
    Newton step `out = -G / (H + lambda)` gives

        num = sum  G^2 / (H + lambda)          <- exactly the L2 score
        den = sum  G^2 * H / (H + lambda)^2

    **At `lambda = 0` those two expressions are the same one**, so the score
    collapses to `sqrt(num)` and, `sqrt` being strictly increasing, has the
    same argmax as L2. Cosine's entire difference from L2 is a function of
    `lambda_l2`, which mojotrees stock now sets to 0. Said the other way: for
    a single child the ratio is `|G| / sqrt(H)`, in which `lambda` has
    cancelled outright -- Cosine is L2 with the regularizer left in the leaf
    value and taken back out of the score. `docs/design/CATBOOST_CATALOG.md`
    A10 carries the full argument and the consequences.

    `finish` and `constrained` are handled exactly as `_split_gain` handles
    them, and for the same reason: CatBoost's monotone branch
    (`scoring.cpp`, `UpdateScores`) calls `AddLeaf` with the *monotonized*
    leaf value, so the accumulators are built from the output the leaf will
    actually emit rather than from the free step. Our monotone rule rejects
    where CatBoost projects, which predates this parameter; a rejection comes
    back as `ok = False` rather than as a zero, because 0.0 is a legitimate
    numerator here and cannot double as a sentinel.

    Addend order is fixed by this function -- left child, then right child,
    in both accumulators -- and never crosses a task boundary, so the value
    does not move with `MOJOTREES_NUM_WORKERS`. CatBoost's SIMD kernel
    accumulates the right child first and seeds the denominator before either
    (`_cosine_score` applies that seed once, at the end, instead); both are
    divergences in association, recorded in A10.
    """
    var left_out = _cosine_out(left_g, left_h, lambda_reg)
    var right_out = _cosine_out(right_g, right_h, lambda_reg)
    if finish:
        left_out = finish_leaf_output(
            left_out, max_delta_step, path_smooth, left_c, parent_output
        )
        right_out = finish_leaf_output(
            right_out, max_delta_step, path_smooth, right_c, parent_output
        )
    if constrained:
        left_out = bounds.clamp(left_out)
        right_out = bounds.clamp(right_out)
        if violates(sign, left_out, right_out):
            return _CosineTerms(0.0, 0.0, False)
    var num = -left_out * left_g
    num += -right_out * right_g
    var den = left_out * left_out * left_h
    den += right_out * right_out * right_h
    return _CosineTerms(num, den, True)


@always_inline
def _cosine_score(num: Float64, den: Float64) -> Float64:
    """`Scores[i][0] / sqrt(Scores[i][1])` with CatBoost's denominator seed
    folded in. Kept as one function so the seed cannot be applied twice or
    forgotten at one of the four call sites."""
    return num / sqrt(den + _COSINE_DEN_FLOOR)


struct SplitInfo(Copyable, Movable, Writable):
    """Best split found for a node. `found` is False when no valid split
    exists (e.g. every candidate violates min_child_hess). `default_left` is
    the direction taken by rows in the feature's missing bin; it is False and
    unused for a feature with no missing bin.

    A numerical split uses `bin` as an inclusive left-going threshold. A
    categorical split sets `is_categorical`, leaves `bin` at -1, and carries
    the bins that route left in `cat_bitset`; every other bin, including the
    missing/unseen bin 0, routes right, so `default_left` is False for it as
    it is in LightGBM."""

    var feature: Int
    var bin: Int
    var gain: Float64
    var found: Bool
    var default_left: Bool
    var is_categorical: Bool
    var cat_bitset: CatBitset

    def __init__(
        out self,
        feature: Int,
        bin: Int,
        gain: Float64,
        found: Bool,
        default_left: Bool = False,
    ):
        """A numerical split, or with found=False the absence of one."""
        self.feature = feature
        self.bin = bin
        self.gain = gain
        self.found = found
        self.default_left = default_left
        self.is_categorical = False
        self.cat_bitset = cat_empty()

    @staticmethod
    def categorical(
        feature: Int, gain: Float64, bitset: CatBitset
    ) -> SplitInfo:
        """A categorical split sending `bitset`'s bins to the left child."""
        var s = SplitInfo(feature, -1, gain, True, False)
        s.is_categorical = True
        s.cat_bitset = bitset
        return s^

    def goes_left(self, bin: Int) -> Bool:
        """Whether a row whose bin for `self.feature` is `bin` routes left.
        Growth, prediction, and GPU partitioning all route through here, so
        the three cannot disagree."""
        if self.is_categorical:
            return cat_contains(self.cat_bitset, bin)
        return bin <= self.bin

    def write_to(self, mut writer: Some[Writer]):
        if not self.found:
            writer.write("SplitInfo(none)")
        elif self.is_categorical:
            writer.write(
                "SplitInfo(feature=",
                self.feature,
                ", categorical, gain=",
                self.gain,
                ")",
            )
        else:
            writer.write(
                "SplitInfo(feature=",
                self.feature,
                ", bin<=",
                self.bin,
                ", gain=",
                self.gain,
                ")",
            )

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


@always_inline
def _split_gain(
    left_g: Float64,
    left_h: Float64,
    right_g: Float64,
    right_h: Float64,
    lambda_reg: Float64,
    parent_score: Float64,
    sign: Int,
    bounds: OutputBounds,
    constrained: Bool,
    finish: Bool = False,
    max_delta_step: Float64 = 0.0,
    path_smooth: Float64 = 0.0,
    left_c: Int = 0,
    right_c: Int = 0,
    parent_output: Float64 = 0.0,
) -> Float64:
    """Gain of one candidate split. `left_g` and `right_g` are the child
    gradient sums after L1 soft-thresholding.

    Unconstrained and unfinished, this is the plain second-order formula.
    Under active monotonic constraints it is LightGBM's constrained form: both
    child outputs are clamped into the node's bounds and scored at those
    outputs, and a candidate whose outputs run against `sign` scores 0.0,
    which no caller accepts because a split must beat a gain of 0.0 to be
    chosen.

    `finish` is the same shape for `max_delta_step` and `path_smooth`: both
    make a child emit something other than the free Newton step, so a
    candidate scored at the free step would not be the gain the tree
    realizes. The children are scored at `finish_leaf_output(...)` instead,
    which is exactly the value `tree._leaf_value` will write, and the monotone
    clamp is then applied on top of the finished output rather than under it,
    so an active constraint still bounds what a leaf can emit.
    """
    if not constrained and not finish:
        return (
            left_g * left_g / (left_h + lambda_reg)
            + right_g * right_g / (right_h + lambda_reg)
            - parent_score
        )
    var left_out = -left_g / (left_h + lambda_reg)
    var right_out = -right_g / (right_h + lambda_reg)
    if finish:
        left_out = finish_leaf_output(
            left_out, max_delta_step, path_smooth, left_c, parent_output
        )
        right_out = finish_leaf_output(
            right_out, max_delta_step, path_smooth, right_c, parent_output
        )
    if constrained:
        left_out = bounds.clamp(left_out)
        right_out = bounds.clamp(right_out)
        if violates(sign, left_out, right_out):
            return 0.0
    return (
        output_score(left_g, left_h, lambda_reg, left_out)
        + output_score(right_g, right_h, lambda_reg, right_out)
        - parent_score
    )


@always_inline
def _feature_gain(
    gain: Float64,
    feature: Int,
    extra: ExtraTreeParams,
    extra_active: Bool,
    penalize: Bool,
    sign: Int,
    depth: Int,
    node_rows: Int,
    cegb: CegbNodeCosts,
) raises -> Float64:
    """One feature's best gain after that feature's costs, or 0.0 when the
    result is rejected (a gain of 0.0 never beats the running best, which
    starts there and is compared strictly).

    Called once per feature, which is where LightGBM charges `feature_contri`
    and every CEGB cost, and the only placement where a per-feature cost is
    charged once rather than once per candidate: the split cost is a property
    of the node and the coupled and lazy costs are properties of the feature,
    so charging per candidate would multiply them by however many thresholds
    the feature happens to offer. The order is LightGBM's: the multiplier
    scales the gain, the CEGB costs are then subtracted as absolute amounts,
    the monotone penalty discounts what is left, and `min_gain_to_split` is
    the floor the result must clear.

    `cegb` is the node's costs, prepared once before this scan by
    `cegb.prepare_cegb_node`, so charging them here is one lookup and one
    subtraction with no ledger read and no walk over the node's rows.
    `penalize` gates only the `feature_contri` multiplier; the CEGB terms gate
    themselves on `CegbNodeCosts.active`.

    With no bundle active this returns `gain` unchanged, so the caller's
    comparison is the one the scan made inline before the bundle existed.
    """
    if not extra_active:
        return gain
    var g = gain
    if penalize:
        g = extra.penalties.penalized_gain(g, feature)
    g = cegb.adjusted_gain_at(g, feature, node_rows)
    g = apply_monotone_penalty(g, sign, depth, extra.monotone_penalty)
    if not passes_min_gain(g, extra.min_gain_to_split):
        return 0.0
    return g


def find_best_split(
    hist: Histogram,
    lambda_reg: Float64 = 1.0,
    min_child_hess: Float64 = 1e-3,
    min_data_in_leaf: Int = 0,
    lambda_l1: Float64 = 0.0,
    allowed: List[Bool] = [],
    features: List[Int] = [],
    missing_bins: List[Int] = [],
    monotone: List[Int] = [],
    bounds: OutputBounds = OutputBounds.unbounded(),
    cats: CategoricalSpec = CategoricalSpec.none(),
    cat_params: CategoricalParams = CategoricalParams.default(),
    extra: ExtraTreeParams = ExtraTreeParams(),
    n_rows: Int = 0,
    depth: Int = 0,
    node: Int = 0,
    tree_index: Int = 0,
    parent_output: Float64 = 0.0,
    cegb: CegbNodeCosts = CegbNodeCosts.inactive(),
    settings: DispatchSettings = DispatchSettings.unresolved(),
    score_function: Int = SCORE_L2,
) raises -> SplitInfo:
    """Scan all (feature, bin) split candidates and return the one with the
    highest gain. `lambda_reg` is the L2 penalty on the leaf hessian sum and
    `lambda_l1` the L1 penalty soft-thresholding every gradient sum. Only
    splits with positive gain are returned as found.

    `allowed` is an optional per-feature mask from the node's interaction
    constraints: features with `allowed[f]` false are skipped entirely. An
    empty `allowed` (the default) means every feature is a candidate; a mask
    shorter than `hist.n_features` disallows the features past its end.

    `features` is an optional ascending list of the feature ids this node may
    split on, from feature subsampling. Empty (the default) scans every
    feature; otherwise only the listed ones are scanned, so excluded features
    cost nothing here and their (all-zero) histogram slices are never read.
    The two restrictions compose: a feature must be listed and allowed.

    `missing_bins` is the per-feature missing-bin table of the binned matrix
    (`BinnedMatrix.missing_bin`), or empty when no feature reserves a missing
    bin. A feature whose entry is >= 0 is scanned over its ordinary bins only
    and reports a `default_left` direction for its missing rows.

    `monotone` is the per-feature monotonic constraint vector (`1`, `0`, or
    `-1`) and `bounds` the interval this node's output must lie in. An empty
    `monotone` (the default) means unconstrained and keeps the original
    scoring path; `bounds` is then ignored.

    `cats` marks which features are categorical (`BinnedMatrix.cats`) and
    `cat_params` holds the categorical hyperparameters. A categorical feature
    is handed to `find_best_categorical_split` instead of being scanned by
    threshold, and its `missing_bins` entry is not consulted.

    `extra` is the remaining tree-control bundle
    (`tree_parameters_extra.ExtraTreeParams`) and defaults to inactive, in
    which case this function scans exactly as it did before it existed. When
    it is active:

    - a feature's best candidate is penalized once per feature, after the
      feature's scan and before it is compared against the running best. That
      is LightGBM's placement and the only one where a per-feature cost is
      charged once rather than once per candidate.
    - `min_gain_to_split` is the floor that best candidate must clear. At the
      default 0.0 it is the `gain > 0.0` test the running best already
      applies.
    - `extra_trees` replaces a feature's scan with a single drawn threshold,
      keyed by (`extra_seed`, `tree_index`, `node`, feature). A feature whose
      draw fails `min_data_in_leaf` or `min_child_hess` yields no split for
      that feature rather than falling back to a full scan, as in LightGBM.
    - `max_delta_step` and `path_smooth` change what a child emits, so
      candidates are scored at the finished outputs (see `_split_gain`).
      `parent_output` is this node's own value, which is what its children
      smooth toward, and `n_rows` its row count.
    - `random_strength` adds a seeded normal to every candidate's gain before
      that gain is compared to anything, keyed by (`random_strength_seed`,
      `tree_index`, `node`, feature, bin). Its standard deviation is
      `random_strength * random_score_scale`, and a positive strength with no
      scale is refused rather than silently scaled to nothing. It is refused
      on a matrix with a categorical feature, for the reason `extra_trees`
      is. At the default of 0 not a single instruction of arithmetic is
      added, which is what makes it a no-op rather than a small one.

    `n_rows` is the node's row count, used by the per-split CEGB cost and, at
    0, taken from the histogram's own totals. `depth` is the node's depth in
    edges from the root, which `monotone_penalty` discounts by.

    `cegb` is this node's CEGB costs (`cegb.prepare_cegb_node`), computed once
    before the scan so that charging a candidate is one lookup and one
    subtraction. Only a grower carrying the ensemble's `CegbLedger` can build
    them; a caller that passes none gets the split cost reconstructed from
    `extra.penalties.cegb` and is refused for the two penalties that need the
    ledger.

    `settings` is the fit's `parallel.DispatchSettings` snapshot. The scan
    dispatches once per node, and without a snapshot that dispatch re-reads
    `MOJOTREES_NUM_WORKERS`, the two grain variables, and the two core-pool
    variables, and re-detects the machine's core counts, every time -- to
    answer a question whose inputs were fixed when the fit started. Passing a
    snapshot reads none of them. It cannot move a gain: the fold below is the
    same ascending strict `>` at every task count, which is the property
    `test_cpu_dispatch` pins. The default sentinel keeps the live reads, so a
    caller that has not been wired behaves exactly as it did.

    `score_function` selects which functional of the children's sums is
    maximized. `SCORE_L2` (the default) is the second-order gain this
    function has always computed and takes exactly the path it always took.
    `SCORE_COSINE` is CatBoost's CPU default, a ratio rather than a sum; see
    the module docstring, `_cosine_pair`, and
    `docs/design/CATBOOST_CATALOG.md` A10. It is refused on a matrix with a
    categorical feature, because a categorical candidate is a category *set*
    scored inside `find_best_categorical_split` under the L2 gain and only
    that search's winner reaches here; rescoring the winner under Cosine
    would put two score functions inside one argmax.

    Exclusive feature bundling (efb.mojo) never reaches here: a bundled
    histogram is expanded back to one slice per original feature before the
    search runs (`tree.grow_tree`), so this function reads the per-feature
    histogram it has always read and a `SplitInfo` names an original feature
    and an original bin whatever the training matrix looked like."""
    var best = SplitInfo(-1, -1, 0.0, False)
    if len(missing_bins) > 0 and len(missing_bins) != hist.n_features:
        raise Error("missing_bins length must equal n_features")
    if len(monotone) > 0 and len(monotone) != hist.n_features:
        raise Error("monotone length must equal n_features")
    var constrained = len(monotone) > 0
    if constrained and cats.any_categorical():
        for f in range(hist.n_features):
            if cats.is_cat(f) and monotone_sign(monotone, f) != MONOTONE_FREE:
                raise Error(
                    "monotonic constraints are not supported on categorical"
                    " features"
                )
    var masked = len(allowed) > 0
    var use_all = len(features) == 0
    var n_active = hist.n_features if use_all else len(features)

    # Tested once per node, not once per candidate: an inactive bundle must
    # leave the scan on exactly the path it took before the bundle existed.
    var extra_active = extra.is_active()
    var penalize = extra_active and extra.penalties.contri_active()

    # `cegb` is this node's prepared costs, which only a grower holding the
    # ensemble's ledger can build. A caller that prepared none but did
    # configure CEGB still gets the split cost, from the same numbers through
    # the same multiplication in the same order: `CegbNodeCosts` keeps
    # `split_rate` factored from the row count precisely so the count this
    # scan computes inside its loop is the one charged. The coupled and lazy
    # costs cannot be reconstructed that way, because they read state that
    # spans the ensemble, so they are refused here rather than silently
    # charged as zero.
    var costs = cegb.copy()
    if not costs.active and extra.penalties.cegb.is_active():
        check_cegb_grower_support(extra.penalties.cegb, False, False)
        costs = prepare_cegb_node(
            extra.penalties.cegb, CegbLedger.none(), hist.n_features, 0
        )
    var finish = extra.needs_leaf_finish()
    var draw_one = extra.extra_trees
    if draw_one and cats.any_categorical():
        # LightGBM randomizes the categorical set search too, over partition
        # positions rather than over thresholds. That is a different draw
        # from this one and is not implemented, so the combination is refused
        # rather than silently scoring categoricals exhaustively while every
        # numerical feature gets a single draw.
        raise Error(
            "extra_trees is implemented for numerical thresholds only; a"
            " categorical feature is searched as category partitions, whose"
            " random draw is a separate rule"
        )

    # `random_strength`: CatBoost's seeded noise on a candidate's gain. The
    # standard deviation is `random_strength * random_score_scale`, and the
    # scale is the ensemble's, not the node's, so a caller that set the
    # strength without it is refused here as well as at `check_scalars`:
    # this function is reachable directly and a caller who came in that way
    # would otherwise get an unregularized answer that looked regularized.
    var noise_stdev = extra.random_score_stdev()
    var noisy = extra.random_strength > 0.0
    if noisy and not (noise_stdev > 0.0):
        raise Error(
            "random_strength is set but ExtraTreeParams.random_score_scale"
            " is not. The scale is CatBoost's derivativesStDevFromZero *"
            " modelSizeDecrease, which is a property of the ensemble's"
            " current gradients and cannot be read from a node histogram."
            " Compute it once per tree with"
            " tree_parameters_extra.random_score_scale_from_gradients(grad,"
            " n_rows, iteration * learning_rate)"
        )
    if noisy and cats.any_categorical():
        # A categorical feature's candidates are category *sets*, searched
        # inside `find_best_categorical_split`, and only that search's winner
        # reaches this function. Noising the winner would noise one candidate
        # per feature while every numerical feature had every candidate
        # noised, which is a different regularizer wearing the same name. The
        # combination is refused rather than half-applied; making it work
        # means the draw moving into the partition search, which is
        # categorical.mojo's.
        raise Error(
            "random_strength is implemented for numerical thresholds only; a"
            " categorical feature is searched as category partitions, and"
            " only that search's winner reaches here, so its candidates"
            " cannot each be noised from this function"
        )
    var noise_seed = extra.random_strength_seed

    # Read once per node, never per candidate: an L2 scan must leave this
    # function on exactly the path it took before the parameter existed, and
    # a `Bool` the whole scan holds constant is a branch the compiler can
    # hoist out of the bin loop.
    check_score_function(score_function)
    var cosine = score_function == SCORE_COSINE
    if cosine and cats.any_categorical():
        raise Error(
            "score_function=cosine is implemented for numerical thresholds"
            " only; a categorical feature is searched as category partitions"
            " scored with the L2 gain, and only that search's winner reaches"
            " this function, so the two score functions would end up inside"
            " one argmax"
        )

    comptime W = SIMD_LANES
    # `Histogram` storage is LightGBM's interleaved `hist_t`: cell `i` is the
    # Float64 pair at `_gh[2 * i]` and `_gh[2 * i + 1]`. Two pointers where
    # there were three, and a feature's slice is two sequential streams rather
    # than three.
    var gh_p = hist._gh.unsafe_ptr()
    var count_p = hist._count.unsafe_ptr()

    # One slot per scanned feature. A feature that yields no candidate leaves
    # its gain at 0.0, which the fold below cannot accept because `best.gain`
    # starts at 0.0 and the comparison is strict: exactly the outcome the
    # serial loop reached by never touching `best` for that feature. The
    # slots are plain scalars rather than a `List[SplitInfo]` so that a task
    # writes through an unsafe pointer, which is how every other parallel
    # kernel in this package writes its output.
    var res_gain = List[Float64](capacity=n_active)
    res_gain.resize(n_active, 0.0)
    var res_feature = List[Int](capacity=n_active)
    res_feature.resize(n_active, -1)
    var res_bin = List[Int](capacity=n_active)
    res_bin.resize(n_active, -1)
    var res_flag = List[Int](capacity=n_active)
    res_flag.resize(n_active, 0)
    # A categorical winner also carries its bitset. Allocated only when the
    # matrix has a categorical feature at all, because the common case is a
    # wholly numerical matrix and this runs once per node.
    var any_cat = cats.any_categorical()
    var res_bits = List[UInt64]()
    if any_cat:
        res_bits.resize(CAT_BITSET_WORDS * n_active, UInt64(0))

    # A task cannot raise: the dispatch shapes in `parallel.mojo` take a
    # non-raising body, because an exception escaping one worker while the
    # others are still running has nowhere to go. A feature whose scan raises
    # therefore records that it did and stops, and the fold below re-runs it
    # serially so the error surfaces with its own message, raised from the
    # lowest-indexed feature that failed. That is one wasted scan on a path
    # that ends in an exception, and it keeps the message exact instead of
    # reducing every failure to a flag.
    var res_fail = List[UInt8](capacity=n_active)
    res_fail.resize(n_active, UInt8(0))

    var gain_out = res_gain.unsafe_ptr()
    var feature_out = res_feature.unsafe_ptr()
    var bin_out = res_bin.unsafe_ptr()
    var flag_out = res_flag.unsafe_ptr()
    var bits_out = res_bits.unsafe_ptr()
    var fail_out = res_fail.unsafe_ptr()

    def scan_feature(i_feature: Int) raises {imm}:
        var f = i_feature if use_all else features[i_feature]
        if f < 0 or f >= hist.n_features:
            return
        if masked and (f >= len(allowed) or not allowed[f]):
            return
        var sign = monotone_sign(monotone, f) if constrained else MONOTONE_FREE
        var missing_bin = -1
        if len(missing_bins) > 0:
            missing_bin = missing_bins[f]
            if missing_bin >= hist.n_bins:
                raise Error("missing bin index out of range")
        var base = f * hist.n_bins
        # One 2W-wide pair accumulator where there were two W-wide ones.
        # Lane `2k` of `vp` receives the gradients of bins `base + k`,
        # `base + k + W`, ... in ascending order, which is exactly the
        # sequence lane `k` of the old `vg` received; `deinterleave` is a lane
        # permutation with no arithmetic and `reduce_add` folds the same tree.
        # The scan's totals are therefore the same Float64 they were.
        var vp = SIMD[DType.float64, 2 * W](0.0)
        var vc = SIMD[DType.int, W](0)
        var b = 0
        while b + W <= hist.n_bins:
            vp += gh_p.unsafe_load[width = 2 * W](2 * (base + b))
            vc += count_p.unsafe_load[width=W](base + b)
            b += W
        var halves = vp.deinterleave()
        var total_g = halves[0].reduce_add()
        var total_h = halves[1].reduce_add()
        var total_c = Int(vc.reduce_add())
        while b < hist.n_bins:
            var cell = gh_p.unsafe_load[width=2](2 * (base + b))
            total_g += cell[0]
            total_h += cell[1]
            total_c += count_p.unsafe_load(base + b)
            b += 1
        # This feature's own best candidate, held apart from `best` so a
        # per-feature cost is charged once, after the feature's scan, rather
        # than once per candidate. With no bundle active the two are the same
        # thing: a local best starting at 0.0 followed by a strict comparison
        # against `best` accepts exactly the candidates the inline comparison
        # accepted, in the same order, so the chosen split is unchanged.
        var f_gain = 0.0
        var f_bin = -1
        var f_default_left = False
        var f_found = False
        # The node's rows, which the per-split CEGB cost is charged per. Every
        # feature's bins sum to the same total, so the histogram answers this
        # when the caller does not.
        var node_rows = n_rows if n_rows > 0 else total_c

        # Categorical features are searched as category partitions, never as
        # ordinal thresholds. Bin 0 (missing, unseen, dropped) is excluded
        # from every candidate set there, so this feature's `missing_bins`
        # entry plays no part.
        if cats.is_cat(f):
            var n_cat = cats.n_categories(f)
            if n_cat >= hist.n_bins:
                raise Error(
                    "categorical feature has more categories than bins"
                )
            var cs = find_best_categorical_split(
                hist._gh,
                hist._count,
                base,
                n_cat,
                total_g,
                total_h,
                total_c,
                lambda_reg,
                lambda_l1,
                min_child_hess,
                min_data_in_leaf,
                cat_params,
            )
            if cs.found:
                var g = _feature_gain(
                    cs.gain,
                    f,
                    extra,
                    extra_active,
                    penalize,
                    sign,
                    depth,
                    node_rows,
                    costs,
                )
                gain_out.unsafe_store(i_feature, g)
                feature_out.unsafe_store(i_feature, f)
                flag_out.unsafe_store(i_feature, _FLAG_CATEGORICAL)
                for w in range(CAT_BITSET_WORDS):
                    bits_out.unsafe_store(
                        CAT_BITSET_WORDS * i_feature + w, cs.bitset[w]
                    )
            return

        var parent_g = soft_threshold_l1(total_g, lambda_l1)
        var parent_score = parent_g * parent_g / (total_h + lambda_reg)
        # The same functional applied to the node without a split, which is
        # CatBoost's `CalcScoreWithoutSplit`. Constant across this node's
        # candidates, as `parent_score` is, so subtracting it only sets the
        # zero point the `> f_gain` test measures from.
        var parent_cos = 0.0
        if cosine:
            var pt = _cosine_unsplit(parent_g, total_h, lambda_reg)
            parent_cos = _cosine_score(pt.num, pt.den)

        # Ordinary bins are [0, n_scan); the missing bin sits at n_scan and is
        # never a threshold, only a side to route.
        var n_scan = missing_bin if missing_bin >= 0 else hist.n_bins
        var miss_g = 0.0
        var miss_h = 0.0
        var miss_c = 0
        if missing_bin >= 0:
            miss_g = hist.grad_at(base + missing_bin)
            miss_h = hist.hess_at(base + missing_bin)
            miss_c = hist.count_at(base + missing_bin)

        # `extra_trees`: one drawn threshold for this feature instead of the
        # whole scan. The candidate space is the same one the scan walks, so
        # the top threshold counts only when missing rows can fill the right
        # child, and a feature that offers no candidate offers no split.
        var pick = -1
        if draw_one:
            var n_candidates = n_scan if miss_c > 0 else n_scan - 1
            pick = extra_threshold_index(
                n_candidates, extra.extra_seed, tree_index, node, f
            )
            if pick < 0:
                return

        var left_g = 0.0
        var left_h = 0.0
        var left_c = 0
        for b in range(n_scan):
            # The top threshold puts every ordinary bin left, so it is only a
            # split at all when missing rows are there to fill the right child.
            if b == n_scan - 1 and miss_c == 0:
                break
            left_g += hist.grad_at(base + b)
            left_h += hist.hess_at(base + b)
            left_c += hist.count_at(base + b)
            # The prefix sums above are what the drawn threshold is built
            # from, so they are accumulated for every bin below it and only
            # the drawn one is scored.
            if draw_one and b != pick:
                continue

            # This threshold's noise draw, taken once and shared by the two
            # routing directions below. Keyed by (seed, tree, node, feature,
            # bin), so it does not depend on this scan having reached bin `b`
            # by walking bins 0..b-1, nor on which task this feature ran on.
            var noise = 0.0
            if noisy:
                noise = random_score_noise(
                    noise_stdev, noise_seed, tree_index, node, f, b
                )

            # Missing to the left, scored first so an exact tie keeps
            # default_left, as in LightGBM. With no missing rows in this node
            # the two candidates coincide, and the left default is recorded
            # for whatever missing value arrives at predict time.
            if missing_bin >= 0:
                var dl_left_g = left_g + miss_g
                var dl_left_h = left_h + miss_h
                var dl_left_c = left_c + miss_c
                var dl_right_g = total_g - dl_left_g
                var dl_right_h = total_h - dl_left_h
                if not (
                    dl_left_h < min_child_hess
                    or dl_right_h < min_child_hess
                    or dl_left_c < min_data_in_leaf
                    or total_c - dl_left_c < min_data_in_leaf
                ):
                    var tl = soft_threshold_l1(dl_left_g, lambda_l1)
                    var tr = soft_threshold_l1(dl_right_g, lambda_l1)
                    var gain: Float64
                    if cosine:
                        var ct = _cosine_pair(
                            tl,
                            dl_left_h,
                            tr,
                            dl_right_h,
                            lambda_reg,
                            sign,
                            bounds,
                            constrained,
                            finish,
                            extra.max_delta_step,
                            extra.path_smooth,
                            dl_left_c,
                            total_c - dl_left_c,
                            parent_output,
                        )
                        # A monotone rejection is `ok = False`, and 0.0 is the
                        # value `_split_gain` returns for it: a candidate must
                        # beat the running best, which starts at 0.0 under a
                        # strict `>`, so the rejection is expressed the same
                        # way under both score functions.
                        gain = 0.0
                        if ct.ok:
                            gain = _cosine_score(ct.num, ct.den) - parent_cos
                    else:
                        gain = _split_gain(
                            tl,
                            dl_left_h,
                            tr,
                            dl_right_h,
                            lambda_reg,
                            parent_score,
                            sign,
                            bounds,
                            constrained,
                            finish,
                            extra.max_delta_step,
                            extra.path_smooth,
                            dl_left_c,
                            total_c - dl_left_c,
                            parent_output,
                        )
                    if noisy:
                        gain += noise
                    if gain > f_gain:
                        f_gain = gain
                        f_bin = b
                        f_default_left = True
                        f_found = True

            # Missing to the right. For a feature with no missing bin this is
            # the only candidate and the scan is exactly the ordinal one.
            if missing_bin < 0 or miss_c > 0:
                var right_g = total_g - left_g
                var right_h = total_h - left_h
                if not (
                    left_h < min_child_hess
                    or right_h < min_child_hess
                    or left_c < min_data_in_leaf
                    or total_c - left_c < min_data_in_leaf
                ):
                    var tl = soft_threshold_l1(left_g, lambda_l1)
                    var tr = soft_threshold_l1(right_g, lambda_l1)
                    var gain: Float64
                    if cosine:
                        var ct = _cosine_pair(
                            tl,
                            left_h,
                            tr,
                            right_h,
                            lambda_reg,
                            sign,
                            bounds,
                            constrained,
                            finish,
                            extra.max_delta_step,
                            extra.path_smooth,
                            left_c,
                            total_c - left_c,
                            parent_output,
                        )
                        gain = 0.0
                        if ct.ok:
                            gain = _cosine_score(ct.num, ct.den) - parent_cos
                    else:
                        gain = _split_gain(
                            tl,
                            left_h,
                            tr,
                            right_h,
                            lambda_reg,
                            parent_score,
                            sign,
                            bounds,
                            constrained,
                            finish,
                            extra.max_delta_step,
                            extra.path_smooth,
                            left_c,
                            total_c - left_c,
                            parent_output,
                        )
                    if noisy:
                        gain += noise
                    if gain > f_gain:
                        f_gain = gain
                        f_bin = b
                        f_default_left = False
                        f_found = True

            if draw_one:
                break

        if f_found:
            var g = _feature_gain(
                f_gain,
                f,
                extra,
                extra_active,
                penalize,
                sign,
                depth,
                node_rows,
                costs,
            )
            gain_out.unsafe_store(i_feature, g)
            feature_out.unsafe_store(i_feature, f)
            bin_out.unsafe_store(i_feature, f_bin)
            if f_default_left:
                flag_out.unsafe_store(i_feature, _FLAG_DEFAULT_LEFT)

    def scan_one(i_feature: Int) {imm}:
        try:
            scan_feature(i_feature)
        except:
            fail_out.unsafe_store(i_feature, UInt8(1))

    dispatch_features_with(
        settings,
        scan_one,
        n_active,
        split_scan_ops(n_active, hist.n_bins, len(missing_bins) > 0),
    )

    # The serial fold. Ascending scan order and a strict `>`, which is the
    # comparison the loop above used to make inline, so the winner and the
    # tie-break are the ones this function has always chosen.
    for i_feature in range(n_active):
        if res_fail[i_feature] != UInt8(0):
            scan_feature(i_feature)
            raise Error("split scan failed on feature ", i_feature)
        var g = gain_out.unsafe_load(i_feature)
        if g > best.gain:
            var flag = flag_out.unsafe_load(i_feature)
            if (flag & _FLAG_CATEGORICAL) != 0:
                var bits = cat_empty()
                for w in range(CAT_BITSET_WORDS):
                    bits[w] = bits_out.unsafe_load(
                        CAT_BITSET_WORDS * i_feature + w
                    )
                best = SplitInfo.categorical(
                    feature_out.unsafe_load(i_feature), g, bits
                )
            else:
                best = SplitInfo(
                    feature_out.unsafe_load(i_feature),
                    bin_out.unsafe_load(i_feature),
                    g,
                    True,
                    (flag & _FLAG_DEFAULT_LEFT) != 0,
                )

    return best^


def find_best_split_shared(
    mut audit: SharedSplitAudit,
    hists: List[Histogram],
    lambda_reg: Float64 = 1.0,
    min_child_hess: Float64 = 1e-3,
    min_data_in_leaf: Int = 0,
    lambda_l1: Float64 = 0.0,
    allowed: List[Bool] = [],
    features: List[Int] = [],
    missing_bins: List[Int] = [],
    monotone: List[Int] = [],
    bounds: List[OutputBounds] = [],
    parent_outputs: List[Float64] = [],
    extra: ExtraTreeParams = ExtraTreeParams(),
    n_rows: Int = 0,
    depth: Int = 0,
    node: Int = 0,
    tree_index: Int = 0,
    settings: DispatchSettings = DispatchSettings.unresolved(),
    score_function: Int = SCORE_L2,
) raises -> SplitInfo:
    """The one split a whole level of leaves shares, for `grow_policy =
    oblivious` (growth_policy.mojo, `docs/design/OBLIVIOUS.md` B2).

    `hists` is one histogram per leaf of the level, in the order the grower
    will apply the split in, which `tree.grow_tree` fixes as ascending node
    id. A candidate is a (feature, bin, missing direction) as it always was;
    what changes is that its gain is the **sum over every leaf of the level**
    of that leaf's own gain at that candidate, computed from that leaf's own
    left/right sums, and the winner is applied to all of them. That is what
    makes the tree symmetric and a row's leaf a bit pattern rather than a
    traversal.

    Where the cross-leaf reduction runs, which is the point
    ------------------------------------------------------
    It is **not a second pass**. `docs/design/OBLIVIOUS.md` B2 requires the
    reduction be fused into a launch that already runs, and B5 registers "it
    needs its own dispatch" as the first thing that would kill the design. It
    does not need one. A candidate (f, b) reads feature f's histogram slice
    and nothing else, in every leaf, so the whole reduction for feature f
    fits inside feature f's own task of the single feature-parallel dispatch
    this function makes -- the same `dispatch_features_with` that
    `find_best_split` makes for one node. The leaf loop is the OUTER loop
    inside that task: each leaf's slice is walked once, ascending by bin,
    accumulating that leaf's contribution into a per-bin accumulator that the
    task owns. There is one dispatch per level, not one per leaf and not one
    per leaf plus a reduce.

    Derived bound (arithmetic, not a measurement): a level of L leaves costs
    L `find_best_split` dispatches under leaf-wise or depth-wise growth and
    exactly 1 here. A depth-6 tree's search dispatches go from
    1+2+4+8+16+32 = 63 to 6. The scan itself reads the same cells either way
    (L slices of `n_active * n_bins`), so this removes fan-out and barrier
    cost, not arithmetic.

    Per-leaf legality contributes zero and is recorded
    -------------------------------------------------
    A leaf that cannot satisfy `min_data_in_leaf` or `min_child_hess` at a
    candidate does NOT veto it: it adds 0.0 to that candidate's sum and is
    split anyway, possibly into an empty child. Vetoing would let one narrow
    leaf out of sixteen decide the level, which is the opposite of what a
    symmetric tree is for. `audit` records how many leaves contributed zero
    at the *chosen* candidate, because "legal for 1 leaf of 16" and "legal
    for 16 of 16" are different objects and a reader has to be able to tell
    them apart.

    **This rule is OURS. It is NOT verified from CatBoost source and cannot
    be.** `docs/design/OBLIVIOUS.md` B2 said, marked *verify*, that "CatBoost
    scores leaves that fail as zero contribution". That sentence has no
    referent: CatBoost's `SymmetricTree` has no `min_data_in_leaf` (their
    documentation scopes it to Depthwise and Lossguide; the CPU code reads it
    only in `GreedyTensorSearchDepthwise` and `FindBestCandidate`, where it
    gates whether an *already existing* leaf is expanded rather than whether
    a candidate's child is admissible; the CUDA searcher switches even that
    off for symmetric trees), and CatBoost has no `min_sum_hessian_in_leaf`
    parameter at all. They never had to define this case.

    **Why zero-contribution rather than a veto:** the GPU device implements
    zero-contribution-without-veto, and host and device must grow the same
    tree. That is the reason. Everything else is corroboration: a veto would
    let one narrow leaf of sixteen decide a whole level, and CatBoost's
    nearest analogue is a zero *guard* rather than a zero *penalty*
    (`CalcAverage` in
    `catboost/private/libs/algo_helpers/online_predictor.h` returns 0 for a
    child of zero weight, so an empty child adds nothing and the candidate
    survives). Do not cite a CatBoost file for the rule itself; there is
    none.

    What the summed gain is, against CatBoost's
    -------------------------------------------
    Each leaf's contribution is `split._split_gain`, which subtracts that
    leaf's parent score. CatBoost's `L2` score calcer
    (`catboost/private/libs/algo/score_calcers.cpp`, `TL2ScoreCalcer`) sums
    `sumDer^2 / (sumWeight + l2)` over every child of every leaf and does not
    subtract anything; the parent term comes off once per level as
    `gain = score - scoreBeforeSplit` in `SelectBestCandidate`. The sum of
    the per-leaf parent scores is a constant across the candidates of one
    level -- every feature's bins total to the same per-leaf sums -- so the
    two forms have the same argmax, and ours is LightGBM's spelling of it.
    CatBoost's CPU *default* is `Cosine`, which is a ratio of two cross-leaf
    sums (`Scores[i][0] / sqrt(Scores[i][1])`) and is NOT this. It is
    available here as `score_function=SCORE_COSINE`, off by default; see
    below and `docs/design/CATBOOST_CATALOG.md` A10.

    `score_function` under a shared split
    -------------------------------------
    `SCORE_L2` (the default) is everything described above and takes exactly
    the path it took before the parameter existed. `SCORE_COSINE` is a ratio,
    so a level's candidate cannot be a sum of per-leaf gains: CatBoost keeps
    `numScoreBlocks = 1` for `SymmetricTree`
    (`catboost/cuda/methods/greedy_subsets_searcher/greedy_search_helper.cpp`)
    precisely because the two accumulators are summed across every leaf of the
    level and ONE ratio is taken per candidate. That is what is built here:
    two accumulator planes per (bin, direction) instead of one, folded over
    leaves in the same ascending order, and the level's unsplit score
    subtracted once at the end.

    **A leaf that fails a minimum, or whose candidate an active monotonic
    constraint rejects, contributes its UNSPLIT terms** rather than nothing.
    That is the exact generalization of the L2 rule and not a new one: under
    L2, `sum over legal leaves of (child - parent)` is identically
    `(sum over legal of child + sum over illegal of parent) - sum over all of
    parent`. Cosine's ratio admits only the second spelling. The illegal-leaf
    count `audit` reports is unchanged and counts the same leaves.

    **At `lambda_l2 = 0` this changes nothing.** Numerator and denominator
    become the same expression, the ratio collapses to `sqrt` of the L2 sum,
    and `sqrt` is strictly increasing, so the level picks the same candidate.
    mojotrees stock is `lambda_l2 = 0`. See A10; this is a result rather than
    a reason to leave the parameter out.

    Cosine's extra planes are allocated with length 0 when it is off, so an
    L2 level allocates exactly what it allocated before.

    Determinism
    -----------
    Two sums and one maximum, and none of the three crosses a task boundary.
    A leaf's prefix sums run ascending by bin inside one task, as they do in
    `find_best_split`. The cross-leaf sum runs ascending by position in
    `hists` inside that same task, so the order of the addends is a property
    of the argument and not of the worker count. The choice among features is
    a maximum, folded afterwards in ascending scan order under the same strict
    `>` the serial loop would have used. Values are therefore identical at
    `MOJOTREES_NUM_WORKERS` 1, 3 and 8.

    Scope, and what is refused rather than half-applied
    ---------------------------------------------------
    Numerical thresholds only. A categorical feature's candidates are
    category *sets* whose order is derived from that node's own
    gradient/hessian ratios, so there is no one set to share across a level
    without a different search; the caller refuses the combination
    (`tree.grow_tree`). `extra_trees` draws one threshold per *node* and has
    no meaning for a level, and the CEGB penalties that read the ensemble
    ledger charge per node; both are refused here. `min_gain_to_split`,
    `feature_contri`, `monotone_penalty` and `random_strength` are live and
    apply to the level: the first three are charged once per feature against
    the summed gain (`_feature_gain`, LightGBM's placement), and
    `random_strength` draws once per (feature, bin) candidate keyed by
    `node`, which the grower passes as the level's lowest node id, so the
    level gets one draw per candidate rather than one per leaf.

    `bounds` and `parent_outputs`, when non-empty, are per leaf and parallel
    to `hists`: the monotone interval a leaf's output must lie in, and the
    value it currently emits (what `path_smooth` shrinks its children
    toward). Empty means unbounded and 0.0 for every leaf.
    """
    var n_leaves = len(hists)
    audit = SharedSplitAudit.none()
    if n_leaves <= 0:
        raise Error(
            "find_best_split_shared needs one histogram per leaf of the level"
        )
    var n_features = hists[0].n_features
    var n_bins = hists[0].n_bins
    for l in range(1, n_leaves):
        if hists[l].n_features != n_features or hists[l].n_bins != n_bins:
            raise Error(
                "every leaf histogram of a level must have the same shape"
            )
    if len(missing_bins) > 0 and len(missing_bins) != n_features:
        raise Error("missing_bins length must equal n_features")
    if len(monotone) > 0 and len(monotone) != n_features:
        raise Error("monotone length must equal n_features")
    if len(bounds) > 0 and len(bounds) != n_leaves:
        raise Error("bounds must hold one interval per leaf of the level")
    if len(parent_outputs) > 0 and len(parent_outputs) != n_leaves:
        raise Error("parent_outputs must hold one value per leaf of the level")
    if extra.extra_trees:
        raise Error(
            "extra_trees draws one threshold per node and a level of an"
            " oblivious tree has one split for every node in it; the two"
            " rules cannot both hold"
        )
    if extra.penalties.cegb.is_active():
        raise Error(
            "the CEGB penalties are charged per node against a ledger that"
            " spans the ensemble; a level's shared split is charged once and"
            " that accounting is not written. Leave the cegb_penalty_*"
            " parameters at 0 under grow_policy=oblivious"
        )

    var constrained = len(monotone) > 0
    var masked = len(allowed) > 0
    var use_all = len(features) == 0
    var n_active = n_features if use_all else len(features)

    var extra_active = extra.is_active()
    var penalize = extra_active and extra.penalties.contri_active()
    var costs = CegbNodeCosts.inactive()
    var finish = extra.needs_leaf_finish()

    var noise_stdev = extra.random_score_stdev()
    var noisy = extra.random_strength > 0.0
    if noisy and not (noise_stdev > 0.0):
        raise Error(
            "random_strength is set but ExtraTreeParams.random_score_scale"
            " is not; see find_best_split"
        )
    var noise_seed = extra.random_strength_seed

    check_score_function(score_function)
    var cosine = score_function == SCORE_COSINE

    # The two accumulators a feature's task folds its leaves into, one per
    # bin per direction, plus the illegal-leaf counts beside them. Allocated
    # once for the whole dispatch and striped by feature slot, so a task
    # writes only its own `[i_feature * n_bins, (i_feature + 1) * n_bins)`
    # range and no task allocates. At 50 features and 255 bins that is
    # 50 * 255 * (8 + 8 + 8 + 8) = 408 KB for a level of any width, because
    # the leaves are folded in rather than held.
    var stripe = n_active * n_bins
    var acc_left = List[Float64](capacity=stripe)
    acc_left.resize(stripe, 0.0)
    var acc_right = List[Float64](capacity=stripe)
    acc_right.resize(stripe, 0.0)
    var ill_left = List[Int](capacity=stripe)
    ill_left.resize(stripe, 0)
    var ill_right = List[Int](capacity=stripe)
    ill_right.resize(stripe, 0)
    # Cosine's second accumulator, one plane per direction. Zero-length when
    # the score function is L2, which is the `res_bits` pattern in
    # `find_best_split`: the pointer is taken either way and dereferenced
    # only under the flag that sized the list.
    var cos_stripe = stripe if cosine else 0
    var den_left = List[Float64](capacity=cos_stripe)
    den_left.resize(cos_stripe, 0.0)
    var den_right = List[Float64](capacity=cos_stripe)
    den_right.resize(cos_stripe, 0.0)

    var res_gain = List[Float64](capacity=n_active)
    res_gain.resize(n_active, 0.0)
    var res_feature = List[Int](capacity=n_active)
    res_feature.resize(n_active, -1)
    var res_bin = List[Int](capacity=n_active)
    res_bin.resize(n_active, -1)
    var res_flag = List[Int](capacity=n_active)
    res_flag.resize(n_active, 0)
    var res_ill = List[Int](capacity=n_active)
    res_ill.resize(n_active, 0)
    var res_fail = List[UInt8](capacity=n_active)
    res_fail.resize(n_active, UInt8(0))

    var accl_out = acc_left.unsafe_ptr()
    var accr_out = acc_right.unsafe_ptr()
    var denl_out = den_left.unsafe_ptr()
    var denr_out = den_right.unsafe_ptr()
    var illl_out = ill_left.unsafe_ptr()
    var illr_out = ill_right.unsafe_ptr()
    var gain_out = res_gain.unsafe_ptr()
    var feature_out = res_feature.unsafe_ptr()
    var bin_out = res_bin.unsafe_ptr()
    var flag_out = res_flag.unsafe_ptr()
    var ill_out = res_ill.unsafe_ptr()
    var fail_out = res_fail.unsafe_ptr()

    def scan_feature(i_feature: Int) raises {imm}:
        var f = i_feature if use_all else features[i_feature]
        if f < 0 or f >= n_features:
            return
        if masked and (f >= len(allowed) or not allowed[f]):
            return
        var sign = monotone_sign(monotone, f) if constrained else MONOTONE_FREE
        var missing_bin = -1
        if len(missing_bins) > 0:
            missing_bin = missing_bins[f]
            if missing_bin >= n_bins:
                raise Error("missing bin index out of range")
        var base = f * n_bins
        var off = i_feature * n_bins

        # Ordinary bins are [0, n_scan); the missing bin sits at n_scan and is
        # never a threshold, only a side to route.
        var n_scan = missing_bin if missing_bin >= 0 else n_bins

        # Whether ANY leaf of the level has rows in the missing bin. It is a
        # level-wide question because the candidate is level-wide: with no
        # missing rows anywhere, "every ordinary bin left, missing alone
        # right" puts every row of every leaf on one side and is not a split,
        # and the two routing directions coincide so only one of them is
        # scored (which is what keeps `default_left` on an exact tie, as in
        # `find_best_split`).
        var level_miss_c = 0
        if missing_bin >= 0:
            for l in range(n_leaves):
                level_miss_c += hists[l].count_at(base + missing_bin)
        var n_top = n_scan if level_miss_c > 0 else n_scan - 1
        if n_top <= 0:
            return
        var score_left = missing_bin >= 0
        var score_right = missing_bin < 0 or level_miss_c > 0

        for b in range(n_top):
            accl_out.unsafe_store(off + b, 0.0)
            accr_out.unsafe_store(off + b, 0.0)
            illl_out.unsafe_store(off + b, 0)
            illr_out.unsafe_store(off + b, 0)
        if cosine:
            for b in range(n_top):
                denl_out.unsafe_store(off + b, 0.0)
                denr_out.unsafe_store(off + b, 0.0)

        # The level's own unsplit score under Cosine, accumulated leaf by leaf
        # beside the candidates in the same loop below. Constant across this
        # feature's candidates -- every feature's bins total to the same
        # per-leaf sums -- so subtracting it sets the zero point the `>
        # f_gain` test measures from and nothing else.
        var p_num = 0.0
        var p_den = 0.0

        # THE CROSS-LEAF REDUCTION. Outer loop over leaves, ascending, inside
        # this one feature's task: each leaf's slice is read once in bin order
        # and folded straight into the per-bin accumulators above. No second
        # pass, no second dispatch, and the addends of every sum are ordered
        # by the loops rather than by the scheduler.
        var level_c = 0
        for l in range(n_leaves):
            var b0 = 0
            var vp = SIMD[DType.float64, 2 * SIMD_LANES](0.0)
            var vc = SIMD[DType.int, SIMD_LANES](0)
            var ghp = hists[l]._gh.unsafe_ptr()
            var cp = hists[l]._count.unsafe_ptr()
            while b0 + SIMD_LANES <= n_bins:
                vp += ghp.unsafe_load[width = 2 * SIMD_LANES](2 * (base + b0))
                vc += cp.unsafe_load[width=SIMD_LANES](base + b0)
                b0 += SIMD_LANES
            var halves = vp.deinterleave()
            var total_g = halves[0].reduce_add()
            var total_h = halves[1].reduce_add()
            var total_c = Int(vc.reduce_add())
            while b0 < n_bins:
                var cell = ghp.unsafe_load[width=2](2 * (base + b0))
                total_g += cell[0]
                total_h += cell[1]
                total_c += cp.unsafe_load(base + b0)
                b0 += 1
            level_c += total_c

            var parent_g = soft_threshold_l1(total_g, lambda_l1)
            var parent_score = parent_g * parent_g / (total_h + lambda_reg)
            # This leaf's unsplit terms: the level's zero point, and also what
            # this leaf contributes at a candidate it cannot take.
            var pt = _CosineTerms(0.0, 0.0, True)
            if cosine:
                pt = _cosine_unsplit(parent_g, total_h, lambda_reg)
                p_num += pt.num
                p_den += pt.den
            var lb = bounds[l].copy() if len(bounds) > 0 else (
                OutputBounds.unbounded()
            )
            var pout = parent_outputs[l] if len(parent_outputs) > 0 else 0.0

            var miss_g = 0.0
            var miss_h = 0.0
            var miss_c = 0
            if missing_bin >= 0:
                # One line for the pair, where two planes cost two.
                var mcell = ghp.unsafe_load[width=2](2 * (base + missing_bin))
                miss_g = mcell[0]
                miss_h = mcell[1]
                miss_c = cp.unsafe_load(base + missing_bin)

            var left_g = 0.0
            var left_h = 0.0
            var left_c = 0
            for b in range(n_top):
                var lcell = ghp.unsafe_load[width=2](2 * (base + b))
                left_g += lcell[0]
                left_h += lcell[1]
                left_c += cp.unsafe_load(base + b)

                if score_left:
                    var dl_left_g = left_g + miss_g
                    var dl_left_h = left_h + miss_h
                    var dl_left_c = left_c + miss_c
                    var dl_right_g = total_g - dl_left_g
                    var dl_right_h = total_h - dl_left_h
                    if (
                        dl_left_h < min_child_hess
                        or dl_right_h < min_child_hess
                        or dl_left_c < min_data_in_leaf
                        or total_c - dl_left_c < min_data_in_leaf
                    ):
                        illl_out.unsafe_store(
                            off + b, illl_out.unsafe_load(off + b) + 1
                        )
                        if cosine:
                            # An illegal leaf stays as it is. Under L2 that is
                            # written as adding 0.0 to a sum of differences;
                            # under a ratio the only way to write it is to add
                            # the leaf's unsplit terms to both accumulators,
                            # and the two are arithmetically the same rule.
                            accl_out.unsafe_store(
                                off + b, accl_out.unsafe_load(off + b) + pt.num
                            )
                            denl_out.unsafe_store(
                                off + b, denl_out.unsafe_load(off + b) + pt.den
                            )
                    else:
                        var tl = soft_threshold_l1(dl_left_g, lambda_l1)
                        var tr = soft_threshold_l1(dl_right_g, lambda_l1)
                        if cosine:
                            var ct = _cosine_pair(
                                tl,
                                dl_left_h,
                                tr,
                                dl_right_h,
                                lambda_reg,
                                sign,
                                lb,
                                constrained,
                                finish,
                                extra.max_delta_step,
                                extra.path_smooth,
                                dl_left_c,
                                total_c - dl_left_c,
                                pout,
                            )
                            # A monotone rejection is the illegal case again:
                            # this leaf stays as it is and does not veto the
                            # level, which is what `_split_gain` returning 0.0
                            # means on the L2 path.
                            var cn = ct.num if ct.ok else pt.num
                            var cd = ct.den if ct.ok else pt.den
                            accl_out.unsafe_store(
                                off + b, accl_out.unsafe_load(off + b) + cn
                            )
                            denl_out.unsafe_store(
                                off + b, denl_out.unsafe_load(off + b) + cd
                            )
                        else:
                            accl_out.unsafe_store(
                                off + b,
                                accl_out.unsafe_load(off + b)
                                + _split_gain(
                                    tl,
                                    dl_left_h,
                                    tr,
                                    dl_right_h,
                                    lambda_reg,
                                    parent_score,
                                    sign,
                                    lb,
                                    constrained,
                                    finish,
                                    extra.max_delta_step,
                                    extra.path_smooth,
                                    dl_left_c,
                                    total_c - dl_left_c,
                                    pout,
                                ),
                            )

                if score_right:
                    var right_g = total_g - left_g
                    var right_h = total_h - left_h
                    if (
                        left_h < min_child_hess
                        or right_h < min_child_hess
                        or left_c < min_data_in_leaf
                        or total_c - left_c < min_data_in_leaf
                    ):
                        illr_out.unsafe_store(
                            off + b, illr_out.unsafe_load(off + b) + 1
                        )
                        if cosine:
                            accr_out.unsafe_store(
                                off + b, accr_out.unsafe_load(off + b) + pt.num
                            )
                            denr_out.unsafe_store(
                                off + b, denr_out.unsafe_load(off + b) + pt.den
                            )
                    else:
                        var tl = soft_threshold_l1(left_g, lambda_l1)
                        var tr = soft_threshold_l1(right_g, lambda_l1)
                        if cosine:
                            var ct = _cosine_pair(
                                tl,
                                left_h,
                                tr,
                                right_h,
                                lambda_reg,
                                sign,
                                lb,
                                constrained,
                                finish,
                                extra.max_delta_step,
                                extra.path_smooth,
                                left_c,
                                total_c - left_c,
                                pout,
                            )
                            var cn = ct.num if ct.ok else pt.num
                            var cd = ct.den if ct.ok else pt.den
                            accr_out.unsafe_store(
                                off + b, accr_out.unsafe_load(off + b) + cn
                            )
                            denr_out.unsafe_store(
                                off + b, denr_out.unsafe_load(off + b) + cd
                            )
                        else:
                            accr_out.unsafe_store(
                                off + b,
                                accr_out.unsafe_load(off + b)
                                + _split_gain(
                                    tl,
                                    left_h,
                                    tr,
                                    right_h,
                                    lambda_reg,
                                    parent_score,
                                    sign,
                                    lb,
                                    constrained,
                                    finish,
                                    extra.max_delta_step,
                                    extra.path_smooth,
                                    left_c,
                                    total_c - left_c,
                                    pout,
                                ),
                            )

        # This feature's best candidate, in the scan order `find_best_split`
        # uses -- ascending bin, missing-left before missing-right -- under
        # the same strict `>`, so an exact tie keeps the lower bin and, at
        # that bin, `default_left`.
        var f_gain = 0.0
        var f_bin = -1
        var f_default_left = False
        var f_ill = 0
        var f_found = False
        # One ratio per candidate for the whole level, which is CatBoost's
        # `numScoreBlocks = 1`, minus the level's unsplit score.
        var level_parent = _cosine_score(p_num, p_den) if cosine else 0.0
        for b in range(n_top):
            # One draw per (feature, bin) candidate, shared by the two routing
            # directions and by every leaf: the noise belongs to the split the
            # level takes, and the level takes one. Added here rather than
            # inside the fold so it is added once and not `n_leaves` times.
            var noise = 0.0
            if noisy:
                noise = random_score_noise(
                    noise_stdev, noise_seed, tree_index, node, f, b
                )
            if score_left:
                var g = accl_out.unsafe_load(off + b)
                if cosine:
                    g = (
                        _cosine_score(g, denl_out.unsafe_load(off + b))
                        - level_parent
                    )
                g += noise
                if g > f_gain:
                    f_gain = g
                    f_bin = b
                    f_default_left = True
                    f_ill = illl_out.unsafe_load(off + b)
                    f_found = True
            if score_right:
                var g = accr_out.unsafe_load(off + b)
                if cosine:
                    g = (
                        _cosine_score(g, denr_out.unsafe_load(off + b))
                        - level_parent
                    )
                g += noise
                if g > f_gain:
                    f_gain = g
                    f_bin = b
                    f_default_left = False
                    f_ill = illr_out.unsafe_load(off + b)
                    f_found = True

        if f_found:
            var node_rows = n_rows if n_rows > 0 else level_c
            var g = _feature_gain(
                f_gain,
                f,
                extra,
                extra_active,
                penalize,
                sign,
                depth,
                node_rows,
                costs,
            )
            gain_out.unsafe_store(i_feature, g)
            feature_out.unsafe_store(i_feature, f)
            bin_out.unsafe_store(i_feature, f_bin)
            ill_out.unsafe_store(i_feature, f_ill)
            if f_default_left:
                flag_out.unsafe_store(i_feature, _FLAG_DEFAULT_LEFT)

    def scan_one(i_feature: Int) {imm}:
        try:
            scan_feature(i_feature)
        except:
            fail_out.unsafe_store(i_feature, UInt8(1))

    # ONE dispatch for the whole level. `split_scan_ops` estimates one node's
    # scan, so the level's estimate is that times the number of leaves folded
    # into it, which is exactly the work this dispatch does.
    dispatch_features_with(
        settings,
        scan_one,
        n_active,
        n_leaves * split_scan_ops(n_active, n_bins, len(missing_bins) > 0),
    )

    var best = SplitInfo(-1, -1, 0.0, False)
    var best_ill = 0
    for i_feature in range(n_active):
        if res_fail[i_feature] != UInt8(0):
            scan_feature(i_feature)
            raise Error("shared split scan failed on feature ", i_feature)
        var g = gain_out.unsafe_load(i_feature)
        if g > best.gain:
            var flag = flag_out.unsafe_load(i_feature)
            best = SplitInfo(
                feature_out.unsafe_load(i_feature),
                bin_out.unsafe_load(i_feature),
                g,
                True,
                (flag & _FLAG_DEFAULT_LEFT) != 0,
            )
            best_ill = ill_out.unsafe_load(i_feature)

    if best.found:
        audit = SharedSplitAudit(n_leaves, best_ill, n_leaves - best_ill)
    return best^
