"""Linear leaves: LightGBM's `linear_tree`, as an algorithm core.

A linear tree routes exactly as an ordinary tree does and then, at the leaf,
emits an affine function of a few raw features instead of a constant. This
module owns that leaf: the sufficient statistics, the regularized solve, the
eligibility rules, the numerical fallbacks, the prediction arithmetic, and
the bytes the model format would have to gain. Nothing here is reachable
from a public entry point; `check_linear_tree_public` says so and names what
is missing (see "Reachability" below).

Why a sidecar and not a field on `Tree`
---------------------------------------
`tree.Tree` stores one `Float64` per node in `value`. There is no array on
it that can hold a coefficient vector, and there is no honest way to encode
one in the fields it has: `threshold_bin`, `missing_bin`, and `cat_offset`
are all consumed by routing. A linear leaf therefore cannot be represented
by today's `Tree`, and this module does not pretend otherwise.

The smallest extension that fixes that is *not* new arrays on `Tree`. `Tree`
is constructed positionally in seven places (`tree.grow_tree`,
`tree_sparse`, the two `train_gpu` growers, the distributed merge,
`serialize._read_trees`, and the LightGBM importer), read by
`contrib.mojo`, `model_dump.mojo`, `gpu_predict.mojo`, and the C ABI, and
its node-array layout is a documented invariant. Widening it touches all of
that for a feature most models never use.

Instead this module defines a *sidecar*, `LinearEnsemble`, keyed by
`(tree index, leaf ordinal)`:

- The tree index is the position in `Booster.trees`, which is round-major
  for multiclass and is the order serialization writes.
- The leaf ordinal is `Tree.leaf_ordinals()`, which `tree.mojo` already
  documents as stable across save and load.

So the whole model-representation extension is one optional field on
`Booster` and `MulticlassBooster` (`var linear: LinearEnsemble`, empty
meaning constant leaves) plus one optional serialized section. `Tree` itself
is untouched, and every consumer that does not know about linear leaves
keeps compiling.

The constant fallback is exact, not a stub
------------------------------------------
`LinearLeaf.intercept` is *by construction* the value the ordinary grower
already wrote into `Tree.value[leaf]`, and the linear part is expressed in
features centered on their in-leaf hessian-weighted means:

    leaf(x) = intercept + sum_j coef[j] * (x[feature[j]] - center[j])

A consumer that ignores the sidecar therefore does not read garbage: it
predicts the linear leaf evaluated at its own training centroid, which is a
well-defined (less accurate) constant-leaf model. That is what makes it
possible for `contrib.mojo`, `gpu_predict.mojo`, and the LightGBM exporter
to *refuse explicitly* rather than silently disagree with the CPU predictor:
they can detect the sidecar and raise, and if a later decision is to degrade
instead of raise, the degradation is a defined model rather than an
accident. `linear_leaf_reduces_to_constant` is that property as a predicate.

The algorithm
-------------
Growth is unchanged. Splits are found from gradient histograms exactly as
they are today, the tree is grown to completion with constant leaves, and
then `refit_linear_tree` replaces each leaf's constant with an affine
function fitted on the rows that reached it. This is a deliberate
difference from LightGBM, which scores split candidates under the linear
fit; see "LightGBM differences" below.

For one leaf, with rows R, per-row gradient g_r and hessian h_r, an
intercept b already fixed by the grower, and candidate features S:

    center[j] = (sum_r h_r x_rj) / (sum_r h_r)
    xt_rj     = x_rj - center[j]
    A[j][k]   = sum_r h_r xt_rj xt_rk          (m by m, symmetric PSD)
    q[j]      = sum_r g_r xt_rj

    (A + linear_lambda * I) c = -q

Centering is what makes this a two-line derivation rather than an
(m+1)-by-(m+1) solve. The leaf objective is

    F(b, c) = sum_r [ g_r f_r + h_r f_r^2 / 2 ]
            + reg(b) + linear_lambda ||c||^2 / 2

with `f_r = b + c . xt_r`. Because `sum_r h_r xt_rj = 0` by construction, the
cross term `b * sum_r h_r xt_rj` vanishes for every j, so `dF/dc` does not
mention `b` at all:

    dF/dc_j = q[j] + (A c)[j] + linear_lambda * c[j]

Three things follow, and all three are load-bearing:

1. The coefficient solve is correct whatever the intercept is. The grower's
   intercept is not the unregularized Newton step in general: it has been
   through `max_delta_step`, `path_smooth`, and the monotone clamp. Taking
   it as given costs nothing.
2. `linear_lambda -> infinity` drives `c -> 0` and the leaf becomes exactly
   today's constant leaf, bit for bit, because the intercept was never
   recomputed.
3. The training-objective improvement is available in closed form from the
   solved coefficients alone, with no second pass over the rows:

       improvement = -( q . c + c^T A c / 2 + linear_lambda ||c||^2 / 2 )

   which is `c^T (A + linear_lambda I) c / 2 >= 0` in exact arithmetic. It is
   computed from the returned `c` rather than from that identity, so
   roundoff that made the fit worse than the constant is caught and the leaf
   falls back.

`linear_lambda` regularizes the coefficients only. The intercept keeps the
regularization it already had, `lambda_l1` and `lambda_l2` through
`tree._leaf_value`. Those are separable terms in `F`, so they do not disturb
the decoupling above.

Which features a leaf may use
-----------------------------
`eligible_leaf_features` starts from the leaf's *branch* set (the features
split on between the root and it, LightGBM's rule) and then removes, in
order:

- categorical features. Their bins are set members, not magnitudes; a
  coefficient on an integer category code is meaningless. LightGBM excludes
  them too.
- features with a non-finite raw value on any row reaching the leaf. A
  linear term cannot evaluate on a `NaN`, and one `inf` poisons the whole
  normal-equation system. See "Missing values" below.
- features whose in-leaf hessian-weighted variance is at or below
  `LINEAR_MIN_VARIANCE` relative to their scale. A constant column
  contributes a zero row and column to `A` and would be dropped by the
  solve anyway; catching it here keeps the solve's drop path for genuine
  rank deficiency.
- features beyond `max_leaf_features`, ranked by the univariate objective
  reduction `q[j]^2 / (A[j][j] + linear_lambda)`, descending, ties broken by
  ascending feature index so the selection is deterministic. -1, the
  default, keeps them all.
- every feature, if `min_data_per_linear_feature > 0` and the leaf has fewer
  than `m * min_data_per_linear_feature` rows; the set is trimmed from the
  bottom of the same ranking until it fits. 0, the default, is off and
  leaves rank deficiency to the solve.

The surviving set is ascending by feature index, which is what
serialization and inspection read.

Numerical stability
-------------------
Everything accumulates in `Float64`, in two passes: the weighted means
first, then the centered second moments. The one-pass
`sum(xy) - sum(x)sum(y)/n` form loses most of its significant digits on a
column whose values are large relative to their spread, which is the common
case for a feature the tree has already split on twice.

The solve is a Cholesky factorization of `M = A + linear_lambda I`,
symmetrically equilibrated by `D = diag(1/sqrt(M_jj))` so every diagonal is
1 before the factorization runs. Equilibration is exact (it is a change of
variable, undone on the way out) and it is applied *after* the ridge term,
so `linear_lambda` keeps its meaning in the raw feature scale rather than in
a normalized one.

Failure is handled by dropping features, not by returning something
plausible:

- a nonpositive diagonal, or a pivot that falls to `LINEAR_PIVOT_TOL` of its
  starting value, drops that feature and restarts the factorization;
- `LinearParams.ridge_eps` adds a relative jitter, `eps * trace(A) / m`, to
  the diagonal, so an exactly singular `A` with `linear_lambda = 0` is still
  factorizable;
- a non-finite coefficient drops the largest-magnitude one and restarts;
- a fit whose computed improvement is not positive and finite is discarded
  and the leaf stays constant;
- `max_linear_deviation`, if positive, discards a fit whose linear part
  exceeds it in absolute value on any row of the leaf. This is the one check
  that costs a second pass over the rows, and it is off by default.

Each drop strictly shrinks the active set, so the loop runs at most `m`
times.

Missing values
--------------
Two separate rules, and they are not the same rule.

*Training*: a feature with a non-finite value on any row reaching the leaf
is not eligible for that leaf at all. Imputing during the fit would put a
made-up number into `A`, and excluding only the offending rows would fit
different features on different row sets, which is not a least-squares
problem. Note that this is a per-leaf test on the raw values, not a test on
`BinnedMatrix.missing_bin`: `use_missing=False` bins `NaN` as `0.0` and
reserves no bin, so the binned matrix cannot answer the question.

*Prediction*: an eligible feature can still be handed a `NaN` or an infinity
at prediction time, on a row unlike anything in training. Such a value is
replaced by that leaf's `center[j]`, so the term contributes exactly zero
and the leaf falls back to its intercept along that one axis. This is
mojotrees-defined; it is not read off LightGBM. It is why `center` is stored
rather than folded into the intercept.

Routing is untouched by all of this. A missing value still follows
`default_left` to a leaf, exactly as in a constant-leaf tree.

Categorical features
--------------------
Categorical features route normally and are never linear terms. A model may
mix them freely with linear leaves; `eligible_leaf_features` removes them
from the branch set and the rest of the tree is unaffected.

Monotonic constraints
---------------------
Linear leaves and active monotonic constraints are refused together, by
`check_monotone_compatible`. This is not conservatism, it is the honest
reading of the proof in `monotone.mojo`.

That proof works by giving every node a closed output interval and clamping
the leaf's output into it, so that two examples separated at a node that
splits on the constrained feature satisfy `pred(x) <= mid <= pred(x')`. A
leaf whose output is an affine function of `x` does not have "an output" to
clamp. Requiring the *function* to lie in a bounded interval for all `x`
forces every coefficient to zero, since a nonconstant affine function on an
unbounded domain is unbounded. So the mechanism as written admits only
constant leaves.

The refinement that would work is real but is a separate piece of
machinery. A leaf's region is an axis-aligned box in raw feature space (the
intersection of its ancestors' bin thresholds, mapped back through the bin
edges), and an affine function on a box attains its extrema at a corner, so
the interval test becomes a test on two corner values. That needs bin edges
at the leaf (the grower is handed a `BinnedMatrix`, which has none),
per-feature box bounds carried down the frontier, and the sign condition
`sign_f * coef_f >= 0` on every constrained feature. `docs/LINEAR_TREES.md`
states the full condition. Until it exists, the combination raises.

Objectives
----------
Any objective that supplies a gradient and a positive hessian works, since
the solve reads nothing else. Two families are refused by
`check_objective_compatible`:

- `QUANTILE`, `L1`, and `MAPE` renew their leaves after growth
  (`boosting.objective_renews_leaves`), replacing the Newton value with a
  weighted residual percentile. That percentile is not the minimizer of any
  quadratic, so it does not decouple from the coefficients the way the
  intercept above does, and fitting the two independently would give a leaf
  that is neither the renewed constant nor the linear least-squares fit.
- `LAMBDARANK` produces gradients from query-group pairs. The leaf solve is
  well-defined on them, but the resulting model has never been checked
  against anything, and a ranking objective with a per-leaf regression is
  not a LightGBM-documented combination. Refused rather than guessed at.

`CUSTOM` is allowed: `objective.check_custom_grad_hess` already guarantees
the finite, nonnegative hessians the solve needs.

Multiclass
----------
Nothing special. `LinearEnsemble.trees` is parallel to `Booster.trees`, and
`MulticlassBooster.trees` is round-major, so class `k`'s tree in iteration
`i` is at `i * n_classes + k` in both. Each class's leaves are fitted from
that class's own gradients and hessians, and each class's leaf may end up
with a different feature set. The per-class *raw* score is piecewise affine;
the softmax probabilities are not, exactly as they are not monotone under
monotone constraints.

Continued training
------------------
Two obligations, both stated as requirements rather than assumed.

1. The raw scores `train_more` resumes from must be computed *through* the
   linear leaves. `predict_ensemble_raw` is that pass. Resuming from the
   constant fallback would restart boosting from a model the ensemble does
   not actually predict, and every subsequent tree would be fitted to the
   wrong residual.
2. Continued training needs the raw feature matrix, not just the binned one.
   `train_more` today takes a `BinnedMatrix`; a linear continuation needs
   the raw values for both the resume pass and the new trees' leaf fits.

`check_continuation_compatible` enforces the invariants that can be checked
from the two ensembles: the sidecar must be as long as the tree list, its
feature count must match, and an ensemble that already has linear leaves
cannot be continued as a constant-leaf one or the reverse.

Per-tree weights (DART)
-----------------------
`alternate_boosting.fold_weights_into_trees` multiplies each tree's node
values by that tree's drop weight, so that an ensemble with per-tree weights
becomes one a single-shrinkage `Booster` represents exactly.
`LinearEnsemble.scale_all` is its other half: folding the constants without
the coefficients would leave a leaf whose affine function no longer passes
through the value its tree carries, and the two would describe different
models. The two calls belong together, with the same weight vector, in the
same place.

Serialization
-------------
A new model format version, v4. `linear_section_text` writes the section and
`read_linear_section` reads it, both in this module and both in the same
raw-bit-pattern token encoding `serialize.mojo` uses, so the cross-lane
change is a call in each direction rather than a new parser. A model with no
linear leaves writes nothing, so v4 files for constant-leaf models are byte
for byte what v3 wrote and old readers keep working. A v4 reader must refuse
a file it cannot evaluate; `read_linear_section` validates ordinals, feature
ids, and lengths against the trees it is given.

LightGBM differences
--------------------
- **Split gains ignore the linear fit.** LightGBM evaluates candidate splits
  under the per-leaf regression, so its trees differ in *shape* from
  constant-leaf ones. mojotrees grows the constant-leaf tree and refits its
  leaves. The fitted leaves are optimal for the tree that was grown; the
  tree is not the one LightGBM would have grown. This is the largest and
  most visible difference and it is not a rounding matter.
  `accumulate_leaf_stats` and `solve_leaf_coefficients` are the pieces a
  linear-aware split search would reuse; the cost is one solve per candidate
  rather than one per leaf.
- **Regularization placement.** `linear_lambda` is applied to the
  coefficients and not to the intercept. LightGBM's `linear_lambda` is added
  to the diagonal of its own `(m+1)`-square system; whether its intercept
  row is included has not been read off its source, so this is stated as a
  mojotrees choice and flagged in `docs/LINEAR_TREES.md` as parity-unverified.
- **Missing values.** The eligibility rule and the centre substitution above
  are mojotrees-defined. LightGBM's behaviour here has not been read off its
  source.
- **`ridge_eps`, `max_leaf_features`, `min_data_per_linear_feature`, and
  `max_linear_deviation`** have no LightGBM counterpart. All four default to
  inactive, so a default `LinearParams` is the LightGBM-shaped
  configuration.

Reachability
------------
The feature is not public and must not become public before the
representation is real. `tree_parameters_extra.check_extra_option_supported`
refuses `linear_tree` and `linear_lambda` by name, `params.mojo` routes
through it, and `lgbm_model_io.mojo` refuses `is_linear=1`, `leaf_const`,
and `leaf_coeff` on the way in. None of that changes here.
`check_linear_tree_public` is the one gate inside this module: it raises
with the list of cross-lane changes that are still outstanding, and
`handoffs/remaining_03_linear_trees.md` carries each of them as a
ready-to-apply patch.
"""

from std.math import isfinite, sqrt
from std.memory import bitcast

from .binning import BinnedMatrix
from .categorical import CategoricalSpec
from .monotone import MonotoneConstraints
from .tree import Tree

# ---------------------------------------------------------------------------
# Objective codes, mirrored
# ---------------------------------------------------------------------------
#
# This module sits *below* `boosting.mojo` in the import graph, because
# `Booster` is what will hold a `LinearEnsemble` (see the handoff) and
# `src/mojotrees` has no mutual imports anywhere. So the four objective codes
# the leaf-compatibility gate needs are mirrored here rather than imported,
# the way `model_dump.mojo` mirrors `categorical._MAX_CATEGORY` for the same
# kind of reason.
#
# The canonical definitions are `boosting.QUANTILE`, `boosting.L1`,
# `boosting.MAPE`, and `ranking.LAMBDARANK`. They are part of a stable public
# numbering (the objective code is serialized in every model file and crosses
# the C ABI), so they do not move; if one ever does, this block and
# `boosting.objective_renews_leaves` have to move together, and the handoff
# asks the boosting lane for a cross-check that would fail if they did not.

comptime _QUANTILE = 4
comptime _L1 = 5
comptime _LAMBDARANK = 7
comptime _MAPE = 10


def _objective_renews_leaves(objective: Int) -> Bool:
    """Mirror of `boosting.objective_renews_leaves`: the objectives that
    replace every leaf's Newton value with a weighted residual percentile
    after growth."""
    return (
        objective == _QUANTILE or objective == _L1 or objective == _MAPE
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The model format version a file carrying linear leaves has to declare.
# `serialize.mojo` writes v4 today (split gains, a cover presence flag, and
# optional feature names); a `linear` section is what makes a file v5. A v5
# file with no linear leaves is byte-identical to the v4 one, so the bump
# costs nothing for models that do not use the feature.
#
# The number is read off `serialize._VERSION` rather than assumed: v4 was
# taken by another lane while this module was being written, which is
# exactly the hazard `linear_model_format_version` exists to contain. If
# `_VERSION` moves again before this is wired, this constant moves with it
# and nothing else here changes.
comptime LINEAR_MODEL_FORMAT_VERSION = 5

# The token that opens the optional serialized section, and the section's own
# revision. The section carries its own revision so a later change to what a
# leaf stores (a per-leaf scale, say) does not need another model-format bump.
comptime LINEAR_SECTION_TAG = "linear"
comptime LINEAR_SECTION_REVISION = 1

# Whether the feature is reachable from a public entry point. It is not, and
# `check_linear_tree_public` is what says so. Flipping this constant is not
# what makes the feature real; the handoff's patches are.
comptime LINEAR_TREE_PUBLIC = False

# A column whose in-leaf hessian-weighted variance is at or below this
# fraction of its mean-square is treated as constant on the leaf and dropped
# before the solve. Scale-free, so it means the same thing for a feature
# measured in metres and one measured in millimetres.
comptime LINEAR_MIN_VARIANCE = 1e-12

# Cholesky gives up on a pivot that has fallen to this fraction of the
# equilibrated diagonal it started at (which is 1 by construction). Below it
# the column is a linear combination of the ones already factored to within
# the precision Float64 can carry.
comptime LINEAR_PIVOT_TOL = 1e-11

# Reasons a leaf ended up constant. Reported by `LeafFitReport` so a caller
# (and later, inspection) can say *why* rather than only that it happened.
comptime LINEAR_FIT_OK = 0
comptime LINEAR_FIT_NO_FEATURES = 1
comptime LINEAR_FIT_RANK_DEFICIENT = 2
comptime LINEAR_FIT_NOT_FINITE = 3
comptime LINEAR_FIT_NO_IMPROVEMENT = 4
comptime LINEAR_FIT_DEVIATION_CAP = 5
comptime LINEAR_FIT_NO_WEIGHT = 6


def linear_fit_reason_name(reason: Int) -> String:
    """A short name for a `LINEAR_FIT_*` code, for error text and dumps."""
    if reason == LINEAR_FIT_OK:
        return String("ok")
    if reason == LINEAR_FIT_NO_FEATURES:
        return String("no_eligible_features")
    if reason == LINEAR_FIT_RANK_DEFICIENT:
        return String("rank_deficient")
    if reason == LINEAR_FIT_NOT_FINITE:
        return String("not_finite")
    if reason == LINEAR_FIT_NO_IMPROVEMENT:
        return String("no_improvement")
    if reason == LINEAR_FIT_DEVIATION_CAP:
        return String("deviation_cap")
    if reason == LINEAR_FIT_NO_WEIGHT:
        return String("no_hessian_weight")
    return String("unknown")


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


struct LinearParams(Copyable, Movable):
    """How leaves are fitted. The default is inactive: `enabled` is False, so
    a `LinearParams` that nobody configured describes today's constant-leaf
    model and `refit_linear_tree` returns an empty sidecar.

    - `enabled`: LightGBM's `linear_tree`.
    - `linear_lambda`: LightGBM's `linear_lambda`, the L2 on the
      coefficients. The intercept is not penalized by it; it keeps the
      `lambda_l1`/`lambda_l2` the grower already applied.
    - `max_leaf_features`: cap on how many features one leaf may use, -1 for
      no cap. mojotrees extension.
    - `min_data_per_linear_feature`: a leaf must hold at least this many rows
      per fitted feature. 0 is off. mojotrees extension.
    - `ridge_eps`: relative diagonal jitter, `eps * trace(A) / m`, so a
      singular system with `linear_lambda = 0` still factorizes. mojotrees
      extension.
    - `max_linear_deviation`: discard a fit whose linear part exceeds this in
      absolute value on any row of the leaf. 0.0 is off, and it is the only
      check that costs a second pass over the leaf's rows. mojotrees
      extension.
    """

    var enabled: Bool
    var linear_lambda: Float64
    var max_leaf_features: Int
    var min_data_per_linear_feature: Int
    var ridge_eps: Float64
    var max_linear_deviation: Float64

    def __init__(
        out self,
        enabled: Bool = False,
        linear_lambda: Float64 = 0.0,
        max_leaf_features: Int = -1,
        min_data_per_linear_feature: Int = 0,
        ridge_eps: Float64 = 1e-10,
        max_linear_deviation: Float64 = 0.0,
    ):
        self.enabled = enabled
        self.linear_lambda = linear_lambda
        self.max_leaf_features = max_leaf_features
        self.min_data_per_linear_feature = min_data_per_linear_feature
        self.ridge_eps = ridge_eps
        self.max_linear_deviation = max_linear_deviation

    @staticmethod
    def disabled() -> LinearParams:
        """Constant leaves: what every trainer uses today."""
        return LinearParams()

    def is_active(self) -> Bool:
        return self.enabled

    def check(self) raises:
        """Reject a configuration before any leaf is fitted, so a bad value is
        named once rather than found part way down a tree."""
        if not self.enabled:
            return
        if not isfinite(self.linear_lambda) or self.linear_lambda < 0.0:
            raise Error(
                "linear_lambda must be finite and nonnegative, got ",
                self.linear_lambda,
            )
        if self.max_leaf_features == 0 or self.max_leaf_features < -1:
            raise Error(
                "max_leaf_features must be positive, or -1 for no cap, got ",
                self.max_leaf_features,
            )
        if self.min_data_per_linear_feature < 0:
            raise Error(
                "min_data_per_linear_feature must not be negative, got ",
                self.min_data_per_linear_feature,
            )
        if not isfinite(self.ridge_eps) or self.ridge_eps < 0.0:
            raise Error(
                "ridge_eps must be finite and nonnegative, got ",
                self.ridge_eps,
            )
        if (
            not isfinite(self.max_linear_deviation)
            or self.max_linear_deviation < 0.0
        ):
            raise Error(
                "max_linear_deviation must be finite and nonnegative (0.0"
                " disables it), got ",
                self.max_linear_deviation,
            )


# ---------------------------------------------------------------------------
# The model representation
# ---------------------------------------------------------------------------


struct LinearLeaf(Copyable, Movable):
    """One leaf's affine function.

        leaf(x) = intercept + sum_j coef[j] * (x[feature[j]] - center[j])

    `feature` is ascending and holds numerical feature indices only. `coef`
    and `center` are the same length as `feature`. An empty `feature` is a
    constant leaf holding `intercept`, which is what a leaf that could not be
    fitted (or was never eligible) carries.

    `intercept` is the value `Tree.value[node]` already holds for this leaf,
    unshrunk: `Booster` applies `learning_rate` at prediction time rather
    than baking it into leaf values, so the coefficients are unshrunk too and
    the caller scales the whole affine function by one factor.

    `center[j]` is the in-leaf hessian-weighted mean of feature `feature[j]`
    over the rows the leaf was fitted from. It is stored, rather than folded
    into the intercept, for two reasons: it is the substitute a non-finite
    value takes at prediction time, and keeping it out of the intercept is
    what makes `intercept` equal to the constant-leaf value exactly.

    `reason` records why a constant leaf is constant (`LINEAR_FIT_*`). It is
    training metadata: it is not serialized and a loaded model reports
    `LINEAR_FIT_OK` for every leaf it actually carries coefficients for.
    """

    var intercept: Float64
    var feature: List[Int]
    var coef: List[Float64]
    var center: List[Float64]
    var improvement: Float64
    var reason: Int

    def __init__(
        out self,
        intercept: Float64,
        var feature: List[Int] = [],
        var coef: List[Float64] = [],
        var center: List[Float64] = [],
        improvement: Float64 = 0.0,
        reason: Int = LINEAR_FIT_OK,
    ):
        self.intercept = intercept
        self.feature = feature^
        self.coef = coef^
        self.center = center^
        self.improvement = improvement
        self.reason = reason

    @staticmethod
    def constant(value: Float64, reason: Int = LINEAR_FIT_OK) -> LinearLeaf:
        """A leaf that emits `value` for every row."""
        return LinearLeaf(value, reason=reason)

    def n_terms(self) -> Int:
        return len(self.feature)

    def is_linear(self) -> Bool:
        return len(self.feature) > 0

    def check_shape(self) raises:
        """Raise unless the three arrays agree and `feature` is ascending.
        Called once per leaf on load, so prediction never has to guard."""
        var m = len(self.feature)
        if len(self.coef) != m or len(self.center) != m:
            raise Error(
                "linear leaf: ",
                m,
                " features but ",
                len(self.coef),
                " coefficients and ",
                len(self.center),
                " centres",
            )
        for j in range(m):
            if self.feature[j] < 0:
                raise Error("linear leaf: negative feature index")
            if j > 0 and self.feature[j] <= self.feature[j - 1]:
                raise Error(
                    "linear leaf: feature indices must be strictly ascending"
                )
            if not isfinite(self.coef[j]) or not isfinite(self.center[j]):
                raise Error("linear leaf: non-finite coefficient or centre")
        if not isfinite(self.intercept):
            raise Error("linear leaf: non-finite intercept")

    def predict(self, raw_row: List[Float64]) -> Float64:
        """Evaluate on one raw example (length n_features, unbinned).

        A non-finite value on a fitted feature is replaced by that feature's
        centre, so its term contributes zero; see the module docstring. The
        substitution is on the *value*, not on the routing, which happened
        before this was reached.
        """
        var out = self.intercept
        for j in range(len(self.feature)):
            var v = raw_row[self.feature[j]]
            if not isfinite(v):
                continue
            out += self.coef[j] * (v - self.center[j])
        return out

    def predict_column_major(
        self, raw: List[Float64], n_rows: Int, row: Int
    ) -> Float64:
        """`predict` against a column-major matrix, `raw[f * n_rows + r]`,
        which is the layout `model.fit` and `BinMapper.transform` take."""
        var out = self.intercept
        for j in range(len(self.feature)):
            var v = raw[self.feature[j] * n_rows + row]
            if not isfinite(v):
                continue
            out += self.coef[j] * (v - self.center[j])
        return out

    def max_abs_coef(self) -> Float64:
        var m = 0.0
        for j in range(len(self.coef)):
            var a = abs(self.coef[j])
            if a > m:
                m = a
        return m


struct LinearTree(Copyable, Movable):
    """One tree's leaves, indexed by leaf ordinal.

    `leaf[o]` is the leaf whose ordinal is `o` in `Tree.leaf_ordinals()`, so
    `len(leaf)` is `Tree.n_leaves`. Ordinals rather than node ids: node ids
    number internal nodes and leaves together and shift as a tree grows,
    while ordinals are fixed once the tree is grown and survive save and load
    unchanged, which `tree.mojo` documents.

    An empty `leaf` list means this tree has constant leaves, which is how a
    sidecar covers an ensemble where only some trees were fitted linearly
    (continued training from a constant-leaf model, for instance).
    """

    var leaf: List[LinearLeaf]

    def __init__(out self, var leaf: List[LinearLeaf] = []):
        self.leaf = leaf^

    @staticmethod
    def constant() -> LinearTree:
        """A tree whose leaves are all constant: an empty entry."""
        return LinearTree()

    def is_active(self) -> Bool:
        return len(self.leaf) > 0

    def n_linear_leaves(self) -> Int:
        var n = 0
        for o in range(len(self.leaf)):
            if self.leaf[o].is_linear():
                n += 1
        return n

    def check_against(self, tree: Tree) raises:
        """Raise unless this entry describes `tree`: one leaf per ordinal, and
        every fitted feature within the tree's feature range."""
        if len(self.leaf) == 0:
            return
        if len(self.leaf) != tree.n_leaves:
            raise Error(
                "linear tree: ",
                len(self.leaf),
                " leaf entries for a tree with ",
                tree.n_leaves,
                " leaves",
            )
        for o in range(len(self.leaf)):
            self.leaf[o].check_shape()


struct LinearEnsemble(Copyable, Movable):
    """The whole sidecar: one `LinearTree` per tree of a `Booster`.

    `trees[i]` describes `Booster.trees[i]`, in the same order, which for a
    `MulticlassBooster` is round-major. `n_features` is the width of the raw
    matrix the leaves were fitted on, kept so that a loaded model can reject
    a raw row of the wrong shape before it indexes into it.

    An empty `trees` list is the inactive sidecar and is what every model
    carries today. `is_active` is the single test a consumer branches on.
    """

    var trees: List[LinearTree]
    var n_features: Int

    def __init__(
        out self, var trees: List[LinearTree] = [], n_features: Int = 0
    ):
        self.trees = trees^
        self.n_features = n_features

    @staticmethod
    def inactive() -> LinearEnsemble:
        """No linear leaves anywhere: the constant-leaf model."""
        return LinearEnsemble()

    def is_active(self) -> Bool:
        """Whether any tree carries linear leaves. This is the predicate every
        consumer that cannot evaluate a linear leaf must test before it
        assumes `Tree.value` is the whole story."""
        for i in range(len(self.trees)):
            if self.trees[i].is_active():
                return True
        return False

    def n_linear_leaves(self) -> Int:
        var n = 0
        for i in range(len(self.trees)):
            n += self.trees[i].n_linear_leaves()
        return n

    def has_entry(self, tree_index: Int) -> Bool:
        """Whether tree `tree_index` has linear leaves. False for an index
        past the sidecar, which is a model continued from a constant-leaf one
        and is not an error."""
        if tree_index < 0 or tree_index >= len(self.trees):
            return False
        return self.trees[tree_index].is_active()

    def entry(self, tree_index: Int) -> LinearTree:
        """A *copy* of tree `tree_index`'s entry, or a constant one when the
        sidecar is shorter than the ensemble.

        For a caller that wants one tree's leaves in hand. Prediction does
        not use it: copying a leaf list per tree per row costs more than the
        prediction, so `predict_tree_raw` indexes `trees` directly and
        `has_entry` is the cheap test."""
        if tree_index < 0 or tree_index >= len(self.trees):
            return LinearTree.constant()
        return self.trees[tree_index].copy()

    def scale(mut self, tree_index: Int, factor: Float64):
        """Multiply one tree's leaves by `factor`, intercept and coefficients
        together.

        This is what a per-tree weight has to do to a linear leaf. DART folds
        its drop weights into `Tree.value` (see
        `alternate_boosting.fold_weights_into_trees`), and folding the
        constants without the coefficients would leave a leaf whose affine
        function no longer passes through the value the tree carries: the
        sidecar and the tree would describe two different models, and
        `linear_leaf_reduces_to_constant` would start returning False.

        Scaling both keeps every invariant: the intercept still equals
        `Tree.value` after the tree is folded by the same factor, the centres
        are untouched (they are feature-space quantities, not output-space
        ones), and the leaf's function is exactly `factor` times what it
        was."""
        if tree_index < 0 or tree_index >= len(self.trees):
            return
        # A weight of exactly 1.0 is skipped rather than multiplied through,
        # so an untouched tree's leaves come back bit-identical, which is the
        # rule `fold_weights_into_trees` follows on the tree side.
        if factor == 1.0:
            return
        ref entry = self.trees[tree_index]
        for o in range(len(entry.leaf)):
            ref lf = entry.leaf[o]
            lf.intercept = lf.intercept * factor
            for j in range(len(lf.coef)):
                lf.coef[j] = lf.coef[j] * factor

    def scale_all(mut self, weights: List[Float64]) raises:
        """`scale` over the whole sidecar, one weight per tree. The
        counterpart of `alternate_boosting.fold_weights_into_trees`, and it
        must be called with the same weight vector, in the same call, or the
        two halves of the model disagree."""
        if len(self.trees) == 0:
            return
        if len(weights) != len(self.trees):
            raise Error(
                "linear sidecar: ",
                len(weights),
                " tree weights for ",
                len(self.trees),
                " trees",
            )
        for i in range(len(self.trees)):
            self.scale(i, weights[i])

    def check_against(self, trees: List[Tree], n_features: Int) raises:
        """Raise unless this sidecar describes `trees`. Called once on load
        and once before continued training, so nothing downstream guards."""
        if len(self.trees) == 0:
            return
        if len(self.trees) != len(trees):
            raise Error(
                "linear sidecar covers ",
                len(self.trees),
                " trees but the ensemble has ",
                len(trees),
            )
        if self.n_features != n_features:
            raise Error(
                "linear sidecar was fitted on ",
                self.n_features,
                " features but the model has ",
                n_features,
            )
        for i in range(len(self.trees)):
            self.trees[i].check_against(trees[i])
            for o in range(len(self.trees[i].leaf)):
                ref lf = self.trees[i].leaf[o]
                for j in range(len(lf.feature)):
                    if lf.feature[j] >= n_features:
                        raise Error(
                            "linear sidecar: tree ",
                            i,
                            " leaf ",
                            o,
                            " uses feature ",
                            lf.feature[j],
                            " but the model has ",
                            n_features,
                        )


def linear_leaf_reduces_to_constant(
    leaf: LinearLeaf, tree_value: Float64
) -> Bool:
    """Whether ignoring this leaf's coefficients gives a defined constant leaf
    with the value the tree already carries.

    True exactly when `intercept` is the tree's own leaf value, which is how
    `refit_linear_tree` builds every leaf. It is the property that lets a
    consumer which cannot evaluate a linear leaf (`contrib.mojo`,
    `gpu_predict.mojo`, the LightGBM exporter) state what it would be
    predicting instead, rather than discovering it. Consumers are still
    expected to raise; this exists so the choice is informed.
    """
    return leaf.intercept == tree_value


# ---------------------------------------------------------------------------
# Compatibility gates
# ---------------------------------------------------------------------------


def check_objective_compatible(objective: Int) raises:
    """Raise for an objective whose leaves cannot be linear.

    Leaf-renewing objectives (`QUANTILE`, `L1`, `MAPE`) replace the Newton
    value with a weighted residual percentile, which is not the minimizer of
    the quadratic the coefficients solve, so the intercept and the
    coefficients would no longer describe one fit. `LAMBDARANK`'s gradients
    come from query-group pairs and the combination is unvalidated. See the
    module docstring.
    """
    if _objective_renews_leaves(objective):
        var name = String("a leaf-renewing")
        if objective == _QUANTILE:
            name = String("QUANTILE")
        elif objective == _L1:
            name = String("L1")
        elif objective == _MAPE:
            name = String("MAPE")
        raise Error(
            "linear leaves are not available for the ",
            name,
            " objective: it renews every leaf after growth, replacing the"
            " Newton value with a weighted residual percentile, and that"
            " percentile is not the intercept the coefficient solve is"
            " decoupled from. Use a Newton objective, or leave linear_tree"
            " off",
        )
    if objective == _LAMBDARANK:
        raise Error(
            "linear leaves are not available for LAMBDARANK: its gradients"
            " come from query-group pairs, and a per-leaf regression on them"
            " is not a combination LightGBM documents or mojotrees has"
            " checked"
        )


def check_monotone_compatible(monotone: MonotoneConstraints) raises:
    """Raise when monotonic constraints are active.

    `monotone.mojo` proves monotonicity by clamping each leaf's *output* into
    an interval. An affine leaf has no single output to clamp, and requiring
    the function to lie in a bounded interval over an unbounded domain forces
    every coefficient to zero. The box-corner refinement that would make the
    two work together is stated in `docs/LINEAR_TREES.md` and is not built.
    """
    var active = False
    for f in range(len(monotone.signs)):
        if monotone.signs[f] != 0:
            active = True
            break
    if not active:
        return
    raise Error(
        "linear leaves and monotone_constraints cannot be used together:"
        " monotonicity is enforced by clamping each leaf's constant output"
        " into an interval, and an affine leaf has no constant output to"
        " clamp. Bounding the affine function over the leaf's box (which is"
        " what would work) needs bin edges at the leaf and per-feature box"
        " bounds carried down the frontier, neither of which the grower has."
        " Drop one of the two"
    )


def check_continuation_compatible(
    existing: LinearEnsemble,
    existing_trees: Int,
    params: LinearParams,
    n_features: Int,
) raises:
    """Raise unless more trees can be added to a model carrying `existing`.

    A `Booster` holds one sidecar for all of its trees, so an ensemble cannot
    change halfway from constant leaves to linear ones or back: the two
    describe different models and the resume pass would read the wrong one.
    """
    if existing.is_active() and not params.is_active():
        raise Error(
            "continued training cannot turn linear_tree off: the ensemble's"
            " existing trees carry per-leaf coefficients, and its raw scores"
            " are computed through them"
        )
    if (
        params.is_active()
        and len(existing.trees) > 0
        and not existing.is_active()
    ):
        raise Error(
            "continued training cannot turn linear_tree on: the ensemble's"
            " existing trees were fitted with constant leaves and their raw"
            " scores would be resumed from a different model than the one"
            " that produced them"
        )
    if not existing.is_active():
        return
    if len(existing.trees) != existing_trees:
        raise Error(
            "linear sidecar covers ",
            len(existing.trees),
            " trees but the ensemble has ",
            existing_trees,
            "; the model cannot be continued until they agree",
        )
    if existing.n_features != n_features:
        raise Error(
            "continued training must use the same feature width the linear"
            " leaves were fitted on: ",
            existing.n_features,
            " then, ",
            n_features,
            " now",
        )


def check_linear_tree_public() raises:
    """Raise: linear trees are not reachable from a public entry point.

    Called by anything that would expose the feature. The message names what
    is outstanding, because "not implemented" without a list is what makes a
    deferred subsystem look like a bug.
    """
    if LINEAR_TREE_PUBLIC:
        return
    raise Error(
        "linear trees are implemented but not connected. The algorithm core"
        " (linear_tree.mojo) fits, predicts, validates, and serializes"
        " per-leaf coefficients, but reaching it needs, in other lanes: a"
        " `linear` field on Booster and MulticlassBooster; the raw feature"
        " matrix kept alongside the binned one through training and"
        " continued training; the v4 `linear` section wired into"
        " serialize.mojo; a linear-aware raw-score pass in boosting.mojo;"
        " explicit refusals in contrib.mojo, gpu_predict.mojo, and"
        " lgbm_model_io.mojo; leaf_const/leaf_coeff in model_dump.mojo;"
        " LinearEnsemble.scale_all beside DART's weight folding in"
        " alternate_boosting.mojo; and the parameter accepted by params.mojo"
        " instead of refused. See handoffs/remaining_03_linear_trees.md"
    )


def linear_tree_available() -> Bool:
    """Whether a public caller can ask for linear trees. False, and it stays
    False until the handoff's patches land."""
    return LINEAR_TREE_PUBLIC


# ---------------------------------------------------------------------------
# Branch feature sets
# ---------------------------------------------------------------------------


def branch_features(tree: Tree) -> List[List[Int]]:
    """Per-node ascending list of the features split on between the root and
    that node. The root's is empty.

    LightGBM fits a leaf's regression on exactly this set, which is why it is
    computed from the tree rather than from a training-time frontier: a
    loaded model can be refitted, and the sidecar can be rebuilt, without the
    growth state that produced it.

    Both growers append children after their parent, so node ids increase
    down the tree and one ascending pass fills the table, the same argument
    `tree.node_bounds` relies on.
    """
    var n_nodes = len(tree.feature)
    var out = List[List[Int]](capacity=n_nodes)
    for _ in range(n_nodes):
        out.append(List[Int]())
    for node in range(n_nodes):
        var f = tree.feature[node]
        if f < 0:
            continue
        var extended = _insert_ascending(out[node], f)
        out[tree.left[node]] = extended.copy()
        out[tree.right[node]] = extended.copy()
    return out^


def _insert_ascending(base: List[Int], value: Int) -> List[Int]:
    """`base` with `value` inserted, keeping it ascending and duplicate-free.
    A feature split on twice down one path appears once."""
    var out = List[Int](capacity=len(base) + 1)
    var placed = False
    for i in range(len(base)):
        if not placed and base[i] == value:
            placed = True
        elif not placed and base[i] > value:
            out.append(value)
            placed = True
        out.append(base[i])
    if not placed:
        out.append(value)
    return out^


# ---------------------------------------------------------------------------
# Sufficient statistics
# ---------------------------------------------------------------------------


struct LeafStats(Copyable, Movable):
    """One leaf's centered second-order statistics over a candidate feature
    set.

    `feature[j]` is the j-th candidate. `center[j]` is its hessian-weighted
    mean over the leaf's rows. `a` is the m-by-m matrix of centered weighted
    second moments, row-major (`a[j * m + k]`), symmetric. `q[j]` is the
    gradient's projection onto the centered column. `sum_hess` and
    `sum_grad` are the leaf's totals, kept for the improvement report and for
    the caller that wants to check the intercept it was handed.

    `n_rows` is how many rows the leaf holds, which is what
    `min_data_per_linear_feature` is measured against.
    """

    var feature: List[Int]
    var center: List[Float64]
    var a: List[Float64]
    var q: List[Float64]
    var sum_hess: Float64
    var sum_grad: Float64
    var n_rows: Int

    def __init__(
        out self,
        var feature: List[Int] = [],
        var center: List[Float64] = [],
        var a: List[Float64] = [],
        var q: List[Float64] = [],
        sum_hess: Float64 = 0.0,
        sum_grad: Float64 = 0.0,
        n_rows: Int = 0,
    ):
        self.feature = feature^
        self.center = center^
        self.a = a^
        self.q = q^
        self.sum_hess = sum_hess
        self.sum_grad = sum_grad
        self.n_rows = n_rows

    def m(self) -> Int:
        return len(self.feature)

    def trace(self) -> Float64:
        var m = len(self.feature)
        var t = 0.0
        for j in range(m):
            t += self.a[j * m + j]
        return t


def eligible_leaf_features(
    candidates: List[Int],
    rows: List[Int],
    raw: List[Float64],
    n_rows_total: Int,
    cats: CategoricalSpec,
) -> List[Int]:
    """Trim a leaf's candidate features to the ones a linear term can use.

    Removes categorical features, and features with a non-finite raw value on
    any row of the leaf. Order is preserved, so an ascending `candidates`
    stays ascending. The variance filter and the cap are applied later, in
    `select_leaf_features`, because both need the statistics.

    `raw` is column-major over the *whole* training matrix
    (`raw[f * n_rows_total + r]`), which is the layout `model.fit` and
    `BinMapper.transform` take; `rows` indexes into it.
    """
    var out = List[Int](capacity=len(candidates))
    for i in range(len(candidates)):
        var f = candidates[i]
        if cats.is_cat(f):
            continue
        var base = f * n_rows_total
        var ok = True
        for k in range(len(rows)):
            if not isfinite(raw[base + rows[k]]):
                ok = False
                break
        if ok:
            out.append(f)
    return out^


def accumulate_leaf_stats(
    features: List[Int],
    rows: List[Int],
    raw: List[Float64],
    n_rows_total: Int,
    grad: List[Float64],
    hess: List[Float64],
) -> LeafStats:
    """Build one leaf's statistics over `features`, in two passes.

    Pass one takes the hessian-weighted means; pass two takes the centered
    second moments and the gradient projections. The one-pass
    `sum(xy) - sum(x) sum(y) / n` identity is algebraically the same and
    numerically much worse: a feature the tree has already split on twice has
    a small spread relative to its magnitude, which is exactly the case where
    cancellation eats the significant digits.

    `A` is filled on and above the diagonal and mirrored, so the matrix is
    exactly symmetric rather than symmetric up to summation order. That
    matters: Cholesky reads the lower triangle and an asymmetry of one ulp
    would make the factorization describe a different matrix than the
    improvement check scores.

    A leaf with no hessian weight returns zero statistics; the caller sees it
    through `sum_hess` and falls back to a constant.
    """
    var m = len(features)
    var stats = LeafStats()
    stats.n_rows = len(rows)

    var sum_h = 0.0
    var sum_g = 0.0
    for k in range(len(rows)):
        sum_h += hess[rows[k]]
        sum_g += grad[rows[k]]
    stats.sum_hess = sum_h
    stats.sum_grad = sum_g
    if m == 0 or not (sum_h > 0.0) or not isfinite(sum_h):
        return stats^

    stats.feature = features.copy()
    stats.center = List[Float64](capacity=m)
    stats.center.resize(m, 0.0)
    stats.a = List[Float64](capacity=m * m)
    stats.a.resize(m * m, 0.0)
    stats.q = List[Float64](capacity=m)
    stats.q.resize(m, 0.0)

    # Pass one: hessian-weighted means.
    for j in range(m):
        var base = features[j] * n_rows_total
        var acc = 0.0
        for k in range(len(rows)):
            var r = rows[k]
            acc += hess[r] * raw[base + r]
        stats.center[j] = acc / sum_h

    # Pass two: centered moments. One row at a time so each row's centered
    # values are computed once and reused across the m * (m + 1) / 2 products.
    var x = List[Float64](capacity=m)
    x.resize(m, 0.0)
    for k in range(len(rows)):
        var r = rows[k]
        var h = hess[r]
        var g = grad[r]
        for j in range(m):
            x[j] = raw[features[j] * n_rows_total + r] - stats.center[j]
        for j in range(m):
            stats.q[j] += g * x[j]
            var hx = h * x[j]
            for kk in range(j, m):
                stats.a[j * m + kk] += hx * x[kk]
    for j in range(m):
        for kk in range(j + 1, m):
            stats.a[kk * m + j] = stats.a[j * m + kk]
    return stats^


def select_leaf_features(
    stats: LeafStats, params: LinearParams
) -> List[Int]:
    """Which of `stats.feature` survive the variance filter, the cap, and the
    rows-per-feature floor. Returns positions into `stats.feature`, ascending.

    Ranking is by the univariate objective reduction
    `q[j]^2 / (A[j][j] + linear_lambda)`, which is what a leaf with that one
    feature would gain, descending. Ties go to the lower feature index, so
    the selection is a function of the statistics and nothing else: two runs
    on the same data pick the same features, whatever order the rows were
    accumulated in.
    """
    var m = stats.m()
    var keep = List[Int](capacity=m)
    if m == 0:
        return keep^

    # Variance filter. `A[j][j]` is `sum_r h_r (x - mean)^2`; comparing it to
    # `sum_hess * mean^2` makes the test scale-free. A feature centred exactly
    # on zero has no scale to compare against, so it is judged against its own
    # magnitude instead, which for an all-zero column is zero and drops it.
    var scores = List[Float64](capacity=m)
    scores.resize(m, 0.0)
    for j in range(m):
        var vjj = stats.a[j * m + j]
        var scale = stats.sum_hess * stats.center[j] * stats.center[j]
        if scale <= 0.0:
            scale = vjj
        if not (vjj > 0.0) or not isfinite(vjj):
            scores[j] = -1.0
            continue
        if scale > 0.0 and vjj <= LINEAR_MIN_VARIANCE * scale:
            scores[j] = -1.0
            continue
        var denom = vjj + params.linear_lambda
        if not (denom > 0.0):
            scores[j] = -1.0
            continue
        scores[j] = stats.q[j] * stats.q[j] / denom

    var order = List[Int](capacity=m)
    for j in range(m):
        if scores[j] >= 0.0:
            order.append(j)
    # Insertion sort: m is the branch depth of one leaf, single digits in
    # practice, and a stable hand-written sort keeps the tie rule explicit.
    for i in range(1, len(order)):
        var v = order[i]
        var j = i - 1
        while j >= 0 and (
            scores[order[j]] < scores[v]
            or (scores[order[j]] == scores[v] and order[j] > v)
        ):
            order[j + 1] = order[j]
            j -= 1
        order[j + 1] = v

    var limit = len(order)
    if params.max_leaf_features > 0 and params.max_leaf_features < limit:
        limit = params.max_leaf_features
    if params.min_data_per_linear_feature > 0:
        var affordable = stats.n_rows // params.min_data_per_linear_feature
        if affordable < limit:
            limit = affordable
    if limit < 0:
        limit = 0

    for i in range(limit):
        keep.append(order[i])
    # Back to ascending feature order, which is what `LinearLeaf` stores.
    for i in range(1, len(keep)):
        var v = keep[i]
        var j = i - 1
        while j >= 0 and keep[j] > v:
            keep[j + 1] = keep[j]
            j -= 1
        keep[j + 1] = v
    return keep^


# ---------------------------------------------------------------------------
# The solve
# ---------------------------------------------------------------------------


struct LinearSolution(Copyable, Movable):
    """The outcome of one leaf's ridge solve.

    `active` holds positions into the statistics' feature list that survived,
    ascending; `coef[i]` is the coefficient for `active[i]`. `improvement` is
    the training-objective reduction the coefficients buy over the same leaf
    with `c = 0`, computed from the returned coefficients rather than from
    the closed form, so roundoff that made the fit worse is visible.
    `reason` is a `LINEAR_FIT_*` code and is `LINEAR_FIT_OK` only when
    `active` is non-empty and the improvement is positive and finite.
    """

    var active: List[Int]
    var coef: List[Float64]
    var improvement: Float64
    var reason: Int

    def __init__(
        out self,
        var active: List[Int] = [],
        var coef: List[Float64] = [],
        improvement: Float64 = 0.0,
        reason: Int = LINEAR_FIT_OK,
    ):
        self.active = active^
        self.coef = coef^
        self.improvement = improvement
        self.reason = reason

    @staticmethod
    def failed(reason: Int) -> LinearSolution:
        return LinearSolution(reason=reason)

    def ok(self) -> Bool:
        return self.reason == LINEAR_FIT_OK and len(self.active) > 0


def _drop_at(active: List[Int], index: Int) -> List[Int]:
    """`active` without its `index`-th entry, order preserved.

    The solve's retry path needs exactly this and nothing else. Written out
    rather than reached for as an indexed `pop`, so the one operation the
    termination argument rests on -- every retry strictly shrinks the active
    set -- is visible at the call site.
    """
    var out = List[Int](capacity=len(active))
    for i in range(len(active)):
        if i != index:
            out.append(active[i])
    return out^


def _cholesky_in_place(mut a: List[Float64], n: Int, tol: Float64) -> Int:
    """Lower Cholesky of an n-by-n symmetric matrix, in place in the lower
    triangle. Returns -1 on success, or the index of the first column whose
    pivot fell to or below `tol`.

    The caller equilibrates first, so every diagonal starts at 1 and `tol` is
    an absolute test that means the same thing for every column.
    """
    for j in range(n):
        var d = a[j * n + j]
        for k in range(j):
            var l = a[j * n + k]
            d -= l * l
        if not (d > tol) or not isfinite(d):
            return j
        var root = sqrt(d)
        a[j * n + j] = root
        for i in range(j + 1, n):
            var s = a[i * n + j]
            for k in range(j):
                s -= a[i * n + k] * a[j * n + k]
            a[i * n + j] = s / root
    return -1


def _cholesky_solve(l: List[Float64], n: Int, mut b: List[Float64]):
    """Solve `L L^T x = b` in place in `b`, given the lower factor from
    `_cholesky_in_place`."""
    for i in range(n):
        var s = b[i]
        for k in range(i):
            s -= l[i * n + k] * b[k]
        b[i] = s / l[i * n + i]
    for ii in range(n):
        var i = n - 1 - ii
        var s = b[i]
        for k in range(i + 1, n):
            s -= l[k * n + i] * b[k]
        b[i] = s / l[i * n + i]


def solve_leaf_coefficients(
    stats: LeafStats, subset: List[Int], params: LinearParams
) -> LinearSolution:
    """Solve `(A + linear_lambda I) c = -q` over `subset`, dropping features
    until it factorizes.

    `subset` holds positions into `stats.feature`. The returned `active` is a
    subset of it, in the same ascending order.

    The system is built over the active set, given the ridge term and the
    relative jitter, then symmetrically equilibrated by
    `D = diag(1 / sqrt(M_jj))` so the factorization sees unit diagonals. The
    equilibration is exact and is undone on the way out; it is applied after
    the ridge so `linear_lambda` keeps its meaning in the feature's own
    scale.

    On a failed pivot the offending feature is removed and the whole build
    repeats. On a non-finite coefficient the largest-magnitude one is
    removed. Every retry strictly shrinks the active set, so this terminates
    in at most `len(subset)` rounds.
    """
    var m = stats.m()
    var active = subset.copy()
    if len(active) == 0:
        return LinearSolution.failed(LINEAR_FIT_NO_FEATURES)
    if not (stats.sum_hess > 0.0):
        return LinearSolution.failed(LINEAR_FIT_NO_WEIGHT)

    # `ridge_eps` is relative to the average diagonal of the full candidate
    # matrix, not of the active one, so dropping a feature does not change
    # the jitter the survivors see.
    var jitter = 0.0
    if params.ridge_eps > 0.0 and m > 0:
        var t = stats.trace()
        if t > 0.0 and isfinite(t):
            jitter = params.ridge_eps * t / Float64(m)

    while len(active) > 0:
        var n = len(active)
        var mat = List[Float64](capacity=n * n)
        mat.resize(n * n, 0.0)
        var rhs = List[Float64](capacity=n)
        rhs.resize(n, 0.0)
        var scale = List[Float64](capacity=n)
        scale.resize(n, 0.0)

        for i in range(n):
            for j in range(n):
                mat[i * n + j] = stats.a[active[i] * m + active[j]]
            mat[i * n + i] += params.linear_lambda + jitter
            rhs[i] = -stats.q[active[i]]

        var bad = -1
        for i in range(n):
            var d = mat[i * n + i]
            if not (d > 0.0) or not isfinite(d):
                bad = i
                break
            scale[i] = 1.0 / sqrt(d)
        if bad >= 0:
            active = _drop_at(active, bad)
            continue

        for i in range(n):
            for j in range(n):
                mat[i * n + j] *= scale[i] * scale[j]
            rhs[i] *= scale[i]

        var failed = _cholesky_in_place(mat, n, LINEAR_PIVOT_TOL)
        if failed >= 0:
            active = _drop_at(active, failed)
            continue

        _cholesky_solve(mat, n, rhs)

        var coef = List[Float64](capacity=n)
        coef.resize(n, 0.0)
        var worst = -1
        var worst_mag = -1.0
        for i in range(n):
            var c = rhs[i] * scale[i]
            coef[i] = c
            if not isfinite(c):
                var mag = abs(rhs[i])
                if not isfinite(mag) or mag > worst_mag:
                    worst = i
                    worst_mag = mag
        if worst >= 0:
            active = _drop_at(active, worst)
            continue

        # The improvement, from the coefficients that actually came back:
        #   -( q.c + c^T A c / 2 + linear_lambda ||c||^2 / 2 )
        # In exact arithmetic this is `c^T (A + lambda I) c / 2 >= 0`. Scoring
        # the returned `c` instead catches a solve that lost too much to
        # roundoff to be worth keeping.
        var qc = 0.0
        var cac = 0.0
        var cc = 0.0
        for i in range(n):
            qc += stats.q[active[i]] * coef[i]
            cc += coef[i] * coef[i]
            for j in range(n):
                cac += coef[i] * stats.a[active[i] * m + active[j]] * coef[j]
        var improvement = -(
            qc + 0.5 * cac + 0.5 * params.linear_lambda * cc
        )
        if not isfinite(improvement):
            return LinearSolution.failed(LINEAR_FIT_NOT_FINITE)
        if not (improvement > 0.0):
            return LinearSolution.failed(LINEAR_FIT_NO_IMPROVEMENT)
        return LinearSolution(active^, coef^, improvement, LINEAR_FIT_OK)

    return LinearSolution.failed(LINEAR_FIT_RANK_DEFICIENT)


# ---------------------------------------------------------------------------
# Fitting one leaf
# ---------------------------------------------------------------------------


def fit_leaf(
    intercept: Float64,
    candidates: List[Int],
    rows: List[Int],
    raw: List[Float64],
    n_rows_total: Int,
    grad: List[Float64],
    hess: List[Float64],
    cats: CategoricalSpec,
    params: LinearParams,
) -> LinearLeaf:
    """Fit one leaf's affine function, or return the constant it was handed.

    `intercept` is the value the grower already wrote into `Tree.value` for
    this leaf, unshrunk and after `max_delta_step`, `path_smooth`, and any
    monotone clamp. It is never recomputed: the coefficient solve is
    decoupled from it (see the module docstring), so keeping it is both
    correct and what makes an all-zero coefficient vector reproduce today's
    leaf exactly.

    Every failure path returns a constant leaf carrying the reason, so a
    caller never has to distinguish "not fitted" from "fitted to zero".
    """
    if not params.is_active():
        return LinearLeaf.constant(intercept)
    if len(rows) == 0:
        return LinearLeaf.constant(intercept, LINEAR_FIT_NO_WEIGHT)

    var eligible = eligible_leaf_features(
        candidates, rows, raw, n_rows_total, cats
    )
    if len(eligible) == 0:
        return LinearLeaf.constant(intercept, LINEAR_FIT_NO_FEATURES)

    var stats = accumulate_leaf_stats(
        eligible, rows, raw, n_rows_total, grad, hess
    )
    if not (stats.sum_hess > 0.0) or stats.m() == 0:
        return LinearLeaf.constant(intercept, LINEAR_FIT_NO_WEIGHT)

    var chosen = select_leaf_features(stats, params)
    if len(chosen) == 0:
        return LinearLeaf.constant(intercept, LINEAR_FIT_NO_FEATURES)

    var solution = solve_leaf_coefficients(stats, chosen, params)
    if not solution.ok():
        return LinearLeaf.constant(intercept, solution.reason)

    var n = len(solution.active)
    var feature = List[Int](capacity=n)
    var coef = List[Float64](capacity=n)
    var center = List[Float64](capacity=n)
    for i in range(n):
        feature.append(stats.feature[solution.active[i]])
        coef.append(solution.coef[i])
        center.append(stats.center[solution.active[i]])

    var leaf = LinearLeaf(
        intercept,
        feature^,
        coef^,
        center^,
        solution.improvement,
        LINEAR_FIT_OK,
    )

    # The one check that costs a pass over the rows, and the only one that is
    # about the fitted function rather than about the system that produced
    # it: a leaf whose linear part swings further than the caller will accept
    # is discarded whole rather than clipped, because clipping would make the
    # stored coefficients describe a function the leaf does not evaluate.
    if params.max_linear_deviation > 0.0:
        for k in range(len(rows)):
            var dev = leaf.predict_column_major(raw, n_rows_total, rows[k])
            dev -= leaf.intercept
            if abs(dev) > params.max_linear_deviation:
                return LinearLeaf.constant(
                    intercept, LINEAR_FIT_DEVIATION_CAP
                )

    return leaf^


# ---------------------------------------------------------------------------
# Fitting a whole tree and a whole ensemble
# ---------------------------------------------------------------------------


def leaf_row_lists(
    tree: Tree, data: BinnedMatrix, rows: List[Int]
) raises -> List[List[Int]]:
    """Group `rows` by the leaf ordinal each reaches in `tree`.

    `rows` is the row set the tree was grown on: the bag under bagging or
    GOSS, every row otherwise. Fitting from exactly those rows is what keeps
    the coefficients consistent with `Tree.count`, with the constant leaf
    value they sit beside, and with the gains the splits were chosen on. An
    empty `rows` means every row of `data`.

    Routing goes through `Tree.leaf_index_row`, so missing values follow
    `default_left` and categorical nodes follow their set, exactly as in
    prediction. Rows keep their ascending order within a leaf, which makes
    the accumulation order (and so the last bits of the statistics) a
    function of the data alone.
    """
    var out = List[List[Int]](capacity=tree.n_leaves)
    for _ in range(tree.n_leaves):
        out.append(List[Int]())
    var ordinals = tree.leaf_ordinals()
    if len(rows) == 0:
        for r in range(data.n_rows):
            var o = ordinals[tree.leaf_index_row(data, r)]
            out[o].append(r)
        return out^
    for k in range(len(rows)):
        var r = rows[k]
        if r < 0 or r >= data.n_rows:
            raise Error("linear leaf fit: row index out of range")
        var o = ordinals[tree.leaf_index_row(data, r)]
        out[o].append(r)
    return out^


def refit_linear_tree(
    tree: Tree,
    data: BinnedMatrix,
    raw: List[Float64],
    grad: List[Float64],
    hess: List[Float64],
    params: LinearParams,
    rows: List[Int] = [],
) raises -> LinearTree:
    """Replace every leaf of one grown tree with an affine function.

    The tree itself is not modified: routing, split gains, and node covers
    are what growth produced, and `Tree.value` keeps the constants that
    become the leaves' intercepts. Only the sidecar is built.

    `raw` is the column-major raw feature matrix
    (`raw[f * data.n_rows + r]`), unbinned. `grad` and `hess` are the round's
    per-row derivatives, the same arrays growth was handed. `rows` is the row
    set the tree was grown on; empty means all of them.

    Returns a constant entry (`LinearTree.constant()`) when `params` is
    inactive, so a caller can hand this the parameters unconditionally.
    """
    if not params.is_active():
        return LinearTree.constant()
    params.check()
    if len(raw) != data.n_rows * data.n_features:
        raise Error(
            "linear leaf fit: raw matrix has ",
            len(raw),
            " values for ",
            data.n_rows,
            " rows and ",
            data.n_features,
            " features",
        )
    if len(grad) != data.n_rows or len(hess) != data.n_rows:
        raise Error(
            "linear leaf fit: gradient and hessian must have one entry per"
            " row"
        )

    var branches = branch_features(tree)
    var groups = leaf_row_lists(tree, data, rows)
    var ordinals = tree.leaf_ordinals()

    var leaves = List[LinearLeaf](capacity=tree.n_leaves)
    for _ in range(tree.n_leaves):
        leaves.append(LinearLeaf.constant(0.0))
    for node in range(len(tree.feature)):
        if tree.feature[node] >= 0:
            continue
        var o = ordinals[node]
        leaves[o] = fit_leaf(
            tree.value[node],
            branches[node],
            groups[o],
            raw,
            data.n_rows,
            grad,
            hess,
            data.cats,
            params,
        )
    return LinearTree(leaves^)


def refit_linear_ensemble(
    trees: List[Tree],
    data: BinnedMatrix,
    raw: List[Float64],
    grad: List[Float64],
    hess: List[Float64],
    params: LinearParams,
) raises -> LinearEnsemble:
    """`refit_linear_tree` over a whole ensemble, from one round's gradients.

    This is the *diagnostic* entry point, not the training one. Every tree of
    a boosted ensemble was fitted to its own round's residuals, so refitting
    them all from a single `grad`/`hess` pair reproduces the trained model
    only for a one-tree ensemble. Training builds the sidecar one tree at a
    time, in the boosting loop, which is what the handoff's `boosting.mojo`
    patch does.
    """
    if not params.is_active():
        return LinearEnsemble.inactive()
    var out = List[LinearTree](capacity=len(trees))
    for i in range(len(trees)):
        out.append(
            refit_linear_tree(trees[i], data, raw, grad, hess, params)
        )
    return LinearEnsemble(out^, data.n_features)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict_tree_raw(
    tree: Tree,
    linear: LinearEnsemble,
    tree_index: Int,
    bins: List[Int],
    raw_row: List[Float64],
) -> Float64:
    """One tree's unshrunk output for one example.

    Routing reads the binned example, the leaf reads the raw one: the two
    describe the same row and both are needed, which is the whole reason a
    linear predictor cannot be handed a `BinnedMatrix` alone. A tree with no
    sidecar entry falls through to `Tree.value`, so a mixed ensemble costs
    nothing on its constant trees.

    The sidecar is indexed here rather than handed in by value: a
    `LinearTree` owns a list of leaves, and copying one per tree per row --
    which returning it from `LinearEnsemble.entry` would do -- costs more
    than the whole prediction. `predict_tree_raw_at` is the variant for a
    caller that already holds the leaf-ordinal table.

    The caller multiplies by `learning_rate`; `Booster` applies shrinkage at
    prediction time rather than baking it into leaf values, so coefficients
    and intercept scale together by construction.
    """
    var node = 0
    while tree.feature[node] >= 0:
        if tree.goes_left(node, bins[tree.feature[node]]):
            node = tree.left[node]
        else:
            node = tree.right[node]
    if tree_index < 0 or tree_index >= len(linear.trees):
        return tree.value[node]
    if not linear.trees[tree_index].is_active():
        return tree.value[node]
    var ordinal = 0
    for i in range(node):
        if tree.feature[i] < 0:
            ordinal += 1
    return linear.trees[tree_index].leaf[ordinal].predict(raw_row)


def predict_tree_raw_at(
    tree: Tree,
    linear: LinearEnsemble,
    tree_index: Int,
    ordinals: List[Int],
    bins: List[Int],
    raw_row: List[Float64],
) -> Float64:
    """`predict_tree_raw` given this tree's precomputed
    `Tree.leaf_ordinals()` table.

    The per-row form counts the leaves before the node it landed on, which is
    O(nodes) on top of an O(depth) walk. Batched prediction builds the table
    once per tree instead; the two return the same number by construction,
    since the table is the same count.
    """
    var node = 0
    while tree.feature[node] >= 0:
        if tree.goes_left(node, bins[tree.feature[node]]):
            node = tree.left[node]
        else:
            node = tree.right[node]
    if tree_index < 0 or tree_index >= len(linear.trees):
        return tree.value[node]
    if not linear.trees[tree_index].is_active():
        return tree.value[node]
    return linear.trees[tree_index].leaf[ordinals[node]].predict(raw_row)


def predict_ensemble_raw(
    trees: List[Tree],
    linear: LinearEnsemble,
    base_score: Float64,
    learning_rate: Float64,
    bins: List[Int],
    raw_row: List[Float64],
    start: Int = 0,
    stop: Int = -1,
) -> Float64:
    """A single-output ensemble's raw score for one example, through the
    linear leaves.

    `start` and `stop` are a half-open tree range, `stop < 0` meaning every
    remaining tree; the base score is included exactly when `start` is 0,
    which is `IterationRange`'s rule. This is the pass continued training has
    to resume from: resuming from the constant fallback would fit every later
    tree to a residual the ensemble does not actually leave behind.
    """
    var hi = len(trees) if stop < 0 else stop
    if hi > len(trees):
        hi = len(trees)
    var s = base_score if start == 0 else 0.0
    for i in range(start, hi):
        s += learning_rate * predict_tree_raw(
            trees[i], linear, i, bins, raw_row
        )
    return s


def predict_multiclass_raw(
    trees: List[Tree],
    linear: LinearEnsemble,
    base_scores: List[Float64],
    learning_rate: Float64,
    n_classes: Int,
    bins: List[Int],
    raw_row: List[Float64],
    start_iteration: Int = 0,
    stop_iteration: Int = -1,
) -> List[Float64]:
    """Per-class raw scores for one example, through the linear leaves.

    Trees are round-major, so iteration `i`'s class `k` is at
    `i * n_classes + k` in both `trees` and the sidecar. The base scores are
    included exactly when `start_iteration` is 0.

    The per-class raw scores are piecewise affine in the raw features. The
    softmax probabilities are not, for the same reason they are not monotone
    under monotone constraints: each depends on every class's score.
    """
    var n_iter = len(trees) // n_classes if n_classes > 0 else 0
    var hi = n_iter if stop_iteration < 0 else stop_iteration
    if hi > n_iter:
        hi = n_iter
    var out = List[Float64](capacity=n_classes)
    for k in range(n_classes):
        out.append(base_scores[k] if start_iteration == 0 else 0.0)
    for i in range(start_iteration, hi):
        for k in range(n_classes):
            var t = i * n_classes + k
            out[k] += learning_rate * predict_tree_raw(
                trees[t], linear, t, bins, raw_row
            )
    return out^


def predict_batch_raw(
    trees: List[Tree],
    linear: LinearEnsemble,
    base_score: Float64,
    learning_rate: Float64,
    data: BinnedMatrix,
    raw: List[Float64],
    start: Int = 0,
    stop: Int = -1,
) raises -> List[Float64]:
    """Batched raw scores over a whole binned matrix and its raw counterpart.

    Two things are hoisted out of the row loop that the per-row entry point
    cannot hoist: the bin and raw vectors are refilled in place rather than
    reallocated, and each tree's leaf-ordinal table is built once instead of
    being recounted per row. The arithmetic is identical, tree by tree and
    term by term, to `predict_ensemble_raw` on the same row.
    """
    if len(raw) != data.n_rows * data.n_features:
        raise Error(
            "linear prediction: raw matrix has ",
            len(raw),
            " values for ",
            data.n_rows,
            " rows and ",
            data.n_features,
            " features",
        )
    var hi = len(trees) if stop < 0 else stop
    if hi > len(trees):
        hi = len(trees)
    var lo = start
    if lo < 0:
        lo = 0

    # One ordinal table per tree in the range, built once. A tree with no
    # sidecar entry gets an empty table: `predict_tree_raw_at` never reads it,
    # because it returns the constant leaf value first.
    var n_tables = hi - lo
    if n_tables < 0:
        n_tables = 0
    var tables = List[List[Int]](capacity=n_tables)
    for i in range(lo, hi):
        if i < len(linear.trees) and linear.trees[i].is_active():
            tables.append(trees[i].leaf_ordinals())
        else:
            tables.append(List[Int]())

    var out = List[Float64](capacity=data.n_rows)
    var bins = List[Int](capacity=data.n_features)
    bins.resize(data.n_features, 0)
    var row = List[Float64](capacity=data.n_features)
    row.resize(data.n_features, 0.0)
    for r in range(data.n_rows):
        for f in range(data.n_features):
            bins[f] = data.bin_at(r, f)
            row[f] = raw[f * data.n_rows + r]
        var s = base_score if lo == 0 else 0.0
        for i in range(lo, hi):
            s += learning_rate * predict_tree_raw_at(
                trees[i], linear, i, tables[i - lo], bins, row
            )
        out.append(s)
    return out^


def resume_raw_scores(
    trees: List[Tree],
    linear: LinearEnsemble,
    base_score: Float64,
    learning_rate: Float64,
    data: BinnedMatrix,
    raw: List[Float64],
    init_score: List[Float64] = [],
) raises -> List[Float64]:
    """The per-row raw scores continued training resumes from.

    `boosting.train_more` recomputes these from the model on every call; the
    linear version has to add `init_score` the same way, since it is training
    state the ensemble does not carry. This is the function the handoff's
    `train_more` patch substitutes for the constant-leaf resume loop.
    """
    if len(init_score) != 0 and len(init_score) != data.n_rows:
        raise Error(
            "init_score must be empty or have one entry per row"
        )
    var out = predict_batch_raw(
        trees, linear, base_score, learning_rate, data, raw
    )
    if len(init_score) == data.n_rows:
        for r in range(data.n_rows):
            out[r] += init_score[r]
    return out^


# ---------------------------------------------------------------------------
# Inspection data
# ---------------------------------------------------------------------------


def linear_feature_use_counts(
    linear: LinearEnsemble, n_features: Int
) -> List[Int]:
    """How many leaves in the whole ensemble carry a coefficient on each
    feature.

    A companion to split-count importance, not a replacement: a feature can
    reach a leaf's regression only if it was split on above that leaf, so
    this is always dominated by the split counts. It answers a question the
    split counts cannot, which is where the model is actually using a feature
    continuously rather than as a threshold.
    """
    var out = List[Int](capacity=n_features)
    out.resize(n_features, 0)
    for i in range(len(linear.trees)):
        ref entry = linear.trees[i]
        for o in range(len(entry.leaf)):
            ref lf = entry.leaf[o]
            for j in range(len(lf.feature)):
                if lf.feature[j] < n_features:
                    out[lf.feature[j]] += 1
    return out^


def linear_coefficient_l1(
    linear: LinearEnsemble, n_features: Int
) -> List[Float64]:
    """Summed absolute coefficient per feature over the whole ensemble.

    Comparable across features only when the features are, since a
    coefficient carries the reciprocal of its feature's unit. Reported as
    data rather than as an importance ranking for that reason.
    """
    var out = List[Float64](capacity=n_features)
    out.resize(n_features, 0.0)
    for i in range(len(linear.trees)):
        ref entry = linear.trees[i]
        for o in range(len(entry.leaf)):
            ref lf = entry.leaf[o]
            for j in range(len(lf.feature)):
                if lf.feature[j] < n_features:
                    out[lf.feature[j]] += abs(lf.coef[j])
    return out^


def linear_fit_summary(linear: LinearEnsemble) -> List[Int]:
    """Counts per `LINEAR_FIT_*` reason across every leaf of the sidecar,
    indexed by the reason code.

    Training diagnostics: a run whose leaves are mostly `rank_deficient` or
    `no_eligible_features` has a data problem (constant columns, missing
    values on the split features) rather than a modelling one, and the
    difference is invisible from the fitted model alone.
    """
    var out = List[Int](capacity=LINEAR_FIT_NO_WEIGHT + 1)
    out.resize(LINEAR_FIT_NO_WEIGHT + 1, 0)
    for i in range(len(linear.trees)):
        ref entry = linear.trees[i]
        for o in range(len(entry.leaf)):
            var reason = entry.leaf[o].reason
            if reason >= 0 and reason < len(out):
                out[reason] += 1
    return out^


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------
#
# The section is written and read here, in `serialize.mojo`'s own encoding
# (floats as decimal UInt64 bit patterns, whitespace-separated tokens), so
# the cross-lane change is one call in each direction rather than a second
# parser. See the handoff for the exact patch.
#
# Layout, after the trees:
#
#     linear <revision> <n_trees> <n_features>
#     ltree <n_leaves> <n_linear_leaves>              (once per tree)
#     <ordinal> <m> <feature...> <coef...> <center...>  (per linear leaf)
#
# Constant leaves are not written: a tree with three constant leaves and one
# linear one writes `ltree 4 1` and a single leaf record. A model with no
# linear leaves writes no section at all, so a v4 file for a constant-leaf
# model is byte for byte what v3 wrote.


def _f64_token(x: Float64) -> String:
    return String(x.to_bits())


def _parse_u64_token(token: String) raises -> UInt64:
    if token.byte_length() == 0:
        raise Error("linear section: empty token where integer expected")
    comptime _U64_MAX = ~UInt64(0)
    var out: UInt64 = 0
    for b in token.as_bytes():
        if b < 48 or b > 57:
            raise Error("linear section: invalid digit in integer token")
        # Coefficients arrive as float bit patterns, so every u64 is a legal
        # value and the overflow check has to be exact rather than a digit
        # cap. Without it a long digit run wraps into an arbitrary
        # coefficient instead of raising.
        var digit = UInt64(Int(b) - 48)
        if out > (_U64_MAX - digit) // 10:
            raise Error(
                "linear section: integer token does not fit in 64 bits: "
                + String(token)
            )
        out = out * 10 + digit
    return out


def _parse_f64_token(token: String) raises -> Float64:
    return bitcast[DType.float64, 1](
        SIMD[DType.uint64, 1](_parse_u64_token(token))
    )


struct LinearSectionResult(Copyable, Movable):
    """What `read_linear_section` returns: the sidecar and where the reader
    should continue. `next_pos` is the token index just past the section, and
    equals the position it was given when there was no section to read."""

    var linear: LinearEnsemble
    var next_pos: Int

    def __init__(out self, var linear: LinearEnsemble, next_pos: Int):
        self.linear = linear^
        self.next_pos = next_pos


def linear_section_text(linear: LinearEnsemble) -> String:
    """The serialized `linear` section, or an empty string when there is
    nothing to write.

    An inactive sidecar writes nothing, which is what keeps a v4 file for a
    constant-leaf model identical to the v3 one. Callers append the result
    unconditionally.
    """
    if not linear.is_active():
        return String("")
    var out = String("")
    out += LINEAR_SECTION_TAG
    out += " " + String(LINEAR_SECTION_REVISION)
    out += " " + String(len(linear.trees))
    out += " " + String(linear.n_features) + "\n"
    for i in range(len(linear.trees)):
        ref entry = linear.trees[i]
        out += (
            "ltree "
            + String(len(entry.leaf))
            + " "
            + String(entry.n_linear_leaves())
            + "\n"
        )
        for o in range(len(entry.leaf)):
            ref lf = entry.leaf[o]
            if not lf.is_linear():
                continue
            var m = len(lf.feature)
            out += String(o) + " " + String(m)
            for j in range(m):
                out += " " + String(lf.feature[j])
            for j in range(m):
                out += " " + _f64_token(lf.coef[j])
            for j in range(m):
                out += " " + _f64_token(lf.center[j])
            out += "\n"
    return out^


def read_linear_section(
    tokens: List[String],
    pos: Int,
    trees: List[Tree],
    n_features: Int,
) raises -> LinearSectionResult:
    """Read the optional `linear` section starting at token `pos`.

    A file without it (every v1, v2, and v3 model, and every v4 model with
    constant leaves) leaves `pos` untouched and returns an inactive sidecar.

    Intercepts are not stored: a leaf's intercept is `Tree.value` at that
    leaf, which the tree section already carries and which
    `refit_linear_tree` guarantees it equals. Reading it from the tree rather
    than writing it twice removes the only way the two could disagree.

    Everything is validated against `trees` before it is used: the tree
    count, the per-tree leaf count, each ordinal's range and uniqueness, each
    feature index, and the finiteness of every stored float. A model whose
    section does not describe its trees is a corrupt file, not a model to
    predict from.
    """
    if pos >= len(tokens) or tokens[pos] != LINEAR_SECTION_TAG:
        return LinearSectionResult(LinearEnsemble.inactive(), pos)
    var p = pos + 1

    var revision = Int(_next_token(tokens, p))
    p += 1
    if revision != LINEAR_SECTION_REVISION:
        raise Error(
            "unsupported linear section revision ",
            revision,
            "; this build reads revision ",
            LINEAR_SECTION_REVISION,
        )
    var n_trees = Int(_next_token(tokens, p))
    p += 1
    var width = Int(_next_token(tokens, p))
    p += 1
    if n_trees != len(trees):
        raise Error(
            "corrupt model file: linear section covers ",
            n_trees,
            " trees but the file holds ",
            len(trees),
        )
    if width != n_features:
        raise Error(
            "corrupt model file: linear section was fitted on ",
            width,
            " features but the mapper has ",
            n_features,
        )

    var entries = List[LinearTree](capacity=n_trees)
    for t in range(n_trees):
        if _next_token(tokens, p) != "ltree":
            raise Error("corrupt model file: expected 'ltree'")
        p += 1
        var n_leaves = Int(_next_token(tokens, p))
        p += 1
        var n_linear = Int(_next_token(tokens, p))
        p += 1
        if n_leaves != trees[t].n_leaves:
            raise Error(
                "corrupt model file: linear section gives tree ",
                t,
                " ",
                n_leaves,
                " leaves, the tree has ",
                trees[t].n_leaves,
            )
        if n_linear < 0 or n_linear > n_leaves:
            raise Error(
                "corrupt model file: linear leaf count out of range"
            )

        var ordinals = trees[t].leaf_ordinals()
        var value_of = List[Float64](capacity=n_leaves)
        value_of.resize(n_leaves, 0.0)
        for node in range(len(trees[t].feature)):
            if trees[t].feature[node] < 0:
                value_of[ordinals[node]] = trees[t].value[node]

        var leaves = List[LinearLeaf](capacity=n_leaves)
        for o in range(n_leaves):
            leaves.append(LinearLeaf.constant(value_of[o]))
        var seen = List[Bool](capacity=n_leaves)
        seen.resize(n_leaves, False)

        for _ in range(n_linear):
            var ordinal = Int(_next_token(tokens, p))
            p += 1
            if ordinal < 0 or ordinal >= n_leaves:
                raise Error(
                    "corrupt model file: linear leaf ordinal out of range"
                )
            if seen[ordinal]:
                raise Error(
                    "corrupt model file: linear leaf ordinal ",
                    ordinal,
                    " appears twice in tree ",
                    t,
                )
            seen[ordinal] = True
            var m = Int(_next_token(tokens, p))
            p += 1
            if m < 1 or m > width:
                raise Error(
                    "corrupt model file: linear leaf term count out of range"
                )
            var feature = List[Int](capacity=m)
            for _ in range(m):
                var f = Int(_next_token(tokens, p))
                p += 1
                if f < 0 or f >= width:
                    raise Error(
                        "corrupt model file: linear leaf feature out of range"
                    )
                feature.append(f)
            var coef = List[Float64](capacity=m)
            for _ in range(m):
                coef.append(_parse_f64_token(_next_token(tokens, p)))
                p += 1
            var center = List[Float64](capacity=m)
            for _ in range(m):
                center.append(_parse_f64_token(_next_token(tokens, p)))
                p += 1
            var leaf = LinearLeaf(
                value_of[ordinal], feature^, coef^, center^
            )
            leaf.check_shape()
            leaves[ordinal] = leaf^
        entries.append(LinearTree(leaves^))

    var out = LinearEnsemble(entries^, n_features)
    out.check_against(trees, n_features)
    return LinearSectionResult(out^, p)


def _next_token(tokens: List[String], pos: Int) raises -> String:
    if pos >= len(tokens):
        raise Error("unexpected end of model file inside the linear section")
    return tokens[pos].copy()


def linear_model_format_version(linear: LinearEnsemble, base: Int) -> Int:
    """The format version a model carrying `linear` must be written at:
    `LINEAR_MODEL_FORMAT_VERSION` when the sidecar is active, and `base`
    (whatever `serialize.mojo` writes today) when it is not.

    Keeping the bump conditional is the point. A constant-leaf model written
    by a build that knows about linear trees stays readable by one that does
    not, so the version is a statement about the file rather than about the
    binary that wrote it.
    """
    if linear.is_active():
        return LINEAR_MODEL_FORMAT_VERSION
    return base
