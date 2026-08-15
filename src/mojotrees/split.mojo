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
"""

from .apple_cpu_policy import split_scan_ops
from .parallel import dispatch_features
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
)


# How a scanned feature's slot records the two booleans a `SplitInfo` needs,
# so that a task writes only scalars through unsafe pointers. Absence of both
# is a numerical split routing missing rows right, which is also what a
# feature that produced nothing leaves behind; the fold never reads the flags
# of a feature whose gain did not win, so the two cases never have to be
# told apart.
comptime _FLAG_DEFAULT_LEFT = 1
comptime _FLAG_CATEGORICAL = 2


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

    `n_rows` is the node's row count, used by the per-split CEGB cost and, at
    0, taken from the histogram's own totals. `depth` is the node's depth in
    edges from the root, which `monotone_penalty` discounts by.

    `cegb` is this node's CEGB costs (`cegb.prepare_cegb_node`), computed once
    before the scan so that charging a candidate is one lookup and one
    subtraction. Only a grower carrying the ensemble's `CegbLedger` can build
    them; a caller that passes none gets the split cost reconstructed from
    `extra.penalties.cegb` and is refused for the two penalties that need the
    ledger.

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

    comptime W = SIMD_LANES
    var grad_p = hist.grad.unsafe_ptr()
    var hess_p = hist.hess.unsafe_ptr()
    var count_p = hist.count.unsafe_ptr()

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
        var vg = SIMD[DType.float64, W](0.0)
        var vh = SIMD[DType.float64, W](0.0)
        var vc = SIMD[DType.int, W](0)
        var b = 0
        while b + W <= hist.n_bins:
            vg += grad_p.unsafe_load[width=W](base + b)
            vh += hess_p.unsafe_load[width=W](base + b)
            vc += count_p.unsafe_load[width=W](base + b)
            b += W
        var total_g = vg.reduce_add()
        var total_h = vh.reduce_add()
        var total_c = Int(vc.reduce_add())
        while b < hist.n_bins:
            total_g += grad_p.unsafe_load(base + b)
            total_h += hess_p.unsafe_load(base + b)
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
                hist.grad,
                hist.hess,
                hist.count,
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

        # Ordinary bins are [0, n_scan); the missing bin sits at n_scan and is
        # never a threshold, only a side to route.
        var n_scan = missing_bin if missing_bin >= 0 else hist.n_bins
        var miss_g = 0.0
        var miss_h = 0.0
        var miss_c = 0
        if missing_bin >= 0:
            miss_g = hist.grad[base + missing_bin]
            miss_h = hist.hess[base + missing_bin]
            miss_c = hist.count[base + missing_bin]

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
            left_g += hist.grad[base + b]
            left_h += hist.hess[base + b]
            left_c += hist.count[base + b]
            # The prefix sums above are what the drawn threshold is built
            # from, so they are accumulated for every bin below it and only
            # the drawn one is scored.
            if draw_one and b != pick:
                continue

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
                    var gain = _split_gain(
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
                    var gain = _split_gain(
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

    dispatch_features(
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
