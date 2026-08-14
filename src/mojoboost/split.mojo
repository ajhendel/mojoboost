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

Monotonic constraints
---------------------
The search also takes the node's constraint vector and output bounds. With an
active vector, every candidate is scored from child outputs clamped into the
node's bounds, and a candidate whose outputs run against its feature's
constraint is rejected whatever its gain. The bounds apply to candidates on
every feature, constrained or not, because a node under a constrained split
owes its interval no matter which feature it goes on to split. An empty
vector keeps the original unconstrained scoring path (see monotone.mojo).
"""

from .categorical import (
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
) -> Float64:
    """Gain of one candidate split. `left_g` and `right_g` are the child
    gradient sums after L1 soft-thresholding.

    Unconstrained, this is the plain second-order formula. Under active
    monotonic constraints it is LightGBM's constrained form: both child
    outputs are clamped into the node's bounds and scored at those outputs,
    and a candidate whose outputs run against `sign` scores 0.0, which no
    caller accepts because a split must beat a gain of 0.0 to be chosen.
    """
    if not constrained:
        return (
            left_g * left_g / (left_h + lambda_reg)
            + right_g * right_g / (right_h + lambda_reg)
            - parent_score
        )
    var left_out = bounds.clamp(-left_g / (left_h + lambda_reg))
    var right_out = bounds.clamp(-right_g / (right_h + lambda_reg))
    if violates(sign, left_out, right_out):
        return 0.0
    return (
        output_score(left_g, left_h, lambda_reg, left_out)
        + output_score(right_g, right_h, lambda_reg, right_out)
        - parent_score
    )


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
    threshold, and its `missing_bins` entry is not consulted."""
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

    comptime W = SIMD_LANES
    var grad_p = hist.grad.unsafe_ptr()
    var hess_p = hist.hess.unsafe_ptr()
    var count_p = hist.count.unsafe_ptr()
    for i_feature in range(n_active):
        var f = i_feature if use_all else features[i_feature]
        if f < 0 or f >= hist.n_features:
            continue
        if masked and (f >= len(allowed) or not allowed[f]):
            continue
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
            if cs.found and cs.gain > best.gain:
                best = SplitInfo.categorical(f, cs.gain, cs.bitset)
            continue

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
                    )
                    if gain > best.gain:
                        best = SplitInfo(f, b, gain, True, True)

            # Missing to the right. For a feature with no missing bin this is
            # the only candidate and the scan is exactly the ordinal one.
            if missing_bin < 0 or miss_c > 0:
                var right_g = total_g - left_g
                var right_h = total_h - left_h
                if left_h < min_child_hess or right_h < min_child_hess:
                    continue
                if (
                    left_c < min_data_in_leaf
                    or total_c - left_c < min_data_in_leaf
                ):
                    continue
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
                )
                if gain > best.gain:
                    best = SplitInfo(f, b, gain, True, False)

    return best^
