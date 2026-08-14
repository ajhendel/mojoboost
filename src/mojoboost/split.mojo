"""Best-split search over histograms.

Uses the standard second-order gain formula (XGBoost/LightGBM), with L1
regularization applied by soft-thresholding each gradient sum:

    T(G) = sign(G) * max(0, |G| - lambda_l1)
    gain = T(GL)^2 / (HL + lambda_l2)
         + T(GR)^2 / (HR + lambda_l2)
         - T(G)^2  / (H + lambda_l2)

With lambda_l1 = 0 (the default) T is the identity and this is the plain
second-order formula. A split at (feature, bin) sends rows with bin value
<= bin to the left child.

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

from .histogram import Histogram, SIMD_LANES
from .monotone import (
    MONOTONE_FREE,
    OutputBounds,
    monotone_sign,
    output_score,
    violates,
)


@always_inline
def soft_threshold_l1(s: Float64, lambda_l1: Float64) -> Float64:
    """LightGBM's ThresholdL1: shrink a gradient sum toward zero by
    `lambda_l1`, clamping at zero. Returns `s` unchanged when `lambda_l1`
    is not positive."""
    if lambda_l1 <= 0.0:
        return s
    var mag = abs(s) - lambda_l1
    if mag <= 0.0:
        return 0.0
    return mag if s > 0.0 else -mag


struct SplitInfo(Copyable, Movable, Writable):
    """Best split found for a node. `found` is False when no valid split
    exists (e.g. every candidate violates min_child_hess). `default_left` is
    the direction taken by rows in the feature's missing bin; it is False and
    unused for a feature with no missing bin."""

    var feature: Int
    var bin: Int
    var gain: Float64
    var found: Bool
    var default_left: Bool

    def __init__(
        out self,
        feature: Int,
        bin: Int,
        gain: Float64,
        found: Bool,
        default_left: Bool = False,
    ):
        self.feature = feature
        self.bin = bin
        self.gain = gain
        self.found = found
        self.default_left = default_left


def find_best_split(
    hist: Histogram,
    lambda_reg: Float64 = 1.0,
    min_child_hess: Float64 = 1e-3,
    min_data_in_leaf: Int = 0,
    lambda_l1: Float64 = 0.0,
    allowed: List[Bool] = [],
    features: List[Int] = [],
    missing_bins: List[Int] = [],
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
    and reports a `default_left` direction for its missing rows."""
    var best = SplitInfo(-1, -1, 0.0, False)
    if len(missing_bins) > 0 and len(missing_bins) != hist.n_features:
        raise Error("missing_bins length must equal n_features")
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
                    var gain = (
                        tl * tl / (dl_left_h + lambda_reg)
                        + tr * tr / (dl_right_h + lambda_reg)
                        - parent_score
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
                var gain = (
                    tl * tl / (left_h + lambda_reg)
                    + tr * tr / (right_h + lambda_reg)
                    - parent_score
                )
                if gain > best.gain:
                    best = SplitInfo(f, b, gain, True, False)

    return best^
