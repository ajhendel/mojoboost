"""Best-split search over histograms.

Uses the standard second-order gain formula (XGBoost/LightGBM):

    gain = GL^2 / (HL + lambda) + GR^2 / (HR + lambda) - G^2 / (H + lambda)

A split at (feature, bin) sends rows with bin value <= bin to the left child.
"""

from .histogram import Histogram


@fieldwise_init
struct SplitInfo(Copyable, Movable, Writable):
    """Best split found for a node. `found` is False when no valid split
    exists (e.g. every candidate violates min_child_hess)."""

    var feature: Int
    var bin: Int
    var gain: Float64
    var found: Bool


def find_best_split(
    hist: Histogram,
    lambda_reg: Float64 = 1.0,
    min_child_hess: Float64 = 1e-3,
    min_data_in_leaf: Int = 0,
) -> SplitInfo:
    """Scan all (feature, bin) split candidates and return the one with the
    highest gain. Only splits with positive gain are returned as found."""
    var best = SplitInfo(-1, -1, 0.0, False)

    for f in range(hist.n_features):
        var base = f * hist.n_bins
        var total_g = 0.0
        var total_h = 0.0
        var total_c = 0
        for b in range(hist.n_bins):
            total_g += hist.grad[base + b]
            total_h += hist.hess[base + b]
            total_c += hist.count[base + b]
        var parent_score = total_g * total_g / (total_h + lambda_reg)

        var left_g = 0.0
        var left_h = 0.0
        var left_c = 0
        for b in range(hist.n_bins - 1):
            left_g += hist.grad[base + b]
            left_h += hist.hess[base + b]
            left_c += hist.count[base + b]
            var right_g = total_g - left_g
            var right_h = total_h - left_h
            if left_h < min_child_hess or right_h < min_child_hess:
                continue
            if left_c < min_data_in_leaf or total_c - left_c < min_data_in_leaf:
                continue
            var gain = (
                left_g * left_g / (left_h + lambda_reg)
                + right_g * right_g / (right_h + lambda_reg)
                - parent_score
            )
            if gain > best.gain:
                best = SplitInfo(f, b, gain, True)

    return best^
