"""Shared split-gain arithmetic.

The second-order gain formula and its L1 soft-thresholding are needed by
both split searches: the ordinal threshold scan in `split.mojo` and the
category partition search in `categorical.mojo`. They live here so the two
cannot drift, and so neither module has to import the other.

    T(G) = sign(G) * max(0, |G| - lambda_l1)
    score(G, H) = T(G)^2 / (H + lambda_l2)
    gain = score(GL, HL) + score(GR, HR) - score(G, H)

`split.mojo` re-exports `soft_threshold_l1` under its historical name.
"""


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


@always_inline
def leaf_score(
    g: Float64, h: Float64, lambda_l1: Float64, lambda_l2: Float64
) -> Float64:
    """The second-order objective improvement of a leaf holding (g, h)."""
    var t = soft_threshold_l1(g, lambda_l1)
    return t * t / (h + lambda_l2)
