"""Feature binning.

Maps raw feature values to small integer bins so that split finding can
operate on fixed-size histograms instead of sorted feature values.

v0 uses equal-width binning. Quantile (equal-frequency) binning, which is
what LightGBM uses, is on the roadmap.
"""


@fieldwise_init
struct BinnedMatrix(Copyable, Movable):
    """Column-major binned feature matrix.

    Bin for (row r, feature f) is stored at `bins[f * n_rows + r]`.
    """

    var bins: List[UInt8]
    var n_rows: Int
    var n_features: Int
    var n_bins: Int

    def bin_at(self, row: Int, feature: Int) -> Int:
        return Int(self.bins[feature * self.n_rows + row])


def bin_equal_width(
    features: List[Float64], n_rows: Int, n_features: Int, n_bins: Int
) raises -> BinnedMatrix:
    """Bin a column-major feature matrix (`features[f * n_rows + r]`) into
    equal-width bins per feature."""
    if n_bins < 2 or n_bins > 256:
        raise Error("n_bins must be in [2, 256]")
    if len(features) != n_rows * n_features:
        raise Error("features length must equal n_rows * n_features")

    var bins = List[UInt8](capacity=n_rows * n_features)
    for f in range(n_features):
        var col = f * n_rows
        var lo = features[col]
        var hi = lo
        for r in range(1, n_rows):
            var v = features[col + r]
            if v < lo:
                lo = v
            if v > hi:
                hi = v
        var width = (hi - lo) / Float64(n_bins)
        for r in range(n_rows):
            var b: Int
            if width <= 0.0:
                b = 0
            else:
                b = Int((features[col + r] - lo) / width)
                if b >= n_bins:
                    b = n_bins - 1
                if b < 0:
                    b = 0
            bins.append(UInt8(b))
    return BinnedMatrix(bins^, n_rows, n_features, n_bins)
