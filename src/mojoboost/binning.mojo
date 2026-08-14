"""Feature binning.

Maps raw feature values to small integer bins so that split finding can
operate on fixed-size histograms instead of sorted feature values.

`fit_bins` performs quantile (equal-frequency) binning, LightGBM style,
and returns a `BinMapper` whose stored edges let a trained model bin raw,
unseen feature values at prediction time. `bin_equal_width` remains as a
simple mapper-free alternative for experiments.
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


@fieldwise_init
struct BinMapper(Copyable, Movable):
    """Per-feature bin edges fit on training data.

    Feature f's edges are `edges[edge_offsets[f] : edge_offsets[f + 1]]`,
    strictly increasing. A value v maps to the first bin b whose edge
    satisfies v <= edge[b]; values above every edge map to the last bin.
    A feature with k edges uses k + 1 bins (k + 1 <= n_bins).
    """

    var edges: List[Float64]
    var edge_offsets: List[Int]
    var n_features: Int
    var n_bins: Int

    def bin_value(self, feature: Int, v: Float64) -> Int:
        var lo = self.edge_offsets[feature]
        var left = lo
        var right = self.edge_offsets[feature + 1]
        while left < right:
            var mid = (left + right) // 2
            if v <= self.edges[mid]:
                right = mid
            else:
                left = mid + 1
        return left - lo

    def transform(
        self, features: List[Float64], n_rows: Int
    ) raises -> BinnedMatrix:
        """Bin a column-major feature matrix (`features[f * n_rows + r]`)."""
        if len(features) != n_rows * self.n_features:
            raise Error("features length must equal n_rows * n_features")
        var bins = List[UInt8](capacity=n_rows * self.n_features)
        for f in range(self.n_features):
            var col = f * n_rows
            for r in range(n_rows):
                bins.append(UInt8(self.bin_value(f, features[col + r])))
        return BinnedMatrix(bins^, n_rows, self.n_features, self.n_bins)

    def bin_row(self, row: List[Float64]) raises -> List[Int]:
        """Bin one example (length n_features) for prediction."""
        if len(row) != self.n_features:
            raise Error("row length must equal n_features")
        var out = List[Int](capacity=self.n_features)
        for f in range(self.n_features):
            out.append(self.bin_value(f, row[f]))
        return out^


def fit_bins(
    features: List[Float64],
    n_rows: Int,
    n_features: Int,
    max_bins: Int = 255,
) raises -> BinMapper:
    """Fit quantile (equal-frequency) bin edges on a column-major feature
    matrix. Edges are midpoints between distinct values at quantile
    boundaries, so duplicate-heavy features simply use fewer bins."""
    if max_bins < 2 or max_bins > 256:
        raise Error("max_bins must be in [2, 256]")
    if n_rows < 1:
        raise Error("n_rows must be positive")
    if len(features) != n_rows * n_features:
        raise Error("features length must equal n_rows * n_features")

    var edges = List[Float64]()
    var offsets = List[Int]()
    offsets.append(0)
    for f in range(n_features):
        var col = List[Float64](capacity=n_rows)
        for r in range(n_rows):
            col.append(features[f * n_rows + r])
        sort(col)
        for b in range(1, max_bins):
            var idx = b * n_rows // max_bins
            if idx <= 0 or idx >= n_rows:
                continue
            var below = col[idx - 1]
            var above = col[idx]
            if above <= below:
                continue
            var edge = (below + above) / 2.0
            # Repeated quantile indices (n_rows < max_bins) revisit the same
            # boundary; keep edges strictly increasing.
            if len(edges) > offsets[f] and edge <= edges[len(edges) - 1]:
                continue
            edges.append(edge)
        offsets.append(len(edges))
    return BinMapper(edges^, offsets^, n_features, max_bins)


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
