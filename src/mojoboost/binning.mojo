"""Feature binning.

Maps raw feature values to small integer bins so that split finding can
operate on fixed-size histograms instead of sorted feature values.

`fit_bins` performs quantile (equal-frequency) binning, LightGBM style,
and returns a `BinMapper` whose stored edges let a trained model bin raw,
unseen feature values at prediction time. `bin_equal_width` remains as a
simple mapper-free alternative for experiments.

Fitting and transforming parallelize across features: every feature's edge
computation (column copy + sort) and bin assignment is independent and
writes only its own output range, so workers need no synchronization and
the result is bit-identical to the serial path. Inputs too small to
amortize task-scheduling overhead stay serial.
"""

from .parallel import dispatch_features


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
        var n_features = self.n_features
        var bins = List[UInt8](capacity=n_rows * n_features)
        bins.resize(n_rows * n_features, 0)

        var bins_p = bins.unsafe_ptr()
        var feat_p = features.unsafe_ptr()
        var edges_p = self.edges.unsafe_ptr()
        var offs_p = self.edge_offsets.unsafe_ptr()

        def do_feature(f: Int) {imm}:
            var lo = offs_p.unsafe_load(f)
            var hi = offs_p.unsafe_load(f + 1)
            var col = f * n_rows
            for r in range(n_rows):
                var v = feat_p.unsafe_load(col + r)
                var left = lo
                var right = hi
                while left < right:
                    var mid = (left + right) // 2
                    if v <= edges_p.unsafe_load(mid):
                        right = mid
                    else:
                        left = mid + 1
                bins_p.unsafe_store(col + r, UInt8(left - lo))

        dispatch_features(do_feature, n_features, n_features * n_rows)
        return BinnedMatrix(bins^, n_rows, n_features, self.n_bins)

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

    # Each feature's edges land in its own fixed-stride scratch slice; a
    # serial pass then concatenates them in feature order, so the result is
    # identical whichever path ran.
    var max_edges = max_bins - 1
    var scratch = List[Float64](capacity=n_features * max_edges)
    scratch.resize(n_features * max_edges, 0.0)
    var counts = List[Int](capacity=n_features)
    counts.resize(n_features, 0)

    var scratch_p = scratch.unsafe_ptr()
    var counts_p = counts.unsafe_ptr()
    var feat_p = features.unsafe_ptr()

    def do_feature(f: Int) {imm}:
        var col = List[Float64](capacity=n_rows)
        for r in range(n_rows):
            col.append(feat_p.unsafe_load(f * n_rows + r))
        sort(col)
        var out = scratch_p.unsafe_offset(f * max_edges)
        var n_out = 0
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
            if n_out > 0 and edge <= out.unsafe_load(n_out - 1):
                continue
            out.unsafe_store(n_out, edge)
            n_out += 1
        counts_p.unsafe_store(f, n_out)

    dispatch_features(do_feature, n_features, n_features * n_rows)

    var edges = List[Float64]()
    var offsets = List[Int](capacity=n_features + 1)
    offsets.append(0)
    for f in range(n_features):
        var base = f * max_edges
        for i in range(counts[f]):
            edges.append(scratch[base + i])
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
