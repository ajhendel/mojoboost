"""Feature binning.

Maps raw feature values to small integer bins so that split finding can
operate on fixed-size histograms instead of sorted feature values.

`fit_bins` performs quantile (equal-frequency) binning, LightGBM style,
and returns a `BinMapper` whose stored edges let a trained model bin raw,
unseen feature values at prediction time. `bin_equal_width` remains as a
simple mapper-free alternative for experiments.

Missing values
--------------
`NaN` is the missing marker for numerical features. A feature whose training
column contains at least one `NaN` reserves one extra bin, immediately above
its ordinary bins, for missing values; `missing_bin[f]` is that bin's index,
or -1 for a feature that reserves none. Reserving costs that feature one
ordinary bin out of the `max_bins` budget, LightGBM style. `NaN` never
reaches the ordinary quantile comparisons: it is routed to the reserved bin
before the binary search runs, and it is left out of the quantile
computation when the edges are fit.

A `NaN` presented at prediction time for a feature with no reserved bin is
binned as the value 0.0, which is what LightGBM does for a feature whose
`missing_type` is `None`. `use_missing=False` reserves nothing for any
feature, so every `NaN` is binned as 0.0, again matching LightGBM.

`+inf` and `-inf` are ordinary finite-side extremes, not missing values: bin
edges are clamped to +/-1e300 (LightGBM's `Common::AvoidInf`), so `+inf`
always lands in a feature's highest ordinary bin and `-inf` in bin 0.

Categorical features are not covered by any of this: they reserve no missing
bin and keep `missing_bin[f] = -1`, because `categorical.mojo` already sends
`NaN`, negatives, and unseen codes to its own bin 0.

Features named in `categorical_features` are binned by `categorical.mojo`
instead: they get no quantile edges at all, and their raw integer codes map
to bins through a fitted category table. The two paths never mix, so adding
a categorical column changes nothing about how the numerical columns are
binned.

Fitting and transforming parallelize across features: every feature's edge
computation (column copy + sort) and bin assignment is independent and
writes only its own output range, so workers need no synchronization and
the result is bit-identical to the serial path. Inputs too small to
amortize task-scheduling overhead stay serial.
"""

from std.math import isnan

from .categorical import CategoricalSpec, fit_categorical_spec
from .parallel import dispatch_features

# LightGBM's Common::kMaxDouble: bin edges are clamped here so an infinite
# training value cannot produce an infinite (or non-increasing) edge.
comptime MAX_EDGE = 1e300


def _avoid_inf(x: Float64) -> Float64:
    """LightGBM's `Common::AvoidInf`, clamping an edge into +/-1e300."""
    if isnan(x):
        return 0.0
    if x >= MAX_EDGE:
        return MAX_EDGE
    if x <= -MAX_EDGE:
        return -MAX_EDGE
    return x


def no_missing_bins(n_features: Int) -> List[Int]:
    """A per-feature missing-bin table for data with no missing support:
    every entry -1."""
    var out = List[Int](capacity=n_features)
    out.resize(n_features, -1)
    return out^


def _sized_missing_bins(var table: List[Int], n_features: Int) -> List[Int]:
    """`table` when it is already per-feature, otherwise an all -1 table."""
    if len(table) == n_features:
        return table^
    return no_missing_bins(n_features)


struct BinnedMatrix(Copyable, Movable):
    """Column-major binned feature matrix.

    Bin for (row r, feature f) is stored at `bins[f * n_rows + r]`. `cats`
    records which features are categorical, so split finding knows to search
    category partitions rather than ordinal thresholds; an empty spec means
    every feature is numerical. `missing_bin[f]` is the bin reserved for
    missing values of feature f, or -1 when that feature reserves none.
    """

    var bins: List[UInt8]
    var n_rows: Int
    var n_features: Int
    var n_bins: Int
    var cats: CategoricalSpec
    var missing_bin: List[Int]

    def __init__(
        out self,
        var bins: List[UInt8],
        n_rows: Int,
        n_features: Int,
        n_bins: Int,
    ):
        self.bins = bins^
        self.n_rows = n_rows
        self.n_features = n_features
        self.n_bins = n_bins
        self.cats = CategoricalSpec.none()
        self.missing_bin = no_missing_bins(n_features)

    def __init__(
        out self,
        var bins: List[UInt8],
        n_rows: Int,
        n_features: Int,
        n_bins: Int,
        var cats: CategoricalSpec,
        var missing_bin: List[Int] = [],
    ):
        """A `missing_bin` table of the wrong length (the empty default
        included) means no feature reserves a missing bin."""
        self.bins = bins^
        self.n_rows = n_rows
        self.n_features = n_features
        self.n_bins = n_bins
        self.cats = cats^
        self.missing_bin = _sized_missing_bins(missing_bin^, n_features)

    def bin_at(self, row: Int, feature: Int) -> Int:
        return Int(self.bins[feature * self.n_rows + row])

    def is_missing(self, row: Int, feature: Int) -> Bool:
        """Whether (row, feature) holds a missing value."""
        return self.bin_at(row, feature) == self.missing_bin[feature]


struct BinMapper(Copyable, Movable):
    """Per-feature bin edges fit on training data.

    Feature f's edges are `edges[edge_offsets[f] : edge_offsets[f + 1]]`,
    strictly increasing. A value v maps to the first bin b whose edge
    satisfies v <= edge[b]; values above every edge map to the last bin.
    A feature with k edges uses k + 1 bins (k + 1 <= n_bins).

    Categorical features carry no edges; `cats` holds their category tables
    and `bin_value` routes them through it instead.

    A numerical feature with `missing_bin[f] >= 0` reserves that bin for
    missing values, so its k edges give ordinary bins 0..k and a missing bin
    at k + 1. `missing_bin[f] = -1` means no reservation.
    """

    var edges: List[Float64]
    var edge_offsets: List[Int]
    var n_features: Int
    var n_bins: Int
    var cats: CategoricalSpec
    var missing_bin: List[Int]

    def __init__(
        out self,
        var edges: List[Float64],
        var edge_offsets: List[Int],
        n_features: Int,
        n_bins: Int,
    ):
        self.edges = edges^
        self.edge_offsets = edge_offsets^
        self.n_features = n_features
        self.n_bins = n_bins
        self.cats = CategoricalSpec.all_numerical(n_features)
        self.missing_bin = no_missing_bins(n_features)

    def __init__(
        out self,
        var edges: List[Float64],
        var edge_offsets: List[Int],
        n_features: Int,
        n_bins: Int,
        var cats: CategoricalSpec,
        var missing_bin: List[Int] = [],
    ):
        """A `missing_bin` table of the wrong length (the empty default
        included) means no feature reserves a missing bin."""
        self.edges = edges^
        self.edge_offsets = edge_offsets^
        self.n_features = n_features
        self.n_bins = n_bins
        self.cats = cats^
        self.missing_bin = _sized_missing_bins(missing_bin^, n_features)

    def has_missing(self) -> Bool:
        """Whether any feature reserves a missing bin."""
        for f in range(self.n_features):
            if self.missing_bin[f] >= 0:
                return True
        return False

    def matches(self, other: BinMapper) -> Bool:
        """Whether two mappers bin every value the same way.

        Equality of the fitted binning, not of the objects: same features,
        same edges, same missing reservations, same category tables. It is
        what lets a fitted model take more trees from a dataset that was
        binned separately (see boosting.train_more) without any chance of a
        bin index meaning two different things.
        """
        if self.n_features != other.n_features:
            return False
        if self.n_bins != other.n_bins:
            return False
        if len(self.edges) != len(other.edges):
            return False
        if len(self.edge_offsets) != len(other.edge_offsets):
            return False
        for i in range(len(self.edges)):
            if self.edges[i] != other.edges[i]:
                return False
        for i in range(len(self.edge_offsets)):
            if self.edge_offsets[i] != other.edge_offsets[i]:
                return False
        if len(self.missing_bin) != len(other.missing_bin):
            return False
        for i in range(len(self.missing_bin)):
            if self.missing_bin[i] != other.missing_bin[i]:
                return False
        for f in range(self.n_features):
            if self.cats.is_cat(f) != other.cats.is_cat(f):
                return False
        if len(self.cats.codes) != len(other.cats.codes):
            return False
        for i in range(len(self.cats.codes)):
            if self.cats.codes[i] != other.cats.codes[i]:
                return False
        if len(self.cats.offsets) != len(other.cats.offsets):
            return False
        for i in range(len(self.cats.offsets)):
            if self.cats.offsets[i] != other.cats.offsets[i]:
                return False
        return True

    def bin_value(self, feature: Int, v: Float64) -> Int:
        if self.cats.is_cat(feature):
            return self.cats.bin_of(feature, v)
        var value = v
        if isnan(value):
            var mb = self.missing_bin[feature]
            if mb >= 0:
                return mb
            # No reserved bin: LightGBM bins NaN as 0.0 for a feature whose
            # missing_type is None.
            value = 0.0
        var lo = self.edge_offsets[feature]
        var left = lo
        var right = self.edge_offsets[feature + 1]
        while left < right:
            var mid = (left + right) // 2
            if value <= self.edges[mid]:
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
        var miss_p = self.missing_bin.unsafe_ptr()
        ref cats = self.cats

        def do_feature(f: Int) {imm}:
            var col = f * n_rows
            if cats.is_cat(f):
                for r in range(n_rows):
                    bins_p.unsafe_store(
                        col + r,
                        UInt8(cats.bin_of(f, feat_p.unsafe_load(col + r))),
                    )
                return
            var lo = offs_p.unsafe_load(f)
            var hi = offs_p.unsafe_load(f + 1)
            var mb = miss_p.unsafe_load(f)
            for r in range(n_rows):
                var v = feat_p.unsafe_load(col + r)
                # NaN is routed before any comparison, so it never takes part
                # in the quantile search (see `bin_value`).
                if isnan(v):
                    if mb >= 0:
                        bins_p.unsafe_store(col + r, UInt8(mb))
                        continue
                    v = 0.0
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
        return BinnedMatrix(
            bins^,
            n_rows,
            n_features,
            self.n_bins,
            self.cats.copy(),
            self.missing_bin.copy(),
        )

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
    categorical_features: List[Int] = [],
    use_missing: Bool = True,
) raises -> BinMapper:
    """Fit quantile (equal-frequency) bin edges on a column-major feature
    matrix. Edges are midpoints between distinct values at quantile
    boundaries, so duplicate-heavy features simply use fewer bins.

    Feature indices listed in `categorical_features` are treated as
    integer-coded categoricals: they are excluded from quantile binning
    entirely and get a category table instead (see `categorical.mojo`).

    With `use_missing` (the default), a numerical feature whose column holds
    any `NaN` reserves its highest bin for missing values and fits its edges
    over the remaining `max_bins - 1` bins from the non-missing values alone,
    so `NaN` never enters a quantile comparison. `use_missing=False` reserves
    nothing and bins `NaN` as 0.0, matching LightGBM's `use_missing=false`."""
    if max_bins < 2 or max_bins > 256:
        raise Error("max_bins must be in [2, 256]")
    if n_rows < 1:
        raise Error("n_rows must be positive")
    if len(features) != n_rows * n_features:
        raise Error("features length must equal n_rows * n_features")

    var cats = fit_categorical_spec(
        features, n_rows, n_features, categorical_features, max_bins
    )

    # Each feature's edges land in its own fixed-stride scratch slice; a
    # serial pass then concatenates them in feature order, so the result is
    # identical whichever path ran.
    var max_edges = max_bins - 1
    var scratch = List[Float64](capacity=n_features * max_edges)
    scratch.resize(n_features * max_edges, 0.0)
    var counts = List[Int](capacity=n_features)
    counts.resize(n_features, 0)
    var missing_bin = no_missing_bins(n_features)

    var scratch_p = scratch.unsafe_ptr()
    var counts_p = counts.unsafe_ptr()
    var missing_p = missing_bin.unsafe_ptr()
    var feat_p = features.unsafe_ptr()
    ref spec = cats

    def do_feature(f: Int) {imm}:
        # Categorical columns never enter quantile binning.
        if spec.is_cat(f):
            counts_p.unsafe_store(f, 0)
            return
        # NaN is dropped before the sort, so it never takes part in a
        # quantile comparison, and a column that has any gives up one bin to
        # hold its missing values.
        var col = List[Float64](capacity=n_rows)
        for r in range(n_rows):
            var v = feat_p.unsafe_load(f * n_rows + r)
            if not isnan(v):
                col.append(v)
        var n_valid = len(col)
        var reserve = use_missing and n_valid < n_rows
        var n_ordinary = max_bins - 1 if reserve else max_bins
        sort(col)
        var out = scratch_p.unsafe_offset(f * max_edges)
        var n_out = 0
        for b in range(1, n_ordinary):
            var idx = b * n_valid // n_ordinary
            if idx <= 0 or idx >= n_valid:
                continue
            var below = col[idx - 1]
            var above = col[idx]
            if above <= below:
                continue
            var edge = _avoid_inf((below + above) / 2.0)
            # Repeated quantile indices (n_valid < n_ordinary) revisit the
            # same boundary, and clamping an infinite midpoint can repeat the
            # previous edge; keep edges strictly increasing either way.
            if n_out > 0 and edge <= out.unsafe_load(n_out - 1):
                continue
            out.unsafe_store(n_out, edge)
            n_out += 1
        counts_p.unsafe_store(f, n_out)
        # k edges give ordinary bins 0..k, so the missing bin is k + 1.
        missing_p.unsafe_store(f, n_out + 1 if reserve else -1)

    dispatch_features(do_feature, n_features, n_features * n_rows)

    var edges = List[Float64]()
    var offsets = List[Int](capacity=n_features + 1)
    offsets.append(0)
    for f in range(n_features):
        var base = f * max_edges
        for i in range(counts[f]):
            edges.append(scratch[base + i])
        offsets.append(len(edges))
    return BinMapper(
        edges^, offsets^, n_features, max_bins, cats^, missing_bin^
    )


def bin_equal_width(
    features: List[Float64], n_rows: Int, n_features: Int, n_bins: Int
) raises -> BinnedMatrix:
    """Bin a column-major feature matrix (`features[f * n_rows + r]`) into
    equal-width bins per feature. This mapper-free path has no missing-value
    support: it reserves no bin and sends `NaN` to bin 0."""
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
            var v = features[col + r]
            if width <= 0.0 or isnan(v):
                b = 0
            else:
                b = Int((v - lo) / width)
                if b >= n_bins:
                    b = n_bins - 1
                if b < 0:
                    b = 0
            bins.append(UInt8(b))
    return BinnedMatrix(bins^, n_rows, n_features, n_bins)
