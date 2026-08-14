"""Sparse feature matrices: CSC/CSR storage, sparse binning, binned data.

MojoBoost's histogram work is feature-oriented, so the internal
representation is compressed sparse column (CSC): every feature owns one
contiguous run of (row index, value) pairs. CSR input is transposed to CSC
once on the way in, and back to a row view for prediction.

Implicit-zero semantics
-----------------------
An entry absent from the sparse structure is the numerical value `0.0`, not
a missing value. A sparse matrix is therefore exactly equivalent to the
dense matrix obtained by filling the gaps with zeros: implicit zeros
participate in the quantile bin edges like any other value, land in whatever
bin contains 0.0, and route through splits accordingly. Explicitly stored
zeros (which SciPy keeps until `eliminate_zeros()` is called) are treated
identically to implicit ones.

This matches LightGBM's default `zero_as_missing=false`. LightGBM's
`zero_as_missing=true` mode is not implemented, and neither is missing-value
handling in general (NaN routed to a learned default child): MojoBoost has no
missing-value support on the dense path either, and NaN inputs are
unsupported on both.

Canonical form
--------------
CSC requires strictly ascending row indices within each column, and CSR
strictly ascending column indices within each row. That is SciPy's canonical
form (`sum_duplicates()` then `sort_indices()`), and it also rules out
duplicate entries. `validate` enforces it.
"""

from .binning import BinMapper, BinnedMatrix
from .parallel import dispatch_features


@fieldwise_init
struct CscMatrix(Copyable, Movable):
    """Compressed sparse column matrix of raw feature values.

    Feature f owns entries `[col_offsets[f], col_offsets[f + 1])`; entry i is
    the value `values[i]` at row `row_index[i]`. Row indices are strictly
    ascending within a column. Rows with no entry for a feature carry the
    value 0.0 (see the module docstring).
    """

    var row_index: List[Int]
    var values: List[Float64]
    var col_offsets: List[Int]
    var n_rows: Int
    var n_features: Int

    def nnz(self) -> Int:
        return len(self.values)

    def density(self) -> Float64:
        return Float64(len(self.values)) / Float64(
            self.n_rows * self.n_features
        )

    def validate(self) raises:
        _check_compressed(
            self.col_offsets,
            self.row_index,
            self.values,
            self.n_features,
            self.n_rows,
            "CSC",
            "column",
            "row",
        )

    def to_csr(self) raises -> CsrMatrix:
        """Transpose to a row view. O(nnz + n_rows), no densification."""
        self.validate()
        var nnz = len(self.values)
        var row_offsets = List[Int](capacity=self.n_rows + 1)
        row_offsets.resize(self.n_rows + 1, 0)
        for i in range(nnz):
            row_offsets[self.row_index[i] + 1] += 1
        for r in range(self.n_rows):
            row_offsets[r + 1] += row_offsets[r]

        var col_index = List[Int](capacity=nnz)
        col_index.resize(nnz, 0)
        var values = List[Float64](capacity=nnz)
        values.resize(nnz, 0.0)
        var pos = List[Int](capacity=self.n_rows)
        for r in range(self.n_rows):
            pos.append(row_offsets[r])
        # Features are visited in ascending order, so each row's column
        # indices come out ascending: the result is canonical CSR.
        for f in range(self.n_features):
            for i in range(self.col_offsets[f], self.col_offsets[f + 1]):
                var p = pos[self.row_index[i]]
                col_index[p] = f
                values[p] = self.values[i]
                pos[self.row_index[i]] = p + 1
        return CsrMatrix(
            col_index^, values^, row_offsets^, self.n_rows, self.n_features
        )

    def to_dense(self) raises -> List[Float64]:
        """Densify to a column-major matrix. Test and benchmark helper only:
        allocating n_rows * n_features floats is exactly what the sparse path
        exists to avoid."""
        self.validate()
        var out = List[Float64](capacity=self.n_rows * self.n_features)
        out.resize(self.n_rows * self.n_features, 0.0)
        for f in range(self.n_features):
            for i in range(self.col_offsets[f], self.col_offsets[f + 1]):
                out[f * self.n_rows + self.row_index[i]] = self.values[i]
        return out^


@fieldwise_init
struct CsrMatrix(Copyable, Movable):
    """Compressed sparse row matrix of raw feature values.

    Row r owns entries `[row_offsets[r], row_offsets[r + 1])`; entry i is the
    value `values[i]` of feature `col_index[i]`. Column indices are strictly
    ascending within a row, so a row's features can be looked up by binary
    search. This is the prediction-side layout.
    """

    var col_index: List[Int]
    var values: List[Float64]
    var row_offsets: List[Int]
    var n_rows: Int
    var n_features: Int

    def nnz(self) -> Int:
        return len(self.values)

    def validate(self) raises:
        _check_compressed(
            self.row_offsets,
            self.col_index,
            self.values,
            self.n_rows,
            self.n_features,
            "CSR",
            "row",
            "column",
        )

    def lookup(self, row: Int, feature: Int) -> Float64:
        """Value at (row, feature), 0.0 when the entry is absent."""
        var lo = self.row_offsets[row]
        var hi = self.row_offsets[row + 1]
        var end = hi
        while lo < hi:
            var mid = (lo + hi) // 2
            if self.col_index[mid] < feature:
                lo = mid + 1
            else:
                hi = mid
        if lo < end and self.col_index[lo] == feature:
            return self.values[lo]
        return 0.0

    def to_csc(self) raises -> CscMatrix:
        """Transpose to the feature-oriented layout. O(nnz + n_features)."""
        self.validate()
        var nnz = len(self.values)
        var col_offsets = List[Int](capacity=self.n_features + 1)
        col_offsets.resize(self.n_features + 1, 0)
        for i in range(nnz):
            col_offsets[self.col_index[i] + 1] += 1
        for f in range(self.n_features):
            col_offsets[f + 1] += col_offsets[f]

        var row_index = List[Int](capacity=nnz)
        row_index.resize(nnz, 0)
        var values = List[Float64](capacity=nnz)
        values.resize(nnz, 0.0)
        var pos = List[Int](capacity=self.n_features)
        for f in range(self.n_features):
            pos.append(col_offsets[f])
        # Rows are visited in ascending order, so each column's row indices
        # come out ascending: the result is canonical CSC.
        for r in range(self.n_rows):
            for i in range(self.row_offsets[r], self.row_offsets[r + 1]):
                var p = pos[self.col_index[i]]
                row_index[p] = r
                values[p] = self.values[i]
                pos[self.col_index[i]] = p + 1
        return CscMatrix(
            row_index^, values^, col_offsets^, self.n_rows, self.n_features
        )

    def to_dense(self) raises -> List[Float64]:
        """Densify to a column-major matrix. Test and benchmark helper only."""
        self.validate()
        var out = List[Float64](capacity=self.n_rows * self.n_features)
        out.resize(self.n_rows * self.n_features, 0.0)
        for r in range(self.n_rows):
            for i in range(self.row_offsets[r], self.row_offsets[r + 1]):
                out[self.col_index[i] * self.n_rows + r] = self.values[i]
        return out^


def _check_compressed(
    offsets: List[Int],
    indices: List[Int],
    values: List[Float64],
    n_outer: Int,
    n_inner: Int,
    kind: String,
    outer: String,
    inner: String,
) raises:
    """Shared CSC/CSR structural validation. `outer` is the compressed axis
    (columns for CSC, rows for CSR) and `inner` the indexed one."""
    if n_outer < 1 or n_inner < 1:
        raise Error(kind + " matrix must have positive dimensions")
    if len(offsets) != n_outer + 1:
        raise Error(kind + " offsets must have length n_" + outer + "s + 1")
    if offsets[0] != 0:
        raise Error(kind + " offsets must start at 0")
    if len(indices) != len(values):
        raise Error(kind + " indices and values must have equal length")
    if offsets[n_outer] != len(values):
        raise Error(kind + " offsets must end at nnz")
    for k in range(n_outer):
        var lo = offsets[k]
        var hi = offsets[k + 1]
        if hi < lo:
            raise Error(kind + " offsets must be non-decreasing")
        for i in range(lo, hi):
            if indices[i] < 0 or indices[i] >= n_inner:
                raise Error(kind + " " + inner + " index out of range")
            if i > lo and indices[i] <= indices[i - 1]:
                raise Error(
                    kind
                    + " "
                    + inner
                    + " indices must be strictly ascending within each "
                    + outer
                )


def csc_from_dense(
    features: List[Float64], n_rows: Int, n_features: Int
) raises -> CscMatrix:
    """Build a CSC matrix from a column-major dense matrix, dropping zeros.
    Test and benchmark helper: real callers should never materialize the
    dense matrix in the first place."""
    if len(features) != n_rows * n_features:
        raise Error("features length must equal n_rows * n_features")
    var row_index = List[Int]()
    var values = List[Float64]()
    var col_offsets = List[Int](capacity=n_features + 1)
    col_offsets.append(0)
    for f in range(n_features):
        for r in range(n_rows):
            var v = features[f * n_rows + r]
            if v != 0.0:
                row_index.append(r)
                values.append(v)
        col_offsets.append(len(values))
    return CscMatrix(row_index^, values^, col_offsets^, n_rows, n_features)


def _sorted_column_at(
    stored: List[Float64], n_neg: Int, n_zero: Int, n_implicit: Int, i: Int
) -> Float64:
    """Element `i` of the ascending sorted dense column, reconstructed from
    the ascending sorted stored values plus `n_implicit` implicit zeros.

    `stored` is laid out as [negatives, stored zeros, positives]; the dense
    column is the same with `n_implicit` extra zeros spliced into the middle
    block, which holds `n_zero` zeros in total.
    """
    if i < n_neg:
        return stored[i]
    if i < n_neg + n_zero:
        return 0.0
    return stored[i - n_implicit]


def fit_bins_csc(csc: CscMatrix, max_bins: Int = 255) raises -> BinMapper:
    """Fit quantile bin edges on a sparse matrix, without densifying.

    Produces bit-identical edges to `fit_bins` on the densified matrix: per
    feature it sorts only the stored values (O(nnz_f log nnz_f)) and indexes
    the implied dense sorted column through `_sorted_column_at` instead of
    materializing n_rows values.
    """
    if max_bins < 2 or max_bins > 256:
        raise Error("max_bins must be in [2, 256]")
    csc.validate()

    var n_rows = csc.n_rows
    var n_features = csc.n_features
    var max_edges = max_bins - 1
    var scratch = List[Float64](capacity=n_features * max_edges)
    scratch.resize(n_features * max_edges, 0.0)
    var counts = List[Int](capacity=n_features)
    counts.resize(n_features, 0)

    var scratch_p = scratch.unsafe_ptr()
    var counts_p = counts.unsafe_ptr()
    var val_p = csc.values.unsafe_ptr()
    var off_p = csc.col_offsets.unsafe_ptr()

    def do_feature(f: Int) {imm}:
        var lo = off_p.unsafe_load(f)
        var hi = off_p.unsafe_load(f + 1)
        var n_stored = hi - lo
        var stored = List[Float64](capacity=n_stored)
        for i in range(lo, hi):
            stored.append(val_p.unsafe_load(i))
        sort(stored)

        var n_implicit = n_rows - n_stored
        var n_neg = 0
        while n_neg < n_stored and stored[n_neg] < 0.0:
            n_neg += 1
        var n_stored_zero = 0
        while (
            n_neg + n_stored_zero < n_stored
            and stored[n_neg + n_stored_zero] == 0.0
        ):
            n_stored_zero += 1
        var n_zero = n_stored_zero + n_implicit

        var out = scratch_p.unsafe_offset(f * max_edges)
        var n_out = 0
        for b in range(1, max_bins):
            var idx = b * n_rows // max_bins
            if idx <= 0 or idx >= n_rows:
                continue
            var below = _sorted_column_at(
                stored, n_neg, n_zero, n_implicit, idx - 1
            )
            var above = _sorted_column_at(
                stored, n_neg, n_zero, n_implicit, idx
            )
            if above <= below:
                continue
            var edge = (below + above) / 2.0
            if n_out > 0 and edge <= out.unsafe_load(n_out - 1):
                continue
            out.unsafe_store(n_out, edge)
            n_out += 1
        counts_p.unsafe_store(f, n_out)

    dispatch_features(
        do_feature, n_features, csc.nnz() + n_features * max_bins
    )

    var edges = List[Float64]()
    var offsets = List[Int](capacity=n_features + 1)
    offsets.append(0)
    for f in range(n_features):
        var base = f * max_edges
        for i in range(counts[f]):
            edges.append(scratch[base + i])
        offsets.append(len(edges))
    return BinMapper(edges^, offsets^, n_features, max_bins)


@fieldwise_init
struct SparseBinnedMatrix(Copyable, Movable):
    """Binned CSC matrix.

    Stored entry i belongs to row `row_index[i]` and falls in bin `bin[i]`;
    feature f owns `[col_offsets[f], col_offsets[f + 1])`. Every row without
    a stored entry for feature f falls in `default_bin[f]`, the bin that
    contains 0.0.

    Stored entries whose value happens to bin to `default_bin[f]` are kept
    rather than compacted away; the accumulator handles the coincidence
    (see `histogram_sparse.mojo`), and dropping them would cost an extra
    pass for no change in results.
    """

    var row_index: List[Int]
    var bin: List[UInt8]
    var col_offsets: List[Int]
    var default_bin: List[UInt8]
    var n_rows: Int
    var n_features: Int
    var n_bins: Int

    def nnz(self) -> Int:
        return len(self.bin)

    def bin_at(self, row: Int, feature: Int) -> Int:
        """Bin of (row, feature), by binary search over the feature's stored
        entries. O(log nnz_f); the training path never uses it."""
        var lo = self.col_offsets[feature]
        var end = self.col_offsets[feature + 1]
        var hi = end
        while lo < hi:
            var mid = (lo + hi) // 2
            if self.row_index[mid] < row:
                lo = mid + 1
            else:
                hi = mid
        if lo < end and self.row_index[lo] == row:
            return Int(self.bin[lo])
        return Int(self.default_bin[feature])

    def to_dense(self) raises -> BinnedMatrix:
        """Densify to a `BinnedMatrix`. Test and benchmark helper only."""
        var size = self.n_rows * self.n_features
        var bins = List[UInt8](capacity=size)
        bins.resize(size, 0)
        for f in range(self.n_features):
            var col = f * self.n_rows
            for r in range(self.n_rows):
                bins[col + r] = self.default_bin[f]
            for i in range(self.col_offsets[f], self.col_offsets[f + 1]):
                bins[col + self.row_index[i]] = self.bin[i]
        return BinnedMatrix(bins^, self.n_rows, self.n_features, self.n_bins)

    def to_rows(self) raises -> SparseBinnedRows:
        """Row-oriented view for per-row tree walks. O(nnz + n_rows)."""
        var nnz = len(self.bin)
        var row_offsets = List[Int](capacity=self.n_rows + 1)
        row_offsets.resize(self.n_rows + 1, 0)
        for i in range(nnz):
            row_offsets[self.row_index[i] + 1] += 1
        for r in range(self.n_rows):
            row_offsets[r + 1] += row_offsets[r]

        var feature_index = List[Int](capacity=nnz)
        feature_index.resize(nnz, 0)
        var bin = List[UInt8](capacity=nnz)
        bin.resize(nnz, 0)
        var pos = List[Int](capacity=self.n_rows)
        for r in range(self.n_rows):
            pos.append(row_offsets[r])
        for f in range(self.n_features):
            for i in range(self.col_offsets[f], self.col_offsets[f + 1]):
                var p = pos[self.row_index[i]]
                feature_index[p] = f
                bin[p] = self.bin[i]
                pos[self.row_index[i]] = p + 1
        return SparseBinnedRows(
            feature_index^,
            bin^,
            row_offsets^,
            self.default_bin.copy(),
            self.n_rows,
            self.n_features,
            self.n_bins,
        )


@fieldwise_init
struct SparseBinnedRows(Copyable, Movable):
    """Row-oriented binned sparse data, for prediction.

    Row r owns entries `[row_offsets[r], row_offsets[r + 1])` with ascending
    feature indices, so `bin_at` binary-searches only that row's own entries
    rather than a whole feature column.
    """

    var feature_index: List[Int]
    var bin: List[UInt8]
    var row_offsets: List[Int]
    var default_bin: List[UInt8]
    var n_rows: Int
    var n_features: Int
    var n_bins: Int

    def bin_at(self, row: Int, feature: Int) -> Int:
        var lo = self.row_offsets[row]
        var end = self.row_offsets[row + 1]
        var hi = end
        while lo < hi:
            var mid = (lo + hi) // 2
            if self.feature_index[mid] < feature:
                lo = mid + 1
            else:
                hi = mid
        if lo < end and self.feature_index[lo] == feature:
            return Int(self.bin[lo])
        return Int(self.default_bin[feature])


def default_bins(mapper: BinMapper) raises -> List[UInt8]:
    """The bin holding the implicit zero, per feature."""
    var out = List[UInt8](capacity=mapper.n_features)
    for f in range(mapper.n_features):
        out.append(UInt8(mapper.bin_value(f, 0.0)))
    return out^


def transform_csc(
    mapper: BinMapper, csc: CscMatrix
) raises -> SparseBinnedMatrix:
    """Bin a sparse matrix, keeping it sparse.

    The result is bin-for-bin identical to `mapper.transform` on the
    densified matrix, because an absent entry bins exactly as the stored
    value 0.0 would.
    """
    csc.validate()
    if csc.n_features != mapper.n_features:
        raise Error("csc n_features must equal the mapper's n_features")

    var nnz = csc.nnz()
    var bins = List[UInt8](capacity=nnz)
    bins.resize(nnz, 0)

    var bins_p = bins.unsafe_ptr()
    var val_p = csc.values.unsafe_ptr()
    var off_p = csc.col_offsets.unsafe_ptr()
    var edges_p = mapper.edges.unsafe_ptr()
    var edge_offs_p = mapper.edge_offsets.unsafe_ptr()

    def do_feature(f: Int) {imm}:
        var elo = edge_offs_p.unsafe_load(f)
        var ehi = edge_offs_p.unsafe_load(f + 1)
        for i in range(off_p.unsafe_load(f), off_p.unsafe_load(f + 1)):
            var v = val_p.unsafe_load(i)
            var left = elo
            var right = ehi
            while left < right:
                var mid = (left + right) // 2
                if v <= edges_p.unsafe_load(mid):
                    right = mid
                else:
                    left = mid + 1
            bins_p.unsafe_store(i, UInt8(left - elo))

    dispatch_features(do_feature, csc.n_features, nnz + csc.n_features)

    return SparseBinnedMatrix(
        csc.row_index.copy(),
        bins^,
        csc.col_offsets.copy(),
        default_bins(mapper),
        csc.n_rows,
        csc.n_features,
        mapper.n_bins,
    )
