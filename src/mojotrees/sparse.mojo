"""Sparse feature matrices: CSC/CSR storage, sparse binning, binned data.

MojoTrees's histogram work is feature-oriented, so the internal
representation is compressed sparse column (CSC): every feature owns one
contiguous run of (row index, value) pairs. CSR input is transposed to CSC
once on the way in, and back to a row view for prediction. Neither direction
materializes a dense matrix: every routine here costs O(nnz + n_rows) or
O(nnz + n_features) memory, and the only functions that allocate
`n_rows * n_features` are the `to_dense` test helpers, which say so.

Absent entries, explicit zeros, and NaN
--------------------------------------
The three are specified independently:

- An **absent** entry, one the compressed structure does not store, is the
  numerical value `0.0`. A sparse matrix is therefore exactly equivalent to
  the dense matrix obtained by filling its gaps with zeros: absent entries
  participate in the quantile bin edges like any other value, land in
  whatever bin contains 0.0, and route through splits accordingly.
- An **explicitly stored zero** (SciPy keeps those until `eliminate_zeros()`
  is called) is treated identically to an absent one. Storing it or dropping
  it changes nothing about the fitted bins, the histograms, or the trees.
- A stored **NaN** is a missing value, exactly as on the dense path
  (see binning.mojo): it is excluded from the quantile computation, the
  feature reserves a bin for it when `use_missing` is set, and it routes by
  the node's learned default direction. NaN is the only missing marker;
  absence is not missingness. There is no way to express a missing value by
  leaving an entry out, which is the point of specifying the two separately.

This matches LightGBM's default `zero_as_missing=false`. LightGBM's
`zero_as_missing=true`, which reinterprets both absent entries and stored
zeros as missing, is not implemented, and no alias accepts it.

Canonical form
--------------
CSC requires strictly ascending row indices within each column, and CSR
strictly ascending column indices within each row. That is SciPy's canonical
form (`sum_duplicates()` then `sort_indices()`), and it also rules out
duplicate entries. `validate` enforces it and is called by every routine
that reads a matrix, so a malformed structure raises rather than reading out
of bounds or quietly binning the wrong rows.

Index widths are the caller's problem only at the boundary: indices are
`Int` here, so a producer holding 32-bit index arrays widens them on the way
in. `validate` bounds every index against the matrix shape either way.

Validation, once, at one place
------------------------------
`CscMatrix.validate` and `CsrMatrix.validate` check the raw form;
`SparseBinnedMatrix.validate` checks the binned one, including the
categorical rules stated at the bottom of this module. Every consumer
assumes all of it, so the check lives here rather than being re-derived by
each of them: the CPU trainers call it once per fit, and the device builder
calls it before uploading. `check_sparse_categorical_semantics` is the same
check with its findings returned instead of discarded.
"""

from std.math import isnan

from .binning import (
    BIN_SEARCH_LANES,
    DEFAULT_BIN_CONSTRUCT_SAMPLE_CNT,
    DEFAULT_DATA_RANDOM_SEED,
    DEFAULT_MIN_DATA_IN_BIN,
    POSITIVE_INF,
    BinMapper,
    BinnedMatrix,
    _sized_missing_bins,
    bin_construct_sample_rows,
    distinct_levels_sorted,
    emit_quantile_edges,
    no_missing_bins,
    quantile_boundary_indices,
)
from .categorical import (
    UNKNOWN_BIN,
    CategoricalSpec,
    _MAX_CATEGORY,
    _keep_most_frequent,
)
from .parallel import dispatch_features
from .validation import check_compressed


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

    def lookup(self, row: Int, feature: Int) -> Float64:
        """Value at (row, feature), 0.0 when the entry is absent. O(log
        nnz_f) by binary search over the feature's stored rows."""
        var lo = self.col_offsets[feature]
        var hi = self.col_offsets[feature + 1]
        var end = hi
        while lo < hi:
            var mid = (lo + hi) // 2
            if self.row_index[mid] < row:
                lo = mid + 1
            else:
                hi = mid
        if lo < end and self.row_index[lo] == row:
            return self.values[lo]
        return 0.0

    def row(self, r: Int) -> List[Float64]:
        """One row as `n_features` raw values, absent entries filled with
        0.0. Costs O(n_features + nnz_r) and allocates one row, not a
        matrix: this is the prediction-side accessor."""
        var out = List[Float64](capacity=self.n_features)
        out.resize(self.n_features, 0.0)
        for f in range(self.n_features):
            out[f] = self.lookup(r, f)
        return out^

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

    def row(self, r: Int) -> List[Float64]:
        """One row as `n_features` raw values, absent entries filled with
        0.0. O(n_features + nnz_r), one row's worth of memory."""
        var out = List[Float64](capacity=self.n_features)
        out.resize(self.n_features, 0.0)
        for i in range(self.row_offsets[r], self.row_offsets[r + 1]):
            out[self.col_index[i]] = self.values[i]
        return out^

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
    """Shared CSC/CSR structural validation, `validation.check_compressed`
    with this module's value list unpacked to its length. `outer` is the
    compressed axis (columns for CSC, rows for CSR) and `inner` the indexed
    one; every failure a malformed producer can hand over is caught before
    any index is dereferenced."""
    check_compressed(
        offsets, indices, len(values), n_outer, n_inner, kind, outer, inner
    )


def csc_from_dense[
    features_origin: ImmOrigin, //
](
    features: Span[Float64, features_origin], n_rows: Int, n_features: Int
) raises -> CscMatrix:
    """Build a CSC matrix from a column-major dense matrix, dropping zeros.
    Test and benchmark helper: real callers should never materialize the
    dense matrix in the first place. `NaN` is stored, since it is a missing
    value and not an absent entry (see the module docstring)."""
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


def csr_from_dense[
    features_origin: ImmOrigin, //
](
    features: Span[Float64, features_origin], n_rows: Int, n_features: Int
) raises -> CsrMatrix:
    """`csc_from_dense`'s row-oriented twin, from the same column-major dense
    matrix. Test and benchmark helper."""
    if len(features) != n_rows * n_features:
        raise Error("features length must equal n_rows * n_features")
    var col_index = List[Int]()
    var values = List[Float64]()
    var row_offsets = List[Int](capacity=n_rows + 1)
    row_offsets.append(0)
    for r in range(n_rows):
        for f in range(n_features):
            var v = features[f * n_rows + r]
            if v != 0.0:
                col_index.append(f)
                values.append(v)
        row_offsets.append(len(values))
    return CsrMatrix(col_index^, values^, row_offsets^, n_rows, n_features)


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


def _distinct_codes_and_counts_csc(
    csc: CscMatrix,
    feature: Int,
    mut codes: List[Int],
    mut counts: List[Int],
) raises:
    """Ascending distinct non-missing category codes of one sparse column,
    with their row counts, absent entries folded in as code 0.

    The dense counterpart is `categorical._distinct_codes_and_counts`; this
    one produces the same table for the same logical column without
    materializing it. Negative values and NaN are missing and excluded, as
    they are there.
    """
    var lo = csc.col_offsets[feature]
    var hi = csc.col_offsets[feature + 1]
    var present = List[Int]()
    for i in range(lo, hi):
        var v = csc.values[i]
        # `not (v >= 0.0)` also rejects NaN, matching `CategoricalSpec.bin_of`.
        if not (v >= 0.0):
            continue
        if v >= Float64(_MAX_CATEGORY):
            raise Error(
                "categorical feature values must be below 2^31; use smaller"
                " integer codes"
            )
        present.append(Int(v))
    sort(present)

    # Absent entries are the value 0.0, which is category code 0.
    var n_implicit = csc.n_rows - (hi - lo)
    var emitted_zero = False
    var i = 0
    while i < len(present):
        var j = i
        while j + 1 < len(present) and present[j + 1] == present[i]:
            j += 1
        var count = j - i + 1
        if present[i] == 0:
            count += n_implicit
            emitted_zero = True
        elif not emitted_zero and n_implicit > 0:
            # Codes are ascending, so 0 comes first when it is present at
            # all; reaching a larger code means it was not stored.
            codes.append(0)
            counts.append(n_implicit)
            emitted_zero = True
        codes.append(present[i])
        counts.append(count)
        i = j + 1
    if not emitted_zero and n_implicit > 0:
        codes.append(0)
        counts.append(n_implicit)


def fit_categorical_spec_csc(
    csc: CscMatrix,
    categorical_features: List[Int],
    max_bins: Int,
) raises -> CategoricalSpec:
    """`categorical.fit_categorical_spec` on a sparse matrix.

    Same tables, same tie-breaking, no densification: absent entries count as
    category 0, which is what they are.
    """
    if max_bins < 2:
        raise Error("max_bins must be at least 2")
    var n_features = csc.n_features
    var flags = List[Bool](capacity=n_features)
    for _ in range(n_features):
        flags.append(False)
    for i in range(len(categorical_features)):
        var f = categorical_features[i]
        if f < 0 or f >= n_features:
            raise Error("categorical feature index out of range")
        if flags[f]:
            raise Error("duplicate categorical feature index")
        flags[f] = True

    var codes = List[Int]()
    var offsets = List[Int](capacity=n_features + 1)
    offsets.append(0)
    for f in range(n_features):
        if flags[f]:
            var raw_codes = List[Int]()
            var raw_counts = List[Int]()
            _distinct_codes_and_counts_csc(csc, f, raw_codes, raw_counts)
            var kept = _keep_most_frequent(raw_codes, raw_counts, max_bins - 1)
            for i in range(len(kept)):
                codes.append(kept[i])
        offsets.append(len(codes))
    return CategoricalSpec(flags^, codes^, offsets^)


def _level_counts_sorted(
    stored: List[Float64],
    levels: List[Float64],
    n_implicit: Int,
    mut counts: List[Int],
):
    """How many rows of the implied dense column each level holds.

    `stored` is the column's stored values, ascending and NaN-free; `levels`
    is what `binning.distinct_levels_sorted` reported for it, with the
    implicit-zero level already spliced in when the caller had to add one.
    Both ascend, so this is one merge walk rather than a search per row.

    Why the counts are computed here and the levels are not. The levels are
    `binning.mojo`'s answer and stay `binning.mojo`'s answer, because the
    dense and sparse fits are required to agree on which levels exist. The
    counts are the one extra fact `min_data_in_bin` needs, and the dense side
    gets them fused into its distinct-value scan (`binning.collect_distinct`)
    where no sorted order exists to exploit. Here the values are already
    sorted, so counting is a walk of `len(stored)` comparisons and no table.

    The `n_implicit` absent rows are the value 0.0 and are added to whichever
    level holds zero. The caller guarantees that level exists whenever
    `n_implicit > 0`: it is the zero it splices in above, and the one case
    where it could not splice one in (the level table was already full)
    is exactly the case that gives up on levels altogether.

    Non-raising, because the sparse fitter calls this from inside a worker
    closure that cannot raise. The walk advances on `<` rather than on `!=`
    so that it lands on the right level for every value it is given and stays
    inside the array for any value it is not. `-0.0` and `0.0` compare equal
    under both, which is the normalization `distinct_levels_sorted` applies.
    """
    counts.clear()
    counts.resize(len(levels), 0)
    if len(levels) == 0:
        return
    var li = 0
    for i in range(len(stored)):
        var v = stored[i]
        while li + 1 < len(levels) and levels[li] < v:
            li += 1
        counts[li] += 1
    if n_implicit > 0:
        for j in range(len(levels)):
            if levels[j] == 0.0:
                counts[j] += n_implicit
                break


def _sample_membership(sample: List[Int], n_rows: Int) -> List[UInt64]:
    """A bitmap of the rows a sampled fit reads, one bit per row.

    Drawn once per fit and shared by every feature, so a column asks "is this
    stored row in the sample" in one load and one shift instead of merging
    against the ascending sample array. At 1,000,000 rows that is 125 KB read
    by every column, against 1.6 MB for the index array, and it is the whole
    reason a sampled sparse fit stays O(nnz) rather than
    O(n_features * sample_cnt).
    """
    var words = (n_rows + 63) // 64
    var bits = List[UInt64](capacity=words)
    bits.resize(words, UInt64(0))
    for i in range(len(sample)):
        var r = sample[i]
        bits[r >> 6] |= UInt64(1) << UInt64(r & 63)
    return bits^


def fit_bins_csc(
    csc: CscMatrix,
    max_bins: Int = 255,
    categorical_features: List[Int] = [],
    use_missing: Bool = True,
    min_data_in_bin: Int = DEFAULT_MIN_DATA_IN_BIN,
    bin_construct_sample_cnt: Int = DEFAULT_BIN_CONSTRUCT_SAMPLE_CNT,
    data_random_seed: Int = DEFAULT_DATA_RANDOM_SEED,
) raises -> BinMapper:
    """Fit quantile bin edges on a sparse matrix, without densifying.

    Produces bit-identical edges, category tables, and missing-bin
    reservations to `binning.fit_bins` on the densified matrix, **at the same
    arguments and therefore at the default**. Per feature it sorts only the
    stored values (O(nnz_f log nnz_f)) and indexes the implied dense sorted
    column through `_sorted_column_at` instead of materializing n_rows values.

    `categorical_features` and `use_missing` mean exactly what they mean on
    the dense path (see binning.mojo): named features get a category table
    and no edges, and a column holding any stored `NaN` reserves its highest
    bin for missing values, fitting its edges over the remaining
    `max_bins - 1` bins. Absent entries are the value 0.0 throughout, so they
    enter the quantiles and never the missing bin.

    `min_data_in_bin` and `bin_construct_sample_cnt` also mean what they mean
    there, and default to LightGBM's 3 and 200,000 as `fit_bins` does. Neither
    reached this function when they landed, which broke the equivalence stated
    at the top of this module -- the dense side merged rare levels and drew a
    sample and this one did neither -- so both are honored here:

    - `min_data_in_bin` counts the rows behind each level of the implied dense
      column, absent rows included, and merges levels in ascending order until
      each bin holds the minimum; past the bin budget it shrinks the budget to
      `max(1, min(max_bins, n_valid / min_data_in_bin))`, which is the same
      pair of `GreedyFindBin` branches `fit_bins` runs.
    - `bin_construct_sample_cnt` draws the same rows `fit_bins` draws, from
      `binning.bin_construct_sample_rows` on the same `n_rows` and seed, so a
      sampled sparse fit reads the same rows of the same matrix. Sampling
      changes how many rows of a column are absent, so `n_implicit` becomes
      "sampled rows with no stored entry" rather than "rows with no stored
      entry", and every count, rank and missingness test below is against the
      sample rather than against the matrix. A matrix of fewer rows than the
      count is not sampled at all, which is the common case and costs nothing.
    """
    if max_bins < 2 or max_bins > 256:
        raise Error("max_bins must be in [2, 256]")
    if min_data_in_bin < 1:
        raise Error("min_data_in_bin must be positive")
    if bin_construct_sample_cnt < 0:
        raise Error("bin_construct_sample_cnt must be non-negative")
    csc.validate()

    var cats = fit_categorical_spec_csc(csc, categorical_features, max_bins)

    var n_rows = csc.n_rows
    var n_features = csc.n_features
    var max_edges = max_bins - 1
    var scratch = List[Float64](capacity=n_features * max_edges)
    scratch.resize(n_features * max_edges, 0.0)
    var counts = List[Int](capacity=n_features)
    counts.resize(n_features, 0)
    var missing_bin = no_missing_bins(n_features)

    var scratch_p = scratch.unsafe_ptr()
    var counts_p = counts.unsafe_ptr()
    var missing_p = missing_bin.unsafe_ptr()
    var val_p = csc.values.unsafe_ptr()
    var off_p = csc.col_offsets.unsafe_ptr()
    var row_p = csc.row_index.unsafe_ptr()
    ref spec = cats

    # Drawn once, before any feature is dispatched, exactly as `fit_bins`
    # draws it and from the same function on the same `n_rows` and seed, so
    # the dense and sparse fits of one matrix read the same rows. Empty means
    # every row, which is the default below 200,000 rows and keeps the column
    # loop the loop it has always been.
    var sample = bin_construct_sample_rows(
        n_rows, bin_construct_sample_cnt, data_random_seed
    )
    var sampled = len(sample) > 0
    var n_sample = len(sample) if sampled else n_rows
    var sample_bits = _sample_membership(sample, n_rows) if sampled else List[
        UInt64
    ]()
    var bits_p = sample_bits.unsafe_ptr()

    def do_feature(f: Int) {imm}:
        # Categorical columns never enter quantile binning.
        if spec.is_cat(f):
            counts_p.unsafe_store(f, 0)
            return
        var lo = off_p.unsafe_load(f)
        var hi = off_p.unsafe_load(f + 1)
        var n_stored = hi - lo
        # NaN is dropped before the sort, so it never takes part in a
        # quantile comparison, exactly as on the dense path.
        #
        # Under a sample, a stored entry counts only when its row was drawn,
        # and `n_stored_seen` counts the drawn entries whether or not they
        # were NaN -- it is what the sampled rows of this column store, and
        # the rest of the drawn rows are the absent ones.
        var stored = List[Float64](capacity=n_stored)
        var n_stored_seen = n_stored
        if sampled:
            n_stored_seen = 0
            for i in range(lo, hi):
                var r = row_p.unsafe_load(i)
                if (bits_p.unsafe_load(r >> 6) >> UInt64(r & 63)) & 1 == 0:
                    continue
                n_stored_seen += 1
                var v = val_p.unsafe_load(i)
                if not isnan(v):
                    stored.append(v)
        else:
            for i in range(lo, hi):
                var v = val_p.unsafe_load(i)
                if not isnan(v):
                    stored.append(v)
        sort(stored)

        var n_implicit = n_sample - n_stored_seen
        var n_valid = len(stored) + n_implicit
        var reserve = use_missing and n_valid < n_sample
        var n_ordinary = max_bins - 1 if reserve else max_bins
        var n_neg = 0
        while n_neg < len(stored) and stored[n_neg] < 0.0:
            n_neg += 1
        var n_stored_zero = 0
        while (
            n_neg + n_stored_zero < len(stored)
            and stored[n_neg + n_stored_zero] == 0.0
        ):
            n_stored_zero += 1
        var n_zero = n_stored_zero + n_implicit

        # Which boundaries exist and how each becomes an edge are
        # `binning.mojo`'s rules, called rather than restated: the dense and
        # sparse fits are required to agree edge for edge, and two copies of
        # the midpoint-clamp-and-dedupe loop is the way that stops being true.
        # All this path supplies is where a rank's value lives in the implied
        # dense sorted column.
        var below = List[Float64]()
        var above = List[Float64]()

        # Few enough levels to give each one a bin, which is the same
        # question and the same answer as on the dense path. The stored
        # values are sorted, so counting them is a walk; the absent entries
        # add the level 0.0, and only when no stored value is already zero.
        var levels = List[Float64]()
        var levels_fit = distinct_levels_sorted(
            stored, len(stored), n_ordinary, levels
        )
        if levels_fit and n_implicit > 0:
            var has_zero = False
            for i in range(len(levels)):
                if levels[i] == 0.0:
                    has_zero = True
                    break
                if levels[i] > 0.0:
                    break
            if not has_zero:
                if len(levels) >= n_ordinary:
                    levels_fit = False
                else:
                    levels.append(0.0)
                    sort(levels)
        elif not levels_fit and len(stored) == 0 and n_implicit > 0:
            # Nothing stored and rows absent: one level, the implicit zero.
            # `distinct_levels_sorted` was handed an empty column and could
            # not say so.
            levels.append(0.0)
            levels_fit = True

        if levels_fit:
            if min_data_in_bin <= 1:
                for j in range(len(levels) - 1):
                    below.append(levels[j])
                    above.append(levels[j + 1])
            else:
                # `fit_bins`'s levels branch, over the same levels: walk them
                # accumulating their row counts and cut only once the
                # accumulator has reached the minimum, resetting it at each
                # cut. The absent rows are counted into the zero level by
                # `_level_counts_sorted`, which is the whole difference
                # between this and the dense loop.
                var level_counts = List[Int]()
                _level_counts_sorted(stored, levels, n_implicit, level_counts)
                var acc = 0
                for j in range(len(levels) - 1):
                    acc += level_counts[j]
                    if acc >= min_data_in_bin:
                        below.append(levels[j])
                        above.append(levels[j + 1])
                        acc = 0
        else:
            # `fit_bins`'s other branch shrinks the bin budget before it cuts:
            # `max_bin = max(1, min(max_bin, total_cnt / min_data_in_bin))`.
            # `total_cnt` is the implied dense column's row count, absent rows
            # included, which is `n_valid`.
            var n_ord = n_ordinary
            if min_data_in_bin > 1:
                var cap = n_valid // min_data_in_bin
                if cap < 1:
                    cap = 1
                if cap < n_ord:
                    n_ord = cap
            var idxs = List[Int]()
            quantile_boundary_indices(n_valid, n_ord, idxs)
            for j in range(len(idxs)):
                var idx = idxs[j]
                var w = _sorted_column_at(
                    stored, n_neg, n_zero, n_implicit, idx - 1
                )
                below.append(w)
                # The next distinct value above `w`, which is what the edge
                # rule cuts against (see `binning.emit_quantile_edges`). The
                # implied dense column is sorted and indexable by rank, so the
                # run `w` belongs to is found by bisecting it rather than
                # walking it -- which matters here, where a column's implicit
                # zeros can be a run of nearly every row.
                var left = idx
                var right = n_valid
                while left < right:
                    var mid = (left + right) // 2
                    if (
                        _sorted_column_at(
                            stored, n_neg, n_zero, n_implicit, mid
                        )
                        > w
                    ):
                        right = mid
                    else:
                        left = mid + 1
                if left < n_valid:
                    above.append(
                        _sorted_column_at(
                            stored, n_neg, n_zero, n_implicit, left
                        )
                    )
                else:
                    above.append(w)
        var edge_buf = List[Float64]()
        emit_quantile_edges(below, above, edge_buf)

        var out = scratch_p.unsafe_offset(f * max_edges)
        var n_out = len(edge_buf)
        for i in range(n_out):
            out.unsafe_store(i, edge_buf[i])
        counts_p.unsafe_store(f, n_out)
        # k edges give ordinary bins 0..k, so the missing bin is k + 1.
        missing_p.unsafe_store(f, n_out + 1 if reserve else -1)

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
    return BinMapper(
        edges^, offsets^, n_features, max_bins, cats^, missing_bin^
    )


struct SparseBinnedMatrix(Copyable, Movable):
    """Binned CSC matrix.

    Stored entry i belongs to row `row_index[i]` and falls in bin `bin[i]`;
    feature f owns `[col_offsets[f], col_offsets[f + 1])`. Every row without
    a stored entry for feature f falls in `default_bin[f]`, the bin that
    contains 0.0.

    `cats` and `missing_bin` carry the same meaning as on `BinnedMatrix`: the
    categorical features and the per-feature bin reserved for missing values
    (-1 for a feature that reserves none). They are properties of the fitted
    binning, so they are the mapper's, copied here for the accumulators and
    split search that read them.

    Stored entries whose value happens to bin to `default_bin[f]` are kept
    rather than compacted away; the accumulator handles the coincidence
    (see `histogram_sparse.mojo`), and dropping them would cost an extra
    pass for no change in results. A stored `NaN` bins to `missing_bin[f]`,
    which is never a default bin, so missing rows stay distinguishable from
    absent ones.
    """

    var row_index: List[Int]
    var bin: List[UInt8]
    var col_offsets: List[Int]
    var default_bin: List[UInt8]
    var n_rows: Int
    var n_features: Int
    var n_bins: Int
    var cats: CategoricalSpec
    var missing_bin: List[Int]

    def __init__(
        out self,
        var row_index: List[Int],
        var bin: List[UInt8],
        var col_offsets: List[Int],
        var default_bin: List[UInt8],
        n_rows: Int,
        n_features: Int,
        n_bins: Int,
        var cats: CategoricalSpec = CategoricalSpec.none(),
        var missing_bin: List[Int] = [],
    ):
        """A `missing_bin` table of the wrong length (the empty default
        included) means no feature reserves a missing bin, matching
        `BinnedMatrix`."""
        self.row_index = row_index^
        self.bin = bin^
        self.col_offsets = col_offsets^
        self.default_bin = default_bin^
        self.n_rows = n_rows
        self.n_features = n_features
        self.n_bins = n_bins
        self.cats = cats^
        self.missing_bin = _sized_missing_bins(missing_bin^, n_features)

    def nnz(self) -> Int:
        return len(self.bin)

    def validate(self) raises:
        """Structural and categorical validation of the binned form.

        The counterpart of `CscMatrix.validate` one level down, and the one
        authoritative statement of what a `SparseBinnedMatrix` must be. Every
        consumer of this structure -- the CPU grower, the entry partition,
        the device upload -- assumes all of it, and each of them used to
        assume a different subset: the device builder checked row and bin
        ranges inline, the accumulators checked nothing, and the categorical
        rules were stated only in the GPU categorical module, where a CPU
        caller could not reach them without pulling in a device dependency.
        Callers now check once, here, at the point a matrix enters training.

        Costs one pass over the stored entries, so it belongs once per fit
        and not once per tree.
        """
        var nnz = len(self.bin)
        if self.n_rows < 1 or self.n_features < 1:
            raise Error("binned sparse matrix must have positive dimensions")
        if self.n_bins < 1 or self.n_bins > 256:
            raise Error("binned sparse matrix must have 1 to 256 bins")
        if len(self.row_index) != nnz:
            raise Error("row_index and bin must have equal length")
        if len(self.col_offsets) != self.n_features + 1:
            raise Error("col_offsets must have length n_features + 1")
        if self.col_offsets[0] != 0:
            raise Error("col_offsets must start at 0")
        if self.col_offsets[self.n_features] != nnz:
            raise Error("col_offsets must end at nnz")
        if len(self.default_bin) != self.n_features:
            raise Error("default_bin must have one entry per feature")
        if len(self.missing_bin) != self.n_features:
            raise Error("missing_bin must have one entry per feature")
        for f in range(self.n_features):
            var lo = self.col_offsets[f]
            var hi = self.col_offsets[f + 1]
            if hi < lo:
                raise Error("col_offsets must be non-decreasing")
            if Int(self.default_bin[f]) >= self.n_bins:
                raise Error("default bin out of range")
            if self.missing_bin[f] < -1 or self.missing_bin[f] >= self.n_bins:
                raise Error("missing bin out of range")
            # The default bin holds the implicit zeros and the missing bin
            # holds `NaN`; a feature whose two coincided would make an absent
            # entry indistinguishable from a missing one, which is the single
            # distinction this whole representation is built on.
            if self.missing_bin[f] == Int(self.default_bin[f]):
                raise Error(
                    "a feature's missing bin must not be its default bin"
                )
            for i in range(lo, hi):
                var r = self.row_index[i]
                if r < 0 or r >= self.n_rows:
                    raise Error("stored entry row index out of range")
                # Ascending row order per column is what `bin_at`'s binary
                # search and the stable entry partition both read.
                if i > lo and r <= self.row_index[i - 1]:
                    raise Error(
                        "row indices must be strictly ascending within each"
                        " column"
                    )
                if Int(self.bin[i]) >= self.n_bins:
                    raise Error("stored entry bin out of range")
        var flagged = List[Int]()
        _check_categorical_columns(self, flagged)

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

    def is_missing(self, row: Int, feature: Int) -> Bool:
        """Whether (row, feature) holds a missing value. Absent entries are
        zeros, not missing values, so this is False for them unless the
        feature's default bin is itself the missing bin, which cannot
        happen."""
        return self.bin_at(row, feature) == self.missing_bin[feature]

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
        return BinnedMatrix(
            bins^,
            self.n_rows,
            self.n_features,
            self.n_bins,
            self.cats.copy(),
            self.missing_bin.copy(),
        )

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

    def bins_of_row(self, row: Int) -> List[Int]:
        """Every feature's bin for one row, the `bin_row` of the sparse path.
        Absent entries take their feature's default bin. O(n_features +
        nnz_r), one row's worth of memory."""
        var out = List[Int](capacity=self.n_features)
        for f in range(self.n_features):
            out.append(Int(self.default_bin[f]))
        for i in range(self.row_offsets[row], self.row_offsets[row + 1]):
            out[self.feature_index[i]] = Int(self.bin[i])
        return out^


def default_bins(mapper: BinMapper) raises -> List[UInt8]:
    """The bin holding the implicit zero, per feature.

    For a categorical feature this is the bin of category 0, or the unknown
    bin when 0 was not kept; for a numerical one it is whichever ordinary bin
    contains 0.0. It is never a missing bin, because `bin_value` routes only
    `NaN` there.
    """
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
    value 0.0 would. Categorical features route through the mapper's category
    tables and stored `NaN` through the reserved missing bin, both exactly as
    the dense transform does -- and now by the same code shape as well, the
    padded lockstep descent of `BIN_SEARCH_LANES`, so the two sides of that
    identity claim can no longer drift apart by one of them being rewritten.
    """
    csc.validate()
    if csc.n_features != mapper.n_features:
        raise Error("csc n_features must equal the mapper's n_features")

    var nnz = csc.nnz()
    var bins = List[UInt8](capacity=nnz)
    bins.resize(nnz, 0)

    # The same per-feature search table `BinMapper.transform` builds, for the
    # same reason and by the same arithmetic: each numerical feature's edges
    # padded up to a power of two with `+inf` sentinels, so a value's bin is a
    # fixed, branch-free descent rather than a data-dependent loop. A sentinel
    # never counts (`+inf < v` is false for every `v`, `+inf` included), so
    # padding cannot move a bin, and `BinMapper.bin_value` keeps the plain
    # search as the reference both descents are compared against.
    #
    # Built here rather than shared with `binning.mojo` because `BinMapper`
    # does not carry the table as a field; the dense transform builds its own
    # local copy in the same way. A shared builder belongs in `binning.mojo`
    # beside `POSITIVE_INF` and is named in this lane's report; duplicating
    # fifteen lines was preferred to reaching into a file this lane does not
    # own. The two constants are imported rather than respelled so that the
    # lane width and the sentinel cannot drift apart.
    var pad = List[Float64](capacity=mapper.n_features * mapper.n_bins)
    var pad_offsets = List[Int](capacity=mapper.n_features + 1)
    var pad_half = List[Int](capacity=mapper.n_features)
    pad_offsets.append(0)
    for f in range(mapper.n_features):
        var elo = mapper.edge_offsets[f]
        var k = mapper.edge_offsets[f + 1] - elo
        # Smallest power of two strictly greater than k, so the descent's
        # largest reachable answer (k) is representable and a feature with no
        # edges still gets one sentinel to point at.
        var m = 1
        while m <= k:
            m += m
        for i in range(k):
            pad.append(mapper.edges[elo + i])
        for _ in range(k, m):
            pad.append(POSITIVE_INF)
        pad_offsets.append(len(pad))
        pad_half.append(m >> 1)

    var bins_p = bins.unsafe_ptr()
    var val_p = csc.values.unsafe_ptr()
    var off_p = csc.col_offsets.unsafe_ptr()
    var pad_p = pad.unsafe_ptr()
    var poff_p = pad_offsets.unsafe_ptr()
    var half_p = pad_half.unsafe_ptr()
    var miss_p = mapper.missing_bin.unsafe_ptr()
    ref cats = mapper.cats
    comptime R = BIN_SEARCH_LANES

    def do_feature(f: Int) {imm}:
        var lo = off_p.unsafe_load(f)
        var hi = off_p.unsafe_load(f + 1)
        if cats.is_cat(f):
            for i in range(lo, hi):
                bins_p.unsafe_store(
                    i, UInt8(cats.bin_of(f, val_p.unsafe_load(i)))
                )
            return
        var pbase = poff_p.unsafe_load(f)
        var half = half_p.unsafe_load(f)
        var mb = miss_p.unsafe_load(f)

        # THE BIN A STORED NaN TAKES, resolved once per column instead of once
        # per entry. It is what lets the entry loop run in lockstep: the shape
        # this replaces branched on `isnan` *before* the descent and left it by
        # `continue`, so interleaved entries would have exited at different
        # steps and there would have been no lockstep to run.
        #
        # `mb >= 0` gives the reserved missing bin. `mb < 0` means no bin was
        # reserved, and LightGBM then bins a NaN as if it were `0.0`
        # (`BinMapper.bin_value`, and the missing_type None rule it cites),
        # which is the descent's answer for a constant and so does not depend
        # on the entry. Either way a NaN's bin is a per-column constant.
        #
        # And the descent is safe to run on a NaN, which is the identity this
        # rests on. Every IEEE comparison against NaN is false, so
        # `border < NaN` is false at every step, `go` is never taken, `pos`
        # stays 0, and the search reads in range and takes exactly the same
        # number of steps as any other value. Its answer is then discarded by
        # the select below.
        var nan_bin = mb
        if nan_bin < 0:
            var p0 = 0
            var s0 = half
            while s0 > 0:
                var n0 = p0 + s0
                var g0 = pad_p.unsafe_load(pbase + n0 - 1) < 0.0
                p0 = n0 if g0 else p0
                s0 = s0 >> 1
            nan_bin = p0

        # Count the edges strictly below `v`, which is the bin the plain
        # search arrived at: it stopped at the first edge with `v <= edge`,
        # and the edges are strictly increasing.
        #
        # `BIN_SEARCH_LANES` stored entries at a time. A column's entries are
        # contiguous in `csc.values` and their bins are contiguous in `bins`,
        # exactly as a dense column's rows are, so the interleave carries over
        # unchanged. The lanes share nothing: each has its own `pos`, reads
        # only its own value, and writes only its own cell. `step` is shared
        # because the trip count is `log2` of the padded table size, a
        # property of the feature and not of the data.
        #
        # BIT-IDENTITY. Lane `k` executes, in order, the same comparisons
        # against the same borders with the same value the plain loop executed
        # for that entry, and arrives at `pos` by the same selects. No float
        # arithmetic exists here to reassociate: the only float operation is a
        # comparison, and a comparison is exact. So every bin byte is the byte
        # the plain loop wrote, and `transform_csc`'s standing claim of
        # bin-for-bin agreement with `BinMapper.transform` holds against the
        # dense side's matching descent rather than merely alongside it. The
        # tail below is the scalar form of the same descent.
        var i = lo
        var vec_end = hi - R + 1
        while i < vec_end:
            var v = val_p.unsafe_load[width=R, alignment=8](i)
            var pos = SIMD[DType.int, R](0)
            var step = half
            while step > 0:
                comptime for k in range(R):
                    var cur = Int(pos[k])
                    var nxt = cur + step
                    var go = pad_p.unsafe_load(pbase + nxt - 1) < v[k]
                    pos[k] = nxt if go else cur
                step = step >> 1
            comptime for k in range(R):
                var b = nan_bin if isnan(v[k]) else Int(pos[k])
                bins_p.unsafe_store(i + k, UInt8(b))
            i += R
        while i < hi:
            var v = val_p.unsafe_load(i)
            var pos = 0
            var step = half
            while step > 0:
                var nxt = pos + step
                var go = pad_p.unsafe_load(pbase + nxt - 1) < v
                pos = nxt if go else pos
                step = step >> 1
            var b = nan_bin if isnan(v) else pos
            bins_p.unsafe_store(i, UInt8(b))
            i += 1

    # Unchanged by the interleave: the same compares are done, just
    # `BIN_SEARCH_LANES` entries' worth at a time, so the same work threshold
    # picks the same schedule it picked before.
    dispatch_features(do_feature, csc.n_features, nnz + csc.n_features)

    return SparseBinnedMatrix(
        csc.row_index.copy(),
        bins^,
        csc.col_offsets.copy(),
        default_bins(mapper),
        csc.n_rows,
        csc.n_features,
        mapper.n_bins,
        mapper.cats.copy(),
        mapper.missing_bin.copy(),
    )


# --- Categorical semantics of a sparse column -----------------------------
#
# These three used to live in `gpu_categorical.mojo`, which made them
# unreachable from the CPU sparse path without importing a device module.
# They are statements about this representation, not about a device, so they
# belong here: `gpu_categorical.mojo` and `gpu_sparse_layout.mojo` import
# them, and so does `boosting_sparse.mojo`. One implementation, three
# consumers, no device dependency for any of them.


def default_category_bin(cats: CategoricalSpec, feature: Int) raises -> Int:
    """The bin an absent entry of a categorical feature falls in.

    Identical to `default_bins(mapper)[feature]` for a categorical column,
    computed from the category table alone so a caller holding a
    `SparseBinnedMatrix` and no mapper can check the matrix against it.
    """
    if not cats.is_cat(feature):
        raise Error("feature is not categorical")
    return cats.bin_of(feature, 0.0)


def absent_is_unknown(cats: CategoricalSpec, feature: Int) raises -> Bool:
    """Whether absent entries of `feature` land in the unknown bin.

    An absent entry of a categorical column is the value 0.0, which is
    *category code 0*, so it takes category 0's own bin when the fitted table
    kept that code. When the table did not keep it (the column never held it,
    or it was dropped as too rare for `max_bins - 1`) the default bin is
    `UNKNOWN_BIN`, and bin 0 is never a member of a split's category set, so
    every absent row then routes right at every categorical node of this
    feature, together with the missing, unseen, and dropped rows.

    Both are correct and the second is the trap: a one-hot-ish column whose
    zeros mean "not this category" behaves completely differently depending
    on whether 0 survived the table, and nothing in the numbers says so. This
    is a modelling fact, reportable before a model is fitted rather than
    discovered inside one.
    """
    return default_category_bin(cats, feature) == UNKNOWN_BIN


def _check_categorical_columns(
    data: SparseBinnedMatrix, mut flagged: List[Int]
) raises:
    """Check every categorical column and append the ascending feature ids
    for which `absent_is_unknown` holds.

    Three things must hold, all of them properties a malformed producer could
    break without any individual number looking wrong:

    - the column's `default_bin` must be the bin of category code 0, because
      an absent entry is the value 0.0 and nothing else;
    - the column must reserve no missing bin: a categorical feature routes
      missing values to `UNKNOWN_BIN` by construction, and a reserved bin
      would give it a second, contradictory missing route;
    - every stored bin must be inside `[0, n_categories]`, since a bin past
      the table indexes a category that was never fitted.
    """
    for f in range(data.n_features):
        if not data.cats.is_cat(f):
            continue
        var expected = default_category_bin(data.cats, f)
        if Int(data.default_bin[f]) != expected:
            raise Error(
                "categorical column's default bin is not the bin of"
                " category code 0"
            )
        if data.missing_bin[f] >= 0:
            raise Error("categorical feature must not reserve a missing bin")
        var n_cat = data.cats.n_categories(f)
        for i in range(data.col_offsets[f], data.col_offsets[f + 1]):
            if Int(data.bin[i]) > n_cat:
                raise Error("stored bin is past the fitted category table")
        if expected == UNKNOWN_BIN:
            flagged.append(f)


def check_sparse_categorical_semantics(
    data: SparseBinnedMatrix,
) raises -> List[Int]:
    """Check a binned sparse matrix against the categorical rules, and report
    which categorical features send their absent rows to the unknown bin.

    The reporting half of `SparseBinnedMatrix.validate`, for a caller that
    wants the flagged features rather than only the guarantee.
    """
    var flagged = List[Int]()
    _check_categorical_columns(data, flagged)
    return flagged^
