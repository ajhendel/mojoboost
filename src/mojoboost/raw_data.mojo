"""Raw training input: one type for dense and sparse feature matrices.

`RawData` is the seam between a caller holding user data and the binning
that every trainer starts from. It carries either a dense column-major
buffer (`values[f * n_rows + r]`) or a CSC matrix, answers the shape
questions both share, and dispatches binning to the matching
implementation. Nothing here converts one representation into the other: a
sparse input is binned by `sparse.fit_bins_csc` and `sparse.transform_csc`,
which never allocate `n_rows * n_features` of anything.

The two binning paths agree bit for bit on the same logical matrix (see
sparse.mojo), so which one a caller took is not observable in the fitted
bins, and a model trained either way predicts identically.

Who consumes this
-----------------
`trainset.Dataset` is the consumer. `Dataset.from_raw` takes a `RawData` by
value, bins it through the two methods below, and owns the result; the
sparse constructors (`Dataset.from_csc`, `Dataset.from_csr`) and every
dataset that retains its input (`keep_raw=True`, which is what `subset` and
reference binning need) go through it. The one path that does not build a
`RawData` is the dense `Dataset(features, ...)` constructor, which is handed
a borrowed matrix it must not copy; it bins that matrix in place with the
same `binning.fit_bins` this type dispatches to, so the two agree by
construction rather than by duplication.

`transform_dense` and `transform_sparse` still return different types,
because the dense grower reads a `BinnedMatrix` and the sparse one a
`SparseBinnedMatrix`. The branch has moved rather than vanished: `Dataset`
takes it once, at construction, and holds whichever matrix binning produced,
so callers of `train_dataset` and its counterparts no longer see it. Folding
the two into a single binned representation is still the step that would
remove it outright.
"""

from .binning import BinMapper, BinnedMatrix, fit_bins
from .sparse import (
    CscMatrix,
    CsrMatrix,
    SparseBinnedMatrix,
    fit_bins_csc,
    transform_csc,
)


def _empty_csc(n_rows: Int, n_features: Int) -> CscMatrix:
    """A structurally valid CSC matrix with no stored entries, which is what
    the sparse field holds while a `RawData` is dense."""
    var offsets = List[Int](capacity=n_features + 1)
    offsets.resize(n_features + 1, 0)
    return CscMatrix(
        List[Int](), List[Float64](), offsets^, n_rows, n_features
    )


struct RawData(Copyable, Movable):
    """A raw feature matrix, dense column-major or sparse CSC.

    Build one with `dense`, `from_csc`, or `from_csr`; the constructors
    validate the shape, so every later call can assume it. `is_sparse` says
    which representation is live, and the unused one is empty rather than
    absent, so the struct stays trivially copyable.
    """

    var values: List[Float64]
    var csc: CscMatrix
    var n_rows: Int
    var n_features: Int
    var is_sparse: Bool

    def __init__(
        out self,
        var values: List[Float64],
        var csc: CscMatrix,
        n_rows: Int,
        n_features: Int,
        is_sparse: Bool,
    ):
        self.values = values^
        self.csc = csc^
        self.n_rows = n_rows
        self.n_features = n_features
        self.is_sparse = is_sparse

    @staticmethod
    def dense(
        var values: List[Float64], n_rows: Int, n_features: Int
    ) raises -> RawData:
        """A column-major dense matrix, `values[f * n_rows + r]`."""
        if n_rows < 1 or n_features < 1:
            raise Error("matrix must have positive dimensions")
        if len(values) != n_rows * n_features:
            raise Error("features length must equal n_rows * n_features")
        return RawData(
            values^, _empty_csc(n_rows, n_features), n_rows, n_features, False
        )

    @staticmethod
    def from_csc(var csc: CscMatrix) raises -> RawData:
        """A CSC matrix, validated on the way in."""
        csc.validate()
        var n_rows = csc.n_rows
        var n_features = csc.n_features
        return RawData(List[Float64](), csc^, n_rows, n_features, True)

    @staticmethod
    def from_csr(csr: CsrMatrix) raises -> RawData:
        """A CSR matrix, transposed once to the feature-oriented layout the
        histogram builders need. O(nnz + n_features), no densification."""
        return RawData.from_csc(csr.to_csc())

    @staticmethod
    def none() -> RawData:
        """A `RawData` holding nothing, for a field that must exist whether
        or not the input was retained.

        `Dataset` keeps its raw matrix only when it was built with
        `keep_raw=True`, and a struct field cannot be absent, so a dataset
        that dropped its input holds this. `is_empty` is what distinguishes
        it from real data; every accessor that would read the matrix checks
        first and raises rather than reading a zero-sized one.
        """
        return RawData(List[Float64](), _empty_csc(0, 0), 0, 0, False)

    def is_empty(self) -> Bool:
        """Whether this holds no matrix at all, as `RawData.none()` does. A
        real matrix has positive dimensions, which the constructors check."""
        return self.n_rows < 1 or self.n_features < 1

    def nnz(self) -> Int:
        """Stored entries for a sparse matrix, every cell for a dense one."""
        if self.is_sparse:
            return self.csc.nnz()
        return self.n_rows * self.n_features

    def row(self, r: Int) raises -> List[Float64]:
        """One row as `n_features` raw values, for prediction. Absent sparse
        entries come back as 0.0, which is what they are."""
        if r < 0 or r >= self.n_rows:
            raise Error("row index out of range")
        if self.is_sparse:
            return self.csc.row(r)
        var out = List[Float64](capacity=self.n_features)
        for f in range(self.n_features):
            out.append(self.values[f * self.n_rows + r])
        return out^

    def check_rows(self, rows: List[Int]) raises:
        """The row-selection contract `subset` enforces, on its own so a
        caller can check a selection before building anything from it.

        Rows must be strictly ascending, in range, and there must be at
        least one. Ascending is not tidiness: a CSC column stores its row
        indices in ascending order, so a selection that reordered or
        repeated rows would either produce a matrix that violates that
        invariant or need a sort per column to repair it. Rejecting the
        selection says so once, where the caller can fix it, instead of
        making every later reader pay for the repair.
        """
        if len(rows) < 1:
            raise Error("a subset needs at least one row")
        for i in range(len(rows)):
            if rows[i] < 0 or rows[i] >= self.n_rows:
                raise Error("subset row index out of range")
            if i > 0 and rows[i] <= rows[i - 1]:
                raise Error("subset rows must be strictly ascending")

    def subset(self, rows: List[Int]) raises -> RawData:
        """The named rows, in the same representation, as their own matrix.

        This is row selection on the *raw* input, before any binning, which
        is what makes it safe to bin the result on its own: the rows that
        were left out had no say in the edges the subset fits. Selecting
        rows out of an already binned matrix would carry the whole matrix's
        quantiles into the part, which is the leak `trainset.Dataset.subset`
        exists to avoid.

        Sparse input stays sparse: the entries of the rows that were not
        taken are dropped and the survivors are renumbered, so the result
        costs O(nnz) and never allocates a dense cell.
        """
        self.check_rows(rows)
        var n = len(rows)
        if not self.is_sparse:
            var out = List[Float64](capacity=n * self.n_features)
            out.resize(n * self.n_features, 0.0)
            for f in range(self.n_features):
                var src = f * self.n_rows
                var dst = f * n
                for i in range(n):
                    out[dst + i] = self.values[src + rows[i]]
            return RawData(
                out^, _empty_csc(n, self.n_features), n, self.n_features, False
            )

        # Old row -> its position in the subset, -1 for a row not taken.
        var position = List[Int](capacity=self.n_rows)
        position.resize(self.n_rows, -1)
        for i in range(n):
            position[rows[i]] = i
        var row_index = List[Int]()
        var values = List[Float64]()
        var offsets = List[Int](capacity=self.n_features + 1)
        offsets.append(0)
        for f in range(self.n_features):
            var start = self.csc.col_offsets[f]
            var end = self.csc.col_offsets[f + 1]
            for e in range(start, end):
                var p = position[self.csc.row_index[e]]
                if p >= 0:
                    # `rows` ascends, so `p` does too within the column and
                    # the result is canonical CSC without a sort.
                    row_index.append(p)
                    values.append(self.csc.values[e])
            offsets.append(len(values))
        return RawData(
            List[Float64](),
            CscMatrix(row_index^, values^, offsets^, n, self.n_features),
            n,
            self.n_features,
            True,
        )

    def fit_mapper(
        self,
        max_bins: Int = 255,
        categorical_features: List[Int] = [],
        use_missing: Bool = True,
    ) raises -> BinMapper:
        """Fit quantile bin edges, category tables, and missing-bin
        reservations. Both representations produce the same mapper for the
        same logical matrix."""
        if self.is_sparse:
            return fit_bins_csc(
                self.csc, max_bins, categorical_features, use_missing
            )
        return fit_bins(
            self.values,
            self.n_rows,
            self.n_features,
            max_bins,
            categorical_features,
            use_missing,
        )

    def transform_dense(self, mapper: BinMapper) raises -> BinnedMatrix:
        """Bin a dense matrix. Raises on sparse input rather than densifying
        it: use `transform_sparse`."""
        if self.is_sparse:
            raise Error(
                "sparse input bins to a SparseBinnedMatrix; call"
                " transform_sparse"
            )
        return mapper.transform(self.values, self.n_rows)

    def transform_sparse(
        self, mapper: BinMapper
    ) raises -> SparseBinnedMatrix:
        """Bin a sparse matrix, keeping it sparse. Raises on dense input."""
        if not self.is_sparse:
            raise Error(
                "dense input bins to a BinnedMatrix; call transform_dense"
            )
        return transform_csc(mapper, self.csc)
