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

What this type does not do yet is pick a trainer. `transform_dense` and
`transform_sparse` return different types, because the dense grower reads a
`BinnedMatrix` and the sparse one a `SparseBinnedMatrix`, so a caller still
branches once on `is_sparse` after binning. Folding the two into a single
binned representation, so that `dataset.Dataset`, boosting, ranking,
multiclass, and the prediction paths take sparse input without knowing it,
is the remaining step; until it lands, sparse input reaches training only
through the sparse-only entry points, and `Dataset` stays dense.
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
