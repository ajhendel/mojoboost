"""Sparse (CSC/CSR) training and prediction.

Three kinds of check:

- structural validation of the CSC/CSR forms and their conversions;
- equivalence with the dense path, which is the whole contract of the
  sparse implementation: an implicit zero is a numerical zero, so a sparse
  fit must agree with the dense fit of the same matrix with the gaps filled
  in. Bin edges and bin ids agree exactly; histograms and predictions agree
  to floating-point rounding, because the sparse accumulator derives each
  feature's zero bin by subtraction (the same trade the dense path already
  makes for sibling subtraction);
- determinism: results must not depend on how the per-feature work was
  scheduled.
"""

from std.os import remove, setenv
from std.testing import assert_equal, assert_true, assert_false, TestSuite

from mojoboost.bagging import BaggingParams
from mojoboost.binning import BinMapper, fit_bins
from mojoboost.boosting import (
    BINARY_LOGISTIC,
    L1,
    QUANTILE,
    SQUARED_ERROR,
    Booster,
    BoosterParams,
    train,
    train_multiclass,
    train_multiclass_with_valid,
    train_with_valid,
)
from mojoboost.boosting_sparse import (
    train_multiclass_sparse,
    train_sparse,
    train_sparse_with_valid,
)
from mojoboost.histogram import build_histogram, build_histogram_subset
from mojoboost.histogram_sparse import (
    SparseEntryOrder,
    SparseNodeEntries,
    build_histogram_sparse,
    build_histogram_sparse_subset,
    sum_all,
)
from mojoboost.model import fit
from mojoboost.model_sparse import (
    fit_csc,
    fit_multiclass_csc,
    predict_csr,
    predict_proba_csr,
    predict_raw_csr,
)
from mojoboost.serialize import load_model, save_model
from mojoboost.sparse import (
    CscMatrix,
    CsrMatrix,
    csc_from_dense,
    fit_bins_csc,
    transform_csc,
)
from mojoboost.tree import Tree, TreeParams, grow_tree
from mojoboost.tree_sparse import grow_tree_sparse, predict_row_sparse


comptime _TMP_PATH = "./.test_sparse_roundtrip.tmp"


def _splitmix64(state: UInt64) -> UInt64:
    var z = state + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _uniform(counter: UInt64) -> Float64:
    return Float64(_splitmix64(counter) >> 11) * (1.0 / 9007199254740992.0)


def _sparse_dense(
    n_rows: Int, n_features: Int, density: Float64, seed: UInt64
) -> List[Float64]:
    """A column-major matrix that is mostly exact zeros, with values on both
    sides of zero so the zero bin lands in the middle of the range."""
    var out = List[Float64](capacity=n_rows * n_features)
    for k in range(n_rows * n_features):
        var u = _uniform(seed + UInt64(k))
        if u < density:
            out.append(4.0 * (u / density) - 2.0)
        else:
            out.append(0.0)
    return out^


def _target(dense: List[Float64], n_rows: Int, seed: UInt64) -> List[Float64]:
    """A linear target with a distinct coefficient per feature.

    Deliberately asymmetric: two features that enter the target the same way
    produce split candidates whose gains agree to the last few bits, and
    which of them wins is then decided by rounding rather than by the data.
    That is a property of the dataset, not of either accumulator, and it
    would make the equivalence assertions below measure noise.
    """
    var n_features = len(dense) // n_rows
    var out = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        var total = 0.0
        for f in range(n_features):
            total += (1.0 + 0.37 * Float64(f)) * dense[f * n_rows + r]
        out.append(total / 4.0 + 0.05 * (_uniform(seed + UInt64(r)) - 0.5))
    return out^


def _row(dense: List[Float64], n_rows: Int, n_features: Int, r: Int) -> List[
    Float64
]:
    var row = List[Float64](capacity=n_features)
    for f in range(n_features):
        row.append(dense[f * n_rows + r])
    return row^


def _grads(n_rows: Int, seed: UInt64) -> List[Float64]:
    var g = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        g.append(2.0 * _uniform(seed + UInt64(r)) - 1.0)
    return g^


def _hessians(n_rows: Int, seed: UInt64) -> List[Float64]:
    var h = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        h.append(_uniform(seed + UInt64(r)) + 0.01)
    return h^


def _raises(matrix: CscMatrix) -> Bool:
    try:
        matrix.validate()
    except:
        return True
    return False


# ---------------------------------------------------------------- structure


def test_csc_validation() raises:
    var rows: List[Int] = [0, 2, 1]
    var vals: List[Float64] = [1.0, 2.0, 3.0]
    var offs: List[Int] = [0, 2, 3]
    CscMatrix(rows.copy(), vals.copy(), offs.copy(), 3, 2).validate()

    # Offsets of the wrong length, not starting at 0, or not ending at nnz.
    var short: List[Int] = [0, 2]
    assert_true(_raises(CscMatrix(rows.copy(), vals.copy(), short^, 3, 2)))
    var nonzero_start: List[Int] = [1, 2, 3]
    assert_true(
        _raises(CscMatrix(rows.copy(), vals.copy(), nonzero_start^, 3, 2))
    )
    var short_end: List[Int] = [0, 2, 2]
    assert_true(_raises(CscMatrix(rows.copy(), vals.copy(), short_end^, 3, 2)))

    # An empty trailing column is legal.
    var empty_last: List[Int] = [0, 3, 3]
    var ascending: List[Int] = [0, 1, 2]
    assert_false(
        _raises(CscMatrix(ascending^, vals.copy(), empty_last^, 3, 2))
    )

    # Row index out of range.
    var oob: List[Int] = [0, 9, 1]
    assert_true(_raises(CscMatrix(oob^, vals.copy(), offs.copy(), 3, 2)))

    # Row indices must be strictly ascending inside a column, which also
    # rules out duplicate entries.
    var unsorted: List[Int] = [2, 0, 1]
    assert_true(_raises(CscMatrix(unsorted^, vals.copy(), offs.copy(), 3, 2)))
    var duplicate: List[Int] = [1, 1, 0]
    assert_true(_raises(CscMatrix(duplicate^, vals.copy(), offs.copy(), 3, 2)))

    # Values and indices must be the same length.
    var short_vals: List[Float64] = [1.0, 2.0]
    assert_true(_raises(CscMatrix(rows.copy(), short_vals^, offs.copy(), 3, 2)))

    # Degenerate dimensions.
    var zero_offs: List[Int] = [0]
    assert_true(
        _raises(CscMatrix(List[Int](), List[Float64](), zero_offs^, 0, 0))
    )


def test_csr_validation() raises:
    var cols: List[Int] = [0, 1, 1]
    var vals: List[Float64] = [1.0, 2.0, 3.0]
    var offs: List[Int] = [0, 2, 3]
    CsrMatrix(cols.copy(), vals.copy(), offs.copy(), 2, 2).validate()

    var unsorted: List[Int] = [1, 0, 1]
    var raised = False
    try:
        CsrMatrix(unsorted^, vals.copy(), offs.copy(), 2, 2).validate()
    except:
        raised = True
    assert_true(raised)


def test_conversions_round_trip() raises:
    var n_rows = 37
    var n_features = 11
    var dense = _sparse_dense(n_rows, n_features, 0.2, UInt64(1))
    var csc = csc_from_dense(dense, n_rows, n_features)
    csc.validate()

    var csr = csc.to_csr()
    csr.validate()
    assert_equal(csr.nnz(), csc.nnz())
    var back = csr.to_csc()
    back.validate()
    assert_equal(back.nnz(), csc.nnz())
    for i in range(csc.nnz()):
        assert_equal(back.row_index[i], csc.row_index[i])
        assert_equal(back.values[i], csc.values[i])

    # Both views densify to the original matrix.
    var from_csc = csc.to_dense()
    var from_csr = csr.to_dense()
    for i in range(n_rows * n_features):
        assert_equal(from_csc[i], dense[i])
        assert_equal(from_csr[i], dense[i])

    # Random access agrees with the dense matrix.
    for r in range(n_rows):
        for f in range(n_features):
            assert_equal(csr.lookup(r, f), dense[f * n_rows + r])


# ------------------------------------------------------------------ binning


def test_sparse_binning_matches_dense_exactly() raises:
    # Sparse binning indexes the implied dense sorted column instead of
    # materializing it, so the edges must come out bit-identical.
    var n_rows = 900
    var n_features = 13
    var dense = _sparse_dense(n_rows, n_features, 0.15, UInt64(2))
    var csc = csc_from_dense(dense, n_rows, n_features)

    for max_bins in [2, 7, 64, 255]:
        var dm = fit_bins(dense, n_rows, n_features, max_bins)
        var sm = fit_bins_csc(csc, max_bins)
        assert_equal(sm.n_features, dm.n_features)
        assert_equal(sm.n_bins, dm.n_bins)
        assert_equal(len(sm.edges), len(dm.edges))
        for i in range(len(dm.edges)):
            assert_equal(sm.edges[i], dm.edges[i])
        for f in range(n_features + 1):
            assert_equal(sm.edge_offsets[f], dm.edge_offsets[f])


def test_sparse_binned_matrix_matches_dense() raises:
    var n_rows = 400
    var n_features = 9
    var dense = _sparse_dense(n_rows, n_features, 0.25, UInt64(3))
    var csc = csc_from_dense(dense, n_rows, n_features)
    var mapper = fit_bins_csc(csc, 32)
    var sparse = transform_csc(mapper, csc)
    var expected = mapper.transform(dense, n_rows)
    var densified = sparse.to_dense()
    var rows = sparse.to_rows()

    for f in range(n_features):
        for r in range(n_rows):
            var want = expected.bin_at(r, f)
            assert_equal(densified.bin_at(r, f), want)
            assert_equal(sparse.bin_at(r, f), want)
            assert_equal(rows.bin_at(r, f), want)


def test_implicit_zero_is_a_numerical_zero() raises:
    # One feature, values -1 for three rows and +1 for three rows, with four
    # rows implicitly zero. Zero must bin between them like a real value.
    var row_index: List[Int] = [0, 1, 2, 7, 8, 9]
    var values: List[Float64] = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
    var offsets: List[Int] = [0, 6]
    var csc = CscMatrix(row_index^, values^, offsets^, 10, 1)
    var mapper = fit_bins_csc(csc, 8)
    var data = transform_csc(mapper, csc)

    var zero_bin = Int(data.default_bin[0])
    assert_equal(data.bin_at(0, 0), 0)
    assert_true(zero_bin > 0)
    assert_true(data.bin_at(7, 0) > zero_bin)
    for r in range(3, 7):
        assert_equal(data.bin_at(r, 0), zero_bin)

    # The zero rows are ordinary rows in an ordinary bin, not missing values.
    assert_equal(mapper.missing_bin[0], -1)
    var grad: List[Float64] = [
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    ]
    var hist = build_histogram_sparse(data, grad, grad)
    assert_equal(hist.count[zero_bin], 4)


def test_explicit_zeros_match_implicit_zeros() raises:
    # SciPy keeps explicitly stored zeros until eliminate_zeros() is called.
    # They must behave exactly like the gaps.
    var n_rows = 200
    var n_features = 5
    var dense = _sparse_dense(n_rows, n_features, 0.3, UInt64(4))
    var compact = csc_from_dense(dense, n_rows, n_features)

    # The same matrix with every third gap stored as an explicit 0.0.
    var row_index = List[Int]()
    var values = List[Float64]()
    var offsets: List[Int] = [0]
    for f in range(n_features):
        for r in range(n_rows):
            var v = dense[f * n_rows + r]
            if v != 0.0 or r % 3 == 0:
                row_index.append(r)
                values.append(v)
        offsets.append(len(values))
    var padded = CscMatrix(row_index^, values^, offsets^, n_rows, n_features)
    assert_true(padded.nnz() > compact.nnz())

    var ma = fit_bins_csc(compact, 16)
    var mb = fit_bins_csc(padded, 16)
    for i in range(len(ma.edges)):
        assert_equal(ma.edges[i], mb.edges[i])

    var da = transform_csc(ma, compact)
    var db = transform_csc(mb, padded)
    for f in range(n_features):
        for r in range(n_rows):
            assert_equal(da.bin_at(r, f), db.bin_at(r, f))

    var target = _target(dense, n_rows, UInt64(500_000))
    var params = BoosterParams(15, 0.1, TreeParams(8, 5, 1.0, 1e-3))
    var ba = train_sparse(da, target, SQUARED_ERROR, params)
    var bb = train_sparse(db, target, SQUARED_ERROR, params)
    assert_equal(len(ba.trees), len(bb.trees))
    # Bins are identical, so the two fits are the same computation over a
    # different number of stored terms: the sums agree to rounding, not to
    # the bit.
    for r in range(n_rows):
        var row = _row(dense, n_rows, n_features, r)
        assert_true(
            abs(
                ba.predict_bins(ma.bin_row(row))
                - bb.predict_bins(mb.bin_row(row))
            )
            < 1e-9
        )


def _with_missing(
    dense: List[Float64], n_rows: Int, n_features: Int, every: Int
) -> List[Float64]:
    """Turn some stored values into NaN, leaving the implicit zeros alone."""
    var out = dense.copy()
    var k = 0
    for f in range(n_features):
        for r in range(n_rows):
            if out[f * n_rows + r] != 0.0:
                k += 1
                if k % every == 0:
                    out[f * n_rows + r] = _nan()
    return out^


def _nan() -> Float64:
    var zero = 0.0
    return zero / zero


def test_sparse_missing_values_match_dense() raises:
    # NaN in a stored value is the missing marker on both paths: it is
    # dropped before the quantile fit, reserves the feature's top bin, and
    # routes by the split's default direction. The implicit zeros are
    # untouched by any of that.
    var n_rows = 800
    var n_features = 7
    var base = _sparse_dense(n_rows, n_features, 0.3, UInt64(24))
    var dense = _with_missing(base, n_rows, n_features, 5)
    var csc = csc_from_dense(dense, n_rows, n_features)

    var dm = fit_bins(dense, n_rows, n_features, 16)
    var sm = fit_bins_csc(csc, 16)
    assert_equal(len(sm.edges), len(dm.edges))
    for i in range(len(dm.edges)):
        assert_equal(sm.edges[i], dm.edges[i])
    var any_reserved = False
    for f in range(n_features):
        assert_equal(sm.missing_bin[f], dm.missing_bin[f])
        if sm.missing_bin[f] >= 0:
            any_reserved = True
    assert_true(any_reserved)

    var sparse = transform_csc(sm, csc)
    var binned = dm.transform(dense, n_rows)
    for f in range(n_features):
        assert_equal(sparse.missing_bin[f], binned.missing_bin[f])
        for r in range(n_rows):
            assert_equal(sparse.bin_at(r, f), binned.bin_at(r, f))

    # An implicit zero is never routed as missing.
    for f in range(n_features):
        if sparse.missing_bin[f] >= 0:
            assert_true(
                Int(sparse.default_bin[f]) != sparse.missing_bin[f]
            )

    var target = _target(base, n_rows, UInt64(2_400_000))
    var params = BoosterParams(20, 0.1, TreeParams(10, 20, 1.0, 1e-3))
    var want = train(binned, target, SQUARED_ERROR, params)
    var got = train_sparse(sparse, target, SQUARED_ERROR, params)
    assert_equal(len(got.trees), len(want.trees))
    for t in range(len(want.trees)):
        _assert_same_tree(want.trees[t], got.trees[t], 1e-12)


def test_sparse_use_missing_off_matches_dense() raises:
    var n_rows = 400
    var n_features = 5
    var base = _sparse_dense(n_rows, n_features, 0.3, UInt64(25))
    var dense = _with_missing(base, n_rows, n_features, 4)
    var csc = csc_from_dense(dense, n_rows, n_features)

    var dm = fit_bins(dense, n_rows, n_features, 16, [], False)
    var sm = fit_bins_csc(csc, 16, [], False)
    assert_equal(len(sm.edges), len(dm.edges))
    for i in range(len(dm.edges)):
        assert_equal(sm.edges[i], dm.edges[i])
    for f in range(n_features):
        assert_equal(sm.missing_bin[f], -1)
        assert_equal(dm.missing_bin[f], -1)
    var sparse = transform_csc(sm, csc)
    var binned = dm.transform(dense, n_rows)
    for f in range(n_features):
        for r in range(n_rows):
            assert_equal(sparse.bin_at(r, f), binned.bin_at(r, f))


def test_sparse_categorical_matches_dense() raises:
    # Categorical columns bin through the mapper's category tables on both
    # paths. An absent entry is the code 0, so a sparse categorical column is
    # the dense one with its zeros left out, exactly as for a numerical one.
    var n_rows = 300
    var n_features = 4
    var dense = List[Float64](capacity=n_rows * n_features)
    dense.resize(n_rows * n_features, 0.0)
    for f in range(n_features):
        for r in range(n_rows):
            dense[f * n_rows + r] = Float64((r * (f + 1)) % 5)
    var csc = csc_from_dense(dense, n_rows, n_features)
    var cat_features: List[Int] = [1, 3]

    var dm = fit_bins(dense, n_rows, n_features, 16, cat_features)
    var sm = fit_bins_csc(csc, 16, cat_features)
    assert_true(dm.cats.any_categorical())
    assert_true(sm.cats.any_categorical())
    var sparse = transform_csc(sm, csc)
    var binned = dm.transform(dense, n_rows)
    for f in range(n_features):
        for r in range(n_rows):
            assert_equal(sparse.bin_at(r, f), binned.bin_at(r, f))

    # One tree, not an ensemble: this grid of category codes is full of
    # split candidates whose gains are equal to the last bits, so over many
    # rounds the two accumulators pick different (equally good) winners and
    # separate. See test_sparse_renewed_objectives_match_dense for the same
    # effect on a numerical dataset.
    var grad = _grads(n_rows, UInt64(2_500_000))
    var hess = _hessians(n_rows, UInt64(2_600_000))
    var tree_params = TreeParams(8, 10, 1.0, 1e-3)
    _assert_same_tree(
        grow_tree(binned, grad, hess, tree_params),
        grow_tree_sparse(sparse, grad, hess, tree_params).tree,
        1e-12,
    )


# --------------------------------------------------------------- histograms


def _assert_counts_equal(a: List[Int], b: List[Int]) raises:
    assert_equal(len(a), len(b))
    for i in range(len(a)):
        assert_equal(a[i], b[i])


def _max_abs_diff(a: List[Float64], b: List[Float64]) -> Float64:
    var worst = 0.0
    for i in range(len(a)):
        var d = abs(a[i] - b[i])
        if d > worst:
            worst = d
    return worst


def test_sparse_histogram_matches_dense() raises:
    var n_rows = 1500
    var n_features = 12
    var dense = _sparse_dense(n_rows, n_features, 0.1, UInt64(5))
    var csc = csc_from_dense(dense, n_rows, n_features)
    var mapper = fit_bins_csc(csc, 32)
    var sparse = transform_csc(mapper, csc)
    var binned = mapper.transform(dense, n_rows)
    var grad = _grads(n_rows, UInt64(1_000_000))
    var hess = _hessians(n_rows, UInt64(2_000_000))

    var want = build_histogram(binned, grad, hess)
    var got = build_histogram_sparse(sparse, grad, hess)
    assert_equal(got.n_features, want.n_features)
    assert_equal(got.n_bins, want.n_bins)
    # Counts are integer on both paths and must match exactly.
    _assert_counts_equal(want.count, got.count)
    # Gradient and hessian sums differ only by the rounding of one
    # subtraction per (feature, default bin).
    assert_true(_max_abs_diff(want.grad, got.grad) < 1e-9)
    assert_true(_max_abs_diff(want.hess, got.hess) < 1e-9)

    var rows = List[Int]()
    for r in range(0, n_rows, 3):
        rows.append(r)
    var want_sub = build_histogram_subset(binned, grad, hess, rows)
    var got_sub = build_histogram_sparse_subset(sparse, grad, hess, rows)
    _assert_counts_equal(want_sub.count, got_sub.count)
    assert_true(_max_abs_diff(want_sub.grad, got_sub.grad) < 1e-9)
    assert_true(_max_abs_diff(want_sub.hess, got_sub.hess) < 1e-9)


def test_sparse_histogram_grouped_matches_masked() raises:
    # The grouped-entry builder used by tree growth and the mask-filtering
    # builder must agree on the whole dataset, bit for bit: they visit each
    # feature's entries in the same order.
    var n_rows = 700
    var n_features = 6
    var dense = _sparse_dense(n_rows, n_features, 0.2, UInt64(6))
    var csc = csc_from_dense(dense, n_rows, n_features)
    var sparse = transform_csc(fit_bins_csc(csc, 16), csc)
    var grad = _grads(n_rows, UInt64(3_000_000))
    var hess = _hessians(n_rows, UInt64(4_000_000))

    var all_rows = List[Int]()
    for r in range(n_rows):
        all_rows.append(r)
    var masked = build_histogram_sparse_subset(sparse, grad, hess, all_rows)
    var grouped = build_histogram_sparse(sparse, grad, hess)
    _assert_counts_equal(masked.count, grouped.count)
    for i in range(len(masked.grad)):
        assert_equal(masked.grad[i], grouped.grad[i])
        assert_equal(masked.hess[i], grouped.hess[i])


def test_sparse_histogram_rejects_bad_input() raises:
    var dense = _sparse_dense(20, 3, 0.4, UInt64(7))
    var csc = csc_from_dense(dense, 20, 3)
    var sparse = transform_csc(fit_bins_csc(csc, 8), csc)
    var short = _grads(5, UInt64(1))
    var raised = False
    try:
        _ = build_histogram_sparse(sparse, short, short)
    except:
        raised = True
    assert_true(raised)

    var grad = _grads(20, UInt64(2))
    var bad_rows: List[Int] = [0, 1, 1]
    raised = False
    try:
        _ = build_histogram_sparse_subset(sparse, grad, grad, bad_rows)
    except:
        raised = True
    assert_true(raised)

    var oob_rows: List[Int] = [0, 99]
    raised = False
    try:
        _ = build_histogram_sparse_subset(sparse, grad, grad, oob_rows)
    except:
        raised = True
    assert_true(raised)


# ------------------------------------------------------------- tree growth


def _assert_same_tree(want: Tree, got: Tree, tol: Float64) raises:
    assert_equal(got.n_leaves, want.n_leaves)
    assert_equal(len(got.feature), len(want.feature))
    for i in range(len(want.feature)):
        assert_equal(got.feature[i], want.feature[i])
        assert_equal(got.threshold_bin[i], want.threshold_bin[i])
        assert_equal(got.left[i], want.left[i])
        assert_equal(got.right[i], want.right[i])
        assert_equal(got.default_left[i], want.default_left[i])
        assert_equal(got.missing_bin[i], want.missing_bin[i])
        assert_true(abs(got.value[i] - want.value[i]) <= tol)


def test_sparse_tree_matches_dense() raises:
    var n_rows = 2000
    var n_features = 15
    var dense = _sparse_dense(n_rows, n_features, 0.12, UInt64(8))
    var csc = csc_from_dense(dense, n_rows, n_features)
    var mapper = fit_bins_csc(csc, 32)
    var sparse = transform_csc(mapper, csc)
    var binned = mapper.transform(dense, n_rows)
    var grad = _grads(n_rows, UInt64(5_000_000))
    var hess = _hessians(n_rows, UInt64(6_000_000))

    var params = TreeParams(15, 20, 1.0, 1e-3)
    var want = grow_tree(binned, grad, hess, params)
    var got = grow_tree_sparse(sparse, grad, hess, params)
    _assert_same_tree(want, got.tree, 1e-9)

    # The row-to-leaf assignment must agree with walking the tree.
    var rows = sparse.to_rows()
    for r in range(n_rows):
        assert_true(got.row_leaf[r] >= 0)
        assert_equal(
            got.tree.value[got.row_leaf[r]],
            predict_row_sparse(got.tree, rows, r),
        )
        assert_true(
            abs(want.predict_row(binned, r) - got.tree.value[got.row_leaf[r]])
            < 1e-9
        )


def test_sparse_tree_honours_tree_params() raises:
    var n_rows = 1200
    var n_features = 10
    var dense = _sparse_dense(n_rows, n_features, 0.2, UInt64(9))
    var csc = csc_from_dense(dense, n_rows, n_features)
    var mapper = fit_bins_csc(csc, 32)
    var sparse = transform_csc(mapper, csc)
    var binned = mapper.transform(dense, n_rows)
    var grad = _grads(n_rows, UInt64(7_000_000))
    var hess = _hessians(n_rows, UInt64(8_000_000))

    # max_depth, lambda_l1, and per-tree feature subsampling all live above
    # the accumulator, so the sparse grower must reproduce them exactly.
    var depth_params = TreeParams(31, 5, 1.0, 1e-3, 0.5, max_depth=2)
    var got = grow_tree_sparse(sparse, grad, hess, depth_params)
    assert_true(got.tree.depth() <= 2)
    _assert_same_tree(
        grow_tree(binned, grad, hess, depth_params), got.tree, 1e-9
    )

    var sampled = TreeParams(
        15, 20, 1.0, 1e-3, 0.0, feature_fraction=0.4,
        feature_fraction_bynode=0.7, feature_fraction_seed=17,
    )
    for tree_index in range(3):
        _assert_same_tree(
            grow_tree(binned, grad, hess, sampled, [], tree_index),
            grow_tree_sparse(sparse, grad, hess, sampled, [], tree_index).tree,
            1e-9,
        )


def test_sparse_tree_bagging_matches_dense() raises:
    var n_rows = 1000
    var n_features = 8
    var dense = _sparse_dense(n_rows, n_features, 0.2, UInt64(10))
    var csc = csc_from_dense(dense, n_rows, n_features)
    var mapper = fit_bins_csc(csc, 16)
    var sparse = transform_csc(mapper, csc)
    var binned = mapper.transform(dense, n_rows)
    var grad = _grads(n_rows, UInt64(9_100_000))
    var hess = _hessians(n_rows, UInt64(9_200_000))

    var bag = List[Int]()
    for r in range(0, n_rows, 2):
        bag.append(r)
    var params = TreeParams(10, 10, 1.0, 1e-3)
    var got = grow_tree_sparse(sparse, grad, hess, params, bag)
    _assert_same_tree(grow_tree(binned, grad, hess, params, bag), got.tree, 1e-9)

    # Rows outside the bag are not assigned to a leaf.
    for r in range(n_rows):
        if r % 2 == 0:
            assert_true(got.row_leaf[r] >= 0)
        else:
            assert_equal(got.row_leaf[r], -1)

    # Duplicate bag rows are rejected rather than silently double-counted.
    var duplicated: List[Int] = [0, 1, 1, 2]
    var raised = False
    try:
        _ = grow_tree_sparse(sparse, grad, hess, params, duplicated)
    except:
        raised = True
    assert_true(raised)


# ------------------------------------------------------------------ training


def _worst_prediction_gap(
    dense: List[Float64],
    n_rows: Int,
    n_features: Int,
    mapper_a: BinMapper,
    booster_a: Booster,
    mapper_b: BinMapper,
    booster_b: Booster,
) raises -> Float64:
    var worst = 0.0
    for r in range(n_rows):
        var row = _row(dense, n_rows, n_features, r)
        var d = abs(
            booster_a.predict_bins(mapper_a.bin_row(row))
            - booster_b.predict_bins(mapper_b.bin_row(row))
        )
        if d > worst:
            worst = d
    return worst


def test_sparse_training_matches_dense() raises:
    var n_rows = 1500
    var n_features = 12
    var dense = _sparse_dense(n_rows, n_features, 0.15, UInt64(11))
    var csc = csc_from_dense(dense, n_rows, n_features)
    var mapper = fit_bins_csc(csc, 32)
    var sparse = transform_csc(mapper, csc)
    var binned = mapper.transform(dense, n_rows)
    var target = _target(dense, n_rows, UInt64(600_000))
    var params = BoosterParams(30, 0.1, TreeParams(15, 20, 1.0, 1e-3))

    var want = train(binned, target, SQUARED_ERROR, params)
    var got = train_sparse(sparse, target, SQUARED_ERROR, params)
    assert_equal(len(got.trees), len(want.trees))
    assert_true(
        _worst_prediction_gap(
            dense, n_rows, n_features, mapper, want, mapper, got
        )
        < 1e-9
    )

    # Binary logistic on a thresholded target.
    var labels = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        labels.append(1.0 if target[r] > 0.0 else 0.0)
    var want_bin = train(binned, labels, BINARY_LOGISTIC, params)
    var got_bin = train_sparse(sparse, labels, BINARY_LOGISTIC, params)
    assert_equal(len(got_bin.trees), len(want_bin.trees))
    assert_true(
        _worst_prediction_gap(
            dense, n_rows, n_features, mapper, want_bin, mapper, got_bin
        )
        < 1e-9
    )


def test_sparse_renewed_objectives_match_dense() raises:
    """QUANTILE and L1, the two objectives that renew leaf values.

    Held to a shorter run than the smooth objectives on purpose. Their
    gradient is a step function of sign(raw - target), so once two split
    candidates are tied to within the rounding that separates the two
    accumulators -- which happens on this dataset around round 13, where the
    winner and the runner-up agree to 13 significant digits -- the two fits
    pick different (equally good) splits and then keep diverging. That is a
    property of tied data, not of either accumulator; up to that point the
    two paths agree to the last bits, which is what this asserts.
    """
    var n_rows = 1500
    var n_features = 12
    var dense = _sparse_dense(n_rows, n_features, 0.15, UInt64(11))
    var csc = csc_from_dense(dense, n_rows, n_features)
    var mapper = fit_bins_csc(csc, 32)
    var sparse = transform_csc(mapper, csc)
    var binned = mapper.transform(dense, n_rows)
    var target = _target(dense, n_rows, UInt64(600_000))
    var params = BoosterParams(12, 0.1, TreeParams(15, 20, 1.0, 1e-3))

    for objective in [QUANTILE, L1]:
        var want = train(binned, target, objective, params)
        var got = train_sparse(sparse, target, objective, params)
        assert_equal(len(got.trees), len(want.trees))
        assert_true(len(got.trees) > 0)
        for t in range(len(want.trees)):
            _assert_same_tree(want.trees[t], got.trees[t], 1e-12)
        assert_true(
            _worst_prediction_gap(
                dense, n_rows, n_features, mapper, want, mapper, got
            )
            < 1e-9
        )


def test_sparse_training_with_weights_matches_dense() raises:
    var n_rows = 900
    var n_features = 8
    var dense = _sparse_dense(n_rows, n_features, 0.2, UInt64(12))
    var csc = csc_from_dense(dense, n_rows, n_features)
    var mapper = fit_bins_csc(csc, 16)
    var sparse = transform_csc(mapper, csc)
    var binned = mapper.transform(dense, n_rows)
    var target = _target(dense, n_rows, UInt64(700_000))
    var weights = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        weights.append(0.5 + 2.0 * _uniform(UInt64(800_000 + r)))

    var params = BoosterParams(20, 0.15, TreeParams(10, 10, 1.0, 1e-3))
    var want = train(binned, target, SQUARED_ERROR, params, weights)
    var got = train_sparse(sparse, target, SQUARED_ERROR, params, weights)
    assert_equal(len(got.trees), len(want.trees))
    assert_true(
        _worst_prediction_gap(
            dense, n_rows, n_features, mapper, want, mapper, got
        )
        < 1e-8
    )


def test_sparse_training_with_bagging_matches_dense() raises:
    var n_rows = 900
    var n_features = 8
    var dense = _sparse_dense(n_rows, n_features, 0.2, UInt64(13))
    var csc = csc_from_dense(dense, n_rows, n_features)
    var mapper = fit_bins_csc(csc, 16)
    var sparse = transform_csc(mapper, csc)
    var binned = mapper.transform(dense, n_rows)
    var target = _target(dense, n_rows, UInt64(900_000))

    var params = BoosterParams(20, 0.15, TreeParams(10, 10, 1.0, 1e-3))
    var bagging = BaggingParams(0.6, 1, 5)
    var want = train(binned, target, SQUARED_ERROR, params, [], 0.9, bagging)
    var got = train_sparse(
        sparse, target, SQUARED_ERROR, params, [], 0.9, bagging
    )
    assert_equal(len(got.trees), len(want.trees))
    assert_true(len(got.trees) > 0)
    assert_true(
        _worst_prediction_gap(
            dense, n_rows, n_features, mapper, want, mapper, got
        )
        < 1e-8
    )


def test_sparse_early_stopping_matches_dense() raises:
    var n_rows = 800
    var n_valid = 300
    var n_features = 8
    var dense = _sparse_dense(n_rows, n_features, 0.2, UInt64(14))
    var valid_dense = _sparse_dense(n_valid, n_features, 0.2, UInt64(15))
    var csc = csc_from_dense(dense, n_rows, n_features)
    var valid_csc = csc_from_dense(valid_dense, n_valid, n_features)
    var mapper = fit_bins_csc(csc, 16)
    var sparse = transform_csc(mapper, csc)
    var valid_sparse = transform_csc(mapper, valid_csc)
    var binned = mapper.transform(dense, n_rows)
    var valid_binned = mapper.transform(valid_dense, n_valid)
    var target = _target(dense, n_rows, UInt64(1_100_000))
    var valid_target = _target(valid_dense, n_valid, UInt64(1_200_000))

    var params = BoosterParams(200, 0.2, TreeParams(10, 20, 1.0, 1e-3))
    var want = train_with_valid(
        binned, target, valid_binned, valid_target, SQUARED_ERROR, params, 5,
    )
    var got = train_sparse_with_valid(
        sparse, target, valid_sparse, valid_target, SQUARED_ERROR, params, 5,
    )
    assert_true(len(got.trees) > 0)
    assert_true(len(got.trees) < 200)
    assert_equal(len(got.trees), len(want.trees))
    assert_true(
        _worst_prediction_gap(
            dense, n_rows, n_features, mapper, want, mapper, got
        )
        < 1e-8
    )


def test_sparse_multiclass_matches_dense() raises:
    var n_rows = 700
    var n_features = 8
    var n_classes = 3
    var dense = _sparse_dense(n_rows, n_features, 0.25, UInt64(16))
    var csc = csc_from_dense(dense, n_rows, n_features)
    var mapper = fit_bins_csc(csc, 16)
    var sparse = transform_csc(mapper, csc)
    var binned = mapper.transform(dense, n_rows)
    var score = _target(dense, n_rows, UInt64(1_300_000))
    var labels = List[Int](capacity=n_rows)
    for r in range(n_rows):
        if score[r] < -1.0:
            labels.append(0)
        elif score[r] < 1.0:
            labels.append(1)
        else:
            labels.append(2)

    var params = BoosterParams(15, 0.2, TreeParams(8, 10, 1.0, 1e-3))
    var want = train_multiclass(binned, labels, n_classes, params)
    var got = train_multiclass_sparse(sparse, labels, n_classes, params)
    assert_equal(len(got.trees), len(want.trees))
    for r in range(n_rows):
        var bins = mapper.bin_row(_row(dense, n_rows, n_features, r))
        var a = want.predict_proba_bins(bins)
        var b = got.predict_proba_bins(bins)
        for k in range(n_classes):
            assert_true(abs(a[k] - b[k]) < 1e-9)


# ---------------------------------------------------------------- end to end


def test_fit_csc_matches_fit() raises:
    var n_rows = 1200
    var n_features = 10
    var dense = _sparse_dense(n_rows, n_features, 0.15, UInt64(17))
    var csc = csc_from_dense(dense, n_rows, n_features)
    var target = _target(dense, n_rows, UInt64(1_400_000))
    var params = BoosterParams(25, 0.1, TreeParams(12, 20, 1.0, 1e-3))

    var want = fit(dense, n_rows, n_features, target, SQUARED_ERROR, params, 32)
    var got = fit_csc(csc, target, SQUARED_ERROR, params, 32)
    assert_equal(len(got.booster.trees), len(want.booster.trees))
    for r in range(n_rows):
        var row = _row(dense, n_rows, n_features, r)
        assert_true(abs(want.predict(row) - got.predict(row)) < 1e-8)


def test_predict_csr_matches_dense_rows() raises:
    var n_rows = 800
    var n_features = 9
    var dense = _sparse_dense(n_rows, n_features, 0.2, UInt64(18))
    var csc = csc_from_dense(dense, n_rows, n_features)
    var csr = csc.to_csr()
    var target = _target(dense, n_rows, UInt64(1_500_000))
    var params = BoosterParams(20, 0.15, TreeParams(10, 20, 1.0, 1e-3))

    var model = fit_csc(csc, target, SQUARED_ERROR, params, 32)
    var predicted = predict_csr(model, csr)
    var raw = predict_raw_csr(model, csr)
    assert_equal(len(predicted), n_rows)
    for r in range(n_rows):
        var row = _row(dense, n_rows, n_features, r)
        # Same model, same bins: the sparse walk must be exact, not close.
        assert_equal(predicted[r], model.predict(row))
        assert_equal(raw[r], model.predict_raw(row))

    # A binary model routes through the logistic link on both paths.
    var labels = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        labels.append(1.0 if target[r] > 0.0 else 0.0)
    var clf = fit_csc(csc, labels, BINARY_LOGISTIC, params, 32)
    var proba = predict_csr(clf, csr)
    for r in range(n_rows):
        assert_true(proba[r] >= 0.0 and proba[r] <= 1.0)
        assert_equal(proba[r], clf.predict(_row(dense, n_rows, n_features, r)))


def test_predict_proba_csr_matches_dense_rows() raises:
    var n_rows = 500
    var n_features = 7
    var n_classes = 3
    var dense = _sparse_dense(n_rows, n_features, 0.25, UInt64(19))
    var csc = csc_from_dense(dense, n_rows, n_features)
    var csr = csc.to_csr()
    var score = _target(dense, n_rows, UInt64(1_600_000))
    var labels = List[Int](capacity=n_rows)
    for r in range(n_rows):
        if score[r] < -1.0:
            labels.append(0)
        elif score[r] < 1.0:
            labels.append(1)
        else:
            labels.append(2)

    var params = BoosterParams(12, 0.2, TreeParams(8, 10, 1.0, 1e-3))
    var model = fit_multiclass_csc(csc, labels, n_classes, params, 16)
    var proba = predict_proba_csr(model, csr)
    for r in range(n_rows):
        var want = model.predict_proba(_row(dense, n_rows, n_features, r))
        var total = 0.0
        for k in range(n_classes):
            assert_equal(proba[r * n_classes + k], want[k])
            total += proba[r * n_classes + k]
        assert_true(abs(total - 1.0) < 1e-9)


def test_sparse_model_serializes_unchanged() raises:
    # A sparse fit produces an ordinary Model, so the v1 format needs no
    # change and a round trip must be bit-exact.
    var n_rows = 400
    var n_features = 6
    var dense = _sparse_dense(n_rows, n_features, 0.25, UInt64(20))
    var csc = csc_from_dense(dense, n_rows, n_features)
    var csr = csc.to_csr()
    var target = _target(dense, n_rows, UInt64(1_700_000))
    var params = BoosterParams(15, 0.1, TreeParams(8, 10, 1.0, 1e-3))
    var model = fit_csc(csc, target, SQUARED_ERROR, params, 16)
    var before = predict_csr(model, csr)

    save_model(model, _TMP_PATH)
    var loaded = load_model(_TMP_PATH)
    var after = predict_csr(loaded, csr)
    remove(_TMP_PATH)
    for r in range(n_rows):
        assert_equal(before[r], after[r])


def test_predict_csr_validates_shape() raises:
    var n_rows = 50
    var n_features = 4
    var dense = _sparse_dense(n_rows, n_features, 0.3, UInt64(21))
    var csc = csc_from_dense(dense, n_rows, n_features)
    var target = _target(dense, n_rows, UInt64(1_800_000))
    var model = fit_csc(
        csc, target, SQUARED_ERROR,
        BoosterParams(5, 0.1, TreeParams(4, 5, 1.0, 1e-3)), 8,
    )

    var other = csc_from_dense(
        _sparse_dense(10, 3, 0.4, UInt64(22)), 10, 3
    ).to_csr()
    var raised = False
    try:
        _ = predict_csr(model, other)
    except:
        raised = True
    assert_true(raised)


# --------------------------------------------------------------- determinism


def test_worker_settings_do_not_change_results() raises:
    # One test owns all env mutation so no other test sees a dirty
    # environment regardless of suite ordering; empty string means unset.
    var n_rows = 600
    var n_features = 7
    var dense = _sparse_dense(n_rows, n_features, 0.2, UInt64(23))
    var csc = csc_from_dense(dense, n_rows, n_features)
    var target = _target(dense, n_rows, UInt64(1_900_000))
    var params = BoosterParams(12, 0.15, TreeParams(10, 10, 1.0, 1e-3))
    var grad = _grads(n_rows, UInt64(2_100_000))
    var hess = _hessians(n_rows, UInt64(2_200_000))

    _ = setenv("MOJOBOOST_NUM_WORKERS", "1")
    var mapper = fit_bins_csc(csc, 16)
    var sparse = transform_csc(mapper, csc)
    var serial_hist = build_histogram_sparse(sparse, grad, hess)
    var serial_tree = grow_tree_sparse(sparse, grad, hess, params.tree)
    var serial_model = fit_csc(csc, target, SQUARED_ERROR, params, 16)

    for workers in ["2", "8", ""]:
        _ = setenv("MOJOBOOST_NUM_WORKERS", workers)
        _ = setenv("MOJOBOOST_PARALLEL_MIN_OPS", "1")
        var m = fit_bins_csc(csc, 16)
        for i in range(len(m.edges)):
            assert_equal(m.edges[i], mapper.edges[i])
        var s = transform_csc(m, csc)
        for i in range(s.nnz()):
            assert_equal(s.bin[i], sparse.bin[i])
        var h = build_histogram_sparse(s, grad, hess)
        for i in range(len(h.grad)):
            assert_equal(h.grad[i], serial_hist.grad[i])
            assert_equal(h.hess[i], serial_hist.hess[i])
            assert_equal(h.count[i], serial_hist.count[i])
        var t = grow_tree_sparse(s, grad, hess, params.tree)
        _assert_same_tree(serial_tree.tree, t.tree, 0.0)
        for r in range(n_rows):
            assert_equal(t.row_leaf[r], serial_tree.row_leaf[r])
        var model = fit_csc(csc, target, SQUARED_ERROR, params, 16)
        for r in range(n_rows):
            var row = _row(dense, n_rows, n_features, r)
            assert_equal(model.predict(row), serial_model.predict(row))

    _ = setenv("MOJOBOOST_NUM_WORKERS", "")
    _ = setenv("MOJOBOOST_PARALLEL_MIN_OPS", "")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
