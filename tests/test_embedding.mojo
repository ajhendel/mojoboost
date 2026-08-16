"""CatBoost's `LDA` and `KNN` embedding estimators
(src/mojotrees/embedding.mojo, docs/design/CATBOOST_CATALOG.md A20).

Every expected number below was worked out by hand from the definition
before the code was run, not read out of a mojotrees run. The four-row LDA
fixture is small enough that the whole generalized eigenproblem is solvable
on paper, and the derivation is written out beside the assertion.

The first two blocks are the point of the lane: the online pass computes row
`i` from a strict prefix, so the first row of a permutation is scored against
nothing and a KNN row never sees its own target. Those are the leakage
assertions and they come first.
"""

from std.math import sqrt
from std.memory import bitcast
from std.os import setenv
from std.testing import assert_equal, assert_false, assert_raises, assert_true
from std.testing import TestSuite

from mojotrees.embedding import (
    CATBOOST_KNN_K_DEFAULT,
    CATBOOST_LDA_REG_DEFAULT,
    EmbeddingEstimatorParams,
    EmbeddingMatrix,
    Eigen,
    KnnParams,
    LdaParams,
    LDA_COMPONENTS_CATBOOST_DEFAULT,
    apply_knn,
    apply_lda,
    canonical_sign,
    cholesky_lower,
    check_permutation,
    class_ids_from_targets,
    compute_online_features,
    fit_knn,
    fit_lda,
    identity_permutation,
    jacobi_eigen,
    knn_feature_count,
    knn_online_features,
    lda_online_features,
    online_feature_count,
    sort_eigenpairs,
)


def close(a: Float64, b: Float64) -> Bool:
    return abs(a - b) < 1e-9


def nan_value() -> Float64:
    return bitcast[DType.float64, 1](SIMD[DType.uint64, 1](0x7FF8000000000000))


def line_1d(var xs: List[Float64]) raises -> EmbeddingMatrix:
    """`n` points on a line, one dimension each."""
    var n = len(xs)
    return EmbeddingMatrix(n, 1, xs^)


def knn_on(k: Int) -> KnnParams:
    return KnnParams(True, k, 1000)


# ---------------------------------------------------------------------------
# The leakage rule. `ComputeOnlineFeatures` is Compute-then-Update, so row
# `i` sees a strict prefix and the first row of the permutation sees nothing.
# ---------------------------------------------------------------------------


def test_knn_first_permutation_row_has_an_empty_neighbourhood() raises:
    # Four points on a line, alternating classes. CatBoost's HNSW cloud is
    # empty when the first row is queried and `GetNearestNeighbors` returns
    # nothing, so `TKNNCalcer::Compute` leaves its zero-initialized result
    # alone. Ours must do the same, exactly, not approximately.
    var e = line_1d([0.0, 1.0, 2.0, 3.0])
    var targets: List[Float64] = [0.0, 1.0, 0.0, 1.0]
    var cols = knn_online_features(
        e, targets, 2, identity_permutation(4), knn_on(1)
    )
    assert_equal(len(cols), 2)
    assert_equal(cols[0][0], 0.0)
    assert_equal(cols[1][0], 0.0)


def test_knn_never_counts_a_rows_own_class() raises:
    # Hand derivation, k = 1, identity permutation, classes 0,1,0,1:
    #   pos 0 (row 0): prefix empty                      -> [0, 0]
    #   pos 1 (row 1): prefix {0}, d=1, class 0          -> [1, 0]
    #   pos 2 (row 2): prefix {0:d=4, 1:d=1}, nearest 1  -> [0, 1]
    #   pos 3 (row 3): prefix {0:9, 1:4, 2:1}, nearest 2 -> [1, 0]
    # Row 3's own class is 1 and its feature says class 0. That is the
    # exclusion, and it is structural: row 3 is not in its own prefix.
    var e = line_1d([0.0, 1.0, 2.0, 3.0])
    var targets: List[Float64] = [0.0, 1.0, 0.0, 1.0]
    var cols = knn_online_features(
        e, targets, 2, identity_permutation(4), knn_on(1)
    )
    var class0 = cols[0].copy()
    var class1 = cols[1].copy()
    assert_equal(class0[0], 0.0)
    assert_equal(class1[0], 0.0)
    assert_equal(class0[1], 1.0)
    assert_equal(class1[1], 0.0)
    assert_equal(class0[2], 0.0)
    assert_equal(class1[2], 1.0)
    assert_equal(class0[3], 1.0)
    assert_equal(class1[3], 0.0)


def test_knn_counts_are_raw_and_unnormalized() raises:
    # `++result[TargetClasses.at(...)]` -- counts, not proportions. With
    # k = 3 and a prefix of three, the row's three counts must sum to 3.
    var e = line_1d([0.0, 1.0, 2.0, 3.0])
    var targets: List[Float64] = [0.0, 1.0, 0.0, 1.0]
    var cols = knn_online_features(
        e, targets, 2, identity_permutation(4), knn_on(3)
    )
    assert_equal(cols[0][3] + cols[1][3], 3.0)
    # rows 0, 1, 2 have classes 0, 1, 0 -> two zeros and one one.
    assert_equal(cols[0][3], 2.0)
    assert_equal(cols[1][3], 1.0)
    # And a short prefix gives a short count: row 1's prefix holds one row.
    assert_equal(cols[0][1] + cols[1][1], 1.0)


def test_knn_regression_is_the_prefix_mean_and_zero_when_empty() raises:
    # k = 2, identity permutation, targets 10/20/30/40 at x = 0/1/2/3:
    #   pos 0: empty                       -> 0.0 (CatBoost's explicit guard)
    #   pos 1: {0}                         -> 10
    #   pos 2: {0:d=4, 1:d=1} both taken   -> (10 + 20)/2 = 15
    #   pos 3: nearest two are rows 2, 1   -> (30 + 20)/2 = 25
    var e = line_1d([0.0, 1.0, 2.0, 3.0])
    var targets: List[Float64] = [10.0, 20.0, 30.0, 40.0]
    var cols = knn_online_features(
        e, targets, 0, identity_permutation(4), knn_on(2)
    )
    assert_equal(len(cols), 1)
    assert_equal(cols[0][0], 0.0)
    assert_equal(cols[0][1], 10.0)
    assert_equal(cols[0][2], 15.0)
    assert_equal(cols[0][3], 25.0)


def test_lda_first_permutation_row_projects_to_zero() raises:
    # `ProjectionMatrix` is constructed zeroed and `Compute` runs against it,
    # so the first row of the permutation projects to exactly 0.
    var e = EmbeddingMatrix(
        4, 2, [-1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    )
    var class_ids: List[Int] = [0, 0, 1, 1]
    var p = LdaParams(
        True, LDA_COMPONENTS_CATBOOST_DEFAULT, CATBOOST_LDA_REG_DEFAULT, False, 60
    )
    var cols = lda_online_features(e, class_ids, 2, identity_permutation(4), p)
    assert_equal(len(cols), 1)
    assert_equal(cols[0][0], 0.0)


# ---------------------------------------------------------------------------
# The train-versus-predict asymmetry: two entry points, two different
# numbers for the same row, on purpose.
# ---------------------------------------------------------------------------


def test_train_and_predict_values_differ_for_the_same_row() raises:
    var e = line_1d([0.0, 1.0, 2.0, 3.0])
    var targets: List[Float64] = [10.0, 20.0, 30.0, 40.0]

    var online = knn_online_features(
        e, targets, 0, identity_permutation(4), knn_on(1)
    )
    var model = fit_knn(e, targets, 0, knn_on(1))
    var predicted = apply_knn(model, e)

    # Row 0 online: empty prefix, 0.0. Row 0 through the fitted cloud: the
    # cloud contains row 0 itself at distance 0, so the "neighbour" is the
    # row and the feature is its own target. That is precisely why the
    # fitted cloud must never be used on training rows, and the test states
    # it as a fact about the two functions rather than as a warning.
    assert_equal(online[0][0], 0.0)
    assert_equal(predicted[0][0], 10.0)
    assert_true(online[0][0] != predicted[0][0])


def test_knn_predict_on_unseen_rows_uses_the_whole_cloud() raises:
    var learn = line_1d([0.0, 1.0, 2.0, 3.0])
    var targets: List[Float64] = [0.0, 0.0, 1.0, 1.0]
    var model = fit_knn(learn, targets, 2, knn_on(3))
    # A query at 2.9 is not in the cloud. Its three nearest are rows 3, 2, 1
    # (d = 0.01, 0.81, 3.61), classes 1, 1, 0.
    var query = line_1d([2.9])
    var cols = apply_knn(model, query)
    assert_equal(len(cols), 2)
    assert_equal(cols[0][0], 1.0)
    assert_equal(cols[1][0], 2.0)


# ---------------------------------------------------------------------------
# Determinism: the KNN tie-break, and worker-count invariance.
# ---------------------------------------------------------------------------


def test_knn_ties_break_by_row_index_not_by_examination_order() raises:
    # Rows 0 and 1 sit at -1 and +1; row 2 sits at 0 and is equidistant from
    # both. `_cand_less` orders on (distance, row index), so row 0 wins and
    # row 2's single neighbour has class 0.
    var e = line_1d([-1.0, 1.0, 0.0])
    var targets: List[Float64] = [0.0, 1.0, 0.0]

    var forward = knn_online_features(
        e, targets, 2, identity_permutation(3), knn_on(1)
    )
    assert_equal(forward[0][2], 1.0)
    assert_equal(forward[1][2], 0.0)

    # Same query, prefix examined in the opposite order. Row 2 is still last
    # in the permutation, so its candidate set is unchanged and only the
    # order it is walked in differs. A comparator that broke ties by
    # "whichever came first" would flip here.
    var reversed_prefix: List[Int] = [1, 0, 2]
    var backward = knn_online_features(e, targets, 2, reversed_prefix, knn_on(1))
    assert_equal(backward[0][2], 1.0)
    assert_equal(backward[1][2], 0.0)


def test_a_non_finite_distance_loses_to_every_finite_one() raises:
    # Row 0's coordinate is NaN, so every distance to it is NaN and is mapped
    # to +inf. Row 2's nearest is therefore row 1, class 1 -- not row 0,
    # which a raw NaN comparison could have let win by making `<` false in
    # both directions.
    var e = line_1d([nan_value(), 0.0, 0.5])
    var targets: List[Float64] = [0.0, 1.0, 0.0]
    var cols = knn_online_features(
        e, targets, 2, identity_permutation(3), knn_on(1)
    )
    assert_equal(cols[0][2], 0.0)
    assert_equal(cols[1][2], 1.0)


def test_knn_is_identical_at_every_worker_count() raises:
    # 40 rows is far below the auto-parallel threshold, so the floor is
    # dropped to force the fan-out to actually happen; otherwise this would
    # compare the serial path against itself three times.
    var n = 40
    var xs = List[Float64]()
    var targets = List[Float64]()
    for i in range(n):
        xs.append(Float64(i) * 0.25)
        targets.append(Float64(i % 3))
    var e = line_1d(xs^)
    var perm = identity_permutation(n)

    _ = setenv("MOJOTREES_PARALLEL_MIN_OPS", "1")
    var settings: List[String] = ["1", "2", "4", "0"]
    var baseline = List[Float64]()
    for s in range(len(settings)):
        _ = setenv("MOJOTREES_NUM_WORKERS", settings[s])
        var cols = knn_online_features(e, targets, 3, perm, knn_on(4))
        var flat = List[Float64]()
        for j in range(len(cols)):
            for r in range(n):
                flat.append(cols[j][r])
        if s == 0:
            baseline = flat^
        else:
            assert_equal(len(flat), len(baseline))
            for i in range(len(flat)):
                assert_equal(flat[i], baseline[i])
    _ = setenv("MOJOTREES_NUM_WORKERS", "")
    _ = setenv("MOJOTREES_PARALLEL_MIN_OPS", "")


# ---------------------------------------------------------------------------
# Determinism: the eigenvector sign and the eigenpair order.
# ---------------------------------------------------------------------------


def test_canonical_sign_pins_the_arbitrary_half_of_an_eigenvector() raises:
    # Largest absolute component is negative -> negate the whole vector.
    var a: List[Float64] = [-3.0, 2.0]
    canonical_sign(a, 0, 2)
    assert_equal(a[0], 3.0)
    assert_equal(a[1], -2.0)

    # Tie in absolute value -> the lowest index decides, and it is already
    # positive, so nothing moves.
    var b: List[Float64] = [1.0, -1.0]
    canonical_sign(b, 0, 2)
    assert_equal(b[0], 1.0)
    assert_equal(b[1], -1.0)

    # Same tie, opposite sign at the lowest index -> negate.
    var c: List[Float64] = [-1.0, 1.0]
    canonical_sign(c, 0, 2)
    assert_equal(c[0], 1.0)
    assert_equal(c[1], -1.0)

    # An all-zero vector has no largest component and is left alone rather
    # than being negated into a different all-zero vector.
    var z: List[Float64] = [0.0, 0.0]
    canonical_sign(z, 0, 2)
    assert_equal(z[0], 0.0)
    assert_equal(z[1], 0.0)


def test_cholesky_is_the_spotrf_catboost_never_calls() raises:
    # A = [[4, 2], [2, 5]] -> L = [[2, 0], [1, 2]], exact in binary.
    var a: List[Float64] = [4.0, 2.0, 2.0, 5.0]
    var l = cholesky_lower(a, 2)
    assert_equal(l[0], 2.0)
    assert_equal(l[1], 0.0)
    assert_equal(l[2], 1.0)
    assert_equal(l[3], 2.0)


def test_cholesky_refuses_a_singular_scatter() raises:
    # `reg = 0` on a rank-deficient within-class scatter. CatBoost would
    # hand this to `ssygst` and get whatever the arithmetic produced.
    var a: List[Float64] = [1.0, 1.0, 1.0, 1.0]
    with assert_raises():
        _ = cholesky_lower(a, 2)


def test_eigen_of_a_diagonal_matrix_sorts_descending() raises:
    var a: List[Float64] = [2.0, 0.0, 0.0, 1.0]
    var e = jacobi_eigen(a, 2, 60)
    sort_eigenpairs(e)
    assert_equal(e.values[0], 2.0)
    assert_equal(e.values[1], 1.0)
    # Column 0 is (1, 0), column 1 is (0, 1).
    assert_equal(e.vectors[0 * 2 + 0], 1.0)
    assert_equal(e.vectors[1 * 2 + 0], 0.0)
    assert_equal(e.vectors[0 * 2 + 1], 0.0)
    assert_equal(e.vectors[1 * 2 + 1], 1.0)


def test_eigen_of_a_rank_one_matrix() raises:
    # A = [[1, 1], [1, 1]]: eigenvalues 2 and 0, eigenvectors (1,1)/sqrt(2)
    # and (1,-1)/sqrt(2). Worked through the single Jacobi rotation by hand:
    # theta = 0, t = 1, c = s = 1/sqrt(2).
    var a: List[Float64] = [1.0, 1.0, 1.0, 1.0]
    var e = jacobi_eigen(a, 2, 60)
    sort_eigenpairs(e)
    var r = 1.0 / sqrt(2.0)
    assert_true(close(e.values[0], 2.0))
    assert_true(close(e.values[1], 0.0))
    assert_true(close(e.vectors[0 * 2 + 0], r))
    assert_true(close(e.vectors[1 * 2 + 0], r))
    # The zero-eigenvalue vector is (1, -1)/sqrt(2), sign-fixed so that its
    # first component (tied in absolute value) is positive.
    assert_true(close(e.vectors[0 * 2 + 1], r))
    assert_true(close(e.vectors[1 * 2 + 1], -r))


def test_eigenpair_ties_are_broken_by_the_vector_not_by_luck() raises:
    # The identity matrix has one eigenvalue of multiplicity 2, so the
    # eigenvalue comparison cannot order the pair and the lexicographic
    # tiebreak has to. Ascending lexicographic puts (0, 1) before (1, 0).
    var a: List[Float64] = [1.0, 0.0, 0.0, 1.0]
    var e = jacobi_eigen(a, 2, 60)
    sort_eigenpairs(e)
    assert_equal(e.values[0], 1.0)
    assert_equal(e.values[1], 1.0)
    assert_equal(e.vectors[0 * 2 + 0], 0.0)
    assert_equal(e.vectors[1 * 2 + 0], 1.0)
    assert_equal(e.vectors[0 * 2 + 1], 1.0)
    assert_equal(e.vectors[1 * 2 + 1], 0.0)


# ---------------------------------------------------------------------------
# LDA arithmetic, worked on paper first.
# ---------------------------------------------------------------------------


def test_lda_finds_the_separating_axis() raises:
    # Four points, two classes, separated along axis 0 and noisy along axis 1:
    #   class 0: (-1, 0), (-1, 1)      class 1: (1, 0), (1, 1)
    # mu_0 = (-1, 0.5), mu_1 = (1, 0.5), mu = (0, 0.5).
    # Cov_c = [[0, 0], [0, 0.25]] for both, so S_W = [[0,0],[0,0.25]] + reg I.
    # S_B = 0.5 mu_0 mu_0^T + 0.5 mu_1 mu_1^T - mu mu^T = [[1, 0], [0, 0]].
    # The generalized problem is diagonal: eigenvalues 1/reg (for (1,0)) and
    # 0 (for (0,1)). The top component is therefore (1, 0), normalized and
    # sign-fixed to itself.
    var e = EmbeddingMatrix(
        4, 2, [-1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    )
    var class_ids: List[Int] = [0, 0, 1, 1]
    var p = LdaParams(True, 1, CATBOOST_LDA_REG_DEFAULT, False, 60)
    var model = fit_lda(e, class_ids, 2, p)
    assert_equal(model.components, 1)
    assert_true(close(model.projection[0], 1.0))
    assert_true(close(model.projection[1], 0.0))

    # Applying it recovers the x coordinate, which is the whole point of the
    # projection: one number that separates the classes.
    var cols = apply_lda(model, e)
    assert_equal(len(cols), 1)
    assert_true(close(cols[0][0], -1.0))
    assert_true(close(cols[0][1], -1.0))
    assert_true(close(cols[0][2], 1.0))
    assert_true(close(cols[0][3], 1.0))


def test_lda_components_default_is_catboosts_cap() raises:
    var p = LdaParams.default()
    # min(n_classes - 1, dim - 1).
    assert_equal(p.resolved_components(2, 8), 1)
    assert_equal(p.resolved_components(5, 3), 2)
    assert_equal(p.resolved_components(10, 64), 9)
    # CB_ENSURE(ProjectionDim > 0) and CB_ENSURE(ProjectionDim < dim).
    with assert_raises():
        _ = p.resolved_components(1, 8)
    var explicit = LdaParams(True, 8, CATBOOST_LDA_REG_DEFAULT, False, 60)
    with assert_raises():
        _ = explicit.resolved_components(2, 8)
    # An explicit request above the rank of S_B is legal, as it is in
    # CatBoost, and is a bad idea; nothing here refuses it.
    var over = LdaParams(True, 5, CATBOOST_LDA_REG_DEFAULT, False, 60)
    assert_equal(over.resolved_components(2, 8), 5)


def test_lda_is_refused_for_regression() raises:
    # With one class cloud CatBoost's between-class scatter is identically
    # zero, so the projection is an arbitrary direction. CatBoost emits it;
    # this raises.
    var e = line_1d([0.0, 1.0, 2.0])
    var e2 = EmbeddingMatrix(3, 2, [0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
    var ids: List[Int] = [0, 0, 0]
    var p = LdaParams(True, 1, CATBOOST_LDA_REG_DEFAULT, False, 60)
    with assert_raises():
        _ = fit_lda(e2, ids, 1, p)
    with assert_raises():
        _ = lda_online_features(e2, ids, 1, identity_permutation(3), p)
    _ = e


def test_the_final_flush_quirk_is_a_switch_and_defaults_to_correct() raises:
    # CatBoost's `Flush` is not on `IEmbeddingCalcerVisitor`, so
    # `EstimateFeatureCalcer` never calls it and the deployed projection is
    # the one fitted at the largest power of two <= n. With five rows that
    # is four, and the fifth row -- the only class-1 row far from the rest --
    # never reaches the solve. The two arms must therefore disagree.
    var e = EmbeddingMatrix(
        5,
        2,
        [
            -1.0, 0.0,
            -1.0, 1.0,
            1.0, 0.0,
            1.0, 1.0,
            9.0, 9.0,
        ],
    )
    var ids: List[Int] = [0, 0, 1, 1, 1]
    var ours = LdaParams(True, 1, CATBOOST_LDA_REG_DEFAULT, False, 60)
    var theirs = LdaParams(True, 1, CATBOOST_LDA_REG_DEFAULT, True, 60)
    var m_ours = fit_lda(e, ids, 2, ours)
    var m_theirs = fit_lda(e, ids, 2, theirs)
    var same = close(m_ours.projection[0], m_theirs.projection[0]) and close(
        m_ours.projection[1], m_theirs.projection[1]
    )
    assert_false(same)


# ---------------------------------------------------------------------------
# Refusals, defaults, and the combined entry point.
# ---------------------------------------------------------------------------


def test_everything_is_off_by_default() raises:
    var p = EmbeddingEstimatorParams.default()
    assert_false(p.lda.enabled)
    assert_false(p.knn.enabled)
    assert_equal(p.knn.k, CATBOOST_KNN_K_DEFAULT)
    assert_equal(p.lda.reg, CATBOOST_LDA_REG_DEFAULT)
    assert_equal(p.lda.components, LDA_COMPONENTS_CATBOOST_DEFAULT)
    assert_false(p.lda.catboost_final_flush_only)

    var e = line_1d([0.0, 1.0])
    var targets: List[Float64] = [0.0, 1.0]
    var perm = identity_permutation(2)
    with assert_raises():
        _ = knn_online_features(e, targets, 2, perm, p.knn)
    with assert_raises():
        _ = compute_online_features(e, targets, 2, perm, p)


def test_catboost_defaults_turn_both_on() raises:
    # `DefaultEmbeddingCalcers` is {LDA, KNN}: both estimators run for every
    # embedding column over there. This has to be asked for by name here.
    var p = EmbeddingEstimatorParams.catboost_defaults()
    assert_true(p.lda.enabled)
    assert_true(p.knn.enabled)
    assert_equal(p.knn.k, 5)
    # 2 classes, 3 dimensions: LDA contributes min(1, 2) = 1 column and KNN
    # contributes one per class.
    assert_equal(online_feature_count(2, 3, p), 3)
    assert_equal(knn_feature_count(2), 2)
    assert_equal(knn_feature_count(0), 1)


def test_combined_output_is_lda_columns_then_knn_columns() raises:
    var e = EmbeddingMatrix(
        4, 2, [-1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    )
    var targets: List[Float64] = [0.0, 0.0, 1.0, 1.0]
    var p = EmbeddingEstimatorParams(
        LdaParams(True, 1, CATBOOST_LDA_REG_DEFAULT, False, 60), knn_on(1)
    )
    var perm = identity_permutation(4)
    var cols = compute_online_features(e, targets, 2, perm, p)
    assert_equal(len(cols), online_feature_count(2, 2, p))
    assert_equal(len(cols), 3)
    for j in range(3):
        assert_equal(len(cols[j]), 4)
    # The first column is LDA's and its first entry is the zero projection.
    assert_equal(cols[0][0], 0.0)
    # The last two are KNN's class counts, and the first row's are empty.
    assert_equal(cols[1][0], 0.0)
    assert_equal(cols[2][0], 0.0)


def test_knn_refuses_above_the_row_bound() raises:
    var e = line_1d([0.0, 1.0, 2.0])
    var targets: List[Float64] = [0.0, 1.0, 0.0]
    var bounded = KnnParams(True, 1, 2)
    with assert_raises():
        _ = knn_online_features(e, targets, 2, identity_permutation(3), bounded)
    with assert_raises():
        _ = fit_knn(e, targets, 2, bounded)


def test_permutation_is_checked_rather_than_trusted() raises:
    var e = line_1d([0.0, 1.0, 2.0])
    var targets: List[Float64] = [0.0, 1.0, 0.0]
    var repeated: List[Int] = [0, 0, 2]
    var short_perm: List[Int] = [0, 1]
    var out_of_range: List[Int] = [0, 1, 3]
    with assert_raises():
        check_permutation(repeated, 3)
    with assert_raises():
        check_permutation(short_perm, 3)
    with assert_raises():
        check_permutation(out_of_range, 3)
    with assert_raises():
        _ = knn_online_features(e, targets, 2, repeated, knn_on(1))
    check_permutation(identity_permutation(3), 3)


def test_class_targets_must_be_exact_integers_in_range() raises:
    var ok: List[Float64] = [0.0, 1.0, 2.0]
    var ids = class_ids_from_targets(ok, 3)
    assert_equal(ids[0], 0)
    assert_equal(ids[2], 2)
    var fractional: List[Float64] = [0.0, 1.5]
    var high: List[Float64] = [0.0, 3.0]
    var negative: List[Float64] = [-1.0, 0.0]
    with assert_raises():
        _ = class_ids_from_targets(fractional, 3)
    with assert_raises():
        _ = class_ids_from_targets(high, 3)
    with assert_raises():
        _ = class_ids_from_targets(negative, 3)


def test_embedding_matrix_rejects_a_bad_shape() raises:
    with assert_raises():
        _ = EmbeddingMatrix(2, 3, [1.0, 2.0, 3.0])
    with assert_raises():
        _ = EmbeddingMatrix(2, 0, [])
    var e = EmbeddingMatrix(2, 2, [1.0, 2.0, 3.0, 4.0])
    assert_equal(e.at(1, 0), 3.0)
    assert_equal(e.offset(1), 2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
