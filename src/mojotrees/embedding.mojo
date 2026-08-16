"""CatBoost's embedding feature estimators, `LDA` and `KNN`.

CatBoost accepts a raw embedding column and mojotrees cannot. What CatBoost
actually does with one is not "train on 768 float columns": it runs a set of
**estimators** that turn the embedding into a handful of numeric columns, and
those columns are then quantized by the ordinary float binning path and are
indistinguishable from any other float feature from there on. This module is
the generation half of that. It writes no trainer, no split rule and no
histogram; it produces `List[List[Float64]]` columns and hands them to the
binning path that already exists.

**Everything here is off by default and nothing in the mojotrees package
imports this module.** No existing default moves. `EmbeddingEstimatorParams`
carries `enabled = False` and every entry point refuses a disabled block by
name rather than quietly returning nothing.

Import direction, checked rather than assumed: this module imports `parallel`
and `std` and nothing else from the package, and no package module imports
it. `parallel` imports `apple_cpu_policy`. The one edge added therefore points
away from the data layer, so it cannot participate in the
`efb -> binning -> tree_parameters_extra` cycle this repository has paid for.

Source, verified
----------------

CatBoost `master`, read 2026-08-16. `docs/design/CATBOOST_CATALOG.md` A19 has
the full transcription; the load-bearing citations are repeated here.

- `catboost/private/libs/embedding_features/lda.{h,cpp}` --
  `TLinearDACalcer`, `IncrementalCloud`, `TotalScatterCalculation`,
  `CalculateProjection`, `TLinearDACalcerVisitor::Update`/`Flush`.
- `catboost/private/libs/embedding_features/knn.{h,cpp}` -- `TKNNCalcer`,
  `TKNNCalcer::Compute`, `TKNNCalcerVisitor::Update`, `TL2Distance`.
- `catboost/private/libs/feature_estimator/base_embedding_feature_estimator.h`
  -- `TEmbeddingBaseEstimator::ComputeOnlineFeatures` (44-82),
  `::EstimateFeatureCalcer` (137-151), `::Calc` (107-132),
  `::MakeFinalFeatureCalcer` (94-104).
- `catboost/private/libs/feature_estimator/embedding_feature_estimators.cpp`
  -- `TLDAEstimator` (10-79) and `TKNNEstimator` (81-121). **The reachable
  defaults are here**, not in the calcer constructors.
- `catboost/private/libs/options/embedding_processing_options.h`
  (`DefaultEmbeddingCalcers`, 37-42) -- both estimators are on by default for
  every embedding column.
- `catboost/private/libs/algo/fold.cpp`
  (`TFold::InitOnlineEstimatedFeatures`, 377-394, called at 200 and 298) and
  `catboost/private/libs/algo/estimated_features.cpp` (448-464).

The leakage rule, which is the whole design
-------------------------------------------

Both estimators read the target, so both leak if computed naively. CatBoost
prevents it the same way it prevents CTR leakage, with the same object: the
fold's permutation. `ComputeOnlineFeatures` is

    for (ui64 line : learnPermutation) {
        Compute(featureCalcer, vector, line, ...);      // FIRST
        calcerVisitor.Update(target[line], vector, ...); // SECOND
    }

**Compute strictly precedes Update.** Row `i`'s feature is a function of the
rows that precede `i` in the permutation and of nothing else. Two things fall
out that an implementation gets wrong if it does not read this loop:

- The first row of the permutation is scored against an empty state. Its KNN
  neighbourhood is empty (all-zero class counts, a regression mean of 0.0
  under CatBoost's explicit `if (neighbors.size())` guard) and its LDA
  projection matrix is still zero.
- **The KNN query point is excluded from its own neighbourhood by
  construction, not by a filter.** There is no self-skip anywhere in
  `knn.cpp`; the point is simply not in the cloud yet, because `AddItem` runs
  after the query. Fit the structure over the whole training set first and
  that exclusion silently disappears, and every training row becomes its own
  nearest neighbour at distance zero.

The train-versus-predict asymmetry
----------------------------------

At **train** time the feature is a function of a strict prefix of a
permutation. At **predict** time it is a function of the entire learn set:
test rows go through `Calc` against `EstimateFeatureCalcer`, which loops the
whole learn set in dataset order calling only `Update`, and
`MakeFinalFeatureCalcer` -- the calcer serialized into the model -- is that
same full-data object. So the mapping the deployed model applies was never
applied to a single training row, and row `i`'s training feature and the
value the model would compute for row `i` are different numbers on purpose.
There is no target for the query row at predict time and none is needed: the
target-awareness lives entirely in the fitted state.

This module therefore has **two** entry points per estimator and they are not
interchangeable:

- `lda_online_features` / `knn_online_features` -- train. Prefix-only.
- `fit_lda` + `apply_lda`, `fit_knn` + `apply_knn` -- predict. Full learn set.

Using the fitted model at train time reintroduces exactly the leak the loop
exists to prevent. Using a prefix at predict time throws away half the data
for nothing.

The permutation is an argument, never generated here
----------------------------------------------------

This module owns no RNG, no shuffle and no fold. `compute_online_features`
takes the permutation as a `List[Int]`. CatBoost calls
`CreateEstimatedFeaturesData` and `InitOnlineCtrs` off the same `TFold`
array a few lines apart, and two permutations where CatBoost has one would be
both slower and a different model. `identity_permutation` exists for tests
and is not a training permutation.

What LDA computes, and two LAPACK calls CatBoost is missing
-----------------------------------------------------------

The problem is the classical Fisher one, `S_B v = lambda S_W v`, with the
within-class scatter regularized on its diagonal. CatBoost's identifiers are
misnamed -- `BetweenMatrix` holds the **within**-class scatter and the local
`totalScatter` holds the **between**-class scatter -- but the mechanism is
standard.

`CalculateProjection` (`lda.cpp:10-37`) calls `ssygst_` then `ssyev_` then
copies the tail of the reduced matrix out. Two steps of the standard recipe
are absent, and both are bugs:

1. **`spotrf_` is never called.** LAPACK's `ssygst` documents that its `B`
   argument must be the Cholesky factor returned by `SPOTRF`. CatBoost hands
   it the raw regularized within-class matrix, so the reduction is by a
   matrix treated as if it were its own Cholesky factor.
2. **The back-transform is never applied.** After `ssyev` the vectors belong
   to the reduced matrix; the generalized eigenvectors need
   `x = inv(L^T) y`, a `strsm` that is not there.

We put both back. Consequence, stated rather than buried: **our LDA column is
not CatBoost's LDA column and a numeric parity test against it would fail by
design.** The first omission is not even reproducible -- what `ssygst` does
with a non-factor `B` is whatever the arithmetic does, and that differs
between LAPACK builds -- so a parity claim would be a claim about one
machine's reference BLAS.

The one subtle step CatBoost gets right, and we copy: `ssyev` returns
eigenvalues **ascending** and eigenvectors as **columns** in column-major
order, so the tail of the buffer is exactly the top-`k` eigenvectors laid out
contiguously, and reading that tail row-major recovers them as rows.

What KNN computes, and why it is bounded here
---------------------------------------------

Squared L2 (`NHnsw::TL2SqrDistance<float>`, `knn.h:17`), `k = 5` by default,
class **counts** among the neighbours for classification (not normalized, not
distance-weighted) and the unweighted target mean for regression.

CatBoost's structure is an **online HNSW graph**, built with
`TOnlineHnswBuildOptions({CloseNum, 300})`. That is approximate. CatBoost's
KNN feature is not the exact k nearest neighbours and does not claim to be;
what it returns depends on insertion order, which is the fold permutation.

We do it exactly, and exactly is quadratic. **Derived bound, not measured:**
the online pass over `n` rows performs `sum_{i<n} i = n(n-1)/2` distance
evaluations of `d` multiply-adds each, `Theta(n^2 d / 2)`. At `n = 10^6` and
`d = 64` that is `3.2e13` multiply-adds for one column of one fold. That does
not finish, so **exact KNN cannot appear in a 1M-row comparison** and this
module does not pretend it can: `KnnParams.max_rows` defaults to 50,000 and
the entry points raise above it rather than silently entering an `n^2` loop.
Refusing loudly beats an approximate index whose output we could not
reproduce.

LDA's bound is linear in rows and quadratic in width: `Theta(n d^2)` for the
scatter accumulation, plus `O(log n)` eigensolves at `O(d^3)` per sweep. The
`O(log n)` is CatBoost's doubling flush schedule, reproduced here, and it is
the reason the eigensolve is not where the time goes.

Determinism
-----------

Bit-identity against CatBoost is neither required nor claimed. Determinism
across `MOJOTREES_NUM_WORKERS` and across machines is required, and this
module has three hazards the rest of the package does not.

**An eigenvector's sign is arbitrary.** If `v` is an eigenvector so is `-v`,
and a solver may return either; a projection feature whose sign flips run to
run is not wrong and is not reproducible. Pinned by `canonical_sign`: the
component of largest absolute value is made positive, ties in absolute value
go to the lowest index, an all-zero vector is left alone. No free choice in
it.

**Eigenvalue ties leave the ordering free**, and they will occur: with `C`
classes, `rank(S_B) <= C - 1`, so any component past that sits in a null
space of exact zeros. Pinned by a total order on eigenpairs -- descending
eigenvalue, then ascending lexicographic order of the sign-fixed vector --
applied by a deterministic insertion sort, not a library sort whose tie
behavior we do not control.

**KNN neighbour ties must not be broken by whichever thread got there
first.** Pinned by the total order `(distance, row index)` ascending, with a
non-finite distance mapped to `+inf` so a NaN coordinate cannot poison a
comparison into non-transitivity. Row indices are unique, so the selected `k`
is a set-valued function of the candidate set alone and is independent of the
order the candidates were examined in.

Where the parallelism is: the KNN pass fans out over **query positions**,
each of which scans its own prefix serially and writes only its own output
slots, so worker count cannot change a value. The LDA pass is **serial by
construction** -- each row depends on the state after the previous one -- and
its scatter sums are accumulated in ascending permutation order, so the float
addend order is fixed by the loop and not by a scheduler. No reduction here
is split across workers.

Deliberate divergences from CatBoost
------------------------------------

Each is a decision, not an oversight, and each is in A19 with its reason.

1. The generalized eigenproblem is solved properly (see above).
2. The eigensolver is cyclic Jacobi with a fixed sweep order, for
   determinism and to avoid a LAPACK dependency.
3. **LDA is refused for regression.** With one class cloud CatBoost's
   between-class scatter is `1.0 * mu mu^T - mu mu^T`, identically zero, so
   every eigenvalue is zero and the projection is an arbitrary direction
   carrying no target information. CatBoost emits it anyway. We raise.
4. Exact KNN with a row bound and a refusal, instead of HNSW.
5. Class scatter is accumulated as raw second moments
   (`Cov_c = M_c/n_c - mu_c mu_c^T`) rather than CatBoost's shifted
   incremental update. One pass and exact in exact arithmetic; in float it is
   the less stable formulation for data far from the origin. Embeddings are
   conventionally near-centered, which is the reason the trade is acceptable
   and not the reason it is invisible.
6. **CatBoost's final calcer is under-fitted and we fix it by default.**
   `IEmbeddingCalcerVisitor` declares only `Update`;
   `TLinearDACalcerVisitor::Flush` is not virtual and not on the interface,
   so `EstimateFeatureCalcer` cannot call it and does not. The only caller is
   `Update`'s doubling schedule, so the deployed projection is the one fitted
   at the largest power of two `<= n` -- on 1,000 rows, the first 512.
   `LdaParams.catboost_final_flush_only` defaults `False` (re-solve on all
   `n`); set it `True` for that parity. The *online* doubling schedule is
   reproduced unconditionally, because there it is not a quirk but the reason
   the pass costs `O(log n)` eigensolves instead of `n`.
7. `Float64` throughout, where CatBoost is `float`.
8. `likelihood` (LDA's optional per-class normalized Mahalanobis kernels) is
   not implemented. It is off by CatBoost's own default.
"""

from std.math import isfinite, sqrt
from std.memory import bitcast

from .parallel import dispatch_rows


comptime POSITIVE_INF = bitcast[DType.float64, 1](
    SIMD[DType.uint64, 1](0x7FF0000000000000)
)
"""`+inf`, spelled from its bit pattern exactly as `binning.POSITIVE_INF` is,
so this module needs no library constant and no import from `binning`. It is
where a non-finite KNN distance goes; see `_finite_or_inf`."""


# ---------------------------------------------------------------------------
# CatBoost's reachable defaults.
#
# "Reachable" is load-bearing. `TLinearDACalcer`'s constructor defaults
# (`totalDimension=2, projectionDimension=1, regularization=0.01`) are dead:
# `TLDAEstimator::CreateFeatureCalcer` always passes its own values. The
# numbers below are the estimator's, which is what a user actually gets.
# ---------------------------------------------------------------------------

comptime CATBOOST_LDA_REG_DEFAULT = 0.00005
"""`TLDAEstimator`, `embedding_feature_estimators.cpp:28-32`. NOT the 0.01 in
`TLinearDACalcer`'s signature, which the estimator always overrides."""

comptime CATBOOST_KNN_K_DEFAULT = 5
"""`TKNNEstimator`, `embedding_feature_estimators.cpp:96-100`. Agrees with
the calcer signature, unlike LDA's regularizer."""

comptime CATBOOST_HNSW_SEARCH_NEIGHBORHOOD = 300
"""`TOnlineHnswBuildOptions({CloseNum, 300})`, `knn.h:109`. Recorded because
it is half of what makes CatBoost's neighbourhood approximate; nothing here
reads it, because nothing here is approximate."""

comptime KNN_MAX_ROWS_DEFAULT = 50000
"""OURS, not CatBoost's. The row count above which exact KNN is refused
rather than run. See the module docstring's derived bound: the pass is
`Theta(n^2 d / 2)` and 50,000 rows at width 64 is already `8e10`
multiply-adds."""

comptime LDA_COMPONENTS_CATBOOST_DEFAULT = -1
"""Sentinel for `components`: resolve to CatBoost's
`min(n_classes - 1, dim - 1)` rather than a fixed number."""

comptime JACOBI_MAX_SWEEPS_DEFAULT = 60
"""Cyclic Jacobi sweep cap. Convergence is normally reached in well under ten
sweeps for a symmetric matrix; the cap exists so a pathological input
terminates rather than spins, and it is a fixed constant so two machines
perform the same rotations."""

comptime JACOBI_TOLERANCE = 1.0e-14
"""Relative off-diagonal Frobenius threshold at which a sweep stops. Fixed,
not derived from machine epsilon at runtime, so the stopping point is a
property of the input and not of the host."""


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------


struct EmbeddingMatrix(Copyable, Movable):
    """One embedding column: `n_rows` vectors of `dim` floats, row-major.

    Row-major because every consumer here walks one row at a time (a KNN
    distance, an outer product into a class scatter), which is the opposite
    of the column-major layout the binned matrix wants and the reason this is
    its own container rather than a reuse of one.
    """

    var n_rows: Int
    var dim: Int
    var values: List[Float64]

    def __init__(
        out self, n_rows: Int, dim: Int, var values: List[Float64]
    ) raises:
        if n_rows < 0:
            raise Error("EmbeddingMatrix: n_rows must not be negative")
        if dim < 1:
            raise Error("EmbeddingMatrix: dim must be at least 1")
        if len(values) != n_rows * dim:
            raise Error(
                "EmbeddingMatrix: expected n_rows * dim values, got a"
                " different count"
            )
        self.n_rows = n_rows
        self.dim = dim
        self.values = values^

    @always_inline
    def offset(self, row: Int) -> Int:
        return row * self.dim

    @always_inline
    def at(self, row: Int, j: Int) -> Float64:
        return self.values[row * self.dim + j]


@fieldwise_init
struct LdaParams(Copyable, Movable):
    """CatBoost's `LDA` estimator options, plus two of ours.

    `components` and `reg` are CatBoost's `components` and `reg` JSON keys.
    `catboost_final_flush_only` and `jacobi_max_sweeps` are ours and have no
    CatBoost equivalent; see divergences 2 and 6 in the module docstring.
    """

    var enabled: Bool
    """Off by default. Nothing in this module runs against a disabled block;
    it raises instead of returning empty columns."""

    var components: Int
    """`ProjectionDimension`. `LDA_COMPONENTS_CATBOOST_DEFAULT` (-1) resolves
    to CatBoost's `min(n_classes - 1, dim - 1)`."""

    var reg: Float64
    """Added to the diagonal of the within-class scatter and to nothing else
    (`lda.cpp:194-196`). CatBoost permits 0; at 0 a rank-deficient scatter
    has nothing holding it up and the Cholesky here will refuse it, which is
    a better failure than an arbitrary projection."""

    var catboost_final_flush_only: Bool
    """OURS. `False` (default) re-solves the final calcer on all `n` rows.
    `True` reproduces CatBoost's largest-power-of-two-prefix behavior, which
    is a consequence of `Flush` not being on the visitor interface."""

    var jacobi_max_sweeps: Int
    """OURS. Sweep cap for the eigensolver."""

    @staticmethod
    def default() -> Self:
        """CatBoost's reachable defaults, with the estimator **off**."""
        return Self(
            False,
            LDA_COMPONENTS_CATBOOST_DEFAULT,
            CATBOOST_LDA_REG_DEFAULT,
            False,
            JACOBI_MAX_SWEEPS_DEFAULT,
        )

    def resolved_components(self, n_classes: Int, dim: Int) raises -> Int:
        """CatBoost's default and its two guards
        (`embedding_feature_estimators.cpp:20-27, 42-49`).

        The default cap of `n_classes - 1` is the right one and CatBoost
        picks it: `S_B` is a sum of `C` rank-one terms under one linear
        constraint, so `rank(S_B) <= C - 1` and any further component has
        eigenvalue zero. **The cap is only the default.** An explicit larger
        `components` is accepted here as it is there, and the extra columns
        are then directions in a null space whose only content is what the
        arithmetic left behind. It is a legal request and a bad one.
        """
        var k: Int
        if self.components == LDA_COMPONENTS_CATBOOST_DEFAULT:
            k = n_classes - 1
            if dim - 1 < k:
                k = dim - 1
        else:
            k = self.components
        if k <= 0:
            raise Error(
                "lda: components must be positive; a 1-class problem or a"
                " 1-dimensional embedding has no discriminant direction"
            )
        if k >= dim:
            raise Error(
                "lda: components must be strictly less than the embedding"
                " dimension, as CatBoost's CB_ENSURE requires"
            )
        return k


@fieldwise_init
struct KnnParams(Copyable, Movable):
    """CatBoost's `KNN` estimator options, plus our row bound."""

    var enabled: Bool
    """Off by default."""

    var k: Int
    """`CloseNum`, CatBoost's `k` key, default 5."""

    var max_rows: Int
    """OURS. Above this the exact pass is refused rather than run; see the
    module docstring's quadratic bound. There is no CatBoost equivalent
    because CatBoost's structure is not exact."""

    @staticmethod
    def default() -> Self:
        return Self(False, CATBOOST_KNN_K_DEFAULT, KNN_MAX_ROWS_DEFAULT)


@fieldwise_init
struct EmbeddingEstimatorParams(Copyable, Movable):
    """Both estimators for one embedding column.

    CatBoost's `DefaultEmbeddingCalcers` (`embedding_processing_options.h:
    37-42`) is `{LDA, KNN}`, so **both are on by default for every embedding
    column** over there. Ours are both off, which is the whole difference in
    posture: this is an opt-in mechanism in a package whose defaults are
    LightGBM's.
    """

    var lda: LdaParams
    var knn: KnnParams

    @staticmethod
    def default() -> Self:
        return Self(LdaParams.default(), KnnParams.default())

    @staticmethod
    def catboost_defaults() -> Self:
        """Both estimators on at CatBoost's parameter values, which is what a
        CatBoost run with an embedding column and no other options does. Not
        a default here; a caller has to ask for it by name."""
        return Self(
            LdaParams(
                True,
                LDA_COMPONENTS_CATBOOST_DEFAULT,
                CATBOOST_LDA_REG_DEFAULT,
                False,
                JACOBI_MAX_SWEEPS_DEFAULT,
            ),
            KnnParams(True, CATBOOST_KNN_K_DEFAULT, KNN_MAX_ROWS_DEFAULT),
        )


# ---------------------------------------------------------------------------
# Permutation helpers. This module generates no permutation; see the module
# docstring and A19 section 9.
# ---------------------------------------------------------------------------


def identity_permutation(n: Int) -> List[Int]:
    """`[0, 1, ..., n-1]`.

    **This is not a training permutation.** It exists so a test can exercise
    the prefix loop without pulling in an RNG. Handing it to
    `lda_online_features` on real data means the "ordered" statistic is
    ordered by whatever order the rows arrived in, which is exactly the
    correlation an ordered statistic exists to break.
    """
    var out = List[Int]()
    for i in range(n):
        out.append(i)
    return out^


def check_permutation(permutation: List[Int], n_rows: Int) raises:
    """Every row exactly once. A permutation with a repeat would let one row
    into another's prefix twice and a permutation with a gap would leave a
    row's feature unwritten, and both are silent."""
    if len(permutation) != n_rows:
        raise Error("embedding: permutation length must equal the row count")
    var seen = List[Bool]()
    for _ in range(n_rows):
        seen.append(False)
    for pos in range(n_rows):
        var r = permutation[pos]
        if r < 0 or r >= n_rows:
            raise Error("embedding: permutation entry out of range")
        if seen[r]:
            raise Error("embedding: permutation repeats a row")
        seen[r] = True


def class_ids_from_targets(
    targets: List[Float64], n_classes: Int
) raises -> List[Int]:
    """CatBoost's `(size_t)target` / `(ui32)target` cast (`lda.cpp:167`,
    `knn.cpp:63`), with the bounds check CatBoost leaves to `.at()`.

    CatBoost truncates toward zero and indexes a vector sized `numClasses`,
    so a target of 2.7 in a 3-class problem lands in class 2 and a target of
    3.0 is undefined behavior over there. We refuse anything that is not an
    exact non-negative integer below `n_classes`.
    """
    var out = List[Int]()
    for i in range(len(targets)):
        var t = targets[i]
        if not isfinite(t):
            raise Error("embedding: class target must be finite")
        var c = Int(t)
        if Float64(c) != t:
            raise Error(
                "embedding: class target must be an exact integer; CatBoost"
                " truncates silently and this refuses instead"
            )
        if c < 0 or c >= n_classes:
            raise Error("embedding: class target out of range")
        out.append(c)
    return out^


# ---------------------------------------------------------------------------
# Dense linear algebra. Everything is row-major `d x d` in a flat
# `List[Float64]`, everything is serial, and every loop order is fixed by the
# source rather than by a scheduler.
# ---------------------------------------------------------------------------


def _identity(d: Int) -> List[Float64]:
    var m = List[Float64]()
    for i in range(d):
        for j in range(d):
            if i == j:
                m.append(1.0)
            else:
                m.append(0.0)
    return m^


def cholesky_lower(a: List[Float64], d: Int) raises -> List[Float64]:
    """`L` with `L L^T = A`, lower triangular, zeros above the diagonal.

    This is the `spotrf_` CatBoost does not call. It raises on a
    non-positive-definite input, which is the honest outcome for a singular
    within-class scatter: there is no discriminant direction to find, and
    `reg` is the knob that fixes it.
    """
    var l = List[Float64]()
    for _ in range(d * d):
        l.append(0.0)
    for i in range(d):
        for j in range(i + 1):
            var s = a[i * d + j]
            for k in range(j):
                s -= l[i * d + k] * l[j * d + k]
            if i == j:
                if not (s > 0.0):
                    raise Error(
                        "lda: within-class scatter is not positive definite;"
                        " raise `reg` or reduce the embedding dimension"
                    )
                l[i * d + i] = sqrt(s)
            else:
                l[i * d + j] = s / l[j * d + j]
    return l^


def _solve_lower(l: List[Float64], d: Int, b: List[Float64]) -> List[Float64]:
    """`X` with `L X = B`, forward substitution, `B` and `X` both `d x d`
    row-major. Column `c` of `X` depends only on column `c` of `B`."""
    var x = List[Float64]()
    for _ in range(d * d):
        x.append(0.0)
    for c in range(d):
        for i in range(d):
            var s = b[i * d + c]
            for k in range(i):
                s -= l[i * d + k] * x[k * d + c]
            x[i * d + c] = s / l[i * d + i]
    return x^


def _solve_lower_transpose_vec(
    l: List[Float64], d: Int, b: List[Float64], b_offset: Int
) -> List[Float64]:
    """`x` with `L^T x = b`, back substitution. This is the `strsm` CatBoost
    does not call: it turns an eigenvector of the reduced problem into an
    eigenvector of the generalized one."""
    var x = List[Float64]()
    for _ in range(d):
        x.append(0.0)
    var i = d - 1
    while i >= 0:
        var s = b[b_offset + i]
        for k in range(i + 1, d):
            s -= l[k * d + i] * x[k]
        x[i] = s / l[i * d + i]
        i -= 1
    return x^


def reduce_to_standard(
    l: List[Float64], d: Int, sb: List[Float64]
) -> List[Float64]:
    """`C = inv(L) S_B inv(L)^T`, the reduction `ssygst` is supposed to do.

    Computed as two triangular solves: `W = inv(L) S_B`, then `C^T =
    inv(L) W^T` -- and `C` is symmetric, so that second solve gives `C`
    directly. The result is symmetrized afterwards, because the two halves
    are computed by different chains of floating-point operations and may
    differ in the last bit; an asymmetric input to a symmetric eigensolver is
    a silent wrong answer rather than an error.
    """
    var w = _solve_lower(l, d, sb)
    var wt = List[Float64]()
    for _ in range(d * d):
        wt.append(0.0)
    for i in range(d):
        for j in range(d):
            wt[i * d + j] = w[j * d + i]
    var c = _solve_lower(l, d, wt)
    for i in range(d):
        for j in range(i + 1, d):
            var m = 0.5 * (c[i * d + j] + c[j * d + i])
            c[i * d + j] = m
            c[j * d + i] = m
    return c^


struct Eigen(Copyable, Movable):
    """A symmetric eigendecomposition: `values[j]` with eigenvector
    `vectors[:, j]` (column `j` of a `d x d` row-major buffer)."""

    var d: Int
    var values: List[Float64]
    var vectors: List[Float64]

    def __init__(
        out self, d: Int, var values: List[Float64], var vectors: List[Float64]
    ):
        self.d = d
        self.values = values^
        self.vectors = vectors^


def jacobi_eigen(
    a_in: List[Float64], d: Int, max_sweeps: Int
) raises -> Eigen:
    """Cyclic Jacobi eigendecomposition of a symmetric `d x d` matrix.

    Chosen over a LAPACK driver for two reasons, both determinism: the sweep
    order over `(p, q)` pairs is fixed by these loops, and the rotation for a
    pair is a closed form with no pivoting and no blocking, so two machines
    perform the same arithmetic in the same order. A blocked `ssyev` makes no
    such promise, and A19 records that an eigenvector's sign being arbitrary
    is enough on its own to make a feature irreproducible.

    Cost is `O(d^3)` per sweep with a small sweep count, worse than a tuned
    `ssyev` by a constant. On a `d x d` problem where `d` is an embedding
    width, run `O(log n)` times per fold, that constant is not where the time
    goes: the `Theta(n d^2)` scatter accumulation is.

    Eigenvalues come back in **no particular order**. `sort_eigenpairs`
    imposes one, and it is a total one.
    """
    if max_sweeps < 1:
        raise Error("jacobi_eigen: max_sweeps must be at least 1")
    var a = a_in.copy()
    var v = _identity(d)
    if d == 1:
        var one_val = List[Float64]()
        one_val.append(a[0])
        var one_vec = List[Float64]()
        one_vec.append(1.0)
        return Eigen(1, one_val^, one_vec^)

    # Scale reference for the stopping test. Fixed relative threshold, not a
    # runtime machine epsilon, so where the sweep stops is a property of the
    # input and not of the host.
    var frob = 0.0
    for i in range(d * d):
        frob += a[i] * a[i]
    var stop = JACOBI_TOLERANCE * JACOBI_TOLERANCE * frob

    for _ in range(max_sweeps):
        var off = 0.0
        for p in range(d):
            for q in range(p + 1, d):
                off += 2.0 * a[p * d + q] * a[p * d + q]
        if off <= stop:
            break
        for p in range(d):
            for q in range(p + 1, d):
                var apq = a[p * d + q]
                if apq == 0.0:
                    continue
                var theta = (a[q * d + q] - a[p * d + p]) / (2.0 * apq)
                var denom = abs(theta) + sqrt(theta * theta + 1.0)
                var t: Float64
                if theta >= 0.0:
                    t = 1.0 / denom
                else:
                    t = -1.0 / denom
                var c = 1.0 / sqrt(t * t + 1.0)
                var s = t * c
                # Rotate rows/cols p and q.
                for k in range(d):
                    var akp = a[k * d + p]
                    var akq = a[k * d + q]
                    a[k * d + p] = c * akp - s * akq
                    a[k * d + q] = s * akp + c * akq
                for k in range(d):
                    var apk = a[p * d + k]
                    var aqk = a[q * d + k]
                    a[p * d + k] = c * apk - s * aqk
                    a[q * d + k] = s * apk + c * aqk
                for k in range(d):
                    var vkp = v[k * d + p]
                    var vkq = v[k * d + q]
                    v[k * d + p] = c * vkp - s * vkq
                    v[k * d + q] = s * vkp + c * vkq

    var vals = List[Float64]()
    for i in range(d):
        vals.append(a[i * d + i])
    return Eigen(d, vals^, v^)


def canonical_sign(mut vec: List[Float64], offset: Int, d: Int):
    """Pin an eigenvector's sign. See the module docstring's first hazard.

    The component of largest absolute value is made positive; a tie in
    absolute value goes to the **lowest index**; an all-zero vector is left
    alone. There is no free choice anywhere in that rule, which is the point
    -- an eigenvector and its negation are both correct answers and a solver
    is free to return either, so a feature built on one is nondeterministic
    without ever being wrong.
    """
    var best = 0
    var best_abs = -1.0
    for j in range(d):
        var m = abs(vec[offset + j])
        if m > best_abs:
            best_abs = m
            best = j
    if best_abs <= 0.0:
        return
    if vec[offset + best] < 0.0:
        for j in range(d):
            vec[offset + j] = -vec[offset + j]


def _vector_less(
    a: List[Float64], a_off: Int, b: List[Float64], b_off: Int, d: Int
) -> Bool:
    """Ascending lexicographic order on two `d`-vectors. The tiebreak of last
    resort in `sort_eigenpairs`; if it also ties, the vectors are equal and
    the order genuinely does not matter."""
    for j in range(d):
        var x = a[a_off + j]
        var y = b[b_off + j]
        if x != y:
            return x < y
    return False


def sort_eigenpairs(mut e: Eigen):
    """Descending eigenvalue, ties broken by ascending lexicographic order of
    the sign-fixed eigenvector.

    Ties are not hypothetical: with `C` classes `rank(S_B) <= C - 1`, so
    every component past that has an eigenvalue of exactly zero and a whole
    null space to pick a direction from. Insertion sort, written out, rather
    than a library sort whose tie behavior is not part of its contract.
    """
    var d = e.d
    # Sign-fix every column first, so the lexicographic tiebreak compares
    # canonical representatives rather than whichever sign came out.
    var col = List[Float64]()
    for _ in range(d):
        col.append(0.0)
    for j in range(d):
        for i in range(d):
            col[i] = e.vectors[i * d + j]
        canonical_sign(col, 0, d)
        for i in range(d):
            e.vectors[i * d + j] = col[i]

    # Insertion sort over column indices under the total order.
    var order = List[Int]()
    for j in range(d):
        order.append(j)
    var flat = List[Float64]()
    for j in range(d):
        for i in range(d):
            flat.append(e.vectors[i * d + j])
    for j in range(1, d):
        var cur = order[j]
        var i = j - 1
        while i >= 0:
            var prev = order[i]
            var swap: Bool
            if e.values[prev] != e.values[cur]:
                swap = e.values[prev] < e.values[cur]
            else:
                swap = _vector_less(flat, cur * d, flat, prev * d, d)
            if not swap:
                break
            order[i + 1] = prev
            i -= 1
        order[i + 1] = cur

    var new_vals = List[Float64]()
    var new_vecs = List[Float64]()
    for _ in range(d * d):
        new_vecs.append(0.0)
    for j in range(d):
        new_vals.append(e.values[order[j]])
        for i in range(d):
            new_vecs[i * d + j] = flat[order[j] * d + i]
    e.values = new_vals^
    e.vectors = new_vecs^


# ---------------------------------------------------------------------------
# LDA
# ---------------------------------------------------------------------------


struct LdaModel(Copyable, Movable):
    """A fitted projection: `components x dim`, row-major, one eigenvector
    per row.

    That layout is CatBoost's and it is not an accident: `ssyev`'s
    column-major eigenvector columns, read as a row-major `k x d` block, are
    exactly this, and `TLinearDACalcer::Compute`'s
    `cblas_sgemv(RowMajor, NoTrans, M=k, N=d, lda=d)` consumes it directly.
    """

    var dim: Int
    var components: Int
    var projection: List[Float64]

    def __init__(
        out self, dim: Int, components: Int, var projection: List[Float64]
    ) raises:
        if len(projection) != dim * components:
            raise Error("LdaModel: projection must be components * dim")
        self.dim = dim
        self.components = components
        self.projection = projection^

    @staticmethod
    def zero(dim: Int, components: Int) raises -> Self:
        """The state before the first flush. CatBoost's `ProjectionMatrix` is
        constructed zeroed and `Compute` runs against it, so the first rows
        of a permutation legitimately project to zero."""
        var p = List[Float64]()
        for _ in range(dim * components):
            p.append(0.0)
        return Self(dim, components, p^)

    def apply_into(
        self, x: List[Float64], x_offset: Int, mut out: List[Float64]
    ):
        """`out[j] = sum_k P[j][k] * x[k]`, the matrix-vector product of
        `TLinearDACalcer::Compute`."""
        for j in range(self.components):
            var s = 0.0
            for k in range(self.dim):
                s += self.projection[j * self.dim + k] * x[x_offset + k]
            out[j] = s


struct _LdaAccumulator(Copyable, Movable):
    """Per-class first and second moments, and the flush schedule.

    CatBoost keeps a running centered covariance per class
    (`IncrementalCloud`). We keep raw sums instead --- `n_c`, `sum_c`,
    `M_c = sum x x^T` --- and form the covariance at flush as
    `M_c/n_c - mu_c mu_c^T`. One pass, no shift bookkeeping, exact in exact
    arithmetic; less stable in float for data far from the origin, which is
    divergence 5 and is a trade rather than an improvement.
    """

    var dim: Int
    var n_classes: Int
    var counts: List[Float64]
    var sums: List[Float64]
    var moments: List[Float64]
    var size: Int
    var last_flush: Int

    def __init__(out self, dim: Int, n_classes: Int):
        self.dim = dim
        self.n_classes = n_classes
        self.counts = List[Float64]()
        for _ in range(n_classes):
            self.counts.append(0.0)
        self.sums = List[Float64]()
        for _ in range(n_classes * dim):
            self.sums.append(0.0)
        self.moments = List[Float64]()
        for _ in range(n_classes * dim * dim):
            self.moments.append(0.0)
        self.size = 0
        self.last_flush = 0

    def add(mut self, x: List[Float64], x_offset: Int, class_id: Int):
        """One row into its class cloud. `Theta(d^2)` per row, which is the
        term that dominates the LDA pass."""
        var d = self.dim
        self.counts[class_id] += 1.0
        var s_off = class_id * d
        var m_off = class_id * d * d
        for i in range(d):
            var xi = x[x_offset + i]
            self.sums[s_off + i] += xi
            var row = m_off + i * d
            for j in range(d):
                self.moments[row + j] += xi * x[x_offset + j]
        self.size += 1

    def wants_flush(self) -> Bool:
        """CatBoost's `if (2 * LastFlush <= Size)`
        (`lda.cpp:169-172`), which fires at sizes 1, 2, 4, 8, ... That is
        what makes the online pass cost `O(log n)` eigensolves instead of
        `n`, and it is reproduced unconditionally for that reason."""
        return 2 * self.last_flush <= self.size

    def within_scatter(self, reg: Float64) raises -> List[Float64]:
        """`S_W = (1/N) sum_c (M_c - n_c mu_c mu_c^T)`, plus `reg` on the
        diagonal.

        Algebraically `sum_c (n_c/N) Cov_c`, which is CatBoost's
        `BetweenMatrix` --- a name that means the opposite of what it holds
        (`lda.cpp:181-196`). The form used here avoids dividing by `n_c` and
        then multiplying by it again.
        """
        var d = self.dim
        var n = Float64(self.size)
        var sw = List[Float64]()
        for _ in range(d * d):
            sw.append(0.0)
        if self.size == 0:
            raise Error("lda: cannot build a scatter from zero rows")
        for c in range(self.n_classes):
            var nc = self.counts[c]
            if nc <= 0.0:
                continue
            var s_off = c * d
            var m_off = c * d * d
            for i in range(d):
                var mui = self.sums[s_off + i] / nc
                for j in range(d):
                    var muj = self.sums[s_off + j] / nc
                    sw[i * d + j] += (
                        self.moments[m_off + i * d + j] - nc * mui * muj
                    ) / n
        for i in range(d):
            sw[i * d + i] += reg
        return sw^

    def between_scatter(self) raises -> List[Float64]:
        """`S_B = sum_c (n_c/N) mu_c mu_c^T - mu mu^T`, CatBoost's
        `TotalScatterCalculation` (`lda.cpp:140-161`) --- again a name that
        means the opposite of what it holds."""
        var d = self.dim
        var n = Float64(self.size)
        if self.size == 0:
            raise Error("lda: cannot build a scatter from zero rows")
        var sb = List[Float64]()
        for _ in range(d * d):
            sb.append(0.0)
        var mean = List[Float64]()
        for i in range(d):
            var acc = 0.0
            for c in range(self.n_classes):
                acc += self.sums[c * d + i]
            mean.append(acc / n)
        for c in range(self.n_classes):
            var nc = self.counts[c]
            if nc <= 0.0:
                continue
            var w = nc / n
            var s_off = c * d
            for i in range(d):
                var mui = self.sums[s_off + i] / nc
                for j in range(d):
                    sb[i * d + j] += w * mui * (self.sums[s_off + j] / nc)
        for i in range(d):
            for j in range(d):
                sb[i * d + j] -= mean[i] * mean[j]
        return sb^

    def solve(self, params: LdaParams, components: Int) raises -> LdaModel:
        """The projection at the current state.

        This is `CalculateProjection` with the two missing LAPACK calls put
        back: Cholesky the regularized within-class scatter, reduce the
        between-class scatter by it, symmetric eigensolve, order and
        sign-fix, back-substitute to the generalized eigenvectors, normalize.
        """
        var d = self.dim
        var sw = self.within_scatter(params.reg)
        var sb = self.between_scatter()
        var l = cholesky_lower(sw, d)
        var c = reduce_to_standard(l, d, sb)
        var e = jacobi_eigen(c, d, params.jacobi_max_sweeps)
        sort_eigenpairs(e)

        var proj = List[Float64]()
        for _ in range(components * d):
            proj.append(0.0)
        var col = List[Float64]()
        for _ in range(d):
            col.append(0.0)
        for j in range(components):
            for i in range(d):
                col[i] = e.vectors[i * d + j]
            var x = _solve_lower_transpose_vec(l, d, col, 0)
            # Unit Euclidean norm, so the feature's scale does not depend on
            # the arbitrary scale the solver happened to produce. Then the
            # sign is pinned again: the back-substitution is a linear map and
            # preserves sign, but the normalization is where a caller would
            # expect the canonical form to hold, so it is asserted here.
            var nrm = 0.0
            for i in range(d):
                nrm += x[i] * x[i]
            nrm = sqrt(nrm)
            if nrm > 0.0:
                for i in range(d):
                    x[i] = x[i] / nrm
            canonical_sign(x, 0, d)
            for i in range(d):
                proj[j * d + i] = x[i]
        return LdaModel(d, components, proj^)


def fit_lda(
    embeddings: EmbeddingMatrix,
    class_ids: List[Int],
    n_classes: Int,
    params: LdaParams,
) raises -> LdaModel:
    """The **predict-side** calcer: fitted on the whole learn set, in dataset
    order, and applied to rows that are not in it.

    This is `EstimateFeatureCalcer` (`base_embedding_feature_estimator.h:
    137-151`) and, through it, `MakeFinalFeatureCalcer` --- the object
    serialized into a CatBoost model. **Do not use it to produce training
    features.** Every training row is inside this fit, so the projection has
    seen every training target, and a tree built on the result is fitting a
    leaked target. That is what `lda_online_features` is for.

    `params.catboost_final_flush_only` selects between our behavior and
    CatBoost's; see divergence 6.
    """
    if not params.enabled:
        raise Error("fit_lda: LdaParams.enabled is False")
    if n_classes < 2:
        raise Error(
            "lda: refused for regression and for a single class. CatBoost"
            " permits it, but with one class cloud its between-class scatter"
            " is `1.0 * mu mu^T - mu mu^T`, identically zero, so every"
            " eigenvalue is zero and the projection is an arbitrary"
            " direction carrying no target information"
        )
    if len(class_ids) != embeddings.n_rows:
        raise Error("lda: class id count must equal the row count")
    var components = params.resolved_components(n_classes, embeddings.dim)
    var acc = _LdaAccumulator(embeddings.dim, n_classes)
    var last_solved = 0
    var model = LdaModel.zero(embeddings.dim, components)
    for row in range(embeddings.n_rows):
        acc.add(embeddings.values, embeddings.offset(row), class_ids[row])
        if acc.wants_flush():
            model = acc.solve(params, components)
            acc.last_flush = acc.size
            last_solved = acc.size
    if not params.catboost_final_flush_only:
        if last_solved != acc.size and acc.size > 0:
            model = acc.solve(params, components)
    return model^


def apply_lda(
    model: LdaModel, embeddings: EmbeddingMatrix
) raises -> List[List[Float64]]:
    """Apply a fitted projection to rows the model has not seen. Returns one
    column per component, each `n_rows` long, ready for the binning path."""
    if model.dim != embeddings.dim:
        raise Error("apply_lda: embedding dimension does not match the model")
    var cols = List[List[Float64]]()
    for _ in range(model.components):
        var c = List[Float64]()
        for _ in range(embeddings.n_rows):
            c.append(0.0)
        cols.append(c^)
    var out = List[Float64]()
    for _ in range(model.components):
        out.append(0.0)
    for row in range(embeddings.n_rows):
        model.apply_into(embeddings.values, embeddings.offset(row), out)
        for j in range(model.components):
            cols[j][row] = out[j]
    return cols^


def lda_online_features(
    embeddings: EmbeddingMatrix,
    class_ids: List[Int],
    n_classes: Int,
    permutation: List[Int],
    params: LdaParams,
) raises -> List[List[Float64]]:
    """The **train-side** features: row `i` from the strict prefix before it.

    This is `ComputeOnlineFeatures` (`base_embedding_feature_estimator.h:
    44-82`), whose loop body is Compute-then-Update in that order and no
    other. The row at permutation position 0 projects to zero, because the
    projection matrix has not been solved yet --- that is CatBoost's behavior
    and it is not a bug to fix.

    Serial by construction: each row's value depends on the state after the
    previous row, and the scatter sums are accumulated in ascending
    permutation order, so the float addend order is fixed by this loop and
    identical at every `MOJOTREES_NUM_WORKERS`.
    """
    if not params.enabled:
        raise Error("lda_online_features: LdaParams.enabled is False")
    if n_classes < 2:
        raise Error(
            "lda: refused for regression and for a single class; see fit_lda"
        )
    if len(class_ids) != embeddings.n_rows:
        raise Error("lda: class id count must equal the row count")
    check_permutation(permutation, embeddings.n_rows)

    var components = params.resolved_components(n_classes, embeddings.dim)
    var cols = List[List[Float64]]()
    for _ in range(components):
        var c = List[Float64]()
        for _ in range(embeddings.n_rows):
            c.append(0.0)
        cols.append(c^)

    var acc = _LdaAccumulator(embeddings.dim, n_classes)
    var model = LdaModel.zero(embeddings.dim, components)
    var out = List[Float64]()
    for _ in range(components):
        out.append(0.0)

    for pos in range(len(permutation)):
        var row = permutation[pos]
        # Compute FIRST. The state has seen only the rows before `pos`.
        model.apply_into(embeddings.values, embeddings.offset(row), out)
        for j in range(components):
            cols[j][row] = out[j]
        # Update SECOND.
        acc.add(embeddings.values, embeddings.offset(row), class_ids[row])
        if acc.wants_flush():
            model = acc.solve(params, components)
            acc.last_flush = acc.size
    return cols^


# ---------------------------------------------------------------------------
# KNN
# ---------------------------------------------------------------------------


@always_inline
def _finite_or_inf(x: Float64) -> Float64:
    """A non-finite distance becomes `+inf`.

    A NaN distance would make `<` non-transitive and would let the selected
    neighbourhood depend on comparison order, which is the failure the whole
    tie-breaking rule exists to prevent. Mapping it to `+inf` keeps the order
    total: such a candidate loses to every finite one and is ordered against
    other infinities by its row index.
    """
    if isfinite(x):
        return x
    return POSITIVE_INF


@always_inline
def _cand_less(d1: Float64, i1: Int, d2: Float64, i2: Int) -> Bool:
    """The total order on KNN candidates: `(distance, row index)` ascending.

    Row indices are unique, so this is a strict total order and the selected
    `k` is a set-valued function of the candidate set alone --- independent
    of the order the candidates were examined in, and therefore independent
    of the worker count. That is the guarantee the brief asks for, and it is
    a property of the comparator rather than of the loop.
    """
    if d1 != d2:
        return d1 < d2
    return i1 < i2


def _squared_l2(
    a: List[Float64], a_off: Int, b: List[Float64], b_off: Int, d: Int
) -> Float64:
    """`NHnsw::TL2SqrDistance<float>` (`knn.h:17, 48-61`). Squared, no root:
    the root is monotone so it changes no ordering, and CatBoost does not
    take it either. No normalization, which does matter --- this is not
    cosine similarity, and an unnormalized embedding gets a
    magnitude-sensitive neighbourhood."""
    var s = 0.0
    for j in range(d):
        var t = a[a_off + j] - b[b_off + j]
        s += t * t
    return s


def _knn_query(
    embeddings: EmbeddingMatrix,
    query_row: Int,
    candidates: List[Int],
    n_candidates: Int,
    k: Int,
    mut best_d: List[Float64],
    mut best_i: List[Int],
) -> Int:
    """The `min(k, n_candidates)` nearest candidates, under `_cand_less`.

    `candidates[0 : n_candidates]` are row indices; for the online pass they
    are the permutation prefix strictly before the query's own position,
    which is how the query is excluded from its own neighbourhood. Returns
    how many were found; fewer than `k` only when there were fewer
    candidates, which is CatBoost's empty-and-short-prefix behavior.

    A sorted insert into a `k`-long array, `O(n_candidates * (d + k))`. The
    array is kept in ascending `_cand_less` order, so the worst element is
    always at the end and the early-out is one comparison.
    """
    var d = embeddings.dim
    var q_off = embeddings.offset(query_row)
    var count = 0
    for c in range(n_candidates):
        var row = candidates[c]
        var dist = _finite_or_inf(
            _squared_l2(
                embeddings.values,
                q_off,
                embeddings.values,
                embeddings.offset(row),
                d,
            )
        )
        if count == k and not _cand_less(
            dist, row, best_d[k - 1], best_i[k - 1]
        ):
            continue
        var pos = count
        if pos == k:
            pos = k - 1
        else:
            count += 1
        while pos > 0 and _cand_less(dist, row, best_d[pos - 1], best_i[pos - 1]):
            best_d[pos] = best_d[pos - 1]
            best_i[pos] = best_i[pos - 1]
            pos -= 1
        best_d[pos] = dist
        best_i[pos] = row
    return count


struct KnnModel(Copyable, Movable):
    """A fitted cloud: every learn row's embedding and target, plus `k`.

    CatBoost's fitted state is an HNSW graph and ours is the point set, which
    is the whole of divergence 4: exact and reproducible up to a bounded row
    count, rather than approximate and irreproducible at any row count.
    """

    var dim: Int
    var k: Int
    var n_classes: Int
    """0 means regression: one feature, the neighbour target mean."""
    var n_points: Int
    var points: List[Float64]
    var targets: List[Float64]

    def __init__(
        out self,
        dim: Int,
        k: Int,
        n_classes: Int,
        n_points: Int,
        var points: List[Float64],
        var targets: List[Float64],
    ) raises:
        if len(points) != dim * n_points:
            raise Error("KnnModel: points must be n_points * dim")
        if len(targets) != n_points:
            raise Error("KnnModel: one target per point")
        self.dim = dim
        self.k = k
        self.n_classes = n_classes
        self.n_points = n_points
        self.points = points^
        self.targets = targets^

    def feature_count(self) -> Int:
        """`embedding_feature_estimators.cpp:91-95`: `num_classes` for
        classification, 1 for regression."""
        if self.n_classes > 0:
            return self.n_classes
        return 1


def knn_feature_count(n_classes: Int) -> Int:
    """`num_classes` features for classification, one for regression."""
    if n_classes > 0:
        return n_classes
    return 1


def _knn_emit(
    n_classes: Int,
    best_i: List[Int],
    found: Int,
    targets: List[Float64],
    class_ids: List[Int],
    mut out: List[Float64],
):
    """`TKNNCalcer::Compute`'s two branches (`knn.cpp:30-52`).

    Classification: raw **counts** per class. Not normalized and not
    distance-weighted, so with `k = 5` these are integers in `[0, 5]` living
    in a float column --- a low-cardinality numeric feature, which the
    binning path's one-bin-per-distinct-value rule quantizes exactly.

    Regression: the unweighted mean, guarded to 0.0 on an empty
    neighbourhood, which is CatBoost's explicit `if (neighbors.size())`.
    """
    if n_classes > 0:
        for c in range(n_classes):
            out[c] = 0.0
        for j in range(found):
            out[class_ids[best_i[j]]] += 1.0
    else:
        out[0] = 0.0
        if found > 0:
            var s = 0.0
            for j in range(found):
                s += targets[best_i[j]]
            out[0] = s / Float64(found)


def _check_knn_size(params: KnnParams, n_rows: Int) raises:
    if not params.enabled:
        raise Error("knn: KnnParams.enabled is False")
    if params.k < 1:
        raise Error("knn: k must be at least 1")
    if n_rows > params.max_rows:
        raise Error(
            "knn: refused above `max_rows`. This is an exact brute-force"
            " neighbour search, so the online pass is Theta(n^2 d / 2);"
            " CatBoost avoids that with an approximate online HNSW index"
            " that mojotrees does not have. Raise `max_rows` only if you"
            " have decided the quadratic cost is acceptable"
        )


def fit_knn(
    embeddings: EmbeddingMatrix,
    targets: List[Float64],
    n_classes: Int,
    params: KnnParams,
) raises -> KnnModel:
    """The **predict-side** cloud: every learn row.

    `TKNNCalcer::Compute` against a full-data calcer, which is what test rows
    and a deployed model see. Query rows are not in this cloud, so the
    self-exclusion the online pass gets for free is not needed and is not
    performed --- **which is exactly why this must not be used on training
    rows.** Query a training row against a cloud containing itself and it is
    its own nearest neighbour at distance zero, and the feature becomes the
    row's own target.
    """
    _check_knn_size(params, embeddings.n_rows)
    if len(targets) != embeddings.n_rows:
        raise Error("knn: target count must equal the row count")
    return KnnModel(
        embeddings.dim,
        params.k,
        n_classes,
        embeddings.n_rows,
        embeddings.values.copy(),
        targets.copy(),
    )


def apply_knn(
    model: KnnModel, embeddings: EmbeddingMatrix
) raises -> List[List[Float64]]:
    """Apply a fitted cloud to rows that are not in it."""
    if model.dim != embeddings.dim:
        raise Error("apply_knn: embedding dimension does not match the model")
    var n_features = model.feature_count()
    var class_ids = List[Int]()
    if model.n_classes > 0:
        class_ids = class_ids_from_targets(model.targets, model.n_classes)
    var cloud = EmbeddingMatrix(
        model.n_points, model.dim, model.points.copy()
    )
    var all_rows = identity_permutation(model.n_points)

    var cols = List[List[Float64]]()
    for _ in range(n_features):
        var c = List[Float64]()
        for _ in range(embeddings.n_rows):
            c.append(0.0)
        cols.append(c^)

    var best_d = List[Float64]()
    var best_i = List[Int]()
    for _ in range(model.k):
        best_d.append(0.0)
        best_i.append(0)
    var out = List[Float64]()
    for _ in range(n_features):
        out.append(0.0)

    # The query rows and the cloud are different point sets, so the query is
    # scored against a temporary that concatenates neither: the distance is
    # computed from the query matrix and the cloud matrix separately, which
    # is why the cloud is rebuilt as an `EmbeddingMatrix` above.
    for row in range(embeddings.n_rows):
        var count = 0
        var k = model.k
        for c in range(model.n_points):
            var dist = _finite_or_inf(
                _squared_l2(
                    embeddings.values,
                    embeddings.offset(row),
                    cloud.values,
                    cloud.offset(all_rows[c]),
                    model.dim,
                )
            )
            var cand = all_rows[c]
            if count == k and not _cand_less(
                dist, cand, best_d[k - 1], best_i[k - 1]
            ):
                continue
            var pos = count
            if pos == k:
                pos = k - 1
            else:
                count += 1
            while pos > 0 and _cand_less(
                dist, cand, best_d[pos - 1], best_i[pos - 1]
            ):
                best_d[pos] = best_d[pos - 1]
                best_i[pos] = best_i[pos - 1]
                pos -= 1
            best_d[pos] = dist
            best_i[pos] = cand
        _knn_emit(
            model.n_classes, best_i, count, model.targets, class_ids, out
        )
        for j in range(n_features):
            cols[j][row] = out[j]
    return cols^


def knn_online_features(
    embeddings: EmbeddingMatrix,
    targets: List[Float64],
    n_classes: Int,
    permutation: List[Int],
    params: KnnParams,
) raises -> List[List[Float64]]:
    """The **train-side** features: row `i`'s neighbours are drawn from the
    strict prefix before it in the permutation.

    That is the self-exclusion, and it is the same one CatBoost gets: the
    query point is not in the candidate set because it has not been added
    yet. The row at permutation position 0 has an empty candidate set and
    gets all-zero class counts, or a regression mean of 0.0 under CatBoost's
    `if (neighbors.size())` guard.

    Parallel over **query positions**, not over candidates. Position `pos`
    scans `permutation[0 : pos]` serially in ascending order and writes only
    its own output slots, so no two workers touch the same storage and no
    reduction is split. Combined with the total order in `_cand_less`, the
    result is identical at every `MOJOTREES_NUM_WORKERS` and on every
    machine.

    The fan-out is over equal-length position blocks while the work at
    position `pos` is proportional to `pos`, so the blocks are imbalanced by
    construction. That is a scheduling cost, not a correctness one, and it is
    left alone rather than fixed with a work-stealing scheme whose row-to-
    worker assignment would be nondeterministic.
    """
    _check_knn_size(params, embeddings.n_rows)
    if len(targets) != embeddings.n_rows:
        raise Error("knn: target count must equal the row count")
    check_permutation(permutation, embeddings.n_rows)

    var class_ids = List[Int]()
    if n_classes > 0:
        class_ids = class_ids_from_targets(targets, n_classes)
    var n_features = knn_feature_count(n_classes)
    var n_rows = embeddings.n_rows
    var k = params.k

    var flat = List[Float64]()
    for _ in range(n_features * n_rows):
        flat.append(0.0)

    var flat_p = flat.unsafe_ptr()

    def scan_positions(start: Int, end: Int) {imm}:
        # Per-worker scratch, so nothing is shared but `flat`, and the slots
        # of `flat` a worker writes are indexed by rows only that worker
        # visits.
        var best_d = List[Float64]()
        var best_i = List[Int]()
        for _ in range(k):
            best_d.append(0.0)
            best_i.append(0)
        var out = List[Float64]()
        for _ in range(n_features):
            out.append(0.0)
        for pos in range(start, end):
            var row = permutation[pos]
            var found = _knn_query(
                embeddings, row, permutation, pos, k, best_d, best_i
            )
            _knn_emit(n_classes, best_i, found, targets, class_ids, out)
            for j in range(n_features):
                flat_p.unsafe_store(j * n_rows + row, out[j])

    # Work estimate in the usual op-equivalents: the union of all prefixes is
    # n(n-1)/2 candidate visits of `dim` multiply-adds each.
    var ops = (n_rows * (n_rows - 1) // 2) * embeddings.dim
    dispatch_rows(scan_positions, n_rows, ops)

    var cols = List[List[Float64]]()
    for j in range(n_features):
        var c = List[Float64]()
        for r in range(n_rows):
            c.append(flat[j * n_rows + r])
        cols.append(c^)
    return cols^


# ---------------------------------------------------------------------------
# Both estimators together
# ---------------------------------------------------------------------------


def compute_online_features(
    embeddings: EmbeddingMatrix,
    targets: List[Float64],
    n_classes: Int,
    permutation: List[Int],
    params: EmbeddingEstimatorParams,
) raises -> List[List[Float64]]:
    """Every enabled estimator's train-side columns for one embedding column,
    LDA's first and KNN's after, concatenated.

    CatBoost's `DefaultEmbeddingCalcers` runs both, in that order, for every
    embedding column. Here neither runs unless asked. The columns are
    ordinary floats from this point on and go straight into the existing
    binning path; nothing downstream knows an embedding feature from any
    other float feature, which is CatBoost's design and the reason this lane
    wrote no trainer.
    """
    var cols = List[List[Float64]]()
    if params.lda.enabled:
        var class_ids = class_ids_from_targets(targets, n_classes)
        var lda_cols = lda_online_features(
            embeddings, class_ids, n_classes, permutation, params.lda
        )
        for j in range(len(lda_cols)):
            cols.append(lda_cols[j].copy())
    if params.knn.enabled:
        var knn_cols = knn_online_features(
            embeddings, targets, n_classes, permutation, params.knn
        )
        for j in range(len(knn_cols)):
            cols.append(knn_cols[j].copy())
    if len(cols) == 0:
        raise Error(
            "compute_online_features: no estimator is enabled; this refuses"
            " rather than returning zero columns"
        )
    return cols^


def online_feature_count(
    n_classes: Int, dim: Int, params: EmbeddingEstimatorParams
) raises -> Int:
    """How many columns `compute_online_features` will produce, without
    producing them --- so a caller can size a feature matrix before paying
    for the pass."""
    var n = 0
    if params.lda.enabled:
        n += params.lda.resolved_components(n_classes, dim)
    if params.knn.enabled:
        n += knn_feature_count(n_classes)
    return n
