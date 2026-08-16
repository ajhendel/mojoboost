"""A binned training set that outlives the model trained on it.

`Dataset` is the Mojo side of LightGBM's dataset object: a feature matrix
that has already been binned, together with the label, weights, query
groups, and init scores that belong to it. Binning is the expensive part of
starting a run, so a `Dataset` is constructed once and then trained on as
many times as the caller likes, with `train_dataset` and its multiclass and
ranking counterparts, and extended with `update_dataset`.

Dense and sparse
----------------
A dataset owns one binned matrix, and which kind it is was decided once, at
construction, from the representation of the input:

- `Dataset(features, n_rows, n_features, ...)` takes a borrowed dense
  column-major matrix and holds a `BinnedMatrix`.
- `Dataset.from_csc` / `Dataset.from_csr` / `Dataset.from_raw` take a
  `raw_data.RawData` and hold a `SparseBinnedMatrix` when it is sparse.

`is_sparse` says which. Every training entry point below dispatches on it, so
a caller trains a sparse dataset with the same `train_dataset` call it trains
a dense one with, and the paths that have no sparse implementation yet (the
GPU trainer, LambdaRank, continued training) say so instead of densifying
behind the caller's back. Nothing here converts one representation into the
other in either direction.

The dense constructor is the one exception to `RawData` owning ingestion, and
deliberately: it is handed a matrix it borrows, and materializing a `RawData`
around it would copy `n_rows * n_features` floats for nothing. It bins that
matrix with the same `binning.fit_bins` that `RawData.fit_mapper` dispatches
to, so the two agree by construction. Ask for `keep_raw=True` and it does
build one, because a retained matrix has to be owned.

What a `Dataset` owns

- the fitted `BinMapper` and the matrix it produced, which is the training
  data itself
- the raw input, but only when it was built with `keep_raw=True`. That is
  what `subset` needs, and what lets one dataset be re-binned over part of
  its rows without the caller holding the matrix a second time; without it
  the raw matrix is dropped after binning, as it always was
- the label, sample weights, per-query row counts, and init scores, each
  optional and each validated against the row count on construction
- the feature names and the categorical feature declaration, which are
  metadata a model file does not carry

Reference binning and leakage
-----------------------------
Bin edges are fitted from data, so a matrix binned over rows that a model
will later be scored on has already leaked. Two constructions keep the two
cases apart, and they are named for which one they are:

- `Dataset.subset(rows)` and `Dataset.from_raw` **fit** bins over the rows
  they were given and nothing else. This is what a cross-validation fold or
  a held-out split wants: the rows that were left out had no say in the
  edges.
- `Dataset.from_reference(reference, ...)` and
  `Dataset.subset_shared_binning(rows)` **reuse** a reference's fitted
  mapper. This is what a validation set that will be scored by a model
  trained on the reference wants, and what continued training requires,
  because a bin index has to mean the same thing to the trees already grown
  and the trees about to be. Using it for a fold is the leak.

What a `Dataset` deliberately does not offer is LightGBM's family of
post-construction mutators (`set_label`, `set_field`,
`set_categorical_feature`, and the rest). Bin edges are fitted from the data
and the categorical declaration, so changing either afterwards would leave
the binned matrix describing data the dataset no longer holds. Fields are
supplied at construction; change one by constructing another dataset.

A prepared table serializes, and separately from a model.
`serialize.save_dataset` / `serialize.load_dataset` write and read the
binning and the columns as their own file kind, with their own magic and
their own version, so no loader can mistake one for the other. It lives
there rather than here because the mapper reader and writer do, and one
mapper codec is the point. `Dataset.from_binned_dense` and
`from_binned_sparse` are the entry points it reads back through; a table
read from a file carries no raw matrix, so it cannot be `subset`.
"""

from std.math import isinf
from std.memory import unsafe_memcpy

from .bagging import BaggingParams
from .binning import BinMapper, BinnedMatrix, fit_bins
from .boosting import (
    Booster,
    BoosterParams,
    train,
    train_more,
    train_multiclass,
    train_multiclass_more,
)
from .boosting_sparse import train_multiclass_sparse, train_sparse
from .device import CPU_DEVICE, GPU_DEVICE, resolve_device
from .goss import GossParams
from .model import Model, MulticlassModel
from .parallel import plan_row_blocks, run_row_blocks
from .ranking import RankerParams, groups_from_counts, train_ranker
from .ranking_advanced import (
    AdvancedRankParams,
    PositionMap,
    advanced_ranking_requested,
    train_ranker_advanced,
)
from .raw_data import RawData
from .sparse import CscMatrix, CsrMatrix, SparseBinnedMatrix, transform_csc
from .train_gpu import train_gpu, train_multiclass_gpu
from .validation import (
    check_categorical_features,
    check_class_codes,
    check_column_length,
    check_group_counts,
    check_relevance_labels,
    check_required_length,
    check_shape,
)


comptime INGEST_TILE_BYTES = 32 * 1024
"""Source bytes one ingest tile holds while the feature loop walks it.

`transpose_to_column_major` reads the caller's row-major matrix and writes a
column-major one, so exactly one of the two sides can be sequential. The
write is chosen, because a column of the destination is `n_rows` contiguous
doubles and a whole column-run is one stream; the read is then strided by
`n_features * 8` bytes and would touch a fresh cache line per element if it
ran the full height of the matrix. Tiling the rows bounds that: a tile of
`tile_rows * n_features * 8` source bytes is fetched once and then read
`n_features` times from cache, once per column pass.

32 KiB, and the same number and the same reasoning as
`binning.ROW_MAJOR_TILE_BYTES`, which tiles the mirror-image transpose of the
binned matrix: a conservative floor for a private L1, on the grounds
`apple_cpu_policy.ASSUMED_L1D_BYTES` states. **Untuned. Nothing here measured
it**, and the only claim being made for it is the one arithmetic supports,
that a tiled read of a tile that fits in L1 fetches each source line once
instead of once per feature.
"""


def transpose_to_column_major[
    src_origin: ImmOrigin, dst_origin: MutOrigin, //
](
    src: Span[Float64, src_origin],
    dst: Span[Float64, dst_origin],
    n_rows: Int,
    n_features: Int,
) raises -> Bool:
    """Write `src[r * n_features + f]` into `dst[f * n_rows + r]`, and say
    whether any value was infinite.

    This is the whole of ingestion for a caller holding a C-ordered matrix,
    which is what NumPy hands out by default and therefore what nearly every
    `fit(X, y)` starts from. `binning.fit_bins` reads
    `features[f * n_rows + r]`, so somebody has to do this; the point of
    doing it here is that it is **one** pass rather than the two the NumPy
    side took (`asfortranarray`, then `isinf(...).any()` over the result),
    and that the pass is parallel and tiled rather than serial and strided.

    Returning the infinity flag from the same pass is not a convenience. The
    caller has to reject `+inf` and `-inf` before binning (`NaN` is the
    missing-value marker and is allowed; see `binning.mojo`), and asking that
    question afterwards means a second full read of `n_rows * n_features`
    doubles plus a bool array of `n_rows * n_features` bytes to reduce. Asked
    here, it is one compare on a value already in a register.

    Determinism: the row blocks are disjoint and each destination slot is
    written by exactly one task, so nothing is reassociated and no value is
    combined with any other. The result is byte-identical at every
    `MOJOTREES_NUM_WORKERS`, and it is the same bytes the caller passed in --
    a transpose moves values between slots and does no arithmetic on them.
    The infinity flag is an OR over per-block flags, which is
    order-independent for the same reason.
    """
    if n_rows < 1 or n_features < 1:
        raise Error("matrix must have positive dimensions")
    if len(src) != n_rows * n_features:
        raise Error("source length must equal n_rows * n_features")
    if len(dst) != n_rows * n_features:
        raise Error("destination length must equal n_rows * n_features")

    var src_p = src.unsafe_ptr()
    var dst_p = dst.unsafe_ptr()
    var tile = INGEST_TILE_BYTES // (n_features * 8)
    if tile < 1:
        tile = 1

    # The block plan is taken explicitly rather than through `dispatch_rows`
    # because each block needs a slot of its own to record whether it saw an
    # infinity, and a block id is what indexes it. Same geometry either way.
    var blocks = plan_row_blocks(n_rows, n_rows * n_features)
    var saw = List[Int](capacity=blocks.n_blocks)
    saw.resize(blocks.n_blocks, 0)
    var saw_p = saw.unsafe_ptr()

    def fill(b: Int) {imm}:
        var start = blocks.start(b)
        var end = blocks.end(b)
        var bad = 0
        var t0 = start
        while t0 < end:
            var t1 = t0 + tile
            if t1 > end:
                t1 = end
            for f in range(n_features):
                var col = f * n_rows
                for r in range(t0, t1):
                    var v = src_p.unsafe_load(r * n_features + f)
                    dst_p.unsafe_store(col + r, v)
                    if isinf(v):
                        bad = 1
            t0 = t1
        if bad != 0:
            saw_p.unsafe_store(b, 1)

    run_row_blocks(blocks, fill)

    var any_inf = False
    for b in range(blocks.n_blocks):
        if saw[b] != 0:
            any_inf = True
    return any_inf


def to_column_major[
    src_origin: ImmOrigin, //
](
    src: Span[Float64, src_origin], n_rows: Int, n_features: Int
) raises -> List[Float64]:
    """`transpose_to_column_major` into a freshly allocated buffer.

    For a caller that owns both sides, a benchmark or a test. The Python
    boundary does not come this way: there the destination is the NumPy
    array the wrapper must hand back, so the binding transposes into a buffer
    the caller already allocated and nothing here allocates a third copy of
    the matrix.

    The infinity flag is dropped; a caller who has to reject infinities
    should call `transpose_to_column_major` and read it.
    """
    var out = List[Float64](unsafe_uninit_length=n_rows * n_features)
    _ = transpose_to_column_major(src, out, n_rows, n_features)
    return out^


def has_infinite[
    values_origin: ImmOrigin, //
](values: Span[Float64, values_origin]) raises -> Bool:
    """True when any value is `+inf` or `-inf`. `NaN` is not infinite.

    The check on its own, for the one input shape that needs no transpose:
    a caller whose matrix is already column-major, whose buffer is therefore
    handed to the binner as it lies. It is a parallel read with no
    allocation, which is what the NumPy expression it replaces
    (`np.isinf(Xa).any()`) is not: that one materializes an
    `n_rows * n_features` byte array before reducing it.
    """
    var n = len(values)
    if n == 0:
        return False
    var p = values.unsafe_ptr()
    var blocks = plan_row_blocks(n, n)
    var saw = List[Int](capacity=blocks.n_blocks)
    saw.resize(blocks.n_blocks, 0)
    var saw_p = saw.unsafe_ptr()

    def scan(b: Int) {imm}:
        var bad = 0
        for i in range(blocks.start(b), blocks.end(b)):
            if isinf(p.unsafe_load(i)):
                bad = 1
        if bad != 0:
            saw_p.unsafe_store(b, 1)

    run_row_blocks(blocks, scan)

    for b in range(blocks.n_blocks):
        if saw[b] != 0:
            return True
    return False


def _check_labels(label: List[Float64], n_rows: Int) raises:
    """A trainer needs one label per row. The rule is `validation`'s; this
    wrapper only names the column."""
    check_required_length(len(label), n_rows, "label")


def _int_labels(label: List[Float64], n_classes: Int) raises -> List[Int]:
    """Class codes from a float64 label column: `validation.check_class_codes`
    (whole numbers in `[0, n_classes)`, at least two classes)."""
    return check_class_codes(label, n_classes)


def _relevance_labels(label: List[Float64]) raises -> List[Int]:
    """Graded relevances from a float64 label column:
    `validation.check_relevance_labels` (whole numbers in
    `[0, MAX_RELEVANCE]`, the same range `ranking.train_ranker` enforces)."""
    return check_relevance_labels(label)


def _check_columns(
    n_rows: Int,
    n_features: Int,
    label: List[Float64],
    weight: List[Float64],
    group: List[Int],
    init_score: List[Float64],
    feature_names: List[String],
    categorical_features: List[Int],
) raises:
    """Every column a dataset can carry, checked against the shape it was
    built with.

    This runs at construction rather than at train time, so a mismatch is
    reported while the caller still knows which array it passed. An empty
    column means the dataset has none. One implementation, used by every
    constructor, so a dataset built from a CSC matrix rejects exactly what a
    dataset built from a dense one rejects. Each rule is `validation.mojo`'s;
    only the feature-name count stays here, because `validation` takes
    primitives and this is the one column that is a list of strings.
    """
    check_shape(n_rows, n_features)
    check_column_length(len(label), n_rows, "label")
    check_column_length(len(weight), n_rows, "weight")
    check_column_length(len(init_score), n_rows, "init_score")
    if len(feature_names) != 0 and len(feature_names) != n_features:
        raise Error(
            "feature_name must have one name per feature: got ",
            len(feature_names),
            " for ",
            n_features,
            " features",
        )
    if len(group) != 0:
        _ = check_group_counts(group, n_rows)
    check_categorical_features(categorical_features, n_features)


def _empty_binned(n_features: Int) -> BinnedMatrix:
    """The dense matrix field of a sparse dataset: structurally valid, no
    rows. A struct field cannot be absent, and `is_sparse` says which of the
    two matrices is the live one."""
    return BinnedMatrix(List[UInt8](), 0, n_features, 0)


def _empty_sparse_binned(n_features: Int) -> SparseBinnedMatrix:
    """The sparse matrix field of a dense dataset. As `_empty_binned`."""
    var offsets = List[Int](capacity=n_features + 1)
    offsets.resize(n_features + 1, 0)
    var defaults = List[UInt8](capacity=n_features)
    defaults.resize(n_features, 0)
    return SparseBinnedMatrix(
        List[Int](), List[UInt8](), offsets^, defaults^, 0, n_features, 0
    )


def _subset_column(column: List[Float64], rows: List[Int]) -> List[Float64]:
    """One of a dataset's per-row columns restricted to `rows`, empty for a
    column the dataset does not have."""
    if len(column) == 0:
        return List[Float64]()
    var out = List[Float64](capacity=len(rows))
    for i in range(len(rows)):
        out.append(column[rows[i]])
    return out^


def _subset_group(
    group: List[Int], rows: List[Int], n_rows: Int
) raises -> List[Int]:
    """The per-query row counts of an ascending row selection.

    A query is an atom: its rows belong to one side of a split together, or
    the model learns the ordering of a query it is then scored on. So a
    selection that takes part of a query is rejected rather than repaired,
    and the counts that come back describe whole queries in query order,
    which is what `group` means everywhere else.
    """
    if len(group) == 0:
        return List[Int]()
    var owner = List[Int](capacity=n_rows)
    for q in range(len(group)):
        for _ in range(group[q]):
            owner.append(q)
    var counts = List[Int]()
    var queries = List[Int]()
    var current = -1
    for i in range(len(rows)):
        var q = owner[rows[i]]
        if q != current:
            counts.append(0)
            queries.append(q)
            current = q
        counts[len(counts) - 1] += 1
    for i in range(len(counts)):
        if counts[i] != group[queries[i]]:
            raise Error(
                "a subset of a ranking dataset takes whole queries: this one"
                " takes part of a query, whose remaining rows would be"
                " scored by a model that learned its ordering"
            )
    return counts^


struct Dataset(Copyable, Movable, Writable):
    """A binned feature matrix, dense or sparse, and the columns that go
    with it."""

    var mapper: BinMapper
    var data: BinnedMatrix
    var sparse_data: SparseBinnedMatrix
    var is_sparse: Bool
    var raw: RawData
    var borrowed_binning: Bool
    var n_rows: Int
    var n_features: Int
    var label: List[Float64]
    var weight: List[Float64]
    var group: List[Int]
    var init_score: List[Float64]
    var feature_names: List[String]
    var categorical_features: List[Int]
    var max_bin: Int
    var use_missing: Bool

    def __init__[
        features_origin: ImmOrigin, //
    ](
        out self,
        features: Span[Float64, features_origin],
        n_rows: Int,
        n_features: Int,
        var label: List[Float64] = [],
        var weight: List[Float64] = [],
        var group: List[Int] = [],
        var init_score: List[Float64] = [],
        var feature_names: List[String] = [],
        var categorical_features: List[Int] = [],
        max_bin: Int = 255,
        use_missing: Bool = True,
        keep_raw: Bool = False,
    ) raises:
        """Bin a column-major raw feature matrix (`features[f * n_rows + r]`)
        and take ownership of the columns that describe its rows.

        The matrix is borrowed, and binned where it lies: nothing here copies
        it. `keep_raw=True` retains a copy so the dataset can later be
        re-binned over part of its rows (`subset`), which is the one thing
        that needs the raw values after construction; it costs one extra
        `n_rows * n_features` buffer for as long as the dataset lives.
        """
        _check_columns(
            n_rows,
            n_features,
            label,
            weight,
            group,
            init_score,
            feature_names,
            categorical_features,
        )
        if len(features) != n_rows * n_features:
            raise Error("features length must equal n_rows * n_features")

        self.mapper = fit_bins(
            features,
            n_rows,
            n_features,
            max_bin,
            use_missing=use_missing,
            categorical_features=categorical_features,
        )
        self.data = self.mapper.transform(features, n_rows)
        self.sparse_data = _empty_sparse_binned(n_features)
        self.is_sparse = False
        if keep_raw:
            # The only copy of the raw matrix left on this path, and the one
            # that cannot be avoided: `keep_raw` is the caller asking the
            # dataset to outlive their buffer, so the dataset has to own it.
            # Binning above read the borrowed view directly.
            var owned = List[Float64](unsafe_uninit_length=len(features))
            unsafe_memcpy(
                dest=owned.unsafe_ptr(),
                src=features.unsafe_ptr(),
                count=len(features),
            )
            self.raw = RawData.dense(owned^, n_rows, n_features)
        else:
            self.raw = RawData.none()
        self.borrowed_binning = False
        self.n_rows = n_rows
        self.n_features = n_features
        self.label = label^
        self.weight = weight^
        self.group = group^
        self.init_score = init_score^
        self.feature_names = feature_names^
        self.categorical_features = categorical_features^
        self.max_bin = max_bin
        self.use_missing = use_missing

    def __init__(
        out self,
        var mapper: BinMapper,
        var data: BinnedMatrix,
        var sparse_data: SparseBinnedMatrix,
        is_sparse: Bool,
        var raw: RawData,
        borrowed_binning: Bool,
        n_rows: Int,
        n_features: Int,
        var label: List[Float64],
        var weight: List[Float64],
        var group: List[Int],
        var init_score: List[Float64],
        var feature_names: List[String],
        var categorical_features: List[Int],
        max_bin: Int,
        use_missing: Bool,
    ):
        """Assemble a dataset from a binning that has already happened.

        Internal: the static constructors below bin their input and then come
        here, so that every dataset, however it was built, is assembled in
        one place. Nothing is validated here, because everything was
        validated on the way in.
        """
        self.mapper = mapper^
        self.data = data^
        self.sparse_data = sparse_data^
        self.is_sparse = is_sparse
        self.raw = raw^
        self.borrowed_binning = borrowed_binning
        self.n_rows = n_rows
        self.n_features = n_features
        self.label = label^
        self.weight = weight^
        self.group = group^
        self.init_score = init_score^
        self.feature_names = feature_names^
        self.categorical_features = categorical_features^
        self.max_bin = max_bin
        self.use_missing = use_missing

    @staticmethod
    def from_raw(
        var raw: RawData,
        var label: List[Float64] = [],
        var weight: List[Float64] = [],
        var group: List[Int] = [],
        var init_score: List[Float64] = [],
        var feature_names: List[String] = [],
        var categorical_features: List[Int] = [],
        max_bin: Int = 255,
        use_missing: Bool = True,
        keep_raw: Bool = False,
    ) raises -> Dataset:
        """Bin a `RawData`, dense or sparse, and own the result.

        The general constructor: `from_csc`, `from_csr`, and `subset` all
        arrive here, and so does any caller that already holds a `RawData`.
        The bins are **fitted on these rows**, so this is the constructor a
        fold or a held-out split wants; `from_reference` is the one that
        reuses someone else's binning.

        `raw` is taken by value and binned in place, so a dense caller pays
        no copy for coming this way rather than through the dense
        constructor. It is retained afterwards only when `keep_raw` asks for
        it, and dropped otherwise.
        """
        var n_rows = raw.n_rows
        var n_features = raw.n_features
        _check_columns(
            n_rows,
            n_features,
            label,
            weight,
            group,
            init_score,
            feature_names,
            categorical_features,
        )
        var mapper = raw.fit_mapper(
            max_bin, categorical_features, use_missing
        )
        var is_sparse = raw.is_sparse
        var data = _empty_binned(n_features)
        var sparse_data = _empty_sparse_binned(n_features)
        if is_sparse:
            sparse_data = raw.transform_sparse(mapper)
        else:
            data = raw.transform_dense(mapper)
        var kept: RawData
        if keep_raw:
            kept = raw^
        else:
            kept = RawData.none()
        return Dataset(
            mapper^,
            data^,
            sparse_data^,
            is_sparse,
            kept^,
            False,
            n_rows,
            n_features,
            label^,
            weight^,
            group^,
            init_score^,
            feature_names^,
            categorical_features^,
            max_bin,
            use_missing,
        )

    @staticmethod
    def from_csc(
        var csc: CscMatrix,
        var label: List[Float64] = [],
        var weight: List[Float64] = [],
        var group: List[Int] = [],
        var init_score: List[Float64] = [],
        var feature_names: List[String] = [],
        var categorical_features: List[Int] = [],
        max_bin: Int = 255,
        use_missing: Bool = True,
        keep_raw: Bool = False,
    ) raises -> Dataset:
        """A dataset over a sparse CSC matrix, which stays sparse: the
        binned matrix is a `SparseBinnedMatrix` and training never allocates
        `n_rows * n_features` of anything."""
        return Dataset.from_raw(
            RawData.from_csc(csc^),
            label^,
            weight^,
            group^,
            init_score^,
            feature_names^,
            categorical_features^,
            max_bin,
            use_missing,
            keep_raw,
        )

    @staticmethod
    def from_csr(
        csr: CsrMatrix,
        var label: List[Float64] = [],
        var weight: List[Float64] = [],
        var group: List[Int] = [],
        var init_score: List[Float64] = [],
        var feature_names: List[String] = [],
        var categorical_features: List[Int] = [],
        max_bin: Int = 255,
        use_missing: Bool = True,
        keep_raw: Bool = False,
    ) raises -> Dataset:
        """A dataset over a sparse CSR matrix, transposed once to the
        feature-oriented layout the histogram builders read."""
        return Dataset.from_raw(
            RawData.from_csr(csr),
            label^,
            weight^,
            group^,
            init_score^,
            feature_names^,
            categorical_features^,
            max_bin,
            use_missing,
            keep_raw,
        )

    @staticmethod
    def from_binned_dense(
        var mapper: BinMapper,
        var data: BinnedMatrix,
        var label: List[Float64] = [],
        var weight: List[Float64] = [],
        var group: List[Int] = [],
        var init_score: List[Float64] = [],
        var feature_names: List[String] = [],
        var categorical_features: List[Int] = [],
        max_bin: Int = 255,
        use_missing: Bool = True,
        borrowed_binning: Bool = False,
    ) raises -> Dataset:
        """A dataset over a matrix that has *already* been binned.

        For a caller holding a mapper and the matrix it produced rather than
        raw values: `serialize.load_dataset` reads a prepared table this way,
        and a shard that was binned elsewhere arrives this way. The mapper
        and the matrix are checked against each other, so a file or a
        transport that lost a byte is refused here rather than producing bin
        indices that mean nothing.

        The result retains no raw matrix, so it cannot be `subset`. Bins
        cannot be refitted from bins; that is the whole reason `subset` needs
        `keep_raw`.
        """
        if data.n_features != mapper.n_features:
            raise Error(
                "a binned matrix and its mapper must agree on the feature"
                " count"
            )
        if data.n_bins != mapper.n_bins:
            raise Error("a binned matrix and its mapper must agree on n_bins")
        if len(data.bins) != data.n_rows * data.n_features:
            raise Error("binned matrix length must equal n_rows * n_features")
        var n_rows = data.n_rows
        var n_features = data.n_features
        _check_columns(
            n_rows,
            n_features,
            label,
            weight,
            group,
            init_score,
            feature_names,
            categorical_features,
        )
        return Dataset(
            mapper^,
            data^,
            _empty_sparse_binned(n_features),
            False,
            RawData.none(),
            borrowed_binning,
            n_rows,
            n_features,
            label^,
            weight^,
            group^,
            init_score^,
            feature_names^,
            categorical_features^,
            max_bin,
            use_missing,
        )

    @staticmethod
    def from_binned_sparse(
        var mapper: BinMapper,
        var sparse_data: SparseBinnedMatrix,
        var label: List[Float64] = [],
        var weight: List[Float64] = [],
        var group: List[Int] = [],
        var init_score: List[Float64] = [],
        var feature_names: List[String] = [],
        var categorical_features: List[Int] = [],
        max_bin: Int = 255,
        use_missing: Bool = True,
        borrowed_binning: Bool = False,
    ) raises -> Dataset:
        """The sparse counterpart of `from_binned_dense`, with the same
        checks: the mapper and the matrix must agree, and the offsets must
        describe the features they claim to."""
        if sparse_data.n_features != mapper.n_features:
            raise Error(
                "a binned matrix and its mapper must agree on the feature"
                " count"
            )
        if sparse_data.n_bins != mapper.n_bins:
            raise Error("a binned matrix and its mapper must agree on n_bins")
        if len(sparse_data.col_offsets) != sparse_data.n_features + 1:
            raise Error(
                "a binned sparse matrix needs one column offset per feature,"
                " plus a final total"
            )
        if len(sparse_data.default_bin) != sparse_data.n_features:
            raise Error(
                "a binned sparse matrix needs one default bin per feature"
            )
        if len(sparse_data.row_index) != len(sparse_data.bin):
            raise Error("row indices and bins must be one per stored entry")
        if sparse_data.col_offsets[sparse_data.n_features] != len(
            sparse_data.bin
        ):
            raise Error("column offsets must end at the stored-entry count")
        var n_rows = sparse_data.n_rows
        var n_features = sparse_data.n_features
        _check_columns(
            n_rows,
            n_features,
            label,
            weight,
            group,
            init_score,
            feature_names,
            categorical_features,
        )
        return Dataset(
            mapper^,
            _empty_binned(n_features),
            sparse_data^,
            True,
            RawData.none(),
            borrowed_binning,
            n_rows,
            n_features,
            label^,
            weight^,
            group^,
            init_score^,
            feature_names^,
            categorical_features^,
            max_bin,
            use_missing,
        )

    @staticmethod
    def from_reference(
        reference: Dataset,
        var raw: RawData,
        var label: List[Float64] = [],
        var weight: List[Float64] = [],
        var group: List[Int] = [],
        var init_score: List[Float64] = [],
        keep_raw: Bool = False,
    ) raises -> Dataset:
        """A dataset binned by `reference`'s mapper instead of its own.

        This is LightGBM's `reference=` and it means one specific thing: the
        bin indices in the result mean what they mean in the reference. That
        is what a validation set needs, because it will be scored by a model
        trained on the reference, and what continued training requires,
        because `update_dataset` refuses a dataset binned any other way.

        It is also the wrong constructor for a cross-validation fold, and for
        the same reason it is the right one here: the reference's edges were
        fitted over rows this dataset does not hold, so a fold built this way
        would be scored under quantiles its held-out rows helped choose.
        `Dataset.from_raw` and `Dataset.subset` fit their own.

        The feature count, the binning parameters, the feature names, and the
        categorical declaration all come from the reference, because they
        describe the columns rather than the rows. Only the rows are this
        dataset's own.
        """
        if raw.n_features != reference.n_features:
            raise Error(
                "a dataset built from a reference must have the reference's"
                " features: it is binned by the reference's mapper"
            )
        var n_rows = raw.n_rows
        var n_features = raw.n_features
        _check_columns(
            n_rows,
            n_features,
            label,
            weight,
            group,
            init_score,
            reference.feature_names,
            reference.categorical_features,
        )
        var mapper = reference.mapper.copy()
        var is_sparse = raw.is_sparse
        var data = _empty_binned(n_features)
        var sparse_data = _empty_sparse_binned(n_features)
        if is_sparse:
            sparse_data = transform_csc(mapper, raw.csc)
        else:
            data = raw.transform_dense(mapper)
        var kept: RawData
        if keep_raw:
            kept = raw^
        else:
            kept = RawData.none()
        return Dataset(
            mapper^,
            data^,
            sparse_data^,
            is_sparse,
            kept^,
            True,
            n_rows,
            n_features,
            label^,
            weight^,
            group^,
            init_score^,
            reference.feature_names.copy(),
            reference.categorical_features.copy(),
            reference.max_bin,
            reference.use_missing,
        )

    @staticmethod
    def from_reference_dense[
        features_origin: ImmOrigin, //
    ](
        reference: Dataset,
        features: Span[Float64, features_origin],
        n_rows: Int,
        var label: List[Float64] = [],
        var weight: List[Float64] = [],
        var group: List[Int] = [],
        var init_score: List[Float64] = [],
        keep_raw: Bool = False,
    ) raises -> Dataset:
        """`from_reference` over a **borrowed** dense column-major matrix.

        The same constructor, taking the matrix the way the dense
        `Dataset.__init__` takes it. `from_reference` requires a `RawData`,
        which owns its values, so a caller holding a matrix on the other side
        of the Python boundary had to copy `n_rows * n_features` doubles into
        one before the reference's mapper could read them -- and then, unless
        `keep_raw` asked for it, that copy was dropped the moment binning
        finished. This one reads the caller's buffer where it lies and copies
        only when `keep_raw` means the dataset has to outlive it.

        Sparse input still goes through `from_reference`: a CSC matrix is
        already the representation the sparse binner reads, so there is no
        copy there to remove.
        """
        if reference.is_sparse:
            raise Error(
                "a dense dataset cannot be binned by a sparse reference's"
                " mapper"
            )
        _check_columns(
            n_rows,
            reference.n_features,
            label,
            weight,
            group,
            init_score,
            reference.feature_names,
            reference.categorical_features,
        )
        if len(features) != n_rows * reference.n_features:
            raise Error("features length must equal n_rows * n_features")

        var mapper = reference.mapper.copy()
        var data = mapper.transform(features, n_rows)
        var kept: RawData
        if keep_raw:
            var owned = List[Float64](unsafe_uninit_length=len(features))
            unsafe_memcpy(
                dest=owned.unsafe_ptr(),
                src=features.unsafe_ptr(),
                count=len(features),
            )
            kept = RawData.dense(owned^, n_rows, reference.n_features)
        else:
            kept = RawData.none()
        return Dataset(
            mapper^,
            data^,
            _empty_sparse_binned(reference.n_features),
            False,
            kept^,
            True,
            n_rows,
            reference.n_features,
            label^,
            weight^,
            group^,
            init_score^,
            reference.feature_names.copy(),
            reference.categorical_features.copy(),
            reference.max_bin,
            reference.use_missing,
        )

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "Dataset(n_rows=",
            self.n_rows,
            ", n_features=",
            self.n_features,
            ", sparse=",
            self.is_sparse,
            ")",
        )

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def num_data(self) -> Int:
        """Rows in the dataset."""
        return self.n_rows

    def num_feature(self) -> Int:
        """Features in the dataset."""
        return self.n_features

    def num_bin(self) -> Int:
        """Bins the binning reserved per feature, the effective `max_bin`."""
        return self.mapper.n_bins

    def nnz(self) -> Int:
        """Stored entries in the binned matrix: every cell for a dense
        dataset, the stored ones for a sparse dataset."""
        if self.is_sparse:
            return self.sparse_data.nnz()
        return self.n_rows * self.n_features

    def has_label(self) -> Bool:
        return len(self.label) != 0

    def has_weight(self) -> Bool:
        return len(self.weight) != 0

    def has_group(self) -> Bool:
        return len(self.group) != 0

    def has_init_score(self) -> Bool:
        return len(self.init_score) != 0

    def has_raw(self) -> Bool:
        """Whether the raw input was retained (`keep_raw=True`), which is
        what `subset` needs."""
        return not self.raw.is_empty()

    def feature_name(self, index: Int) raises -> String:
        """One feature's name, LightGBM's `Column_<i>` for a dataset built
        without names."""
        if index < 0 or index >= self.n_features:
            raise Error("feature index out of range")
        if len(self.feature_names) == 0:
            return String("Column_") + String(index)
        return self.feature_names[index].copy()

    def is_categorical(self, feature: Int) raises -> Bool:
        """Whether a feature was declared categorical for this binning."""
        if feature < 0 or feature >= self.n_features:
            raise Error("feature index out of range")
        for i in range(len(self.categorical_features)):
            if self.categorical_features[i] == feature:
                return True
        return False

    def matches_binning(self, other: Dataset) -> Bool:
        """Whether two datasets bin every value the same way, so that a bin
        index means the same thing in both.

        The mapper's own equality (see `BinMapper.matches`), plus the
        representation: a dense and a sparse dataset that bin identically
        still hold different matrix types, and no trainer reads both.
        """
        if self.is_sparse != other.is_sparse:
            return False
        return self.mapper.matches(other.mapper)

    def raw_matrix(self) raises -> RawData:
        """A copy of the retained raw input.

        Raises when the dataset did not keep it, which is the default: a
        dataset that dropped its raw matrix cannot produce it, and returning
        an empty one would look like an empty matrix.
        """
        if not self.has_raw():
            raise Error(
                "this dataset did not retain its raw matrix; build it with"
                " keep_raw=True to subset or re-bin it later"
            )
        return self.raw.copy()

    def subset(self, rows: List[Int], keep_raw: Bool = True) raises -> Dataset:
        """The named rows as their own dataset, **binned over those rows**.

        Row selection happens on the raw matrix and the bins are fitted
        afterwards, so the rows that were left out had no say in the edges.
        That is what makes this the constructor for a fold or a held-out
        split, and it is why the dataset has to have been built with
        `keep_raw=True`: bins cannot be refitted from bins.

        `rows` must be strictly ascending and in range (see
        `RawData.check_rows`). The label, the weights, and the init scores
        follow the rows; a ranking dataset's `group` is recomputed, and a
        selection that takes part of a query is refused.

        The subset keeps its own raw matrix by default, so a fold can be
        subset again; pass `keep_raw=False` for a leaf that will only be
        trained on.
        """
        var raw = self.raw_matrix()
        # Ahead of the group work below, which indexes by row and would
        # otherwise read out of range for a selection `subset` will reject.
        raw.check_rows(rows)
        var group = _subset_group(self.group, rows, self.n_rows)
        return Dataset.from_raw(
            raw.subset(rows),
            _subset_column(self.label, rows),
            _subset_column(self.weight, rows),
            group^,
            _subset_column(self.init_score, rows),
            self.feature_names.copy(),
            self.categorical_features.copy(),
            self.max_bin,
            self.use_missing,
            keep_raw,
        )

    def subset_shared_binning(
        self, rows: List[Int], keep_raw: Bool = False
    ) raises -> Dataset:
        """The named rows, binned by **this dataset's** mapper.

        LightGBM's `Dataset.subset`: the part is binned as the whole was, so
        its bin indices mean what the whole's mean and a model trained on the
        whole can be scored on it. That is exactly what makes it wrong for a
        cross-validation fold, whose whole point is that the held-out rows
        did not shape the binning; use `subset` for that.
        """
        var raw = self.raw_matrix()
        # As in `subset`: check before the group work indexes by row.
        raw.check_rows(rows)
        var group = _subset_group(self.group, rows, self.n_rows)
        return Dataset.from_reference(
            self,
            raw.subset(rows),
            _subset_column(self.label, rows),
            _subset_column(self.weight, rows),
            group^,
            _subset_column(self.init_score, rows),
            keep_raw,
        )


def _no_gpu_for_sparse() raises:
    raise Error(
        "a sparse dataset trains on the CPU: the GPU trainer reads a dense"
        " binned matrix, and densifying it is what the sparse path exists to"
        " avoid. Use device='cpu' or device='auto', or build the dataset"
        " from a dense matrix"
    )


def train_dataset(
    dataset: Dataset,
    objective: Int,
    params: BoosterParams,
    alpha: Float64 = 0.9,
    device: Int = CPU_DEVICE,
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
) raises -> Model:
    """Train a single-output model on an already binned dataset.

    The counterpart of `model.fit` for a dataset that is being reused: the
    binning is the dataset's, so `max_bin`, `use_missing`, and the
    categorical declaration come from it and are not passed again. The
    returned `Model` carries a copy of the dataset's mapper, which is what
    lets it predict on raw feature values.

    A sparse dataset trains through the sparse grower, which reads the
    matrix it holds. The model that comes back is an ordinary one either
    way: it serializes, loads, and predicts on dense rows identically.
    """
    _check_labels(dataset.label, dataset.n_rows)
    var booster: Booster
    if dataset.is_sparse:
        if device == GPU_DEVICE:
            _no_gpu_for_sparse()
        booster = train_sparse(
            dataset.sparse_data,
            dataset.label,
            objective,
            params,
            dataset.weight,
            alpha,
            bagging,
            goss,
            dataset.init_score,
        )
        return Model(dataset.mapper.copy(), booster^)

    # See `external_memory.train_external`: the objective is part of the
    # crossover rule and dropping it declines the GPU unconditionally.
    var backend = resolve_device(
        device, dataset.n_rows, dataset.n_features, 1, objective
    )
    if backend == GPU_DEVICE:
        if len(dataset.init_score) != 0:
            raise Error("init_score is a CPU training path; use device='cpu'")
        booster = train_gpu(
            dataset.data,
            dataset.label,
            objective,
            params,
            dataset.weight,
            alpha,
            bagging,
            goss,
        )
    else:
        booster = train(
            dataset.data,
            dataset.label,
            objective,
            params,
            dataset.weight,
            alpha,
            bagging,
            goss,
            dataset.init_score,
        )
    return Model(dataset.mapper.copy(), booster^)


def train_dataset_multiclass(
    dataset: Dataset,
    n_classes: Int,
    params: BoosterParams,
    device: Int = CPU_DEVICE,
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
) raises -> MulticlassModel:
    """Train a softmax model on an already binned dataset, on the backend
    `device` resolves to, exactly as in `model.fit_multiclass`.

    That sentence used to read "multiclass training is CPU-only, so
    GPU_DEVICE raises and AUTO_DEVICE resolves to the CPU", and the code
    matched it: the resolved backend was discarded and the CPU trainer was
    called unconditionally. It stopped being true when `train_multiclass_gpu`
    landed and `model.fit_multiclass` began dispatching on it, and the two
    entry points then disagreed in silence. A caller reaching multiclass
    through a `Dataset` got the CPU whatever it asked for, while the same
    request through `fit_multiclass` got the device.

    The evidence that this was real rather than theoretical is in the
    benchmark record: in `bench/real_data/results/20260815T023123Z`, the
    covertype scenario's CPU and GPU arms carry byte-identical
    `predictions_sha256`, while `dense_regression` and `imbalanced_binary`,
    which reach the device through the single-output path, do not. A
    multiclass GPU timing taken through this function was a CPU timing
    wearing a GPU label.

    A sparse dataset trains through `train_multiclass_sparse`, which does
    not implement GOSS; asking for it raises rather than training every row
    and reporting a run that sampled.
    """
    _check_labels(dataset.label, dataset.n_rows)
    if len(dataset.init_score) != 0:
        raise Error(
            "init_score is not supported for multiclass training: one offset"
            " per row cannot say what each class starts from"
        )
    if dataset.is_sparse:
        if device == GPU_DEVICE:
            _no_gpu_for_sparse()
        if goss.enabled:
            raise Error(
                "GOSS is not implemented for sparse multiclass training:"
                " train_multiclass_sparse trains on every row. Disable goss,"
                " or build the dataset from a dense matrix"
            )
        var sparse_booster = train_multiclass_sparse(
            dataset.sparse_data,
            _int_labels(dataset.label, n_classes),
            n_classes,
            params,
            dataset.weight,
            bagging,
        )
        return MulticlassModel(dataset.mapper.copy(), sparse_booster^)

    var backend = resolve_device(
        device, dataset.n_rows, dataset.n_features, n_classes, MULTICLASS
    )  # MULTICLASS is the registry's code for a softmax fit, which this
    # entry point is by construction and takes no objective argument for.
    var booster: MulticlassBooster
    if backend == GPU_DEVICE:
        booster = train_multiclass_gpu(
            dataset.data,
            _int_labels(dataset.label, n_classes),
            n_classes,
            params,
            dataset.weight,
            bagging,
            goss,
        )
    else:
        booster = train_multiclass(
            dataset.data,
            _int_labels(dataset.label, n_classes),
            n_classes,
            params,
            dataset.weight,
            bagging,
            goss,
        )
    return MulticlassModel(dataset.mapper.copy(), booster^)


def train_dataset_ranker(
    dataset: Dataset,
    params: BoosterParams,
    rank_params: RankerParams = RankerParams.default(),
    bagging: BaggingParams = BaggingParams.disabled(),
) raises -> Model:
    """Train a LambdaRank model on an already binned dataset, whose `group`
    holds the per-query row counts. CPU only, as `ranking.fit_ranker` is,
    and dense only: `train_ranker` reads a `BinnedMatrix`."""
    _check_labels(dataset.label, dataset.n_rows)
    if dataset.is_sparse:
        raise Error(
            "LambdaRank has no sparse trainer: train_ranker reads a dense"
            " binned matrix. Build the dataset from a dense matrix"
        )
    if len(dataset.group) == 0:
        raise Error(
            "a ranking dataset needs `group`: the number of rows in each"
            " query, in row order"
        )
    if len(dataset.init_score) != 0:
        raise Error(
            "init_score is not supported for ranking: lambdas are computed"
            " within a query and start from a score of 0"
        )
    var booster = train_ranker(
        dataset.data,
        _relevance_labels(dataset.label),
        groups_from_counts(dataset.group),
        params,
        rank_params,
        dataset.weight,
        bagging,
    )
    return Model(dataset.mapper.copy(), booster^)


def train_dataset_ranker_advanced(
    dataset: Dataset,
    params: BoosterParams,
    rank_params: AdvancedRankParams,
    positions: PositionMap = PositionMap.absent(),
    bagging: BaggingParams = BaggingParams.disabled(),
) raises -> Model:
    """`train_dataset_ranker` with the advanced ranking parameters
    (`label_gain`, position bias, pair sampling, a decoupled maxDCG cutoff).

    Routing is `ranking_advanced.advanced_ranking_requested`: when nothing
    advanced is asked for this is exactly `train_dataset_ranker` with
    `rank_params.base`, so a default run trains the same model it always
    did; otherwise the ensemble comes from `train_ranker_advanced`, whose
    learned position biases are training state that no model file holds.
    """
    if not advanced_ranking_requested(rank_params, positions):
        return train_dataset_ranker(dataset, params, rank_params.base, bagging)
    _check_labels(dataset.label, dataset.n_rows)
    if dataset.is_sparse:
        raise Error(
            "LambdaRank has no sparse trainer: train_ranker reads a dense"
            " binned matrix. Build the dataset from a dense matrix"
        )
    if len(dataset.group) == 0:
        raise Error(
            "a ranking dataset needs `group`: the number of rows in each"
            " query, in row order"
        )
    if len(dataset.init_score) != 0:
        raise Error(
            "init_score is not supported for ranking: lambdas are computed"
            " within a query and start from a score of 0"
        )
    var trained = train_ranker_advanced(
        dataset.data,
        _relevance_labels(dataset.label),
        groups_from_counts(dataset.group),
        params,
        rank_params,
        positions,
        dataset.weight,
        bagging,
    )
    return Model(dataset.mapper.copy(), trained.booster.copy())


def update_dataset(
    mut model: Model,
    dataset: Dataset,
    params: BoosterParams,
    alpha: Float64 = 0.9,
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
) raises -> Int:
    """Append `params.n_estimators` more rounds to `model` from `dataset`,
    returning how many trees were added.

    The dataset must be binned by the mapper the model was trained under, or
    a bin index would mean one thing to the trees already in the model and
    another to the ones about to be grown. That is checked here rather than
    assumed: a dataset built from the same data with the same binning
    parameters passes, and any other dataset raises. `Dataset.from_reference`
    is the constructor that guarantees it for a dataset built separately.

    Continued training runs on the CPU, and on a dense dataset. The GPU
    trainer grows trees from a device-resident state that starts at the base
    score, so resuming from an existing ensemble has no GPU path yet; the
    trees it would produce are the CPU trees either way (see
    docs/GPU_VALIDATION.md). The sparse grower has no `train_more`
    counterpart, so a sparse dataset is refused rather than densified.
    """
    if dataset.is_sparse:
        raise Error(
            "continued training has no sparse path: boosting_sparse has no"
            " train_more counterpart, so the rounds would have to be grown"
            " from a densified matrix. Train a sparse dataset in one call"
        )
    if not model.mapper.matches(dataset.mapper):
        raise Error(
            "continued training needs the dataset the model was trained on:"
            " this one is binned differently"
        )
    _check_labels(dataset.label, dataset.n_rows)
    return train_more(
        model.booster,
        dataset.data,
        dataset.label,
        params,
        dataset.weight,
        alpha,
        bagging,
        goss,
        dataset.init_score,
    )


def update_dataset_multiclass(
    mut model: MulticlassModel,
    dataset: Dataset,
    params: BoosterParams,
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
) raises -> Int:
    """Append `params.n_estimators` more softmax rounds to `model` from
    `dataset`, returning how many rounds were added (one round is one tree
    per class).

    The multiclass counterpart of `update_dataset`, with the same binning
    requirement: the dataset must be binned by the mapper the model was
    trained under, or a bin index would mean one thing to the trees already
    in the model and another to the ones about to be grown.

    The class count is the model's, and the labels are checked against it
    rather than against a count passed in again: a dataset whose labels have
    outgrown the ensemble's classes cannot be continued into, because the
    ensemble has no tree sequence for a class it was never fitted with.
    Continued training runs on the CPU and on a dense dataset, as
    `update_dataset` does.
    """
    if dataset.is_sparse:
        raise Error(
            "continued training has no sparse path: boosting_sparse has no"
            " train_multiclass_more counterpart. Train a sparse dataset in"
            " one call"
        )
    if not model.mapper.matches(dataset.mapper):
        raise Error(
            "continued training needs the dataset the model was trained on:"
            " this one is binned differently"
        )
    _check_labels(dataset.label, dataset.n_rows)
    if len(dataset.init_score) != 0:
        raise Error(
            "init_score is not supported for multiclass training: one offset"
            " per row cannot say what each class starts from"
        )
    return train_multiclass_more(
        model.booster,
        dataset.data,
        _int_labels(dataset.label, model.booster.n_classes),
        params,
        dataset.weight,
        bagging,
        goss,
    )
