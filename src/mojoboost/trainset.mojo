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

Prepared tables are not serialized here. Writing a binned matrix and its
mapper to a file the way `serialize.mojo` writes a model needs that module's
mapper reader and writer, which are private to it; see
handoffs/connect_12_dataset_cv.md for the exact request.
"""

from .bagging import BaggingParams
from .binning import BinMapper, BinnedMatrix, BinnedMatrix, fit_bins
from .boosting import (
    Booster,
    BoosterParams,
    MulticlassBooster,
    train,
    train_more,
    train_multiclass,
    train_multiclass_more,
)
from .boosting_sparse import train_multiclass_sparse, train_sparse
from .device import CPU_DEVICE, GPU_DEVICE, resolve_device
from .goss import GossParams
from .model import Model, MulticlassModel
from .ranking import RankerParams, groups_from_counts, train_ranker
from .raw_data import RawData
from .sparse import CscMatrix, CsrMatrix, SparseBinnedMatrix, transform_csc
from .train_gpu import train_gpu


def _check_labels(label: List[Float64], n_rows: Int) raises:
    if len(label) != n_rows:
        raise Error("a dataset needs one label per row to train on")


def _int_labels(label: List[Float64], n_classes: Int) raises -> List[Int]:
    """Class codes from a float64 label column, which is how labels reach a
    dataset. Whole numbers in `[0, n_classes)` only: a fractional code would
    otherwise truncate into a neighboring class."""
    var out = List[Int](capacity=len(label))
    for r in range(len(label)):
        var v = label[r]
        if v != Float64(Int(v)):
            raise Error("class labels must be whole numbers")
        var code = Int(v)
        if code < 0 or code >= n_classes:
            raise Error("class label out of range")
        out.append(code)
    return out^


def _relevance_labels(label: List[Float64]) raises -> List[Int]:
    """Graded relevances from a float64 label column. `train_ranker` checks
    the range; this only rejects the fractional values that would otherwise
    truncate silently."""
    var out = List[Int](capacity=len(label))
    for r in range(len(label)):
        var v = label[r]
        if v != Float64(Int(v)):
            raise Error("relevance labels must be whole numbers")
        out.append(Int(v))
    return out^


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
    dataset built from a dense one rejects.
    """
    if n_rows < 1:
        raise Error("a dataset needs at least one row")
    if n_features < 1:
        raise Error("a dataset needs at least one feature")
    if len(label) != 0 and len(label) != n_rows:
        raise Error("label length must equal n_rows")
    if len(weight) != 0 and len(weight) != n_rows:
        raise Error("weight length must equal n_rows")
    if len(init_score) != 0 and len(init_score) != n_rows:
        raise Error("init_score length must equal n_rows")
    if len(feature_names) != 0 and len(feature_names) != n_features:
        raise Error("feature_name must have one name per feature")
    if len(group) != 0:
        var total = 0
        for q in range(len(group)):
            if group[q] < 1:
                raise Error("group counts must be positive")
            total += group[q]
        if total != n_rows:
            raise Error("group counts must sum to n_rows")
    for i in range(len(categorical_features)):
        var f = categorical_features[i]
        if f < 0 or f >= n_features:
            raise Error("categorical feature index out of range")
        for j in range(i):
            if categorical_features[j] == f:
                raise Error("categorical feature index listed twice")


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

    def __init__(
        out self,
        features: List[Float64],
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
            self.raw = RawData.dense(features.copy(), n_rows, n_features)
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
        var kept = RawData.none()
        if keep_raw:
            kept = raw^
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
        var kept = RawData.none()
        if keep_raw:
            kept = raw^
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
        return self.feature_names[index]

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

    var backend = resolve_device(
        device, dataset.n_rows, dataset.n_features, 1
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
    """Train a softmax model on an already binned dataset. Multiclass
    training is CPU-only, so GPU_DEVICE raises and AUTO_DEVICE resolves to
    the CPU, exactly as in `model.fit_multiclass`.

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

    _ = resolve_device(
        device, dataset.n_rows, dataset.n_features, n_classes
    )
    var booster = train_multiclass(
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
