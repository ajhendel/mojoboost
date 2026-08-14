"""A binned training set that outlives the model trained on it.

`Dataset` is the Mojo side of LightGBM's dataset object: a feature matrix
that has already been binned, together with the label, weights, query
groups, and init scores that belong to it. Binning is the expensive part of
starting a run, so a `Dataset` is constructed once and then trained on as
many times as the caller likes, with `train_dataset` and its multiclass and
ranking counterparts, and extended with `update_dataset`.

What a `Dataset` owns

- the fitted `BinMapper` and the `BinnedMatrix` it produced, which is the
  training data itself; the raw float64 matrix is not retained, so callers
  that want it keep their own copy (the Python wrapper does, unless it is
  told to free it)
- the label, sample weights, per-query row counts, and init scores, each
  optional and each validated against the row count on construction
- the feature names and the categorical feature declaration, which are
  metadata a model file does not carry

What it deliberately does not offer is LightGBM's family of post-construction
mutators (`set_label`, `set_field`, `set_categorical_feature`, and the rest).
Bin edges are fitted from the data and the categorical declaration, so
changing either afterwards would leave the binned matrix describing data the
dataset no longer holds. Fields are supplied at construction; change one by
constructing another dataset.
"""

from .bagging import BaggingParams
from .binning import BinMapper, BinnedMatrix, fit_bins
from .boosting import (
    Booster,
    BoosterParams,
    MulticlassBooster,
    train,
    train_more,
    train_multiclass,
)
from .device import CPU_DEVICE, GPU_DEVICE, resolve_device
from .goss import GossParams
from .model import Model, MulticlassModel
from .ranking import (
    RankerParams,
    groups_from_counts,
    train_ranker,
)
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


struct Dataset(Copyable, Movable, Writable):
    """A binned feature matrix and the columns that go with it."""

    var mapper: BinMapper
    var data: BinnedMatrix
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
    ) raises:
        """Bin a column-major raw feature matrix (`features[f * n_rows + r]`)
        and take ownership of the columns that describe its rows.

        Every optional column is validated against `n_rows` here rather than
        at train time, so a mismatch is reported while the caller still knows
        which array it passed. An empty column means the dataset has none.
        """
        if n_rows < 1:
            raise Error("a dataset needs at least one row")
        if n_features < 1:
            raise Error("a dataset needs at least one feature")
        if len(features) != n_rows * n_features:
            raise Error("features length must equal n_rows * n_features")
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

        self.mapper = fit_bins(
            features,
            n_rows,
            n_features,
            max_bin,
            use_missing=use_missing,
            categorical_features=categorical_features,
        )
        self.data = self.mapper.transform(features, n_rows)
        self.label = label^
        self.weight = weight^
        self.group = group^
        self.init_score = init_score^
        self.feature_names = feature_names^
        self.categorical_features = categorical_features^
        self.max_bin = max_bin
        self.use_missing = use_missing

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "Dataset(n_rows=",
            self.data.n_rows,
            ", n_features=",
            self.data.n_features,
            ")",
        )

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def num_data(self) -> Int:
        """Rows in the dataset."""
        return self.data.n_rows

    def num_feature(self) -> Int:
        """Features in the dataset."""
        return self.data.n_features

    def num_bin(self) -> Int:
        """Bins the binning reserved per feature, the effective `max_bin`."""
        return self.mapper.n_bins


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
    """
    _check_labels(dataset.label, dataset.data.n_rows)
    var backend = resolve_device(
        device, dataset.data.n_rows, dataset.data.n_features, 1
    )
    var booster: Booster
    if backend == GPU_DEVICE:
        if len(dataset.init_score) != 0:
            raise Error(
                "init_score is a CPU training path; use device='cpu'"
            )
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
    the CPU, exactly as in `model.fit_multiclass`."""
    _check_labels(dataset.label, dataset.data.n_rows)
    if len(dataset.init_score) != 0:
        raise Error(
            "init_score is not supported for multiclass training: one offset"
            " per row cannot say what each class starts from"
        )
    _ = resolve_device(
        device, dataset.data.n_rows, dataset.data.n_features, n_classes
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
    holds the per-query row counts. CPU only, as `ranking.fit_ranker` is."""
    _check_labels(dataset.label, dataset.data.n_rows)
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
    parameters passes, and any other dataset raises.

    Continued training runs on the CPU. The GPU trainer grows trees from a
    device-resident state that starts at the base score, so resuming from an
    existing ensemble has no GPU path yet; the trees it would produce are the
    CPU trees either way (see docs/GPU_VALIDATION.md).
    """
    if not model.mapper.matches(dataset.mapper):
        raise Error(
            "continued training needs the dataset the model was trained on:"
            " this one is binned differently"
        )
    _check_labels(dataset.label, dataset.data.n_rows)
    return train_more(
        model.booster,
        dataset.data,
        dataset.label,
        params,
        dataset.weight,
        alpha,
        bagging,
        goss,
    )
