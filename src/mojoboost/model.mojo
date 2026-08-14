"""End-to-end model: raw features in, predictions out.

`fit` bins the raw training matrix with quantile binning, trains a boosted
ensemble, and returns a `Model` that carries the fitted `BinMapper` so it
can predict on raw, unseen feature values.

Both entry points take a `device` (see device.mojo). The device chooses
which trainer grows the trees; it is not a property of the fitted model,
which is the same tree ensemble either way and serializes identically.
"""

from .bagging import BaggingParams
from .binning import BinMapper, BinnedMatrix, fit_bins
from .boosting import (
    Booster,
    BoosterParams,
    IterationRange,
    MulticlassBooster,
    train,
    train_multiclass,
)
from .device import CPU_DEVICE, GPU_DEVICE, resolve_device
from .goss import GossParams
from .gpu_predict import (
    predict_gpu,
    predict_proba_gpu,
    predict_raw_gpu,
    predict_raw_multiclass_gpu,
)
from .objective import GradHessFn, train_custom
from .train_gpu import train_gpu, train_multiclass_gpu


@fieldwise_init
struct Model(Copyable, Movable, Writable):
    var mapper: BinMapper
    var booster: Booster

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Model(n_trees=", len(self.booster.trees), ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def predict(self, row: List[Float64]) raises -> Float64:
        """Response-scale prediction for one raw example (length n_features)."""
        return self.booster.predict_bins(self.mapper.bin_row(row))

    def predict_raw(self, row: List[Float64]) raises -> Float64:
        """Raw-score prediction (log-odds for BINARY_LOGISTIC)."""
        return self.booster.predict_raw_bins(self.mapper.bin_row(row))

    def n_iterations(self) -> Int:
        """Boosting iterations the fitted ensemble kept."""
        return self.booster.n_iterations()

    def predict_range(
        self, row: List[Float64], rng: IterationRange
    ) raises -> Float64:
        """Response-scale prediction from the boosting iterations in `rng`
        alone. See IterationRange for how the range is interpreted, including
        where the base score sits."""
        return self.booster.predict_bins_range(self.mapper.bin_row(row), rng)

    def predict_raw_range(
        self, row: List[Float64], rng: IterationRange
    ) raises -> Float64:
        """Raw-score prediction from the iterations in `rng` alone."""
        return self.booster.predict_raw_bins_range(
            self.mapper.bin_row(row), rng
        )

    def leaf_indices(
        self, row: List[Float64], rng: IterationRange
    ) raises -> List[Int]:
        """The leaf ordinal this raw example reaches in each tree of `rng`,
        one entry per iteration. See `Tree.leaf_ordinals` for the numbering
        and its stability guarantees."""
        return self.booster.leaf_indices_bins(self.mapper.bin_row(row), rng)

    def predict_batch(
        self,
        features: List[Float64],
        n_rows: Int,
        rng: IterationRange,
        raw_score: Bool = False,
        device: Int = CPU_DEVICE,
    ) raises -> List[Float64]:
        """Batched prediction on a raw column-major matrix
        (`features[f * n_rows + r]`), one output per row.

        `device` follows the training-time vocabulary in device.mojo:
        CPU_DEVICE walks the trees on the host, GPU_DEVICE walks them on the
        accelerator and raises when none is available, and AUTO_DEVICE keeps
        resolving to the CPU until a measured crossover exists. Binning
        always runs on the host (bin edges are Float64; a Float32 edge
        search could move a row a whole bin), so the two devices route every
        row to the same leaf and differ only by the Float32 accumulation of
        leaf values, the same contract the GPU trainer ships."""
        var resolved = resolve_device(
            device, n_rows, self.mapper.n_features, 1
        )
        var data = self.mapper.transform(features, n_rows)
        if resolved == GPU_DEVICE:
            if raw_score:
                return predict_raw_gpu(self.booster, data, rng)
            return predict_gpu(self.booster, data, rng)
        var out = List[Float64](capacity=n_rows)
        var bins = List[Int](capacity=self.mapper.n_features)
        for r in range(n_rows):
            bins.clear()
            for f in range(self.mapper.n_features):
                bins.append(Int(data.bins[f * n_rows + r]))
            if raw_score:
                out.append(self.booster.predict_raw_bins_range(bins, rng))
            else:
                out.append(self.booster.predict_bins_range(bins, rng))
        return out^


@fieldwise_init
struct MulticlassModel(Copyable, Movable, Writable):
    var mapper: BinMapper
    var booster: MulticlassBooster

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "MulticlassModel(n_classes=",
            self.booster.n_classes,
            ", n_trees=",
            len(self.booster.trees),
            ")",
        )

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def predict_proba(self, row: List[Float64]) raises -> List[Float64]:
        """Class probabilities for one raw example (length n_features)."""
        return self.booster.predict_proba_bins(self.mapper.bin_row(row))

    def predict_raw(self, row: List[Float64]) raises -> List[Float64]:
        """Raw per-class scores before the softmax."""
        return self.booster.predict_raw_bins(self.mapper.bin_row(row))

    def n_iterations(self) -> Int:
        """Boosting iterations the fitted ensemble kept: one iteration grows
        one tree per class."""
        return self.booster.n_iterations()

    def predict_proba_range(
        self, row: List[Float64], rng: IterationRange
    ) raises -> List[Float64]:
        """Class probabilities from the boosting iterations in `rng` alone."""
        return self.booster.predict_proba_bins_range(
            self.mapper.bin_row(row), rng
        )

    def predict_raw_range(
        self, row: List[Float64], rng: IterationRange
    ) raises -> List[Float64]:
        """Per-class raw scores from the iterations in `rng` alone."""
        return self.booster.predict_raw_bins_range(
            self.mapper.bin_row(row), rng
        )

    def leaf_indices(
        self, row: List[Float64], rng: IterationRange
    ) raises -> List[Int]:
        """The leaf ordinal this raw example reaches in each tree of `rng`,
        round-major: entry `i * n_classes + k` is class k's tree in the
        range's iteration i."""
        return self.booster.leaf_indices_bins(self.mapper.bin_row(row), rng)

    def predict_batch(
        self,
        features: List[Float64],
        n_rows: Int,
        rng: IterationRange,
        raw_score: Bool = False,
        device: Int = CPU_DEVICE,
    ) raises -> List[Float64]:
        """Batched prediction on a raw column-major matrix
        (`features[f * n_rows + r]`), row-major `[r * n_classes + k]`
        outputs: class probabilities, or per-class raw scores with
        `raw_score`.

        `device` carries the same meaning as in `Model.predict_batch`, and
        the same contract holds: binning always runs on the host, so both
        devices route every row to the same leaves and differ only by
        Float32 accumulation (and, for probabilities, the Float32
        softmax)."""
        var resolved = resolve_device(
            device, n_rows, self.mapper.n_features, self.booster.n_classes
        )
        var data = self.mapper.transform(features, n_rows)
        if resolved == GPU_DEVICE:
            if raw_score:
                return predict_raw_multiclass_gpu(self.booster, data, rng)
            return predict_proba_gpu(self.booster, data, rng)
        var n_classes = self.booster.n_classes
        var out = List[Float64](capacity=n_rows * n_classes)
        var bins = List[Int](capacity=self.mapper.n_features)
        for r in range(n_rows):
            bins.clear()
            for f in range(self.mapper.n_features):
                bins.append(Int(data.bins[f * n_rows + r]))
            var scores: List[Float64]
            if raw_score:
                scores = self.booster.predict_raw_bins_range(bins, rng)
            else:
                scores = self.booster.predict_proba_bins_range(bins, rng)
            for k in range(n_classes):
                out.append(scores[k])
        return out^

    def predict_class(self, row: List[Float64]) raises -> Int:
        """The argmax class for one raw example."""
        var raw = self.predict_raw(row)
        var argmax = 0
        for k in range(1, len(raw)):
            if raw[k] > raw[argmax]:
                argmax = k
        return argmax


def fit(
    features: List[Float64],
    n_rows: Int,
    n_features: Int,
    target: List[Float64],
    objective: Int,
    params: BoosterParams,
    max_bins: Int = 255,
    sample_weight: List[Float64] = [],
    alpha: Float64 = 0.9,
    device: Int = CPU_DEVICE,
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
    use_missing: Bool = True,
    categorical_features: List[Int] = [],
) raises -> Model:
    """Fit on a column-major raw feature matrix (`features[f * n_rows + r]`).
    `alpha` is the target quantile for QUANTILE and the huber transition
    point for HUBER; other objectives ignore it. `device` is CPU_DEVICE,
    GPU_DEVICE, or AUTO_DEVICE; GPU_DEVICE raises when no accelerator is
    available instead of falling back. `bagging` samples training rows per
    tree (see bagging.mojo) and draws the same rows on either device; `goss`
    is the gradient-based alternative (see goss.mojo), likewise identical on
    either device. `use_missing` is LightGBM's parameter of that name: with
    it, `NaN` feature values train and predict as missing (see binning.mojo);
    without it they are binned as 0.0. `categorical_features` lists the
    feature indices to treat as integer-coded categoricals: they are split by
    category set rather than by threshold, and missing, unseen, and dropped
    categories route right (see categorical.mojo)."""
    var backend = resolve_device(device, n_rows, n_features, 1)
    var mapper = fit_bins(
        features,
        n_rows,
        n_features,
        max_bins,
        use_missing=use_missing,
        categorical_features=categorical_features,
    )
    var data = mapper.transform(features, n_rows)
    var booster: Booster
    if backend == GPU_DEVICE:
        booster = train_gpu(
            data,
            target,
            objective,
            params,
            sample_weight,
            alpha,
            bagging,
            goss,
        )
    else:
        booster = train(
            data,
            target,
            objective,
            params,
            sample_weight,
            alpha,
            bagging,
            goss,
        )
    return Model(mapper^, booster^)


def fit_multiclass(
    features: List[Float64],
    n_rows: Int,
    n_features: Int,
    labels: List[Int],
    n_classes: Int,
    params: BoosterParams,
    max_bins: Int = 255,
    sample_weight: List[Float64] = [],
    device: Int = CPU_DEVICE,
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
    use_missing: Bool = True,
    categorical_features: List[Int] = [],
) raises -> MulticlassModel:
    """Fit a softmax multiclass model on a column-major raw feature matrix
    (`features[f * n_rows + r]`), labels in 0..n_classes-1. `device` carries
    the same meaning as in `fit`: multiclass runs on either backend, growing
    one tree per class per round, so GPU_DEVICE trains on the device rather
    than raising. `bagging` draws one bag per round, shared by every class's
    tree in that round, and `goss` draws its gradient-based sample on the
    same once-per-round schedule; both draw identical rows on either device.
    `use_missing` and `categorical_features` carry the same meaning as in
    `fit`."""
    var backend = resolve_device(device, n_rows, n_features, n_classes)
    var mapper = fit_bins(
        features,
        n_rows,
        n_features,
        max_bins,
        use_missing=use_missing,
        categorical_features=categorical_features,
    )
    var data = mapper.transform(features, n_rows)
    var booster: MulticlassBooster
    if backend == GPU_DEVICE:
        booster = train_multiclass_gpu(
            data, labels, n_classes, params, sample_weight, bagging, goss
        )
    else:
        booster = train_multiclass(
            data, labels, n_classes, params, sample_weight, bagging, goss
        )
    return MulticlassModel(mapper^, booster^)


def fit_custom[F: GradHessFn](
    features: List[Float64],
    n_rows: Int,
    n_features: Int,
    target: List[Float64],
    grad_hess: F,
    params: BoosterParams,
    max_bins: Int = 255,
    sample_weight: List[Float64] = [],
    base_score: Float64 = 0.0,
    use_missing: Bool = True,
    categorical_features: List[Int] = [],
) raises -> Model:
    """Fit a caller-supplied objective on a column-major raw feature matrix
    (`features[f * n_rows + r]`), the `fit` counterpart of `train_custom`
    (see objective.mojo for the callback contract).

    `Model.predict` returns the raw score for a custom-objective model,
    since the framework does not know the inverse link. CPU only: there is
    no `device` argument, use `train_custom_gpu` on a pre-binned matrix for
    GPU tree growth. `use_missing` and `categorical_features` carry the same
    meaning as in `fit`."""
    var mapper = fit_bins(
        features,
        n_rows,
        n_features,
        max_bins,
        use_missing=use_missing,
        categorical_features=categorical_features,
    )
    var data = mapper.transform(features, n_rows)
    var booster = train_custom(
        data, target, grad_hess, params, sample_weight, base_score
    )
    return Model(mapper^, booster^)
