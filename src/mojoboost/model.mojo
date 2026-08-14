"""End-to-end model: raw features in, predictions out.

`fit` bins the raw training matrix with quantile binning, trains a boosted
ensemble, and returns a `Model` that carries the fitted `BinMapper` so it
can predict on raw, unseen feature values.
"""

from .binning import BinMapper, BinnedMatrix, fit_bins
from .boosting import (
    Booster,
    BoosterParams,
    MulticlassBooster,
    train,
    train_multiclass,
)


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
) raises -> Model:
    """Fit on a column-major raw feature matrix (`features[f * n_rows + r]`)."""
    var mapper = fit_bins(features, n_rows, n_features, max_bins)
    var data = mapper.transform(features, n_rows)
    var booster = train(data, target, objective, params, sample_weight)
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
) raises -> MulticlassModel:
    """Fit a softmax multiclass model on a column-major raw feature matrix
    (`features[f * n_rows + r]`), labels in 0..n_classes-1."""
    var mapper = fit_bins(features, n_rows, n_features, max_bins)
    var data = mapper.transform(features, n_rows)
    var booster = train_multiclass(data, labels, n_classes, params, sample_weight)
    return MulticlassModel(mapper^, booster^)
