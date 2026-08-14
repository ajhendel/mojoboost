"""End-to-end model: raw features in, predictions out.

`fit` bins the raw training matrix with quantile binning, trains a boosted
ensemble, and returns a `Model` that carries the fitted `BinMapper` so it
can predict on raw, unseen feature values.
"""

from .binning import BinMapper, BinnedMatrix, fit_bins
from .boosting import Booster, BoosterParams, train


@fieldwise_init
struct Model(Copyable, Movable):
    var mapper: BinMapper
    var booster: Booster

    def predict(self, row: List[Float64]) raises -> Float64:
        """Response-scale prediction for one raw example (length n_features)."""
        return self.booster.predict_bins(self.mapper.bin_row(row))

    def predict_raw(self, row: List[Float64]) raises -> Float64:
        """Raw-score prediction (log-odds for BINARY_LOGISTIC)."""
        return self.booster.predict_raw_bins(self.mapper.bin_row(row))


def fit(
    features: List[Float64],
    n_rows: Int,
    n_features: Int,
    target: List[Float64],
    objective: Int,
    params: BoosterParams,
    max_bins: Int = 255,
) raises -> Model:
    """Fit on a column-major raw feature matrix (`features[f * n_rows + r]`)."""
    var mapper = fit_bins(features, n_rows, n_features, max_bins)
    var data = mapper.transform(features, n_rows)
    var booster = train(data, target, objective, params)
    return Model(mapper^, booster^)
