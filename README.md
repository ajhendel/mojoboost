# mojoboost

Gradient boosted decision trees in [Mojo](https://www.modular.com/mojo).

mojoboost is a from-scratch GBDT library in the LightGBM family. It uses
histogram-based split finding and leaf-wise (best-first) tree growth, and it
matches LightGBM's default hyperparameters so results are directly comparable.

## Why Mojo

LightGBM and XGBoost are excellent, mature C++ libraries. The bet here is
that Mojo enables a simpler codebase that is at least as fast, because the
hot loop of gradient boosting (histogram accumulation over binned features)
is exactly what Mojo is built for. Explicit SIMD with per-chip vector widths,
compile-time specialization of the training loop for bin count and dtype
(no virtual dispatch or runtime branching), precise control of memory layout
for cache tiling, and structured parallelism, all in one language that also
targets GPUs from the same source.

## Status

Early development. Training works end to end. What works today, each piece
with tests

- equal-width feature binning into `uint8` bins
- histogram accumulation with the sibling subtraction trick
- best-split search with the standard second-order gain formula,
  `min_data_in_leaf` and hessian constraints
- leaf-wise (best-first) tree growth with `num_leaves` cap and Newton-step
  leaf values
- boosting loop with squared-error and binary-logistic objectives

```mojo
from mojoboost import (
    BINARY_LOGISTIC, BoosterParams, TreeParams, bin_equal_width, train,
)

def main() raises:
    var data = bin_equal_width(features, n_rows, n_features, n_bins=255)
    var params = BoosterParams(100, 0.1, TreeParams.default())
    var model = train(data, labels, BINARY_LOGISTIC, params)
    var p = model.predict_row(data, 0)
```

## Roadmap

1. Quantile (equal-frequency) binning, LightGBM style, with stored bin edges
   so prediction works on raw (unbinned) data
2. Multiclass objective and validation-set early stopping
3. SIMD histogram kernels and multicore training
4. scikit-learn style `fit`/`predict` Python API via Mojo interop
5. Reproducible benchmark suite vs LightGBM and XGBoost (same defaults, same
   datasets, hardware documented)
6. GPU training

## Defaults

Matched to LightGBM so comparisons are apples to apples.

| Parameter | Default |
|---|---|
| `num_leaves` | 31 |
| `learning_rate` | 0.1 |
| `n_estimators` | 100 |
| `min_data_in_leaf` | 20 |
| `max_bin` | 255 |
| `lambda_l2` | 0.0 (gain search default 1.0 until the boosting loop lands) |

## Development

Requires [pixi](https://pixi.sh).

```sh
pixi install
pixi run test
```

## Benchmarks

Coming with the boosting loop. Nothing will be claimed here that is not
reproducible from scripts in `benchmarks/`.

## License

Apache-2.0
