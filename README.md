# mojoboost

[![CI](https://github.com/ajhendel/mojoboost/actions/workflows/ci.yml/badge.svg)](https://github.com/ajhendel/mojoboost/actions/workflows/ci.yml)

Gradient boosted decision trees in [Mojo](https://www.modular.com/mojo).

mojoboost is a from-scratch GBDT library in the LightGBM family. It uses
histogram-based split finding and leaf-wise (best-first) tree growth. Its
benchmark configuration aligns important parameters with LightGBM for
reproducible comparisons.

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

- quantile (equal-frequency) feature binning into `uint8` bins, LightGBM
  style, with stored edges so trained models predict on raw, unseen data
- histogram accumulation with the sibling subtraction trick
- best-split search with the standard second-order gain formula,
  `min_data_in_leaf` and hessian constraints
- leaf-wise (best-first) tree growth with `num_leaves` cap and Newton-step
  leaf values
- objectives: squared error, binary logistic, poisson, and multiclass
  softmax
- validation-set early stopping with `min_delta` for every objective,
  truncating to the best round
- evaluation metrics: RMSE, log loss, accuracy, and ROC AUC with
  sklearn-matching tie handling
- sample weights for every objective, LightGBM semantics (weighted
  gradients, hessians, and base scores; zero-weight rows are ignored)
- feature importance, split counts and total gain, matching LightGBM's
  two importance types
- SIMD histogram kernels (pointer-based scatter accumulation, vectorized
  sibling subtraction and split scans)
- multicore CPU histogram accumulation across independent features
- experimental portable GPU histogram accumulation, tested for correctness
  on Apple Metal; CUDA and HIP validation is still required
- model serialization: `save_model`/`load_model` with a versioned text
  format that stores floats as raw bit patterns, so loaded models predict
  bit-exactly; multiclass models via `save_multiclass_model` and
  `load_multiclass_model`
- scikit-learn style Python API (`MojoBoostRegressor`,
  `MojoBoostClassifier`) backed by a CPython extension module built from
  the same Mojo code, with sample weights and exact save/load
- multiclass end to end on raw data: `fit_multiclass` returns a
  `MulticlassModel` with `predict_proba` and `predict_class`

```mojo
from mojoboost import BINARY_LOGISTIC, BoosterParams, TreeParams, fit

def main() raises:
    # features is column-major: features[f * n_rows + r]
    var params = BoosterParams(100, 0.1, TreeParams.default())
    var model = fit(features, n_rows, n_features, labels,
                    BINARY_LOGISTIC, params)
    var p = model.predict(row)   # raw feature values in, probability out
```

Lower-level entry points `train`, `train_with_valid`, and
`train_multiclass` operate on pre-binned matrices.

## Python API

Build the extension once with `bindings/build.sh`, then use the
scikit-learn style estimators in `python/mojoboost`:

```python
from mojoboost import MojoBoostRegressor, MojoBoostClassifier

model = MojoBoostRegressor(num_leaves=31, n_estimators=100).fit(X, y)
pred = model.predict(X)          # numpy in/out when numpy is available
model.save("model.mbst")
model = MojoBoostRegressor.load("model.mbst")

clf = MojoBoostClassifier().fit(X, labels)   # binary or multiclass by labels
proba = clf.predict_proba(X)
```

`fit` accepts `sample_weight`, hyperparameters mirror the Mojo defaults,
and saved models round-trip bit-exactly. numpy is optional; plain Python
sequences work without it.

## Roadmap

1. Integrate the GPU histogram backend into end-to-end training while keeping
   intermediate state device-resident
2. Scale GPU histograms beyond one threadgroup per feature and validate on
   Apple, NVIDIA, and AMD hardware
3. Package the Python API for distribution (wheels, PyPI)
4. Broader benchmark suite (XGBoost and real datasets)

## Defaults

Matched to LightGBM so comparisons are apples to apples.

| Parameter | Default |
|---|---|
| `num_leaves` | 31 |
| `learning_rate` | 0.1 |
| `n_estimators` | 100 |
| `min_data_in_leaf` | 20 |
| `max_bin` | 255 |
| `lambda_l2` | 1.0 (LightGBM's own default is 0; benchmarks set both to 1.0) |

## Development

Requires [pixi](https://pixi.sh).

```sh
pixi install
pixi run test
```

The test command includes CPU/GPU equivalence checks. They run when a
supported accelerator is present and skip cleanly on CPU-only machines.

## Benchmarks

Reproducible from `bench/` (methodology, exact parameters, and caveats in
[bench/README.md](bench/README.md)). Both drivers generate bit-identical
synthetic data from the same splitmix64 stream and train with matched
parameters. The table below preserves the original single-thread baseline;
rerun the commands for current multicore results. 100,000 rows x 100
features, 100 rounds, Apple M4:

| | mojoboost (1 thread) | LightGBM (1 thread) |
|---|---|---|
| Regression: training | 3.53 s | 2.41 s |
| Regression: binning | 0.55 s | 0.81 s |
| Regression: train MSE | 0.003615 | 0.003797 |
| Binary: training | 3.50 s | 2.32 s |
| Binary: train logloss | 0.267034 | 0.267168 |

The original implementation was within 1.5x of single-threaded LightGBM on
training and faster at binning, before multicore histogram accumulation.

```sh
pixi run bench                 # mojoboost
pixi run -e bench bench-lgbm --threads 1
pixi run bench-hist            # CPU/GPU histogram microbenchmark
```

The GPU microbenchmark separates first-use setup from repeated builds. It is
a kernel-development measurement, not an end-to-end GPU-training claim.

## License

Apache-2.0
