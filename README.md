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
- L1 and L2 split regularization (`lambda_l1`, `lambda_l2`) with LightGBM's
  soft-thresholding of gradient sums, applied consistently to split gains
  and leaf values on both the CPU and GPU trainers
- objectives: squared error, binary logistic, poisson, huber, quantile,
  L1 (mean absolute error), and multiclass softmax; quantile and L1
  renew leaf values with residual percentiles the way LightGBM does, and
  `alpha` follows LightGBM's meaning for huber and quantile
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
- one device vocabulary across Mojo and Python (`cpu`, `gpu`, `auto`),
  where `gpu` raises rather than falling back silently (see
  [Device selection](#device-selection))
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
sequences work without it. The regressor takes LightGBM objective names
(`objective="regression"`, `"huber"`, `"quantile"`, or `"mae"`) with
`alpha` as the quantile level or huber transition point. Both estimators
take `device` and record the backend that ran on `device_`; see
[Device selection](#device-selection).

`pixi run test-wheel` builds a self-contained wheel (`pixi run build-wheel`)
and validates it in a clean venv. The wheel bundles the Mojo runtime
dylibs the extension links (delocate-style, with an `@loader_path` rpath),
so installing it requires no Mojo or MAX toolchain. Wheels currently
target macOS on Apple silicon; Linux wheels need a manylinux build.

## Device selection

The same three values everywhere, in Mojo (`fit(..., device=CPU_DEVICE)`)
and in Python (`MojoBoostRegressor(device="cpu")`):

| `device` | Behavior |
|---|---|
| `cpu` | The default and the dependable backend: Float64, every objective, every entry point |
| `gpu` | Device-resident tree growth. Raises when no accelerator is present, or on a workload the GPU path does not cover, instead of falling back |
| `auto` | The GPU when it is available, covers the workload, and the size heuristic selects it; the CPU otherwise |

The GPU path covers single-output training (squared error, binary
logistic, poisson, huber, quantile, L1). Multiclass grows one tree per
class per round on the CPU only, so `gpu` raises for it and `auto`
chooses the CPU.

`auto`'s size heuristic ships disabled, so `auto` currently always
resolves to the CPU. No benchmark on any device has established a
workload size where end-to-end GPU training beats the CPU trainer, and
shipping a crossover threshold before then would be a performance claim
with nothing behind it. `MOJOBOOST_AUTO_MIN_CELLS` enables it as an
integer cell count (`n_rows * n_features`) at or above which `auto`
chooses the GPU, which is the knob for running the crossover benchmark
that would justify a default. `MOJOBOOST_DISABLE_GPU=1` makes the library
report no accelerator, so `gpu` raises and `auto` chooses the CPU on a
machine that has one; it pins a mixed fleet to the CPU and exercises the
unavailable-GPU path in tests.

Fitted Python estimators record the backend that actually ran on
`device_`. The device is a training choice rather than part of the model,
so the serialization format is unchanged and a loaded estimator carries no
`device_`.

LightGBM difference: LightGBM spells this `device_type` with `cpu`, `gpu`,
and `cuda`, and has no `auto`. mojoboost has one portable GPU backend
rather than separate OpenCL and CUDA ones, so `gpu` covers every supported
accelerator, and `auto` is an addition.

## Roadmap

1. Integrate the GPU histogram backend into end-to-end training while keeping
   intermediate state device-resident
2. Scale GPU histograms beyond one threadgroup per feature and validate on
   Apple, NVIDIA, and AMD hardware
3. Publish the Python API to PyPI (macOS arm64 wheels build and validate
   today; Linux needs a manylinux build)
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
| `lambda_l1` | 0.0 (LightGBM's default) |

### L1 regularization

`lambda_l1` follows LightGBM's `ThresholdL1`: every gradient sum is shrunk
toward zero by `lambda_l1` and clamped there,

    T(G) = sign(G) * max(0, |G| - lambda_l1)

and `T` is applied to the parent, left, and right sums of the split gain and
to the gradient sum of the Newton leaf value `-T(G) / (H + lambda_l2)`. A
leaf whose gradients all fall inside the threshold gets value zero, and a
split whose children both fall inside it has no gain and is never taken. The
penalty acts on absolute gradient sums, so sample weights scale what the
threshold removes: a leaf of heavily weighted rows keeps more of its
gradient than the same rows at weight one.

Two intentional consequences, both matching LightGBM:

- for the `mae`/`quantile` objectives, leaf values are replaced afterwards by
  residual percentiles (LightGBM's `RenewTreeOutput`), so `lambda_l1` shapes
  which splits are chosen but not the final leaf value
- `lambda_l1` is a training parameter only; it changes the trees, not the
  model format, so serialized models are unaffected

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
