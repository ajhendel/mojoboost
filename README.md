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

Early development. What works today, each piece with tests

- equal-width feature binning into `uint8` bins
- histogram accumulation (gradient sums, hessian sums, counts)
- best-split search with the standard second-order gain formula

## Roadmap

1. Leaf-wise tree growth with `num_leaves` cap and histogram subtraction trick
2. Quantile (equal-frequency) binning, LightGBM style
3. Boosting loop with logistic and squared-error objectives, early stopping
4. SIMD histogram kernels and multicore training
5. scikit-learn style `fit`/`predict` Python API via Mojo interop
6. Reproducible benchmark suite vs LightGBM and XGBoost (same defaults, same
   datasets, hardware documented)
7. GPU training

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
