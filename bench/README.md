# Benchmarks

Reproducible mojoboost vs LightGBM comparison on identical synthetic data.

Both drivers generate the dataset from the same counter-based splitmix64
stream, so every feature value and label is bit-identical between the two
(no files exchanged, no RNG library differences). The target mixes linear,
interaction, and quadratic terms of the first four features plus uniform
noise; the remaining features are pure noise.

Parameters match on both sides (mojoboost defaults): 100 boosting rounds,
`num_leaves=31`, `learning_rate=0.1`, `min_data_in_leaf=20`,
`min_sum_hessian_in_leaf=1e-3`, `lambda_l2=1.0`, `max_bin=255`. LightGBM
additionally runs with `enable_bundle=false` (mojoboost has no EFB yet) and
`force_row_wise=true`. mojoboost is single-threaded, so the 1-thread
LightGBM column is the apples-to-apples one.

## Running

```sh
pixi run bench                 # mojoboost, defaults: 100000 rows x 100 features, reg
pixi run bench 100000 100 binary
pixi run -e bench bench-lgbm --rows 100000 --features 100 --objective reg --threads 1
```

## Results

100,000 rows x 100 features, 100 rounds. Apple M4, Mojo 1.0.0,
LightGBM 4.x via conda-forge. Times in seconds (binning + training reported
separately; loss is on the training set).

| | mojoboost (1 thread) | LightGBM (1 thread) | LightGBM (10 threads) |
|---|---|---|---|
| Regression: binning | 0.55 | 0.81 | 0.20 |
| Regression: training | 3.53 | 2.41 | 1.06 |
| Regression: train MSE | 0.003615 | 0.003797 | 0.003797 |
| Binary: binning | 0.54 | 0.80 | |
| Binary: training | 3.50 | 2.32 | |
| Binary: train logloss | 0.267034 | 0.267168 | |

Headline: a from-scratch Mojo GBDT is within 1.5x of single-threaded
LightGBM on training, faster at binning, and matches its accuracy, with no
multithreading, GOSS, or EFB yet.

## Caveats

- LightGBM's `min_data_in_bin=3` (its default) has no mojoboost equivalent;
  both still fit 255-bin quantile histograms.
- Losses differ slightly because tree growth diverges after the first
  floating-point tie; the ~4% MSE gap in mojoboost's favor is within
  seed-to-seed variation, not a quality claim.
- Single machine, single run; rerun both commands back to back on an idle
  machine before quoting numbers.
