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
`force_row_wise=true`. The table below records the original single-threaded
mojoboost baseline. The current implementation parallelizes histogram
accumulation across features, so new runs should record available CPU cores
and compare against both 1-thread and machine-wide LightGBM.

## Running

```sh
pixi run bench                 # mojoboost, defaults: 100000 rows x 100 features, reg
pixi run bench 100000 100 binary
pixi run -e bench bench-lgbm --rows 100000 --features 100 --objective reg --threads 1
pixi run bench-hist
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

Headline for this original baseline: a from-scratch Mojo GBDT is within 1.5x
of single-threaded LightGBM on training, faster at binning, and matches its
accuracy, before multicore histogram accumulation and without GOSS or EFB.

## CPU/GPU histogram microbenchmark

`bench_histogram.mojo` compares the multicore CPU implementation with the
experimental portable GPU kernel. It separates first-use setup from repeated
device-resident builds. This is not an end-to-end GPU-training benchmark.

Results from one GPU family must not be presented as representative of other
devices. Record the accelerator, Mojo/MAX version, dataset dimensions, and
repetition count with every result.

## Per-device GPU validation report

`bench_gpu_validation.mojo` is the cross-vendor driver. It prints device
identity and capability attributes once, then per dataset shape the launch
geometry the histogram kernel uses and a phase breakdown: binning, setup,
gradient upload, partition kernel, per-node histogram, a same-size
device-to-host probe, and complete training on both backends with training
MSE for each.

```sh
pixi run gpu-validate                  # built-in four-shape sweep
pixi run gpu-validate 20 250000 200    # rounds, then (rows, features) pairs
```

The phases are host-visible wall clock. The exact kernel-versus-transfer
split comes from the vendor profiler, and
[docs/GPU_VALIDATION.md](../docs/GPU_VALIDATION.md) has those commands, the
full procedure, and the record of which devices have actually been run. As of
this writing that record is Apple Metal only: no NVIDIA or AMD device has
executed this code, so no CUDA or HIP number should appear anywhere.

## Custom objectives

Two drivers, because the two paths cost different things.

`bench_custom_objective.mojo` times the native Mojo interface: the built-in
`SQUARED_ERROR` objective, the same derivatives through `train_custom`, and
the same again through a closure that captures state. All three grow
identical trees (the custom runs start from the label mean), and the driver
checks that before reporting, so what is timed is only the objective
plumbing.

```sh
pixi run bench-custom                     # 100000 rows x 20 features, 100 rounds
pixi run bench-custom 250000 40 100
```

`bench_custom_objective.py` times the Python callback path: the same fit
with `objective="regression"` and with a Python callable computing the same
derivatives. It refuses to report unless both fits produced identical
predictions.

```sh
pixi run -e bench bench-custom-py
pixi run -e bench bench-custom-py --rows 250000 --rounds 100 --repeat 3
```

Measured on Apple M4, Mojo 1.0.0, 100 rounds, best of 3:

| rows x features | built-in | Python callback | overhead per round |
|---|---|---|---|
| 100,000 x 20 | 2.49 s | 3.38 s | 8.9 ms (+36%) |
| 20,000 x 10 (30 rounds) | 0.118 s | 0.142 s | 0.81 ms (+20%) |

The overhead is one Python call per round plus a copy of the raw scores out
and the gradients and hessians back, so it scales with rows, not with tree
work. Quote it per round rather than as a percentage: the percentage falls
as `num_leaves`, features, or bins grow, and rises as they shrink.

## Caveats

- LightGBM's `min_data_in_bin=3` (its default) has no mojoboost equivalent;
  both still fit 255-bin quantile histograms.
- Losses differ slightly because tree growth diverges after the first
  floating-point tie; the ~4% MSE gap in mojoboost's favor is within
  seed-to-seed variation, not a quality claim.
- Single machine, single run; rerun both commands back to back on an idle
  machine before quoting numbers.
