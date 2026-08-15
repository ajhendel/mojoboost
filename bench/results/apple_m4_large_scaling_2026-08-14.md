# Apple M4 large-data scaling, 2026-08-14

Raw benchmark record for mojotrees CPU, mojotrees Metal, and LightGBM CPU on
the same deterministic synthetic regression problem. This is a development
result, not a production-performance claim.

## Environment

- Apple M4, 10 physical CPU cores, 10 GPU multiprocessors, 16 GB memory
- macOS 26.5.2 (25F84), arm64
- Mojo 1.0.0 (ed45d567), MAX 26.5.0
- LightGBM 4.7.0 from conda-forge
- 100 trees, 31 leaves, learning rate 0.1, 255 bins, L2 regularization 1.0
- 50 dense Float64 input features; training loss is measured after timing
- Initial load average: 1.33, 1.42, 1.36
- No benchmark arms ran concurrently

The seed is an offset into the shared splitmix64 counter stream. Both drivers
therefore receive bit-identical features and labels for a given seed. Seed 0
is backward-compatible with the original benchmark dataset.

Commands:

```sh
pixi run bench-train-gpu ROWS 50 reg REPEATS cpu,gpu SEED
pixi run -e bench bench-lgbm --rows ROWS --features 50 \
  --objective reg --threads THREADS --seed SEED
```

`/usr/bin/time -l` wrapped every process to record maximum resident set size.

## 1,000,000 rows x 50 features

The mojotrees driver used three alternating CPU/GPU repeats for every seed;
its table reports the minimum, as specified by the driver. LightGBM ran once
per seed and thread count.

| seed | MB bin | MB CPU train | MB GPU train | MB CPU loss | MB GPU loss | LGBM 1t bin | LGBM 1t train | LGBM 10t bin | LGBM 10t train | LGBM loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.173163 | 11.094292 | 4.288977 | 0.0035182734 | 0.0034559283 | 1.181759 | 8.985073 | 0.316469 | 2.701332 | 0.0034900988 |
| 1 | 0.163968 | 11.345849 | 4.381703 | 0.0035086437 | 0.0034953874 | 1.220286 | 8.908440 | 0.302864 | 2.802940 | 0.0035291874 |
| 2 | 0.164192 | 11.706220 | 4.377537 | 0.0035675740 | 0.0036006199 | 1.205639 | 8.902984 | 0.360753 | 2.950542 | 0.0034522718 |

Times are seconds. Across seed-level values:

- mojotrees GPU training: 4.289-4.382 s, mean 4.349 s
- mojotrees CPU training: 11.094-11.706 s, mean 11.382 s
- LightGBM 1-thread training: 8.903-8.985 s, mean 8.932 s
- LightGBM 10-thread training: 2.701-2.951 s, mean 2.818 s
- Metal is 2.62x faster than mojotrees CPU and 2.05x faster than LightGBM
  1-thread on the means
- Metal is 54% slower than LightGBM 10-thread on training and 44% slower on
  binning plus training
- observed process maximum RSS: mojotrees 0.84-0.96 GB; LightGBM about 1.66 GB

## 5,000,000 rows x 50 features

Seed 0 used three alternating mojotrees CPU/GPU repeats. Seeds 1 and 2 used
one repeat because the complete pair takes about two minutes; the driver
correctly marks their within-seed deltas as unresolvable. Treat the
cross-seed range as replication across datasets, not as a substitute for
within-seed timing repeats. LightGBM ran once per seed and thread count.

| seed | MB bin | MB CPU train | MB GPU train | MB CPU loss | MB GPU loss | LGBM 1t bin | LGBM 1t train | LGBM 10t bin | LGBM 10t train | LGBM loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.027547 | 55.137767 | 17.483050 | 0.0034420427 | 0.0034532385 | 3.402448 | 45.264195 | 0.852435 | 12.216644 | 0.0034651964 |
| 1 | 0.807454 | 57.649708 | 16.684288 | 0.0033948420 | 0.0035449161 | 3.348532 | 45.128597 | 0.847395 | 12.253733 | 0.0034607956 |
| 2 | 0.841680 | 57.918384 | 17.318753 | 0.0033966929 | 0.0034438549 | 3.294226 | 46.486568 | 0.932349 | 12.916045 | 0.0033832411 |

Times are seconds. Across seed-level values:

- mojotrees GPU training: 16.684-17.483 s, mean 17.162 s
- mojotrees CPU training: 55.138-57.918 s, mean 56.902 s
- LightGBM 1-thread training: 45.129-46.487 s, mean 45.626 s
- LightGBM 10-thread training: 12.217-12.916 s, mean 12.462 s
- Metal is 3.32x faster than mojotrees CPU and 2.66x faster than LightGBM
  1-thread on the means
- Metal is 38% slower than LightGBM 10-thread on training and 35% slower on
  binning plus training
- observed process maximum RSS: mojotrees 2.88-3.04 GB; LightGBM 6.02-6.50 GB

The RSS comparison is process-level, not an allocator-normalized library
measurement. In particular, the Python LightGBM driver materializes NumPy
arrays and LightGBM's dataset in one process, while the mojotrees driver runs
both training arms over one Mojo-owned dataset. It is evidence of this
harness's peak memory, not proof that the libraries themselves differ by the
same factor in an application.

## Reading the result

On this workload, Metal scales better than mojotrees CPU as row count grows
and beats single-threaded LightGBM decisively. It does not beat LightGBM using
all ten M4 CPU cores. The mean Metal training deficit against 10-thread
LightGBM narrows from 54% at one million rows to 38% at five million rows.
Training losses are of the same order but are not identical models or a
held-out accuracy comparison.

The recent packaging, project-link, and release-hardening commits do not touch
the training hot path. These measurements do not show a regression caused by
that work. They establish a dated baseline for the next profiling and hybrid
CPU/GPU optimization pass.
