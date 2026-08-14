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

## CPU stage profile

`bench_profile.mojo` is the driver for the CPU backend itself. It times every
stage a boosting round spends real work in, each one twice on identical data:
once with `MOJOBOOST_NUM_WORKERS=1` and once in auto mode. The ratio isolates
what the scheduler is worth for that stage; the serial column is what
single-core work moves.

| Stage | What it covers |
|---|---|
| `bin_fit` | quantile edge fitting per feature (sort-bound) |
| `bin_transform` | raw values to bin ids per feature (binary-search-bound) |
| `grad_hess` | per-row gradients and hessians (elementwise) |
| `hist_full` | root histogram over every row (scatter-bound) |
| `hist_subset50`, `hist_subset10` | child histograms over half and a tenth of the rows |
| `hist_subtract` | sibling histogram by subtraction (SIMD elementwise) |
| `split_scan` | best-split search over a built histogram |
| `partition` | routing a node's rows to its two children |
| `grow_tree` | one whole tree, as an integration check on the above |
| `predict` | scoring every row through a grown tree |

It prints a machine header first (ISA, core counts, SIMD width, and the
scheduler's resolved task counts), which must accompany any number taken from
it.

```sh
pixi run bench-profile                  # 100000 rows x 100 features, 3 reps
pixi run bench-profile 200000 50 5      # rows, features, reps
```

Sweeping `TASKS_PER_CORE` in `src/mojoboost/parallel.mojo` and rerunning this
is how that constant should be settled; it is currently an unmeasured starting
value of 4.

**No numbers are recorded here yet.** Every run taken during the work that
added this driver was on a machine carrying a load average above 8 on 10
cores, from concurrent Mojo builds, which makes the absolute times meaningless
and the ratios unreliable. Run it on an idle machine and add a table.

### How the scheduler decides

Both dispatch shapes in `parallel.mojo` take a work estimate in
*histogram-op equivalents*, one scattered read-modify-write of a gradient, a
hessian, and a count. Callers whose per-row work is cheaper or dearer than
that scale their estimate: gradient generation divides by 16 (a few flops on
sequential arrays), row partitioning multiplies by 3 (two passes, each an
indirect load). Task count then follows two rules, both in `plan_tasks`:

- below one grain (`MOJOBOOST_PARALLEL_MIN_OPS`, default 65536) of estimated
  work, stay serial;
- above it, never give a task less than a grain, and never exceed
  `TASKS_PER_CORE` tasks per physical core.

The grain cap is the rule that matters for cheap stages. Without it, 100k rows
of elementwise gradient work asked for one task per core, and the profiler
timed that fan-out well below the serial path.

`MOJOBOOST_NUM_WORKERS` overrides both rules, which is how the tests force a
particular path, and is what to pin when comparing runs.

## mojoboost vs LightGBM at matched thread counts

The comparison the CPU work is aimed at needs four runs, back to back on an
idle machine. `MOJOBOOST_NUM_WORKERS=1` is what makes the mojoboost side
genuinely single-threaded; without it, auto mode uses the machine.

```sh
MOJOBOOST_NUM_WORKERS=1 pixi run bench 100000 100 reg
pixi run -e bench bench-lgbm --rows 100000 --features 100 --objective reg --threads 1
pixi run bench 100000 100 reg                 # auto: machine-wide
pixi run -e bench bench-lgbm --rows 100000 --features 100 --objective reg --threads 10
```

Record physical core count with the result, and set the LightGBM thread count
to that same number rather than to the logical count. **No post-optimization
numbers are recorded here yet**, for the load reason above; the table under
Results is the pre-multicore single-threaded baseline and is labelled as such.

## CPU/GPU histogram microbenchmark

`bench_histogram.mojo` compares the multicore CPU implementation with the
experimental portable GPU kernel. It separates first-use setup from repeated
device-resident builds. This is not an end-to-end GPU-training benchmark.

Results from one GPU family must not be presented as representative of other
devices. Record the accelerator, Mojo/MAX version, dataset dimensions, and
repetition count with every result.

## GPU histogram scaling and phase breakdown

`bench_histogram_scaling.mojo` is the driver for the histogram kernel
itself. It runs the same data through both accumulation strategies,
`tiled` (per-threadgroup partials plus a deterministic reduction kernel) and
`atomic` (the preserved fallback), and reports the launch geometry each one
resolved to along with six phases timed separately:

| Phase | What it covers | How often it is paid |
|---|---|---|
| `setup_s` | context, allocations, binned matrix upload | once per dataset |
| `convert_s` | Float64 gradients to the device's Float32 | once per round |
| `upload_s` | that staged pair, host to device | once per round |
| `kernel_root_s`, `kernel_leaf_s` | histogram kernels only | once per node |
| `download_s` | fixed-point histogram, device to host | once per node |
| `back_convert_s` | those integers to the Float64 histogram | once per node |

Kernel and transfer phases are timed after a warm-up build, so shader or PTX
compilation never lands inside a reported number. The run also checks that
the two strategies produced bit-identical histograms and prints the verdict,
so a timing comparison is never read apart from correctness.

```sh
pixi run bench-hist-scaling                    # small, medium, large shapes
pixi run bench-hist-scaling 20 1000000 50      # reps, then (rows, features)
```

`MOJOBOOST_GPU_HIST_STRATEGY`, `MOJOBOOST_GPU_ROW_TILE`, and
`MOJOBOOST_GPU_BLOCK_THREADS` (see `src/mojoboost/gpu_tiling.mojo`) sweep the
tiling by hand, which is how the defaults in that module were chosen. Pin
them when comparing runs.

The same caveat as above applies with more force here: the tiling is derived
from the device's own reported capabilities, so numbers from one GPU say
nothing about another. Record the device, driver, and Mojo/MAX version.

### Results

Apple M4 (10 multiprocessors), Mojo 1.0.0, 255 bins, `reps=30`. Kernel times
are milliseconds, `best` of 30 interleaved repetitions. **The machine was not
idle for this run**: other Mojo builds were running throughout, which is why
only `best` is quoted. Rerun on an idle machine before quoting these
anywhere else.

| shape | tiles | groups | root: atomic / tiled | leaf: atomic / tiled |
|---|---|---|---|---|
| 20,000 x 20 | 4 | 80 | 0.56 / 0.75 | 0.64 / 0.58 |
| 200,000 x 100 | 1 | 100 | 4.87 / 4.80 | 5.31 / 4.97 |
| 1,000,000 x 50 | 2 | 100 | 13.08 / 10.48 | 8.81 / 6.75 |

Both strategies were bit-identical on every shape, root and leaf.

Reading it: the tiled path wins where it was built to win, the large shape
(20% on the root build, 23% on a leaf build), and on leaf builds generally,
which is where a tree spends `num_leaves - 1` of its `num_leaves`
histograms. It loses about 0.2 ms on the small root build, which is the
second kernel launch showing through when the whole build is under a
millisecond. That cost is paid once per tree against a saving paid on every
node after the root.

Transfers and conversion on the same run, for scale: gradient upload
0.5 ms to 1.5 ms, histogram download 0.3 ms to 0.7 ms, Float64-to-Float32
conversion 0.2 ms to 5.6 ms (it scales with rows, not with the histogram),
and the fixed-point-to-Float64 conversion under 0.7 ms. One-time setup, which
is dominated by the binned matrix upload, ran 0.14 s at 20,000 x 20 and
1.6 s to 1.9 s at 1,000,000 x 50. Binning itself, on the CPU, cost 11.4 s at
the largest shape and dwarfs everything the GPU does.

Not yet measured: a sweep of `MOJOBOOST_GPU_ROW_TILE` at the largest shape,
which is what would confirm or move `TARGET_BLOCKS_PER_SM` in
`gpu_tiling.mojo`. The default of 8 threadgroups per multiprocessor gives
only 2 tiles and 100 threadgroups at 1,000,000 x 50 on this device, and
whether more tiles would help there is an open question, not a settled one.

## What a launch and a wait cost

`bench_launch_cost.mojo` measures the two fixed costs the GPU trainer pays
per split, with an empty kernel so the work is excluded: submitting one
kernel, and one round trip to the device and back. Read it before proposing
any change whose whole benefit is fewer kernel launches or fewer
synchronizations, because those two numbers set the price of the change and
they are a property of the device, not of this repository.

```sh
pixi run bench-launch-cost             # 200 launches per sample, 5 trials
pixi run bench-launch-cost 500 5
```

Both arms alternate inside one process, as `bench_train_gpu.mojo` does, and
both are warmed first: an unwarmed round-trip arm reads about 60% high. The
summary prices a device-resident split (eight launches and one wait) against
the result, and states what one launch removed is worth over a default
100-round, 31-leaf run.

On an Apple M4 that is roughly 20us to submit a launch and 126us for the
wait, so about 280us a split, or 0.85s of a 3.05s run at 50,000 x 100 --
about a third of the device path is fixed cost, and one launch removed is
worth near 2%, which is inside `bench_train_gpu.mojo`'s noise floor. The
consequence is recorded in `_device_search_resident` in `train_gpu.mojo`: a
kernel fusion here has to justify itself as strictly less work for a
bit-identical result, not by a measured speedup. Numbers from one GPU family
say nothing about another; rerun this before quoting it elsewhere.

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

Python callback path, measured with that driver on Apple M4, Mojo 1.0.0,
best of 3 (no native-interface numbers are recorded here yet; run
`pixi run bench-custom` and add them with the machine):

| rows x features | built-in | Python callback | overhead per round |
|---|---|---|---|
| 100,000 x 20 | 2.49 s | 3.38 s | 8.9 ms (+36%) |
| 20,000 x 10 (30 rounds) | 0.118 s | 0.142 s | 0.81 ms (+20%) |

**The machine was not idle for this run**: other Mojo builds were running
throughout. Both fits in a pair ran back to back under the same load and the
overhead is a difference between them, so the ratio is more trustworthy than
either absolute time; rerun on an idle machine before quoting these anywhere
else.

The overhead is one Python call per round plus a copy of the raw scores out
and the gradients and hessians back, so it scales with rows, not with tree
work. Quote it per round rather than as a percentage: the percentage falls
as `num_leaves`, features, or bins grow, and rises as they shrink.

## GOSS

`bench_goss.mojo` trains one dataset twice with identical parameters, once
on every row and once with Gradient-based One-Side Sampling, and reports
both wall times and both training losses. GOSS trades accuracy for speed,
so the driver prints both halves of the trade: a speedup and a loss ratio.
A speedup with a materially worse loss is not a win, and neither number
means anything without the other.

```sh
pixi run bench-goss                        # 100000 x 100, reg, 0.2 / 0.1
pixi run bench-goss 100000 100 binary
pixi run bench-goss 200000 50 reg 0.1 0.1  # rates as the last two arguments
```

Two things to keep in mind when reading a run:

- LightGBM skips sampling for the first `int(1 / learning_rate)` rounds, so
  at the default learning rate the first 10 of 100 rounds train on every
  row. The driver prints that count; a short run spends a large share of
  its rounds on full data and will show less speedup than a long one.
- The sampled run keeps `top_rate + other_rate` of the rows, so the
  expected upper bound on the speedup of the histogram work is roughly
  `1 / (top_rate + other_rate)` for the sampled rounds only. Binning, score
  updates, and prediction are not sampled.

**No numbers are recorded here yet.** This machine has been running
concurrent Mojo builds throughout the work that added GOSS, which makes any
timing taken now meaningless. Run both commands back to back on an idle
machine, record the machine, Mojo version, dimensions, and rates, and add a
row here.

## Learning to rank

`compare_ranking.py` is a correctness comparison, not a timing one. It
builds synthetic queries, splits them into train and validation by query,
and reports two things:

- the NDCG metric cross-check: mojoboost's `ndcg_score` applied to
  LightGBM's own validation predictions, against the `ndcg@k` LightGBM
  reports for those same scores. Same scores, same labels, same
  boundaries, so a disagreement is a disagreement about the metric. Expect
  around 1e-9, which is Mojo's `log2` error in the position discounts, not
  a difference in definition.
- validation NDCG at several cutoffs for two independently trained models.
  These are not expected to match: tree growth diverges after the first
  floating-point tie and LightGBM reads its pairwise sigmoid from a lookup
  table. Comparable NDCG is the claim.

```sh
pixi run -e bench compare-ranking
pixi run -e bench compare-ranking --queries 2000 --rounds 200
```

## Missing-value semantics against LightGBM

`compare_missing_lightgbm.py` is the missing-value comparison. It trains both
libraries on four probe datasets built from the same closed form of the row
index (no file is exchanged) and prints their routing decisions side by side:
whether a missing bin is reserved, whether the root isolates the missing rows,
the node's default direction, what a `NaN` predicts on a model trained without
any, `use_missing=false`, and where the infinities bin.

```sh
pixi run -e bench compare-missing
```

As of LightGBM 4.7 every routing decision matches. Leaf values differ by the
factor `H / (H + lambda_l2)`, because mojoboost defaults `lambda_l2` to 1.0
where LightGBM defaults it to 0.0; the script prints those two rows for
context and says so rather than counting them as disagreements.

## Sparse vs dense

`bench_sparse.mojo` runs both paths over the same genuinely sparse dataset,
generated from the same counter-based splitmix64 stream, and reports the
exact size of the arrays each path has to hold along with binning, transform,
and training time.

```sh
pixi run bench-sparse                  # 200,000 x 500, 10 nonzeros per row
pixi run mojo run -I src bench/bench_sparse.mojo 100000 500 5
```

The dense matrix is materialized only so the dense path has something to run
on; a real sparse workload never builds it, which is what the memory column
is measuring. Two configurations on an Apple M4, 50 rounds, `num_leaves=31`,
`max_bin=255`:

| | 50,000 x 300, 2.7% dense | 100,000 x 500, 1.0% dense |
|---|---|---|
| nonzeros | 400,000 | 500,000 |
| dense memory (raw + bins) | 128.7 MB | 429.2 MB |
| sparse memory (CSC + bins + growth index) | 15.6 MB | 19.6 MB |
| sparse / dense memory | 0.12x | 0.046x |
| dense total time | 2.73 s | 6.34 s |
| sparse total time | 1.55 s | 3.31 s |
| sparse speedup | 1.76x | 1.92x |
| train MSE, both paths | 0.0023802419115210 | 0.0052833930695555 |

Both paths reported the same training MSE to every printed digit in these
runs, which is the correctness half of the benchmark: an absent entry is a
numerical zero, so the two are fitting the same model.

The memory figures are exact array sizes computed from the shapes, not a
process-level measurement, and they include the O(nnz) entry permutation and
scratch buffer that sparse tree growth needs and the dense path does not.
They are deterministic: the same shapes always give the same numbers.

The times are not. Each column is one run on one machine, and a repeat of
the first column on the same machine gave 2.22 s dense against 1.28 s
sparse, a 1.74x speedup rather than 1.76x. Treat the ratio as the
measurement and the absolute seconds as the conditions it was taken under,
and rerun both before quoting any of it.

The speedup shrinks as density rises and reverses eventually: the sparse
accumulator costs O(nnz_in_node) per node against the dense O(rows * features),
so it wins while `density * n_features` stays below the dense per-row cost.
Neither path is the right default for the other's shape, which is why both
exist.

## Caveats

- LightGBM's `min_data_in_bin=3` (its default) has no mojoboost equivalent;
  both still fit 255-bin quantile histograms.
- Losses differ slightly because tree growth diverges after the first
  floating-point tie; the ~4% MSE gap in mojoboost's favor is within
  seed-to-seed variation, not a quality claim.
- Single machine, single run; rerun both commands back to back on an idle
  machine before quoting numbers.
- The sparse table above was measured on synthetic data with a fixed number
  of nonzeros per row. Real sparse datasets have skewed column densities,
  which changes the per-feature work distribution and so the speedup.
