# Benchmarks

Reproducible mojotrees vs LightGBM comparison on identical synthetic data.

Every `pixi run` command in this file names a task that exists. Which of them
have ever produced a recorded number is a different question, and it is answered
per driver in [`results/INSTRUCTION_AUDIT.md`](results/INSTRUCTION_AUDIT.md).
Several sections below say "no numbers are recorded here yet" and mean it.

Both drivers generate the dataset from the same counter-based splitmix64
stream, so every feature value and label is bit-identical between the two
(no files exchanged, no RNG library differences). The target mixes linear,
interaction, and quadratic terms of the first four features plus uniform
noise; the remaining features are pure noise.

Parameters match on both sides (mojotrees defaults): 100 boosting rounds,
`num_leaves=31`, `learning_rate=0.1`, `min_data_in_leaf=20`,
`min_sum_hessian_in_leaf=1e-3`, `lambda_l2=1.0`, `max_bin=255`. LightGBM
additionally runs with `enable_bundle=false` (mojotrees has no EFB yet) and
`force_row_wise=true`. The table below records the original single-threaded
mojotrees baseline. The current implementation parallelizes histogram
accumulation across features, so new runs should record available CPU cores
and compare against both 1-thread and machine-wide LightGBM.

## Running

```sh
pixi run bench                 # mojotrees, defaults: 100000 rows x 100 features, reg
pixi run bench 100000 100 binary
pixi run bench 100000 100 reg 1             # final argument is the data seed
pixi run -e bench bench-lgbm --rows 100000 --features 100 --objective reg --threads 1
pixi run -e bench bench-lgbm --rows 100000 --features 100 --objective reg --threads 1 --seed 1
pixi run bench-hist
```

## Results

100,000 rows x 100 features, 100 rounds. Apple M4, Mojo 1.0.0,
LightGBM 4.x via conda-forge. Times in seconds (binning + training reported
separately; loss is on the training set).

| | mojotrees (1 thread) | LightGBM (1 thread) | LightGBM (10 threads) |
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

**Do not carry that 1.5x anywhere else.** The serial ratio has since been
measured at a different shape and it is worse: at 1,000,000 x 50 we are
15.96s at one worker against LightGBM's 8.82s at one thread, so **1.81x**.
The same record also shows we get 2.29x from ten cores where LightGBM gets
3.08x, so the inner loop and the parallel scaling are both behind, and
either one alone is an incomplete account.

The current comparison is
[`results/sweep2_2026-08-15/RESULTS.md`](results/sweep2_2026-08-15/RESULTS.md),
with
[`results/profile_2026-08-15/RESULTS.md`](results/profile_2026-08-15/RESULTS.md)
behind it for the shapes the sweep did not repeat. Between them they
supersede both this baseline and
[`results/apple_m4_large_scaling_2026-08-14.md`](results/apple_m4_large_scaling_2026-08-14.md)
for anything quoted as current. In seconds of training with binning
excluded, 100 rounds, 31 leaves, 255 bins, squared error, Apple M4, arms
interleaved, **measured** as the median of five repeats:

| shape | our CPU | our GPU, leaf-wise | our GPU, depth-wise | LightGBM, 10 threads |
|---|---|---|---|---|
| 250,000 x 50 | 1.649 | 1.967 | 1.909 | **1.023** |
| 1,000,000 x 50 | 5.942 | 3.756 | **2.587** | 2.767 |
| 2,000,000 x 50 | 13.483 | 6.093 | 5.417 | **5.228** |

Spreads: our GPU 1.1 / 1.8 / 12.5 percent, depth-wise 6.7 / 0.3 / 10.8, our
CPU 9.3 / 13.5 / 35.6. The two-million-row CPU arm is too noisy to carry a
verdict.

Three readings, and the order matters.

**The depth-wise GPU arm beats LightGBM at 1,000,000 x 50**, 2.587 against
2.767, which is 6.5 percent faster at a 0.3 percent spread, the tightest arm
in the sweep. It is 3.6 percent behind at 2,000,000, which is parity. That is
the first measured win this project has against LightGBM on training speed at
a large shape, and four conditions travel with it: it grows a **different
tree** than LightGBM's leaf-wise growth, **no training loss was recorded for
any arm of this sweep** so its accuracy against that leaf-wise model is
unmeasured, it is one machine and one generator, and **the spreads quoted
above are ours alone**. At the time of the sweep, `bench_lightgbm.py` trained
once in a separate process with no repeat loop and no median, so every
LightGBM cell in this table is a single sample whose noise floor is unknown,
on a machine this file elsewhere documents as drifting by a factor of two to
three across time windows. Our own leaf-wise arm is still behind LightGBM
everywhere: 1.92x, 1.36x, 1.17x.

**That last condition now has a fix and the table has not yet been retaken
under it.** `bench_train_gpu.mojo` has a `lightgbm` arm that runs inside the
interleaved loop, so the comparator gets the same repeat count, the same
minimum-with-spread reduction, and the same resolved-versus-indistinguishable
verdict as every other arm; see "Interleaved LightGBM arm" below. Until the
1,000,000-row row is retaken that way, the 6.5 percent margin is a comparison
against an unrepeated single sample and should be quoted as one.

The arm now runs end to end and reports what the margin was missing. Per
arm, LightGBM included, it prints `<arm>_train_s_samples` — every repeat in
the order it ran — beside `<arm>_spread_pct` and a second
`<arm>_spread_pct_of_median`, and it writes the same lists into a one-line
`json_summary` record (also to a file under `MOJOTREES_BENCH_JSON`). So the
noise floor the 6.5 percent has to clear is now a quantity the run produces
rather than one nobody has. **No LightGBM spread has been measured yet.**
Nothing in this file was retaken for this change and no number above moved:
the arm was exercised only at a toy shape to prove it executes, on a machine
that was busy at the time, and a toy shape on a busy machine is not a
measurement. What exists is the instrument. The command that produces the
figure is the first one under "Interleaved LightGBM arm", and until it has
been run on a quiet box the 1,000,000-row cell stands exactly as written
above.

**Our marginal cost per row now equals LightGBM's on ten CPU cores.**
**Fitted** from the measured points, ours is 2.385 then 2.337 microseconds
per row across the two segments and LightGBM's is 2.325 then 2.461. The four
slopes interleave. **Derived** from those fits, our intercept is about 1.42
seconds against LightGBM's 0.44, so roughly one second of fixed cost is the
whole remaining deficit.

**That one second is worth less than it looks.** Removing all of it puts
1,000,000 rows near 2.8 seconds, which is parity rather than a win
(**estimated**). The margin has to come from the histogram kernel, and
depth-wise beat parity precisely because level-batched histograms move the
slope and not only the intercept.

None of that replaces the 100,000-row baseline, because the shapes and the
implementation revisions are different; it replaces any extrapolation from
it.

The loss columns in the baseline table above have no counterpart in the
sweep. `bench_train_gpu.mojo` prints `<arm>_train_loss` on the first repeat
and `bench_lightgbm.py` prints `train_mse`, and the committed sweep record
kept only the timing lines. Any rerun of these arms should keep the loss
lines, because a speed table with no loss beside it cannot answer the only
question a growth-policy change raises.

## CPU stage profile

`bench_profile.mojo` is the driver for the CPU backend itself. It times every
stage a boosting round spends real work in, each one twice on identical data:
once with `MOJOTREES_NUM_WORKERS=1` and once in auto mode. The ratio isolates
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

Sweeping the tasks-per-core constant and rerunning this is how it should be
settled; it is currently an unmeasured starting value of 4. **This no longer
means editing the source.** `MOJOTREES_CPU_TASKS_PER_CORE` overrides
`TASKS_PER_CORE` at runtime (`src/mojotrees/apple_cpu_policy.mojo`, and
`src/mojotrees/parallel.mojo:172` names it for exactly this purpose), so the
sweep is a loop over the environment variable rather than a rebuild per point,
which also makes the points comparable on a machine that drifts.

Nobody has run that sweep. The constant is still whatever
`DEFAULT_TASKS_PER_CORE` says.

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

- below one grain (`MOJOTREES_PARALLEL_MIN_OPS`, default 65536) of estimated
  work, stay serial;
- above it, never give a task less than a grain, and never exceed
  `TASKS_PER_CORE` tasks per physical core.

The grain cap is the rule that matters for cheap stages. Without it, 100k rows
of elementwise gradient work asked for one task per core, and the profiler
timed that fan-out well below the serial path.

`MOJOTREES_NUM_WORKERS` overrides both rules, which is how the tests force a
particular path, and is what to pin when comparing runs.

## mojotrees vs LightGBM at matched thread counts

The comparison the CPU work is aimed at needs four runs, back to back on an
idle machine. `MOJOTREES_NUM_WORKERS=1` is what makes the mojotrees side
genuinely single-threaded; without it, auto mode uses the machine.

```sh
MOJOTREES_NUM_WORKERS=1 pixi run bench 100000 100 reg
pixi run -e bench bench-lgbm --rows 100000 --features 100 --objective reg --threads 1
pixi run bench 100000 100 reg                 # auto: machine-wide
pixi run -e bench bench-lgbm --rows 100000 --features 100 --objective reg --threads 10
```

Record physical core count with the result, and set the LightGBM thread count
to that same number rather than to the logical count. **No post-optimization
numbers are recorded here yet**, for the load reason above; the table under
Results is the pre-multicore single-threaded baseline and is labelled as such.

Those four runs put the two engines in four separate processes at four
different moments, which is exactly the protocol this file forbids everywhere
else. They remain useful for a factor-of-two question and are useless for a
few-percent one. Use the interleaved arm below for anything narrower than the
machine's drift.

## Interleaved LightGBM arm

`lightgbm` is an arm of `bench_train_gpu.mojo` like `cpu` and `gpu-device`
are. It reaches LightGBM through Mojo's Python interop, so the comparator is
measured in the same process, in the same time window, alternating with the
mojotrees arms, under the same repeat count and the same reduction and the
same verdict rule. It needs the `bench` environment, which is where LightGBM
lives, and it is the same task with the environment named:

```sh
pixi run -e bench bench-train-gpu 1000000 50 reg 5 gpu-device-depth,lightgbm
MOJOTREES_LGBM_THREADS=10 pixi run -e bench bench-train-gpu 1000000 50 reg 5 cpu,lightgbm
```

The arm has been exercised end to end and it runs. That is a smoke test at a
toy shape on a loaded machine and **no timing was taken from it**; nothing in
this file changed as a result.

The arm prints `lightgbm_train_s_samples`, `lightgbm_train_s`,
`lightgbm_train_s_median`, `lightgbm_train_s_max`, `lightgbm_spread_pct`,
`lightgbm_spread_pct_of_median`, `lightgbm_n_trees` and
`lightgbm_train_loss` under the same names every other arm uses, plus
`lightgbm_threads`, `lightgbm_binning_s` and `lightgbm_params` once before
the loop. The loss is the mean squared residual for `reg` and the mean log
loss for `binary`, in the same definition `_train_loss` uses in the harness,
so the two sit in one column and are read against each other. Record the
`lightgbm_params` line with any result: it is the resolved parameter set the
run actually used rather than the one a second file says it should have.

### Per-repeat spread, for every arm

`<arm>_train_s_samples` holds one arm's repeats alone, in the order they ran,
and every arm gets one. The `run N <arm> train_s` lines above it already
carry the same numbers, but interleaved, so recovering a single arm's
dispersion from them means picking every n-th line out of a transcript by
hand — and a spread that is only recoverable by hand is a spread that gets
dropped when a result is copied into a table. That is not hypothetical here:
the 6.5 percent margin above was quoted beside a spread that described one
side only, because the other side had no repeat loop at all.

Two spreads are printed and they are not interchangeable.
`<arm>_spread_pct` is `(max - min) / min`, unchanged, and is the quantity the
`<arm>_vs_<baseline>` verdict is computed against, so figures recorded before
this line existed still mean what they meant. `<arm>_spread_pct_of_median` is
`(max - min) / median`: the arm's dispersion rather than its excursion above
its own best sample, and the one to quote beside a median. Neither is chosen
for the reader, because the choice belongs to whoever quotes the number.

The whole run also prints as one line of JSON, `json_summary:`, carrying the
per-arm sample lists, both spreads, the losses, the tree counts, the
comparisons with their verdicts, and LightGBM's resolved threads, binning
time and parameters. `MOJOTREES_BENCH_JSON=<path>` writes the same record to
a file. It is printed unconditionally rather than only on request, because
the transcript is the artifact that actually survives and the per-repeat
samples are the first thing a hand-copied summary loses.
`noise_floor_pct` is `-1` at one repeat, which is the null and not a floor of
zero.

```sh
MOJOTREES_BENCH_JSON=bench/results/lgbm_1m.json \
  pixi run -e bench bench-train-gpu 1000000 50 reg 5 gpu-device-depth,lightgbm
```

What this makes comparable is the time window, which was the hole. What it
does not:

- The two engines generate the dataset separately. Same counter-based
  splitmix64 sequence, bit-identical values, different code.
- Only the boosting run is inside the clock on either side. Our arms are
  handed an already binned matrix and LightGBM trains on an already
  constructed Dataset, with both binning times reported separately.
- Thread counts are matched by number, not by meaning.
  `MOJOTREES_LGBM_THREADS`, or `MOJOTREES_NUM_WORKERS` when it is unset,
  pins LightGBM's; nothing checks that the number buys the same amount of
  machine on both sides, which is why it is printed.
- Loading LightGBM changes the process it is loaded into, through the memory
  its regenerated feature matrix holds and the thread pool it parks between
  repeats. An arm timing from a run containing a `lightgbm` arm is not
  interchangeable with the same arm's timing from a run without one. Compare
  inside a run.
- The arm is single output, `reg` or `binary`. Multiclass would need the
  harness's quantile bucketing replicated on the Python side, and
  `lightgbm-depth` is refused because LightGBM has no depth-wise grow policy
  to select.

Two measurement changes ride along with the arm and both make LightGBM's
number smaller, so every LightGBM figure recorded before them reads slightly
slow and the two sets are not interchangeable to better than about a percent.
`lgb.train` is now called with `keep_training_booster=True`, which removes
the model serialize-and-reload round trip it otherwise performs inside the
timed call and which mojotrees pays no counterpart to. And the parameters now
come from `real_data/scenarios.py`, which adds the binning alignment the
standalone script was missing (`min_data_in_bin=1`, `feature_pre_filter=false`
and `bin_construct_sample_cnt` at the row count); `deterministic` is the one
entry deliberately not inherited, because it is a reproducibility setting
whose documented cost would land entirely on the comparator's side of a speed
comparison.

`bench-lgbm` itself now takes `--repeats` and reports minimum, median,
maximum and spread under the same reduction, so even the separate-process
form has a measured noise floor:

```sh
pixi run -e bench bench-lgbm --rows 1000000 --features 50 --threads 10 --repeats 5
```

## CPU/GPU histogram microbenchmark

`bench_histogram.mojo` compares the multicore CPU implementation with the
experimental portable GPU kernel. It separates first-use setup from repeated
device-resident builds. This is not an end-to-end GPU-training benchmark.

Results from one GPU family must not be presented as representative of other
devices. Record the accelerator, Mojo/MAX version, dataset dimensions, and
repetition count with every result.

### Launch-shape A/Bs

Two interleaved A/Bs run after the CPU/GPU comparison, both alternating
their arms inside one process because this machine's device timings drift
several-fold across time windows and only adjacent samples compare.

`feature_group_*` is how many feature slots one histogram threadgroup
accumulates, one against two.

`row_unroll_*` is how many rows one thread keeps in flight in the histogram
row loop: `HIST_ROW_UNROLL` (on, the default) against one row per iteration
(off), through `GpuActiveRows.set_row_unroll`. It prints
`row_unroll_on_samples_ms` and `row_unroll_off_samples_ms` — every repeat, in
order — then the minimum, the median, the speedup and a verdict against the
wider of the two arms' quiet bands.

**Neither knob can change a histogram, and the arm names should not be read
as if one might.** Both row-walk arms visit the same rows of the same range
and add the same fixed-point integers into the same bins; only the order of
the adds and of the loads feeding them differs, and integer addition is
associative and commutative. What differs is instruction count and how many
memory requests a thread has outstanding, against a higher live register
count and therefore a residency risk this backend does not let anyone query.
That is the open question and it is why the knob is a runtime argument
rather than a comptime one: a comptime knob would have forced a two-build
comparison, and two builds minutes apart cannot settle anything on this
machine.

**No row-unroll numbers are recorded here.** Both A/Bs compile and neither
has been run for a timing; they need a quiet box. A smoke run of the
end-to-end pair at 2,000 x 8 did have the two arms agree on the training loss
to the last digit and on the tree count, which is the "cannot change a model"
claim holding and is not a measurement of anything else.

The same A/B also runs **end to end**, as the `row-unroll-on` /
`row-unroll-off` arms of `bench_train_gpu.mojo`, once
`GpuHistogramBuilder.set_row_unroll` and `train_gpu`'s `row_unroll` argument
existed to carry it. Both benchmarks are kept and **both should be run**:

```sh
pixi run bench-hist 1000000 50 20                                        # isolated
pixi run bench-train-gpu 1000000 50 reg 5 row-unroll-on,row-unroll-off   # end to end
```

They answer different questions, and the end-to-end one is the question.
**An isolated histogram win is a hypothesis about a fit, not a result about
one.** The row-tile floor is why that sentence is here rather than in a
footnote: it measured well in isolation on this repo and turned out to be a
22 to 36 percent regression across a whole fit, because the isolated shape
did not carry the partial traffic a real round does. If the two disagree,
that disagreement is the finding and neither number supersedes the other.

Both end-to-end arms run under `SPLIT_SEARCH_AUTO`, so the pair holds every
condition but the knob constant. `gpu-unroll` / `gpu-nounroll` parse to the
same two arms under the same two labels, and any GPU arm takes a `-unroll` or
`-nounroll` suffix when the strategy needs pinning as well
(`gpu-device-nounroll`, `gpu-device-depth-nounroll`, in either suffix order).
`cpu-nounroll`, `lightgbm-nounroll` and any multiclass unroll arm are refused
rather than run, because in each case the knob reaches nothing and the arm
would print a number under a label it had not earned; `train_multiclass_gpu`
in particular takes no `row_unroll` argument, so both arms of the pair would
be the same run.

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

`MOJOTREES_GPU_HIST_STRATEGY`, `MOJOTREES_GPU_ROW_TILE`, and
`MOJOTREES_GPU_BLOCK_THREADS` (see `src/mojotrees/gpu_tiling.mojo`) sweep the
tiling by hand. Pin them when comparing runs.

**Only one of the three has settled a default.** `MOJOTREES_GPU_HIST_STRATEGY`
produced the `tiled` against `atomic` table below, which is a **measured**
result. `MOJOTREES_GPU_ROW_TILE` and `MOJOTREES_GPU_BLOCK_THREADS` have never
been swept and recorded — the "Not yet measured" note further down says so, and
`gpu_tiling.mojo` is explicit that `TARGET_BLOCKS_PER_SM = 8` rests on a
**derived bound** rather than a measurement: a small shared-memory footprint "is
evidence that 8 blocks are not excluded, not evidence that more than 8 are
resident". This paragraph used to claim all three knobs were how the module's
defaults were chosen, which contradicted that note twenty lines away.

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

Not yet measured: a sweep of `MOJOTREES_GPU_ROW_TILE` at the largest shape,
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
worth near 2%, which is inside `bench_train_gpu.mojo`'s noise floor.

Those three figures have since been confirmed by an independent instrument,
a Metal System Trace (`docs/METAL_TIMELINE.md`). The commit call measures
12.62us at the median on an instrumented process, so 20us is a sound upper
bound; the completion notification measures 101.33us against the 126us
here; and the median serialized turnaround, from the GPU finishing a
readback to the host committing the next command buffer, is 285us against
the 280us derived here. `bench/README.md` was also right to call these
properties of the device rather than of this repository: between a 50,000
and a 200,000 row capture the median enqueue moved from 12.67 to 12.62us
while the compute between them changed by a factor of 2.3.

One figure the trace does **not** confirm is "about a third of the device
path is fixed cost", and the difference is a denominator rather than an
error. The 280us here is a launch plus a wait with an empty kernel. What a
real blocking readback costs end to end, commit to next commit, is 606us at
the median, and 32 of those per round is 20.16ms of a 23.50ms round, or
85.8%. The extra 326us is mostly the readback queued behind compute the host
had already submitted, half of which is the pipeline working as intended and
half of which is the GPU idle with a command buffer it has not started. So
read 280us as the floor for one round trip and 606us as what one costs in
place.

The
consequence is recorded in `_device_search_resident` in `train_gpu.mojo`: a
kernel fusion here has to justify itself as strictly less work for a
bit-identical result, not by a measured speedup. Numbers from one GPU family
say nothing about another; rerun this before quoting it elsewhere.

## What a transfer costs on unified memory

`bench/apple/unified_memory.mojo` runs one bandwidth-bound checksum kernel
over the same payload through every host-to-device delivery route this Mojo
version exposes (pinned staged copy, heap copy, mapped write, host pointer
passed straight to the kernel) and one device-to-host route, and reports
per route where the time went, whether the device saw the right bytes, and
how many bytes mojotrees asked the runtime to copy. The methodology, the
protocol a run must follow, and what each outcome licenses are in
[docs/APPLE_UNIFIED_MEMORY.md](../docs/APPLE_UNIFIED_MEMORY.md).

```sh
pixi run bench-unified-memory              # 256 MiB, 8 rounds, rewrite mode
pixi run bench-unified-memory 1024 8
MOJOTREES_UM_MODE=resident pixi run bench-unified-memory 1024 8
```

The first run, UM-2026-08-15-M4-01, is
[`results/apple_m4_unified_memory_2026-08-15.md`](results/apple_m4_unified_memory_2026-08-15.md).
On that stack the staged copy ran at 75 to 85 GB/s (1 GiB in 12 to 14 ms),
the host write of the payload was 85% to 90% of a round, the mapped-write
route was correct and issued no copy and was 45% to 60% slower, and both
host-pointer routes read or wrote the wrong bytes. The transfer is not where
a GPU round's time goes on this machine, which is what the doc's "read a
route win as a reason to run the trainer, not a substitute for it" was
written to make explicit.

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

## The two GPU training crossovers

`crossover_rules()` in `src/mojotrees/device_policy.mojo` now holds one rule
for the **dense** crossover, scoped to Metal on an Apple M4 for unweighted
squared error at 1,000,000 rows by 50 features and above. The **sparse**
crossover has still never been measured and has no rule. This section is the
protocol for both, and the dense half is the worked example of it rather
than an outstanding task.

Two things about the dense rule are worth stating here, because a reader of
this file will otherwise draw the wrong conclusion from `auto` still
choosing the CPU. The rule's floor is the measured shape and not a fraction
of it, so it says nothing about 250,000 rows, where the device in fact loses
to our own CPU (1.89s against 1.66s, and 1.967 against 1.649 when the second
sweep repeated it). That holds for leaf-wise growth. Depth-wise growth on a
forced device split search runs 250,000 rows in 1.214s and beats our own CPU
there, which is a property of the automatic gate rather than of the rule, and
is written up in
[`results/sweep2_2026-08-15/RESULTS.md`](results/sweep2_2026-08-15/RESULTS.md).
And the rule cannot fire through
`resolve_device` at all, because `DeviceCapabilities.detect()` opens no
device and a hardware-scoped rule cannot match a profile that names no
hardware. That is a wiring gap, not a missing measurement.

**No further crossover rule may be written into `device_policy.mojo` until
its own numbers exist.** A rule is a performance claim that carries an
`evidence_id`, and a threshold derived from reasoning rather than from a
recorded run is the one thing that module refuses to ship. Adding a rule is
a benchmarking result, not a code change, and it is out of order before the
sweeps below have been run and recorded, whatever anyone expects the numbers
to be.

Both drivers call the trainers directly, so neither run needs
`MOJOTREES_AUTO_MIN_CELLS`. That variable exists for exercising the policy
path itself and has no part in these measurements.

Two rules govern every command here.

- Arms are compared only inside one invocation. This machine drifts by a
  factor of 2 to 3 across time windows, so a number from one run and a
  number from another are not comparable even when the two commands were
  identical. Both drivers interleave their arms for that reason, one repeat
  running every arm before the next repeat starts, and both report each
  arm's own spread as the noise floor a delta has to clear.
- Five repeats, and never fewer than three. Each driver prints a warning
  below three and refuses to call anything resolved at one repeat, because a
  single sample has a spread of zero by construction rather than a noise
  floor of zero.

Pin the CPU side before the first command. `MOJOTREES_NUM_WORKERS` takes the
performance-core count from `sysctl hw.perflevel0.physicalcpu`, per the
thread-matching rule in
[docs/APPLE_GPU_BENCHMARK_PROTOCOL.md](../docs/APPLE_GPU_BENCHMARK_PROTOCOL.md),
and `MOJOTREES_PARALLEL_MIN_OPS` stays at its default and is recorded with
the result.

### Dense crossover

`bench_train_gpu.mojo`, arguments
`[n_rows] [n_features] [reg|binary] [repeats] [arms] [seed]`. Two steps,
because the GPU side has two split-search strategies and racing the slower
one against the CPU answers a question nobody asked.

Step one settles the strategy at each row count.

```sh
pixi run bench-train-gpu 250000 100 reg 5 gpu-host,gpu-device
pixi run bench-train-gpu 1000000 100 reg 5 gpu-host,gpu-device
```

Step two races the winner of step one against the CPU. The commands below
assume `gpu-device` won; substitute `gpu-host` at any row count where it did
not. The 50,000 row point is included because it is the shape the device
path is already known to lose, and a sweep with no losing shape in it has
not bracketed anything.

```sh
pixi run bench-train-gpu 50000 100 reg 5 cpu,gpu-device
pixi run bench-train-gpu 250000 100 reg 5 cpu,gpu-device
pixi run bench-train-gpu 1000000 100 reg 5 cpu,gpu-device
```

Then repeat every row count that came back `resolved` in the GPU's favor at
two more data seeds, which is the sixth argument, and once under `binary` so
the verdict is not read as objective independent when only one objective was
measured.

```sh
pixi run bench-train-gpu 250000 100 reg 5 cpu,gpu-device 1
pixi run bench-train-gpu 250000 100 reg 5 cpu,gpu-device 2
pixi run bench-train-gpu 250000 100 binary 5 cpu,gpu-device
```

### Sparse crossover, `w4_sparse`

`bench_train_gpu_sparse.mojo`, arguments
`[n_rows] [n_features] [density_pct] [reg|binary] [repeats] [arms] [seed]`.
It races `train_sparse` on the CPU against `train_gpu_sparse` on the device
over one `SparseBinnedMatrix`, binned once outside the timed region, and it
reports the same spread and the same resolved-versus-indistinguishable
verdict the dense driver does. There is no `gpu-host` or `gpu-device` arm
here, because the sparse grower always searches splits on the host.

Density is an argument because the crossover moves with it. The sparse
accumulator costs O(nnz in node) per node against the dense
O(rows x features), so where the device starts winning depends on how many
entries a node holds and not on the row count alone. One row count at three
densities is therefore the minimum useful sweep, and a single density is not
a crossover.

The first command is the `w4_sparse` shape of the Apple protocol exactly,
since 500 features at 2.0% is 10 nonzeros per row.

```sh
pixi run bench-train-gpu-sparse 200000 500 2.0 reg 5 cpu,gpu
pixi run bench-train-gpu-sparse 200000 500 1.0 reg 5 cpu,gpu
pixi run bench-train-gpu-sparse 200000 500 5.0 reg 5 cpu,gpu
pixi run bench-train-gpu-sparse 1000000 500 2.0 reg 5 cpu,gpu
```

Then two more data seeds, which is the seventh argument, at every
(rows, density) point that came back `resolved` in the GPU's favor.

```sh
pixi run bench-train-gpu-sparse 200000 500 2.0 reg 5 cpu,gpu 1
pixi run bench-train-gpu-sparse 200000 500 2.0 reg 5 cpu,gpu 2
```

Without the pixi task, the driver runs directly.

```sh
pixi run mojo run -I src bench/bench_train_gpu_sparse.mojo 200000 500 2.0 reg 5 cpu,gpu
```

The driver never configures exclusive feature bundling, a custom objective,
or an eval set, and it must not be changed to. `train_gpu_sparse` refuses
all three by name rather than approximating them, so a harness that
configured one would measure a refusal.

At `seed 0` the generated dataset is entry for entry the one
`bench_sparse.mojo` builds at the same shape, so a sparse number from this
driver and the CPU sparse-versus-dense table above describe the same data.
The CSC arrays are built by counting sort rather than from a materialized
dense matrix, which is what makes the 1,000,000 x 500 point runnable.

### What each verdict licenses

| Verdict | What it means | What may be written |
|---|---|---|
| `resolved`, GPU faster | The gap cleared the wider of the two arms' spreads | A candidate crossover rule, once the conditions below all hold |
| `resolved`, CPU faster | Also a result, and worth recording | Nothing, and it forbids a rule at that shape and below |
| `indistinguishable` | The gap is inside the noise floor of the run | Nothing for either backend |
| `unresolvable` | One repeat was used | Nothing, rerun with five |

A crossover rule is justified only when all four of these hold together.

1. The shape reports `resolved` with the GPU arm faster.
2. The same shape resolves the same way at three data seeds, in three
   separate invocations. Only the verdict carries across time windows; the
   seconds do not.
3. The two arms' training losses agree to the precision the driver prints.
   Device histograms are fixed-point, so Float32-level agreement is what to
   expect; a materially different loss means the arms did not fit the same
   model and the wall clocks are not a comparison.
4. Some smaller shape in the same sweep came back either
   `indistinguishable` or `resolved` the other way, so the crossover is
   bracketed from below rather than assumed to sit under the smallest shape
   measured.

The rule that may then be written is the narrowest one the sweep supports.
Its `min_rows` and `min_cells` are the smallest measured shape that passed
all four conditions, never an interpolated or extrapolated one. Its `api`
and `apple_generation` are the device that was measured, its `objective` is
the objective that was measured, its `evidence_id` names the result file,
and `POLICY_VERSION` is bumped with it. Widening any of those fields past
what was run is the same unearned claim as writing the rule with no run at
all.

Record every sweep under `results/` beside the existing files, one per
crossover, carrying the machine, the OS version, the Mojo version, the
resolved worker count, and the full command lines. The dense sweep is
recorded in `results/apple_m4_large_scaling_2026-08-14.md` and
`results/profile_2026-08-15/RESULTS.md`, and one rule was written from them.
**The sparse sweep has no numbers yet**, and `crossover_rules()` carries no
sparse rule for that reason.

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

- the NDCG metric cross-check: mojotrees's `ndcg_score` applied to
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
factor `H / (H + lambda_l2)`, because mojotrees defaults `lambda_l2` to 1.0
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

- LightGBM's `min_data_in_bin=3` (its default) has no mojotrees equivalent;
  both still fit 255-bin quantile histograms.
- Losses differ slightly because tree growth diverges after the first
  floating-point tie; the ~4% MSE gap in mojotrees's favor is within
  seed-to-seed variation, not a quality claim.
- Single machine, single run; rerun both commands back to back on an idle
  machine before quoting numbers.
- The sparse table above was measured on synthetic data with a fixed number
  of nonzeros per row. Real sparse datasets have skewed column densities,
  which changes the per-feature work distribution and so the speedup.
