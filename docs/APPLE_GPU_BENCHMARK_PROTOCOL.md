# Apple silicon benchmark protocol

Protocol version 1.0.0. Result schema `bench/apple/schema.json`, version
1.0.0. Runner `bench/apple/suite.py`, version 1.0.0.

mojotrees trains gradient-boosted trees on the GPU that is already in every
Apple Silicon Mac. This document is how that claim gets tested, on which
machines, against which alternatives, and what has to be true before a
number from it is allowed to leave the repository.

It is a procedure, not a result. No run has been performed under it.

## Status

Every row of the fleet is unmeasured. The suite has been written and
statically checked; it has not been executed on any machine.

| Chip family | Machines run | Correctness gate | Timings | Energy | Thermals |
|---|---|---|---|---|---|
| M1 | 0 | not run | not run | not run | not run |
| M2 | 0 | not run | not run | not run | not run |
| M3 | 0 | not run | not run | not run | not run |
| M4 | 0 | not run | not run | not run | not run |
| M5 | 0 | not run | not run | not run | not run |

The scattered numbers already in `bench/README.md` and
`docs/GPU_VALIDATION.md` were taken on a loaded development machine, and
several are labeled there as unquotable for that reason. They are not
results under this protocol and do not fill any cell above. This protocol
supersedes nothing in those files; it is the way the empty cells get filled.

## What is being asked

Four questions, in this order. The later ones are not worth asking until
the earlier ones are answered.

1. **Does the GPU path produce the same model as the CPU path?** A speed
   comparison between two backends that fit different models is not a
   comparison. Correctness gates everything below it.
2. **Where is the crossover?** Which dataset shapes does the GPU win on,
   which does it lose on, and where does the line sit on each chip family.
   A single winning shape is not an answer.
3. **How does it compare to what people already run?** LightGBM and XGBoost
   at matched thread counts on the same machine, on the same data.
4. **What does it cost in energy?** On a laptop this is a first-class
   question, not a footnote, and it is the one question the GPU might win
   even where it loses on wall clock.

## What a result is

A quotable number satisfies all of the following. A number that fails any
of them stays in the record file and stays out of the README.

- It came from a run whose `conditions.idle_gate_passed` is true and whose
  `conditions.throttle_detected` is false.
- It is a median over at least four non-warmup repetitions, quoted with the
  minimum and maximum, and its `summary.spread_ratio` is at most 1.25.
- It has a quality number from the same fit next to it. A throughput figure
  without a loss figure is not a result. This rule is inherited from
  `docs/GPU_VALIDATION.md` and applies here unchanged.
- It names the machine, the OS build, the Mojo and MAX versions, and the
  versions of every library it was compared against.
- It came from a run at `--scale full`. A smoke run proves the harness
  works and is never quoted.
- The comparison it makes is between rows of one record file. Two records
  from two machines, two OS versions, or two protocol versions are not
  comparable, and neither is a mojotrees number from one day against a
  LightGBM number from another.

## The fleet

The M1 through M5 matrix. What identifies a machine is `hw.model` plus
`machdep.cpu.brand_string`, both captured automatically into every record,
not a marketing name. Within a generation the base, Pro, Max, and Ultra
parts differ in GPU core count by more than a factor of eight, so a result
from one says little about another and the record keeps them apart.

Minimum useful fleet, in priority order:

1. One base chip of the newest generation available. This is the machine
   most users have, and the one where a small GPU has the least room to
   win.
2. One Max or Ultra part of any generation. This is where the GPU path
   should look best, and where a failure to scale with GPU cores would be
   most informative.
3. One M1 of any part. This is the oldest supported hardware and the
   floor for any claim of the form "every Apple Silicon Mac".

Every machine records its GPU core count from `system_profiler`. When that
key is absent on the installed macOS the field stays null; it is never
filled in from the chip name.

Different generations of the same tier are compared only on ratios, never
on absolute seconds, and never without stating both OS versions.

## Workloads

Eight, defined in `bench/apple/suite.py` and reproduced here so the two can
be checked against each other. Every one is generated from the same
counter-based splitmix64 stream the existing drivers use, so no file is
exchanged between engines and every engine in a comparison provably
received identical bytes. The `data_digest` field on each measurement is
what proves it.

| id | Shape | Task | What it is for |
|---|---|---|---|
| `w1_small_dense` | 20,000 x 20 | regression | Launch and fixed-cost bound. The shape the GPU is expected to lose on, reported because it is the shape a first-time user tries. |
| `w2_medium_dense` | 200,000 x 100 | regression | The middle of the range, where the crossover is expected to sit. |
| `w3_large_dense` | 1,000,000 x 50 | regression | Large enough to matter, small enough for a base-configuration Mac. |
| `w4_sparse` | 200,000 x 500, 10 nonzeros per row | regression | The sparse path: `train_gpu_sparse` on the device against `train_sparse` on the CPU. Its crossover is unmeasured, which is why `auto` keeps the CPU for sparse input; this workload is the measurement. |
| `w5_multiclass` | 200,000 x 50, 5 classes | multiclass | Multiplies per-round work and gradient traffic by the class count, which is the shape most sensitive to transfer cost. |
| `w6_missing_categorical` | 200,000 x 50, 10% missing, 5 categorical columns of 40 levels | regression | The two split paths that are not the plain numerical scan. |
| `w7_repeated_fit` | 200,000 x 50, 5 fits in one process | regression | Separates one-time cost (library import, kernel compilation, device context, allocation) from steady state. |
| `w8_prediction` | 200,000 x 50, 10 batches of 20,000 | regression | Inference, which is a different question from training and answered by different code. |

All eight or none. A suite run that reports only `w3_large_dense` is a
marketing exercise and this protocol does not authorize it.

Shared fitting parameters are mojotrees's defaults, which are LightGBM's
defaults: 100 rounds, `num_leaves=31`, `learning_rate=0.1`,
`min_data_in_leaf=20`, `min_sum_hessian_in_leaf=1e-3`, `lambda_l2=1.0`,
`max_bin=255`. The point is to compare implementations, not
configurations.

## Engines

| Engine | Device | Notes |
|---|---|---|
| `mojotrees_cpu` | CPU | Threads set by `MOJOTREES_NUM_WORKERS`, which is the only control; there is no thread parameter. |
| `mojotrees_gpu` | GPU | `device="gpu"` never falls back. Unavailable means `unsupported` on the record, not a CPU number in a GPU row. |
| `lightgbm_cpu` | CPU | `num_threads`, `enable_bundle=false`, `force_row_wise=true`, `deterministic=true`. |
| `xgboost_cpu` | CPU | `tree_method="hist"`, `grow_policy="lossguide"`, `max_depth=0`, `nthread`. |
| `xgboost_gpu` | GPU | Always `unsupported` on Apple silicon. XGBoost's GPU tree method targets CUDA and there is no Metal backend. The row exists so the record states this rather than leaving a blank a reader can misread. |

Parameter matching cannot make the engines identical, and the record does
not pretend otherwise. Every measurement carries a `comparability` list
naming the differences that survive:

- LightGBM's `min_data_in_bin=3` has no mojotrees equivalent, so bin edges
  can differ.
- LightGBM's exclusive feature bundling is forced off, because mojotrees
  has none.
- XGBoost grows depth-wise by default; leaf-wise growth is approximated
  with `grow_policy=lossguide` and `max_depth=0`, which is close but not
  the same rule.
- XGBoost's `min_child_weight` is a hessian sum, not a row count, so
  `min_data_in_leaf` has no exact counterpart.
- XGBoost splits categoricals by its own partitioning rule, not
  LightGBM's. Timings on `w6` are comparable; the fitted models are not.

When a difference cannot be removed, it is named on the record and in any
table drawn from it.

## Thread matching

Apple silicon has two core types, and the distinction matters more here
than the physical and logical distinction does on x86.

- The matched thread counts are **1** and the **performance-core count**,
  read from `sysctl hw.perflevel0.physicalcpu`.
- Efficiency cores are excluded by default. Including them changes the
  ratio between engines rather than scaling it, because the libraries
  distribute work differently across asymmetric cores, and a ratio that
  moves with the thread count is not a property of either library.
- A third point at performance plus efficiency cores may be recorded, and
  when it is, it is recorded as its own row rather than replacing the
  performance-core row.
- `MOJOTREES_NUM_WORKERS=1` is what makes the mojotrees side genuinely
  single threaded. Without it, auto mode uses the machine and the
  comparison is not the comparison it claims to be.
- `MOJOTREES_PARALLEL_MIN_OPS` is left at its default and recorded. Pin it
  explicitly if a run sweeps it.
- `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS` are
  set in the worker environment before the library is imported, which is
  one of the reasons every measurement is its own process.

The GPU rows still use the CPU for binning and for host-side work, so they
carry a thread count too, and it is matched the same way.

## Preconditions, the idle gate

Checked automatically by the suite before the first measurement, and
recorded in `conditions`. A failure stops the run unless `--allow-busy` is
passed, which marks the record unquotable rather than making it acceptable.

1. **One minute load average at or below 0.75.** On a ten core machine a
   single busy core is enough to move a histogram timing.
2. **No competing build.** No `mojo`, `pixi`, `pytest`, `cmake`, `clang`,
   or Xcode process running. Parallel development sessions in particular
   have already ruined every timing currently in `bench/README.md`.
3. **AC power.** On battery, Apple silicon changes both clocks and power
   limits, and the energy numbers stop being about the workload.
4. **Low Power Mode off.**
5. **Not already throttled.** `pmset -g therm` reports
   `CPU_Speed_Limit = 100` before the run starts.

Do these by hand as well, since the suite cannot see them:

- Quit browsers, editors, chat clients, and anything syncing files.
- Let Spotlight indexing and Time Machine finish. A backup starting mid-run
  is the most common cause of a single anomalous repetition.
- Disable display sleep for the duration (`caffeinate -dimsu` around the
  run), because a display sleep transition changes clocks.
- Do not touch the machine while it runs. Do not open a terminal to watch
  it.
- Leave the lid open on a laptop, and do not run on a soft surface.

## Cooldown

Apple laptops in particular reach a thermal limit within a few minutes of
sustained multicore load, and a limited machine produces timings that fall
monotonically through a run. The cooldown rules exist to keep that out of
the numbers.

- **Before the run.** Machine idle and cool for at least five minutes.
  `pmset -g therm` clean.
- **Between measurements.** The suite sleeps `--cooldown` seconds, default
  60. On a fanless MacBook Air, use 180.
- **Between workloads at the largest shape.** Take a thermal sample and, if
  the speed limit is below 100, stop the run rather than continuing into
  degraded numbers.
- **After the run.** A final thermal sample is recorded, and
  `throttle_detected` on the record is true if any sample in the run showed
  a limit. A record with `throttle_detected` true is a record of thermal
  behavior, not of library performance.

Detecting throttling is not the same as avoiding it. If a shape cannot be
run without throttling on a given machine, that is a finding about that
machine, and the honest report is the finding rather than a lower number
taken after a longer wait.

## What gets measured

### Phases

Wall clock, host visible, `time.perf_counter` around regions that do not
overlap.

| Field | Region |
|---|---|
| `import_s` | Importing the engine library in a fresh process. First repetition only. |
| `data_gen_s` | Synthetic data construction. Not part of any comparison; recorded so a long total can be attributed. |
| `binning_s` | Quantile binning of the training matrix, before any tree is grown. `Dataset.construct()` for mojotrees and LightGBM, `QuantileDMatrix` construction for XGBoost. |
| `train_s` | Boosting with the binned dataset already built. |
| `fit_s` | Binning plus training, which is what a user calling `fit()` waits for. The headline metric for most workloads. |
| `first_fit_s`, `steady_fit_s` | `w7` only. The first fit in a process against the median of the later ones. |
| `predict_s`, `predict_rows_per_s` | Scoring the held-out matrix. The headline metric for `w8`. |
| `total_s` | Everything the worker spent on the measured region, data generation excluded. |

Sparse input has no separable binning step in mojotrees, because there is
no `Dataset` path for it, so `binning_s` is null there rather than guessed
at.

### Warmup and compilation

The first repetition of every measurement is marked `warmup: true` and
excluded from the summary statistics. It is kept, not deleted, because the
gap between it and the rest is the compilation and first-touch cost.

On the GPU path that gap contains shader compilation, device context
creation, and the first allocations. Splitting it into named phases needs
instrumentation inside the training run, which the Python layer cannot see.
Until that exists, `warmup_compile_s` is null and the warmup repetition
itself carries the information. `w7_repeated_fit` is the workload designed
to make that cost legible at process scale.

### Device phases

`device_phases` on each repetition holds the inside of a GPU fit, with
field names matching the phases `bench/bench_gpu_validation.mojo` already
prints, so a record and a Mojo driver run can be read side by side.

Three sources, and the record says which one it used:

- `instrumentation`, counters exported from the training run itself. This
  does not exist yet. `handoffs/apple_a8_benchmarks.md` describes the hook
  it is waiting on.
- `mojo_driver`, a separate run of `pixi run gpu-validate` or
  `pixi run bench-hist-scaling` on the same shape. That is a different run
  of different code, and mixing its phase times with this record's totals
  is only legitimate if labeled as such, which this field does.
- `unavailable`, the honest default and what the suite writes today.

### Quality

Computed by the suite from raw predictions, with one scorer for every
engine, so a quality difference is a model difference and not a metric
difference.

- regression, `valid_rmse`
- binary, `valid_logloss` and `valid_accuracy`
- multiclass, `valid_multi_logloss` and `valid_accuracy`

Every measurement also records a `prediction_digest`. Equal digests across
repetitions of one engine is the determinism check. Across CPU and GPU the
digests are expected to differ, because the device accumulates in Float32,
and what is checked there is `max_abs_diff_vs_cpu` against the documented
tolerance rather than equality.

Losses between mojotrees and LightGBM differ slightly because tree growth
diverges after the first floating-point tie. A gap of a few percent is
seed-to-seed variation and is not a quality claim in either direction.

### Memory

`peak_rss_bytes` is `getrusage(RUSAGE_SELF).ru_maxrss` of the worker
process, in bytes on macOS. It covers the whole worker including the
generated data, which is why `data_bytes`, computed exactly from the
shapes, is recorded next to it.

On unified memory a device allocation the process owns is counted in its
resident size, so a GPU row's peak memory is not comparable to a discrete
GPU's host-side peak. Say so wherever these numbers are shown.
`footprint(1)` fills `footprint_bytes` when it is run by hand; the suite
does not run it.

### Energy and average power

The suite measures energy with `powermetrics` or it reports none. It never
derives energy from CPU time, core count, or a TDP figure. Four facts
govern how these numbers may be read:

1. **powermetrics needs root.** Without it the energy block is
   `available: false` with that reason, and the run is still valid for
   every other quantity.
2. **It reports the machine, not the process.** Everything else running
   contributes, which is what the idle gate is protecting.
3. **An idle baseline is required.** The suite samples the idle machine for
   `--idle-baseline-seconds` before the workloads, and
   `energy_above_idle_j` is the workload's energy net of that baseline.
   Without a baseline, the absolute joules mostly measure the machine being
   switched on.
4. **The key names move.** `powermetrics` output differs across macOS
   releases and chips, so the parser records `keys_seen`. A record where
   `available` is true and `keys_seen` is empty is a parser to fix, not a
   machine that used no power.

The comparison worth making is energy per fit at equal quality, between
`mojotrees_gpu` and the matched-thread CPU engines on the same machine in
the same run. Energy per fit against a different machine is not a
comparison, and average power without a window length is not a number.

If a CPU engine and the GPU engine reach the same held-out loss and the GPU
uses materially less energy while being slower on wall clock, that is a
real and reportable result. It is also the result this protocol is most
likely to find on a laptop, which is exactly why energy is measured rather
than assumed.

### Thermals

`pmset -g therm` before the run, after every measurement, and at the end.
`CPU_Speed_Limit` below 100 or a thermal pressure above nominal sets
`throttle_detected`, which invalidates every timing in the record. With
`--energy` the `powermetrics` thermal sampler fills `thermal_pressure` as
well.

### Environment

Captured automatically into every record: chip brand string, `hw.model`,
physical, performance, efficiency, and logical core counts, GPU core count
and name, memory size, macOS version and build, kernel, Python, Mojo, MAX,
mojotrees, `mojotrees.gpu_available()`, LightGBM, XGBoost, numpy, scipy,
the git commit, and whether the tree was dirty.

A run from a dirty tree is fine for exploration and is not quotable.

## Running it

```sh
python bench/apple/suite.py --list                  # the catalog
python bench/apple/suite.py --plan                  # what a run would do
python bench/apple/suite.py --self-check            # static validation only
python bench/apple/suite.py --run --scale smoke     # harness check, not a result
```

The real run, on an idle machine, from a clean tree:

```sh
caffeinate -dimsu python bench/apple/suite.py --run \
    --scale full --repetitions 5 --cooldown 60 \
    --note "fleet: M4 base, 24 GB, macOS 15.6"
```

With energy, which needs root and therefore an explicit decision:

```sh
sudo -E caffeinate -dimsu python bench/apple/suite.py --run \
    --scale full --repetitions 5 --cooldown 60 --energy \
    --idle-baseline-seconds 30
```

One record lands in `bench/apple/results/<run_id>.json`, validated against
the schema before it is written. Schema violations are printed and do not
suppress the file, because a malformed record is still evidence.

Expect hours, not minutes. Eight workloads times four engines times two
thread counts times five repetitions, with a minute of cooldown between
measurements, is a long afternoon at full scale. Run it once, properly,
rather than four times in a hurry.

## Reporting rules

- Publish the record file alongside any table drawn from it. A table
  without its record is not reviewable.
- Quote medians with minimum and maximum. Never a single repetition.
- State the machine, the OS build, and every library version in the same
  place as the numbers.
- State the thread count in the same sentence as any CPU comparison.
- Never present a result from one chip as representative of Apple silicon.
  Say M4 base, or M1 Max, or whatever it was.
- Never present a GPU histogram microbenchmark as end-to-end training, and
  never present a phase time from a Mojo driver as a phase of a run in this
  record.
- Never compare against a competitor's published number. Rerun it here or
  do not make the claim.
- A workload an engine cannot run is reported as unsupported, with the
  reason, in any table that includes that engine. Dropping the row makes
  the engine look better than it is.

## Prohibited

- Fabricating, estimating, extrapolating, or interpolating any number in a
  record file, including energy from CPU time and a chip's timings from a
  neighboring chip's.
- Quoting a smoke-scale run, an `--allow-busy` run, a throttled run, or a
  run whose spread ratio exceeded the threshold.
- Filling a null field by hand. A null is a measurement that did not
  happen, and the fix is to make it happen.
- Editing a record file after the fact. Rerun, and keep both records.

## Open items

Named here so the gaps are visible rather than implied.

- No run has been performed. Every cell of the status table is empty.
- No phase counters are exported to Python, so `device_phases` is
  `unavailable` on every measurement the suite can currently produce.
  `handoffs/apple_a8_benchmarks.md` describes the hook.
- `resolved_threads` is null for every engine, because none of the three
  libraries exposes what it actually used. Thread matching is therefore
  requested and not verified.
- Sparse GPU training does not exist, so `w4` has no `mojotrees_gpu` row
  and will not until a sparse kernel does.
- Energy attribution is machine-wide. Per-process energy is not available
  from `powermetrics` on Apple silicon and this protocol does not claim it.
- XGBoost's leaf-wise approximation is close but not exact. If the
  difference turns out to matter at the shapes here, the honest fix is to
  report both growth policies rather than to pick the flattering one.
