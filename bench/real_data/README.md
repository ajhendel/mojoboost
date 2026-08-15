# Real-data differential harness

A reproducible comparison of mojotrees against LightGBM across six problem
shapes, on pinned public datasets or deterministic generators, measuring
quality, time, memory, and model size, and separating the numbers that can
fail a build from the numbers that cannot.

**Nothing in this directory has been run.** It was written as a harness and
committed unexecuted. There are no results here, no numbers in this README,
and no numbers anywhere else in the repository that came from it. When it is
first run, the results land under `results/` and are not committed either;
`results/README.md` explains why.

## The six scenarios

| id | shape | real dataset | generator |
| --- | --- | --- | --- |
| `dense_regression` | dense float matrix, squared error | YearPredictionMSD | interactions plus noise features |
| `imbalanced_binary` | rare positive class | Bank Marketing | logistic model at a solved-for positive rate |
| `multiclass` | softmax, one tree per class per round | Covertype | skewed-prior softmax |
| `ranking` | LambdaRank on graded relevance | MSLR-WEB10K (manual) | graded queries of varying length |
| `categorical_missing` | integer categories and real holes | Adult | high-cardinality effects, missing not at random |
| `sparse_highdim` | CSC, features outnumber rows | RCV1 | power-law column density |

Two more datasets sit in a `large` tier that is never in the default run:
HIGGS for row scaling and news20 for feature scaling.

## Running it

```
pixi run -e bench real-data-fetch --list          # what is registered, what is pinned
pixi run -e bench real-data-fetch adult --pin     # download once, record the digest
pixi run -e bench real-data --dry-run             # the job matrix, nothing executed
pixi run -e bench real-data --tier smoke          # seconds, proves the wiring
pixi run -e bench real-data                       # the standard tier
pixi run -e bench real-data-verify results/<run_id>
pixi run -e bench real-data-report results/<run_id>
```

The pixi tasks are not wired yet. `handoffs/task19_real_data.md` has the
task and dependency entries to add, because this lane does not edit
`pixi.toml`. Until they land, call the scripts directly:

```
python bench/real_data/run.py --dry-run
python bench/real_data/run.py --tier standard --device cpu gpu
python bench/real_data/verify.py results/<run_id>
python bench/real_data/report.py results/<run_id>
```

## How it is put together

```
sources.json          dataset registry: url, version, expected shape, split rule, licence
checksums.lock.json   digests fetch.py --pin observed. Written by a fetch, never by hand
fetch.py              download, verify against the lock, or record a new pin
loaders.py            pinned files into arrays, with the shape checked against the registry
generators.py         deterministic data for every scenario, so the suite runs offline
scenarios.py          the six scenarios and the parameter alignment between the engines
engines.py            one adapter per library, same phases measured on both
quality.py            one implementation of every metric, applied to both engines
measure.py            timing, peak memory, model size, digests
envinfo.py            machine, versions, git state, thermal and load conditions
worker.py             one measured run in its own process, one JSON record out
run.py                builds the matrix and runs it sequentially
thresholds.json       the correctness tolerances, with the reasoning for each
verify.py             the gate: applies thresholds, exits non-zero on a failure
report.py             the timings: prints distributions, decides nothing
schema.json           JSON Schema for a result record
```

## What makes it a differential harness rather than two benchmarks

**Both engines are asked only for predictions.** Every metric in a record is
computed by `quality.py` from those predictions. No engine's self-reported
metric is ever compared against another's. A metric name does not pin down a
metric: log loss differs in its clipping, AUC in its tie handling, NDCG in
what it does with a query nobody can get wrong, and each of those is worth
about as much as the differences the harness exists to detect.

**Both engines get the same data, and it is checked.** Each run rebuilds the
data in its own process from the same deterministic recipe and records a
digest of the exact bytes it will hand over. `verify.py` fails a scenario
whose two engines report different digests before it looks at anything else.

**The parameters are aligned deliberately, and the alignment is written
down.** `scenarios.py` sets `lambda_l2` explicitly because the two defaults
differ, disables exclusive feature bundling because mojotrees has none,
disables LightGBM's feature pre-filter because it deletes columns at Dataset
build time, raises `bin_construct_sample_cnt` to the training row count
because LightGBM otherwise builds its bin edges from a 200000-row subsample
while mojotrees bins from every row, and sets `min_data_in_bin` to 1 because
mojotrees's numerical binner has no minimum-population rule and LightGBM's
default of 3 would merge low-cardinality levels that mojotrees keeps apart.
That last one moves LightGBM onto our rule and leaves us with the larger bin
counts, so it makes the comparison stricter against us rather than laxer.
Each one is justified where it is set. Threads are matched by count rather
than by parameter name: LightGBM reads `num_threads`, mojotrees reads
`MOJOTREES_NUM_WORKERS`, and the runner sets both from one number before
either library is imported.

**The record holds the dicts that were passed, not a reconstruction of
them.** Each adapter returns the parameter dict it used, so the additions it
makes after translation are in the record rather than lost from it. The
record also carries which histogram construction LightGBM ran, since
row-wise and col-wise are different algorithms and a training time cannot be
repeated without knowing which produced it, and the per-feature bin counts
from both engines, which is what makes the binning alignment checkable by
measurement instead of by argument.

**Where alignment is impossible, the record says so.** Scenario caveats are
copied into every record they touch and reprinted under every table.
LightGBM reads its pairwise ranking sigmoid from a lookup table where
mojotrees evaluates it, so ranking models diverge from the first pair; the
ranking thresholds are correspondingly loose and say why.

## Correctness and performance are separated on purpose

`verify.py` gates on quality and exits non-zero. It checks completeness,
input agreement, dataset pinning, bit-identical repeat training, the
mojotrees-against-LightGBM differential in the direction the metric runs, a
floor against the trivial model, and accelerator-against-CPU agreement on
the predictions themselves. Every tolerance lives in `thresholds.json` with
its reasoning attached.

`report.py` prints timings and decides nothing. It shows the median across
repeats with the min and max around it, marks any cell whose spread is wide,
refuses to print a ratio from fewer than three repeats, refuses to put runs
from different machines or builds in one table, and reprints the thread
count, device, data kind, battery state, and thermal state under every
table. Beside each of the two threaded phases it prints parallel efficiency,
CPU seconds over wall seconds, because a seconds column on its own cannot
tell a slow engine from a serial one and the mistake runs in a predictable
direction. `records.csv` reduces repeated samples with the same median the
table uses, so a column and a cell of the same name are the same number.
There is no headline number anywhere in it.

The reason for the split is that a quality difference between two engines
fitting the same objective on the same bins is small, stable, and
reproducible, so a threshold on it is a real test. A timing on a laptop with
a thermal budget is none of those things, so a threshold on it would be a
coin toss wearing a lab coat.

## Measurements taken

Per run: import time, warmup on the target device, binning, training,
batch prediction, single-row prediction latency, wall and CPU time for each
(their ratio catches a thread setting that did not take), peak resident set
of the process, serialised model size and tree count, and the full
environment. Each run gets a fresh process, which is what makes the memory
figure attributable and the warmup figure real.

Several measurements are recorded as unavailable rather than estimated.
Host-to-device transfer time is not exposed to Python by either engine;
instrumenting it is a change to the Mojo accelerator sources and is listed
in the handoff. On the sparse path, mojotrees bins inside `fit`, so binning
time cannot be separated from training time there, and the estimator keeps
its Dataset to itself, so the bin counts cannot be read back either. And a
LightGBM run that forces neither histogram builder has no recoverable
resolved builder, because 4.7 keeps that choice in the tree learner's share
state and reports it only in a log line this harness silences. Every one of
them appears as a null with a reason, which is the only form a missing
measurement takes in this harness.

## Accelerator mode

`--device gpu` runs mojotrees on the accelerator. LightGBM stays on the CPU:
its GPU support is a compile-time option the bench environment does not
install, so a mojotrees accelerator row against a LightGBM CPU row would be
a comparison of two different things. The harness therefore treats
CPU-against-accelerator as a mojotrees-internal check, records it as such,
and skips the cross-engine cell with that reason attached.

The sparse scenario is CPU only on both sides, because mojotrees's CSC path
reports `cpu` whatever device it is given.
