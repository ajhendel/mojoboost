# Real-data differential harness

A reproducible comparison of mojotrees against LightGBM across six problem
shapes, on pinned public datasets or deterministic generators, measuring
quality, time, memory, and model size, and separating the numbers that can
fail a build from the numbers that cannot.

**It has been run, on 2026-08-15, and one file from each complete run is
committed.** The sentence here used to read "Nothing in this directory has
been run", and that stopped being true when `results/20260815T014842Z` and
`results/20260815T023123Z` were produced. The bulk of a run is still
gitignored by design and `results/README.md` explains which parts and why.
What is committed is `results/<run_id>/summary.json`, written by
`summarize.py`: the metrics, the data and prediction digests, the parameters
each engine was passed, and the environment, with no prediction vectors and
no timings in it. It exists because a run readable only on the machine that
took it cannot support a claim made in the repository, which stopped being a
hypothesis when a reviewer read the empty results directory in a fresh
worktree and retracted an accuracy claim that was true. There are still no
numbers in this README; `results/README.md` has them, as a range rather than
a best case.

That run is nonetheless cited twice in the source tree, and for one reason
worth knowing about before reading any record from this harness. It is the
run in which covertype's CPU and GPU arms came back with byte-identical
`predictions_sha256`, which is what exposed
`trainset.train_dataset_multiclass` resolving the device and discarding the
answer. `backend_proof.py` exists because of it.

### What a GPU run of this harness does and does not produce

Three of the eight scenarios declare `devices=["cpu", "gpu"]`
(`dense_regression`, `imbalanced_binary`, `multiclass`). The other five
declare the CPU only: `ranking`, `categorical_missing`,
`high_cardinality_categorical` and `sparse_highdim` have no accelerator path
the harness is willing to compare, and `ordered_boosting_small` is pinned to
a row count chosen for CatBoost's boosting-scheme threshold, which sits below
the size at which this library sends work to the accelerator at all. A
`--device gpu` run therefore records those five as **skipped, with the reason
attached**, which is a different record from a failure and must not be read
as one.

Two corrections follow, because both readings have circulated. "The GPU
errored on four of six scenarios" is wrong twice over: the scenarios without
a GPU arm are three, not four, and they are skipped rather than errored. And
the error records that do exist in that run are not a GPU story at all. They
came from a package rename that had not propagated, and they hit **all six**
scenarios on **both** backends. An error that lands on the CPU arm too is
evidence about the harness environment, not about the accelerator.

## The eight scenarios

| id | shape | real dataset | generator |
| --- | --- | --- | --- |
| `dense_regression` | dense float matrix, squared error | YearPredictionMSD | interactions plus noise features |
| `imbalanced_binary` | rare positive class | Bank Marketing | logistic model at a solved-for positive rate |
| `multiclass` | softmax, one tree per class per round | Covertype | skewed-prior softmax |
| `ranking` | LambdaRank on graded relevance | MSLR-WEB10K (manual) | graded queries of varying length |
| `categorical_missing` | integer categories and real holes | Adult | high-cardinality effects, missing not at random |
| `high_cardinality_categorical` | 1,000,000 rows, five cardinalities, one interaction | none yet | level counts pinned to rows per level |
| `ordered_boosting_small` | 50,000 rows, where CatBoost's default is `Ordered` | none | the dense recipe at a higher noise level |
| `sparse_highdim` | CSC, features outnumber rows | RCV1 | power-law column density |

Two more datasets sit in a `large` tier that is never in the default run:
HIGGS for row scaling and news20 for feature scaling.

The last two are the newest and are the ones that make a comparison possible
rather than adding a shape to it, so what each is for is worth stating.
`high_cardinality_categorical` is the shape ordered target statistics and CTR
feature combinations exist for, and before it landed no scenario in this
suite exercised any of that machinery, so none of it could reach a published
number however well it worked. `ordered_boosting_small` is pinned to a row
count chosen so that CatBoost runs its own `Ordered` default rather than
`Plain`; that row count is an assumption and `scenarios.ORDERED_BOOSTING_ROWS`
records exactly how much of it is verified, which is none of it.

Neither has a real dataset. For the first that is a gap and not a
preference: the real reading of the shape is a click log, and registering one
is a `sources.json` entry, a loader, a licence check and a multi-gigabyte
fetch. A ninth scenario, `text_features`, is **specified and not built** and
is deliberately outside `SCENARIOS`: `scenarios.TEXT_SCENARIO_PENDING` holds
the design and names the three things the input contract has to settle before
it can be registered, and `selfcheck.check_pending_scenarios` fails if it is
registered before they are.

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
python bench/real_data/summarize.py results/<run_id>
```

The last of those is the one that leaves something behind in the
repository. Run it after `verify.py`, commit the `summary.json` it writes,
and the run stops being readable only on the machine that took it.

## How it is put together

```
sources.json          dataset registry: url, version, expected shape, split rule, licence
checksums.lock.json   digests fetch.py --pin observed. Written by a fetch, never by hand
fetch.py              download, verify against the lock, or record a new pin
loaders.py            pinned files into arrays, with the shape checked against the registry
generators.py         deterministic data for every scenario, so the suite runs offline
scenarios.py          the eight scenarios and the parameter alignment between the engines
engines.py            one adapter per library, same phases measured on both
quality.py            one implementation of every metric, applied to both engines
measure.py            timing, peak memory, model size, digests
backend_proof.py      what the trainer emitted about which backend it ran on
envinfo.py            machine, versions, git state, thermal and load conditions
worker.py             one measured run in its own process, one JSON record out
run.py                builds the matrix and runs it sequentially; exits 2 if any cell produced no result
thresholds.json       the correctness tolerances, with the reasoning for each
verify.py             the gate: applies thresholds, exits non-zero on a failure
report.py             the timings: prints distributions, decides nothing
summarize.py          one run reduced to the one file that is committed
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
down.** There is exactly one comparator, and it is named in every run.

**`stock+det`, which is LightGBM at its own defaults plus
`deterministic=true`.** Registered as section C9 of
`bench/results/PROFILE_PROTOCOL.md`. One arm,
one label, and no other LightGBM configuration is published. `run.py`
prints the whole configuration before the first cell, writes it into the
manifest and into `records.json`, and puts its id in a column of every CSV
row, so a table without it is missing a field rather than missing a
convention.

`deterministic=true` is the only deviation from pure stock that is not a
feature-space pin, and the reason it is there is worth stating plainly
because a reader will ask. Our arm is reproducible across thread counts at
no cost, so this is the setting that makes the two sides comparable rather
than one that handicaps either. It does not fully succeed. In the first
real-data run LightGBM produced two distinct prediction digests across
three repeats on `sparse_highdim`, with `deterministic=true` already set
and a fixed seed, while our arm was bit-identical across all three. The
LightGBM side of the repeat-determinism check is non-gating for exactly
that reason. LightGBM's own documentation also advises pairing
`deterministic` with a forced histogram builder; this comparator forces
neither, because choosing the comparator's algorithm for it is the larger
distortion, and the repeats that disagreed above were taken with
`force_row_wise` set anyway.

Superseded 2026-08-16. `scenarios.py` used to set `lambda_l2`
explicitly because the two defaults differed. mojotrees now defaults
to LightGBM's stock 0.0, so the pin is gone and both engines take
the same value by default rather than by agreement.

The two disabled switches are both feature-space pins and neither is a
leftover. `feature_pre_filter` deletes features that cannot satisfy
`min_data_in_leaf` before training starts, which removes them from the
matrix, from the pool `feature_fraction` samples, and from every feature
index; mojotrees does not do it, so leaving it on would compare two engines
fitting different feature spaces. It comes out the day mojotrees implements
the filter, and a lane is on it. `enable_bundle` merges mutually exclusive
sparse features before binning, which is the same kind of change.
mojotrees's EFB is reachable from Python now, so that one is closer to
coming out than it was, but the ranking trainer refuses an active bundling
switch by name, so turning it on for both engines would raise on one of the
eight scenarios rather than compare it.

Those two used to be pinned and are not any more, which is worth recording
because the reasoning that justified the pins is the reasoning that now
forbids them. `bin_construct_sample_cnt` was raised to the training row
count, because LightGBM built its bin edges from a 200000-row subsample
while mojotrees binned every row; `min_data_in_bin` was set to 1, because
mojotrees's numerical binner had no minimum-population rule and LightGBM's
default of 3 would have merged low-cardinality levels that mojotrees kept
apart. Both pins moved LightGBM onto our rule, which made the comparison
stricter against us rather than laxer, and both were correct while they
held. mojotrees's binner now defaults to LightGBM's own 3 and 200000, so
the same pins would move LightGBM *away* from a rule it already shares with
us and make it do strictly more binning work than the subject does. They are
inverted rather than merely stale, so they are gone and both engines bin
stock.

Two more settings left with them, for the same kind of reason.
`force_row_wise=true` chose LightGBM's histogram builder for it; stock
LightGBM times both on its first iterations and keeps the winner, which is
both what a user gets and the harder thing to beat. And
`zero_as_missing`, `min_gain_to_split` and `boost_from_average` were
restatements of LightGBM's own defaults, so they are recorded in
`scenarios.LIGHTGBM_STOCK_DEFAULTS` rather than passed.

Every remaining setting is justified where it is set, and
`selfcheck.check_comparator` fails if the dict grows one that is not.
Threads are matched by count rather
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

## Three exit codes, and what each of them is about

`run.py` reports whether the **matrix ran**. It exits 0 when every cell
that was meant to run produced a result, and **2 when any cell produced no
result at all**: an engine that would not import, a worker that died, a
timeout, or a record that came back without an ok status. It prints a
block naming every failed cell, the distinct causes, and, when the cause is
the one that has actually happened, the command that fixes it
(`pixi run build-python`). Exit code 1 is deliberately unused, so a caller
that reads the number can tell "the matrix did not run" from "the results
were bad".

That distinction is not a refinement. The first time this harness ran it
produced 44 cells of which 27 failed, every mojotrees row dying on `cannot
import name '_mojotrees'` because nothing in the run path builds the
extension, and the run was read as having happened. A cell with no result
is an infrastructure failure and it is a different finding from a red
verdict.

`verify.py` reports whether the **results were good**, and it remains the
sole judge of that. `run.py` decides nothing about quality.

## Correctness and performance are separated on purpose

`verify.py` gates on quality and exits non-zero. It checks completeness,
input agreement, dataset pinning, bit-identical repeat training, proof of
the backend a run actually used, the mojotrees-against-LightGBM
differential in the direction the metric runs, a floor against the trivial
model, and accelerator-against-CPU agreement on the predictions themselves.
Every tolerance lives in `thresholds.json` with its reasoning attached.

The backend check is newer than the rest and exists because the rest could
not see past it. `device_used` is a label the Python side resolved, and for
an unknown stretch every multiclass record written through the `Dataset`
path carried `device_used="gpu"` over a fit that ran on the CPU, because
`trainset.train_dataset_multiclass` resolved the device and then discarded
the answer. Completeness, the self-check, the differential, device
agreement and repeat determinism all passed, because all of them are
consistent with a CPU fit wearing a GPU label. A human caught it by
noticing two identical prediction digests. So every measured fit now runs
with the trainer's phase profile on, the trainer's own report goes into
`backend_proof`, and an accelerator row that carries no proof **and**
reproduces its CPU twin's prediction digest is refused. Either of those on
its own is reported and not refused: an uninstrumented run is not a lie,
and two backends can in principle agree bit for bit, since the device
histogram reduction is fixed point precisely so that they can.

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
its Dataset to itself, so the bin counts cannot be read back either. And
the resolved histogram builder is unavailable on **every** LightGBM cell
now that the comparator forces neither: 4.7 keeps that choice in the tree
learner's share state and reports it only in a log line this harness
silences. That is a real loss against the old pinned comparator and it is
the price of letting LightGBM pick its own algorithm. Every one of them
appears as a null with a reason, which is the only form a missing
measurement takes in this harness.

Two measurements got *less* reduced rather than more. Each record now
carries the per-feature bin counts themselves, not only their total,
maximum, minimum, mean and digest, because a lane diagnosing an accuracy
gap had to reconstruct them by arithmetic over the recorded total and the
list would have answered the question by being read. The digest is still
taken over the whole vector; the stored list is truncated with a marker
above 65,536 features, which only the sparse `large` tier reaches.

## Accelerator mode

`--device gpu` runs mojotrees on the accelerator. LightGBM stays on the CPU:
its GPU support is a compile-time option the bench environment does not
install, so a mojotrees accelerator row against a LightGBM CPU row would be
a comparison of two different things. The harness therefore treats
CPU-against-accelerator as a mojotrees-internal check, records it as such,
and skips the cross-engine cell with that reason attached.

The sparse scenario is CPU only on both sides, because mojotrees's CSC path
reports `cpu` whatever device it is given.
