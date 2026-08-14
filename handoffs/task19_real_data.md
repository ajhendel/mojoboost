# Task 19 handoff: real-data differential harness

**Status: authored, unexecuted.** Every file below was written and statically
checked. Nothing was trained, no dataset was downloaded, no checksum was
computed, and there are no results anywhere in the repository from this lane.
Nothing was committed or staged.

Everything lives under `bench/real_data/`. No file outside that directory and
this handoff was touched, so the wiring that belongs in shared files is
written out here for whoever holds `pixi.toml`.

## What is there

| file | role |
| --- | --- |
| `README.md` | what the harness is, how to run it, and the policy it follows |
| `sources.json` | dataset registry: url, version, expected shape, split rule, licence, citation |
| `checksums.lock.json` | empty pin file, written only by `fetch.py --pin` |
| `fetch.py` | download, verify against the lock, or record a first pin |
| `loaders.py` | pinned files into arrays, shapes checked against the registry |
| `generators.py` | deterministic splitmix64 data for all six scenarios |
| `scenarios.py` | the six scenarios and the engine parameter alignment |
| `engines.py` | one adapter per library, identical phases measured on both |
| `quality.py` | one implementation of every metric, applied to both engines |
| `measure.py` | timings, peak RSS, model size, digests |
| `envinfo.py` | machine, versions, git state, load and thermal conditions |
| `worker.py` | one measured run per process, one JSON record out |
| `run.py` | builds the job matrix and runs it sequentially |
| `thresholds.json` | correctness tolerances, each with its reasoning |
| `verify.py` | the gate: exits non-zero on a correctness failure |
| `report.py` | the timings: prints distributions, gates nothing |
| `schema.json` | JSON Schema for a result record |
| `selfcheck.py` | static consistency and metric checks, no training, under a second |
| `results/README.md` | why results are never committed |
| `.gitignore` | keeps `data/` and `results/` out of the repository |

## Wiring needed in `pixi.toml`

Not applied by this lane. Add to `[feature.bench.dependencies]`:

```toml
scipy = "*"     # CSC matrices for the sparse scenario and the LIBSVM loader
pandas = "*"    # fast CSV reading; loaders.py falls back to the csv module
```

`numpy` and `lightgbm` are already there. Nothing else is needed: fetching
uses `urllib`, and the archive handling uses `zipfile`, `gzip`, and `bz2`.

Add to `[feature.bench.tasks]`:

```toml
real-data          = "python bench/real_data/run.py"
real-data-fetch    = "python bench/real_data/fetch.py"
real-data-verify   = "python bench/real_data/verify.py"
real-data-report   = "python bench/real_data/report.py"
real-data-check    = "python bench/real_data/selfcheck.py"
```

`real-data-check` is the only one worth putting anywhere automatic. It needs
nothing but numpy and the standard library, finishes in well under a second,
trains nothing, and downloads nothing, so it can also sit in the default
environment if that is more convenient than the bench one.

## The order to run it in, first time

```
pixi run -e bench real-data-check                     # the harness is consistent
pixi run -e bench real-data --dry-run --tier smoke    # the matrix, nothing executed
pixi run -e bench real-data --tier smoke              # minutes, synthetic only
pixi run -e bench real-data-fetch --list              # what is registered
pixi run -e bench real-data-fetch adult --pin         # one dataset, record its digest
pixi run -e bench real-data --tier standard
pixi run -e bench real-data-verify bench/real_data/results/<run_id>
pixi run -e bench real-data-report bench/real_data/results/<run_id>
```

The smoke tier runs entirely on generators, so it is the honest first step:
it proves the wiring without asking anyone to trust a URL yet.

## What still needs a decision or a download

**The pins are empty and that is deliberate.** This lane had no network, and
a checksum nobody computed is worse than no checksum: it looks like
provenance. `sha256` is `null` for every entry in `sources.json` and
`checksums.lock.json` has no pins. The first person to fetch each dataset
runs `fetch.py --pin <id>`, reads what it wrote, and commits the lock. Until
a dataset is pinned, `loaders.load` refuses it and the run falls back to the
generator with the reason recorded; `--allow-unpinned` overrides that and
stamps `pinned: false` on the record, which `verify.py` then fails on unless
told otherwise.

**The URLs are unverified.** They were written from the shape of the UCI
static export endpoints and the LIBSVM dataset page and have not been
requested. The first `fetch.py` run is where they are checked, and a wrong
one is a one-line fix in `sources.json`. Two are worth watching:
`bank_marketing` needs a nested zip (`bank-additional.zip!bank-additional/
bank-additional-full.csv`), which the fetcher supports but which is exactly
the sort of layout that gets repackaged, and `covertype` has a gzip inside
its zip.

**MSLR-WEB10K is manual.** Its download link redirects through an endpoint
that changes, so pinning a URL for it would be pinning a lie. Point
`MOJOBOOST_BENCH_DATA` at a directory holding `Fold1`, then
`fetch.py --pin mslr_web10k` checksums the two files the loader reads. Until
then the ranking scenario runs on its generator.

**Nobody has read `adult.test` with this parser.** Its first line is a
comment and its labels carry a trailing period; both are handled in
`sources.json` and `loaders.py`, and both are the kind of thing that is
wrong until it has been run once.

## Asks on the library side

None of these block the harness. Each one turns a `null` with a reason into
a measurement, which is the only way this harness ever gains a number.

1. **Host-to-device transfer time is not observable from Python.** Every
   accelerator record carries
   `transfers: {value: null, unavailable_reason: ...}`. Making it a number
   means instrumenting `src/mojoboost/histogram_gpu.mojo` and
   `src/mojoboost/train_gpu.mojo` and exposing the total through the
   binding. Worth doing when the accelerator story is being written up,
   because upload cost is most of what the M4 numbers are currently
   measuring.
2. **`mojoboost.Dataset` does not take a SciPy sparse matrix.** The sparse
   scenario therefore goes through `MojoBoostClassifier.fit`, where binning
   happens inside `fit` and cannot be timed separately, and where the device
   is `cpu` whatever was asked for. Folding CSC into `Dataset` would make
   the sparse row measurable on the same footing as every other row, and
   would give it an `eval_set` as a side effect.
3. **Thread count has no parameter.** mojoboost reads
   `MOJOBOOST_NUM_WORKERS` where LightGBM takes `num_threads`. The runner
   sets the environment before the worker process starts and the adapter
   asserts it rather than setting it late, so this is handled, but a
   `num_threads` parameter that fed the same dispatcher would remove a
   whole class of ways to get a threaded benchmark wrong.
4. **`best_score_` is a scalar where LightGBM's is a dict.** Not used by
   this harness, which computes its own metrics, but it is the same
   compatibility gap Task 20 is looking at and it is visible from here.

## Choices worth arguing with

**Both engines are asked only for predictions.** Every metric in a record is
computed by `quality.py`. No engine's self-reported metric is ever compared
against another's, because a metric name does not pin down a metric: log
loss differs in its clipping, AUC in its tie handling, NDCG in what it does
with a query that has no relevant document. Those differences are the same
size as the differences the harness exists to detect.

**One NDCG convention had to be picked.** A query with no positive label has
no defined NDCG. LightGBM's own metric counts it as a perfect 1.0; this
harness skips it, so a query nobody can get wrong cannot raise the score.
`quality.ndcg(..., empty_query="one")` reproduces LightGBM's choice. Either
way it is applied to both engines identically, which is what matters.

**LightGBM stays on the CPU.** Its GPU support is a compile-time option the
bench environment does not install, so a mojoboost accelerator row against a
LightGBM CPU row would be a comparison of two different things. The harness
skips that cell with the reason attached and treats CPU-against-accelerator
as a mojoboost-internal agreement check.

**The alignment settings are opinions.** `enable_bundle=false`,
`feature_pre_filter=false`, `force_row_wise=true`,
`bin_construct_sample_cnt` raised to the row count, and `lambda_l2=1.0` on
both sides. Each is argued at the top of `scenarios.py`. The
`bin_construct_sample_cnt` one is the least obvious and the most important:
LightGBM builds its bin edges from a 200000-row subsample by default and
mojoboost bins from every row, so above that row count the two are binning
different data and LightGBM's binning time is not comparable.

**Timings gate nothing.** `verify.py` exits non-zero on quality, determinism,
input agreement, pinning, and CPU-against-accelerator agreement.
`report.py` prints timings, shows the spread, refuses a ratio from fewer
than three repeats, refuses to put runs from different machines or builds in
one table, and has no headline anywhere in it. A quality difference between
two engines fitting the same objective on the same bins is small and
reproducible, so a threshold on it is a test. A timing on a laptop that
throttles is not.

## CI

Suggested, not applied, and it belongs to whoever owns `ci.yml`:

- `selfcheck.py` on every push. It is cheap, needs no build, and is the only
  thing here that catches a typo in the harness before somebody spends an
  afternoon on a run.
- The smoke tier, synthetic only, CPU only, behind `workflow_dispatch`
  rather than on push. It builds the extension and trains eighteen small
  models, which is more than a push should cost.
- Never the standard tier on CI. It needs pinned downloads, and its timings
  from a shared runner would be noise presented as data.

This lane respected the round's CPU protection contract: nothing was
compiled, no test suite was run, and the only executions were
`python bench/real_data/selfcheck.py` and a `--dry-run` of the matrix
builder written to a scratch directory outside the repository.

## Validation actually performed

- `python bench/real_data/selfcheck.py` passes. It compiles all twelve
  modules, parses all four JSON files, and checks that every scenario has a
  generator that exists, a registered dataset, a threshold entry, and a
  primary metric whose direction `quality.py` knows; that every threshold
  gates a metric with a known direction; that every registered dataset has a
  loader and no registry entry carries a hand-written digest; that the two
  parameter translations agree on regularisation, hessian, class count, and
  the ranking parameters; that every tier's generator arguments are accepted
  by the generator; and that the CSV columns match what the flattener
  produces. It also checks the metrics against fixtures with known answers:
  AUC of a perfect, reversed, and fully tied ranking, AUC with one class
  absent, average precision of a perfect ranking, log loss and multiclass
  log loss at uniform probabilities, RMSE of a unit error, NDCG of the ideal
  and the worst ordering, the empty-query convention in both settings, and
  that a group vector which does not sum to the row count is rejected.
- `run.py --dry-run --tier smoke --device cpu --device gpu` writes the
  matrix and runs nothing. It produced 45 runs and 9 skips, each skip
  carrying its reason.
- `verify.py` and `report.py` were exercised against a throwaway record
  fixture in the session scratchpad, which walked the pass, fail, warn, and
  skip paths of the gate and both output formats. The fixture holds
  placeholder values, was never in the repository, and produced no results
  file.
- `git diff --check` on the assigned paths is clean.

No benchmark was executed and no number in this repository came from this
lane.
