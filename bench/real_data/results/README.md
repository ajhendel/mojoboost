# Results

The harness has been run. Two complete runs are summarized here, and one
file from each of them is committed. Everything else those runs produced
stays out of the repository, for the reasons below.

This file used to open "Empty, and not by accident. This harness has never
been run." That was true when it was written and stopped being true on
2026-08-15. It is worth knowing how it was corrected, because the failure it
caused is the reason the summaries exist. A reviewer working in a fresh
worktree read this directory, found nothing in it, read that first line, and
concluded the real-data harness had never been executed. On that basis a
true claim about accuracy parity with LightGBM was retracted. The claim was
correct and the reviewer was reasonable: an accurate claim with no evidence
in the tree is indistinguishable from an inaccurate one.

`RESULTS_TEMPLATE.md`, beside this file, is what a published table is
written from. It carries the comparator block, the run block, and the
"pinned configuration, superseded" banner for figures taken against a
comparator that has since changed.

## What is committed

One file per complete run:

    <run_id>/summary.json

`summarize.py` writes it from a run directory. It carries, per scenario and
per engine and per device: the metrics `quality.py` computed, the trivial
baseline beside them, the digest of the predictions, the digests of the
training and test matrices both engines were handed, the parameter dicts
each engine was passed, the model size and tree count, the backend evidence
the trainer emitted, and the machine, toolchain, and commit that produced
all of it. It also carries the arithmetic that needs no prediction vector:
this project against LightGBM metric by metric, and this project's
accelerator arm against its own CPU arm.

## What is not committed

The manifest, the job files, the per-job records, `records.json`,
`records.csv`, and the prediction vectors. `.gitignore` in the parent
directory keeps them out and says which line does what.

They are excluded for two different reasons that are worth keeping apart.
The prediction vectors are excluded because they are large and because a
digest answers the same questions they do. The timings are excluded because
of what a committed timing does. A benchmark result is a measurement of a
machine on a day. Committed to a repository it becomes a claim about a
library, quoted later by someone who never saw the manifest that said the
laptop was on battery. `report.py` exists to show the distribution to a
person, who then reads it, names the conditions, and writes the sentence.
The summary carries no timing at all, which is deliberate.

Quality is different, and that difference is the whole argument for
committing anything here. A quality difference between two engines fitting
the same objective on the same bins is small, stable, and reproducible;
`thresholds.json` gates on exactly that property and says so at length. A
metric with its digests and its environment beside it survives being read a
year later in a way that a training time does not.

## What a summary proves, and what it does not

It is evidence that a run happened, on the machine and at the commit named
in its `environment` block, against the datasets named by their digests, and
that the numbers quoted elsewhere in this repository are the numbers it
produced.

It is not a reproduction. Every number in it was measured once. Two
summaries from two machines will differ, and most of the ways they can
differ are legitimate: a different LightGBM build, a different thread count,
an accelerator that is present on one machine and not the other, a real
dataset that is pinned on one and falls back to the generator on the other.
A diff between two summaries is a question, not an answer.

It is not a verdict. `verify.py` decides pass or fail, and one of its checks
compares the two prediction vectors row by row, which cannot be done from a
file that carries only their digests. The summary records that check as
unavailable with its reason rather than leaving it out. What the summary
does carry is a `flags` list: conditions already visible in the file that a
reader should not have to find, each stated with the numbers behind it. An
empty list means none of those particular things is true of the run. It does
not mean the run passed.

## The runs

| run | state | summarized |
| --- | --- | --- |
| `20260815T013837Z` | dry run, no records | no |
| `20260815T014008Z` | 30 of 45 records errored: 27 could not import the package, 3 raised `KeyError: 'num_class'` | no |
| `20260815T014842Z` | complete, 45 records | yes |
| `20260815T023123Z` | complete, 45 records | yes |
| `20260815T190344Z` | dry run, no records | no |
| `20260815T190351Z-h2h` | interrupted, no manifest | no |

The rejected runs are listed so that the two that are here can be read as
the complete set rather than as a selection. `summarize.py` refuses each of
them by a rule rather than by judgment, and prints which rule: a run
without a manifest was interrupted, because `run.py` writes the manifest
after the last job; a run whose manifest says `dry_run` executed nothing;
and a run holding an error or timeout record did not complete, so
summarizing the cells that finished would report a broken run as a working
one.

Both summarized runs carry flags worth reading before either is quoted.
`20260815T014842Z` spans five different commits, because the working tree
moved while the run was executing, so it is not one build and `report.py`
would split it into five tables; its multiclass differential is also 14.3%
against a 5% limit, which is the run that exposed the multiclass dispatch
bug rather than a parity result. `20260815T023123Z` is a single build and
every gated cell is inside its threshold, but its covertype accelerator arm
carries the same prediction digest as its CPU arm with nothing proving a
device backend ran, which is the observation that started the whole
`backend_proof.py` line of work. Neither run's records carry a
`backend_proof` field, because both predate it.

Between the two runs, every cell except multiclass produced a byte-identical
prediction digest. The two summaries are therefore the same measurement
twice over, except for the one scenario a fix landed in.

## The parity spread, stated as a range

> **Pinned configuration, superseded.** Both runs below were measured
> against LightGBM pinned to `min_data_in_bin=1`,
> `bin_construct_sample_cnt` at the training row count, `force_row_wise=true`,
> `deterministic=true`, `enable_bundle=false` and `feature_pre_filter=false`.
> That comparator was retired on 2026-08-16 in favor of **`stock+det`**,
> LightGBM at its own defaults plus `deterministic=true`, registered as
> section C9 of `bench/results/PROFILE_PROTOCOL.md`.
>
> The figures stay, because a measurement whose comparator later changed is
> still a record of what was true when it was taken. What they now describe
> is accuracy parity against a LightGBM that was binning on our rule rather
> than its own: `min_data_in_bin=1` moved it onto a minimum-population rule
> mojotrees no longer has, and the row-count sample made it fit its edges
> from every row while mojotrees fit them from 200,000. Both differences are
> in the binning, which is where an accuracy differential on a quantile
> histogram is most sensitive, so **a re-run under `stock+det` is the thing
> to quote and these numbers are not it**.
>
> Nothing here was a timing, so none of the speed caveats apply.

The claim these summaries exist to support is accuracy parity with LightGBM.
From `20260815T023123Z`, the relative difference on each scenario's primary
metric, this project against LightGBM, both scored by `quality.py` from
predictions:

| scenario | dataset | primary metric | relative difference |
| --- | --- | --- | --- |
| `sparse_highdim` | RCV1 | auc | 5.7e-06 |
| `ranking` | generator, see below | ndcg@10 | 1.6e-04 |
| `multiclass` | Covertype | multi_logloss | 4.1e-04 |
| `dense_regression` | YearPredictionMSD | rmse | 5.1e-04 |
| `categorical_missing` | Adult | auc | 5.3e-04 |
| `imbalanced_binary` | Bank Marketing | average_precision | 3.2e-03 |

Five of the six scenarios ran on pinned real data. `ranking` did not:
MSLR-WEB10K has no entry in `checksums.lock.json`, so it fell back to the
generator and the record says so. Across the five real datasets the spread
is 5.7e-06 to 3.2e-03, a factor of about 560 between the closest and the
furthest.

This project has been describing that as "a few parts in ten thousand." For
three of the five real datasets it is accurate. For RCV1 it understates the
result by two orders of magnitude, and for Bank Marketing it overstates it
by one: 3.2e-03 is a few parts in a thousand, not in ten thousand. The
honest form of the claim is the range and not the best case, and the range
is worth quoting with the metric attached, because average precision on a
rare positive class is dominated by the ordering of a few hundred rows and
is the noisiest primary metric in the suite. `thresholds.json` says as much
in the `imbalanced_binary` rationale, and gates that scenario at 0.02
absolute for the same reason.

The secondary metrics are wider still. Log loss differs by 4.9e-03 on both
`imbalanced_binary` and `sparse_highdim`, so the primary-metric figures
above are the closest of the numbers the harness computes rather than a
summary of all of them. Every metric is in each summary's `differential`
block, gated and ungated alike, which is what makes the spread visible
instead of quotable one number at a time.

All of the above is one machine, an Apple M4 at ten threads, on one day.

## Regenerating

    python bench/real_data/summarize.py results/<run_id>

Writes `results/<run_id>/summary.json`. The output is a pure function of the
run directory, with no generation timestamp in it, so regenerating a summary
that is already committed rewrites the same bytes and leaves the working
tree clean. A new run produces an untracked `summary.json` next to a run
directory that is otherwise ignored, which is the prompt to commit it.

    python bench/real_data/summarize.py results/<run_id> --stdout

Writes to standard output instead, which is how the summaries in this
directory were produced from run directories that live on another machine.

## Before anything from here is quoted

- The table it came from was written from `RESULTS_TEMPLATE.md` and carries
  its comparator block. The comparator is `stock+det` and every run records
  the whole configuration in its `manifest.json` and its `records.json`, in
  the `comparator` column of `records.csv`, and on the console before the
  first cell. A figure from an older run carries an older comparator and
  says so under a "pinned configuration, superseded" banner.
- `run.py` exited 0. It exits 2 when any cell produced no result at all,
  which is a different finding from a red verdict and is not a statement
  about quality either way.
- `verify.py` returned zero on the run. A red verdict means the two engines
  were not compared on the same problem, whatever the timings say. No
  verdict has been recorded for either run summarized here, and running one
  against them now would be misleading: `verify.py` looks up records by the
  engine name `mojotrees`, and these runs predate the rename and say
  `mojoboost` in every record, so its differential, backend, determinism and
  device checks would all find nothing to check and pass by vacancy. The
  `differential` block in each summary is the same arithmetic under the same
  thresholds, computed without the engine name.
- The manifest's environment block is quoted with the number: machine,
  thread count, device, mojo and lightgbm versions, and the commit. Each
  summary carries it, and carries every distinct environment its records saw
  rather than the first.
- The record says `data_kind: real` and `pinned: true`, if the number is
  described as a real-data result. Each summary's `datasets` block carries
  both, and carries the fallback reason where a scenario ran on the
  generator.
- The cell is not marked `!` in the report, or the mark is quoted too. That
  mark is about timings, so it lives in `report.py` and not in a summary.
- At least three repeats. `report.py` will not print a ratio from fewer, and
  a ratio it refused to print should not be reconstructed by hand.

A number that fails any of these is not a result yet. It is an observation,
and it belongs in a message to whoever is working on that lane, not in a
README.
