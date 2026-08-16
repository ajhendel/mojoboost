# Results template

Every published table from this harness is written from this file. Copy it,
fill it in, and do not delete a heading because it did not apply: write
"none" under it. A heading that is absent reads as a question nobody asked,
and every incident below was a question nobody asked.

The comparator block is not optional and it is not written by hand. It is
printed by `run.py` before the first cell, stored in `manifest.json` and in
`records.json` under `comparator`, and carried in the `comparator` column of
every row of `records.csv`. Paste it. If a table cannot say which comparator
produced it, it is not a result yet.

The peer arms travel inside that same block, under `comparator.peers`, and
they are not a second mechanism for the same reason: a file that states its
comparator cannot then fail to state what else was in the table. **A peer
arm is never the comparator.** There is exactly one comparator, `stock+det`,
and a section headed by a peer arm says so in its own first line.

## Why this file exists

Four comparator-configuration incidents in three days, three of them caught
only after a number had been published:

- a margin measured against a thermally throttled comparator,
- a binning ratio measured against a comparator forced to bin every row,
- a speculation figure that was a tautology over a conditioned subset,
- a gain form invalid under L1.

In every case the number was real, and in every case it was quoted for a
question it could not answer. The common property was not carelessness. It
was that the result recorded the number and not the configuration that
produced it, so the caveat lived in somebody's memory and the number
travelled without it.

---

# <scenario or campaign>, <date>

## Comparator

    <paste `comparator` from the run's manifest.json, or the block run.py
    printed. At minimum: id, label, the parameters LightGBM was passed, and
    the reproducibility note.>

The comparator is **`stock+det`**: LightGBM at its own defaults plus
`deterministic=true`. Registered as section C9 of
`bench/results/PROFILE_PROTOCOL.md`. One arm, one label. No other LightGBM
configuration is published, and speed and accuracy are reported together
against it.

`deterministic=true` is the only deviation from pure stock that is not a
feature-space pin. It is on because our arm is reproducible across thread
counts at no cost, so it is the setting that makes the two sides comparable
rather than one that handicaps either.

It does not fully succeed, and that belongs here rather than in a later
correction: in the first real-data run **LightGBM produced two distinct
prediction digests across three repeats on `sparse_highdim`, with
`deterministic=true` already set and a fixed seed**, while our arm was
bit-identical across all three.

Two switches are pinned off and neither is a leftover. `feature_pre_filter`
deletes columns at Dataset construction, which mojotrees does not do, so
leaving it on would compare two engines fitting different feature spaces; it
comes out the day mojotrees implements the filter. `enable_bundle` merges
mutually exclusive sparse features before binning, which is the same kind of
change, and mojotrees's EFB is not applied by every trainer this harness
reaches.

## Peer arm: CatBoost

    <paste `comparator.peers.catboost` from the run's manifest.json, or the
    peer block run.py printed. At minimum: id, label, the parameters
    CatBoost was passed, the two matched parameters, the determinism block,
    and `unmatchable`.>

**The headline row above is unchanged and this section does not compete with
it.** CatBoost is a peer column, reported beside `stock+det` and never
instead of it. Nothing in `thresholds.json` is measured against it and
`verify.py`'s differential does not see these rows. A margin quoted from
this section is a margin against CatBoost and must say so in the sentence
that carries it, not in a footnote.

Two rows, both against the same CatBoost arm:

| row | mojotrees side | CatBoost side |
| --- | --- | --- |
| us in CatBoost mode vs CatBoost defaults | `mojotrees_catboost_mode` | `catboost` |
| our defaults vs CatBoost defaults | `mojotrees` | `catboost` |

Both at a **matched tree count** and at **each engine's own learning rate**.
The second half changed on 2026-08-16, the arm id moved from `cb-default` to
`cb-shipped` to mark it, and every CatBoost number taken under the old id is
superseded rather than re-labelled.

CatBoost picks its own rate from the iteration count and the dataset when it is
not given one, and the value moves with the budget: 0.5 at 2 iterations, 0.4273
at 100 and 0.06573 at 1000, measured at 20,000 rows by 20 features. This
harness used to pin it to 0.1 so all three arms ran one rate; it does not any
more, so the CatBoost column trains at whatever CatBoost chose for that cell
and the two arms beside it train at 0.1.

> **Do not read the CatBoost accuracy column as an engine claim.** At a fixed
> 100-tree budget an engine training at roughly 0.43 walks much further down
> its loss curve than one training at 0.1, so this column can improve against
> `cb-default@v1` without a single line of CatBoost running faster or fitting
> better. The difference is a learning rate multiplied by a budget. Read it as
> an out-of-the-box claim, which is what it is: a user who asks CatBoost for
> 100 trees gets CatBoost's rate and a user who asks LightGBM for 100 trees
> gets 0.1.

The `mojotrees_catboost_mode` row takes CatBoost's **resolved** rate for the
same cell, read off the fitted CatBoost model, so "us in CatBoost's shape" is
CatBoost's own number rather than a transcription. Every record carries
`params.resolved_parity`, which is the key-by-key diff between the two engines'
resolved dicts; read it before quoting either row.

### Determinism, which is weaker here than on the comparator

**CatBoost has no `deterministic` flag.** LightGBM has one, which is the
whole reason the comparator is `stock+det`. CatBoost's like-for-like is a
fixed `thread_count` plus a fixed `random_seed`, and that is weaker: nothing
in it is a promise by the library, only two inputs held still. **This arm is
seeded, not guaranteed, and it must be labelled that way wherever it is
quoted.**

It was checked rather than assumed. On catboost 1.2.10, 20,000 rows by 20
features, 100 iterations at learning rate 0.1 and a fixed seed, the
prediction digest was identical across three in-process repeats, across
three separate processes, and across `thread_count` 1, 2, 4 and 8. That is
one shape, one loss, one machine, and no missing values or categorical
features; bit-identity at 20,000 rows is not bit-identity at 1,000,000,
where the parallel reductions are wider. Every repeat records its own
prediction digest, so state what **this** run observed and not what that
note observed.

The precedent for saying it this way is directly above: LightGBM produced
two distinct prediction digests across three repeats on `sparse_highdim`
with `deterministic=true` already set and a fixed seed. The honest form is
to state what the flag does and does not buy, on both arms.

### What each engine's phases contain

The end-to-end headline includes ingestion, and the three libraries put
ingestion and binning in different places. Paste `comparator.phase_shape`
and do not summarize it away.

| engine | ingest | binning | train | e2e |
| --- | --- | --- | --- | --- |
| mojotrees | transpose | `Dataset.construct()` | boosting | ingest + binning + train |
| lightgbm | inside binning | `Dataset.construct()`, contains ingestion | boosting | binning + train |
| catboost | `Pool()`, conversion only | none, it is inside `fit` | `fit()`, contains binning | ingest + train |

**CatBoost's `Pool` does something neither of the others does, and it is
recorded rather than folded in.** `Pool(X, label=y)` is ingestion only: the
pool is not quantized when it returns. `Pool.quantize()` is a separate
public call, and using it to expose a binning number was tried and rejected
on evidence, because it produces a different model above a few hundred
thousand rows. At 300,000 rows by 20 features, a raw-pool fit, a
default-seed quantized fit and a harness-seed quantized fit gave three
distinct prediction digests and 51 against 50 borders on feature 0. CatBoost
draws its border-construction sample under the quantization seed, which the
fit path does not share. So the CatBoost rows carry `binning: null` with
that reason and their binning cost sits inside `train`. A table that adds a
CatBoost `binning_s` to a CatBoost `train_s` is adding a number that is not
there.

### What no parameter closes

Copy `comparator.peers.catboost.unmatchable` in full. The short form, which
is not a substitute for it: CatBoost grows symmetric trees of depth 6 and
mojotrees has no symmetric policy, so the CatBoost-mode arm is depthwise at
depth 6 and is not the same tree. CatBoost subsamples 80 percent of rows per
tree by default under MVS and the other two do not subsample at all.
CatBoost perturbs split scores with `random_strength` and scores them with
`Cosine`. Its `min_data_in_leaf` default is 1 against this harness's shared
20. Its `nan_mode` is `Min` where the other two learn a direction.

### Cost, before the matrix is scheduled

Copy `comparator.peers.catboost.cost_warnings`. As of this arm's first
version there is one, and it is about `sparse_highdim`: CatBoost's `rsm`
default of 1 means it considers every feature at every split, and the smoke
tier alone took 8.5 seconds of fit on two threads where ingestion took
0.011. The standard tier is 25 times each dimension. A cell that hits
`run.py`'s per-run timeout is an infrastructure failure and takes the run's
exit code with it, so schedule that cell knowing this or leave it out
deliberately and say which.

### Scenarios with no CatBoost row

Copy `comparator.peers.catboost.scenarios_not_run`. As of this arm's first
version: `ranking`, because CatBoost has no lambdarank and running YetiRank
would put a third objective in a column headed by the other two's; and
`categorical_missing`, because CatBoost refuses `cat_features` on the
float64 matrix every engine is handed, and converting a copy for CatBoost
alone would break the data digest that makes the records comparable.

## Run

| field | value |
| --- | --- |
| run id | |
| commit | |
| machine, thread count, device | |
| battery and thermal state | |
| repeats per cell | |
| data kind, and pinned or not | |
| `run.py` exit code | |
| `verify.py` verdict | |

A run that exited 2 has cells that produced no result and is not a source of
numbers. A run with no `verify.py` verdict has no correctness statement, and
a timing without one is a measurement of an engine that may have been
solving a different problem.

## Numbers

<the table. Median across repeats with min and max around it, never a lone
figure. State the metric, not just the ratio. Speed and accuracy together.>

## What this does not establish

<the things a reader could reasonably take from the table and should not.
"None" is an acceptable answer only if it is true.>

## Caveats carried from the scenario

<copy the `caveats` list from the records. They are copied into every record
the scenario produces precisely so that they arrive here.>

---

# Superseded figures

A number whose comparator later changed stays where it is, under a banner,
and is not deleted. A measurement is a record of what was true when it was
taken, and deleting it is how a project loses the ability to explain its own
history.

Mark it exactly like this, at the top of the table it applies to:

> **Pinned configuration, superseded.** Measured against LightGBM pinned to
> `min_data_in_bin=1`, `bin_construct_sample_cnt` at the training row count,
> `force_row_wise=true`, `enable_bundle=false` and
> `feature_pre_filter=false`, with `deterministic=true` in the real-data
> harness and not in the speed harness, which is to say against two
> comparators rather than one. That configuration was retired on 2026-08-16
> in favor of `stock+det`. The row-count binning pin made the comparator do
> strictly **more** binning work than mojotrees did, so any binning ratio
> here is wrong in our favor, and the builder pin chose the comparator's
> histogram algorithm for it rather than letting it choose. Not comparable
> with a figure taken under `stock+det`.

Copy the parameter line the run actually recorded into the banner where the
run has one. The list above is the general shape and a specific run is
better evidence than a general shape.
