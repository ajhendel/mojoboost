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
