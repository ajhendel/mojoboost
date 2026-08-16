"""The committed evidence. Reduces a run on disk to one small JSON file.

    python bench/real_data/summarize.py results/<run_id>
    python bench/real_data/summarize.py results/<run_id> --stdout
    python bench/real_data/summarize.py results/*/

A run leaves behind a manifest, forty-five records, forty-five prediction
vectors and two concatenations of the records, and `.gitignore` keeps all
of it out of the repository. That rule is right about the bulk and wrong
about the consequence: with nothing committed, a true claim about accuracy
and an invented one look exactly alike from a clean clone. A reviewer read
the empty `results/` directory in a fresh worktree, concluded the harness
had never been run, and retracted a parity claim that was in fact correct.

So this module writes the one part of a run that is small enough to commit
and specific enough to argue with: `results/<run_id>/summary.json`, holding
the metrics both engines produced, the digests of the data they were given,
the digests of the predictions they produced, the parameters they were
passed, and the machine and toolchain that produced all of it. The
prediction vectors stay ignored. They are large, they are reproducible from
the record, and a digest settles the same questions they do.

What the file is evidence of, and what it is not:

- It is evidence that a run happened, on a named machine, at a named
  commit, against named datasets, and that the numbers quoted elsewhere in
  the repository are the numbers it produced. That is what was missing.
- It is not a reproduction. Every number in it was measured once, on one
  machine, on one day. Two summaries from two machines will differ, and
  most of the ways they differ are legitimate: a different LightGBM build,
  a different thread count, a different accelerator, a fallback from a real
  dataset to the generator. Diffing two summaries is a way to find the
  question, not the answer.
- It is not a verdict. `verify.py` decides pass or fail, and it needs the
  prediction vectors for the device-agreement check, so it runs on the
  machine that holds them. This file carries the inputs to that decision
  and the arithmetic that does not need the vectors, clearly labeled.

Provenance vocabulary, used on every derived field in the output:

    measured    copied from a record exactly as the harness wrote it
    derived     arithmetic over measured values, with no new observation
    unavailable a value the run did not record, with the reason, never an
                estimate and never an omission

Nothing here trains, downloads, imports an engine, or reads a `.npy` file.

The output has no generation timestamp on purpose. A summary is a pure
function of the run directory it was built from, so regenerating one that
is already committed produces the same bytes and leaves the working tree
clean. The timestamp that matters is the run's own.
"""

import argparse
import glob
import json
import re
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import envinfo  # noqa: E402
import verify  # noqa: E402

#: Bumped when the shape of the output changes in a way that would break a
#: reader. A summary written by an older version stays valid; it says which
#: version wrote it.
SUMMARY_VERSION = 1

#: The engine this project ships, under its current name. The package was
#: renamed from `mojoboost` on 2026-08-15 and runs taken before that carry
#: the old name in every record. The summarizer never rewrites it: the
#: summary reports the name the run actually used and notes the rename, so
#: that a reader comparing an old summary against a new one is told why the
#: engine key moved rather than left to guess.
PROJECT_ENGINE = "mojotrees"
FORMER_PROJECT_ENGINE_NAMES = ("mojoboost",)
REFERENCE_ENGINE = "lightgbm"


def _finite(value):
    """A JSON-safe number. NaN and infinity become null.

    `quality.py` returns NaN for a metric that is undefined on the data it
    was given, AUC on a single-class test set being the usual one. JSON has
    no NaN, and the dump below refuses to write one, so the conversion
    happens here where it can be documented: a null in a metrics block is a
    metric the harness scored as NaN, not a metric it skipped.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _metrics(block):
    return {name: _finite(value) for name, value in sorted((block or {}).items())}


def load_run(run_dir):
    """(manifest, records) for a run directory.

    Records are read from the individual files under `records/` rather than
    from `records.json`, because the concatenation is written once at the
    end of a run and an interrupted run has the parts without the whole.
    Reading the parts means an incomplete run can still be described, and
    described as incomplete, instead of failing to open.
    """
    manifest_path = os.path.join(run_dir, "manifest.json")
    manifest = None
    if os.path.exists(manifest_path):
        with open(manifest_path) as handle:
            manifest = json.load(handle)

    records = []
    for path in sorted(glob.glob(os.path.join(run_dir, "records", "*.json"))):
        with open(path) as handle:
            records.append(json.load(handle))
    return manifest, records


def assess(manifest, records):
    """Whether this run is worth committing a summary of.

    Four states, and the difference between them matters more than the
    verdict does, because three of them look identical from a directory
    listing:

    - `no_manifest`: `run.py` writes the manifest last, so a run directory
      without one was interrupted and the record set is a prefix of the
      matrix, not the matrix.
    - `dry_run`: the matrix was written and nothing was executed. The
      directory has a manifest and no records at all.
    - `incomplete`: records exist and some carry `status` error or timeout.
      A summary of only the cells that finished is how a broken run gets
      quoted as a working one.
    - `complete`: a manifest, at least one record, and every record either
      ok or a declared skip.

    Skips are not failures. `run.py` records a scenario with no accelerator
    path as skipped with the reason attached, and a run whose only
    non-ok records are skips is complete.
    """
    counts = {"ok": 0, "error": 0, "timeout": 0, "skipped": 0, "other": 0}
    for record in records:
        status = record.get("status")
        counts[status if status in counts else "other"] += 1

    if manifest is None:
        state = "no_manifest"
        reason = (
            "the directory has no manifest.json. run.py writes it after the "
            "last job, so this run was interrupted and its records are a "
            "prefix of the matrix rather than the matrix"
        )
    elif (manifest.get("arguments") or {}).get("dry_run"):
        state = "dry_run"
        reason = "--dry-run: the job matrix was written and nothing was executed"
    elif not records:
        state = "dry_run"
        reason = "the manifest is present and no record was written"
    elif counts["error"] or counts["timeout"] or counts["other"]:
        state = "incomplete"
        reason = (
            f"{counts['error']} error and {counts['timeout']} timeout records. "
            "A run that did not complete is a failure and not an absence, and "
            "summarizing the cells that finished would report it as a success"
        )
    else:
        state = "complete"
        reason = f"{counts['ok']} records, all ok, {counts['skipped']} declared skips"
    return {"state": state, "reason": reason, "counts": counts}


def project_engine_name(records):
    """The name this project's engine went by in these records.

    Returns None when no such record is present. Raises when two different
    non-LightGBM engine names appear, which would mean two libraries were
    measured under one run id and no comparison in the file means anything.
    """
    names = {
        record.get("engine")
        for record in records
        if record.get("engine") and record.get("engine") != REFERENCE_ENGINE
    }
    if not names:
        return None
    if len(names) > 1:
        raise ValueError(
            f"records name more than one non-{REFERENCE_ENGINE} engine: "
            f"{sorted(names)}"
        )
    return names.pop()


def _cell_key(record):
    """The same key report.py and verify.py group by, so a cell in the
    summary is the same cell they talk about."""
    return (
        record["scenario"],
        record["engine"],
        record.get("device_used") or record.get("device_requested"),
        record["threads"],
    )


def _all_equal(values):
    """True when a list of JSON-able values is one value repeated."""
    encoded = {json.dumps(value, sort_keys=True, default=str) for value in values}
    return len(encoded) == 1


#: Explanations that would otherwise be repeated verbatim in every cell.
#: A cell that needs one carries the short reason and a `see` pointing
#: here, so the file says each of these once and a reader still reaches it
#: from the field it applies to.
NOTES = {
    "backend_proof_absent": (
        "No record in this run carries a `backend_proof` field, so nothing "
        "in it looked for evidence of which backend ran. run.py writes the "
        "field onto every ok record from the trainer's own phase profile; a "
        "record with no field at all predates the field. This is a "
        "different state from a proof that was sought and came back empty, "
        "and the distinction matters: every device label in this run is a "
        "label the Python side resolved and nothing corroborates it. "
        "backend_proof.py says what would count as corroboration."
    ),
    "params_predate_adapter_capture": (
        "`params.engine` in this run is the dict the translator produced, "
        "not the dict the adapter passed. worker.py was later fixed to take "
        "the dicts back from the adapters, because each adapter adds "
        "parameters the scenario module cannot know: "
        "bin_construct_sample_cnt on the LightGBM side, n_estimators on "
        "this project's, both of them alignment settings. `num_boost_round` "
        "was added by the same change, so its absence is the marker for the "
        "older shape. Read these dicts as the alignment that was intended "
        "rather than the alignment that was applied."
    ),
}


def _backend_proof(group):
    """The trainer's own evidence of which backend ran, or why there is none.

    `run.py` parses this out of the worker's stdout and writes it onto every
    ok record, so an absent field is not an empty proof. The two states are
    kept apart deliberately: nobody looked, against somebody looked and
    found nothing. See backend_proof.py.
    """
    proofs = [record.get("backend_proof") for record in group]
    if all(proof is None for proof in proofs):
        return {
            "value": None,
            "unavailable_reason": "no backend_proof field on any record in this cell",
            "see": "notes.backend_proof_absent",
            "provenance": "unavailable",
        }
    if not _all_equal(proofs):
        return {
            "repeats_disagree": True,
            "per_repeat": proofs,
            "provenance": "measured",
        }
    proof = dict(proofs[0] or {})
    proof["provenance"] = "measured"
    return proof


def _params(group):
    """The dicts the engines were actually passed.

    `worker.py` takes these back from the adapter that used them rather than
    rebuilding them from the scenario, because each adapter adds parameters
    the scenario module cannot know: `bin_construct_sample_cnt` on the
    LightGBM side, `n_estimators` on ours. A record written before that fix
    carries the rebuilt dict, which is missing exactly those, and it is
    missing them silently. `num_boost_round` was added by the same change
    and is absent from no record written after it, so its presence is the
    marker that separates the two, and the summary says which it is looking
    at rather than presenting both as the same evidence.
    """
    first = group[0].get("params") or {}
    out = {
        "shared": first.get("shared"),
        "engine": first.get("engine"),
        "dataset": first.get("dataset"),
        "dataset_unavailable_reason": first.get("dataset_unavailable_reason"),
        "num_boost_round": first.get("num_boost_round"),
        "identical_across_repeats": _all_equal(
            [record.get("params") for record in group]
        ),
        "provenance": "measured",
    }
    if "num_boost_round" not in first:
        out["engine_dict_source"] = "translated, not captured from the adapter"
        out["see"] = "notes.params_predate_adapter_capture"
    else:
        out["engine_dict_source"] = "captured from the adapter that used it"
    return out


def _model(group):
    model = group[0].get("model") or {}
    size = model.get("size") or {}
    out = {
        "num_trees": model.get("num_trees"),
        "num_bin": model.get("num_bin"),
        "serialized_bytes": size.get("string_bytes"),
        "serialized_sha256": size.get("file_sha256"),
        "provenance": "measured",
    }
    bins = model.get("bins")
    if isinstance(bins, dict):
        # The per-feature bin-count vector is not carried; its digest and
        # distribution are, which is what makes the binning alignment
        # checkable without storing a vector per feature.
        out["bins"] = bins
    else:
        out["bins"] = {
            "value": None,
            "unavailable_reason": (
                "this run recorded no per-feature bin counts, so the two "
                "binnings cannot be compared from this file"
            ),
        }
    return out


def _encoding(block):
    """The container an engine was handed, in three fields.

    Absent on a record written before the field existed, and that case is
    reported as unknown rather than as canonical. Defaulting it to
    "canonical" would be the cheap choice and the wrong one: it would make
    an old record and a correctly-canonical new one indistinguishable,
    which is precisely the confusion this field was added to remove.
    """
    encoding = block.get("encoding")
    if not encoding:
        return {
            "form": None,
            "is_canonical": None,
            "agrees_with_canonical": None,
            "note": (
                "this record predates the encoding block. Which container "
                "the engine was handed is not recorded, so it is unknown "
                "rather than canonical"
            ),
        }
    return {
        "form": encoding.get("form"),
        "is_canonical": encoding.get("is_canonical"),
        "agrees_with_canonical": encoding.get("agrees_with_canonical"),
    }


def _data(group):
    data = group[0].get("data") or {}
    train = data.get("train") or {}
    test = data.get("test") or {}

    def part(block):
        out = {
            "rows": block.get("rows"),
            "features": block.get("features"),
            "sparse": block.get("sparse"),
            # The digest of the CANONICAL form. Two engines with equal
            # digests were given the same problem, which is the whole basis
            # of the comparison, and it is checkable from this file alone.
            "digest": block.get("digest"),
            # What each engine was PHYSICALLY handed, which since 2026-08-16
            # is not always the canonical container: CatBoost takes a
            # categorical block only in an integer-typed one. Reduced to the
            # three fields a summary reader needs -- the container's name,
            # whether it is the canonical one, and whether the adapter's
            # reconstruction hashed back to the digest above -- with the
            # argument and the per-column proof left in the record. A
            # summary that carried the digest and not this would let a
            # re-encoded row read as an unencoded one.
            "encoding": _encoding(block),
        }
        for optional in (
            "nnz", "density", "missing_fraction", "queries",
            "label_mean", "label_distinct",
        ):
            if optional in block:
                out[optional] = block[optional]
        return out

    return {
        "dataset": data.get("dataset"),
        "train": part(train),
        "test": part(test),
        "provenance": "measured",
    }


def build_cells(ok_records):
    """One entry per (scenario, engine, device, thread count).

    Repeats are collapsed, and the collapse is checked rather than assumed:
    a cell says how many repeats it holds, whether their metrics were
    identical, and every distinct prediction digest across them. That last
    field is determinism as a measurement. One digest over three repeats is
    the property `thresholds.json` requires of this project's trainer; more
    than one is a regression, and the summary shows it without needing the
    vectors.
    """
    groups = {}
    for record in ok_records:
        groups.setdefault(_cell_key(record), []).append(record)

    cells = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda r: r.get("repeat") or 0)
        first = group[0]
        scenario, engine, device, threads = key
        cells.append(
            {
                "scenario": scenario,
                "scenario_title": first.get("scenario_title"),
                "task": first.get("task"),
                "tier": first.get("tier"),
                "engine": engine,
                "engine_version": first.get("engine_version"),
                "device_requested": first.get("device_requested"),
                # A label the Python side resolved, not evidence. The
                # evidence, where there is any, is in backend_proof below.
                "device_used": first.get("device_used"),
                "threads": threads,
                "path": first.get("path"),
                "repeats": len(group),
                "primary_metric": first.get("primary_metric"),
                "quality": _metrics(first.get("quality")),
                "baseline_quality": _metrics(first.get("baseline_quality")),
                "quality_identical_across_repeats": _all_equal(
                    [record.get("quality") for record in group]
                ),
                "predictions": {
                    "sha256_distinct": sorted(
                        {
                            record.get("predictions_sha256")
                            for record in group
                            if record.get("predictions_sha256")
                        }
                    ),
                    "shape": first.get("predictions_shape"),
                    "provenance": "measured",
                },
                "backend_proof": _backend_proof(group),
                "histogram_builder": first.get("histogram_builder"),
                "params": _params(group),
                "data": _data(group),
                "model": _model(group),
                "peak_rss_bytes": first.get("peak_rss_bytes"),
            }
        )
    return cells


def build_datasets(ok_records):
    """The dataset facts that do not vary by engine, said once.

    Source, license, split rule, pinning and any fallback are properties of
    the data rather than of a run of an engine over it, so they live here
    and the cells carry only the digests they each observed. The digests
    stay per cell because their agreeing is the measurement.
    """
    out = {}
    for record in ok_records:
        data = record.get("data") or {}
        name = data.get("dataset")
        if not name or name in out:
            continue
        entry = {
            "data_kind": data.get("data_kind"),
            "pinned": data.get("pinned"),
            "pin_reason": data.get("pin_reason"),
            # Never silent, per the schema: a real-data scenario that ran on
            # the generator says here why, and a summary that did not carry
            # this field would read as real data.
            "fallback_reason": data.get("fallback_reason"),
            "split": data.get("split"),
            "source": data.get("source"),
        }
        for optional in (
            "generator",
            "generator_kwargs",
            "categorical_feature",
            "category_vocab_sha256",
            "positive_rate",
            # Set by worker.py for a scenario that declares no real dataset
            # at all, which is a different thing from one whose real dataset
            # was missing. `fallback_reason` is null in that case on purpose;
            # this is what says why.
            "no_real_variant",
        ):
            if optional in data:
                entry[optional] = data[optional]
        out[name] = entry
    return out


def build_differential(ok_records, config, project_engine):
    """This project against LightGBM on the CPU, metric by metric.

    Derived, not measured: every number here is arithmetic over metrics the
    records already carry, and no prediction vector is read. Two figures per
    metric, because they answer different questions and the project has
    quoted one of them for the other:

    - `relative_difference` is |ours - theirs| / |theirs|, unsigned, always
      present. It is the number to quote when describing how close the two
      engines are, and reporting it for every metric rather than only for
      the gated ones is what makes the real spread across scenarios
      visible.
    - `gate` is `verify.py`'s comparison, in the units and direction
      `thresholds.json` sets for that scenario, present only where there is
      a rule. Some rules are absolute and some relative, so the gate figure
      and the relative difference are not interchangeable.

    The direction and the gate arithmetic come from `verify.py` itself
    rather than from a second implementation here, so a summary can never
    disagree with the gate about which way a metric runs.

    The pairing is by scenario and thread count at `device == "cpu"`, which
    is the only cross-engine comparison this harness makes: LightGBM runs on
    the CPU here, so an accelerator row against it would compare two
    different things.
    """
    cells = {}
    for record in ok_records:
        device = record.get("device_used") or record.get("device_requested")
        cells.setdefault((record["scenario"], record["threads"]), {}).setdefault(
            (record["engine"], device), []
        ).append(record)

    out = []
    for (scenario, threads), by_engine in sorted(cells.items()):
        mine = by_engine.get((project_engine, "cpu"))
        theirs = by_engine.get((REFERENCE_ENGINE, "cpu"))
        if not mine or not theirs:
            continue
        a, b = mine[0], theirs[0]
        rules = verify.scenario_rules(config, a)
        primary = rules.get("primary_metric", a.get("primary_metric"))
        gated = dict(rules.get("secondary") or {})
        if rules.get("differential"):
            gated[primary] = rules["differential"]

        metrics = []
        for name in sorted(set(a.get("quality") or {}) & set(b.get("quality") or {})):
            got, want = _finite((a["quality"] or {})[name]), _finite((b["quality"] or {})[name])
            if got is None or want is None:
                metrics.append(
                    {
                        "metric": name,
                        "role": "primary" if name == primary else "scored",
                        "value": {project_engine: got, REFERENCE_ENGINE: want},
                        "relative_difference": None,
                        "unavailable_reason": "one side scored NaN on this metric",
                    }
                )
                continue
            entry = {
                "metric": name,
                "role": (
                    "primary" if name == primary
                    else "secondary_gated" if name in gated
                    else "scored_not_gated"
                ),
                "value": {project_engine: got, REFERENCE_ENGINE: want},
                "relative_difference": abs(got - want) / abs(want) if want else None,
                "provenance": "derived",
            }
            if not want:
                entry["unavailable_reason"] = (
                    "the reference value is zero, so a relative difference is "
                    "undefined. The two values are above"
                )
            spec = gated.get(name)
            if spec:
                # verify.py's own comparison, private name and all. A second
                # implementation here could disagree with the gate about
                # which direction of a metric is a regression, and a summary
                # that disagreed with the gate would be worse than no
                # summary.
                worse = verify._worse_by(name, got, want, spec["kind"])
                entry["gate"] = {
                    "kind": spec["kind"],
                    "worse_by": worse,
                    "max_worse": spec["max_worse"],
                    "within_limit": worse <= spec["max_worse"],
                    "provenance": "derived",
                }
                if "implausible_better" in spec:
                    entry["gate"]["implausible_better"] = spec["implausible_better"]
                    entry["gate"]["implausibly_better"] = (
                        -worse > spec["implausible_better"]
                    )
            metrics.append(entry)

        out.append(
            {
                "scenario": scenario,
                "threads": threads,
                "device": "cpu",
                "dataset": (a.get("data") or {}).get("dataset"),
                "data_kind": (a.get("data") or {}).get("data_kind"),
                "primary_metric": primary,
                "metrics": metrics,
            }
        )
    return out


def build_device_agreement(ok_records, project_engine):
    """This project's accelerator arm against its own CPU arm.

    Two of `verify.py`'s three device checks can be made from a record. The
    third cannot, and the gap is the clearest example of what a summary is
    not: `verify.py` compares the two prediction vectors row by row, and the
    vectors are not committed. That check is recorded here as unavailable
    with its reason rather than left out, so nobody reads the two that are
    here as all three.

    The digest comparison is the check that caught a real bug. Covertype's
    CPU and accelerator arms came back with byte-identical
    `predictions_sha256`, which is how `train_dataset_multiclass` was found
    resolving the device and then discarding the answer. Equal digests are
    not by themselves wrong, since the device histogram reduction is fixed
    point so that bit-exact agreement between backends is reachable, but
    they are always worth saying out loud.
    """
    cells = {}
    for record in ok_records:
        if record.get("engine") != project_engine:
            continue
        device = record.get("device_used") or record.get("device_requested")
        cells.setdefault((record["scenario"], record["threads"]), {}).setdefault(
            device, record
        )

    out = []
    for (scenario, threads), by_device in sorted(cells.items()):
        cpu = by_device.get("cpu")
        for device, record in sorted(by_device.items()):
            if device == "cpu" or cpu is None:
                continue
            metric = record.get("primary_metric")
            got = _finite((record.get("quality") or {}).get(metric))
            want = _finite((cpu.get("quality") or {}).get(metric))
            out.append(
                {
                    "scenario": scenario,
                    "threads": threads,
                    "device": device,
                    "primary_metric": metric,
                    "value": {device: got, "cpu": want},
                    "relative_difference": (
                        abs(got - want) / abs(want)
                        if got is not None and want else None
                    ),
                    "predictions_sha256": {
                        device: record.get("predictions_sha256"),
                        "cpu": cpu.get("predictions_sha256"),
                        "identical": (
                            record.get("predictions_sha256")
                            == cpu.get("predictions_sha256")
                        ),
                    },
                    "row_level_agreement": {
                        "value": None,
                        "unavailable_reason": (
                            "max |device - cpu| over the rows needs both "
                            "prediction vectors, which are not committed. "
                            "verify.py computes it on the machine that holds "
                            "them"
                        ),
                    },
                    "provenance": "derived",
                }
            )
    return out


_LOCAL_PATH = re.compile(r"/(?:Users|home)/[^\"\s,\]]+")


def redact_local(value):
    """Strip machine-local paths and the hostname from anything committed.

    These summaries go into version control in a public repository, and the
    provenance they carry is the run's identity rather than the operator's
    filesystem. A home directory path and a hostname identify a person and a
    machine and settle nothing a reader of this file needs settled, so they
    are replaced rather than kept.

    Redaction happens here rather than as a pass over the written file so that
    it cannot be forgotten. A summary is a pure function of its run directory,
    and that has to stay true of the redacted form too, or regenerating a
    committed summary would produce a diff.
    """
    if isinstance(value, str):
        out = _LOCAL_PATH.sub("<redacted-local-path>", value)
        return "<redacted-hostname>" if out == "Mac" else out
    if isinstance(value, list):
        return [redact_local(v) for v in value]
    if isinstance(value, dict):
        return {k: redact_local(v) for k, v in value.items()}
    return value


def build_environment(manifest, records):
    """The machine, the toolchain, and whether every record agrees on them.

    The manifest's block is the run's environment, collected once by the
    runner. Each record carries its own, collected in its own worker
    process, and those can legitimately differ from the runner's in the
    thread and device variables the runner sets per job. What must not
    differ is `envinfo.comparable_key`: host, architecture, CPU model, mojo
    and lightgbm versions, and the commit. Every distinct key seen is listed
    rather than reduced to the first, because a run that changed machines
    or builds partway through is not one table of results and should not be
    summarized as though it were.
    """
    keys = []
    for record in records:
        key = envinfo.comparable_key(record.get("environment") or {})
        if key not in keys:
            keys.append(key)
    return {
        "run": (manifest or {}).get("environment"),
        "comparable_keys_across_records": keys,
        "single_environment": len(keys) <= 1,
        "provenance": "measured",
    }


ABOUT = [
    "One committed summary of one real-data differential run. The metrics, "
    "the digests, the parameters and the environment; not the predictions, "
    "not the timings, not a verdict.",
    "Evidence that this run happened and what it produced. Not a "
    "reproduction: every number was measured once, on the machine and at "
    "the commit named in `environment`. Two summaries from two machines "
    "will differ in ways that are legitimate, and a diff between them is a "
    "question rather than an answer.",
    "Timings are deliberately absent. results/README.md gives the reason: a "
    "timing on a laptop with a thermal budget is a fact about an afternoon, "
    "and report.py exists to show its distribution to a person who will "
    "then write the sentence. A committed number would travel without the "
    "distribution.",
    "Fields carry a `provenance`: `measured` is copied from a record as the "
    "harness wrote it, `derived` is arithmetic over measured values with no "
    "new observation, and a value the run did not record appears as null "
    "with an `unavailable_reason` rather than being omitted or estimated.",
    "Regenerate with `python bench/real_data/summarize.py results/<run_id>`. "
    "The output is a pure function of the run directory, so regenerating a "
    "committed summary from the same run rewrites the same bytes.",
]


def summarize(run_dir, allow_incomplete=False):
    manifest, records = load_run(run_dir)
    state = assess(manifest, records)
    if state["state"] != "complete" and not allow_incomplete:
        raise ValueError(f"{os.path.basename(run_dir)}: {state['reason']}")

    ok = [record for record in records if record.get("status") == "ok"]
    project_engine = project_engine_name(ok)
    config = verify.thresholds()

    payload = {
        "artifact": "mojotrees real-data run summary",
        "summary_version": SUMMARY_VERSION,
        "written_by": "bench/real_data/summarize.py",
        "about": ABOUT,
        "run": {
            "run_id": (manifest or {}).get("run_id") or os.path.basename(run_dir.rstrip("/")),
            "directory": os.path.basename(run_dir.rstrip("/")),
            "created_utc": (manifest or {}).get("created_utc"),
            "arguments": (manifest or {}).get("arguments"),
            "completeness": state,
            "record_schema_version": (ok[0].get("schema_version") if ok else None),
            "thresholds_version": config.get("version"),
        },
        "engines": {
            "project": project_engine,
            "reference": REFERENCE_ENGINE,
            "naming_note": (
                f"this project's engine is named `{PROJECT_ENGINE}` today. Runs "
                f"taken before the 2026-08-15 rename carry "
                f"{', '.join('`' + name + '`' for name in FORMER_PROJECT_ENGINE_NAMES)} "
                "in every record and in every key derived from one. The "
                "summarizer reports the name the run used and never rewrites "
                "it"
            ),
        },
        "environment": build_environment(manifest, records),
        "datasets": build_datasets(ok),
        "differential": build_differential(ok, config, project_engine),
        "device_agreement": build_device_agreement(ok, project_engine),
        "cells": build_cells(ok),
        "caveats": _caveats(ok),
        # From the manifest rather than from the records. run.py appends a
        # skipped row to records.json at the end of a run and writes no file
        # for it under records/, so a summary built from the record files
        # alone would report a matrix with nine declared skips as a matrix
        # of forty-five cells and no skips.
        "skipped": _skipped(manifest),
        "not_decided_here": [
            "Pass or fail. verify.py applies thresholds.json and exits "
            "non-zero; this file carries the inputs to that decision.",
            "Row-level agreement between the accelerator and the CPU, which "
            "needs the prediction vectors.",
            "Anything about speed.",
        ],
    }
    payload["flags"] = build_flags(payload)
    payload["notes"] = _referenced_notes(payload)
    # Redacted at the boundary, so nothing machine-local can reach a
    # committed file however the blocks above are later extended.
    return redact_local(payload)


def build_flags(payload):
    """The things in this run a reader should not have to find.

    Derived, and deliberately not a verdict. `verify.py` decides; this is a
    list of facts already in the file that a reader scanning eighty
    kilobytes of JSON would otherwise have to go looking for, each one
    stated with the numbers that produced it. An empty list means none of
    these particular things is true of this run, and it does not mean the
    run passed.

    Each flag restates a condition the harness already treats as important
    somewhere: `thresholds.json` gates on the differential and on repeat
    determinism, `report.py` refuses to put two builds in one table,
    `verify.py` refuses an uncorroborated device claim that reproduced its
    CPU twin, and `loaders.py` records a fallback rather than performing one
    silently.
    """
    flags = []

    keys = payload["environment"]["comparable_keys_across_records"]
    if len(keys) > 1:
        commits = sorted({key.get("git_commit") for key in keys if key.get("git_commit")})
        flags.append(
            {
                "flag": "multiple_builds",
                "scope": "run",
                "detail": (
                    f"records in this run were produced at {len(keys)} distinct "
                    f"environments, {len(commits)} distinct commits: "
                    + ", ".join(commit[:12] for commit in commits)
                    + ". report.py would split this run into one table per "
                    "environment, and a cross-engine comparison inside it is "
                    "not necessarily a comparison at one build"
                ),
            }
        )

    for entry in payload["differential"]:
        for metric in entry["metrics"]:
            gate = metric.get("gate")
            if not gate:
                continue
            if not gate["within_limit"]:
                flags.append(
                    {
                        "flag": "differential_over_threshold",
                        "scope": f"{entry['scenario']}/{metric['metric']}",
                        "detail": (
                            f"worse by {gate['worse_by']:.4g} against a limit of "
                            f"{gate['max_worse']} ({gate['kind']}). This cell is "
                            "not parity and quoting the run as a whole without "
                            "it would be quoting a selection"
                        ),
                    }
                )
            elif gate.get("implausibly_better"):
                flags.append(
                    {
                        "flag": "implausibly_better",
                        "scope": f"{entry['scenario']}/{metric['metric']}",
                        "detail": (
                            f"better by {-gate['worse_by']:.4g}, past the "
                            f"{gate['implausible_better']} that thresholds.json "
                            "reads as a sign the two engines were not given the "
                            "same problem"
                        ),
                    }
                )

    for cell in payload["cells"]:
        if len(cell["predictions"]["sha256_distinct"]) > 1:
            flags.append(
                {
                    "flag": "repeats_not_identical",
                    "scope": f"{cell['scenario']}/{cell['engine']}/{cell['device_used']}",
                    "detail": (
                        f"{len(cell['predictions']['sha256_distinct'])} distinct "
                        f"prediction digests across {cell['repeats']} repeats"
                    ),
                }
            )

    proof_absent = any(
        isinstance(cell["backend_proof"], dict)
        and cell["backend_proof"].get("provenance") == "unavailable"
        for cell in payload["cells"]
        if cell["device_used"] not in (None, "cpu")
    )
    for entry in payload["device_agreement"]:
        if entry["predictions_sha256"]["identical"]:
            flags.append(
                {
                    "flag": "device_digest_equals_cpu",
                    "scope": f"{entry['scenario']}/{entry['device']}",
                    "detail": (
                        f"the {entry['device']} arm and the cpu arm produced the "
                        "same prediction digest"
                        + (
                            ". Nothing in this run proves a device backend ran, "
                            "and those two facts together are what a CPU fit "
                            "wearing a device label looks like from the outside. "
                            "It is also what two backends agreeing bit for bit "
                            "looks like, which is a property the fixed-point "
                            "device reduction is built to have. This file cannot "
                            "tell them apart"
                            if proof_absent
                            else ". The backend proof on this cell says a device "
                            "backend ran, so this is two backends agreeing bit "
                            "for bit"
                        )
                    ),
                }
            )

    for name, dataset in sorted(payload["datasets"].items()):
        if dataset.get("fallback_reason"):
            flags.append(
                {
                    "flag": "generator_fallback",
                    "scope": name,
                    "detail": (
                        "a scenario that names a real dataset ran on the "
                        f"generator instead: {dataset['fallback_reason']}. Its "
                        "numbers are not a real-data result"
                    ),
                }
            )
        if dataset.get("data_kind") == "real" and not dataset.get("pinned"):
            flags.append(
                {
                    "flag": "unpinned_real_data",
                    "scope": name,
                    "detail": dataset.get("pin_reason")
                    or "real data was used without a pin, so the bytes are unverified",
                }
            )

    return flags


def _skipped(manifest):
    """The cells the matrix declared and did not run, with the reason.

    A scenario that declares no accelerator path is skipped rather than
    failed, and the difference has been misread before: "the GPU errored on
    four of six scenarios" was a reading of three declared skips. The
    summary carries them so the same reading cannot be made from it.
    """
    return [
        {
            "scenario": job.get("scenario"),
            "engine": job.get("engine"),
            "device": job.get("device"),
            "reason": job.get("skip"),
        }
        for job in (manifest or {}).get("jobs", [])
        if "skip" in job
    ]


def _referenced_notes(payload):
    """Exactly the notes something in this file points at.

    Carrying the whole table would put an explanation of a missing backend
    proof into a summary of a run that has one, which is how a boilerplate
    caveat becomes furniture nobody reads.
    """
    wanted = set()

    def walk(node):
        if isinstance(node, dict):
            reference = node.get("see")
            if isinstance(reference, str) and reference.startswith("notes."):
                wanted.add(reference.split(".", 1)[1])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(payload)
    return {key: NOTES[key] for key in sorted(wanted) if key in NOTES}


def _caveats(ok_records):
    """The scenario caveats, once per scenario rather than once per cell.

    They are copied into every record the scenario produces, so carrying
    them per cell would repeat a paragraph fifteen times. They are carried
    at all because a caveat is a statement about what the comparison is
    worth, and a summary of a comparison that dropped them would read as a
    cleaner result than the run was.
    """
    out = {}
    for record in ok_records:
        caveats = record.get("caveats") or []
        if caveats:
            out.setdefault(record["scenario"], sorted(set(caveats)))
    return out


def write(payload, path):
    with open(path, "w") as handle:
        # allow_nan=False so a NaN that escaped _finite fails here rather
        # than being written as a token no other JSON parser will read.
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", nargs="+", help="one or more results/<run_id> directories")
    parser.add_argument(
        "--stdout", action="store_true",
        help="write to stdout instead of results/<run_id>/summary.json",
    )
    parser.add_argument(
        "--allow-incomplete", action="store_true",
        help="summarize a run that did not complete. The summary records "
             "which state it was in; committing one is a decision to be "
             "made deliberately, not a default",
    )
    args = parser.parse_args(argv)

    failures = 0
    for run_dir in args.run:
        run_dir = run_dir.rstrip("/")
        try:
            payload = summarize(run_dir, allow_incomplete=args.allow_incomplete)
        except (ValueError, OSError) as exc:
            print(f"skip {exc}", file=sys.stderr)
            failures += 1
            continue
        if args.stdout:
            json.dump(payload, sys.stdout, indent=2, allow_nan=False)
            sys.stdout.write("\n")
            continue
        path = os.path.join(run_dir, "summary.json")
        write(payload, path)
        print(
            f"wrote {path} "
            f"({os.path.getsize(path) / 1024:.1f} KiB, "
            f"{len(payload['cells'])} cells, "
            f"{len(payload['differential'])} differential scenarios)"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
