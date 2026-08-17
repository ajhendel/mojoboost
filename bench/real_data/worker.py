"""One measured run, in its own process.

The runner never trains anything itself. It spawns this module once per
(scenario, ARM, device, threads, repeat), and this module builds the data,
runs one engine, scores the predictions, and writes one JSON record plus the
predictions to disk.

`arm` is the cell's identity within an engine and defaults to the engine
name, so a job from a matrix built without `--arms` carries `arm == engine`
and this module behaves exactly as it did before the dimension existed. A job
from an arm additionally carries `arm_params` and `arm_dataset_params`, which
`engines.build` folds into the training and the binning parameters
respectively, and `axis`, which says which axis of a sweep the arm moves. All
of it is written onto the record, because a timing that cannot say which arm
produced it is a timing of an unknown model.

A separate process per run is not tidiness. It is what makes three of the
measurements mean anything:

- Peak resident set is a property of a process. Measured after a second
  model has been trained in the same process, it is the larger of the two
  and attributable to neither.
- Thread count is read from the environment by one of the two libraries,
  and an environment set after import may or may not have taken.
- Import and first-call warmup happen once per process. In a shared
  process the second engine measured would always look faster to load.

Both engines rebuild the data independently from the same deterministic
recipe, and each records a digest of what it built. The runner refuses to
compare two records whose data digests differ, so "the two engines were
given the same data" is checked rather than assumed.

The same applies to the backend. `device_used` is a label the Python side
resolved, so this module turns the trainer's own instrument on for exactly
the measured fit and leaves the evidence on the process's stdout for
run.py to read back into `backend_proof`. What that establishes, and what
it does not, is backend_proof.py's docstring.

    python bench/real_data/worker.py --job job.json --out record.json
"""

import argparse
import json
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# The mojotrees package this harness measures is the checkout's own
# python/ tree; the bench environment deliberately does not install it, so
# the worker puts it on the path itself, after the harness modules so a
# scenario module can never be shadowed by a package file.
_REPO_PYTHON = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
    "python",
)
if os.path.isdir(_REPO_PYTHON) and _REPO_PYTHON not in sys.path:
    sys.path.insert(1, _REPO_PYTHON)

import backend_proof  # noqa: E402
import engines  # noqa: E402
import envinfo  # noqa: E402
import generators  # noqa: E402
import loaders  # noqa: E402
import measure  # noqa: E402
import quality  # noqa: E402
import scenarios  # noqa: E402


#: The one canonical form of a scenario's data: a float64 feature matrix (or
#: a CSC one), a float64 label, and an int64 group vector where there is one.
#: `data_digest` is a digest of exactly that, and it is what
#: `verify.check_data_agreement` compares across engines.
#:
#: **It is not a digest of what every engine physically receives, and since
#: 2026-08-16 it does not claim to be.** CatBoost refuses `cat_features` on a
#: floating-point array outright, so on a categorical scenario its adapter
#: re-encodes the declared categorical columns into integer columns and hands
#: CatBoost a mixed-dtype frame. That is a second ENCODING of the canonical
#: form and not a second dataset, and the distinction is the whole reason
#: this constant and `ENCODING_CONTRACT` below exist rather than the harness
#: quietly exempting one engine from the digest check.
#:
#: The rule, stated once here and enforced in two places:
#:
#:   1. The canonical digest is computed from the canonical form and from
#:      nothing else. Every engine's record carries the same one on a given
#:      scenario or `verify.py` fails the scenario. There is no per-engine
#:      exemption, because an exemption would turn the one check that proves
#:      the arms saw the same problem into a check that proves nothing.
#:   2. An engine that cannot take the canonical form re-encodes it, and the
#:      re-encoding must be a BIJECTION on the values -- reconstruct the
#:      canonical matrix from the re-encoding, hash it with the same
#:      function, and the two digests must agree. The adapter reports the
#:      reconstruction digest, this module compares it, and the record
#:      carries the verdict per run. A re-encoding that cannot pass that
#:      test is not an encoding of the same data and the adapter refuses it.
#:
#: So a reader of a record sees: one `digest` field, equal across engines,
#: which says they were given the same problem; and one `encoding` block,
#: which says what container each engine was handed and carries the in-run
#: proof that the container did not change the values. Those are two
#: different questions and they now have two different fields.
CANONICAL_ENCODING = "float64_matrix"

ENCODING_CONTRACT = (
    "the canonical form is a float64 feature matrix, a float64 label and an "
    "int64 group vector, and `digest` is over exactly that. An engine that "
    "cannot take it re-encodes it and reports a digest of the canonical "
    "matrix RECONSTRUCTED from the re-encoding; `agrees_with_canonical` is "
    "this module comparing the two. true means the engine was given the "
    "same values in a different container. false means it was given "
    "different data and nothing on that scenario is comparable"
)


def data_digest(part):
    """A digest of exactly the bytes the CANONICAL form holds.

    Not necessarily the bytes an engine physically received: see
    `CANONICAL_ENCODING` for why those are two questions and where the
    second one is answered.

    The byte stream is UNCHANGED from the version that wrote every record in
    bench/results: the matrix's pieces, then the label, then the group, fed
    to one running sha256. The body moved to `measure.canonical_digest` so
    that the CatBoost adapter's reconstruction check hashes the identical
    way rather than a way that looks identical. A digest whose definition
    moves silently makes old records and new ones incomparable, which is the
    failure this note exists to prevent.
    """
    return measure.canonical_digest(
        part["X"], part["y"], part.get("group")
    )


def build_data(spec, variant, allow_unpinned):
    """(train, test, meta). Real data when it is available and pinned and
    the scenario asked for it, the generator otherwise, and the record
    always says which and why."""
    dataset_id = spec.get("dataset")
    if variant != "synthetic" and dataset_id:
        try:
            train, test, meta = loaders.load(dataset_id, allow_unpinned)
            meta["fallback_reason"] = None
            return train, test, meta
        except loaders.DataUnavailable as exc:
            if variant == "real":
                raise
            fallback = str(exc)
    else:
        fallback = None

    generator = generators.GENERATORS[spec["generator"]]
    data = generator(**spec["generator_kwargs"])
    train, test = generators.split(data)
    meta = {
        "dataset": f"generated:{spec['generator']}",
        "data_kind": "synthetic",
        "pinned": True,
        "generator": spec["generator"],
        "generator_kwargs": spec["generator_kwargs"],
        "task": spec["task"],
        "split": {"kind": "hash", "train_fraction": 0.8, "seed": 1900},
        # A fallback is a real dataset that was WANTED and was not there, and
        # `summarize.py` raises a `generator_fallback` flag on it reading "a
        # scenario that names a real dataset ran on the generator instead".
        # A scenario with `dataset=None` never wanted one, so filling this in
        # for it would put a true-sounding and false sentence in every
        # summary. Until 2026-08-16 every scenario named a dataset and the
        # two cases could not be told apart; `high_cardinality_categorical`
        # and `ordered_boosting_small` are the first that have none, so the
        # distinction is recorded in its own field instead.
        "fallback_reason": fallback,
    }
    if not dataset_id:
        meta["no_real_variant"] = (
            "this scenario declares no real dataset, so the generator is not "
            "a fallback, it is the only variant. A record from it is a "
            "synthetic-data result and must not be quoted as a real-data one"
        )
    # The known Bayes floor, on the record, for the generator variant only.
    # `scenarios.bayes_floor` returns None for every scenario that declares
    # none, and this branch is the synthetic one by construction, so a real
    # record never carries the field. Readers fall back to the same helper for
    # records written before 2026-08-17.
    floor = scenarios.bayes_floor(spec["id"], "synthetic", spec["generator_kwargs"])
    if floor:
        # Both floors: the population value the scenario declares and the
        # noise REALIZED on these held-out rows, which is what the excess is
        # measured against. `generators.with_realized_floor` is the one place
        # that knows how to reproduce the second, so a reader of an old record
        # and the writer of a new one cannot disagree about it.
        meta["bayes_floor"] = generators.with_realized_floor(
            floor, spec["generator_kwargs"], meta["split"]
        )
    for key in ("categorical_feature", "n_classes", "sparse"):
        if key in train:
            meta[key] = train[key] if key != "sparse" else True
    return train, test, meta


def describe(part, name):
    x = part["X"]
    out = {
        "rows": int(x.shape[0]),
        "features": int(x.shape[1]),
        "sparse": bool(hasattr(x, "tocsc")),
        "digest": None,
        # Overwritten by `apply_encoding_report` for an engine that could
        # not take the canonical form. Filled in here rather than left
        # absent so that a record from an engine that took it says so,
        # instead of a reader having to infer "no encoding block" as "the
        # canonical one" -- which is exactly the inference that would go
        # wrong the first time a block failed to be written.
        "encoding": {
            "form": CANONICAL_ENCODING,
            "is_canonical": True,
            "agrees_with_canonical": True,
            "contract": ENCODING_CONTRACT,
        },
    }
    if out["sparse"]:
        out["nnz"] = int(x.nnz)
        out["density"] = out["nnz"] / float(out["rows"] * out["features"])
    else:
        out["missing_fraction"] = float(np.isnan(x).mean())
    if part.get("group") is not None:
        out["queries"] = int(len(part["group"]))
    if name == "train":
        y = np.asarray(part["y"], dtype=np.float64)
        out["label_mean"] = float(y.mean())
        out["label_distinct"] = int(min(len(np.unique(y)), 1000))
    return out


def apply_encoding_report(desc, report):
    """Fold an adapter's re-encoding report into a part's description, and
    decide the one question the report cannot decide for itself.

    The adapter knows what it built and can hash the canonical matrix it
    reconstructed from what it built. It does NOT know the canonical digest
    -- that is computed here, from the canonical data, before any engine
    sees it -- so it cannot be the thing that says the two agree. Comparing
    them is this module's job, for the same reason the digest is: a check
    that the object being checked gets to run on itself is not a check.

    `agrees_with_canonical` is therefore one of three values and every one
    of them is a different statement:

      true   the reconstruction hashed to the canonical digest. The engine
             was given the same values in a different container.
      false  it did not. The engine was given different data, the scenario's
             rows are not comparable, and `verify.py` should be reading this
             field. This is a bug in the adapter, not a caveat.
      null   there is nothing to compare against, because the run was asked
             for no data digest (`--no-data-digest`). The re-encoding was
             still checked for losslessness column by column inside the
             adapter, which is a weaker statement, and `proof` carries it.
    """
    if not report:
        return
    canonical = desc.get("digest")
    rebuilt = report.get("canonical_digest_recomputed")
    if canonical is None or rebuilt is None:
        agrees = None
    else:
        agrees = bool(canonical == rebuilt)
    desc["encoding"] = {
        "contract": ENCODING_CONTRACT,
        "is_canonical": report.get("form") == CANONICAL_ENCODING,
        "agrees_with_canonical": agrees,
        **report,
    }
    if agrees is None and canonical is None:
        desc["encoding"]["agrees_unavailable_reason"] = (
            "no canonical digest was computed for this run, so the "
            "reconstruction has nothing to be compared against. The "
            "per-column losslessness proof in `proof` still ran"
        )


def _record_extra(task, train):
    """The per-dataset parameters the record's shared block needs.

    The engines derive `num_class` from the loaded data before training,
    and the shared block has to derive it the same way, or the canonical
    parameters in the record would omit a parameter both engines were
    given. The engine's own dict is not built here at all: it comes back
    from the adapter that used it.
    """
    if task != "multiclass":
        return None
    return {
        "num_class": int(train.get("n_classes") or (np.max(train["y"]) + 1))
    }


def run_job(job):
    spec = scenarios.resolve(job["scenario"], job["tier"], job["variant"])
    train, test, data_meta = build_data(
        spec, job["variant"], job.get("allow_unpinned", False)
    )
    task = data_meta.get("task", spec["task"])
    if task != spec["task"]:
        # The real variant of a scenario can be a different task from its
        # generator; adult is a classifier where the generator is a
        # regression. The record carries whichever actually ran.
        spec["task"] = task
        spec["objective"] = {"binary": "binary", "regression": "regression"}[task]
        spec["primary_metric"] = quality.TASK_METRICS[task][0]

    train_desc, test_desc = describe(train, "train"), describe(test, "test")
    if job.get("data_digest", True):
        train_desc["digest"] = data_digest(train)
        test_desc["digest"] = data_digest(test)

    # CatBoost's resolved parameters for the cells this run has already
    # measured. Only `mojotrees_catboost_mode` reads it, and it refuses to
    # build without the entry for its own cell rather than guessing a learning
    # rate: see scenarios.MOJOTREES_CATBOOST_MODE_FROM_READBACK. The file is
    # written below by whichever CatBoost cell ran first, and `run.py` orders
    # the CatBoost cell ahead of the CatBoost-mode cell inside each round.
    catboost_readback = None
    readback_path = job.get("catboost_readback_path")
    if readback_path and os.path.exists(readback_path):
        with open(readback_path) as handle:
            catboost_readback = json.load(handle)

    # The arm's parameter overrides, carried from the matrix into the engine.
    # `arm_params` folds into the engine's TRAINING parameters and
    # `arm_dataset_params` into the BINNING ones, which is the same split
    # `scenarios.dataset_params` already draws: both libraries reject max_bin
    # on `train`. A job with no arm carries two empty dicts and the engine
    # makes the call it made before this existed.
    engine = engines.build(
        job["engine"], job["threads"], job["device"], catboost_readback,
        arm_params=job.get("arm_params"),
        arm_dataset_params=job.get("arm_dataset_params"),
    )
    engine.load()
    warmup = engine.warmup(spec)
    # The trainer's own instrument, on for the measured fit and off for
    # everything around it. Off for the warm-up above deliberately: the
    # warm-up fits one round on a tiny matrix through the same entry point,
    # so a block from it would be evidence about a shape nobody measured,
    # and evidence about the wrong shape is how a proof stops proving
    # anything. See backend_proof.py for what the blocks establish, what
    # they do not, and what the instrument costs.
    want_proof = job.get("backend_proof", True)
    previous_profile = backend_proof.enable() if want_proof else None
    try:
        result = engine.run(
            spec, train, test, repeats=job.get("predict_repeats", 3)
        )
    finally:
        if want_proof:
            backend_proof.restore(previous_profile)
    predictions = result.pop("predictions")
    # The engine's own dicts, not a second call to the translators. Each
    # adapter adds to the translated dict after it gets it, and rebuilding
    # the dict here dropped exactly those additions: every LightGBM record
    # written before this was missing bin_construct_sample_cnt and every
    # mojotrees record was missing n_estimators. Both are alignment
    # settings, and the records were being read as evidence of alignment.
    params_used = result.pop("params_used")
    # The other half of the handover. A CatBoost cell writes its own resolved
    # parameters into the run's sidecar, keyed by (scenario, tier, variant), so
    # the CatBoost-mode cell for that same cell can read them in a later
    # process. Popped rather than left in `result` because it is a channel
    # between two cells and not a field of this record; what the record carries
    # is `engine_resolved_params`, which was already there.
    readback_entry = result.pop("catboost_readback", None)
    if readback_entry and readback_path:
        scenarios.append_catboost_readback(readback_path, readback_entry)
    dataset_params_used = result.pop("dataset_params_used", None)
    dataset_params_reason = result.pop("dataset_params_unavailable_reason", None)
    num_boost_round = result.pop("num_boost_round", None)
    # What the engine was PHYSICALLY handed, against what the canonical form
    # holds. Popped rather than left in `result` so it lands inside
    # `data.train` and `data.test` beside the digest it has to be read with,
    # rather than in a top-level field a reader could look at on its own.
    encoding_report = result.pop("data_encoding", None) or {}
    apply_encoding_report(train_desc, encoding_report.get("train"))
    apply_encoding_report(test_desc, encoding_report.get("test"))

    scores = quality.score(
        task, test["y"], predictions, group=test.get("group")
    )
    baseline = quality.trivial_baseline(
        task, train["y"], test["y"], test.get("group")
    )

    record = {
        "schema_version": 1,
        "status": "ok",
        "run_id": job["run_id"],
        "job_index": job["job_index"],
        "repeat": job["repeat"],
        # The arm dimension, on the record rather than only in the filename.
        # `arm` defaults to the engine name in `run.py`, so a record from a
        # matrix with no arms carries arm == engine and a reader who groups on
        # `arm` gets the same groups they got from `engine`. A reader of an
        # OLDER record should take `record.get("arm") or record["engine"]`.
        "arm": job.get("arm") or job["engine"],
        "axis": job.get("axis"),
        "axis_value": job.get("axis_value"),
        "arm_block": job.get("arm_block"),
        # What this cell is FOR, as distinct from what it is. `measured` or
        # `oracle`; run.py's `_mark_oracle_cells` decides, because the decision
        # is a property of the whole matrix and this process sees one job. The
        # note travels with it so that a single record file, read on its own,
        # says why one cpu row sits beside three gpu rows.
        "cell_role": job.get("cell_role") or "measured",
        "cell_role_note": job.get("cell_role_note"),
        # What the arm actually asked to move, beside the resolved dicts
        # below. A record that carried only the resolved parameters could not
        # say which of them the arm CHOSE and which the translator supplied.
        "arm_overrides": {
            "params": dict(job.get("arm_params") or {}),
            "dataset_params": dict(job.get("arm_dataset_params") or {}),
            "env": dict(job.get("arm_env") or {}),
        },
        "scenario": spec["id"],
        "scenario_title": spec["title"],
        "tier": spec["tier"],
        "task": task,
        "primary_metric": spec.get("primary_metric", quality.TASK_METRICS[task][0]),
        "threads": job["threads"],
        "data": {**data_meta, "train": train_desc, "test": test_desc},
        "params": {
            "shared": scenarios.shared_params(spec, _record_extra(task, train)),
            "engine": params_used,
            "dataset": dataset_params_used,
            "dataset_unavailable_reason": dataset_params_reason,
            "num_boost_round": num_boost_round,
            # Both engines' resolved dicts and the key-by-key verdict, on
            # EVERY row rather than only in the manifest. A record used to
            # carry what its own engine was passed and nothing about what the
            # engine it is compared against resolved, so a published ratio was
            # readable only by somebody who also held the other row and knew
            # which parameters each library derives for itself.
            "resolved_parity": scenarios.record_parity_block(
                spec,
                job["engine"],
                params_used,
                dataset_params_used,
                catboost_readback,
            ),
        },
        "caveats": list(spec.get("caveats", [])) + list(result.get("notes", [])),
        "quality": scores,
        "baseline_quality": baseline,
        "predictions_sha256": measure.digest(predictions),
        "predictions_shape": list(np.asarray(predictions).shape),
        # Placeholder until run.py, which owns the worker's stdout, fills
        # in what the trainer printed. Never omitted, so a record that was
        # not run under run.py says why it has no proof instead of looking
        # like a run whose proof came back empty.
        "backend_proof": backend_proof.pending(want_proof),
        "warmup": warmup.as_dict(),
        **{k: v for k, v in result.items() if k != "notes"},
    }
    return record, np.asarray(predictions, dtype=np.float64)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--job", required=True, help="path to a job JSON file")
    parser.add_argument("--out", required=True, help="where to write the record")
    parser.add_argument("--predictions", help="where to write predictions as .npy")
    args = parser.parse_args(argv)

    with open(args.job) as handle:
        job = json.load(handle)

    try:
        record, predictions = run_job(job)
        if args.predictions:
            np.save(args.predictions, predictions)
            record["predictions_path"] = os.path.basename(args.predictions)
    except Exception as exc:  # noqa: BLE001 - a failed run is a recorded result
        record = {
            "schema_version": 1,
            "status": "error",
            "run_id": job.get("run_id"),
            "job_index": job.get("job_index"),
            "repeat": job.get("repeat"),
            "scenario": job.get("scenario"),
            "tier": job.get("tier"),
            "engine": job.get("engine"),
            "arm": job.get("arm") or job.get("engine"),
            "axis": job.get("axis"),
            # Written on the error record too (2026-08-17), so every record
            # kind carries the arm's declared block beside its arm id.
            "arm_block": job.get("arm_block"),
            "cell_role": job.get("cell_role") or "measured",
            "device_requested": job.get("device"),
            "threads": job.get("threads"),
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=20),
            },
        }

    record["environment"] = envinfo.collect()
    with open(args.out, "w") as handle:
        json.dump(record, handle, indent=2, sort_keys=False, default=str)
        handle.write("\n")
    return 0 if record["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
