"""One measured run, in its own process.

The runner never trains anything itself. It spawns this module once per
(scenario, engine, device, repeat), and this module builds the data, runs
one engine, scores the predictions, and writes one JSON record plus the
predictions to disk.

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

    python bench/real_data/worker.py --job job.json --out record.json
"""

import argparse
import json
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# The mojoboost package this harness measures is the checkout's own
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

import engines  # noqa: E402
import envinfo  # noqa: E402
import generators  # noqa: E402
import loaders  # noqa: E402
import measure  # noqa: E402
import quality  # noqa: E402
import scenarios  # noqa: E402


def data_digest(part):
    """A digest of exactly the bytes an engine will be given."""
    x = part["X"]
    if hasattr(x, "tocsc"):
        pieces = [x.data, x.indices, x.indptr]
    else:
        pieces = [np.ascontiguousarray(x)]
    pieces.append(np.asarray(part["y"], dtype=np.float64))
    if part.get("group") is not None:
        pieces.append(np.asarray(part["group"], dtype=np.int64))
    import hashlib

    h = hashlib.sha256()
    for piece in pieces:
        h.update(np.ascontiguousarray(piece).tobytes())
    return h.hexdigest()


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
        fallback = None if variant == "synthetic" else "no real dataset for this scenario"

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
        "fallback_reason": fallback,
    }
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


def _record_extra(task, train):
    """The per-dataset parameters the record's translated dicts need.

    The engines derive `num_class` from the loaded data before training;
    the record has to derive it the same way or a multiclass record cannot
    even be written (`lightgbm_params` and `mojoboost_params` both require
    it for a multiclass spec).
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

    engine = engines.build(job["engine"], job["threads"], job["device"])
    engine.load()
    warmup = engine.warmup(spec)
    result = engine.run(spec, train, test, repeats=job.get("predict_repeats", 3))
    predictions = result.pop("predictions")

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
        "scenario": spec["id"],
        "scenario_title": spec["title"],
        "tier": spec["tier"],
        "task": task,
        "primary_metric": spec.get("primary_metric", quality.TASK_METRICS[task][0]),
        "threads": job["threads"],
        "data": {**data_meta, "train": train_desc, "test": test_desc},
        "params": {
            "shared": scenarios.shared_params(spec, _record_extra(task, train)),
            "engine": (
                scenarios.mojoboost_params(
                    spec, job["device"], _record_extra(task, train)
                )
                if job["engine"] == "mojoboost"
                else scenarios.lightgbm_params(
                    spec, job["threads"], _record_extra(task, train)
                )
            ),
        },
        "caveats": list(spec.get("caveats", [])) + list(result.get("notes", [])),
        "quality": scores,
        "baseline_quality": baseline,
        "predictions_sha256": measure.digest(predictions),
        "predictions_shape": list(np.asarray(predictions).shape),
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
