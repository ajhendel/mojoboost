"""The orchestrator. Builds the job matrix and runs it, one process at a
time.

    python bench/real_data/run.py --dry-run
    python bench/real_data/run.py --tier smoke
    python bench/real_data/run.py --scenario dense_regression --device cpu gpu
    python bench/real_data/run.py --tier standard --threads 1 --threads 8

Runs are sequential and that is not negotiable. Two training runs sharing a
machine share its memory bandwidth, its cache, and on a laptop its thermal
budget, and every number either of them produces is then a number about the
other one too. The harness is slower for it and the timings mean something.

What this writes, under `results/<run_id>/`:

    manifest.json   the matrix, the environment, and what was skipped
    jobs/*.json     one job spec per run, so any single run can be repeated
    records/*.json  one record per run
    predictions/*.npy
    records.json    every record, concatenated
    records.csv     the flat view, for a spreadsheet

Nothing in this directory is committed. `results/README.md` says why.

The exit code reports whether the matrix ran, not whether the results were
good. Quality is verify.py's decision and speed is nobody's.
"""

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import backend_proof  # noqa: E402
import envinfo  # noqa: E402
import scenarios  # noqa: E402

DEFAULT_RESULTS = os.path.join(HERE, "results")

#: Thread-count environment for a run. Every library that might spin up its
#: own pool is pinned to the same number, so a background BLAS pool cannot
#: quietly help one engine.
THREAD_ENV = (
    "MOJOTREES_NUM_WORKERS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def default_threads():
    """Physical cores where that is knowable, logical cores otherwise. On a
    machine with performance and efficiency cores this is the whole chip,
    which is what a user gets by default and therefore what to measure."""
    cpu = envinfo._cpu()
    return cpu.get("physical_cores") or cpu.get("logical_cores") or 1


def build_matrix(args):
    jobs = []
    for scenario_id in args.scenario:
        spec = scenarios.resolve(scenario_id, args.tier, args.variant)
        for device in args.device:
            for engine in args.engine:
                if engine not in spec["engines"]:
                    continue
                if device != "cpu":
                    if device not in spec["devices"]:
                        jobs.append(
                            _skip(
                                scenario_id, engine, device, args,
                                f"{scenario_id} declares no {device} support",
                            )
                        )
                        continue
                    if engine == "lightgbm":
                        jobs.append(
                            _skip(
                                scenario_id, engine, device, args,
                                "LightGBM runs on the CPU in this harness; a "
                                "cpu-vs-gpu row is a mojotrees-internal "
                                "comparison and is labelled as one",
                            )
                        )
                        continue
                for threads in args.threads:
                    for repeat in range(args.repeats):
                        jobs.append(
                            {
                                "scenario": scenario_id,
                                "tier": args.tier,
                                "variant": args.variant,
                                "engine": engine,
                                "device": device,
                                "threads": threads,
                                "repeat": repeat,
                                "predict_repeats": args.predict_repeats,
                                "allow_unpinned": args.allow_unpinned,
                                "data_digest": not args.no_data_digest,
                                "backend_proof": not args.no_backend_proof,
                            }
                        )
    for index, job in enumerate(jobs):
        job["job_index"] = index
    return jobs


def _skip(scenario_id, engine, device, args, reason):
    return {
        "scenario": scenario_id,
        "tier": args.tier,
        "variant": args.variant,
        "engine": engine,
        "device": device,
        "threads": args.threads[0],
        "repeat": 0,
        "skip": reason,
    }


def label(job):
    return (
        f"{job['scenario']}.{job['engine']}.{job['device']}."
        f"t{job['threads']}.r{job['repeat']}"
    )


def run_job(job, run_dir, run_id, timeout):
    """Run one job in a fresh process and return its record."""
    name = label(job)
    job_path = os.path.join(run_dir, "jobs", f"{job['job_index']:03d}-{name}.json")
    record_path = os.path.join(run_dir, "records", f"{job['job_index']:03d}-{name}.json")
    pred_path = os.path.join(run_dir, "predictions", f"{job['job_index']:03d}-{name}.npy")

    payload = dict(job, run_id=run_id)
    with open(job_path, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    env = dict(os.environ)
    for name_ in THREAD_ENV:
        env[name_] = str(job["threads"])
    if job["device"] == "cpu":
        # A cpu row must be a cpu row even on a machine with an
        # accelerator, whatever the device parameter would otherwise
        # resolve to.
        env["MOJOTREES_DISABLE_GPU"] = "1"
    else:
        env.pop("MOJOTREES_DISABLE_GPU", None)

    started = time.time()
    proc = subprocess.run(
        [
            sys.executable, os.path.join(HERE, "worker.py"),
            "--job", job_path,
            "--out", record_path,
            "--predictions", pred_path,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    wall = time.time() - started

    if os.path.exists(record_path):
        with open(record_path) as handle:
            record = json.load(handle)
    else:
        record = {
            "schema_version": 1,
            "status": "error",
            "scenario": job["scenario"],
            "engine": job["engine"],
            "device_requested": job["device"],
            "threads": job["threads"],
            "repeat": job["repeat"],
            "error": {
                "type": "WorkerCrash",
                "message": f"worker exited {proc.returncode} with no record",
                "stderr": proc.stderr[-4000:],
            },
        }
        with open(record_path, "w") as handle:
            json.dump(record, handle, indent=2)
            handle.write("\n")
    record["worker"] = {
        "returncode": proc.returncode,
        "wall_s": wall,
        "stderr_tail": proc.stderr[-2000:] if proc.stderr else "",
    }
    # The trainer prints its profile through file descriptor 1 from
    # compiled Mojo, which the worker cannot read back through sys.stdout.
    # This process already holds the whole of that stream, so the evidence
    # is parsed here and written onto the record the worker produced. Only
    # a record that ran is given one; an error record has no fit to prove
    # anything about.
    if record.get("status") == "ok":
        record["backend_proof"] = backend_proof.parse(
            proc.stdout or "", job.get("backend_proof", True)
        )
    return record


CSV_COLUMNS = (
    "run_id", "scenario", "tier", "task", "data_kind", "dataset", "pinned",
    "engine", "engine_version", "device_requested", "device_used",
    "backend_proof", "threads",
    "histogram_builder", "repeat", "status", "primary_metric",
    "primary_value", "train_s", "train_cpu_s", "train_par_eff", "binning_s",
    "predict_batch_s", "predict_batch_par_eff", "predict_row_s",
    "warmup_s", "import_s", "peak_rss_bytes", "model_bytes", "num_trees",
    "num_bin", "bins_total", "bins_sha256",
    "train_rows", "train_features", "predictions_sha256", "data_sha256",
)


def _builder_column(record):
    """The histogram construction, as one cell.

    The resolved value when there is one, and the request marked as such
    when there is not, so a spreadsheet row never says "row-wise" about a
    run where nobody established that. See engines._histogram_builder for
    when the resolved value is knowable.
    """
    builder = record.get("histogram_builder")
    if not isinstance(builder, dict):
        return None
    resolved = builder.get("resolved")
    if isinstance(resolved, str):
        return resolved
    return f"{builder.get('requested')} (unresolved)"


def _flat(record):
    """One record flattened for the CSV.

    Repeated samples are reduced with the median, which is the same
    reduction report.py's `phase_value` uses, so the CSV column and the
    table cell are the same number under the same name. They were not:
    this took the minimum while the table took the median, and a reader
    who compared the two was comparing two different statistics without
    being told.

    The median rather than the minimum, because the minimum is the
    best-case sample and the machine's contention is exactly what these
    runs are trying to expose rather than to filter out. A benchmark that
    reports its luckiest sample reports the machine it wishes it had.
    """

    def phase(name, field="elapsed_s"):
        phases = record.get("phases") or {}
        block = phases.get(name)
        if isinstance(block, dict) and field in block:
            return block[field]
        if isinstance(block, dict) and "measured" in block:
            values = [s[field] for s in block["measured"] if s.get(field) is not None]
            return statistics.median(values) if values else None
        return None

    data = record.get("data") or {}
    train = data.get("train") or {}
    quality_block = record.get("quality") or {}
    primary = record.get("primary_metric")
    model = record.get("model") or {}
    bins = model.get("bins")
    bins = bins if isinstance(bins, dict) else {}
    return {
        "run_id": record.get("run_id"),
        "scenario": record.get("scenario"),
        "tier": record.get("tier"),
        "task": record.get("task"),
        "data_kind": data.get("data_kind"),
        "dataset": data.get("dataset"),
        "pinned": data.get("pinned"),
        "engine": record.get("engine"),
        "engine_version": record.get("engine_version"),
        "device_requested": record.get("device_requested"),
        "device_used": record.get("device_used"),
        # Beside device_used on purpose. The first is what the Python side
        # resolved and the second is what the trainer emitted, and a reader
        # scanning the sheet should be able to see them disagree.
        "backend_proof": backend_proof.csv_cell(record.get("backend_proof")),
        "threads": record.get("threads"),
        "histogram_builder": _builder_column(record),
        "repeat": record.get("repeat"),
        "status": record.get("status"),
        "primary_metric": primary,
        "primary_value": quality_block.get(primary),
        "train_s": phase("train"),
        "train_cpu_s": phase("train", "cpu_s"),
        "train_par_eff": phase("train", "parallel_efficiency"),
        "binning_s": phase("binning"),
        "predict_batch_s": phase("predict_batch"),
        "predict_batch_par_eff": phase("predict_batch", "parallel_efficiency"),
        "predict_row_s": phase("predict_row"),
        "warmup_s": (record.get("warmup") or {}).get("elapsed_s"),
        "import_s": phase("import"),
        "peak_rss_bytes": record.get("peak_rss_bytes"),
        "model_bytes": (model.get("size") or {}).get("string_bytes"),
        "num_trees": model.get("num_trees"),
        "num_bin": model.get("num_bin"),
        "bins_total": bins.get("total"),
        "bins_sha256": bins.get("sha256"),
        "train_rows": train.get("rows"),
        "train_features": train.get("features"),
        "predictions_sha256": record.get("predictions_sha256"),
        "data_sha256": train.get("digest"),
    }


def write_outputs(run_dir, run_id, records, manifest):
    with open(os.path.join(run_dir, "records.json"), "w") as handle:
        json.dump({"run_id": run_id, "records": records}, handle, indent=2, default=str)
        handle.write("\n")
    with open(os.path.join(run_dir, "records.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for record in records:
            writer.writerow(_flat(record))
    with open(os.path.join(run_dir, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2, default=str)
        handle.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scenario", action="append", choices=sorted(scenarios.SCENARIOS),
        help="repeatable; every scenario by default",
    )
    parser.add_argument("--tier", choices=scenarios.TIERS, default="standard")
    parser.add_argument("--variant", choices=("auto", "real", "synthetic"), default="auto")
    parser.add_argument(
        "--engine", action="append", choices=("mojotrees", "lightgbm"),
        help="repeatable; both by default",
    )
    parser.add_argument(
        "--device", action="append", choices=("cpu", "gpu"),
        help="repeatable; cpu by default",
    )
    parser.add_argument(
        "--threads", action="append", type=int,
        help="repeatable; the machine's physical core count by default",
    )
    parser.add_argument(
        "--repeats", type=int, default=3,
        help="whole-process repeats per cell; 3 is the minimum that shows a spread",
    )
    parser.add_argument("--predict-repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=7200, help="seconds per run")
    parser.add_argument("--out", default=DEFAULT_RESULTS)
    parser.add_argument("--tag", default="", help="appended to the run id")
    parser.add_argument("--allow-unpinned", action="store_true")
    parser.add_argument(
        "--no-data-digest", action="store_true",
        help="skip hashing the input matrices; faster, and gives up the "
             "guarantee that both engines saw identical data",
    )
    parser.add_argument(
        "--no-backend-proof", action="store_true",
        help="do not turn the trainer's phase profile on for the measured "
             "fit; the records then carry no evidence of which backend ran "
             "and verify.py refuses every accelerator row that also matches "
             "its CPU twin's prediction digest",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="write the matrix and the job files, run nothing",
    )
    args = parser.parse_args(argv)

    args.scenario = args.scenario or sorted(scenarios.SCENARIOS)
    args.engine = args.engine or ["mojotrees", "lightgbm"]
    args.device = args.device or ["cpu"]
    args.threads = args.threads or [default_threads()]

    jobs = build_matrix(args)
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + (
        f"-{args.tag}" if args.tag else ""
    )
    run_dir = os.path.join(args.out, run_id)
    for sub in ("jobs", "records", "predictions"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)

    manifest = {
        "run_id": run_id,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arguments": vars(args),
        "environment": envinfo.collect(),
        "jobs": jobs,
        "sequential": True,
        "note": (
            "Runs are sequential. Timings from a run whose manifest says "
            "otherwise are timings of a contended machine."
        ),
    }

    runnable = [job for job in jobs if "skip" not in job]
    skipped = [job for job in jobs if "skip" in job]
    print(f"run {run_id}: {len(runnable)} runs, {len(skipped)} skipped")
    for job in skipped:
        print(f"  skip {label(job)}: {job['skip']}")

    if args.dry_run:
        for job in runnable:
            print(f"  would run {label(job)}")
        write_outputs(run_dir, run_id, [], manifest)
        print(f"\nmatrix written to {run_dir}. Nothing was run.")
        return 0

    records = []
    for job in runnable:
        print(f"  {label(job)} ... ", end="", flush=True)
        try:
            record = run_job(job, run_dir, run_id, args.timeout)
        except subprocess.TimeoutExpired:
            record = {
                "schema_version": 1,
                "status": "timeout",
                "scenario": job["scenario"],
                "engine": job["engine"],
                "device_requested": job["device"],
                "threads": job["threads"],
                "repeat": job["repeat"],
                "error": {"type": "Timeout", "message": f"exceeded {args.timeout}s"},
            }
        records.append(record)
        state = record.get("status")
        detail = ""
        if state == "ok":
            metric = record.get("primary_metric")
            detail = f"{metric}={record['quality'].get(metric):.6g}"
        else:
            detail = (record.get("error") or {}).get("message", "")[:120]
        print(f"{state} {detail}")

    for job in skipped:
        records.append(
            {
                "schema_version": 1, "status": "skipped",
                "scenario": job["scenario"], "engine": job["engine"],
                "device_requested": job["device"], "threads": job["threads"],
                "repeat": 0, "skip_reason": job["skip"],
            }
        )

    write_outputs(run_dir, run_id, records, manifest)
    failed = [r for r in records if r.get("status") in ("error", "timeout")]
    print(f"\nwrote {run_dir}/records.json")
    print(
        "next: `python bench/real_data/verify.py "
        f"{os.path.join(run_dir, 'records.json')}` for the correctness "
        "verdict, then report.py for the timings."
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
