"""The orchestrator. Builds the job matrix and runs it, one process at a
time.

    python bench/real_data/run.py --dry-run
    python bench/real_data/run.py --tier smoke
    python bench/real_data/run.py --scenario dense_regression --device cpu gpu
    python bench/real_data/run.py --tier standard --threads 1 --threads 8
    python bench/real_data/run.py --arms frontier

A cell's identity is (scenario, tier, variant, ARM, device, threads, repeat).
`arm` defaults to the engine name, so the cross-product matrix above is one
arm per engine and renders exactly the labels it always did; `--arms` names a
module whose `arms()` returns cells that vary the parameters WITHIN an engine,
which is what a frontier sweep is and what the identity had no room for. See
"The arm dimension" below.

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
good. Quality is verify.py's decision and speed is nobody's. That
separation is deliberate and is not what the exit codes below changed.

    0   every cell that was meant to run produced a result
    2   at least one cell produced NO RESULT AT ALL

Two is an infrastructure failure and it is a different thing from a red
verdict. It comes from an incident: `bench/real_data` had never been run,
the first attempt produced 44 cells of which 27 failed, every mojotrees row
died on `cannot import name '_mojotrees'` because nothing in the run path
builds the extension, and the run was read as having happened. A cell that
produced no result is now counted by a positive test rather than by a list
of known-bad statuses, printed in a block that names every failed cell, and
reported through an exit code that is not the one verify.py uses for a
quality failure.

Exit code 1 is deliberately unused here, so that a caller which only
distinguishes zero from non-zero still fails, and a caller which reads the
number can tell "the matrix did not run" from "the results were bad".
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
# Imported for `ENGINES` alone, so that `--engine` offers exactly the names
# worker.py can build and the two lists cannot drift into a flag that parses
# and then fails in the child. It costs 0.077s and pulls in no engine
# library: every one of those imports sits inside a method.
import engines  # noqa: E402
import envinfo  # noqa: E402
import scenarios  # noqa: E402

DEFAULT_RESULTS = os.path.join(HERE, "results")

#: Exit code for "at least one cell produced no result". Not 1, because
#: verify.py returns 1 for a quality failure and the two are different
#: findings: this one says the matrix did not run, and says nothing at all
#: about whether the cells that did run were any good.
EXIT_INFRASTRUCTURE = 2

#: Substrings that mean the extension was never built. The first is what 27
#: cells printed in the incident that produced these exit codes.
MISSING_EXTENSION_MARKERS = (
    "cannot import name '_mojotrees'",
    "No module named 'mojotrees'",
    "No module named '_mojotrees'",
)

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


#: Arm rank INSIDE a round, for arms one of which must run before another.
#:
#: One dependency today. `mojotrees_catboost_mode` cannot be built until the
#: `catboost` cell for the same scenario, tier and variant has written its
#: resolved parameters into the run's sidecar, because CatBoost derives its own
#: learning rate and this arm takes that value rather than a constant. Every
#: other arm is rank 0 and keeps build order.
#:
#: This is composed UNDER the repeat sort, never in place of it. See the
#: comment at the sort itself.
CELL_ORDER = {
    "catboost": 0,
    "catboost_lossguide": 0,
    "mojotrees_catboost_mode": 1,
}


# ---------------------------------------------------------------------------
# The arm dimension.
# ---------------------------------------------------------------------------
#
# A job's identity was (scenario, tier, variant, engine, device, threads,
# repeat), and an ARM is a variation on the parameters WITHIN one engine, so
# two arms of one engine collided: one identity, one cell key, one output
# filename, and the second arm's record overwrote the first's. Nothing said so.
#
# The dimension is `arm`, a name, and it DEFAULTS TO THE ENGINE NAME. That
# default is what makes this change invisible to the existing matrix: every
# job the old builder produced now carries `arm == engine`, `label()` renders
# the same string it always did, and every file under `jobs/`, `records/` and
# `predictions/` keeps the name it had. A frontier arm carries its own id and
# is addressable end to end.
#
# Beside it a job carries what makes the arm an arm:
#
#   arm_params          overrides folded into the engine's training params
#   arm_dataset_params  overrides folded into the binning params (max_bin)
#   arm_env             per-CELL environment, applied by `run_job`
#   axis                which axis of a sweep this arm moves, or None
#
# `engines.build` takes the first two; `run_job` applies the third; `worker.py`
# writes `arm` and `axis` onto the record. `report.py` and `verify.py` should
# read `record.get("arm")` with a fallback to `record["engine"]`, which is what
# a record written before this change carries.
#
# NOTE the word `arm` is already used by `engines.ENGINE_ARM` for a different
# thing -- the ROLE of an engine in the comparison, subject / comparator /
# peer. That mapping is untouched and is still keyed by engine name. A record
# carries both: `arm` is which cell this is, `ENGINE_ARM[engine]` is what the
# cell is for.

#: An arm's identity includes its RESOLVED parameters, per backend.
#:
#: **The rule (Andrew, 2026-08-16): if a backend resolves a parameter
#: differently from another, that backend's cell is a DECLARED SKIP with the
#: reason, never the same arm name carrying a different model.** It is the
#: arm dimension above, one level deeper. The dimension stops two arms of one
#: engine from colliding on a name; this stops one arm from wearing the same
#: name across two backends that did not train the same thing.
#:
#: The case that produced it: `random_strength` is refused on the GPU when it
#: is NAMED and declines to 0.0 when it arrives as a CatBoost-mode default,
#: because no device round loop computes the per-tree score scale. So a GPU
#: cell of a symmetric-tree arm is either an infrastructure failure or a
#: different regularizer under the CPU cell's name, and a reader comparing the
#: two would see a backend difference and be looking at a model difference.
#:
#: Each entry names the parameter and BOTH values, the way the scenario
#: support tables do, because "GPU skipped" does not tell the next person
#: whether this is a gap with a scheduled exit or a permanent difference.
#:
#: `applies` reads the arm's resolved training parameters. It is deliberately
#: a predicate over what the harness ASKED FOR rather than a copy of
#: `device_policy`: this table's job is to keep a cell off the schedule, and
#: the native policy remains the authority on what a fit may do.
DEVICE_PARAMETER_DIVERGENCE = (
    {
        "parameter": "random_strength",
        "device": "gpu",
        "applies": lambda params: float(params.get("random_strength", 0.0)) > 0.0,
        # A callable, because the value the arm asked for is the whole point
        # of the sentence: "GPU resolves it to 0.0 where this arm asks for
        # 1.0" is the skip a reader can act on, and "where this arm asks for
        # the value it asked for" is not.
        "cpu_value": lambda params: params.get("random_strength"),
        "device_value": "0.0",
        "why": (
            "the per-split draw is staged on the device but its per-tree "
            "SCALE is computed only by the dense CPU round loops "
            "(boosting._round_random_score_scale), so `_parse_params` "
            "declares random_strength_ok as `device == cpu and not sparse`. "
            "A named value is refused there by name; a value inherited from "
            "the symmetrictree mode default declines to 0.0 in silence. "
            "Either way the GPU cell is not this arm"
        ),
        "exit": (
            "a gap with an exit: a device round loop that computes the scale "
            "would close it. Until one does, the CPU cell is the arm"
        ),
    },
    {
        "parameter": "derivative_precision",
        "device": "gpu",
        "applies": lambda params: (
            str(params.get("derivative_precision", "float32")).lower()
            == "float64"
        ),
        "cpu_value": "float64",
        "device_value": "float32",
        "why": (
            "gradients and hessians are carried as Float32 on the device and "
            "there is no Float64 there, so "
            "histogram.check_device_derivative_precision refuses the request "
            "by name at every shape rather than computing the narrow answer "
            "under the wide label"
        ),
        "exit": (
            "PERMANENT on this hardware. The missing thing is a datatype the "
            "device does not have, and it is not fixable by threading"
        ),
    },
)


def backend_divergence(arm_params, device, resolved=None):
    """The reason this arm is not the same arm on `device`, or None.

    Two modes, and the returned sentence says which one it is in rather than
    letting a reader assume the stronger one:

    - `resolved` given: a per-fit RESOLVED-CONFIGURATION record, with
      provenance per value on the same axis as `catboost_value_source`. That
      is the real input, because it answers "what did this backend actually
      resolve" instead of "did something fall back". The other campaign is
      building it; this function takes it when it is there.
    - `resolved` absent: the coarse mode. The declared table above, read
      against what the arm asked for. It cannot see a parameter nobody
      thought to declare, and the sentence says so.

    Nothing here is a quality judgement and nothing here is a fallback. A
    divergence is a skip with a reason, which is a result; running the cell
    anyway would be a row.
    """
    mode = "resolved" if resolved else "declared"
    for rule in DEVICE_PARAMETER_DIVERGENCE:
        if rule["device"] != device:
            continue
        if resolved:
            # The resolved record answers directly: what did this backend
            # resolve this parameter to, and was that the request.
            entry = (resolved.get(rule["parameter"]) or {})
            if not entry or entry.get("agrees_with_request", True):
                continue
            cpu_value = entry.get("requested")
            device_value = entry.get("resolved")
        else:
            if not rule["applies"](arm_params):
                continue
            cpu_value = rule["cpu_value"]
            if callable(cpu_value):
                cpu_value = cpu_value(arm_params)
            device_value = rule["device_value"]
            if callable(device_value):
                device_value = device_value(arm_params)
        return (
            f"{device} resolves {rule['parameter']} to {device_value} where "
            f"this arm asks for {cpu_value}, so the two cells are not the "
            f"same arm and must not share its name. {rule['why']}. "
            f"{rule['exit']}. "
            f"[{mode} mode: "
            + (
                "read off the run's per-fit resolved-configuration record"
                if resolved
                else "read off run.DEVICE_PARAMETER_DIVERGENCE against what "
                "the arm asked for, because no per-fit "
                "resolved-configuration record was available; a parameter "
                "not in that table is not checked"
            )
            + "]"
        )
    return None


def _engine_skip_reason(spec, scenario_id, engine, device, engines_in_run, tier):
    """Why this (engine, device) cell must not be scheduled, or None.

    Lifted out of `build_matrix` unchanged so that the arm path and the
    engine path answer it with one function rather than two copies. **The
    ORDER of these tests is load-bearing and is not an implementation
    detail**; see the comment on the `mojotrees_catboost_mode` case.
    """
    # The CatBoost-mode arm has a DEPENDENCY on the CatBoost arm that no
    # other pair in this matrix has: it takes CatBoost's resolved learning
    # rate for the same cell, so without a CatBoost cell in the same run it
    # raises rather than trains. Declared as a skip here because a raising
    # cell is an infrastructure failure and this harness answers one by
    # withholding the quality verdict for the whole matrix -- so
    # `--engine mojotrees_catboost_mode` on its own would suppress the
    # verdict rather than report a scheduling mistake.
    if engine == "mojotrees_catboost_mode" and "catboost" not in engines_in_run:
        return (
            "the CatBoost-mode arm takes CatBoost's RESOLVED learning rate "
            "for this same cell, which cb-shipped no longer pins, so the "
            "catboost arm has to run in the same run and write it. Add "
            "catboost to --engine, or read "
            "scenarios.CATBOOST_LEARNING_RATE_TRANSITION for why there is no "
            "fallback"
        )
    if engine in scenarios.CATBOOST_ENGINES:
        ok, reason = scenarios.catboost_tier_ok(spec, tier)
        if not ok:
            return reason
    if device != "cpu":
        if device not in spec["devices"]:
            return f"{scenario_id} declares no {device} support"
        if engine == "lightgbm":
            return (
                "LightGBM runs on the CPU in this harness; a cpu-vs-gpu row "
                "is a mojotrees-internal comparison and is labelled as one"
            )
        # The same treatment for the two arms below, and it was missing
        # until 2026-08-16: both refuse a non-CPU device somewhere further
        # in, so without these they were SCHEDULED and then failed at load
        # or fit time. A failing peer cell is an infrastructure failure, and
        # this harness answers an infrastructure failure by withholding the
        # quality verdict for the whole matrix -- so twenty-four cells
        # nobody wanted could have taken the exit code of a run whose
        # comparator rows all succeeded. Found by `--dry-run` before the
        # window rather than inside it.
        if engine in scenarios.CATBOOST_ENGINES:
            return (
                "CatBoost runs on the CPU in this harness: its GPU training "
                "is a different quantization (border_count capped at 255 "
                "against 65535) and so is not the same measurement. "
                "CatBoostEngine.load refuses it by name"
            )
        # The pairing ground is stated first and the device block second, and
        # the order is the point. This arm exists to be read against
        # CatBoost, CatBoost is CPU-only above, so a GPU row here has nothing
        # to pair against no matter what the device can compute.
        # BLOCK_SCORE_FUNCTION is why it cannot run there today; the pairing
        # is why it should not be scheduled there even after a lane teaches
        # the device split search the Cosine ratio and the block goes away.
        # Written this way so that landing Cosine on the device does not
        # silently add a column to a comparison matrix.
        #
        # DEVICE_PARAMETER_DIVERGENCE is checked AFTER this function returns
        # None, never before it, so a resolved-parameter divergence can never
        # absorb this reason or reorder its two halves.
        if engine == "mojotrees_catboost_mode":
            return (
                "the CatBoost-mode arm is read against CatBoost, and "
                "CatBoost is CPU-only in this harness, so a GPU row here has "
                "no counterpart to be read against; separately, the arm sets "
                "score_function=Cosine and the device split search computes "
                "G^2/(H+lambda) only, so device_policy blocks it today "
                "(BLOCK_SCORE_FUNCTION). The first reason outlives the second"
            )
    return None


def _job(scenario_id, engine, device, threads, repeat, args, arm):
    """One runnable job. `arm` is the normalized arm dict."""
    return {
        "scenario": scenario_id,
        "tier": arm["tier"],
        "variant": arm["variant"],
        "engine": engine,
        # The new dimension. Defaults to the engine name, which is what
        # keeps every existing filename and every existing label identical.
        "arm": arm["id"],
        "axis": arm["axis"],
        "axis_value": arm["axis_value"],
        "arm_block": arm["block"],
        "arm_params": dict(arm["params"]),
        "arm_dataset_params": dict(arm["dataset_params"]),
        "arm_env": dict(arm["env"]),
        "device": device,
        "threads": threads,
        "repeat": repeat,
        "predict_repeats": args.predict_repeats,
        "allow_unpinned": args.allow_unpinned,
        "data_digest": not args.no_data_digest,
        "backend_proof": not args.no_backend_proof,
        # Filled in by `main`, which knows the run directory and this
        # function does not. The run's collected CatBoost.get_all_params(),
        # which the CatBoost-mode arm cannot build without.
        "catboost_readback_path": None,
    }


def _normalize_arm(arm, engine, scenario_id, args):
    """An arm dict with every field this module reads, defaults filled in.

    An engine with no arm is an arm: its id is the engine name and its
    override dicts are empty. That is what makes the whole engine path below
    a special case of the arm path rather than a second implementation.
    """
    arm = dict(arm or {})
    arm.setdefault("id", engine)
    arm.setdefault("scenario", scenario_id)
    arm.setdefault("tier", args.tier)
    arm.setdefault("variant", args.variant)
    arm.setdefault("engine", engine)
    arm.setdefault("axis", None)
    arm.setdefault("axis_value", None)
    arm.setdefault("block", None)
    arm.setdefault("params", {})
    arm.setdefault("dataset_params", {})
    arm.setdefault("env", {})
    arm.setdefault("skip", None)
    return arm


def build_matrix(args, arms=None):
    """The job matrix.

    Without `arms` this is the cross product it has always been, one arm per
    engine, and every job it produces is byte-for-byte the job the previous
    version produced plus the arm fields (`arm == engine`, empty overrides).

    With `arms` -- a list of arm dicts, as `frontier.arms()` returns -- each
    arm is one cell of the matrix and carries its own scenario, tier,
    variant, engine, device and parameter overrides. The skip rules are the
    same function in both paths.
    """
    jobs = []
    # Which engines are in THIS run, for the one skip that is a property of
    # the run rather than of the cell: the CatBoost-mode arm cannot be built
    # unless a CatBoost cell runs beside it. Read off the arms when there are
    # arms, because `--engine` does not describe an arm list.
    engines_in_run = (
        set(args.engine)
        if arms is None
        else {arm["engine"] for arm in arms}
    )
    if arms is None:
        for scenario_id in args.scenario:
            spec = scenarios.resolve(scenario_id, args.tier, args.variant)
            for device in args.device:
                for engine in args.engine:
                    if engine not in spec["engines"]:
                        continue
                    arm = _normalize_arm(None, engine, scenario_id, args)
                    jobs.extend(
                        _cell(
                            spec, scenario_id, engine, device, args, arm,
                            engines_in_run,
                        )
                    )
    else:
        for declared in arms:
            engine = declared["engine"]
            scenario_id = declared["scenario"]
            arm = _normalize_arm(declared, engine, scenario_id, args)
            device = declared.get("device", "cpu")
            if arm["skip"]:
                # A skip the arm itself declared. Kept as the arm's own
                # sentence rather than restated here, which is the rule
                # `frontier.check()` enforces from the other side.
                jobs.append(_skip(scenario_id, engine, device, args, arm["skip"], arm))
                continue
            spec = scenarios.resolve(scenario_id, arm["tier"], arm["variant"])
            if engine not in spec["engines"]:
                jobs.append(
                    _skip(
                        scenario_id, engine, device, args,
                        f"{scenario_id} declares no {engine} arm", arm,
                    )
                )
                continue
            jobs.extend(
                _cell(
                    spec, scenario_id, engine, device, args, arm,
                    engines_in_run,
                )
            )
    for index, job in enumerate(jobs):
        job["job_index"] = index
    return jobs


def _cell(spec, scenario_id, engine, device, args, arm, engines_in_run):
    """The jobs for one (arm, device) cell: one skip, or one per repeat."""
    reason = _engine_skip_reason(
        spec, scenario_id, engine, device, engines_in_run, arm["tier"]
    )
    if reason is not None:
        return [_skip(scenario_id, engine, device, args, reason, arm)]
    # Second, and never first. See _engine_skip_reason's closing comment.
    reason = backend_divergence(arm["params"], device)
    if reason is not None:
        return [_skip(scenario_id, engine, device, args, reason, arm)]
    repeats = int(arm.get("repeats") or args.repeats)
    return [
        _job(scenario_id, engine, device, threads, repeat, args, arm)
        for threads in args.threads
        for repeat in range(repeats)
    ]


def _skip(scenario_id, engine, device, args, reason, arm=None):
    arm = arm or _normalize_arm(None, engine, scenario_id, args)
    return {
        "scenario": scenario_id,
        "tier": arm["tier"],
        "variant": arm["variant"],
        "engine": engine,
        "arm": arm["id"],
        "axis": arm["axis"],
        "device": device,
        "threads": args.threads[0],
        "repeat": 0,
        "skip": reason,
    }


def label(job):
    """The cell's name, and the stem of its three filenames.

    `arm` sits where `engine` used to and defaults to the engine name, so a
    matrix with no arms renders exactly the strings it rendered before. A
    frontier arm renders its own id, which is what stops two arms of one
    engine writing one file.
    """
    return (
        f"{job['scenario']}.{_safe(job.get('arm') or job['engine'])}."
        f"{job['device']}.t{job['threads']}.r{job['repeat']}"
    )


def _safe(name):
    """An arm id as one path segment.

    Arm ids are written by hand in a plan file, so this refuses a separator
    rather than rewriting one: a silently mangled id is an id that no longer
    matches the plan it came from.
    """
    text = str(name)
    for bad in ("/", os.sep, "\\", "\n", "\t", " "):
        if bad in text:
            raise ValueError(
                f"arm id {text!r} contains {bad!r}, and an arm id is a path "
                "segment as well as a name. Rename the arm"
            )
    return text


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
        # A GPU cell must not inherit a float64 derivative setting EITHER
        # WAY, and this is where that is enforced.
        #
        # `histogram.check_device_derivative_precision` refuses the request
        # from both entries, the parameter and a live environment read, and it
        # says which of the two is the likelier mistake: a variable exported
        # for a CPU A/B and then left set for a GPU run. This process holds
        # whatever the operator's shell held, and `subprocess` copies it into
        # every child, so a variable nobody typed for this run would decide a
        # cell of it. The parameter is the door now
        # (`derivative_precision` on the estimator and on the parameter
        # string); the variable is unset here so that a GPU cell cannot fail
        # on something the job file does not mention.
        env.pop("MOJOTREES_DERIVATIVE_PRECISION", None)
    # The arm's own environment, applied per CELL. Last, so that a
    # per-cell setting is visible in the job file that produced it; and
    # REFUSED where it would overwrite something this runner owns, because a
    # thread count or a GPU switch quietly replaced by an arm is a cell whose
    # label no longer describes it.
    for key, value in (job.get("arm_env") or {}).items():
        if key in THREAD_ENV or key == "MOJOTREES_DISABLE_GPU":
            raise ValueError(
                f"arm {job.get('arm')!r} sets {key}, which run.py owns: the "
                "thread count comes from --threads and the GPU switch from "
                "--device, and an arm that moved either would be measuring "
                "something its label does not say"
            )
        if key == "MOJOTREES_DERIVATIVE_PRECISION":
            raise ValueError(
                f"arm {job.get('arm')!r} sets MOJOTREES_DERIVATIVE_PRECISION. "
                "That is a parameter now, not a variable: pass "
                "derivative_precision in the arm's params, where it travels "
                "in the job file and in the record instead of being inherited "
                "by whatever else this process starts. See "
                "docs/COMPATIBILITY_POLICY.md section 9.5.1"
            )
        env[key] = str(value)

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
            "arm": job.get("arm") or job["engine"],
            "axis": job.get("axis"),
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
    "run_id", "comparator", "scenario", "tier", "task", "data_kind",
    "dataset", "pinned",
    # `arm` before `engine` because it is the finer of the two: a spreadsheet
    # sorted on `engine` alone puts two arms of one engine in one block, which
    # is the collision the arm dimension exists to remove.
    "arm", "axis", "axis_value",
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
        # On every row rather than once per file. A spreadsheet gets
        # filtered, sorted, and pasted into somewhere else one row at a
        # time, and a row that has left its file still has to say what it
        # was measured against.
        "comparator": scenarios.comparator_id(),
        "scenario": record.get("scenario"),
        "tier": record.get("tier"),
        "task": record.get("task"),
        "data_kind": data.get("data_kind"),
        "dataset": data.get("dataset"),
        "pinned": data.get("pinned"),
        # The fallback is what makes a record written before the arm
        # dimension readable in the same sheet as one written after: it
        # carried no `arm`, and its arm WAS its engine.
        "arm": record.get("arm") or record.get("engine"),
        "axis": record.get("axis"),
        "axis_value": record.get("axis_value"),
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


def comparator_banner():
    """The comparator on the console, before the first cell runs.

    Short enough to read and specific enough to check: the id, the label,
    every parameter the comparator passes, and the one deviation from stock
    that is not a feature-space pin, with the reason it is there. The full
    block goes into the manifest and into records.json.
    """
    block = scenarios.comparator_block()
    passed = " ".join(
        f"{key}={value}" for key, value in sorted(block["lightgbm_passed"].items())
    )
    return (
        f"comparator {block['one_line']}: {block['label']}\n"
        f"  registered: {block['registered']}\n"
        f"  lightgbm gets: {passed}\n"
        f"  everything else is LightGBM's own default "
        f"({block['lightgbm_defaults_source']})\n"
        f"  deterministic=true: {block['reproducibility']['why_deterministic']}\n"
        f"  and: {block['reproducibility']['known_limit']}"
    )


def _cell_error(record):
    """One failed cell as (label, type, message)."""
    error = record.get("error") or {}
    name = record.get("label") or ".".join(
        str(record.get(key))
        for key in ("scenario", "arm", "device_requested", "threads")
    )
    return (
        name,
        error.get("type") or f"status={record.get('status')!r}",
        (error.get("message") or "").strip().splitlines()[0][:200]
        if error.get("message")
        else "no error block; the record simply is not an ok result",
    )


def report_infrastructure_failure(failed, attempted, run_dir):
    """Print the block that the incident this exists for did not have.

    Every failed cell by name, the count against what was attempted, the
    distinct causes once each, and the build hint when the cause is the one
    that has actually happened. Written to stdout and to stderr, because a
    long matrix scrolls and the last thing a reader sees should not be
    "wrote records.json".
    """
    lines = [
        "",
        "=" * 72,
        f"INFRASTRUCTURE FAILURE: {len(failed)} of {attempted} cells produced "
        "no result.",
        "=" * 72,
        "",
        "This is not a quality verdict. These cells did not run, so there is "
        "nothing",
        "for verify.py to judge and nothing in this run to quote. Anything "
        "written",
        f"under {run_dir} is a partial matrix.",
        "",
    ]
    for name, kind, message in (_cell_error(r) for r in failed):
        lines.append(f"  {name}: {kind}: {message}")
    causes = sorted({kind for _, kind, _ in (_cell_error(r) for r in failed)})
    lines.append("")
    lines.append("distinct causes: " + ", ".join(causes))
    blob = " ".join(str(_cell_error(r)) for r in failed)
    if any(marker in blob for marker in MISSING_EXTENSION_MARKERS):
        lines += [
            "",
            "The mojotrees extension is not importable, which is the failure "
            "that produced",
            "27 of 44 dead cells the first time this harness ran. Nothing in "
            "this run path",
            "builds it. Build it and run the matrix again:",
            "",
            "    pixi run build-python",
            "",
        ]
    lines.append("=" * 72)
    text = "\n".join(lines)
    print(text)
    print(text, file=sys.stderr)


def write_outputs(run_dir, run_id, records, manifest):
    with open(os.path.join(run_dir, "records.json"), "w") as handle:
        json.dump(
            {
                "run_id": run_id,
                # Beside the records, not only in the manifest. A records
                # file that travels on its own still says which comparator
                # produced it, which is the whole of the lesson from four
                # comparator-configuration incidents in three days.
                "comparator": scenarios.comparator_block(),
                "records": records,
            },
            handle,
            indent=2,
            default=str,
        )
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
        "--engine", action="append", choices=tuple(engines.ENGINES),
        help="repeatable; mojotrees and lightgbm by default. The peer arms ("
             + ", ".join(scenarios.PEER_ENGINES)
             + ") are selectable but never default: the headline is against "
               "the comparator and a peer arm must not be able to join it by "
               "accident",
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
        "--arms", default=None, metavar="MODULE",
        help="a module in bench/real_data/ exposing arms(), e.g. `frontier`. "
             "Each arm is one cell and carries its own scenario, tier, "
             "variant, engine, device and parameter overrides, so --scenario, "
             "--engine and --device do not apply. Without it the matrix is "
             "the cross product it has always been, one arm per engine",
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

    arms = None
    arms_source = None
    if args.arms:
        # Imported by name from this directory rather than taken as a path,
        # so an arm list is a module in the repository with a `check()` beside
        # it and not a file somebody points at.
        import importlib

        module = importlib.import_module(args.arms)
        if not hasattr(module, "arms"):
            raise SystemExit(
                f"--arms {args.arms}: the module has no arms(). An arm list "
                "is a function returning dicts with at least id, scenario, "
                "engine and device"
            )
        arms = list(module.arms())
        arms_source = {
            "module": args.arms,
            "file": getattr(module, "__file__", None),
            "count": len(arms),
            "runnable": sum(1 for a in arms if not a.get("skip")),
        }
        # The plan checks itself before any of it is scheduled, where it has
        # one, for the reason `comparator_banner` prints before the first
        # cell: a plan that cannot be read correctly must not be run at all.
        if hasattr(module, "check"):
            module.check(arms)

    jobs = build_matrix(args, arms)
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + (
        f"-{args.tag}" if args.tag else ""
    )
    run_dir = os.path.join(args.out, run_id)
    for sub in ("jobs", "records", "predictions"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)

    # Where CatBoost cells leave their resolved parameters for the
    # CatBoost-mode cells to read. One file per run and not per matrix: the
    # rate CatBoost derives depends on the shape, so a read-back from another
    # run's tier is a wrong number rather than a stale one, and a per-run file
    # cannot be mistaken for one. Set here rather than in `build_matrix`,
    # which does not know the run directory.
    readback_path = os.path.join(run_dir, scenarios.CATBOOST_READBACK_FILE)
    for job in jobs:
        if "skip" not in job:
            job["catboost_readback_path"] = readback_path

    manifest = {
        "run_id": run_id,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arguments": vars(args),
        # Not optional and not a convention. Every published table has to
        # say which comparator produced it, so the runner writes the whole
        # configuration into the manifest, into records.json, into a column
        # of every CSV row, and onto the console before the first cell.
        "comparator": scenarios.comparator_block(),
        "environment": envinfo.collect(),
        "jobs": jobs,
        # What produced the arm list, or None for the engine cross product.
        # A records file that says "97 arms" and cannot say which plan they
        # came from is a table nobody can reproduce.
        "arms_source": arms_source,
        "sequential": True,
        "arm_order": "round-interleaved",
        "note": (
            "Runs are sequential. Timings from a run whose manifest says "
            "otherwise are timings of a contended machine. `jobs` is the "
            "matrix in build order; execution is round-interleaved (all arms "
            "at repeat 0, then all arms at repeat 1, ...), so read the "
            "executed order off the records' repeat field, not off this list. "
            "Inside a round, arms keep build order except that catboost runs "
            "before mojotrees_catboost_mode, which cannot be built until "
            "catboost has written its resolved learning rate into "
            "catboost_readback.json: see run.CELL_ORDER. A job's `arm` is "
            "its identity within an engine and defaults to the engine name, "
            "so a matrix with no --arms carries arm == engine on every row "
            "and renders exactly the labels it always did."
        ),
    }

    runnable = [job for job in jobs if "skip" not in job]
    # Interleave the arms. `build_matrix` nests repeat inside engine, so the
    # natural order runs every repeat of one arm and then every repeat of the
    # next, which puts the arms in different thermal windows: at this tier the
    # last arm of a five-arm block starts an hour after the first, and this
    # machine has been measured drifting two to three times across windows that
    # size. The arm-blocked order therefore reports drift as if it were a
    # difference between engines, and it does so silently, because each arm's
    # own spread stays tight inside its own window.
    #
    # A stable sort by repeat turns the matrix into rounds: every arm takes one
    # measurement, then every arm takes the next. Drift then lands on all arms
    # at once and shows up as spread across repeats, which is where it can be
    # read. Arm order inside a round is left fixed, matching the Mojo harness's
    # `for rep: for arm:` at bench/bench_train_gpu.mojo:1772.
    #
    # Ordering only; the set of jobs, their job_index, and their filenames are
    # exactly what build_matrix assigned.
    #
    # The second element of the key is a DEPENDENCY and not a preference, and
    # it is second rather than first on purpose. `repeat` stays the primary
    # key, so the rounds above are unchanged; within one round the CatBoost
    # cell for a scenario runs before the CatBoost-mode cell for it, because
    # `scenarios.mojotrees_catboost_mode_params` takes CatBoost's RESOLVED
    # learning rate for that cell out of the run's catboost_readback.json and
    # refuses by name without it. Putting the arm rank first would silently
    # restore the arm-blocked order this sort exists to remove, which is the
    # defect that cost a real result.
    #
    # Every other arm shares rank 0 and Python's sort is stable, so they keep
    # build order exactly as before.
    runnable.sort(key=lambda job: (job["repeat"], CELL_ORDER.get(job["engine"], 0)))
    skipped = [job for job in jobs if "skip" in job]
    print(f"run {run_id}: {len(runnable)} runs, {len(skipped)} skipped")
    print(comparator_banner())
    for job in skipped:
        print(f"  skip {label(job)}: {job['skip']}")

    if args.dry_run:
        for job in runnable:
            print(f"  would run {label(job)}")
        write_outputs(run_dir, run_id, [], manifest)
        print(f"\nmatrix written to {run_dir}. Nothing was run.")
        return 0

    records = []
    runnable_records = []
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
                "arm": job.get("arm") or job["engine"],
                "axis": job.get("axis"),
                "device_requested": job["device"],
                "threads": job["threads"],
                "repeat": job["repeat"],
                "error": {"type": "Timeout", "message": f"exceeded {args.timeout}s"},
            }
        record.setdefault("label", label(job))
        records.append(record)
        # Held separately from the skips, because a skip is a decision the
        # matrix made with a reason and a failure is a cell that was meant
        # to produce a number and did not. Only this list is judged.
        runnable_records.append(record)
        state = record.get("status")
        detail = ""
        if state == "ok":
            metric = record.get("primary_metric")
            value = (record.get("quality") or {}).get(metric)
            # Formatted defensively. A cell that came back ok with a null
            # metric used to raise TypeError here, which killed the runner
            # in the middle of the matrix and threw away every record it
            # had not written yet.
            detail = f"{metric}={value:.6g}" if isinstance(value, float) else f"{metric}={value}"
        else:
            detail = (record.get("error") or {}).get("message", "")[:120]
        print(f"{state} {detail}")

    for job in skipped:
        records.append(
            {
                "schema_version": 1, "status": "skipped",
                "scenario": job["scenario"], "engine": job["engine"],
                "arm": job.get("arm") or job["engine"],
                "axis": job.get("axis"),
                "device_requested": job["device"], "threads": job["threads"],
                "repeat": 0, "skip_reason": job["skip"],
            }
        )

    write_outputs(run_dir, run_id, records, manifest)
    print(f"\nwrote {run_dir}/records.json")

    # A positive test, not a list of known-bad statuses. A record with no
    # `status` field at all, or with a status nobody has thought of yet,
    # used to fall through the old `status in ("error", "timeout")` filter
    # and be counted as a cell that ran.
    failed = [r for r in runnable_records if r.get("status") != "ok"]
    if not failed:
        print(
            "next: `python bench/real_data/verify.py "
            f"{os.path.join(run_dir, 'records.json')}` for the correctness "
            "verdict, then report.py for the timings."
        )
        return 0

    report_infrastructure_failure(failed, len(runnable_records), run_dir)
    return EXIT_INFRASTRUCTURE


if __name__ == "__main__":
    raise SystemExit(main())
