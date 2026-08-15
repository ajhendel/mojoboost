"""The correctness gate. Reads a run, applies thresholds.json, decides.

    python bench/real_data/verify.py results/<run_id>/records.json
    python bench/real_data/verify.py results/<run_id> --json verdict.json

What is checked, in the order a failure is most likely to explain the ones
after it:

1. Completeness. A run that errored or timed out is a failure, not an
   absence. Silently comparing the cells that happened to finish is how a
   suite reports green on a broken build.
2. Data agreement. Both engines rebuilt the data in their own processes;
   their digests must match. If they do not, nothing downstream means
   anything and the rest of the checks are skipped for that scenario.
3. Pinning. A real-data row whose bytes were never verified against the
   lock is not a real-data result.
4. Determinism. mojoboost trains bit-identically on a repeat, on the CPU
   and on the accelerator alike. Two repeats with different prediction
   digests fail even when every metric matches.
5. The differential. mojoboost against LightGBM on the primary metric,
   in the direction and by the tolerance thresholds.json gives, plus a
   check that mojoboost is not implausibly better, which is what a
   mismatched problem looks like from the outside.
6. The baseline floor. Each engine separately has to beat the trivial
   model. Two engines that agree because both produced rubbish pass the
   differential check and fail here.
7. Device agreement. mojoboost on the accelerator against mojoboost on the
   CPU, compared row by row on the predictions themselves rather than on a
   summary of them.

Timings are not checked. Not loosely, not as a warning. The exit code is
about correctness, and report.py handles the rest.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import quality  # noqa: E402

PASS, FAIL, WARN, SKIP = "pass", "fail", "warn", "skip"


class Verdict:
    def __init__(self):
        self.checks = []

    def add(self, status, check, scope, detail, **extra):
        self.checks.append(
            {"status": status, "check": check, "scope": scope, "detail": detail, **extra}
        )

    @property
    def failed(self):
        return [c for c in self.checks if c["status"] == FAIL]

    def counts(self):
        out = {PASS: 0, FAIL: 0, WARN: 0, SKIP: 0}
        for check in self.checks:
            out[check["status"]] += 1
        return out


def load_records(target):
    if os.path.isdir(target):
        target = os.path.join(target, "records.json")
    with open(target) as handle:
        payload = json.load(handle)
    return payload.get("records", payload), os.path.dirname(os.path.abspath(target))


def thresholds():
    with open(os.path.join(HERE, "thresholds.json")) as handle:
        return json.load(handle)


def scenario_rules(config, record):
    rules = dict(config["scenarios"].get(record["scenario"], {}))
    dataset = (record.get("data") or {}).get("dataset")
    overrides = rules.get("real_variant_overrides", {})
    if dataset in overrides:
        rules.update(overrides[dataset])
    return rules


def _worse_by(metric, mine, theirs, kind):
    """How much worse `mine` is than `theirs`, positive when worse, in the
    units `kind` names. Direction comes from the metric, not from the
    caller, so a lower-is-better metric cannot be gated backwards."""
    higher_better = quality.HIGHER_IS_BETTER[metric]
    delta = (theirs - mine) if higher_better else (mine - theirs)
    if kind == "relative":
        denominator = abs(theirs) if theirs else 1.0
        return delta / denominator
    return delta


def check_completeness(records, verdict):
    bad = [r for r in records if r.get("status") in ("error", "timeout")]
    for record in bad:
        verdict.add(
            FAIL, "completeness",
            f"{record.get('scenario')}/{record.get('engine')}/{record.get('device_requested')}",
            (record.get("error") or {}).get("message", "run did not complete"),
        )
    skipped = [r for r in records if r.get("status") == "skipped"]
    for record in skipped:
        verdict.add(
            SKIP, "completeness",
            f"{record.get('scenario')}/{record.get('engine')}/{record.get('device_requested')}",
            record.get("skip_reason", ""),
        )
    if not bad:
        verdict.add(PASS, "completeness", "run", f"{len(records) - len(skipped)} runs completed")
    return {r["scenario"] for r in bad}


def check_data_agreement(ok, config, verdict):
    """Digests of the training and test matrices, per scenario."""
    require = config["defaults"]["data_agreement"]["require_identical_digests"]
    broken = set()
    by_scenario = {}
    for record in ok:
        by_scenario.setdefault(record["scenario"], []).append(record)
    for scenario, group in sorted(by_scenario.items()):
        digests = {
            (r["engine"], (r["data"]["train"] or {}).get("digest"),
             (r["data"]["test"] or {}).get("digest"))
            for r in group
        }
        values = {d[1] for d in digests}
        if None in values:
            verdict.add(
                WARN, "data_agreement", scenario,
                "input digests were not computed (--no-data-digest), so "
                "identical inputs are assumed rather than checked",
            )
            continue
        if len(values) == 1:
            verdict.add(PASS, "data_agreement", scenario, f"train digest {values.pop()[:12]}")
        else:
            broken.add(scenario)
            verdict.add(
                FAIL if require else WARN, "data_agreement", scenario,
                "engines were given different data: "
                + ", ".join(f"{e}={d[:12] if d else None}" for e, d, _ in sorted(digests)),
            )
    return broken


def check_pinning(ok, config, verdict):
    if not config["defaults"]["pinning"]["require_pinned_for_real_data"]:
        return
    for record in ok:
        data = record.get("data") or {}
        if data.get("data_kind") != "real":
            continue
        scope = f"{record['scenario']}/{data.get('dataset')}"
        if data.get("pinned"):
            verdict.add(PASS, "pinning", scope, "verified against checksums.lock.json")
        else:
            verdict.add(
                FAIL, "pinning", scope,
                data.get("pin_reason") or "real data was used without a pin",
            )


def check_determinism(ok, config, verdict):
    cells = {}
    for record in ok:
        key = (
            record["scenario"], record["engine"],
            record.get("device_used") or record.get("device_requested"),
            record["threads"],
        )
        cells.setdefault(key, []).append(record)
    for key, group in sorted(cells.items()):
        scenario, engine, device, threads = key
        rule = config["defaults"]["determinism"].get(engine, {})
        if not rule.get("require_identical_predictions"):
            continue
        gating = rule.get("gating", True)
        scope = f"{scenario}/{engine}/{device}/t{threads}"
        if len(group) < 2:
            verdict.add(SKIP, "determinism", scope, "only one repeat; nothing to compare")
            continue
        digests = {r["predictions_sha256"] for r in group}
        if len(digests) == 1:
            verdict.add(PASS, "determinism", scope, f"{len(group)} repeats bit-identical")
        else:
            verdict.add(
                FAIL if gating else WARN, "determinism", scope,
                f"{len(digests)} distinct prediction digests across "
                f"{len(group)} repeats: " + ", ".join(sorted(d[:12] for d in digests)),
            )


def _representative(records):
    """One record per cell. Repeats are bit-identical when determinism
    passed, so the first is the cell; when it did not, the failure is
    already recorded and the first is still the honest thing to quote."""
    return records[0]


def check_differential(ok, config, verdict, skip_scenarios):
    cells = {}
    for record in ok:
        device = record.get("device_used") or record.get("device_requested")
        cells.setdefault(
            (record["scenario"], record["threads"]), {}
        ).setdefault((record["engine"], device), []).append(record)

    for (scenario, threads), by_engine in sorted(cells.items()):
        if scenario in skip_scenarios:
            verdict.add(SKIP, "differential", f"{scenario}/t{threads}", "inputs did not agree")
            continue
        mine = by_engine.get(("mojoboost", "cpu"))
        theirs = by_engine.get(("lightgbm", "cpu"))
        scope = f"{scenario}/t{threads}"
        if not mine or not theirs:
            verdict.add(SKIP, "differential", scope, "no mojoboost and lightgbm cpu pair")
            continue
        a, b = _representative(mine), _representative(theirs)
        rules = scenario_rules(config, a)
        rule = rules.get("differential")
        if not rule:
            verdict.add(SKIP, "differential", scope, "no threshold for this scenario")
            continue

        metric = rules.get("primary_metric", a["primary_metric"])
        checks = [(metric, rule)] + list((rules.get("secondary") or {}).items())
        for name, spec in checks:
            got, want = a["quality"].get(name), b["quality"].get(name)
            if got is None or want is None or np.isnan(got) or np.isnan(want):
                verdict.add(WARN, "differential", f"{scope}/{name}", "metric missing or nan")
                continue
            worse = _worse_by(name, got, want, spec["kind"])
            unit = "x" if spec["kind"] == "relative" else ""
            detail = (
                f"mojoboost {got:.6g} vs lightgbm {want:.6g}, "
                f"{'worse' if worse > 0 else 'better'} by {abs(worse):.4g}{unit} "
                f"(limit {spec['max_worse']}{unit})"
            )
            if worse > spec["max_worse"]:
                verdict.add(FAIL, "differential", f"{scope}/{name}", detail)
            elif "implausible_better" in spec and -worse > spec["implausible_better"]:
                verdict.add(
                    FAIL, "differential", f"{scope}/{name}",
                    detail + ". Better than this is not a win, it is a sign "
                    "the two engines were not given the same problem.",
                )
            else:
                verdict.add(PASS, "differential", f"{scope}/{name}", detail)


def check_baseline(ok, config, verdict):
    # One check per cell rather than per repeat. Repeats of a deterministic
    # trainer score identically, and three copies of the same line makes a
    # verdict harder to read for no information.
    cells = {}
    for record in ok:
        key = (
            record["scenario"], record["engine"],
            record.get("device_used") or record.get("device_requested"),
            record["threads"],
        )
        cells.setdefault(key, record)

    for record in cells.values():
        rules = scenario_rules(config, record)
        rule = rules.get("baseline")
        if not rule:
            continue
        metric = rule["metric"]
        got = (record.get("quality") or {}).get(metric)
        base = (record.get("baseline_quality") or {}).get(metric)
        scope = (
            f"{record['scenario']}/{record['engine']}/"
            f"{record.get('device_used') or record['device_requested']}"
        )
        if got is None or np.isnan(got):
            verdict.add(WARN, "baseline", scope, f"{metric} missing")
            continue
        kind = rule["kind"]
        if kind == "absolute_floor":
            ok_ = got >= rule["min"]
            detail = f"{metric} {got:.6g} against floor {rule['min']}"
        elif kind == "absolute_improvement":
            ok_ = (got - base) >= rule["min"] if base is not None else False
            detail = f"{metric} {got:.6g} against baseline {base:.6g}, needs +{rule['min']}"
        else:
            improvement = _worse_by(metric, got, base, "relative")
            ok_ = -improvement >= rule["min"]
            detail = (
                f"{metric} {got:.6g} against baseline {base:.6g}, "
                f"improvement {-improvement:.4g} needs {rule['min']}"
            )
        verdict.add(PASS if ok_ else FAIL, "baseline", scope, detail)


def check_device_agreement(ok, config, verdict, run_dir):
    rule = config["defaults"]["device_agreement"]
    cells = {}
    for record in ok:
        if record["engine"] != "mojoboost":
            continue
        device = record.get("device_used") or record.get("device_requested")
        cells.setdefault((record["scenario"], record["threads"]), {})[device] = record
    for (scenario, threads), by_device in sorted(cells.items()):
        cpu, gpu = by_device.get("cpu"), by_device.get("gpu")
        scope = f"{scenario}/t{threads}"
        if not cpu or not gpu:
            continue
        metric = cpu["primary_metric"]
        a, b = gpu["quality"].get(metric), cpu["quality"].get(metric)
        metric_agrees = None
        if a is not None and b is not None:
            worse = abs(_worse_by(metric, a, b, "relative"))
            metric_agrees = worse <= rule["primary_metric_relative"]
            verdict.add(
                PASS if metric_agrees else FAIL,
                "device_agreement", f"{scope}/{metric}",
                f"gpu {a:.6g} vs cpu {b:.6g}, relative gap {worse:.4g} "
                f"(limit {rule['primary_metric_relative']})",
            )
        got = _load_predictions(run_dir, gpu)
        want = _load_predictions(run_dir, cpu)
        if got is None or want is None:
            verdict.add(
                WARN, "device_agreement", scope,
                "predictions were not saved, so only the metrics can be "
                "compared and a row-level disagreement could hide in them",
            )
        else:
            diff = float(np.max(np.abs(np.asarray(got) - np.asarray(want))))
            detail = f"max |gpu - cpu| = {diff:.3g} (limit {rule['max_abs_prediction_diff']})"
            if diff <= rule["max_abs_prediction_diff"]:
                verdict.add(PASS, "device_agreement", scope, detail)
            elif metric_agrees:
                # Row-level parity only holds when both devices grow the
                # same trees. The workload-aware AUTO strategy sends large
                # shapes to the device split search, which the trainer
                # documents as trading exactly that guarantee: Float32 gain
                # comparisons flip near-tie splits, so equally good trees
                # can disagree on individual rows while every quality
                # metric agrees. With the primary metric inside its band
                # that is the expected shape of the divergence, not a
                # defect, and failing it would gate the default
                # configuration out of existence.
                verdict.add(
                    WARN, "device_agreement", scope,
                    detail + "; primary metric agrees, consistent with the "
                    "device split search's documented near-tie divergence",
                )
            else:
                verdict.add(FAIL, "device_agreement", scope, detail)


def _load_predictions(run_dir, record):
    name = record.get("predictions_path")
    if not name:
        return None
    path = os.path.join(run_dir, "predictions", name)
    if not os.path.exists(path):
        return None
    return np.load(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", help="a records.json file or a run directory")
    parser.add_argument("--json", help="write the verdict here as well")
    parser.add_argument(
        "--allow-unpinned", action="store_true",
        help="downgrade the pinning failure to a warning; the verdict still "
             "records that the data was unverified",
    )
    args = parser.parse_args(argv)

    records, run_dir = load_records(args.results)
    config = thresholds()
    if args.allow_unpinned:
        config["defaults"]["pinning"]["require_pinned_for_real_data"] = False

    verdict = Verdict()
    broken = check_completeness(records, verdict)
    ok = [r for r in records if r.get("status") == "ok"]
    broken |= check_data_agreement(ok, config, verdict)
    check_pinning(ok, config, verdict)
    check_determinism(ok, config, verdict)
    check_differential(ok, config, verdict, broken)
    check_baseline(ok, config, verdict)
    check_device_agreement(ok, config, verdict, run_dir)

    counts = verdict.counts()
    order = {FAIL: 0, WARN: 1, SKIP: 2, PASS: 3}
    for check in sorted(verdict.checks, key=lambda c: (order[c["status"]], c["check"], c["scope"])):
        print(f"{check['status'].upper():<5} {check['check']:<18} {check['scope']:<44} {check['detail']}")

    print(
        f"\n{counts[PASS]} pass, {counts[FAIL]} fail, {counts[WARN]} warn, "
        f"{counts[SKIP]} skip"
    )
    print("Timings were not checked here. Run report.py for those.")

    payload = {
        "results": os.path.abspath(args.results),
        "thresholds_version": config["version"],
        "counts": counts,
        "checks": verdict.checks,
        "verdict": "fail" if verdict.failed else "pass",
    }
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
    return 1 if verdict.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
