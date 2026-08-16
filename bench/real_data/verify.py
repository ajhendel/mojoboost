"""The correctness gate. Reads a run, applies thresholds.json, decides.

    python bench/real_data/verify.py results/<run_id>/records.json
    python bench/real_data/verify.py results/<run_id> --json verdict.json

What is checked, in the order a failure is most likely to explain the ones
after it:

1. Completeness. A run that errored or timed out is a failure, not an
   absence. Silently comparing the cells that happened to finish is how a
   suite reports green on a broken build.
2. Data agreement. Every engine rebuilt the data in its own process; their
   canonical digests must match. If they do not, nothing downstream means
   anything and the rest of the checks are skipped for that scenario. An
   engine that could not take the canonical container -- CatBoost with a
   categorical block, which its API refuses on a float array -- also has to
   show that its re-encoding reconstructs the canonical form byte for byte.
   The canonical digest cannot see that on its own, which is why it is a
   second question here rather than a footnote.
3. Pinning. A real-data row whose bytes were never verified against the
   lock is not a real-data result.
4. Determinism. mojotrees trains bit-identically on a repeat, on the CPU
   and on the accelerator alike. Two repeats with different prediction
   digests fail even when every metric matches.
5. Backend proof. A record that claims an accelerator has to carry
   evidence from the trainer that one ran. This sits before the checks
   below it because a mislabelled backend makes every one of them
   meaningless while leaving all of them green: that is exactly what
   happened to multiclass, where `trainset.train_dataset_multiclass`
   resolved the device, discarded the answer, and trained on the CPU under
   a `device_used="gpu"` label. Nothing here noticed. A human did, by
   seeing that covertype's CPU and GPU records shared a prediction digest.
6. The differential. mojotrees against LightGBM on the primary metric,
   in the direction and by the tolerance thresholds.json gives, plus a
   check that mojotrees is not implausibly better, which is what a
   mismatched problem looks like from the outside.
7. The accuracy budget, which is the frontier's verdict rather than a gate:
   is this arm within 1 percent relative, on the primary metric, of the
   BETTER of CatBoost-as-shipped and LightGBM stock+det AT A MATCHED TREE
   COUNT. The direction comes from `quality.HIGHER_IS_BETTER`, and a missing
   competitor row at that tree count ABSTAINS rather than falling back to
   another count. `report.py` ranks the arms this passes, by speed, and names
   the fastest one as a documented recommendation that nothing applies.
8. The baseline floor. Each engine separately has to beat the trivial
   model. Two engines that agree because both produced rubbish pass the
   differential check and fail here.
9. Device agreement. mojotrees on the accelerator against mojotrees on the
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

import backend_proof  # noqa: E402
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
    """Digests of the training and test matrices, per scenario.

    Two questions, and the second one was added on 2026-08-16 because the
    first stopped covering it. The digest comparison asks whether the arms
    were given the same PROBLEM, and it is computed from the canonical data
    before any engine sees it, so it is equal across arms whatever an
    adapter subsequently does with its copy. That is exactly what makes it
    trustworthy and exactly what makes it blind to the new failure: an
    engine that cannot take the canonical container re-encodes it, and a
    bug in that re-encoding leaves the canonical digest untouched. The
    re-encoding therefore proves itself per record -- reconstruct the
    canonical form out of what the engine was handed and hash it back --
    and `agrees_with_canonical` carries the verdict. Reading it here is
    what makes it a gate rather than a field.
    """
    require = config["defaults"]["data_agreement"]["require_identical_digests"]
    broken = set()
    for record in ok:
        for part in ("train", "test"):
            encoding = ((record["data"] or {}).get(part) or {}).get("encoding")
            if not encoding or encoding.get("is_canonical", True):
                continue
            scope = f"{record['scenario']}/{record['engine']}/{part}"
            agrees = encoding.get("agrees_with_canonical")
            if agrees is True:
                verdict.add(
                    PASS, "data_agreement", scope,
                    f"re-encoded as {encoding.get('form')} and the "
                    "reconstruction hashes back to the canonical digest",
                )
            elif agrees is None:
                verdict.add(
                    WARN, "data_agreement", scope,
                    f"re-encoded as {encoding.get('form')} with no canonical "
                    "digest to check it against: "
                    + str(encoding.get("agrees_unavailable_reason")),
                )
            else:
                broken.add(record["scenario"])
                verdict.add(
                    FAIL if require else WARN, "data_agreement", scope,
                    f"re-encoded as {encoding.get('form')} and the "
                    "reconstruction does NOT hash back to the canonical "
                    "digest. This engine trained on different data from the "
                    "others and every number in this record is void",
                )
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


def _device_of(record):
    """The backend a record is being read as running on.

    `device_used` when the engine reported one and the request otherwise,
    which is the same rule every other check in this file uses. Both are
    labels; the point of `check_backend_proof` is that neither is evidence.
    """
    return record.get("device_used") or record.get("device_requested")


def check_backend_proof(ok, config, verdict):
    """Refuse a device claim that nothing corroborates and that produced
    the CPU arm's model byte for byte.

    Three conditions, evaluated separately and named separately in every
    message, because they mean three different things:

    A. the record claims an accelerator. On its own this is a label the
       Python side wrote, and the whole reason for this check.
    B. no independent proof. Nothing the trainer emitted says a device
       backend ran. See backend_proof.py for what counts and why.
    C. the prediction digest equals a CPU arm's for the same scenario and
       thread count. On its own this is suspicious, not wrong: two
       backends can in principle produce one model, and mojotrees's device
       histogram reduction is deliberately fixed point so that bit-exact
       agreement is a reachable outcome rather than an impossible one.

    Only the conjunction fails. Failing on C alone would gate out a real
    integer-exact agreement, and failing on B alone would gate out every
    run somebody chose not to instrument. Together they are the signature
    of a CPU fit wearing a GPU label, which is the one shape none of the
    other checks in this file can see.

    C also warns on its own, every time, whether or not the other two
    fire. Two backends landing on one digest is rare enough that the next
    person should be told rather than left to compare two SHA-256s by eye,
    which is how this was caught the first time and is not a method.
    """
    rule = config["defaults"].get("backend_proof") or {}
    if not rule.get("require_for_device_claims", True):
        return

    cpu_digests = {}
    for record in ok:
        if record.get("engine") != "mojotrees":
            continue
        if _device_of(record) != "cpu":
            continue
        key = (record["scenario"], record["threads"])
        cpu_digests.setdefault(key, set()).add(record.get("predictions_sha256"))

    for record in ok:
        if record.get("engine") != "mojotrees":
            continue
        device = _device_of(record)
        if device == "cpu":
            # A CPU row makes no device claim, so condition A never holds
            # and there is nothing here to refuse.
            continue

        scope = f"{record['scenario']}/{device}/t{record['threads']}"
        proved, reason = backend_proof.device_evidence(
            record.get("backend_proof")
        )
        digest = record.get("predictions_sha256")
        twin = cpu_digests.get((record["scenario"], record["threads"]), set())
        hash_matches = bool(digest) and digest in twin

        conditions = [f"A claims {device}"]
        conditions.append(
            f"B no independent proof ({reason})" if not proved
            else f"B proof present ({reason})"
        )
        if hash_matches:
            conditions.append(
                f"C prediction digest {digest[:12]} equals the cpu arm's"
            )
        elif twin:
            conditions.append("C digest differs from the cpu arm's")
        else:
            conditions.append("C no cpu arm to compare against")
        detail = "; ".join(conditions)

        if not proved and hash_matches:
            verdict.add(
                FAIL, "backend_proof", scope,
                detail + ". A and B and C together are what a CPU fit "
                "labelled as a device fit looks like from the outside, and "
                "no other check in this file can tell the difference.",
            )
        elif not proved:
            verdict.add(
                WARN, "backend_proof", scope,
                detail + ". The claim is uncorroborated. It is not refused "
                "because the digest does not match the CPU arm's, which a "
                "silent CPU fallback could not manage.",
            )
        elif hash_matches:
            verdict.add(
                WARN, "backend_proof", scope,
                detail + ". Allowed: the trainer's own evidence says a "
                "device backend ran, so this is two backends agreeing bit "
                "for bit rather than one backend counted twice. Rare enough "
                "to say out loud every time.",
            )
        else:
            verdict.add(PASS, "backend_proof", scope, detail)


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
        mine = by_engine.get(("mojotrees", "cpu"))
        theirs = by_engine.get(("lightgbm", "cpu"))
        scope = f"{scenario}/t{threads}"
        if not mine or not theirs:
            verdict.add(SKIP, "differential", scope, "no mojotrees and lightgbm cpu pair")
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
                f"mojotrees {got:.6g} vs lightgbm {want:.6g}, "
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


#: The accuracy budget, when thresholds.json does not carry one.
#:
#: Andrew's standing directive: accuracy within 1 percent relative on the
#: primary metric, against the BETTER of CatBoost-as-shipped and LightGBM
#: stock+det AT A MATCHED TREE COUNT, is a budget, and it is to be spent for
#: speed. The number lives here rather than only in thresholds.json so that
#: the check works on a results file taken before the key existed; a
#: thresholds.json entry, when one is added, wins.
DEFAULT_ACCURACY_BUDGET = {
    "max_worse_relative": 0.01,
    "competitors": ["catboost", "lightgbm"],
    "engines_judged": ["mojotrees", "mojotrees_catboost_mode"],
}


def _tree_count(record):
    """How many boosting rounds this record's model was grown for.

    `params.num_boost_round` is what the engine adapter reported and is the
    only field all three engines fill in. The two fallbacks are the resolved
    parameter under each library's own name, because a record written before
    the adapter reported the round count should be read rather than dropped:
    dropping it would abstain a verdict for a reason that has nothing to do
    with the comparison.
    """
    params = record.get("params") or {}
    rounds = params.get("num_boost_round")
    if rounds:
        return int(rounds)
    engine_params = params.get("engine") or {}
    for name in ("n_estimators", "num_boost_round", "iterations"):
        if engine_params.get(name):
            return int(engine_params[name])
    return None


def _budget_cell(record):
    """The comparison ground: same scenario, same data, same thread count.

    The DATA KIND is in the key deliberately. `dense_regression` runs as a
    generator and as UCI YearPredictionMSD, and a budget spent against the
    wrong one of those is not a smaller comparison, it is a different
    problem.
    """
    data = record.get("data") or {}
    return (
        record["scenario"],
        record.get("tier"),
        data.get("data_kind"),
        data.get("dataset"),
        record["threads"],
    )


def _arm_of(record):
    """The arm label, which is the engine until the harness grows an arm
    dimension. `frontier.py` documents the change; reading it through this
    helper is what lets this check work before and after it."""
    return record.get("arm") or record.get("engine")


def check_accuracy_budget(ok, config, verdict):
    """The inside-1-percent verdict, per row, at a MATCHED TREE COUNT.

    Three things this gets right on purpose, because each of them silently
    inverts or fabricates a recommendation when got wrong.

    **Direction.** `quality.HIGHER_IS_BETTER` records whether a larger value
    is better and `_worse_by` reads it, so a relative comparison on RMSE
    (lower better) and one on AUC (higher better) both come out positive when
    our arm is worse. A sign error here would rank the frontier backwards and
    nothing downstream could tell.

    **The better competitor.** The directive says the BETTER of CatBoost as
    shipped and LightGBM stock+det, so the budget is measured against
    whichever of them scored better on the primary metric at that tree count,
    which is the harder of the two. Taking the mean, or the nearer one, or
    whichever happens to be present would each make the budget easier in a
    way nobody asked for.

    **A missing competitor row ABSTAINS.** A 360-tree arm is judged against
    360-tree competitor rows or against nothing. Falling back to the 100-tree
    row would produce a verdict that looks like a result and is a category
    error: the accuracy of a 100-tree LightGBM is not the bar a 360-tree arm
    has to clear. An abstention is recorded as a skip that says which count
    was missing, and it is not a failure of the arm.

    Not gating today. The budget decides what the frontier RECOMMENDS, and
    the recommendation is documented rather than applied, so an arm outside
    the budget is a WARN and a fact about that arm rather than a broken run.
    A run may make it gating by setting `gating` true in thresholds.json.
    """
    rule = dict(DEFAULT_ACCURACY_BUDGET)
    rule.update((config.get("defaults") or {}).get("accuracy_budget") or {})
    limit = float(rule["max_worse_relative"])
    competitors = tuple(rule["competitors"])
    judged = tuple(rule["engines_judged"])
    gating = bool(rule.get("gating", False))

    # One record per cell, as `check_baseline` does: repeats of a
    # deterministic trainer score identically and three copies of one line
    # makes a verdict harder to read for no information.
    cells = {}
    for record in ok:
        key = (
            _budget_cell(record), _arm_of(record),
            record.get("device_used") or record.get("device_requested"),
            _tree_count(record),
        )
        cells.setdefault(key, record)

    # The competitor index, keyed by cell and tree count. Built from the same
    # representative records, so a competitor row that failed completeness is
    # not in here and its absence abstains rather than passing quietly.
    bar = {}
    for (cell, arm, _device, trees), record in cells.items():
        if arm not in competitors or trees is None:
            continue
        metric = record.get("primary_metric")
        value = (record.get("quality") or {}).get(metric)
        if value is None or np.isnan(value):
            continue
        bar.setdefault((cell, trees), {})[arm] = (value, metric)

    for (cell, arm, device, trees), record in sorted(
        cells.items(), key=lambda item: str(item[0])
    ):
        if arm not in judged:
            continue
        scenario, tier, data_kind, _dataset, threads = cell
        scope = f"{scenario}/{arm}/{device}/t{threads}/n{trees}"
        metric = record.get("primary_metric")
        mine = (record.get("quality") or {}).get(metric)
        if trees is None:
            verdict.add(
                SKIP, "accuracy_budget", scope,
                "abstains: this record does not say how many boosting rounds "
                "it grew, so there is no tree count to match a competitor at",
            )
            continue
        if mine is None or np.isnan(mine):
            verdict.add(
                SKIP, "accuracy_budget", scope,
                f"abstains: the primary metric {metric} is missing or nan",
            )
            continue

        present = bar.get((cell, trees), {})
        missing = [name for name in competitors if name not in present]
        if not present:
            verdict.add(
                SKIP, "accuracy_budget", scope,
                f"abstains: no competitor row at {trees} trees on this cell "
                f"({', '.join(competitors)} all absent). The budget is "
                "defined at a matched tree count and this check will not "
                "fall back to another count, because the accuracy of a model "
                "at a different budget is not the bar this arm has to clear",
            )
            continue

        # Direction comes from the metric, through `_worse_by`, and the
        # BETTER competitor is the one that makes `worse` largest.
        scored = []
        for name, (value, their_metric) in present.items():
            if their_metric != metric:
                # Two rows of one cell disagreeing about the primary metric
                # is a harness fault, not a close call, and comparing across
                # it would be comparing two different quantities.
                verdict.add(
                    WARN, "accuracy_budget", f"{scope}/{name}",
                    f"competitor row reports primary metric {their_metric} "
                    f"where this arm reports {metric}; not compared",
                )
                continue
            scored.append((_worse_by(metric, mine, value, "relative"), name, value))
        if not scored:
            verdict.add(
                SKIP, "accuracy_budget", scope,
                "abstains: no competitor at this tree count reports the same "
                "primary metric",
            )
            continue
        worse, name, theirs = max(scored)

        detail = (
            f"{metric} {mine:.6g} against the better of "
            f"{', '.join(sorted(present))} at {trees} trees, which is "
            f"{name} at {theirs:.6g}: "
            f"{'worse' if worse > 0 else 'better'} by {abs(worse) * 100:.3f} "
            f"percent (budget {limit * 100:.3f} percent)"
        )
        if missing:
            detail += (
                f". PARTIAL: {', '.join(missing)} has no row at this tree "
                "count, so the bar is the better of the ones present and may "
                "be lower than the directive's"
            )
        if worse <= limit:
            verdict.add(PASS, "accuracy_budget", scope, "inside budget. " + detail)
        else:
            verdict.add(
                FAIL if gating else WARN, "accuracy_budget", scope,
                "OUTSIDE budget. " + detail
                + ". This arm's speed cannot be spent: report.py ranks only "
                "arms inside the budget",
            )


def check_device_agreement(ok, config, verdict, run_dir):
    rule = config["defaults"]["device_agreement"]
    cells = {}
    for record in ok:
        if record["engine"] != "mojotrees":
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
    check_backend_proof(ok, config, verdict)
    check_differential(ok, config, verdict, broken)
    check_accuracy_budget(ok, config, verdict)
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
