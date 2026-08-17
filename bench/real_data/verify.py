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
7. The accuracy anchor, which IS the accuracy gate, and it is self-anchored:
   is this arm within tolerance of OUR OWN recorded accuracy for the same arm
   on the same scenario, at the same tree count, on the same device. No peer
   appears in it. The recorded values live in `accuracy_anchors.json` and this
   file only reads them; refreshing one is a deliberate act that shows the old
   and the new number in a diff, which is what stops slow drift.
7b. The peer comparison, which GATES NOTHING and is reported on every run:
   how far this arm is from the BETTER of CatBoost-as-shipped and LightGBM
   stock+det AT A MATCHED TREE COUNT. The direction comes from
   `quality.HIGHER_IS_BETTER`, and a missing competitor row at that tree count
   ABSTAINS rather than falling back to another count. Every line it writes is
   `note`. Until 2026-08-17 this was the accuracy gate and it suppressed a
   speed ranking; see THE TWO ACCURACY AXES below for what changed and why.
8. The baseline floor. Each engine separately has to beat the trivial
   model. Two engines that agree because both produced rubbish pass the
   differential check and fail here.
9. Device agreement. mojotrees on the accelerator against mojotrees on the
   CPU, compared row by row on the predictions themselves rather than on a
   summary of them.
10. Coverage, last, over the verdict itself. Every subject cell has to have
   been NAMED by at least one of the per-cell checks above (4, 5, 7, 8, 9).
   A gate that emits nothing is indistinguishable from a passing gate, and
   on 2026-08-17 the peer comparison did exactly that on an `--arms` run by
   testing engine names against arm ids. This one never fails; it writes
   the silence down as a warning so that it stops reading as green.

Timings are not checked. Not loosely, not as a warning. The exit code is
about correctness, and report.py handles the rest.

Checks 4, 5 and 9 all read a subject arm's CPU row, and from 2026-08-17 that
row is an ORACLE CELL rather than a competitor. It runs so that 5 and 9 have
something to compare an accelerator row against, it may run fewer repeats
than the measured arms, and report.py keeps it out of the speed story. See the
ORACLE CELL block above `ORACLE_CELL_ROLE` for what that changed and, more
importantly, for what it did not.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import backend_proof  # noqa: E402
import generators  # noqa: E402
import quality  # noqa: E402
import scenarios  # noqa: E402

PASS, FAIL, WARN, SKIP = "pass", "fail", "warn", "skip"

#: NOTE, added 2026-08-17. A FIFTH status, and it exists because four were not
#: enough to say one thing this file has to say.
#:
#: The four above are all judgments about the RUN. `fail` and `pass` say the
#: check ran and decided. `warn` says something is missing or unverified, so a
#: reader should look. `skip` says the check declined to run. Every one of them
#: is read as a statement about run health.
#:
#: `note` says: this check ran, it measured what it set out to measure, and the
#: number it produced is a FACT ABOUT A DESIGN CHOICE rather than a statement
#: about whether this run is healthy. The peer accuracy comparison is the only
#: user today. "We are 1.42 percent behind CatBoost" is true, it is worth
#: printing on every run, and it is not a warning: it is what the accuracy of a
#: deliberately different tree shape looks like, and colouring it yellow every
#: run for years teaches a reader to skip yellow.
#:
#: `note` NEVER affects the exit code, exactly as `warn` never did. The exit
#: code is `fail` and nothing else, unchanged by this addition.
NOTE = "note"

#: Every status, in the order a reader should meet them. `note` sits after
#: `warn` and before `skip`: it is not a problem, and it carries more
#: information than a check that declined to run.
STATUSES = (FAIL, WARN, NOTE, SKIP, PASS)


class Verdict:
    def __init__(self):
        self.checks = []

    def add(self, status, check, scope, detail, **extra):
        if status not in STATUSES:
            # A typo'd status used to land silently in `checks` and then
            # KeyError in `counts`, at the end of a run, after the work.
            raise ValueError(f"unknown verdict status {status!r}")
        self.checks.append(
            {"status": status, "check": check, "scope": scope, "detail": detail, **extra}
        )

    @property
    def failed(self):
        return [c for c in self.checks if c["status"] == FAIL]

    def counts(self):
        out = {status: 0 for status in STATUSES}
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


def bayes_floor_of(record):
    """The Bayes floor that applies to this record, or None.

    The record's own `data.bayes_floor` wins, since worker.py stamps it on
    every synthetic record from 2026-08-17; a record written before that is
    resolved through `scenarios.bayes_floor` by scenario name, `data_kind`
    and the recorded generator kwargs. Both paths return None for a real-data
    record and for a scenario that declares no floor, so nothing downstream
    can quote an excess against a floor that was never known. See
    `scenarios.BAYES_FLOOR_RMSE` for the argument.
    """
    data = record.get("data") or {}
    floor = data.get("bayes_floor")
    if not floor:
        floor = scenarios.bayes_floor(
            record.get("scenario"), data.get("data_kind"), data.get("generator_kwargs")
        )
    if not floor:
        return None
    return with_realized_floor(floor, data.get("generator_kwargs"), data.get("split"))


_REALIZED_CACHE = {}


def with_realized_floor(floor, generator_kwargs, split):
    """`generators.with_realized_floor`, cached per (generator, kwargs, split)
    because this runs once per record and regenerating the noise stream is a
    few hundred milliseconds at the large tier. The population floor is what
    the scenario declares; the realized one is what the excess is measured
    against, and the difference is a quarter of a typical arm's excess at the
    standard tier (see `generators.realized_noise_floor`)."""
    if "realized_mse" in floor:
        return dict(floor)
    key = (
        floor.get("generator"),
        json.dumps(generator_kwargs or {}, sort_keys=True),
        json.dumps(split or {}, sort_keys=True),
    )
    if key not in _REALIZED_CACHE:
        _REALIZED_CACHE[key] = generators.with_realized_floor(
            floor, generator_kwargs, split
        )
    return dict(_REALIZED_CACHE[key])


def floor_mse(floor):
    """The MSE an excess is measured against: realized when known, else the
    population value."""
    return float(floor.get("realized_mse", floor["mse"]))


def excess_error(metric, value, floor):
    """The part of `value` the model is responsible for: MSE above the floor.
    Defined for `rmse` (rmse**2 - floor_mse) and for the floor's own metric
    when the two agree; None otherwise, and None when it is not positive,
    because a model at or below the floor has no excess to take a ratio of."""
    if floor is None or value is None or floor.get("metric") != metric:
        return None
    if metric == "rmse":
        excess = float(value) ** 2 - floor_mse(floor)
    else:
        excess = float(value) - float(floor.get("realized_value", floor["value"]))
    return excess if excess > 0 else None


def excess_worse_by(metric, mine, theirs, floor):
    """`_worse_by` on the EXCESS error rather than the raw metric, relative,
    positive when `mine` is worse. Same direction convention as `_worse_by`
    (excess is lower-is-better for every metric that has a floor). None when
    either side has no excess, which is when there is no floor."""
    ours = excess_error(metric, mine, floor)
    peer = excess_error(metric, theirs, floor)
    if ours is None or peer is None:
        return None
    return (ours - peer) / peer


def excess_root_worse_by(metric, mine, theirs, floor):
    """The same gap in the metric's OWN units, sqrt of the excess MSE for
    rmse: ACCURACY_GAP.md quotes both ("28.8 percent behind on excess RMSE,
    which is 66 percent more of it in MSE"), and a reader holding that
    document should be able to find both figures here. None for a metric that
    is not rmse, and whenever `excess_worse_by` is None."""
    if metric != "rmse":
        return None
    ours = excess_error(metric, mine, floor)
    peer = excess_error(metric, theirs, floor)
    if ours is None or peer is None:
        return None
    return (ours ** 0.5 - peer ** 0.5) / peer ** 0.5


def check_completeness(records, verdict):
    bad = [r for r in records if r.get("status") in ("error", "timeout")]
    # The scope names a CELL, so it is the ARM (2026-08-17: cell by arm, role
    # by engine). On an --arms run two failed arms of one engine wrote one
    # scope string here and read as one failure. `_arm_of` is the engine
    # name on a run without --arms, so those scopes are unchanged.
    for record in bad:
        verdict.add(
            FAIL, "completeness",
            f"{record.get('scenario')}/{_arm_of(record)}/{record.get('device_requested')}",
            (record.get("error") or {}).get("message", "run did not complete"),
        )
    skipped = [r for r in records if r.get("status") == "skipped"]
    for record in skipped:
        verdict.add(
            SKIP, "completeness",
            f"{record.get('scenario')}/{_arm_of(record)}/{record.get('device_requested')}",
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
    # Needed only for the one-repeat branch below, and computed once rather
    # than per cell. See the ORACLE CELL block for what it is.
    accelerator_keys = accelerator_cells(ok)
    cells = {}
    for record in ok:
        # KEYED BY ARM, not by engine, since 2026-08-17. Every cell key in
        # this file that groups repeats has to be, and this one was not: on
        # an `--arms` run one engine name covers dozens of arms at different
        # tree counts and bin counts, and an engine key pooled all of them
        # into one "cell" whose digests naturally differed, so this check
        # would have FAILED a perfectly deterministic trainer for the crime
        # of having been asked forty different questions. On a run without
        # `--arms` the arm IS the engine name and nothing here changes.
        # The threshold rule stays keyed by ENGINE, because thresholds.json
        # is written per engine and every arm of one engine shares it.
        key = (
            record["scenario"], _arm_of(record),
            record.get("device_used") or record.get("device_requested"),
            record["threads"],
        )
        cells.setdefault(key, []).append(record)
    for key, group in sorted(cells.items()):
        scenario, arm, device, threads = key
        engine = group[0]["engine"]
        rule = config["defaults"]["determinism"].get(engine, {})
        if not rule.get("require_identical_predictions"):
            continue
        gating = rule.get("gating", True)
        scope = f"{scenario}/{arm}/{device}/t{threads}"
        if len(group) < 2:
            # An ORACLE cell reaching this branch is a DESIGN DECISION rather
            # than a short run, and the two are indistinguishable from the
            # message that used to be here. `--oracle-repeats` defaults to 1,
            # so this is the ordinary shape of every cpu subject cell in an
            # accelerator run, and a reader who takes it for an accident will
            # go looking for a bug that is not there. The knob is named on the
            # row because a verdict is read on its own, away from this file.
            if is_oracle(group[0], accelerator_keys):
                verdict.add(
                    SKIP, "determinism", scope,
                    "only one repeat; nothing to compare. This is an ORACLE "
                    "cell and the single repeat is deliberate, because "
                    "run.py --oracle-repeats defaults to 1 and "
                    "device_agreement and backend_proof each need one cpu "
                    "prediction per cell and no more. What is LOST by it is "
                    "exactly this check on the cpu path, and it is a real "
                    "loss rather than a restatement, because bit-identity "
                    "is a property of each backend separately. This arm's "
                    "accelerator cell keeps the full --repeats count, so the "
                    "property is still checked there whenever that count is "
                    "two or more. Pass --oracle-repeats 2 when the run is "
                    "about cpu determinism",
                )
            else:
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
        if record.get("engine") not in SUBJECT_ENGINES:
            continue
        if _device_of(record) != "cpu":
            continue
        # KEYED BY ARM, same reason as `check_device_agreement`, and it was
        # keyed by ENGINE until 2026-08-17 for the same insufficient reason.
        # The digest set is the arm's OWN cpu twin, and pooling several arms
        # into one set breaks condition C in both directions: a gpu row could
        # match a DIFFERENT arm's cpu digest and be refused for a coincidence,
        # or its own twin could be hidden among others. Condition C only means
        # anything against the same arm on the other backend, and on an
        # `--arms` run one engine name is dozens of arms.
        key = (record["scenario"], record["threads"], _arm_of(record))
        cpu_digests.setdefault(key, set()).add(record.get("predictions_sha256"))

    for record in ok:
        if record.get("engine") not in SUBJECT_ENGINES:
            continue
        device = _device_of(record)
        if device == "cpu":
            # A CPU row makes no device claim, so condition A never holds
            # and there is nothing here to refuse.
            continue

        scope = (
            f"{record['scenario']}/{_arm_of(record)}/{device}"
            f"/t{record['threads']}"
        )
        proved, reason = backend_proof.device_evidence(
            record.get("backend_proof")
        )
        digest = record.get("predictions_sha256")
        twin = cpu_digests.get(
            (record["scenario"], record["threads"], _arm_of(record)), set()
        )
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
        # KEYED BY ARM since 2026-08-17. This is the comparison of the two
        # PLAIN cells, `mojotrees` as shipped against `lightgbm` stock+det,
        # and the thresholds it applies were written for exactly that pair.
        # Under an engine key an `--arms` run poured every mojotrees arm and
        # every lightgbm arm into the two lists and `_representative` picked
        # whichever came first, so a 25-tree frontier arm could be gated
        # against a 200-tree lightgbm arm and reported as the differential
        # verdict. Keying by arm means the two names below match only the
        # plain cells, which are what the rule is about; a run that has no
        # plain pair gets the SKIP line below rather than a made-up pair.
        cells.setdefault(
            (record["scenario"], record["threads"]), {}
        ).setdefault((_arm_of(record), device), []).append(record)

    for (scenario, threads), by_arm in sorted(cells.items()):
        if scenario in skip_scenarios:
            verdict.add(SKIP, "differential", f"{scenario}/t{threads}", "inputs did not agree")
            continue
        mine = by_arm.get(("mojotrees", "cpu"))
        theirs = by_arm.get(("lightgbm", "cpu"))
        scope = f"{scenario}/t{threads}"
        if not mine or not theirs:
            verdict.add(
                SKIP, "differential", scope,
                "no plain mojotrees and lightgbm cpu pair. This check "
                "compares the two shipped configurations and nothing else; "
                "frontier arms of either engine are ranked by "
                "check_accuracy_peer at a matched tree count instead",
            )
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
        # KEYED BY ARM since 2026-08-17. `setdefault` keeps the FIRST record
        # per key, so under an engine key an `--arms` run had exactly one of
        # its forty mojotrees arms checked against the trivial model and the
        # other thirty-nine published unchecked, with no line saying so.
        # Every arm has to beat the baseline on its own.
        key = (
            record["scenario"], _arm_of(record),
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
            f"{record['scenario']}/{_arm_of(record)}/"
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


#: The mojotrees SUBJECT arms, meaning every arm that is our own engine.
#: `engines.ENGINE_ARM` records these as "subject" or "subject_variant", and
#: the competitor arms as "comparator" or "peer".
#:
#: It exists because three checks in this file were written as a literal
#: `!= "mojotrees"` when `mojotrees` was the only arm of our own, and each of
#: them is a SAFETY gate that silently stops covering a new arm the day one is
#: added. `mojotrees_depthwise` was added on 2026-08-17 and would have
#: published a GPU row with no backend proof and no cpu-versus-gpu agreement
#: check, which is exactly the failure `check_backend_proof` exists to
#: prevent: this campaign has five recorded instances of a measurement that
#: never executed the code it was about.
#:
#: `check_differential` is deliberately NOT widened to this set, and the
#: reason is not an oversight. That check is the HEADLINE pairing, the plain
#: mojotrees arm against the registered comparator, and its limits are tuned
#: to two arms growing the same shape of tree. A depth-wise arm grows a
#: different tree by design, so a gap there is a property of the shape and
#: not a defect, and gating on it would fail a run for a result that is
#: working as intended. Accuracy on the variant arms is gated by
#: `check_accuracy_anchor` instead, which compares each arm against OUR OWN
#: recorded accuracy for that same arm and so cannot punish a tree shape for
#: being a different tree shape. `check_accuracy_peer` reports each arm's
#: distance from the peers beside it and gates nothing.
SUBJECT_ENGINES = (
    "mojotrees",
    "mojotrees_depthwise",
    "mojotrees_catboost_mode",
    # The two CORRECTNESS arms, added 2026-08-17. Membership here is not
    # bookkeeping for these two, it is the arm: `check_device_agreement` skips
    # any engine not in this tuple, and that check is the only thing either
    # arm produces. Both cover a configuration under which a live wrong answer
    # was found the same day by reading code rather than by any gate here --
    # `feature_fraction < 1` on the oblivious device path, and
    # `score_function='Cosine'` under leaf-wise growth. See
    # `scenarios.CORRECTNESS_ARMS`, and `selfcheck.check_correctness_arms`,
    # which fails if either name leaves this tuple.
    "mojotrees_symmetric_colsample",
    "mojotrees_cosine_leafwise",
)


# ---------------------------------------------------------------------------
# THE ORACLE CELL, introduced 2026-08-17. A MEANING CHANGED HERE.
# ---------------------------------------------------------------------------
#
# Read this before reading any CPU row of a run taken on or after that date,
# because the same row means something different from what it meant the day
# before, and nothing about its number changed.
#
# An ORACLE CELL is a subject arm on the cpu in a run that ALSO scheduled that
# same arm on an accelerator. Andrew ruled that day that the GPU is the
# product, and that the CPU backend is kept as a correctness oracle and as the
# portability floor rather than as a competitor, and is no longer optimized.
# So a cpu subject row is still measured, still timed, still recorded and
# still printed in the per-engine table, and it is no longer part of the SPEED
# story. `report.py` keeps it out of the accuracy-budget frontier ranking,
# keeps it out of the headline ratio, and labels it `oracle` in every table it
# appears in. Its number is not hidden and it is not presented as ours either.
#
# WHAT THE ORACLE IS FOR, and it is both of these rather than either. Neither
# is decorative and neither survives a run with no cpu row.
#
#   `check_device_agreement` compares every accelerator prediction against its
#   OWN cpu twin, row by row rather than through a metric. On 2026-08-17 that
#   comparison caught a noise hash domain divergence between the cpu and gpu
#   symmetric growers in the shipped defaults. Without a cpu row it does not
#   fail. It does not run at all, because the loop `continue`s on a missing
#   twin. That silence is the reason a WARN was added to the missing-twin
#   branch when this concept landed.
#
#   `check_backend_proof` condition C is "this accelerator row's prediction
#   digest equals its own cpu twin's", which is how a CPU fit wearing a GPU
#   label was found once already. Without a cpu row that condition degrades to
#   "no cpu arm to compare against" and the three-way conjunction the check
#   refuses on can never fire.
#
# Both of them need ONE cpu prediction per cell rather than three, which is
# what `run.py --oracle-repeats` exists to exploit and why its default is 1.
# What one repeat does NOT buy is `check_determinism` on the cpu cell. That
# check needs two rows of a cell to compare and reports a NAMED skip when an
# oracle cell gives it one. The skip says so on the row itself rather than
# only in this comment, because a reader of a verdict does not read this file.
#
# A cpu subject row in a run with NO accelerator cell for that arm is NOT an
# oracle. It is the only backend that ran, so it IS the measurement, and
# nothing here touches its repeats, its ranking or its label.
#
# WHAT THIS DID NOT CHANGE, stated because it is the obvious thing to assume:
# no threshold moved, no gate changed its verdict for the same inputs, and
# every check in this file still covers every subject arm on both backends.
# The oracle concept is about which rows carry the SPEED claim.

#: `cell_role` on a record, written by `run.py`. Two values today.
ORACLE_CELL_ROLE = "oracle"
MEASURED_CELL_ROLE = "measured"


def _oracle_cell_key(record):
    """The cell an oracle row would be the twin of.

    The DEVICE is deliberately not in this key. The whole question is whether
    the same arm also ran on an accelerator, and a key holding the device
    could never answer it.

    `run.py` builds the same key at matrix time out of the job's `variant`
    where this reads the resolved `data_kind`. A run resolves one variant per
    scenario, so the two are the same discriminator seen from before and from
    after resolution. They are not textually equal, which is why the
    `cell_role` field the runner writes is the authority and this key is only
    the fallback for records taken before that field existed.
    """
    return (
        record.get("scenario"),
        record.get("tier"),
        _arm_of(record),
        record.get("threads"),
        (record.get("data") or {}).get("data_kind"),
    )


def accelerator_cells(records):
    """The keys of every cell that produced a non-cpu SUBJECT row.

    Only `ok` records count. A gpu cell that errored is not a twin, and
    treating it as one would turn its cpu row into an oracle whose reason for
    existing never ran.
    """
    return {
        _oracle_cell_key(record)
        for record in records
        if record.get("status") == "ok"
        and record.get("engine") in SUBJECT_ENGINES
        and _device_of(record) not in (None, "cpu")
    }


def is_oracle(record, accelerator_keys=None):
    """Whether this row is an oracle cell, per the block above.

    `cell_role` on the record wins whenever it is present, because the runner
    knew the whole matrix and this function sees one row.

    Records written before 2026-08-17 carry no such field, and for those the
    answer is COMPUTED from the run. That is why `accelerator_keys` is a
    parameter rather than something derived per row. A caller holding one
    record cannot answer the question, and it gets False rather than a guess.
    False is the safe direction, because it means the row keeps the treatment
    it had before this concept existed.
    """
    role = record.get("cell_role")
    if role:
        return role == ORACLE_CELL_ROLE
    if record.get("engine") not in SUBJECT_ENGINES:
        return False
    if _device_of(record) != "cpu":
        return False
    if not accelerator_keys:
        return False
    return _oracle_cell_key(record) in accelerator_keys


# ---------------------------------------------------------------------------
# THE TWO ACCURACY AXES, SEPARATED 2026-08-17. TWO MEANINGS CHANGED HERE.
# ---------------------------------------------------------------------------
#
# Until this date there was ONE accuracy concept in this harness, the "accuracy
# budget": within 1 percent relative of the BETTER of CatBoost-as-shipped and
# LightGBM stock+det, at a matched tree count. It did two jobs at once and it
# did both of them wrong. It is now two checks with two different jobs.
#
#   `check_accuracy_anchor`  THE GATE. Our accuracy against OUR OWN recorded
#                            accuracy for the same arm on the same scenario.
#                            Self-anchored. No peer appears in it.
#
#   `check_accuracy_peer`    THE SCOREBOARD. Our accuracy against the peers,
#                            reported on every run, prominent, and gating
#                            nothing at all.
#
# WHY THE PEER COMPARISON STOPPED BEING A GATE. Andrew, 2026-08-17: "i think we
# may need to modify our rule re adopting things... it should be about trading
# OUR accuracy for speed. not tied to whatever lightgbm or catboost is fucking
# doing". Three failures, and the second is the one that makes the old design
# unsafe rather than merely awkward.
#
#   1. It blocked changes that cost nothing. The leaf-wise GPU arm went from
#      1.043 s to 0.839 s on 2026-08-17 on bit-identical work, and `report.py`
#      would not rank the improvement, because the arm sits 1.42 percent behind
#      CatBoost against a 1 percent bar. A speed result was hidden by an
#      accuracy figure the speed work could not have moved.
#   2. IT PERMITTED REAL ACCURACY LOSS WHENEVER WE WERE AHEAD. The symmetric
#      arm beats CatBoost at 799k rows, 0.303271 against 0.303468 rmse
#      (measured 2026-08-17, handed to this lane, not taken by it). Under a
#      1 percent bar anchored on CatBoost that arm could give away about 1.07
#      percent of its OWN accuracy and still pass. The old gate was measuring
#      the wrong quantity in both directions at once.
#   3. The bar moved when somebody else shipped. A CatBoost release could fail
#      our gate with no change to our code, and a CatBoost regression could
#      hand us headroom we did not earn.
#
# WHAT DID NOT CHANGE. The peer arithmetic. Same 1 percent, same "better of the
# two", same matched tree count, same direction from `quality.HIGHER_IS_BETTER`,
# same abstention when a competitor row is missing. Every one of those was
# right. Only what it GATES changed, and it now gates nothing.
#
# `report.py` no longer suppresses a row for either verdict. Speed is ranked
# for every arm in every frontier table and both accuracy columns are printed
# beside it. See `report._frontier`.


#: The peer comparison, when thresholds.json does not carry one.
#:
#: The values are the old `DEFAULT_ACCURACY_BUDGET` unchanged, and the name
#: changed on 2026-08-17 because "budget" is now the anchor's word. Andrew's
#: standing directive that these numbers came from: accuracy within 1 percent
#: relative on the primary metric, against the BETTER of CatBoost-as-shipped and
#: LightGBM stock+det AT A MATCHED TREE COUNT. That directive is now a
#: DESCRIPTION of where we stand rather than a rule about what we may adopt.
#: The number lives here rather than only in thresholds.json so that the check
#: works on a results file taken before the key existed; a thresholds.json entry
#: under `defaults.accuracy_peer`, when one is added, wins.
#:
#: THE `gating` KEY IS GONE, deliberately, rather than left dormant at false.
#: It let a run make the peer comparison fail the build, and after Andrew's
#: ruling there is no run for which that is the right thing to do. A dormant
#: key that contradicts a written ruling is an invitation to turn it on.
#:
#: **`xgboost` IS DELIBERATELY NOT IN `competitors`, AND THE DECISION IS NOT
#: THIS FILE'S TO MAKE.** An XGBoost peer arm landed on 2026-08-17, and whether
#: it joins the "better of" set the comparison is taken against is a judgment
#: about what the scoreboard SHOWS, not a wiring detail. Adding it can only
#: make us look further behind, never closer, and it would do so on an arm
#: running XGBoost's own learning rate of 0.3 against everyone else's 0.1 at a
#: matched tree count, which `scenarios.XGBOOST_DELIBERATE_DIVERGENCE` records
#: as a reason accuracy on that column is not evidence about either engine.
#: Left at two engines until Andrew decides. This mattered more when the number
#: gated; it is now a column, so the cost of the decision has gone down and the
#: reason to make it deliberately has not.
DEFAULT_ACCURACY_PEER = {
    "max_worse_relative": 0.01,
    "competitors": ["catboost", "lightgbm"],
    "engines_judged": list(SUBJECT_ENGINES),
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


def check_accuracy_peer(ok, config, verdict):
    """Where we stand against the peers, per row, at a MATCHED TREE COUNT.

    **This gates nothing.** Renamed from `check_accuracy_budget` on 2026-08-17
    and its arithmetic is unchanged; see THE TWO ACCURACY AXES above for why it
    stopped being a gate and what took over. Every line it writes is `note`,
    including the ones that used to be `pass`, because a scoreboard that
    reports green for one arm and yellow for another is a gate wearing a
    different word.

    Three things this gets right on purpose, because each of them silently
    inverts or fabricates a comparison when got wrong.

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

    **A missing competitor row ABSTAINS.** A 360-tree arm is compared against
    360-tree competitor rows or against nothing. Falling back to the 100-tree
    row would produce a line that looks like a result and is a category error:
    the accuracy of a 100-tree LightGBM is not the peer of a 360-tree arm. An
    abstention is recorded as a skip that says which count was missing, and it
    is not a mark against the arm.

    The 1 percent band is still computed and still printed, because "inside the
    band we have been quoting" remains a useful shorthand for a reader. It is
    now a DESCRIPTION of a distance and no longer permission to do anything.
    """
    rule = dict(DEFAULT_ACCURACY_PEER)
    # Both key names are read. `accuracy_peer` is the name from 2026-08-17;
    # `accuracy_budget` is what a thresholds.json written before that date
    # would carry, and dropping it would silently ignore a tolerance somebody
    # had set on purpose.
    defaults = config.get("defaults") or {}
    rule.update(defaults.get("accuracy_budget") or {})
    rule.update(defaults.get("accuracy_peer") or {})
    limit = float(rule["max_worse_relative"])
    competitors = tuple(rule["competitors"])
    judged = tuple(rule["engines_judged"])

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
    #
    # MEMBERSHIP IS BY ENGINE, since 2026-08-17. `competitors` and
    # `engines_judged` are lists of ENGINE names, which is what their names
    # say, and until that date both were tested against the ARM id. On a run
    # without `--arms` the two are equal and nothing showed. On an `--arms`
    # run no arm id is an engine name, so no competitor bar was ever built and
    # every one of our arms fell through `continue` below unjudged, WITH NO
    # LINE SAYING SO: the scoreboard was simply absent from the verdict and
    # from every frontier table that reads it. The reverse of the keying
    # mistake in `check_device_agreement`, and the same silence.
    bar = {}
    for (cell, arm, _device, trees), record in cells.items():
        if record.get("engine") not in competitors or trees is None:
            continue
        metric = record.get("primary_metric")
        value = (record.get("quality") or {}).get(metric)
        if value is None or np.isnan(value):
            continue
        bar.setdefault((cell, trees), {})[arm] = (value, metric, record["engine"])

    for (cell, arm, device, trees), record in sorted(
        cells.items(), key=lambda item: str(item[0])
    ):
        if record.get("engine") not in judged:
            continue
        scenario, tier, data_kind, _dataset, threads = cell
        scope = f"{scenario}/{arm}/{device}/t{threads}/n{trees}"
        metric = record.get("primary_metric")
        mine = (record.get("quality") or {}).get(metric)
        if trees is None:
            verdict.add(
                SKIP, "accuracy_peer", scope,
                "abstains: this record does not say how many boosting rounds "
                "it grew, so there is no tree count to match a competitor at",
            )
            continue
        if mine is None or np.isnan(mine):
            verdict.add(
                SKIP, "accuracy_peer", scope,
                f"abstains: the primary metric {metric} is missing or nan",
            )
            continue

        present = bar.get((cell, trees), {})
        present_engines = {engine for _v, _m, engine in present.values()}
        missing = [name for name in competitors if name not in present_engines]
        if not present:
            verdict.add(
                SKIP, "accuracy_peer", scope,
                f"abstains: no competitor row at {trees} trees on this cell "
                f"({', '.join(competitors)} all absent). The comparison is "
                "defined at a matched tree count and this check will not "
                "fall back to another count, because the accuracy of a model "
                "grown for a different number of rounds is not this arm's peer",
            )
            continue

        # Direction comes from the metric, through `_worse_by`, and the
        # BETTER competitor is the one that makes `worse` largest.
        scored = []
        for name, (value, their_metric, _engine) in present.items():
            if their_metric != metric:
                # Two rows of one cell disagreeing about the primary metric
                # is a harness fault, not a close call, and comparing across
                # it would be comparing two different quantities.
                verdict.add(
                    WARN, "accuracy_peer", f"{scope}/{name}",
                    f"competitor row reports primary metric {their_metric} "
                    f"where this arm reports {metric}; not compared",
                )
                continue
            scored.append((_worse_by(metric, mine, value, "relative"), name, value))
        if not scored:
            verdict.add(
                SKIP, "accuracy_peer", scope,
                "abstains: no competitor at this tree count reports the same "
                "primary metric",
            )
            continue
        worse, name, theirs = max(scored)

        detail = (
            f"{metric} {mine:.6g} against the better of "
            f"{', '.join(sorted(present))} at {trees} trees, which is "
            f"{name} at {theirs:.6g}: "
            f"{'behind' if worse > 0 else 'ahead'} by {abs(worse) * 100:.3f} "
            f"percent (the 1 percent band is {limit * 100:.3f} percent)"
        )
        # THE EXCESS LENS, added 2026-08-17. On the generator variant of a
        # scenario that declares a Bayes floor, the raw gap is mostly floor:
        # 1.7 percent of RMSE is 28.8 percent of the error the model is
        # responsible for. Both numbers go on the line and into the extra
        # fields; a record without a floor (every real-data record, every
        # scenario that declares none) gets the raw gap alone, unchanged.
        floor = bayes_floor_of(record)
        excess_worse = excess_worse_by(metric, mine, theirs, floor)
        excess_fields = {}
        if excess_worse is not None:
            excess_fields = {
                "bayes_floor_mse": floor_mse(floor),
                "bayes_floor_population_mse": float(floor["mse"]),
                "excess_value": excess_error(metric, mine, floor),
                "excess_peer_value": excess_error(metric, theirs, floor),
                "excess_worse_relative": float(excess_worse),
            }
            detail += (
                f". EXCESS over the Bayes floor ({floor['metric']} "
                f"{floor['value']:.6g} by construction, realized on the "
                f"held-out rows as mse {floor_mse(floor):.6g}, generator "
                f"variant only): excess mse {excess_fields['excess_value']:.6g} "
                f"against {excess_fields['excess_peer_value']:.6g}, "
                f"{'behind' if excess_worse > 0 else 'ahead'} by "
                f"{abs(excess_worse) * 100:.3f} percent of the model's own error"
            )
            excess_root = excess_root_worse_by(metric, mine, theirs, floor)
            if excess_root is not None:
                excess_fields["excess_root_worse_relative"] = float(excess_root)
                detail += (
                    f" ({abs(excess_root) * 100:.3f} percent as excess "
                    f"{metric}, the document's lens)"
                )
        if missing:
            detail += (
                f". PARTIAL: {', '.join(missing)} has no row at this tree "
                "count, so this is the better of the ones present and the "
                "distance may read smaller than against the full peer set"
            )
        # BOTH BRANCHES ARE `note`, and the symmetry is the point. Being inside
        # the band is not a pass, because there was nothing to pass; being
        # outside it is not a warning, because a deliberately different tree
        # shape landing 1.42 percent from CatBoost is a design fact and not a
        # sign that this run is unhealthy. The gate is `check_accuracy_anchor`
        # and it does not read this number.
        #
        # The two branches still say different words, because the DISTANCE is
        # the information and a reader scanning for "OUTSIDE" should still find
        # it. What they no longer do is carry different severity.
        band = "inside the 1 percent band. " if worse <= limit else (
            "OUTSIDE the 1 percent band. "
        )
        verdict.add(
            NOTE, "accuracy_peer", scope, band + detail
            + ". This is a scoreboard line and it gates nothing: report.py "
            "ranks every arm on speed whatever this says",
            metric=metric, value=float(mine), peer=name, peer_value=float(theirs),
            worse_relative=float(worse), band=float(limit),
            inside_band=bool(worse <= limit), trees=trees, partial=bool(missing),
            **excess_fields,
        )


#: Where the recorded accuracy anchors live.
#:
#: A SEPARATE FILE FROM thresholds.json, and the reason is the same one that
#: put `checksums.lock.json` beside `sources.json`. thresholds.json holds
#: POLICY: numbers somebody chose, each carrying an argument, edited when the
#: argument changes. An anchor is a MEASUREMENT: a number some run produced,
#: carrying provenance rather than an argument, refreshed when the code moves.
#: Putting a measured value in the policy file makes "what we decided" and
#: "what we observed" indistinguishable in a diff, which is the exact confusion
#: this whole redesign is about. The TOLERANCE stays in thresholds.json, where
#: policy belongs, and only the observed values live here.
#:
#: The file is read at check time and NEVER written by `check_accuracy_anchor`.
#: Writing is `--propose-anchors`, which writes a candidate file somewhere else
#: for a person to read and move into place. See `propose_anchors`.
ACCURACY_ANCHORS_PATH = os.path.join(HERE, "accuracy_anchors.json")


#: The self-anchored accuracy gate, when thresholds.json does not carry one.
#:
#: `max_worse_relative` IS A DIFFERENT QUANTITY FROM THE OLD 1 PERCENT and
#: reusing that number would have been wrong. The old 1 percent was a DISTANCE
#: TO A COMPETITOR, a standing gap Andrew was willing to live with. This is
#: HOW MUCH OF OUR OWN ACCURACY ONE ADOPTED CHANGE MAY SILENTLY COST. At 1
#: percent a single change could give away the whole competitive gap in one
#: commit.
#:
#: 0.25 percent, and here is the argument, with the measured and the inferred
#: parts separated because they are not the same kind of claim.
#:
#:   MEASURED, in this repository: our arms are bit-identical across repeats at
#:   a fixed configuration. `check_determinism` gates it and thresholds.json's
#:   determinism rationale states it. So the anchor comparison HAS NO NOISE
#:   FLOOR for a subject arm: the data is generated from a pure seeded
#:   generator (`selfcheck.check_generators_are_pure`), the config is pinned,
#:   and a metric that moves at all moved because our code moved. Any tolerance
#:   above zero is therefore a deliberate allowance, not headroom for noise.
#:
#:   MEASURED, nearest evidence for the size: thresholds.json's dense_regression
#:   differential rationale records that tie breaking in the split search and
#:   the order of floating-point summation "moves RMSE in the third decimal
#:   place". That observation is about two DIFFERENT engines, not about one
#:   engine before and after a reassociation, so it is an upper bound on the
#:   kind of movement we are allowing for and not a measurement of it.
#:
#:   INFERRED, and not measured by this lane: a change that only reassociates a
#:   reduction (threading, layout, tiling) should move a metric by less than
#:   this. 0.25 percent of an rmse of 0.31 is about 0.0008, which is the fourth
#:   decimal place, one place tighter than the recorded cross-engine figure
#:   above. Nobody has measured the same-arm reassociation magnitude on this
#:   codebase. If a legitimate reassociation trips this gate, the finding is
#:   that the number is wrong and it is one key in thresholds.json to move,
#:   with the measurement that justified moving it written beside it.
#:
#: AND THE GATE IS NOT A REFUSAL. Tripping it does not mean the change may not
#: be made. It means the change may not be made SILENTLY: the anchor is
#: refreshed by a deliberate act, which puts the old and the new number side by
#: side in a diff, which is the whole mechanism.
#:
#: `implausible_better_relative` exists for the reason `check_differential`
#: already has one, in that check's own words: "Better than this is not a win,
#: it is a sign the two engines were not actually given the same problem." The
#: same trap catches an anchor. An arm that suddenly scores 8 percent better
#: than its own recorded value is far more likely to be solving a changed
#: problem, or scoring a changed metric, than to have got that much better, and
#: adopting that number as the next anchor would bake the fault in permanently.
#: 5 percent, chosen by the same argument as the differential's 10 percent and
#: set tighter because this compares an arm against ITSELF rather than against
#: another library. Inferred, not measured.
DEFAULT_ACCURACY_ANCHOR = {
    "max_worse_relative": 0.0025,
    "implausible_better_relative": 0.05,
    "engines_judged": list(SUBJECT_ENGINES),
    "gating": True,
}


def _accuracy_environment(record):
    """The parts of the environment an accuracy anchor is valid within.

    NOT `envinfo.comparable_key`, and the difference matters. That key holds
    the git commit, because two TIMINGS from different commits do not belong in
    one table. An accuracy anchor exists precisely to be compared across
    commits, so a key holding the commit would never match and the gate would
    never fire.

    What is left is the hardware and the architecture, because floating-point
    reduction order and the accelerator's own arithmetic are properties of the
    machine, and an anchor recorded on one machine is not evidence about
    another. A mismatch here does not discard the comparison; it downgrades a
    regression from a failure to a warning that names the mismatch, because "we
    got worse" and "we got worse on a different machine" are different claims
    and only the first should stop a run.
    """
    cpu = ((record.get("environment") or {}).get("cpu") or {})
    return {"arch": cpu.get("arch"), "cpu_model": cpu.get("model")}


def _anchor_key(record):
    """The identity an anchor is recorded against.

    THREADS ARE DELIBERATELY NOT IN THIS KEY. Our arms are reproducible across
    thread counts by design and `check_determinism` gates it, so a metric that
    moves when the thread count moves is a defect this gate should catch rather
    than a new cell it should abstain on. (Competitors are not anchored at all,
    which is what makes this safe: LightGBM produced two distinct prediction
    digests across three repeats on `sparse_highdim` with `deterministic=true`
    already set, recorded in PROFILE_PROTOCOL C9, so a thread-free key would be
    wrong for them.)

    THE DEVICE IS IN THIS KEY. The cpu and the accelerator do not compute the
    same arithmetic here (the accelerator histogram reduction is fixed point on
    purpose), so one anchor covering both would let a real accelerator
    regression hide behind the cpu number, or the reverse.

    THE DATASET AND THE TIER ARE IN THIS KEY for the reason `_budget_cell`
    gives: `dense_regression` runs as a generator and as UCI YearPredictionMSD,
    and an anchor spent against the wrong one is not a smaller comparison, it
    is a different problem.
    """
    data = record.get("data") or {}
    return "|".join(
        str(part) for part in (
            record.get("scenario"),
            record.get("tier"),
            data.get("data_kind"),
            data.get("dataset"),
            _arm_of(record),
            record.get("device_used") or record.get("device_requested"),
            f"n{_tree_count(record)}",
        )
    )


#: THE STALE-ANCHOR MECHANISM, registered 2026-08-17.
#:
#: An anchor is an absolute recorded value and rule 3 of `LANE_RULES.md` says a
#: later run may not become its own baseline. That rule is what makes an anchor
#: worth having and it creates one problem it does not solve: **an anchor can
#: stop describing the model we ship without any number in it changing.**
#:
#: The case that produced this. On 2026-08-17 our `lambda_l2` under every
#: non-symmetric growth policy moved from 0.0 to 1.0
#: (`sklearn.py::_LAMBDA_L2`, declared in `check_parity.py::STOCK_DIVERGENCES`,
#: priced in `ACCURACY_BUDGET.md` section 13). Any anchor for an arm that LEFT
#: THAT KEY UNSET was recorded on a model nobody ships any more. The anchor's
#: value is still a correct record of what that run produced; what is no longer
#: true is that it is the reference for this arm.
#:
#: **The wrong fix, and it is the tempting one: recompute the anchor.** Editing
#: an anchor to match a model nobody measured installs exactly the ratchet the
#: file exists to prevent. So a stale anchor is neither edited nor deleted. It is
#: MARKED, it stops gating, and it says so on every run until a run replaces it.
#:
#: **How staleness is decided, and it is not a date.** A date-based rule needs
#: somebody to remember to write the date down. This rule is a comparison:
#:
#:   an anchor is STALE when the arm it was recorded from LEFT a parameter
#:   UNSET, and the value that parameter RESOLVED TO in the anchored run is not
#:   the value it resolves to now.
#:
#: Both halves come off the anchor entry's own `configuration` block, which
#: `propose_anchors` writes from the record, and the live half is read from the
#: package source. Nothing has to be maintained: an anchor adopted from a run
#: taken after a default moves records the new value and is not stale, so **the
#: run clears the staleness by existing**. That is the design requirement,
#: because the alternative is a hand-maintained list that goes stale about
#: staleness.
#:
#: An anchor with no `configuration` block cannot be judged either way. That is
#: an UNKNOWN and it WARNS, on the precedent of C10's missing-cpu-twin and
#: C11's missing anchor: a check that cannot run is dangerous precisely because
#: it is silent, so it says so rather than passing.
#: `constants` is keyed by GROW POLICY because the default an unset key resolves
#: to is per policy, which is the standing mirroring rule: `symmetrictree`
#: mirrors CatBoost and every other policy mirrors LightGBM. `sklearn.py`'s
#: `_Base._params` branches on exactly that
#: (`l2_default = _CATBOOST_L2_LEAF_REG if catboost_mode else _LAMBDA_L2`), so a
#: single constant per parameter would price a symmetric arm against a
#: leaf-wise default and report staleness that is not there. `None` means the
#: value is not a constant at all and cannot be compared; the symmetric
#: learning rate is DERIVED per fit from the row count, the iteration count and
#: the loss, so there is nothing to read.
STALE_ANCHOR_PARAMETERS = {
    "lambda_l2": {
        "constants": {
            "default": "_LAMBDA_L2",
            "symmetrictree": "_CATBOOST_L2_LEAF_REG",
        },
        "why": (
            "the L2 leaf regularizer, and the denominator of every leaf value "
            "and every split gain. It moved from 0.0 to 1.0 on 2026-08-17 under "
            "every non-symmetric growth policy, so an anchor recorded from an "
            "arm that did not name it describes a different model. A change to "
            "this value reorders which candidate split wins, so it is not a "
            "small movement in a metric, it is a different tree"
        ),
    },
    "learning_rate": {
        "constants": {
            "default": "_LEARNING_RATE",
            "symmetrictree": None,
        },
        "why": (
            "the shrinkage rate. An arm that does not name it follows the "
            "policy's default, so a change to that default changes how much "
            "fitting the anchored budget bought"
        ),
    },
}


def stale_anchor_constant(name, grow_policy):
    """The `sklearn.py` constant an unset `name` resolves to under
    `grow_policy`, or None when there is no constant to read."""
    rule = STALE_ANCHOR_PARAMETERS.get(name)
    if rule is None:
        return None
    constants = rule["constants"]
    if grow_policy in constants:
        return constants[grow_policy]
    return constants["default"]

#: Where the live shipped value of each of those parameters is read from.
#: The Python surface, because that is the surface `bench/real_data` fits
#: through: `basic.py::_Config` forwards a training dict into
#: `sklearn.py::MojoTreesRegressor`, so what an unset key resolves to is that
#: module's constant and not `tree.mojo::TreeParams.default`.
SHIPPED_CONSTANTS_SOURCE = os.path.join(
    HERE, "..", "..", "python", "mojotrees", "sklearn.py"
)


def shipped_constant(name):
    """The value of a module-level numeric constant in `sklearn.py`, or None.

    Read from the SOURCE TEXT and not by importing, because
    `python/mojotrees/sklearn.py` imports the compiled extension and every tool
    in this directory must run with nothing built. A constant that stops being a
    bare numeric literal returns None, which the callers treat as "cannot
    determine" rather than as a value, because an expression is not something a
    gate may guess at.
    """
    import re

    try:
        with open(os.path.abspath(SHIPPED_CONSTANTS_SOURCE)) as handle:
            text = handle.read()
    except OSError:
        return None
    found = re.search(
        r"^" + re.escape(name) + r"\s*=\s*(-?[0-9][0-9._eE+-]*)\s*$",
        text,
        re.MULTILINE,
    )
    if found is None:
        return None
    try:
        return float(found.group(1))
    except ValueError:
        return None


def anchor_configuration(record):
    """The part of an arm's configuration an anchor's validity depends on.

    Three fields and each answers a different question.

    `passed` is what the HARNESS handed the estimator, off the record's resolved
    engine parameters. **`None` there means UNSET and no other value does**,
    because that is the only thing `sklearn.py::_Base._l2_named` and
    `_learning_rate_named` read as "the caller did not name it".

    `followed_default` is `passed`'s `None`s, and it is the load-bearing field. A
    value the harness passed is a property of the ARM and moving a package
    default cannot invalidate it; a value passed as `None` is a property of the
    DEFAULT, and moving the default replaces the model under the anchor's name.
    Note that this is deliberately read off what was PASSED and not off the arm's
    own override dict: `scenarios.mojotrees_params` copies `learning_rate` and
    `lambda_l2` out of `BASE_PARAMS` onto every mojotrees arm, so an arm that
    names neither still passes both, and calling that "followed the default"
    would report staleness on arms a default change cannot touch.

    `shipped_at_record` is the value the relevant `sklearn.py` constant held when
    this run was taken. Recording it is what makes staleness a comparison of two
    recorded facts rather than an inference: `anchor_staleness` reads it against
    the constant's value now, and no version history is needed.
    """
    resolved = (record.get("params") or {}).get("engine") or {}
    grow_policy = resolved.get("grow_policy")
    passed = {}
    followed = []
    shipped = {}
    for name in sorted(STALE_ANCHOR_PARAMETERS):
        value = resolved.get(name)
        passed[name] = None if value is None else float(value)
        if value is None:
            followed.append(name)
        constant = stale_anchor_constant(name, grow_policy)
        shipped[name] = None if constant is None else shipped_constant(constant)
    return {
        "passed": passed,
        "followed_default": followed,
        "shipped_at_record": shipped,
        "grow_policy": grow_policy,
        "note": (
            "`passed` is what the harness handed the estimator, where None means "
            "UNSET. `followed_default` is that subset, and it is the only subset "
            "a package default change can invalidate. `shipped_at_record` is "
            "what the matching sklearn.py constant held when this run was taken, "
            "chosen per grow policy because symmetrictree resolves an unset key "
            "from CatBoost's constant and every other policy from LightGBM's. "
            "verify.anchor_staleness compares the two"
        ),
    }


def anchor_staleness(anchor):
    """Why this anchor no longer describes what we ship, or None.

    Three outcomes and they are deliberately different:

    - `None`: the anchor's unset parameters still resolve to what they resolved
      to, so the anchor describes the model it was recorded from.
    - a dict with `"unknown": True`: the anchor carries no `configuration`
      block, so nothing here can tell. WARNs.
    - a dict naming the parameter: STALE. WARNs and does not gate.

    An explicit `"stale"` block on the entry is authoritative and wins over the
    comparison, so a person can retire an anchor for a reason no comparison can
    see. It is read but never written by any code path in this harness.
    """
    declared = anchor.get("stale")
    if declared:
        detail = dict(declared)
        detail.setdefault("parameter", None)
        detail.setdefault("declared", True)
        return detail

    configuration = anchor.get("configuration")
    if not configuration:
        return {
            "unknown": True,
            "why": (
                "this anchor carries no `configuration` block, so nothing here "
                "can tell whether the parameters its arm left UNSET still "
                "resolve to the values it was recorded at. Anchors proposed "
                "before 2026-08-17 have none. Re-propose from a run to give it "
                "one; until then this anchor cannot be shown to describe the "
                "model we ship, and an anchor that cannot be shown to be "
                "current does not gate"
            ),
        }

    was_shipped = configuration.get("shipped_at_record") or {}
    grow_policy = configuration.get("grow_policy")
    for name in configuration.get("followed_default") or ():
        rule = STALE_ANCHOR_PARAMETERS.get(name)
        if rule is None:
            continue
        constant = stale_anchor_constant(name, grow_policy)
        if constant is None:
            continue
        was = was_shipped.get(name)
        now = shipped_constant(constant)
        if was is None or now is None:
            continue
        if float(was) != float(now):
            return {
                "parameter": name,
                "recorded_at_value": float(was),
                "shipped_value": float(now),
                "why": (
                    f"the arm this anchor was recorded from left `{name}` "
                    f"UNSET, so it took the default. That default was {was:g} "
                    f"when the anchor was recorded and it is {now:g} now "
                    f"(sklearn.py::{constant}, the constant an unset `{name}` "
                    f"resolves to under grow_policy {grow_policy!r}). "
                    f"{rule['why']}"
                ),
                "cleared_by": (
                    "a run of this arm on the current default, then "
                    "`verify.py <run> --propose-anchors <path>` and a "
                    "deliberate adoption. The anchor is NOT recomputed by "
                    "arithmetic: editing a recorded value to match a model "
                    "nobody measured is how the ratchet LANE_RULES rule 3 "
                    "designs out gets installed"
                ),
            }
    return None


def load_accuracy_anchors(path=None):
    """The recorded anchors, or an empty set when the file has none.

    A missing file is not an error. It is the state this harness shipped in on
    2026-08-17, when the anchor concept landed with no anchors adopted, and it
    reads as "nothing is covered yet" rather than as "everything is fine",
    because `check_accuracy_anchor` warns per arm on a missing anchor.
    """
    path = path or ACCURACY_ANCHORS_PATH
    if not os.path.exists(path):
        return {}
    with open(path) as handle:
        payload = json.load(handle)
    return payload.get("anchors") or {}


def check_accuracy_anchor(ok, config, verdict, anchors=None):
    """THE ACCURACY GATE. Our accuracy against OUR OWN recorded accuracy.

    Registered 2026-08-17. No peer appears anywhere in this check. See THE TWO
    ACCURACY AXES above for the ruling that replaced the peer-anchored budget
    with this, and `DEFAULT_ACCURACY_ANCHOR` for where the tolerance came from.

    **The ratchet is designed out, and that is the part that needed the care.**
    If the reference were "the previous run" then losing 0.9 percent ten times
    would pass ten times and end nine percent worse, with every individual step
    defensible and the total invisible. So the reference is an ABSOLUTE
    RECORDED VALUE in `accuracy_anchors.json`, which this check only ever
    READS. It is refreshed by a deliberate act that shows the old and the new
    number in one diff. Drift accumulates against a fixed point, which means it
    does not accumulate.

    **An improvement does not move the anchor either.** Andrew leaned deliberate
    both ways for symmetry and there is a second reason: `check_differential`
    already treats a large unexplained improvement as evidence that the two
    sides were not given the same problem, and auto-adopting an improvement
    would auto-adopt exactly that class of event. An improvement is reported
    and the anchor stays where it is until a person moves it.

    **A new arm with no anchor WARNS rather than passing.** Silently passing is
    how a bad number becomes the anchor: the first run of a new arm would
    establish whatever it happened to produce as correct. It is not a FAIL,
    because a legitimately new arm would then fail every run until somebody
    blessed it, and a harness that refuses to run a new arm is a harness people
    route around. The precedent is the missing-cpu-twin WARN in
    `check_device_agreement`: a check that cannot run is dangerous exactly
    because it is silent, so it says so.

    **Only the PRIMARY metric gates.** The full metric set is RECORDED in the
    anchor file, so a later decision to gate a secondary costs no re-run. It is
    not gated today because a secondary is not the quantity any arm was tuned
    for, and making an anchor refresh a negotiation over five numbers is how a
    refresh stops happening. `check_differential` does gate its secondaries
    against LightGBM, so this is a deliberate difference from the file's other
    accuracy check rather than an oversight.
    """
    rule = dict(DEFAULT_ACCURACY_ANCHOR)
    rule.update((config.get("defaults") or {}).get("accuracy_anchor") or {})
    limit = float(rule["max_worse_relative"])
    implausible = float(rule["implausible_better_relative"])
    judged = tuple(rule["engines_judged"])
    gating = bool(rule.get("gating", True))
    if anchors is None:
        anchors = load_accuracy_anchors()

    cells = {}
    for record in ok:
        # By ENGINE: `engines_judged` names engines. See check_accuracy_peer.
        if record.get("engine") not in judged:
            continue
        cells.setdefault(_anchor_key(record), record)

    uncovered = []
    for key, record in sorted(cells.items()):
        metric = record.get("primary_metric")
        mine = (record.get("quality") or {}).get(metric)
        scope = key
        if _tree_count(record) is None:
            verdict.add(
                SKIP, "accuracy_anchor", scope,
                "abstains: this record does not say how many boosting rounds "
                "it grew, so it cannot be matched to an anchor",
            )
            continue
        if mine is None or np.isnan(mine):
            verdict.add(
                SKIP, "accuracy_anchor", scope,
                f"abstains: the primary metric {metric} is missing or nan",
            )
            continue

        anchor = anchors.get(key)
        if not anchor:
            uncovered.append(key)
            verdict.add(
                WARN, "accuracy_anchor", scope,
                f"NO ANCHOR RECORDED for this arm. Measured {metric} "
                f"{mine:.6g}. Nothing here can tell whether that is a "
                "regression, because there is no recorded value to compare it "
                "against, so this arm is UNCOVERED by the accuracy gate. "
                "Adopting this number as the anchor is a deliberate act and "
                "this check will not do it",
                metric=metric, value=float(mine), anchored=False,
            )
            continue
        # STALENESS IS TESTED BEFORE THE VALUE IS USED, and never after it.
        # An anchor that no longer describes what we ship must not be able to
        # produce a PASS or a FAIL, because both of those are statements about
        # this arm against its own reference and a stale anchor is not that
        # arm's reference any more. See the STALE ANCHOR block above for why the
        # remedy is a run rather than an edit.
        stale = anchor_staleness(anchor)
        if stale is not None:
            uncovered.append(key)
            if stale.get("unknown"):
                verdict.add(
                    WARN, "accuracy_anchor", scope,
                    f"ANCHOR CURRENCY UNKNOWN, so it does not gate. Measured "
                    f"{metric} {mine:.6g} against a recorded "
                    f"{anchor.get('value')}. {stale['why']}",
                    metric=metric, value=float(mine), anchored=False,
                    stale=True, stale_reason="unknown",
                )
            else:
                verdict.add(
                    WARN, "accuracy_anchor", scope,
                    f"STALE ANCHOR, so it does not gate. Measured {metric} "
                    f"{mine:.6g} against a recorded {anchor.get('value')}, "
                    f"which is NOT compared. {stale['why']}. "
                    f"Cleared by: {stale.get('cleared_by', 'a fresh run')}",
                    metric=metric, value=float(mine), anchored=False,
                    stale=True,
                    stale_parameter=stale.get("parameter"),
                    stale_reason="parameter_default_moved",
                )
            continue
        if anchor.get("primary_metric") != metric:
            verdict.add(
                WARN, "accuracy_anchor", scope,
                f"the anchor records primary metric "
                f"{anchor.get('primary_metric')} where this run reports "
                f"{metric}; not compared, because those are two different "
                "quantities and comparing them would produce a number that "
                "looks like a verdict",
                anchored=False,
            )
            continue

        theirs = anchor.get("value")
        if theirs is None:
            verdict.add(
                WARN, "accuracy_anchor", scope,
                "the anchor entry carries no value", anchored=False,
            )
            continue

        worse = _worse_by(metric, mine, float(theirs), "relative")
        here = _accuracy_environment(record)
        there = anchor.get("environment") or {}
        same_machine = all(
            there.get(field) in (None, here.get(field)) for field in here
        )
        provenance = anchor.get("recorded_from") or {}
        where = (
            f"anchor {theirs:.6g}, recorded "
            f"{provenance.get('recorded_at') or 'at an unrecorded date'} from "
            f"run {provenance.get('run_id') or 'unrecorded'} at commit "
            f"{(provenance.get('git_commit') or 'unrecorded')[:12]}"
        )
        detail = (
            f"{metric} {mine:.6g} against our own {where}: "
            f"{'worse' if worse > 0 else 'better'} by "
            f"{abs(worse) * 100:.4f} percent (tolerance "
            f"{limit * 100:.4f} percent)"
        )
        if not same_machine:
            detail += (
                f". DIFFERENT MACHINE: the anchor was recorded on "
                f"{there.get('cpu_model')} ({there.get('arch')}) and this run "
                f"is on {here.get('cpu_model')} ({here.get('arch')})"
            )

        common = dict(
            metric=metric, value=float(mine), anchor_value=float(theirs),
            worse_relative=float(worse), tolerance=float(limit),
            anchored=True, same_machine=bool(same_machine),
        )
        if worse > limit:
            # A regression on a DIFFERENT machine is a warning and not a
            # failure. "We got worse" and "we got worse on hardware the anchor
            # was never taken on" are different claims, and only the first one
            # should stop a run. The line says which one it is either way.
            status = FAIL if (gating and same_machine) else WARN
            verdict.add(
                status, "accuracy_anchor", scope,
                "ACCURACY REGRESSION against our own recorded anchor. "
                + detail
                + ". No peer is involved in this verdict. The fix is either to "
                "recover the accuracy or to refresh the anchor deliberately, "
                "which puts the old and the new number in one diff",
                **common,
            )
        elif -worse > implausible:
            verdict.add(
                WARN, "accuracy_anchor", scope,
                "IMPLAUSIBLY BETTER than our own recorded anchor. " + detail
                + ". Better than this is not a win, it is a sign the arm was "
                "not given the same problem the anchor was taken on: a changed "
                "generator, a changed split, a changed metric. Adopting this "
                "as the next anchor would bake that in. Establish what changed "
                "first",
                **common,
            )
        else:
            verdict.add(PASS, "accuracy_anchor", scope, detail, **common)

    if uncovered:
        verdict.add(
            NOTE, "accuracy_anchor", "run",
            f"{len(uncovered)} arm-cell(s) have no USABLE accuracy anchor "
            "(missing, stale, or of unknown currency), "
            "so the accuracy gate covers nothing for them. To adopt anchors "
            "from a run you trust: `python bench/real_data/verify.py <run> "
            "--propose-anchors <path>`, read the file it writes, then move the "
            "entries into bench/real_data/accuracy_anchors.json in a commit "
            "that says what they were adopted for. Nothing in this harness "
            "writes that file for you",
        )


def propose_anchors(ok, config, path, results_path):
    """Write a CANDIDATE anchor file for a person to read and move into place.

    This is the deliberate act, split in two so that the deliberate half is
    done by a human. This function produces a file; nothing produces
    `accuracy_anchors.json`. That separation is the whole reason a drifting
    anchor cannot happen by accident: there is no code path from a run to the
    live anchors.

    That last sentence is enforced at the bottom of this function and was NOT
    true until 2026-08-17, when it was only a description of intent while
    `path` accepted the live anchors file like any other. Read the refusal
    there before trusting this paragraph, and do not remove it: the claim and
    the guard are the same fact stated twice, and deleting one leaves the other
    lying.

    The candidate carries the full metric set and the provenance, because an
    anchor a reader cannot trace back to a run, a commit and a machine is a
    number with no argument behind it, and this file's entire purpose is to be
    the fixed point everything else is measured from.
    """
    rule = dict(DEFAULT_ACCURACY_ANCHOR)
    rule.update((config.get("defaults") or {}).get("accuracy_anchor") or {})
    judged = tuple(rule["engines_judged"])

    cells = {}
    for record in ok:
        # By ENGINE: `engines_judged` names engines. See check_accuracy_peer.
        if record.get("engine") not in judged:
            continue
        if _tree_count(record) is None:
            continue
        cells.setdefault(_anchor_key(record), record)

    anchors = {}
    for key, record in sorted(cells.items()):
        metric = record.get("primary_metric")
        value = (record.get("quality") or {}).get(metric)
        if value is None or np.isnan(value):
            continue
        data = record.get("data") or {}
        environment = record.get("environment") or {}
        anchors[key] = {
            "scenario": record.get("scenario"),
            "tier": record.get("tier"),
            "data_kind": data.get("data_kind"),
            "dataset": data.get("dataset"),
            "arm": _arm_of(record),
            "device": record.get("device_used") or record.get("device_requested"),
            "trees": _tree_count(record),
            "primary_metric": metric,
            "value": float(value),
            "quality": {
                name: (None if v is None or (isinstance(v, float) and np.isnan(v))
                       else float(v))
                for name, v in (record.get("quality") or {}).items()
            },
            "environment": _accuracy_environment(record),
            # THE BLOCK THAT MAKES STALENESS MACHINE-CHECKABLE, added
            # 2026-08-17. Without it an anchor can stop describing what we ship
            # with no number in it changing, and nothing can tell. With it,
            # `anchor_staleness` compares the parameters this arm left UNSET
            # against what they resolve to now, so **an anchor proposed from a
            # run taken after a default moves is current by construction** and
            # the run is what clears the staleness. An anchor without this block
            # is judged UNKNOWN and does not gate.
            "configuration": anchor_configuration(record),
            "recorded_from": {
                "run_id": os.path.basename(os.path.normpath(results_path)),
                "git_commit": (environment.get("git") or {}).get("commit"),
                "recorded_at": None,
                "recorded_by": None,
            },
            "why": None,
        }

    payload = {
        "_about": [
            "CANDIDATE accuracy anchors. This file is a proposal and nothing "
            "reads it.",
            "",
            "Written by `verify.py --propose-anchors`. To adopt: read every "
            "entry, fill in `recorded_at`, `recorded_by` and `why`, and move "
            "the entries into bench/real_data/accuracy_anchors.json in a "
            "commit that says what they were adopted for.",
            "",
            "An anchor adopted without reading it is worse than no anchor, "
            "because a gate measured from a number nobody looked at reports "
            "green about nothing.",
            "",
            "KEEP THE `configuration` BLOCK. It records what the arm ran at and "
            "which parameters the arm left UNSET, and verify.anchor_staleness "
            "reads both to decide whether the anchor still describes the model "
            "we ship. An entry adopted without it is judged UNKNOWN and does "
            "not gate. A stale anchor is never recomputed by arithmetic: it is "
            "marked, it stops gating, and a fresh run replaces it.",
        ],
        "version": 1,
        "proposed_from": os.path.abspath(results_path),
        "anchors": anchors,
    }
    # The guard that makes this function's docstring true. Added 2026-08-17
    # after an audit found the claim "there is no code path from a run to the
    # live anchors" was FALSE as written: `path` is whatever the caller typed,
    # so `--propose-anchors bench/real_data/accuracy_anchors.json` wrote a
    # freshly measured run straight over the fixed point it is supposed to be
    # judged against, installing the exact ratchet that file exists to prevent
    # and destroying the adopted values in the same stroke. Nothing warned.
    #
    # Refusing by resolved path rather than by string, because `./` prefixes,
    # relative invocations and symlinked checkouts all spell the same file
    # differently and a comparison that any of them defeats is decoration.
    if os.path.realpath(path) == os.path.realpath(ACCURACY_ANCHORS_PATH):
        raise SystemExit(
            f"REFUSED: {path} resolves to the live anchors file, "
            f"{ACCURACY_ANCHORS_PATH}. `--propose-anchors` writes a CANDIDATE "
            "for a person to read; it is not the adoption step, and letting it "
            "write here would overwrite the recorded fixed point with the run "
            "being judged. Adoption is a human edit in a commit that says what "
            "the anchors were adopted for. Write the candidate somewhere else, "
            "read every entry, then move the entries across by hand."
        )
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return len(anchors)


def check_device_agreement(ok, config, verdict, run_dir):
    rule = config["defaults"]["device_agreement"]
    cells = {}
    for record in ok:
        if record["engine"] not in SUBJECT_ENGINES:
            continue
        device = record.get("device_used") or record.get("device_requested")
        # KEYED BY ARM AS WELL AS BY CELL, and it has to be. This loop covered
        # one engine when it was written, so (scenario, threads) was a unique
        # key. With three subject arms it is not: two arms' records would
        # overwrite each other in the inner dict and this check would compare
        # one arm's GPU predictions against a DIFFERENT arm's CPU predictions,
        # which is a guaranteed false failure and, worse, a meaningless
        # comparison reported as a device-agreement verdict.
        #
        # THE KEY WAS `record["engine"]` UNTIL 2026-08-17 AND THAT WAS STILL
        # TOO COARSE, silently, on exactly one kind of run. An `--arms` run
        # takes its cells from a module rather than from the engine cross
        # product, and `frontier.arms()` alone produces 40 runnable
        # mojotrees/dense_regression/cpu cells and 16 gpu ones. Under an engine
        # key all 56 collapsed into one entry per device, so the check emitted
        # a SINGLE verdict comparing whichever cpu arm and whichever gpu arm
        # happened to be written last, at different tree counts and different
        # `max_bin`. That is the precise failure the paragraph above says the
        # engine key was added to prevent, one level down, and it reported a
        # green verdict rather than an error. `_arm_of` is the identity the
        # rest of this file already groups by, so it is what this key uses.
        cells.setdefault(
            (record["scenario"], record["threads"], _arm_of(record)), {}
        )[device] = record
    for (scenario, threads, arm), by_device in sorted(cells.items()):
        cpu, gpu = by_device.get("cpu"), by_device.get("gpu")
        scope = f"{scenario}/{arm}/t{threads}"
        if not cpu or not gpu:
            # A missing cpu twin used to leave NO LINE AT ALL, which is the
            # one shape of this check that a reader cannot tell from a pass:
            # the verdict simply had nothing to say about a gpu row whose
            # predictions nobody compared. Said out loud from 2026-08-17,
            # when the cpu cell became the ORACLE cell and the reason for
            # scheduling it became a written rule rather than a habit.
            #
            # WARN and not FAIL, on purpose. A gpu-only run is a legal thing
            # to ask for, and refusing one would be this check deciding the
            # matrix. The point is that the loss is stated.
            if gpu and not cpu:
                verdict.add(
                    WARN, "device_agreement", scope,
                    "an accelerator row with no cpu twin in this run, so this "
                    "check did not run for it and no row-level comparison "
                    "exists. The cpu twin is the ORACLE cell and it is the "
                    "only thing this check compares against; on 2026-08-17 "
                    "that comparison caught a noise hash domain divergence "
                    "between the cpu and gpu symmetric growers in the shipped "
                    "defaults. Schedule the cpu cell to get the verdict back; "
                    "--oracle-repeats 1 is enough for it",
                )
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


#: The checks whose lines COVER a subject cell for `check_coverage`: every
#: check in this file that runs per cell and writes the ARM into its scope as
#: a delimited component. `check_differential` is deliberately absent, its
#: scope is `scenario/tN[/metric]` with no arm and it is the headline pairing
#: rather than a per-cell gate. `check_completeness` is absent because its
#: scope names the ENGINE, which on a plain run equals the arm and would cover
#: by coincidence. `check_accuracy_peer` is absent because it gates nothing:
#: a scoreboard naming an arm says nothing about whether a gate saw it, and
#: on 2026-08-17 it was exactly the check that named nothing.
COVERING_CHECKS = (
    "determinism", "backend_proof", "baseline", "accuracy_anchor",
    "device_agreement",
)


def _scope_components(scope):
    """A scope split into its delimited components. Every per-cell scope in
    this file is `/`-joined except `check_accuracy_anchor`, whose scope is
    `_anchor_key`, which is `|`-joined; both delimiters split. Matching on
    components rather than substrings is the point: an arm id `A` must not
    be covered by a line about `A.b`, and frontier arm ids are dot-joined
    prefixes of one another by construction."""
    return set(scope.replace("|", "/").split("/"))


def check_coverage(ok, verdict):
    """Every subject cell was NAMED by at least one per-cell gate.

    Runs LAST, over the verdict itself. On 2026-08-17 three gates in this file
    were green because they compared nothing or the wrong things, and the
    worst of the three, `check_accuracy_peer` testing engine names against arm
    ids on an `--arms` run, emitted ZERO lines, which reads exactly like a
    clean run. A gate that emits nothing is indistinguishable from a passing
    gate, and no per-check fix closes that class: the next check to be keyed
    on the wrong field will be silent in the same way. This is the structural
    answer. For every ok record whose ENGINE is a subject (roles are by
    engine, `SUBJECT_ENGINES`), take its ARM (`_arm_of`, cells are by arm) and
    require that some line already written by a check in `COVERING_CHECKS`
    carries that scenario and that arm as delimited scope components.

    It NEVER fails. A run shape someone asked for is legal, and a cell no gate
    could speak to (a variant arm with no determinism rule, no baseline rule
    for its scenario, no anchor and no twin) is a fact to state, not a
    defect to exit on. Every uncovered cell is one WARN so that the silence
    is written down where the reader is looking; a line per COVERED cell would
    be one more green line per cell on every run for no information, so the
    covered ones are summed into a single line instead, PASS when every
    subject cell was named and WARN otherwise. No subject cells at all writes
    nothing: a competitor-only run has nothing here to be silent about.
    """
    named = set()
    for check in verdict.checks:
        if check["check"] in COVERING_CHECKS:
            named.add(frozenset(_scope_components(check["scope"])))
    cells = {}
    for record in ok:
        if record.get("engine") not in SUBJECT_ENGINES:
            continue
        key = (
            record["scenario"], _arm_of(record), _device_of(record),
            record["threads"],
        )
        cells.setdefault(key, record)
    if not cells:
        return
    uncovered = []
    for key in sorted(cells):
        scenario, arm, device, threads = key
        if any(scenario in parts and arm in parts for parts in named):
            continue
        uncovered.append(key)
        verdict.add(
            WARN, "coverage", f"{scenario}/{arm}/{device}/t{threads}",
            "NO per-cell check named this subject cell. One of "
            + ", ".join(COVERING_CHECKS)
            + " was expected to write a line whose scope carries this "
            "scenario and this arm, and none did. A gate that emits nothing "
            "is indistinguishable from a passing gate, so this line exists "
            "to say that the silence is silence and not a pass. Either the "
            "run shape gives the gates nothing to say for this cell (say so "
            "in the run notes) or a check is keying on the wrong field again",
        )
    covered = len(cells) - len(uncovered)
    verdict.add(
        PASS if not uncovered else WARN, "coverage", "run",
        f"{covered} of {len(cells)} subject cells named by a per-cell check "
        f"({', '.join(COVERING_CHECKS)})",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", help="a records.json file or a run directory")
    parser.add_argument("--json", help="write the verdict here as well")
    parser.add_argument(
        "--allow-unpinned", action="store_true",
        help="downgrade the pinning failure to a warning; the verdict still "
             "records that the data was unverified",
    )
    parser.add_argument(
        "--propose-anchors", metavar="PATH",
        help="write a CANDIDATE accuracy anchor file here and exit as normal. "
             "It is a proposal: nothing reads it, and adopting it means "
             "reading every entry and moving it into accuracy_anchors.json in "
             "a commit that says what it was adopted for. This flag never "
             "writes the live anchor file",
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
    check_accuracy_anchor(ok, config, verdict)
    check_accuracy_peer(ok, config, verdict)
    check_baseline(ok, config, verdict)
    check_device_agreement(ok, config, verdict, run_dir)
    # LAST, over the verdict the checks above wrote. See check_coverage.
    check_coverage(ok, verdict)

    if args.propose_anchors:
        written = propose_anchors(ok, config, args.propose_anchors, args.results)
        print(
            f"Wrote {written} CANDIDATE anchor(s) to {args.propose_anchors}. "
            "Nothing reads that file. Read every entry, fill in recorded_at, "
            "recorded_by and why, then move them into "
            "bench/real_data/accuracy_anchors.json in a commit that says what "
            "they were adopted for.\n"
        )

    counts = verdict.counts()
    order = {status: index for index, status in enumerate(STATUSES)}
    for check in sorted(verdict.checks, key=lambda c: (order[c["status"]], c["check"], c["scope"])):
        print(f"{check['status'].upper():<5} {check['check']:<18} {check['scope']:<44} {check['detail']}")

    print(
        f"\n{counts[PASS]} pass, {counts[FAIL]} fail, {counts[WARN]} warn, "
        f"{counts[NOTE]} note, {counts[SKIP]} skip"
    )
    print(
        "A `note` is a measured fact rather than a judgment about this run, "
        "and it never affects the exit code. Every accuracy_peer line is one."
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
