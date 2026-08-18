#!/usr/bin/env python3
"""Decide what to run after a change, and print it. Never run it.

    python3 tools/validation_plan.py                       # the working tree
    python3 tools/validation_plan.py --paths src/mojotrees/split.mojo
    python3 tools/validation_plan.py --handoff <path>
    python3 tools/validation_plan.py --subsystem gpu-training --explain
    python3 tools/validation_plan.py --allow differential --budget-seconds 900
    python3 tools/validation_plan.py --format sh --out /tmp/plan.sh
    python3 tools/validation_plan.py --self-check
    python3 tools/validation_plan.py --coverage

This repository has 47 Mojo suites, 18 pytest files, two full-suite pixi
tasks, seven benchmark harnesses, a wheel pipeline, and a hardware protocol.
`pixi run test` compiles and runs all 47 suites. The habit that costs the most
machine time is reaching for it after a two-line change, and the habit that
costs the most trust is reaching for nothing.

So this is a planner. It maps the files that changed onto the smallest set of
commands that would notice, orders them cheapest first, charges them against a
machine budget, and prints them with the reason each one was selected and what
a pass would and would not prove. Then it stops. There is no flag that makes it
execute a validation command, and adding one would defeat the purpose of the
file: the decision to spend twenty minutes of a laptop belongs to the person
holding the laptop.

The data lives in `validation/manifests/`:

    tiers.toml        nine classes of evidence, their budgets, their gates
    jobs.toml         every command this repository can issue, one entry each
    subsystems.toml   changed paths to the jobs that would notice, and the
                      paths no job would notice at all
    handoffs.toml     each handoff to the jobs that would test its claims

Nothing in those files is a measurement. `budget_seconds` is a scheduling
guess, and `provenance` records, per command, whether a CI job runs it, whether
a document records it, or whether this manifest composed it from a documented
pattern and nobody has ever run that composition.

What keeps a plan from turning into the full suite by accident
--------------------------------------------------------------

1. Five of the nine tiers are opt-in. Without `--allow <tier>` their jobs are
   printed under HELD, with the flag that would schedule them, and they never
   consume budget.
2. A plan carries at most one broad suite (`max_broad_jobs` in tiers.toml), and
   never a broad suite beside a hardware job.
3. Everything is charged against `--budget-seconds`. What does not fit is
   printed under OVER BUDGET with its cost, so dropping it is a decision.
4. The emitted shell script refuses to start unless `MOJOTREES_VALIDATION_OPT_IN=1`
   is set in the environment, so a stray `sh plan.sh` does nothing.
5. Heavy jobs are wrapped in `tools/with_build_lock.sh`, the lock this
   repository already uses to keep two sessions from compiling at once.

The one thing this file executes
--------------------------------

`git --no-optional-locks status --porcelain`, and only when no explicit change
set was given. It is read-only, it is the only subprocess call in this file,
and `--paths` skips it. Nothing else here runs, builds, imports the extension,
touches the network, or writes inside the repository.

Standard library only, Python 3.11 or later for tomllib. Exit status is 0 for a
plan, 0 for a clean `--self-check`, and 1 when `--self-check` finds a problem.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFESTS = ROOT / "validation" / "manifests"

# Programs a `pixi run` line may name instead of a pixi task. Anything else
# after `pixi run` has to be a task defined in pixi.toml, which --self-check
# enforces.
KNOWN_PROGRAMS = {"mojo", "python", "python3", "pytest", "sh", "bash", "env"}

# Preconditions a job may declare. The planner never checks whether one holds,
# because checking would mean running something. It prints them.
KNOWN_NEEDS = {
    "toolchain",
    "extension",
    "wheel",
    "gpu",
    "numpy",
    "pytest",
    "network",
    "two-hosts",
    "sudo",
    "outside-pixi",
}

PROVENANCE = {"ci", "documented", "unverified"}

# Environment variables a job may set that are not part of this project's
# MOJOTREES_* contract.
ALLOWED_FOREIGN_ENV: set[str] = set()


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------


class Manifests:
    """The four TOML files, loaded once and cross-indexed."""

    def __init__(self, directory=MANIFESTS):
        self.directory = Path(directory)
        self.tiers_doc = self._load("tiers")
        self.jobs_doc = self._load("jobs")
        self.subsystems_doc = self._load("subsystems")
        self.handoffs_doc = self._load("handoffs")

        self.tiers = {t["id"]: t for t in self.tiers_doc.get("tier", [])}
        self.tier_order = {
            t["id"]: t.get("order", 99) for t in self.tiers.values()
        }
        self.jobs = {}
        for job in self.jobs_doc.get("job", []):
            self.jobs.setdefault(job["id"], job)
        self.defaults = self.jobs_doc.get("defaults", {})
        self.budget = self.tiers_doc.get("budget", {})
        self.locks = self.tiers_doc.get("locks", {})
        self.subsystems = self.subsystems_doc.get("subsystem", [])
        self.gaps = self.subsystems_doc.get("gap", [])
        self.handoffs = self.handoffs_doc.get("handoff", [])
        self.lanes = self.handoffs_doc.get("lane", [])
        self.archive = self.handoffs_doc.get("archive", {})

    def _load(self, name):
        path = self.directory / f"{name}.toml"
        with path.open("rb") as handle:
            return tomllib.load(handle)

    # -- per-job resolution, manifest value then tier default then global ----

    def tier_of(self, job_id):
        return self.tiers.get(self.jobs[job_id]["tier"], {})

    def cost(self, job_id):
        job = self.jobs[job_id]
        if "budget_seconds" in job:
            return int(job["budget_seconds"])
        per_tier = self.defaults.get("budget_seconds", {})
        return int(per_tier.get(job["tier"], 60))

    def timeout(self, job_id):
        job = self.jobs[job_id]
        return int(
            job.get(
                "timeout_seconds",
                self.defaults.get(
                    "timeout_seconds",
                    self.budget.get("default_timeout_seconds", 900),
                ),
            )
        )

    def threads(self, job_id):
        job = self.jobs[job_id]
        return int(
            job.get(
                "threads",
                self.defaults.get(
                    "threads", self.budget.get("default_threads", 4)
                ),
            )
        )

    def exclusive(self, job_id):
        job = self.jobs[job_id]
        if "exclusive" in job:
            return job["exclusive"]
        return self.tier_of(job_id).get("exclusive", "")

    def retired_path(self, path):
        """True when this path is a known casualty of a deliberate deletion.

        `[archive]` in handoffs.toml names a prefix and the commit that removed
        it. A path under that prefix which is absent is retired; one which is
        present is checked normally, so restoring a file needs no edit here.
        """
        if not self.archive.get("retired"):
            return False
        prefix = normalize(self.archive.get("prefix", ""))
        if not prefix:
            return False
        return normalize(path).startswith(prefix) and not exists(path)

    def lock_env(self):
        """The variable the wrapper will read a lock path from, once it does."""
        return self.locks.get("lock_env", "")

    def lock_file_for(self, klass):
        """The lock file an exclusion class should take.

        Falls back to the single shared lock, which is what the wrapper uses
        today regardless of what this returns. See tiers.toml [locks].
        """
        shared = self.locks.get("lock_file", "/tmp/mojotrees-build.lock")
        if not klass:
            return shared
        return self.locks.get("class_lock_files", {}).get(klass, shared)

    def order_key(self, job_id):
        job = self.jobs.get(job_id)
        if job is None:
            return (99, job_id)
        return (self.tier_order.get(job["tier"], 99), job_id)


# ---------------------------------------------------------------------------
# The change set
# ---------------------------------------------------------------------------


def normalize(path):
    text = str(path).strip().replace("\\", "/")
    if text.startswith("./"):
        text = text[2:]
    return text


def git_changed_paths(base=None):
    """Paths git reports as changed. The only subprocess call in this file.

    Read-only: `--no-optional-locks` keeps `git status` from refreshing the
    index on disk. Returns (paths, description). An unavailable or unhappy git
    is not an error here, it is an empty change set and a sentence saying so.
    """
    if base:
        argv = ["git", "--no-optional-locks", "diff", "--name-only", base]
    else:
        argv = [
            "git",
            "--no-optional-locks",
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--no-renames",
        ]
    try:
        done = subprocess.run(
            argv,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [], f"git unavailable ({type(exc).__name__}), empty change set"
    if done.returncode != 0:
        detail = done.stderr.strip().splitlines()
        first = detail[0] if detail else f"exit {done.returncode}"
        return [], f"git failed ({first}), empty change set"

    paths = []
    for line in done.stdout.splitlines():
        if not line.strip():
            continue
        if base:
            paths.append(normalize(line))
            continue
        # porcelain v1: two status characters, a space, then the path
        paths.append(normalize(line[3:]))
    where = f"git diff against {base}" if base else "git working tree"
    return sorted(set(paths)), where


def path_matches(pattern, path):
    """Literal, fnmatch, or directory-prefix match."""
    pattern = normalize(pattern)
    if pattern.endswith("/"):
        return path == pattern.rstrip("/") or path.startswith(pattern)
    if any(ch in pattern for ch in "*?["):
        return fnmatch.fnmatch(path, pattern)
    return path == pattern


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


class Selection:
    """Job ids with the reasons they were selected, in insertion order."""

    def __init__(self):
        self.reasons = {}

    def add(self, job_id, reason):
        self.reasons.setdefault(job_id, [])
        if reason not in self.reasons[job_id]:
            self.reasons[job_id].append(reason)

    def __contains__(self, job_id):
        return job_id in self.reasons

    def __iter__(self):
        return iter(self.reasons)

    def __len__(self):
        return len(self.reasons)


def select_for_paths(man, paths, escalate=False):
    """Jobs for a change set, plus the gap warnings the change set earns."""
    selection = Selection()
    gaps = []
    unmatched = []

    for path in paths:
        matched = False

        for subsystem in man.subsystems:
            if not any(path_matches(p, path) for p in subsystem["paths"]):
                continue
            matched = True
            reason = f"subsystem {subsystem['id']} ({path})"
            for job_id in subsystem.get("jobs", []):
                selection.add(job_id, reason)
            if escalate:
                for job_id in subsystem.get("escalate", []):
                    selection.add(job_id, f"escalation of {subsystem['id']}")

        # A changed suite runs itself. No subsystem entry is needed or wanted
        # for that: the job already names the file in requires_files.
        for job_id, job in man.jobs.items():
            if path in [normalize(p) for p in job.get("requires_files", [])]:
                if path.startswith(("tests/", "python/tests/")):
                    matched = True
                    selection.add(job_id, f"the suite itself changed ({path})")

        for gap in man.gaps:
            if any(path_matches(p, path) for p in gap["paths"]):
                matched = True
                gaps.append((path, gap))
                for job_id in gap.get("fallback", []):
                    selection.add(job_id, f"fallback for an untested path ({path})")

        if not matched:
            unmatched.append(path)

    return selection, gaps, unmatched


def select_for_handoff(man, name, escalate=False):
    """Jobs for a handoff file or a lane id."""
    selection = Selection()
    found = []
    target = normalize(name)
    for entry in man.handoffs:
        paths = [normalize(p) for p in entry["paths"]]
        if target in paths or any(Path(p).name == target for p in paths):
            found.append(target)
            for job_id in entry.get("jobs", []):
                selection.add(job_id, f"handoff {target}")
            if escalate:
                for job_id in entry.get("escalate", []):
                    selection.add(job_id, f"escalation of handoff {target}")
    for lane in man.lanes:
        if target in (lane["id"], normalize(lane["path"]), Path(lane["path"]).name):
            found.append(lane["id"])
            for job_id in lane.get("jobs", []):
                selection.add(job_id, f"lane {lane['id']}")
            if escalate:
                for job_id in lane.get("escalate", []):
                    selection.add(job_id, f"escalation of lane {lane['id']}")
    return selection, found


def add_dependencies(man, selection):
    """Pull in depends_on, transitively, marking why."""
    pending = list(selection)
    while pending:
        job_id = pending.pop()
        for dependency in man.jobs.get(job_id, {}).get("depends_on", []):
            if dependency not in selection:
                selection.add(dependency, f"{job_id} needs it first")
                pending.append(dependency)
    return selection


def order(man, selection):
    """Cheapest tier first, dependencies before the jobs that need them."""
    ordered = []
    placed = set()

    def place(job_id, stack):
        if job_id in placed or job_id not in selection:
            return
        if job_id in stack:  # a cycle; --self-check reports it properly
            return
        stack.add(job_id)
        for dependency in man.jobs.get(job_id, {}).get("depends_on", []):
            place(dependency, stack)
        stack.discard(job_id)
        placed.add(job_id)
        ordered.append(job_id)

    for job_id in sorted(selection, key=man.order_key):
        place(job_id, set())
    return ordered


# ---------------------------------------------------------------------------
# The plan: gates, budget, and mutual exclusion
# ---------------------------------------------------------------------------


class Plan:
    def __init__(self, man, args):
        self.man = man
        self.args = args
        self.scheduled = []
        self.held = []  # (job_id, why)
        self.blocked = []  # (job_id, why)
        self.over_budget = []  # (job_id, cost)
        self.spent = 0
        self.gaps = []
        self.unmatched = []
        self.change_set = []
        self.source = ""

    # -- gates ----------------------------------------------------------

    def gate(self, job_id):
        """Why this job may not be scheduled, or None."""
        man, args = self.man, self.args
        job = man.jobs[job_id]
        tier = man.tier_of(job_id)

        if job.get("blocked"):
            return None  # handled separately, it is not a gate

        if tier.get("opt_in") and job["tier"] not in args.allow:
            return f"tier is opt-in; add --allow {job['tier']}"

        needs = set(job.get("needs", []))
        if args.no_gpu and "gpu" in needs:
            return "--no-gpu was given and this job needs an accelerator"
        if args.offline and "network" in needs:
            return "--offline was given and this job needs the network"
        if args.no_sudo and "sudo" in needs:
            return "--no-sudo was given and this job would ask for sudo"

        for dependency in job.get("depends_on", []):
            if dependency in dict(self.held):
                return f"depends on {dependency}, which is held"
        return None

    def build(self, selection):
        man, args = self.man, self.args
        ordered = order(man, selection)

        # blocked jobs first: they are information, not work
        runnable = []
        for job_id in ordered:
            job = man.jobs.get(job_id)
            if job is None:
                # A manifest names a job that jobs.toml does not define.
                # --self-check fails on this; a plan should still be usable.
                self.held.append(
                    (job_id, "no such job in jobs.toml; run --self-check")
                )
                continue
            if job.get("blocked"):
                self.blocked.append(
                    (job_id, (job.get("blocked_reason") or "").strip())
                )
            else:
                runnable.append(job_id)

        # tier gates and precondition flags
        gated = []
        for job_id in runnable:
            why = self.gate(job_id)
            if why:
                self.held.append((job_id, why))
            else:
                gated.append(job_id)

        # a scheduled job whose dependency got held cannot run either
        held_ids = {job_id for job_id, _ in self.held}
        survivors = []
        for job_id in gated:
            blocked_by = [
                d for d in man.jobs[job_id].get("depends_on", []) if d in held_ids
            ]
            if blocked_by:
                self.held.append(
                    (job_id, f"depends on {', '.join(blocked_by)}, which is held")
                )
                held_ids.add(job_id)
            else:
                survivors.append(job_id)

        survivors = self._limit_broad(survivors)
        survivors = self._one_class_when_measuring(survivors)
        self._charge(survivors)

    def _limit_broad(self, candidates):
        """At most one broad suite, and never one beside a hardware job."""
        man = self.man
        limit = int(man.budget.get("max_broad_jobs", 1))
        hardware = [j for j in candidates if man.jobs[j]["tier"] == "hardware"]
        kept, broad_kept = [], 0
        for job_id in candidates:
            if not man.jobs[job_id].get("broad"):
                kept.append(job_id)
                continue
            if hardware:
                self.held.append(
                    (
                        job_id,
                        "a hardware job is in this plan; a broad suite running "
                        "beside a measurement invalidates the measurement",
                    )
                )
                continue
            if broad_kept >= limit:
                self.held.append(
                    (
                        job_id,
                        f"a plan carries at most {limit} broad suite "
                        "(max_broad_jobs in tiers.toml); schedule this one on "
                        "its own",
                    )
                )
                continue
            broad_kept += 1
            kept.append(job_id)
        return kept

    def _one_class_when_measuring(self, candidates):
        """With --measure, a plan may hold only one exclusion class."""
        man = self.man
        if not self.args.measure:
            return candidates
        chosen = None
        kept = []
        for job_id in candidates:
            klass = man.exclusive(job_id)
            if not klass:
                kept.append(job_id)
                continue
            if chosen is None:
                chosen = klass
            if klass == chosen:
                kept.append(job_id)
            else:
                self.held.append(
                    (
                        job_id,
                        f"--measure holds one exclusion class per plan and this "
                        f"plan is a {chosen} plan; run the {klass} jobs "
                        "separately",
                    )
                )
        return kept

    def _charge(self, candidates):
        man, args = self.man, self.args
        max_jobs = args.max_jobs
        for job_id in candidates:
            cost = man.cost(job_id)
            if len(self.scheduled) >= max_jobs:
                self.over_budget.append((job_id, cost))
                continue
            if self.spent + cost > args.budget_seconds:
                self.over_budget.append((job_id, cost))
                continue
            self.scheduled.append(job_id)
            self.spent += cost

        # a scheduled job whose dependency fell off the budget cannot run
        dropped = {job_id for job_id, _ in self.over_budget}
        if dropped:
            keep = []
            for job_id in self.scheduled:
                missing = [
                    d
                    for d in man.jobs[job_id].get("depends_on", [])
                    if d in dropped
                ]
                if missing:
                    self.over_budget.append((job_id, man.cost(job_id)))
                    self.spent -= man.cost(job_id)
                    dropped.add(job_id)
                else:
                    keep.append(job_id)
            self.scheduled = keep


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render_text(plan, selection, explain=False):
    man = plan.man
    out = []
    add = out.append

    add("mojotrees focused validation plan")
    add(f"  change set     {len(plan.change_set)} path(s) from {plan.source}")
    add(
        f"  budget         {plan.spent}s of {plan.args.budget_seconds}s, "
        f"{len(plan.scheduled)} of at most {plan.args.max_jobs} jobs"
    )
    allowed = ", ".join(sorted(plan.args.allow)) or "none"
    add(f"  opt-in tiers   {allowed}")
    add("")

    if plan.scheduled:
        add(f"SCHEDULED ({len(plan.scheduled)})")
        for job_id in plan.scheduled:
            job = man.jobs[job_id]
            add("")
            add(
                f"  [{job['tier']}] {job_id}  ~{man.cost(job_id)}s"
                + (f"  ({man.exclusive(job_id)} lock)" if man.exclusive(job_id) else "")
            )
            add(f"      $ {job['command']}")
            for name, value in sorted(job.get("env", {}).items()):
                add(f"        env {name}={value}")
            for reason in selection.reasons.get(job_id, []):
                add(f"      because {reason}")
            if explain:
                add(f"      proves  {job.get('proves', '')}")
                if job.get("needs"):
                    add(f"      needs   {', '.join(job['needs'])}")
                add(f"      source  {job.get('provenance', 'unverified')}")
                if job.get("evidence"):
                    add(f"      file to {job['evidence']}")
    else:
        add("SCHEDULED (0)")
        add("  Nothing matched, or everything is held. Try --explain, or name a")
        add("  subsystem with --subsystem, or widen with --escalate.")
    add("")

    if plan.held:
        add(f"HELD ({len(plan.held)})")
        for job_id, why in plan.held:
            add(f"  [{man.jobs.get(job_id, {}).get('tier', '?')}] {job_id}")
            add(f"      {why}")
        add("")

    if plan.blocked:
        add(f"BLOCKED, no command exists ({len(plan.blocked)})")
        for job_id, why in plan.blocked:
            add(f"  {job_id}")
            for line in (why or "").splitlines():
                if line.strip():
                    add(f"      {line.strip()}")
        add("")

    if plan.over_budget:
        add(f"OVER BUDGET ({len(plan.over_budget)})")
        total = sum(cost for _, cost in plan.over_budget)
        for job_id, cost in plan.over_budget:
            tier = man.jobs.get(job_id, {}).get("tier", "?")
            add(f"  [{tier}] {job_id}  ~{cost}s")
        add(f"      {total}s more would run these; raise --budget-seconds to take them")
        add("")

    if plan.gaps:
        add(f"GAPS, changed paths no suite imports ({len(plan.gaps)})")
        for path, gap in plan.gaps:
            add(f"  {path}")
            add(f"      {gap.get('reason', '')}")
            if gap.get("would_close_it"):
                add(f"      to close it: {gap['would_close_it']}")
        add("")

    if plan.unmatched:
        add(f"UNMAPPED ({len(plan.unmatched)})")
        add("  No subsystem, gap, or job names these paths. Either they need no")
        add("  validation, or validation/manifests/subsystems.toml needs an entry.")
        for path in plan.unmatched:
            add(f"  {path}")
        add("")

    add("NOTHING ABOVE WAS RUN. This tool prints commands and never issues one.")
    add("To collect the evidence: --format sh --out <file>, read the file, then")
    add("run it yourself with MOJOTREES_VALIDATION_OPT_IN=1.")
    return "\n".join(out)


def render_json(plan, selection):
    man = plan.man

    def describe(job_id):
        job = man.jobs[job_id]
        return {
            "id": job_id,
            "tier": job["tier"],
            "title": job.get("title", ""),
            "command": job.get("command", ""),
            "env": job.get("env", {}),
            "budget_seconds": man.cost(job_id),
            "timeout_seconds": man.timeout(job_id),
            "threads": man.threads(job_id),
            "exclusive": man.exclusive(job_id),
            "needs": job.get("needs", []),
            "depends_on": job.get("depends_on", []),
            "provenance": job.get("provenance", "unverified"),
            "proves": job.get("proves", ""),
            "evidence": job.get("evidence", ""),
            "because": selection.reasons.get(job_id, []),
        }

    return json.dumps(
        {
            "executed": False,
            "change_set": plan.change_set,
            "change_source": plan.source,
            "budget_seconds": plan.args.budget_seconds,
            "spent_seconds": plan.spent,
            "allowed_tiers": sorted(plan.args.allow),
            "scheduled": [describe(j) for j in plan.scheduled],
            "held": [{"id": j, "why": w} for j, w in plan.held],
            "blocked": [{"id": j, "why": w} for j, w in plan.blocked],
            "over_budget": [
                {"id": j, "budget_seconds": c} for j, c in plan.over_budget
            ],
            "gaps": [
                {"path": p, "reason": g.get("reason", "")} for p, g in plan.gaps
            ],
            "unmapped": plan.unmatched,
        },
        indent=2,
        sort_keys=False,
    )


SCRIPT_HEADER = """\
#!/bin/sh
# Generated by tools/validation_plan.py. Read it before you run it.
#
# This script was written by a planner that has never executed a validation
# command. Every line below is a command the manifests in validation/manifests
# claim is the right one; several of them carry provenance "unverified", which
# means nobody has run that exact composition.
#
# It refuses to start unless MOJOTREES_VALIDATION_OPT_IN=1 is set, so that a
# stray `sh plan.sh` does nothing:
#
#     MOJOTREES_VALIDATION_OPT_IN=1 sh {out}
#
# Jobs run one at a time. Heavy ones take {lock}, the lock this repository
# already uses to keep two sessions from compiling at once, so a plan running
# here will wait for a build running in another terminal instead of fighting it.
# Each exclusive job also exports its class's lock path, which that wrapper does
# not read yet; until it does, cpu, gpu, and build share one lock file.
#
# Wall clock limits are enforced with timeout(1) when one is on PATH. macOS
# ships none by default; `brew install coreutils` provides gtimeout, which this
# script also looks for. Without either, jobs run unbounded and say so.
#
# Memory is NOT limited. `ulimit -v` does nothing useful on macOS, and a limit
# that silently fails to apply is worse than a printed expectation.

set -u

if [ "${{MOJOTREES_VALIDATION_OPT_IN:-}}" != "1" ]; then
    echo "refusing: set MOJOTREES_VALIDATION_OPT_IN=1 to run this plan" >&2
    exit 2
fi

cd "{root}" || exit 2

ulimit -c 0 2>/dev/null || true

MB_LOCK="{lock}"
MB_TIMEOUT=$(command -v timeout 2>/dev/null || command -v gtimeout 2>/dev/null || true)
[ -n "$MB_TIMEOUT" ] || echo "note: no timeout(1) or gtimeout on PATH; jobs run unbounded" >&2

MB_LOG=${{MB_LOG:-${{TMPDIR:-/tmp}}/mojotrees-validation-$(date +%Y%m%dT%H%M%S)}}
mkdir -p "$MB_LOG" || exit 2
echo "transcripts: $MB_LOG"

# One runner, used by every job. The command itself never appears inside
# another layer of quotes: it travels in MB_CMD, which both branches read.
MB_RUNNER='if [ -n "${{MB_TIMEOUT:-}}" ]; then "$MB_TIMEOUT" "$MB_SECS" /bin/sh -c "$MB_CMD"; else /bin/sh -c "$MB_CMD"; fi'
export MB_RUNNER MB_TIMEOUT

MB_PASS=0
MB_FAIL=0
MB_FAILED=""
"""

SCRIPT_JOB = """
# --- {job_id} ({tier}, ~{cost}s, {provenance}) ------------------------------
# {title}
# proves: {proves}
echo ""
echo "==> {job_id}"
rm -f "$MB_LOG/{safe}.status"
(
{exports}    MB_CMD={command}
    MB_SECS={timeout}
    export MB_CMD MB_SECS
    # The status file exists because `cmd | tee` reports tee's status, and
    # pipefail is not POSIX. Without this a failing suite would be recorded
    # as a pass, which is the one failure mode this whole file exists to avoid.
    {{ {invocation}; echo $? > "$MB_LOG/{safe}.status"; }} 2>&1 \\
        | tee "$MB_LOG/{safe}.log"
)
MB_STATUS=$(cat "$MB_LOG/{safe}.status" 2>/dev/null || echo 1)
if [ "$MB_STATUS" = "0" ]; then
    MB_PASS=$((MB_PASS + 1))
    echo "    ok   {job_id}"
else
    MB_FAIL=$((MB_FAIL + 1))
    MB_FAILED="$MB_FAILED {job_id}"
    echo "    FAIL {job_id} (status $MB_STATUS)"
fi
"""

SCRIPT_FOOTER = """
echo ""
echo "passed $MB_PASS, failed $MB_FAIL"
if [ -n "$MB_FAILED" ]; then
    echo "failed:$MB_FAILED"
    echo "transcripts in $MB_LOG"
    exit 1
fi
echo "transcripts in $MB_LOG"
exit 0
"""


def render_sh(plan, out_path):
    man = plan.man
    lock = man.locks.get("wrapper", "tools/with_build_lock.sh")
    chunks = [
        SCRIPT_HEADER.format(
            out=out_path or "plan.sh", root=ROOT, lock=lock
        )
    ]
    for job_id in plan.scheduled:
        job = man.jobs[job_id]
        exports = ""
        env = dict(job.get("env", {}))
        # A thread cap only means something to a job that trains or compiles.
        # Setting it on a file reader would be noise pretending to be a limit.
        if job["tier"] not in ("static", "smoke"):
            env.setdefault("MOJOTREES_NUM_WORKERS", str(man.threads(job_id)))
        for name, value in sorted(env.items()):
            exports += f"    {name}={shlex.quote(str(value))}\n"
            exports += f"    export {name}\n"
        klass = man.exclusive(job_id)
        if klass:
            # The per-class lock path travels in an environment variable the
            # current wrapper does not read, so this changes nothing today and
            # separates the classes the moment the wrapper honors it. Patch P1

            lock_name = man.lock_env()
            if lock_name:
                exports += f"    {lock_name}={shlex.quote(man.lock_file_for(klass))}\n"
                exports += f"    export {lock_name}\n"
            invocation = '"$MB_LOCK" /bin/sh -c "$MB_RUNNER"'
        else:
            invocation = '/bin/sh -c "$MB_RUNNER"'
        chunks.append(
            SCRIPT_JOB.format(
                job_id=job_id,
                safe=job_id.replace(":", "_"),
                tier=job["tier"],
                cost=man.cost(job_id),
                provenance=job.get("provenance", "unverified"),
                title=job.get("title", ""),
                proves=job.get("proves", "").replace("\n", " "),
                exports=exports,
                command=shlex.quote(job["command"]),
                timeout=man.timeout(job_id),
                invocation=invocation,
            )
        )
    if plan.held or plan.over_budget or plan.blocked:
        chunks.append("\n# Not in this script, and why:\n")
        for job_id, why in plan.held:
            chunks.append(f"#   held        {job_id}: {why}\n")
        for job_id, why in plan.blocked:
            first = (why or "").strip().splitlines()
            chunks.append(
                f"#   blocked     {job_id}: {first[0] if first else ''}\n"
            )
        for job_id, cost in plan.over_budget:
            chunks.append(f"#   over budget {job_id}: ~{cost}s\n")
    chunks.append(SCRIPT_FOOTER)
    return "".join(chunks)


# ---------------------------------------------------------------------------
# --self-check
# ---------------------------------------------------------------------------


def pixi_tasks():
    """Task names defined in pixi.toml, across every feature."""
    text = (ROOT / "pixi.toml").read_text()
    names = set()
    in_tasks = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_tasks = stripped.rstrip("]").lstrip("[").endswith("tasks")
            continue
        if in_tasks and "=" in stripped and not stripped.startswith("#"):
            names.add(stripped.split("=", 1)[0].strip().strip('"'))
    return names


SUITE_RE = re.compile(r"tests/test_[A-Za-z0-9_]+\.mojo")


def pixi_suites():
    """Mojo suites some pixi task actually runs.

    A suite named here is one the repository itself considers part of a
    checkable whole. That is a stronger signal than a file sitting in tests/,
    which may be a lane's work in progress, and it is the signal --self-check
    holds the manifests to.
    """
    text = (ROOT / "pixi.toml").read_text()
    return {normalize(m) for m in SUITE_RE.findall(text)}


def repo_corpus():
    """Text of the files a MOJOTREES_* name would be defined in."""
    parts = []
    for pattern in (
        "src/mojotrees/*.mojo",
        "bench/**/*.mojo",
        "bench/**/*.py",
        "bench/**/*.sh",
        "python/mojotrees/*.py",
        "packaging/**/*.py",
        "packaging/**/*.sh",
        "docs/*.md",
        ".github/workflows/*.yml",
        "tools/*.py",
        "capi/*.sh",
        "bindings/*.sh",
    ):
        for path in ROOT.glob(pattern):
            try:
                parts.append(path.read_text(errors="ignore"))
            except OSError:
                continue
    return "\n".join(parts)


def command_targets(command):
    """The pixi task a command runs, when it runs one."""
    try:
        tokens = shlex.split(command, comments=False, posix=True)
    except ValueError as exc:
        raise ValueError(f"unparsable command {command!r}: {exc}") from exc
    targets = []
    index = 0
    while index < len(tokens) - 1:
        if tokens[index] == "pixi" and tokens[index + 1] == "run":
            cursor = index + 2
            if cursor < len(tokens) and tokens[cursor] in ("-e", "--environment"):
                cursor += 2
            if cursor < len(tokens):
                targets.append(tokens[cursor])
            index = cursor
        index += 1
    return targets


def exists(pattern):
    pattern = normalize(pattern)
    if any(ch in pattern for ch in "*?["):
        return bool(list(ROOT.glob(pattern)))
    return (ROOT / pattern).exists()


def self_check(man):
    problems = []
    notes = []

    def fail(message):
        problems.append(message)

    tier_ids = set(man.tiers)
    job_ids = set(man.jobs)
    tasks = pixi_tasks()

    seen = set()
    for job in man.jobs_doc.get("job", []):
        job_id = job["id"]
        if job_id in seen:
            fail(f"jobs: {job_id} is defined twice")
        seen.add(job_id)
        if job["tier"] not in tier_ids:
            fail(f"jobs: {job_id} names tier {job['tier']!r}, which tiers.toml does not define")
            continue
        if not job_id.startswith(job["tier"] + ":"):
            fail(f"jobs: {job_id} is in tier {job['tier']!r} and must be prefixed with it")
        if not job.get("proves", "").strip():
            fail(f"jobs: {job_id} has no `proves`. A command nobody can say the meaning of is not evidence")
        if job.get("provenance", "unverified") not in PROVENANCE:
            fail(f"jobs: {job_id} has provenance {job.get('provenance')!r}; expected one of " + ", ".join(sorted(PROVENANCE)))
        if job.get("blocked"):
            if job.get("command"):
                fail(f"jobs: {job_id} is blocked and still carries a command")
            if not (job.get("blocked_reason") or "").strip():
                fail(f"jobs: {job_id} is blocked with no reason. A hole with no explanation reads as an oversight")
        elif not job.get("command", "").strip():
            fail(f"jobs: {job_id} has neither a command nor blocked = true")

        for need in job.get("needs", []):
            if need not in KNOWN_NEEDS:
                fail(f"jobs: {job_id} needs {need!r}, which is not a known precondition")
        for dependency in job.get("depends_on", []):
            if dependency not in job_ids:
                fail(f"jobs: {job_id} depends on {dependency}, which is not a job")
        for name in job.get("env", {}):
            if not name.startswith("MOJOTREES_") and name not in ALLOWED_FOREIGN_ENV:
                fail(f"jobs: {job_id} sets {name}, which is outside this project's MOJOTREES_* env contract")
        for path in job.get("requires_files", []):
            if not exists(path):
                fail(f"jobs: {job_id} requires {path}, which does not exist")
        try:
            targets = command_targets(job.get("command", ""))
        except ValueError as exc:
            fail(f"jobs: {job_id} {exc}")
            targets = []
        for target in targets:
            if target in KNOWN_PROGRAMS:
                continue
            if target not in tasks:
                fail(f"jobs: {job_id} runs `pixi run {target}`, and pixi.toml defines no such task")

    # dependency cycles
    colors = {}

    def visit(job_id, trail):
        if colors.get(job_id) == "done":
            return
        if colors.get(job_id) == "open":
            fail("jobs: dependency cycle " + " -> ".join(trail + [job_id]))
            return
        colors[job_id] = "open"
        for dependency in man.jobs.get(job_id, {}).get("depends_on", []):
            if dependency in man.jobs:
                visit(dependency, trail + [job_id])
        colors[job_id] = "done"

    for job_id in job_ids:
        visit(job_id, [])

    # environment names this repository has never heard of
    corpus = repo_corpus()
    for job in man.jobs_doc.get("job", []):
        for name in job.get("env", {}):
            if name not in corpus:
                fail(
                    f"jobs: {job['id']} sets {name}, which appears nowhere in "
                    "src, bench, python, packaging, docs, or the workflows. "
                    "Setting a variable nothing reads is a plan that quietly "
                    "does not do what it says"
                )

    # Every suite the repository's own task chain runs must be reachable from a
    # job, or the planner under-plans in the one direction that matters: it
    # tells someone their change needs no test when a test for it exists and
    # runs on every push.
    #
    # This is a failure and not a note on purpose. The other coverage gaps are
    # notes because the tree grows faster than the map and an unmapped new
    # module is normal. This one is different: adding a suite to pixi.toml is a
    # deliberate statement that it belongs to the checkable whole, so a map that
    # has not caught up is wrong rather than merely incomplete. It is also the
    # exact case that went unnoticed until it was found by hand:
    # tests/test_binning.mojo was in the test chain and in no job, so a change
    # to src/mojotrees/binning.mojo planned three suites, none of them the one
    # that tests binning.
    covered_suites = set()
    for job in man.jobs.values():
        for path in job.get("requires_files", []):
            covered_suites.add(normalize(path))
        covered_suites.update(SUITE_RE.findall(job.get("command", "")))
    for suite in sorted(pixi_suites() - covered_suites):
        fail(
            f"jobs: {suite} is run by a pixi task and named by no job, so a "
            "change to what it covers plans without it. Add a [[job]] for it "
            "and put that job in the subsystem that owns the module it imports"
        )

    # the lock table
    wrapper = man.locks.get("wrapper", "")
    if not wrapper:
        fail("tiers: [locks] names no wrapper, so exclusive jobs would run unserialized")
    elif not exists(wrapper):
        fail(f"tiers: [locks] wrapper {wrapper} does not exist")
    classes = set(man.locks.get("classes", []))
    for klass in sorted(man.locks.get("class_lock_files", {})):
        if klass not in classes:
            fail(f"tiers: [locks.class_lock_files] names {klass!r}, which is not an exclusion class")
    lock_env = man.lock_env()
    if lock_env and not lock_env.startswith("MOJOTREES_"):
        fail(f"tiers: [locks] lock_env is {lock_env!r}, which is outside this project's MOJOTREES_* env contract")
    declared = {man.exclusive(job_id) for job_id in man.jobs}
    for klass in sorted(k for k in declared if k):
        if klass not in classes:
            fail(f"tiers: exclusion class {klass!r} is used by a job or tier and is not listed in [locks] classes")
    if lock_env and wrapper and exists(wrapper):
        # Read the wrapper itself rather than the corpus: a mention in a doc is
        # not a wrapper that honors the variable.
        try:
            wrapper_text = (ROOT / normalize(wrapper)).read_text(errors="ignore")
        except OSError:
            wrapper_text = ""
        if lock_env not in wrapper_text:
            notes.append(
                f"{lock_env} is exported by the emitted script and {wrapper} "
                "does not read it, so every exclusion class still shares "
                f"{man.locks.get('lock_file', 'one lock')}. Inert, not broken. "
                "Patch P1"
            )

    # subsystems and gaps
    for subsystem in man.subsystems:
        for key in ("jobs", "escalate"):
            for job_id in subsystem.get(key, []):
                if job_id not in job_ids:
                    fail(f"subsystems: {subsystem['id']} {key} names {job_id}, which is not a job")
        for path in subsystem["paths"]:
            if not exists(path):
                fail(f"subsystems: {subsystem['id']} names {path}, which does not exist")
    for gap in man.gaps:
        for job_id in gap.get("fallback", []):
            if job_id not in job_ids:
                fail(f"subsystems: a gap names fallback {job_id}, which is not a job")
        for path in gap["paths"]:
            if not exists(path):
                fail(f"subsystems: a gap names {path}, which does not exist")
        if not gap.get("reason", "").strip():
            fail(f"subsystems: the gap for {gap['paths'][0]} has no reason")

    # handoffs and lanes
    retired = []
    for entry in man.handoffs:
        for path in entry["paths"]:
            if exists(path):
                continue
            if man.retired_path(path):
                retired.append(normalize(path))
            else:
                fail(f"handoffs: {path} does not exist")
        for key in ("jobs", "escalate"):
            for job_id in entry.get(key, []):
                if job_id not in job_ids:
                    fail(f"handoffs: {entry['paths'][0]} {key} names {job_id}, which is not a job")
    if retired:
        # One note for the whole directory, not one per file. The point of the
        # note is that the mapping outlived the memos, which is a single fact.
        notes.append(
            f"{len(retired)} handoff paths were removed by "
            f"{man.archive.get('commit', 'a deliberate deletion')} "
            f"({man.archive.get('title', 'handoffs retired')}). Their job "
            "mappings are kept on purpose; see the header of handoffs.toml. "
            "--handoff still resolves them, because it matches on names and "
            "never reads the file."
        )
    for lane in man.lanes:
        for key in ("jobs", "escalate"):
            for job_id in lane.get(key, []):
                if job_id not in job_ids:
                    fail(f"handoffs: lane {lane['id']} {key} names {job_id}, which is not a job")
        if not exists(lane["path"]):
            notes.append(f"lane {lane['id']} is pending: {lane['path']} is not written yet")

    notes.extend(coverage_notes(man))
    return problems, notes


def coverage_notes(man):
    """What the manifests do not name. Reported, never fatal: this tree grows
    faster than any manifest in it."""
    notes = []

    covered_files = set()
    for job in man.jobs.values():
        for path in job.get("requires_files", []):
            covered_files.add(normalize(path))

    # `test_*.mojo`, not `*.mojo`: tests/support.mojo holds the shared data
    # generators and is not a suite, so a job naming it would be wrong.
    # tests/parallel/ is gone; the directory was named for the lane that wrote
    # its files, never for how they run.
    suites = sorted(
        normalize(p.relative_to(ROOT))
        for p in (ROOT / "tests").glob("test_*.mojo")
    )
    orphan_suites = [s for s in suites if s not in covered_files]
    if orphan_suites:
        notes.append(
            "no job runs these suites: " + ", ".join(orphan_suites)
        )

    pytests = sorted(
        normalize(p.relative_to(ROOT))
        for p in list((ROOT / "python" / "tests").glob("test_*.py"))
        + list((ROOT / "python" / "tests" / "parallel").glob("test_*.py"))
    )
    orphan_pytests = [p for p in pytests if p not in covered_files]
    if orphan_pytests:
        notes.append("no job runs these pytest files: " + ", ".join(orphan_pytests))

    named_paths = []
    for subsystem in man.subsystems:
        named_paths.extend(subsystem["paths"])
    for gap in man.gaps:
        named_paths.extend(gap["paths"])
    modules = sorted(
        normalize(p.relative_to(ROOT))
        for p in (ROOT / "src" / "mojotrees").glob("*.mojo")
    )
    unmapped = [
        m for m in modules if not any(path_matches(p, m) for p in named_paths)
    ]
    if unmapped:
        notes.append(
            "no subsystem or gap names these modules: " + ", ".join(unmapped)
        )

    named_handoffs = set()
    for entry in man.handoffs:
        named_handoffs.update(normalize(p) for p in entry["paths"])
    for lane in man.lanes:
        named_handoffs.add(normalize(lane["path"]))
    handoff_files = sorted(
        normalize(p.relative_to(ROOT)) for p in (ROOT / "handoffs").glob("*.md")
    )
    unnamed = [h for h in handoff_files if h not in named_handoffs]
    if unnamed:
        notes.append("no entry names these handoffs: " + ", ".join(unnamed))

    return notes


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="validation_plan.py",
        description=(
            "Print the smallest set of validation commands a change earns. "
            "This tool never runs a validation command."
        ),
    )
    what = parser.add_argument_group("what changed")
    what.add_argument("--paths", nargs="+", metavar="PATH", help="explicit change set; skips git entirely")
    what.add_argument("--changed-from", metavar="FILE", help="read the change set from a file, one path per line")
    what.add_argument("--since", metavar="REF", help="use `git diff --name-only REF` as the change set")
    what.add_argument("--handoff", action="append", default=[], metavar="NAME", help="a handoff path, file name, or lane id")
    what.add_argument("--subsystem", action="append", default=[], metavar="ID", help="a subsystem id from subsystems.toml")
    what.add_argument("--job", action="append", default=[], metavar="ID", help="a job id, selected directly")
    what.add_argument("--tier", action="append", default=[], metavar="ID", help="every job in a tier")
    what.add_argument("--all", action="store_true", help="every job in every tier, still only printed")

    how = parser.add_argument_group("how much")
    how.add_argument("--allow", action="append", default=[], metavar="TIER", help="opt in to an opt-in tier; repeatable, or comma separated")
    how.add_argument("--escalate", action="store_true", help="add each subsystem's escalation jobs")
    how.add_argument("--budget-seconds", type=int, default=None, metavar="N")
    how.add_argument("--max-jobs", type=int, default=None, metavar="N")
    how.add_argument("--measure", action="store_true", help="this plan carries timings: hold every job outside one exclusion class")
    how.add_argument("--no-gpu", action="store_true", help="hold every job that needs an accelerator")
    how.add_argument("--offline", action="store_true", help="hold every job that needs the network")
    how.add_argument("--no-sudo", action="store_true", help="hold every job that would ask for sudo")

    out = parser.add_argument_group("output")
    out.add_argument("--format", choices=("text", "json", "sh"), default="text")
    out.add_argument("--out", metavar="FILE", help="write the output to a file instead of stdout")
    out.add_argument("--explain", action="store_true", help="add what each job proves, needs, and where its output is filed")

    checks = parser.add_argument_group("checks on the manifests themselves")
    checks.add_argument("--self-check", action="store_true", help="validate validation/manifests against the repository")
    checks.add_argument("--coverage", action="store_true", help="report what the manifests do not name")
    checks.add_argument("--list-tiers", action="store_true")
    checks.add_argument("--list-jobs", action="store_true")
    checks.add_argument("--list-subsystems", action="store_true")

    args = parser.parse_args(argv)

    allow = set()
    for value in args.allow:
        allow.update(part.strip() for part in value.split(",") if part.strip())
    args.allow = allow
    return args


def list_tiers(man):
    lines = []
    for tier in sorted(man.tiers.values(), key=lambda t: t.get("order", 99)):
        gate = f"--allow {tier['id']}" if tier.get("opt_in") else "on by default"
        count = sum(1 for j in man.jobs.values() if j["tier"] == tier["id"])
        lines.append(f"{tier['id']:14s} {count:3d} jobs  {gate}")
        lines.append(f"               proves: {tier.get('proves', '')}")
        lines.append(f"               not:    {tier.get('does_not_prove', '')}")
    return "\n".join(lines)


def list_jobs(man):
    lines = []
    for job_id in sorted(man.jobs, key=man.order_key):
        job = man.jobs[job_id]
        mark = "BLOCKED" if job.get("blocked") else f"~{man.cost(job_id)}s"
        lines.append(f"{job_id:38s} {mark:>8s}  {job.get('title', '')}")
    return "\n".join(lines)


def list_subsystems(man):
    lines = []
    for subsystem in man.subsystems:
        lines.append(f"{subsystem['id']:24s} {subsystem.get('title', '')}")
        jobs = ", ".join(subsystem.get("jobs", [])) or "none"
        lines.append(f"    {len(subsystem['paths'])} path(s), jobs: {jobs}")
    return "\n".join(lines)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    man = Manifests()

    if args.self_check:
        problems, notes = self_check(man)
        print("checking validation/manifests against the repository")
        print(f"  {len(man.tiers)} tiers, {len(man.jobs)} jobs, "
              f"{len(man.subsystems)} subsystems, {len(man.gaps)} gaps, "
              f"{len(man.handoffs)} handoffs, {len(man.lanes)} lanes")
        for note in notes:
            print(f"  note: {note}")
        if problems:
            print(f"\n{len(problems)} problem(s):")
            for problem in problems:
                print(f"  - {problem}")
            return 1
        print("  ok")
        return 0

    if args.coverage:
        for note in coverage_notes(man) or ["the manifests name everything"]:
            print(f"- {note}")
        return 0

    if args.list_tiers:
        print(list_tiers(man))
        return 0
    if args.list_jobs:
        print(list_jobs(man))
        return 0
    if args.list_subsystems:
        print(list_subsystems(man))
        return 0

    args.budget_seconds = (
        args.budget_seconds
        if args.budget_seconds is not None
        else int(man.budget.get("default_seconds", 600))
    )
    args.max_jobs = (
        args.max_jobs
        if args.max_jobs is not None
        else int(man.budget.get("max_jobs", 12))
    )

    selection = Selection()
    plan = Plan(man, args)

    explicit = bool(
        args.paths
        or args.changed_from
        or args.handoff
        or args.subsystem
        or args.job
        or args.tier
        or args.all
    )

    paths = []
    if args.paths:
        paths = [normalize(p) for p in args.paths]
        plan.source = "--paths"
    elif args.changed_from:
        text = Path(args.changed_from).read_text()
        paths = [normalize(line) for line in text.splitlines() if line.strip()]
        plan.source = args.changed_from
    elif not explicit or args.since:
        paths, plan.source = git_changed_paths(args.since)

    if paths:
        found, gaps, unmatched = select_for_paths(man, paths, args.escalate)
        for job_id in found:
            for reason in found.reasons[job_id]:
                selection.add(job_id, reason)
        plan.gaps = gaps
        plan.unmatched = unmatched
    plan.change_set = paths

    for name in args.handoff:
        found, resolved = select_for_handoff(man, name, args.escalate)
        if not resolved:
            print(f"no handoff or lane matches {name!r}", file=sys.stderr)
            return 2
        for job_id in found:
            for reason in found.reasons[job_id]:
                selection.add(job_id, reason)

    for wanted in args.subsystem:
        matches = [s for s in man.subsystems if s["id"] == wanted]
        if not matches:
            print(f"no subsystem {wanted!r}", file=sys.stderr)
            return 2
        for subsystem in matches:
            for job_id in subsystem.get("jobs", []):
                selection.add(job_id, f"subsystem {subsystem['id']}")
            if args.escalate:
                for job_id in subsystem.get("escalate", []):
                    selection.add(job_id, f"escalation of {subsystem['id']}")

    for job_id in args.job:
        if job_id not in man.jobs:
            print(f"no job {job_id!r}", file=sys.stderr)
            return 2
        selection.add(job_id, "named with --job")

    for tier_id in args.tier:
        if tier_id not in man.tiers:
            print(f"no tier {tier_id!r}", file=sys.stderr)
            return 2
        for job_id, job in man.jobs.items():
            if job["tier"] == tier_id:
                selection.add(job_id, f"every job in tier {tier_id}")

    if args.all:
        for job_id in man.jobs:
            selection.add(job_id, "--all")

    if not plan.source:
        plan.source = "the selection flags"

    add_dependencies(man, selection)
    plan.build(selection)

    if args.format == "json":
        text = render_json(plan, selection)
    elif args.format == "sh":
        text = render_sh(plan, args.out)
    else:
        text = render_text(plan, selection, args.explain)

    if args.out:
        Path(args.out).write_text(text if text.endswith("\n") else text + "\n")
        print(f"wrote {args.out} ({len(plan.scheduled)} jobs, {plan.spent}s)")
        if args.format == "sh":
            print("Read it, then: MOJOTREES_VALIDATION_OPT_IN=1 sh " + args.out)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
