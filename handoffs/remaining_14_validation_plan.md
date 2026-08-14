# Handoff, remaining round task 14: the targeted validation orchestrator

Nothing outside the four assigned paths was touched. No test, build, benchmark,
CI job, profiler, or the new orchestrator itself was run. Nothing was committed
or staged by this lane.

## What this lane produced

| Path | What it is |
| --- | --- |
| `tools/validation_plan.py` | The planner. Reads four manifests, turns a change set into the smallest set of commands that change earns, and prints them. Standard library only. It never executes a validation command |
| `validation/manifests/tiers.toml` | Nine tiers of evidence, their budgets, their opt-in gates, their exclusion classes, and the lock table |
| `validation/manifests/jobs.toml` | 119 jobs. Each carries a command or a `blocked` flag with a reason, a cost guess, a provenance, the files it needs, and a `proves` sentence |
| `validation/manifests/subsystems.toml` | 46 subsystems mapping repository paths to jobs, plus 6 `[[gap]]` entries for paths no job covers |
| `validation/manifests/handoffs.toml` | 42 handoff entries and 15 lane entries, mapping a handoff document to the jobs that would test the work it describes |
| `docs/FOCUSED_VALIDATION_PLAN.md` | Why the tiers are the way they are, how a changed path becomes a command, what the method cannot see, and the known holes |
| `handoffs/remaining_14_validation_plan.md` | This file |

## Read this before trusting any of it

**The planner has never been run.** Not with `--self-check`, not with `--all`,
not with `--format sh`. It parses four TOML files it was written alongside, and
every branch in it is written from a reading of those files, but a program that
has never met its input is a hypothesis. The section "Validation, all of it
UNRUN" at the end says what to run first and what each command should print.

**No number in the manifests is a measurement.** Every `budget_seconds`, every
value under `[budget]`, and every duration in the plan document is an estimate
written by looking at what a job does. They are good enough to order a plan and
to truncate it at a budget. They are not good enough to quote anywhere.
`docs/FOCUSED_VALIDATION_PLAN.md` section 6 says this at length, and the same
sentence belongs in any report that cites one of them.

**`provenance` is the honesty field.** `ci` means a workflow under
`.github/workflows/` runs that exact command. `documented` means a file in this
repository says it is the right command and nobody in this lane confirmed it.
`unverified` means neither. A plan prints the provenance next to the command so
that a job which has never run anywhere does not look like one that runs on
every push.

**Blocked work is listed, not omitted.** `distributed:two-process` carries no
command and a `blocked_reason`, because no two-process path exists in the tree.
A plan prints it under "BLOCKED, no command exists". A hole that prints is a
hole somebody can close.

## What the planner is, in one paragraph

`python3 tools/validation_plan.py` takes a change set, from `--paths`,
`--changed-from`, `--since`, or a read-only `git status`, and maps it through
`subsystems.toml` and `handoffs.toml` to a job set. It then orders the jobs by
tier, drops the tiers the operator did not opt into, holds at most one broad
suite, charges each job against a wall clock budget, and prints what it kept and
what it dropped and why. `--format sh` writes a POSIX script instead of a list.
`--self-check` validates the manifests against the tree. The default output is a
list of commands, and there is no flag anywhere that makes the tool execute one.

The nine tiers, cheapest first, are `static`, `smoke`, `native`, `python`,
`differential`, `wheel`, `distributed`, `hardware`, and `broad`. The last five
are `opt_in`, and no flag turns them on together.

## What was integrated inside ownership

Three connections were made rather than deferred, because both ends were inside
this lane's paths.

1. **The compatibility lane's two tools are wired in.** `tools/api_snapshot.py`
   and `tools/model_fixture_manifest.py` landed from the concurrent task 13 lane
   while this one was writing. They are now `static:api-snapshot` and
   `static:model-fixtures` in `jobs.toml`, reachable through the new
   `compatibility-register` subsystem, and named by the
   `remaining-13-compatibility` lane entry. Both are `unverified` and both fail
   today because their baselines are not in the tree, which is correct behavior
   for a check with a missing baseline and is written into their `proves`
   strings so the first failure is not misread as a regression.

2. **The per-class lock seam.** `[locks]` in `tiers.toml` now carries
   `lock_env = "MOJOBOOST_BUILD_LOCK"` and a `[locks.class_lock_files]` table,
   `Manifest.lock_env()` and `Manifest.lock_file_for()` resolve it, and
   `render_sh` exports it before every exclusive job. The current wrapper reads
   no environment, so the export is inert and all three classes keep sharing one
   lock file, which is today's behavior exactly. Patch P1 makes it live.

3. **`--self-check` covers the lock table.** It fails when the wrapper does not
   exist, when `[locks.class_lock_files]` names a class that is not an exclusion
   class, when `lock_env` falls outside the `MOJOBOOST_*` contract, or when a job
   or tier declares an exclusion class the lock table does not list. It emits a
   note, not a failure, when the wrapper does not read `lock_env`, and it decides
   that by reading the wrapper rather than by grepping the tree, so the note
   disappears the day P1 lands and not a day earlier.

Everything else that would connect this lane to the rest of the repository lands
in a file this lane may not edit, and is below.

---

# READY-TO-APPLY INTEGRATION PATCHES

Six patches. Each is blocked only by ownership. They are ordered so that no
patch depends on a later one.

Every "minimal later validation" line is **UNRUN**.

---

## P1. Let the build lock take a per-class lock file

**Target file and symbol:** `tools/with_build_lock.sh`, the whole script. It is
nine lines today and the lock path is a literal inside the embedded Python.
**Owner:** whoever owns `tools/` shell scripts (shared).
**Depends on:** nothing. The manifest and planner side is already applied.
**Public API effect:** none. This is a developer tool, not a shipped surface.
**Serialization effect:** none. No model file, no wire format.

**Signature.** Unchanged for every existing caller.
`tools/with_build_lock.sh <command> [args...]`. The new input is the environment
variable `MOJOBOOST_BUILD_LOCK`, unset by default.

Replace the script body with:

```sh
#!/bin/sh
# Serialize heavy builds/tests across parallel sessions.
# macOS has no flock(1), so this blocks on a Python fcntl lock without polling.
# Usage: tools/with_build_lock.sh <command> [args...]
#
# MOJOBOOST_BUILD_LOCK chooses the lock file. Jobs that cannot contend for the
# same resource, a CPU suite and a GPU suite for instance, take different files
# and stop waiting on each other. Unset means the historical machine-wide lock,
# so every existing caller keeps its current behavior.
MOJOBOOST_BUILD_LOCK="${MOJOBOOST_BUILD_LOCK:-/tmp/mojoboost-build.lock}"
export MOJOBOOST_BUILD_LOCK
exec /usr/bin/python3 -c '
import fcntl, os, subprocess, sys
f = open(os.environ["MOJOBOOST_BUILD_LOCK"], "w")
fcntl.flock(f, fcntl.LOCK_EX)
sys.exit(subprocess.call(sys.argv[1:]))
' "$@"
```

**Call sites.** Every existing one is unaffected, because the default is the
path the script hardcodes today. Grep finds no shell or workflow caller at all,
only documentation: `docs/STARTUP_LATENCY.md`, `handoffs/apple_a1_active_rows.md`,
`handoffs/integration_07_apple_gpu.md`, `handoffs/performance_15_startup.md`,
and this lane's own files. The first caller that sets the variable is the script
emitted by `validation_plan.py --format sh`, which already exports it.

**State flow.** The variable travels one way, from the emitted script's per-job
subshell into the wrapper process, and is not read back. The lock is held for
exactly the lifetime of the child process, released by process exit, as today.

**Errors.** An unwritable or nonexistent directory in the path raises `OSError`
from `open`, the wrapper exits non-zero before the command starts, and the
emitted script records that as a job failure with the traceback in the job's
log. This is deliberate. A wrapper that silently fell back to the shared lock
when the requested one could not be opened would serialize jobs the operator
asked to separate and would look like a slow machine rather than a
misconfiguration.

**Fallback.** The unset case is the fallback and it is the current behavior. If
this patch is never applied, nothing breaks, the classes keep sharing one file,
and `--self-check` keeps printing its note saying so.

**Minimal later validation, UNRUN:**
`tools/with_build_lock.sh /bin/echo ok` prints `ok` and exits 0, and
`MOJOBOOST_BUILD_LOCK=/tmp/mb-t.lock tools/with_build_lock.sh /bin/echo ok`
prints `ok`, exits 0, and leaves `/tmp/mb-t.lock` on disk. Then
`python3 tools/validation_plan.py --self-check` no longer prints the
`MOJOBOOST_BUILD_LOCK is exported and the wrapper does not read it` note.

---

## P2. Declare scipy in the pytest environment

**Target file and symbol:** `pixi.toml`, `[feature.pytest.dependencies]`.
**Owner:** whoever owns `pixi.toml` (shared).
**Depends on:** nothing.
**Public API effect:** none. It is a test-time dependency and it must not become
a runtime one. `python/pyproject.toml` is not touched by this patch.
**Serialization effect:** none.

The table declares `pytest`, `scikit-learn`, `pandas`, `pyarrow`, and `polars`,
and does not declare scipy. `python/tests/test_contrib.py:504` and
`python/tests/test_validation.py:64` both call
`pytest.importorskip("scipy.sparse")`, and `pixi run -e pytest test-estimators`
is the only task that collects those files. So the sparse paths skip, the run
reports green, and the scipy conversion code in `python/mojoboost/_validation.py`
and `python/mojoboost/_arrow.py` has no covering test that ever executes.

Add one line to `[feature.pytest.dependencies]`:

```toml
# Two tests importorskip scipy.sparse. Without it here they skip in the only
# environment that collects them, and a skip nobody reads is a pass.
scipy = "*"
```

**Signature.** None. A dependency declaration.
**Call site.** `pixi run -e pytest test-estimators`, which the `python` CI job
runs on both matrix runners.
**State flow.** None.
**Errors.** Newly collected assertions in those two tests may fail. That is the
point of the patch, and it is the reason to land it on its own commit rather
than inside an unrelated one. If they fail, the bug is in the sparse conversion
path and it has been there unobserved.
**Fallback.** The tests keep their `importorskip`, so an environment without
scipy still skips cleanly rather than erroring at collection.
**Minimal later validation, UNRUN:**
`pixi run -e pytest test-estimators -k sparse` reports the sparse tests as
passed or failed rather than skipped.

---

## P3. Run the C ABI test in CI

**Target file and symbol:** `.github/workflows/ci.yml`, the `test` job's `steps`
list.
**Owner:** whoever owns the workflows (shared).
**Depends on:** nothing.
**Public API effect:** none directly, but this is the only gate that would catch
a break in `capi/mojoboost.h`, which is a public surface.
**Serialization effect:** none.

`pixi.toml:35` defines `test-c = "capi/run_c_tests.sh"`, which builds the shared
library, compiles `capi/test_capi.c` against the public header with
`-std=c99 -Wall -Wextra -Werror`, and runs it. No workflow invokes it. The Mojo
suites `test-capi` and `test-cli` do run in CI, but they exercise the ABI from
Mojo, so nothing anywhere proves the header compiles under a C compiler.

Add a step to the existing `test` job, after `Run tests`:

```yaml
      # The C ABI from C. `pixi run test` covers it from Mojo only, so this is
      # the only check that the public header in capi/ compiles under a C
      # compiler with -Wall -Wextra -Werror. The script skips itself with
      # status 0 when no compiler is present.
      - name: C ABI tests
        run: pixi run test-c
```

**Signature.** None.
**Call site.** The `test` job, which already runs on `ubuntu-latest` and
`ubuntu-24.04-arm`, so this gets both architectures for free and needs no new
pixi environment.
**State flow.** The script writes `capi/libmojoboost.{so,dylib}` and
`capi/test_capi` into the checkout. Both are build outputs in a throwaway runner
workspace. Confirm they are ignored before running this locally in a dirty tree.
**Errors.** A compile error or a failing assertion fails the job with the
compiler's or the test's own output. Note that `capi/run_c_tests.sh` calls
`pixi run printenv CONDA_PREFIX` internally to find the runtime search path, so
invoking it through `pixi run test-c` nests one pixi invocation inside another.
That works and costs a few seconds. Calling `capi/run_c_tests.sh` directly is
the alternative if the nesting ever misbehaves.
**Fallback.** The script exits 0 with a printed message when `cc` is absent, so
a runner image without a toolchain reports green rather than red. On the GitHub
Ubuntu images `cc` is present, so this will actually execute.
**Dependency.** None on the other patches. It does depend on `capi/build.sh`
succeeding on the runner, which `pixi run test` does not currently prove.
**Minimal later validation, UNRUN:** the step passes on a clean checkout, and
fails when a field is reordered inside a struct in `capi/mojoboost.h` without
`capi/test_capi.c` being updated.

---

## P4. Run the planner's self-check in CI

**Target file and symbol:** `.github/workflows/ci.yml`, the `parity` job's
`steps` list.
**Owner:** whoever owns the workflows (shared).
**Depends on:** nothing, though it is worth landing after the first local
`--self-check` run has been made to pass. See the sequencing note.
**Public API effect:** none.
**Serialization effect:** none.

The manifests describe the tree, and a description of a tree rots at exactly the
speed the tree changes. `--self-check` is standard library only, reads files,
builds nothing, and exits, which is why the `parity` job is the right home for
it. That job already runs `python3 tools/check_parity.py` on a bare runner in
seconds with no pixi setup at all.

Add a step to the `parity` job:

```yaml
      # The validation manifests describe this tree, so they go stale silently.
      # Standard library only and no build, like the parity check above. It
      # never runs a validation command; it only checks that the ones it names
      # still exist.
      - name: Validation manifests
        run: python3 tools/validation_plan.py --self-check
```

**Signature.** `--self-check` takes no arguments and exits 0 when the manifests
are consistent with the tree, non-zero otherwise.
**Call site.** The `parity` job, `runs-on: ubuntu-latest`, which does no
checkout of pixi and needs none.
**State flow.** None. It reads `validation/manifests/*.toml`, `pixi.toml`, and
the paths those manifests name, and writes nothing.
**Errors.** Every failure names the manifest, the entry, and what does not
exist. The classes are a job naming a tier or a pixi task that does not exist, a
job requiring a file that does not exist, a dangling job reference from a
subsystem, gap, handoff, or lane, a dependency cycle, a job env name outside the
`MOJOBOOST_*` contract or absent from the repository, a blocked job that still
carries a command, and the lock table checks listed above. Coverage findings,
including modules no suite reaches, are printed as notes and are not failures,
because this tree grows faster than any manifest in it.
**Sequencing note.** Do not land this step before somebody has run
`--self-check` locally once and fixed what it finds. This lane never ran it, so
the first run is as likely to find a mistake in the manifests as in the tree,
and a red `main` on the commit that adds the gate gets the gate deleted rather
than the manifests fixed.
**Fallback.** None. It is a gate, and a cheap one.
**Dependency.** P6 closes the two known failures that the compatibility lane
owns. Until then `--self-check` may report the missing baselines through the
`requires_files` of `static:api-snapshot` and `static:model-fixtures`; those two
jobs deliberately require only inputs that exist, so this should pass, and the
first local run is what confirms it.
**Minimal later validation, UNRUN:** `python3 tools/validation_plan.py --self-check`
exits 0 on a clean checkout, and exits non-zero after a job in `jobs.toml` is
pointed at a pixi task that does not exist.

---

## P5. A pixi task for the planner

**Target file and symbol:** `pixi.toml`, `[tasks]`, next to `check-parity`.
**Owner:** whoever owns `pixi.toml` (shared).
**Depends on:** nothing. P4 calls the script directly, as the `parity` job does,
so neither patch needs the other.
**Public API effect:** none.
**Serialization effect:** none.

```toml
# Turns a change set into the smallest set of validation commands it earns, and
# prints them. Standard library only, builds nothing, and never executes a
# validation command. See docs/FOCUSED_VALIDATION_PLAN.md.
validation-plan = "python3 tools/validation_plan.py"
validation-self-check = "python3 tools/validation_plan.py --self-check"
```

**Signature.** `pixi run validation-plan -- [flags]`. Note the `--`, without
which pixi eats the flags.
**Call site.** Developers. Nothing else should call it.
**State flow.** None, unless `--out` is passed, which writes the emitted script
to the named path and nowhere else.
**Errors.** Exit 2 on a bad flag or an unreadable manifest, exit 1 from
`--self-check` on an inconsistency, exit 0 otherwise.
**Fallback.** None needed. `python3` absent means the task fails loudly, which
is what `check-parity` already does.
**Minimal later validation, UNRUN:** `pixi run validation-plan -- --list-tiers`
prints nine tiers and runs nothing.

---

## P6. The compatibility baselines the two static jobs need

**Target files:** `compatibility/api_snapshot.json` and
`compatibility/fixtures/checksums.json`, neither of which exists.
**Owner:** the task 13 compatibility lane. See
`handoffs/remaining_13_compatibility.md`.
**Depends on:** nothing here. This is the residual of an integration that is
otherwise already applied.
**Public API effect:** the snapshot file becomes the record of the public
surface. Creating it does not change the surface.
**Serialization effect:** the fixture register becomes the record of which model
format versions the reader accepts. Creating it does not change the format.

The wiring is done. `static:api-snapshot` runs
`python3 tools/api_snapshot.py --check`, `static:model-fixtures` runs
`python3 tools/model_fixture_manifest.py --check`, both are in the
`compatibility-register` subsystem with `escalate` to `native:serialize`,
`python:pickle`, and `static:parity`, and the `remaining-13-compatibility` lane
entry names all four of its jobs. What is missing is only the baselines, which
that lane must generate with `--write` and which this lane must not fabricate.

**Signature.** `tools/api_snapshot.py --write` and
`tools/model_fixture_manifest.py --write`, both documented in that lane's
handoff.
**Call site.** `validation/manifests/jobs.toml`, jobs `static:api-snapshot` and
`static:model-fixtures`, and any CI job that lane adds.
**State flow.** `--write` reads the tree and writes the baseline file, which
`--check` then reads on every later run.
**Errors.** Until the baselines exist both jobs exit non-zero with a
missing-baseline message. That is the correct behavior for a check with no
baseline and it is recorded in each job's `proves` string so the first failure
is not read as a regression.
**Fallback.** None, and none is wanted. A check that passes because its baseline
is absent is worse than one that fails.
**Minimal later validation, UNRUN:** after `--write`,
`python3 tools/validation_plan.py --job static:api-snapshot` prints the command
with provenance `unverified`, and running that command by hand exits 0.
Promote the provenance to `documented` or `ci` in `jobs.toml` only once that is
actually true.

---

# Validation, all of it UNRUN

In this order, cheapest first. Nothing below has been executed.

1. `python3 tools/validation_plan.py --self-check`
   Reads the four manifests and the paths they name. Expect exit 0 and a list of
   coverage notes, including modules no suite reaches, one pending lane if any
   lane document is still unwritten, and the `MOJOBOOST_BUILD_LOCK` note until
   P1 lands. Any failure line names the manifest and the entry. This is the
   first execution of any of this lane's work, and it is the cheapest place for
   the first mistake to surface.

2. `python3 tools/validation_plan.py --list-tiers`
   Expect nine rows, `static` and `smoke` and `native` and `python` not opt-in,
   the other five opt-in.

3. `python3 tools/validation_plan.py --paths src/mojoboost/gain.mojo`
   Expect a short plan, not a broad one. If this prints dozens of suites, the
   subsystem map has drifted toward transitive reach and section 3.1 of
   `docs/FOCUSED_VALIDATION_PLAN.md` explains why that is the failure mode to
   watch for.

4. `python3 tools/validation_plan.py --coverage`
   Expect the list of modules no subsystem names and the list of suites no job
   runs. Both are notes, both are real gaps, and neither is a failure.

5. `python3 tools/validation_plan.py --all --format sh --out /tmp/plan.sh`
   Writes a script and runs nothing. Read it. Confirm the opt-in guard at the
   top, the per-job timeout, the status-file capture, the lock wrapper on
   exclusive jobs, and the `MOJOBOOST_BUILD_LOCK` export next to it.

6. Only then, and only deliberately, `MOJOBOOST_VALIDATION_OPT_IN=1 sh /tmp/plan.sh`
   on a plan you have read, on a machine you are willing to give up for the
   printed budget.

## What this lane did not do

It did not run the planner, any suite, any build, any benchmark, or any CI job.
It did not commit or stage anything. It did not edit any file outside
`tools/validation_plan.py`, `docs/FOCUSED_VALIDATION_PLAN.md`,
`validation/manifests/`, and this document.

It did not fabricate a timing, a provenance, or a baseline. Where a number was
needed for scheduling it is an estimate and is labeled as one. Where a file
would have had to come from running something, the file is absent and the job
that needs it fails.
