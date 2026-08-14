# Handoff: the focused validation planner

Lane: remaining-14. Owned files: `tools/validation_plan.py`,
`docs/FOCUSED_VALIDATION_PLAN.md`, `validation/manifests/`, and this file.

Nothing in this lane has been run. Not the planner, not `--self-check`, not a
suite, not a build, not a benchmark, not CI. Nothing was committed. No file
outside the four paths above was edited.

`docs/FOCUSED_VALIDATION_PLAN.md` is the document. It is normative for how the
planner behaves and what its numbers mean, and it is not repeated here. This
file carries only the things that need somebody else's hands: five patches to
files this lane does not own, and the standing caveat about what a lane that
runs nothing can and cannot claim.

## What this revision changed, in one paragraph

The manifests were audited against the tree by hand, with `grep`, because the
planner may not be run. Three defects were found. `tests/test_binning.mojo` was
in the pixi test chain and named by no job, so a plan for a change to
`src/mojoboost/binning.mojo` returned three suites and none of them was the
binning suite. `tests/parallel/test_gpu_split_policy.mojo` is on disk and run by
nothing, and two separate checks were each structurally unable to notice. And 42
`[[handoff]]` entries named files under `handoffs/`, which commit `21ff9fa`
deleted on purpose, so the first honest `--self-check` would have failed 42
times without a defect behind any of them. The first is now a `--self-check`
failure in its own right; the second is a `blocked` job and patch P4; the third
is an `[archive]` table and one note.

## The caveat, stated once and meant

A hand audit is evidence about the manifests. It is not evidence about the
planner. `tools/validation_plan.py` gained a regular expression, a method
(`Manifests.retired_path`), a helper (`pixi_suites`), an import, and two
branches in `self_check`, and none of it has been parsed by a Python
interpreter. The new failure rule is asserted to pass on the current tree
because the identical set difference was computed with `comm` over 48 suites
named in `pixi.toml` against 49 named in `jobs.toml`. That is a fact about two
files, not about whether the code that computes it imports.

So: if `python3 tools/validation_plan.py --self-check` raises rather than
reporting, that is this revision's bug and not a stale manifest. It is the
cheapest command in the repository and it is the right first thing to run.

## Ready-to-apply patches

Five, none of them inside this lane's ownership, all of them mechanical. Each
one states its effect on serialization and on the public API explicitly, and
both are "none" throughout: nothing here touches a model format, a binding, an
exported symbol, or an estimator parameter. Every validation step is UNRUN.

---

### P1. Give each exclusion class its own lock file

**Target** `tools/with_build_lock.sh`, the whole script.
**Owner** whoever owns `tools/` build scripts; created by the stabilization
gate lane.
**Dependency** none. `[locks].lock_env` and `[locks.class_lock_files]` in
`validation/manifests/tiers.toml` already exist and the emitted script already
exports the variable, so this patch is the only missing half.

**State flow today.** `--format sh` exports `MOJOBOOST_BUILD_LOCK` with the
job's class lock path before every exclusive job, and the wrapper does not read
it, so all three classes (`cpu`, `gpu`, `build`) take
`/tmp/mojoboost-build.lock`. The export is inert. That is the current behavior
and it is safe; the cost is that a CPU job and a GPU job which could overlap do
not.

**Patch.** Replace the hardcoded path with the variable, defaulted to the
current path so behavior is unchanged when nothing sets it:

```sh
#!/bin/sh
# Serialize heavy builds/tests across parallel sessions.
# macOS has no flock(1), so this blocks on a Python fcntl lock without polling.
# Usage: tools/with_build_lock.sh <command> [args...]
#
# MOJOBOOST_BUILD_LOCK picks the lock file. Unset means the single machine-wide
# lock this script has always taken, so a caller that does not set it keeps the
# old behavior exactly. tools/validation_plan.py sets it per exclusion class.
exec /usr/bin/python3 -c '
import fcntl, os, subprocess, sys
path = os.environ.get("MOJOBOOST_BUILD_LOCK") or "/tmp/mojoboost-build.lock"
f = open(path, "w")
fcntl.flock(f, fcntl.LOCK_EX)
sys.exit(subprocess.call(sys.argv[1:]))
' "$@"
```

**Errors.** An unwritable path raises `OSError` from `open` and the wrapped
command does not run, which is the correct failure: a lock that could not be
taken must not be reported as taken. Do not add a fallback to the shared path
on error, because that silently reintroduces the collision this patch removes.

**Fallback** if not applied: none needed. Everything keeps working, all classes
share one lock, and `--self-check` keeps printing its note.

**Serialization effect** none. **Public API effect** none.

**Validation, UNRUN.** `python3 tools/validation_plan.py --self-check` should
stop printing the `MOJOBOOST_BUILD_LOCK` note, because that note is produced by
reading this script's text for the variable name rather than by grepping the
tree. Its disappearance is the check.

---

### P2. Give the pytest environment a scipy so the sparse tests stop skipping

**Target** `pixi.toml`, `[feature.pytest.dependencies]`.
**Owner** packaging / pixi lane.
**Dependency** none.

**Problem.** `python/tests/test_contrib.py:504` and
`python/tests/test_validation.py:64` both call
`pytest.importorskip("scipy.sparse")`. `[feature.pytest.dependencies]` declares
pytest, scikit-learn, pandas, pyarrow, and polars, and no scipy.
`pixi run -e pytest test-estimators` is the only task that collects those files.
So the sparse paths skip, the run reports green, and the scipy conversion code
in `python/mojoboost/_validation.py` and `python/mojoboost/_arrow.py` has no
covering test that ever executes. A skip nobody reads is a pass.

**Patch.** Add one line to `[feature.pytest.dependencies]`:

```toml
scipy = "*"
```

**Errors.** None expected. If solving the environment conflicts, that conflict
is itself the finding and belongs in the packaging lane's notes rather than
being worked around by leaving the tests skipped.

**Fallback** if not applied: the two subsystem entries that escalate to
`python:validation` and `python:contrib` keep their note saying those
escalations skip themselves, so the plan does not overstate what they prove.

**Serialization effect** none. **Public API effect** none. Adding a test-only
dependency to one feature environment does not touch the wheel; the wheel is
built by the `pkg` environment and `test-python` deliberately stays
dependency-free so it also runs against a bare install.

**Validation, UNRUN.** `pixi run -e pytest test-estimators` and read the skip
count for those two files rather than the exit status.

---

### P3. Run the C ABI test somewhere

**Target** `.github/workflows/ci.yml`, the `test` job.
**Owner** CI lane.
**Dependency** none.

**Problem.** `pixi.toml:35` defines `test-c` as `capi/run_c_tests.sh`, which
compiles `capi/test_capi.c` against `capi/mojoboost.h`. Nothing under
`.github/workflows/` invokes it, and the script skips itself when no C compiler
is present. So whether the public C header compiles under an actual C compiler
is unproven on every platform in the matrix.

**Patch.** Add a step to the `test` job, after `Run tests`:

```yaml
      - name: C ABI from C
        run: pixi run test-c
```

The matrix is `ubuntu-latest` and `ubuntu-24.04-arm`, both of which have a
system C compiler, so the self-skip should not trigger. If it does, the script
printing its skip reason is the finding.

**Errors.** A compile failure here is a real defect in `capi/mojoboost.h` and
should fail the job.

**Fallback** if not applied: the job stays in `jobs.toml` as
`native:c-abi-from-c` with `needs = ["toolchain"]` and
`provenance = "documented"`, because only `pixi.toml` attests to it. A
coordinator can run it by hand.

**Serialization effect** none. **Public API effect** none directly, but this is
the only check that the C ABI header is valid C, so it guards the public C API
against a change that only Mojo would accept.

**Validation, UNRUN.** The step passes on both matrix runners, and
`provenance` for `native:c-abi-from-c` in `validation/manifests/jobs.toml` can
then be raised from `documented` to `ci`. That manifest edit is inside this
lane's ownership and should be made when the CI change lands, not before.

---

### P4. Put `test_gpu_split_policy` in the test chain

**Target** `pixi.toml`, the `test` task.
**Owner** pixi / test-registration lane.
**Dependency** none.

**Problem.** `tests/parallel/test_gpu_split_policy.mojo` exists and no pixi task
runs it. It has therefore never run here or in CI.
`tools/check_parity.py` does not catch it: its unwired-suite check reads suites
cited in `docs/LIGHTGBM_PARITY.md`, nothing cites this file, and
`KNOWN_UNWIRED_TESTS` is the empty set, so an uncited suite that no task runs
satisfies both sides of that check.

**Patch.** In the `test` task string, insert after the existing
`tests/parallel/test_gpu_split_search.mojo` clause:

```
 && mojo run -I src tests/parallel/test_gpu_split_policy.mojo
```

**Check first, because this is the part that needs a person.** Every other
`test_gpu_*` suite in `tests/parallel/` is already in the default `test` chain
rather than in `test-gpu`, which is the precedent this follows. But CI runners
are CPU-only Linux, and the repository's rule is that a module-level test
helper compiles unconditionally, so an unguarded helper that calls
device-kernel-launching code fails the GPU-arch constraint on a CPU-only runner
even when only comptime-guarded tests call it. Read this suite's helpers for
that pattern before adding it to the default chain. If any helper is unguarded,
the suite belongs in `test-gpu` instead, and the fix to the helper belongs to
whoever owns the suite.

**Errors.** If the suite fails on first run, that failure is information about
GPU split policy and not about this patch. Do not silence it by moving the
suite back out of the chain.

**Fallback** if not applied: the job stays `blocked = true` in `jobs.toml` as
`native:gpu-split-policy` with `provenance = "unverified"` and a `proves` that
says it proves nothing yet, so the gap has an entry instead of a silence.

**Serialization effect** none. **Public API effect** none.

**Validation, UNRUN.** `pixi run mojo run -I src tests/parallel/test_gpu_split_policy.mojo`
once, alone, before adding it to the chain. Then `--self-check`, which will stop
reporting it under `--coverage`. When it lands, the `blocked` entry in
`jobs.toml` becomes an ordinary native job with a command; that edit is inside
this lane's ownership.

---

### P5. Document `MOJOBOOST_BINNING_SELECT_MIN_ROWS` in the README

**Target** `README.md`, the parallelism bullet that currently reads
"`MOJOBOOST_NUM_WORKERS` and `MOJOBOOST_PARALLEL_MIN_OPS` pin the scheduler for
reproducible runs" (around line 270).
**Owner** README / docs lane.
**Dependency** none.

**Problem.** `src/mojoboost/binning.mojo` reads
`MOJOBOOST_BINNING_SELECT_MIN_ROWS`, and both that module and
`src/mojoboost/parallel.mojo` document it in their docstrings, but the README
list does not have it. `--self-check` validates every variable a job sets
against the corpus, and no job sets this one, so nothing here is broken by the
omission. It is recorded because the next person to look for a documented knob
will not find it, and because the README list is what the rest of the project
treats as the env contract.

**Patch.** Extend that sentence:

```
`MOJOBOOST_NUM_WORKERS` and `MOJOBOOST_PARALLEL_MIN_OPS` pin the scheduler for
reproducible runs, and `MOJOBOOST_BINNING_SELECT_MIN_ROWS` picks between the
two ways a quantile fit resolves its order statistics, rank selection or a full
sort; both resolve the same values, so it moves no edge and no bin
```

**Errors.** None. This is documentation of an existing knob.

**Fallback** if not applied: the variable keeps working and stays documented in
the two module docstrings.

**Serialization effect** none: the two paths this variable selects between
produce identical edges, so a model fitted under either serializes identically.
That is asserted by `tests/test_binning.mojo`, which fits every fixture both
ways and compares edge for edge. **Public API effect** none; it is an
environment variable, not a parameter.

**Validation, UNRUN.** None needed beyond `python3 tools/check_parity.py`,
which will read the changed README line.

---

## Not a patch: what this lane deliberately did not fix

**The two compatibility tools have no baseline.**
`tools/api_snapshot.py --check` needs `compatibility/api_snapshot.json`, which
is not in the tree. `tools/model_fixture_manifest.py --check` needs
`compatibility/fixtures/checksums.json` and the fixtures, and
`compatibility/fixtures/manifest.toml` carries `status = "declared"` with a
comment saying `--write` has not been run. Both are wired into `jobs.toml` as
static jobs with `provenance = "unverified"` so the plan names them, and both
will fail until the owning lane (remaining-13) generates their inputs. That is
correct behavior for a check whose baseline is missing. It is recorded so the
first failure is not read as a regression.

**18 modules are reached by no suite.** They are `[[gap]]` entries in
`subsystems.toml`, each with a `reason`, a `would_close_it`, and
`fallback = ["native:compile-package"]`, which is the honest floor: compiling
proves the module compiles and nothing else.

**`distributed:two-process` has no command,** because
`docs/DISTRIBUTED_TRANSPORT.md` section 1 states nothing here has moved a byte
between two processes.

## The one command to run first

```
python3 tools/validation_plan.py --self-check
```

It reads files and exits. It is in the `static` tier by its own classification.
It is the first execution of anything in this lane, and its output is the
cheapest evidence available about how much of the above survived contact.
