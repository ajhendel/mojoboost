# The focused validation plan

How a change in this repository turns into the smallest set of commands that
could catch what the change broke, and why the machinery that does it refuses
to run any of them.

## Status of this document

The planner, `tools/validation_plan.py`, and its four manifests under
`validation/manifests/` exist and are internally consistent. Nothing in this
lane has been executed. Not the planner, not a suite, not a build, not a
benchmark. Every duration in the manifests is a scheduling guess written by
someone reading source, and section 6 says what follows from that. Every
mapping from a changed file to a command was derived statically, by the method
in section 3, and section 3 also says what that method cannot see.

The map was revised on 2026-08-14 against a tree that had moved under it, and
because the planner still may not be run, that revision was a hand audit: the
same cross-references `--self-check` makes, made with `grep` and read by eye.
It found three things, all now fixed or recorded in section 7. A suite in the
test chain that no job named, which made the binning subsystem plan without its
own tests. A suite on disk that no task runs, which two separate checks were
each structurally unable to see. And 42 handoff entries pointing into a
directory a coordinator had deliberately deleted, which would have made the
first real `--self-check` run fail 42 times for no defect at all. The first of
those is now a `--self-check` failure in its own right, so the next one costs a
command instead of an audit. What that audit could not do is prove the planner
runs; see section 8.

This is not `docs/VALIDATION_CONTRACT.md`. That document is about validating
*user input* to the library. This one is about validating *the library* after
an edit. The two words collide and nothing else about them does.

## 1. The problem this solves

`pixi run test` chains all 47 Mojo suites with `&&`. Each `mojo run` compiles
its suite before running it, and compilation is most of the wall clock, so the
chain costs roughly 47 compilations. `pixi run -e pytest test-estimators`
collects 17 files and depends on `build-python`, which is another build.
`pixi run test-gpu` chains 8 more. On a laptop that is already hosting a
parallel editing session, running any of those out of habit is a decision to
stop working for several minutes, and it is usually made by reflex rather than
on purpose.

The working rule this project already had is the right one. After changing a
module, run the one suite that imports it. The problem with the rule is purely
mechanical. Nobody remembers which of the 47 suites imports
`src/mojotrees/gpu_split_search.mojo`, the answer changes every time a lane
lands, and looking it up by hand costs more attention than just running
everything. So people run everything, or they run nothing.

`tools/validation_plan.py` is the lookup, kept in a file so it stays current,
with the cost model and the refusals attached to it.

## 2. The nine tiers

A tier is a class of machine cost with a claim attached. The full text of each
one, including what it does not prove, is in `validation/manifests/tiers.toml`,
which is normative. This is the summary.

| Tier | Cost shape | Opt-in | Exclusion class |
| --- | --- | --- | --- |
| `static` | Reads files, exits. Standard library only. | no | none |
| `smoke` | One process, one small input, seconds. | no | none |
| `native` | One Mojo suite, compiled and run alone. | no | cpu |
| `python` | One pytest file, on a built extension. | no | cpu |
| `differential` | CPU against GPU, and CPU with the GPU disabled. | yes | gpu |
| `wheel` | Build an artifact, then install it somewhere hostile. | yes | build |
| `distributed` | Collective, wire protocol, Dask contract. | yes | cpu |
| `hardware` | Measurements filed against a named device record. | yes | gpu |
| `broad` | The full suites and the benchmark sweeps. | yes | cpu |

The first four run without asking. The last five require the operator to name
the tier with `--allow`, one at a time. There is no flag that turns all of them
on, and `--all` prints every job in every tier without scheduling any of the
opt-in ones.

Two distinctions in that table are load-bearing and easy to lose.

**`wheel` is not `python`.** `packaging/test_wheel.sh` proves the wheel works
inside the pixi environment that built it. `packaging/matrix/smoke/clean_install_*.sh`
refuses to run inside that environment at all, which is the only reason it can
prove the wheel is self-contained. A plan that runs the first and reports the
second is reporting something it did not check.

**`hardware` is not `broad`.** Both are slow. A broad job is slow because it
does a lot; a hardware job is slow because it is a measurement, and a
measurement taken while a broad suite was compiling on the same machine is a
measurement of both of them. The planner will not put a hardware job and a
broad job in the same plan, and `--measure` narrows a plan to a single
exclusion class for the same reason.

## 3. How a changed path becomes a command

This is the derivation method. It was run once, by hand, over the tree as it
stood when this lane wrote, and its output is `validation/manifests/subsystems.toml`.

Every file in `tests/` was read for its import block, specifically for lines of
the form

```mojo
from mojotrees.<module> import <names>
```

That gives a map from suite to the modules it names. Inverting it gives, for
each module in `src/mojotrees/`, the set of suites that import it directly.
That inverted map is the whole of the targeting signal. When
`src/mojotrees/goss.mojo` changes, the plan is the suites that name
`mojotrees.goss`, plus whatever `escalate` says for that subsystem, and nothing
else.

### 3.1 Why direct imports and not transitive reach

The obvious refinement is to close the import graph transitively, on the theory
that a suite importing `train.mojo` which imports `gain.mojo` is also a test of
`gain.mojo`. That was computed and then thrown away, because it destroys the
thing being built. Under transitive reach, `src/mojotrees/gain.mojo` is reached
by 34 of the 47 suites. A tool that answers "you changed gain.mojo, run 34
suites" has given the same answer as `pixi run test` while costing more to
maintain.

The honest reading is that transitive reach measures blast radius, not test
coverage. A suite that imports `train.mojo` asserts things about training. It
may fail when `gain.mojo` breaks, and it may just as easily pass, because
nothing in it was written to look at gain. Direct import is the closest static
proxy this repository offers for "somebody wrote assertions about this module",
and the planner uses it and says so rather than pretending to more.

### 3.2 What this method cannot see

- **Runtime dispatch.** A module selected through a registry, a function
  pointer, or a string parameter has no import edge, so a suite that exercises
  it through that seam does not appear in the map.
- **Behavior with no suite.** 18 modules under `src/mojotrees/` are imported by
  no suite at all. They are not silently absent. They are `[[gap]]` entries in
  `subsystems.toml`, each with a `reason`, a `would_close_it`, and
  `fallback = ["native:compile-package"]`, which is the honest floor. Compiling
  proves the module still compiles and nothing else. `tools/check_parity.py`
  records the same fact from the symbol side in `KNOWN_UNWIRED_TESTS`.
- **Non-Mojo change sets.** A change to `packaging/`, `.github/workflows/`,
  `docs/`, or `python/` is mapped by hand-written subsystem entries, not by an
  import graph, so those entries are only as current as the last person who
  read them.
- **Anything that landed after this lane wrote.** The map is a snapshot.
  `--self-check` catches a manifest that names a file or a task that no longer
  exists; it cannot catch a new module that nothing names yet, which is what
  `--coverage` reports as a note.

A gap may overlap a subsystem. When it does, the planner emits both, the
subsystem's jobs and the gap's warning, on the grounds that a partially covered
module is worse to be confident about than an uncovered one.

## 4. The four manifests

All four live in `validation/manifests/`, are TOML, and are read with
`tomllib`. There is no schema library, no plugin, and no code in them.

**`tiers.toml`** declares the nine tiers, the budget defaults, the mutual
exclusion classes, and the lock wrapper. Nine `[[tier]]` entries.

**`jobs.toml`** is the command table. 121 `[[job]]` entries, keyed `tier:name`,
each with a `command`, a `requires_files` list that `--self-check` verifies
exists, a `provenance` of `ci`, `documented`, or `unverified`, a `proves`
sentence, and where its output is filed. Distribution by tier is native 46,
python 18, static 14, differential 10, wheel 8, hardware 8, broad 8, smoke 5,
distributed 4. All 49 Mojo suites and all 17 pytest files are named by exactly
one job. Two jobs carry `blocked = true` and no command:
`distributed:two-process`, because `docs/DISTRIBUTED_TRANSPORT.md` section 1
states that nothing in this repository has moved a byte between two processes,
and `native:gpu-split-policy`, because no pixi task runs that suite. Both are
listed rather than omitted so the gap has an entry instead of a silence.

**`subsystems.toml`** is the map from section 3. 46 `[[subsystem]]` entries,
each with `paths`, `jobs`, and `escalate`, plus 6 `[[gap]]` entries.

**`handoffs.toml`** answers the other question a coordinator asks, which is not
"what did I change" but "I just applied the patches from handoff X, what now".
42 `[[handoff]]` entries, grouped where several handoffs share a job set, and 15
`[[lane]]` entries for the remaining-parity lanes. A lane whose file does not
exist yet is reported as pending, not as an error, because these lanes are
written in parallel and a missing file means "not yet", not "wrong".

The handoff directory those 42 entries name is gone. Commit `21ff9fa`, "Remove
obsolete handoff documents", deleted `handoffs/` after this manifest was
written, which under the original rule made `--self-check` emit 42 failures for
one decision somebody made on purpose. The entries are kept and an `[archive]`
table now declares the retirement, so a missing path under that prefix is
reported as retired, once, in a note. The reasoning is in the file's own header:
the mapping is the durable half and the memo was the transient one, so "you
applied the leaf-batching work, run these jobs" is still the right answer with
the memo deleted. An entry whose file comes back is checked again with no edit,
because the rule keys on the file being absent rather than on the entry being
disbelieved, and any `[[handoff]]` outside the retired prefix still fails.

## 5. Using the planner

The planner prints. The only thing `tools/validation_plan.py` executes, ever,
is one read-only `git --no-optional-locks status --porcelain` to discover the
change set, and passing `--paths` skips even that.

```
python3 tools/validation_plan.py                      # plan for the working tree
python3 tools/validation_plan.py --paths src/mojotrees/goss.mojo
python3 tools/validation_plan.py --since main
python3 tools/validation_plan.py --handoff remaining_12_validation
python3 tools/validation_plan.py --subsystem gpu-histogram --explain
```

Selection comes from `--paths`, `--changed-from`, `--since`, `--handoff`,
`--subsystem`, `--job`, `--tier`, or `--all`. Scope comes from `--allow` (one
opt-in tier at a time, repeatable or comma separated), `--escalate`,
`--budget-seconds`, `--max-jobs`, and the three environment refusals `--no-gpu`,
`--offline`, and `--no-sudo`, each of which holds the jobs that declare the
matching `needs` and says why it held them. `--measure` narrows a plan to one
exclusion class. Output is `--format text|json|sh`, optionally to `--out`, with
`--explain` adding what each job proves and where its evidence goes.

Three checks act on the manifests rather than on the tree.
`--self-check` validates every cross-reference, every `requires_files` path,
every `pixi run` task name against the tasks actually declared in `pixi.toml`,
every `MOJOTREES_*` variable against the documented env contract, and that every
Mojo suite the pixi task chain runs is named by some job, and exits 1 on any
problem. `--coverage` reports what the manifests do not name, as notes, never as
failures. `--list-tiers`, `--list-jobs`, and `--list-subsystems` dump the
tables.

That suite rule is the one check here that is a failure rather than a note, and
the asymmetry is deliberate. A new module nothing maps yet is normal; this tree
grows faster than any manifest in it, so `--coverage` says so and moves on. A
suite in `pixi.toml` is different, because putting it there is a statement that
it belongs to the checkable whole. A map that has not caught up with one is not
incomplete, it is wrong, and it is wrong in the direction that matters: it tells
somebody their change needs no test when a test for it exists and runs on every
push.

That is not hypothetical. `tests/test_binning.mojo` was added to the test chain,
and until it was found by hand it was named by no job, so a plan for a change to
`src/mojotrees/binning.mojo` returned `native:kernels`, `native:trainset`, and
`native:histogram-reference`, and not one of them is the binning suite. The rule
exists so the next one costs a failing `--self-check` instead of an audit.

### 5.1 The emitted script

`--format sh` writes a POSIX shell script. It is meant to be read before it is
run, and it is built so that running it by accident is hard.

- It refuses to start unless `MOJOTREES_VALIDATION_OPT_IN=1` is set in the
  environment. Nothing sets that anywhere in this repository.
- It sets `ulimit -c 0`, exports `MOJOTREES_NUM_WORKERS` for every tier except
  `static` and `smoke`, which read files and do not need threads, and applies a
  per-job timeout through `timeout(1)` or `gtimeout(1)`, whichever exists.
  macOS ships neither by default, and the script says so and continues without
  the timeout rather than failing.
- Every job with a non-empty exclusion class is wrapped in
  `tools/with_build_lock.sh`, so a plan run here serializes against a build
  started in another terminal. It also exports `MOJOTREES_BUILD_LOCK` with that
  class's own lock path, which the wrapper does not read yet; see section 7.
- Each job's exit status is captured through a status file rather than through
  `PIPESTATUS`, which is not POSIX. The tee'd log would otherwise report a
  failing suite as a pass. There is a comment in the emitted script saying so,
  because that bug is invisible and it comes back.
- Logs go to `${TMPDIR:-/tmp}/mojotrees-validation-<stamp>/`, one file per job,
  plus the status files.

### 5.2 The five mechanisms against a full-suite run

1. The five expensive tiers are `opt_in` and there is no flag that enables them
   together.
2. `max_broad_jobs = 1`. A plan with two broad suites is not a focused plan.
3. The budget stops adding jobs at `default_seconds`, and prints what it
   dropped, with the cost, so dropping it is visible.
4. `--measure` refuses to mix exclusion classes.
5. The emitted script will not start without an environment variable that
   nothing sets for you.

## 6. No number here is a measurement

Every `budget_seconds` in `jobs.toml`, every value under `[budget]` in
`tiers.toml`, and every duration in this document is a guess. This lane was
told not to run anything and did not run anything, so no suite has been timed,
no build has been timed, and no benchmark has been read for its wall clock.
The numbers were written by looking at what a job does and estimating.

They are good enough for their only job, which is ordering and truncation.
Relative cost is what the budget consumes, and a static file read being cheaper
than a wheel build is not in doubt. They are not good enough to quote. If a
number from these manifests appears in a report, a README, a benchmark table,
or a claim to anybody, it is being used for something it cannot support.

The same rule governs `provenance` on a job. `ci` means a workflow under
`.github/workflows/` runs that command, so it has demonstrably run somewhere.
`documented` means a document in this repository says the command is the right
one, and nothing in this lane confirmed it runs. `unverified` means neither. A
plan prints the provenance next to the command precisely so that a job which
has never run anywhere does not look the same as one that runs on every push.

Rerunning `--self-check` after any lane lands is the cheap way to find out
which of these guesses have gone stale into outright wrong.

## 7. Known holes this plan does not close

These were observed statically while building the map. Where the fix falls
inside this lane's ownership it was applied and is described as applied. Where
it does not, nothing was edited and there is a ready-to-apply patch in
`handoffs/remaining_14_validation_plan.md (deleted, recover with git log --all --diff-filter=D -- handoffs/remaining_14_validation_plan.md)`, identified below by its patch
number.

**The sparse tests skip themselves in the only environment that runs them.**
`python/tests/test_contrib.py:504` and `python/tests/test_validation.py:64`
both call `pytest.importorskip("scipy.sparse")`. `[feature.pytest.dependencies]`
in `pixi.toml` declares pytest, scikit-learn, pandas, pyarrow, and polars, and
does not declare scipy. `pixi run -e pytest test-estimators` is the only task
that collects those files. So the sparse paths report as skipped, the run
reports green, and the scipy conversion code in `python/mojotrees/_validation.py`
and `python/mojotrees/_arrow.py` has no covering test that ever executes. A
skip that nobody reads is indistinguishable from a pass. Patch P2.

**`pixi run test-c` is in no CI job.** `pixi.toml:35` defines it as
`capi/run_c_tests.sh`, which compiles `capi/test_capi.c` against
`capi/mojotrees.h`. Nothing under `.github/workflows/` invokes it, and the
script skips itself when no C compiler is present, so the C header's
compilability under an actual C compiler is unproven on every platform in the
matrix. The job exists in `jobs.toml` as `native:c-abi-from-c`, carrying
`needs = ["toolchain"]`, `provenance = "documented"` because only `pixi.toml`
attests to it, and that sentence in its `proves`. Patch P3.

**The two compatibility tools have nothing to compare against.**
`tools/api_snapshot.py --check` needs `compatibility/api_snapshot.json`, which
is not in the tree. `tools/model_fixture_manifest.py --check` needs
`compatibility/fixtures/checksums.json` and the fixture files themselves, and
`compatibility/fixtures/manifest.toml` carries `status = "declared"` with a
comment saying `--write` has not been run. Both tools are wired into `jobs.toml`
as static jobs with `provenance = "unverified"` so that the plan names them,
and both will fail until the owning lane generates their inputs. That is the
correct behavior for a check whose baseline is missing, and it is recorded here
so the first failure is not read as a regression.

**One lock file for three exclusion classes.** `tools/with_build_lock.sh` takes
a single machine-wide lock. A CPU job and a GPU job that could safely overlap do
not, and a plan is a sequence anyway, so the cost falls on other sessions. The
alternative was a second implementation of fcntl locking inside the planner,
which is the kind of duplicate that later disagrees with itself.

What was done inside ownership is the seam, not the fix. `[locks]` in
`tiers.toml` now carries `lock_env = "MOJOTREES_BUILD_LOCK"` and a
`[locks.class_lock_files]` table, and the emitted script exports that variable
with the job's class lock path before every exclusive job. The wrapper reads no
environment today, so the export is inert and all three classes keep sharing one
file, which is the current behavior. `--self-check` prints a note saying exactly
that, checked by reading the wrapper rather than by grepping the tree, so the
note disappears on its own the day the wrapper honors the variable and not a day
earlier. Patch P1 is the small change to the wrapper that reads it.

**18 modules are reached by no suite.** Listed as `[[gap]]` entries with
`native:compile-package` as the fallback. See section 3.2.

**One distributed job is blocked outright.** `distributed:two-process` has no
command because no two-process path exists. See section 4.

**`tests/test_gpu_split_policy.mojo` is run by nothing.** The suite is
on disk and no task in `pixi.toml` names it, so it has never run here or in CI.
`tools/check_parity.py` does not catch it either: its unwired-suite check reads
the suites cited in `docs/LIGHTGBM_PARITY.md`, nothing cites this file, and
`KNOWN_UNWIRED_TESTS` is empty, so an uncited suite that no task runs passes
both checks. It is `native:gpu-split-policy` here, `blocked = true`, with
`provenance = "unverified"` and a `proves` that says it proves nothing yet.
Giving it a command in this file was the wrong fix, because a job that invents
its own way to run a test is one the next person will not find from `pixi.toml`.
Patch P4 adds it to the test chain, after which the entry becomes ordinary.

**`MOJOTREES_BINNING_SELECT_MIN_ROWS` is not in the README env contract.**
`src/mojotrees/binning.mojo` reads it and both that module and
`src/mojotrees/parallel.mojo` document it in their docstrings, but the README
list that `--self-check` treats as the contract does not have it. It is
scheduling-only, in the sense that both paths it selects between resolve the
same edges, and `tests/test_binning.mojo` asserts that on both sides of the
threshold. No job here sets it, precisely because that suite already sweeps it
itself, so nothing in these manifests is broken by the omission. It is recorded
because the next lane to reach for a documented knob will not find it. Patch P5.

**The binning subsystem pointed at three suites, none of them the binning
suite.** Fixed here, and worth writing down because of how it was found rather
than what it was. `native:sparse` was promoted from an escalation into the
minimum for the binning subsystem at the same time, on evidence rather than
suspicion: `sparse.fit_bins_csc` reimplements the rank walk over an implied
dense column, and on 2026-08-14 a rule added to the dense binner and not to that
one made the two disagree. `test_sparse_binning_matches_dense_exactly` was the
only thing in the tree that noticed. A change to `binning.mojo` that does not
touch `sparse.mojo` is exactly the change that breaks that pair, so it cannot be
an escalation somebody has to remember to ask for.

## 8. What this lane did not do

It did not run the planner, any suite, any build, any benchmark, or any CI job.
It did not commit. It did not edit any file outside `tools/validation_plan.py`,
`docs/FOCUSED_VALIDATION_PLAN.md`, `validation/manifests/`, and
`handoffs/remaining_14_validation_plan.md (deleted, recover with git log --all --diff-filter=D -- handoffs/remaining_14_validation_plan.md)`.

That holds for the 2026-08-14 revision too, and it is the load-bearing caveat on
everything that revision claims. The audit behind it checked the manifests
against the tree; it did not check the planner against the Python interpreter.
`tools/validation_plan.py` gained a regular expression, a method, a helper, and
two branches in `self_check`, and not one line of that has been parsed by
anything but a reader. The new failure rule is the sharpest illustration of the
risk: it is asserted to pass on the current tree because the same set difference
was computed by hand with `comm`, on 48 suites named in `pixi.toml` against 49
named in `jobs.toml`, and a hand-checked set difference is evidence about the
manifests, not about whether the code that computes it runs.

So the first person to run `python3 tools/validation_plan.py --self-check` is
still the first person to execute any of this, and now also the first to find
out whether the file imports. That command reads files and exits, it is in the
`static` tier by its own classification, and its failures are the cheapest
possible evidence that this map has gone stale. If it raises rather than
reporting, that is this revision's bug and not a stale manifest.
