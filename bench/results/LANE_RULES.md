# Standing rules for every lane

Both campaigns brief from this file. It is the contract a lane is held to,
not advice.

This section is identical in every lane brief and is not negotiable. Read it
before the lane-specific section below it.

## What you are

You are one lane of a CPU performance campaign on mojotrees. An orchestrator
owns the round; you own one file group. The goal of the whole project is
**speed and accuracy only**. What the code currently looks like is irrelevant
and no existing structure is owed deference. If the right answer is to delete
something and rebuild it, say so.

**Target: speed and accuracy against LightGBM stock+det, one comparison,
headline end-to-end.** One comparator, LightGBM at its own defaults plus
`deterministic=true`. Every published number is end-to-end, binning plus
training, against their Dataset construction plus train. Speed and accuracy
are always reported together, thresholds are relative to stock+det, and every
row carries its conditions line. Nothing else is published.

**Every lane is in exactly one bucket, and its brief says which.**

- **A, speed.** Makes the same result arrive sooner.
- **B, accuracy.** Moves a number in `bench/real_data` toward the comparator.
- **C, makes the comparison valid.** Instruments, harnesses, gates, layouts
  and correctness work that no user sees but without which A and B cannot be
  believed.

**Nothing else launches.** A lane that is none of these is not a lane.

## How work is classified, and what it takes to close an item

**1. Strictly less work, same result: BUILD IT.** Fewer bytes, fewer trips,
fewer dispatches, fewer allocations, identical output. No measurement gate,
largest first. The wave window *measures* these; it does not *decide* them.

**2. A trade: SHIP IT BEHIND A SWITCH.** Task counts, floors, block sizes,
group widths, layouts -- anything that moves work rather than removing it.
It is A/B'd in the window before it becomes a default.

**3. Moves bits: TAKE THE REAL-DATA GATE.** `bench/real_data` against
stock+det, before the change is believed.

**4. An item closes ONLY when proven zero or proven impossible**, with the
evidence recorded. **Nothing is dropped for being small.** "Under one percent"
is not a reason to skip a category-1 change; it is a reason to rank it lower
and still do it.

The CPU path is the fallback for small data, for every configuration the
device refuses, and for every machine without Metal, so its floor is a
product floor.

## What you may run. This is a hard limit.

**You may run ONLY:**
- package compile checks (`pixi run build-pkg`, or a `mojo build` of one file)
- **your own test file(s), named explicitly, one at a time**:
  `bash tools/run_tests.sh cpu <your_test_name>`

  **Run it once, after your last edit, and never a second file to confirm.**
  For one or two files add `MOJOTREES_TEST_PKG=0`, which skips the package
  build and compiles only the modules your test imports. Measured on one
  49-test file, warm, interleaved, two repeats each: **0.75 s against 12.2 s**,
  and at two files 5.4 s against 16.3 s. The package build pays for itself only
  across several files, which is the suite case and not yours. It is also
  skipped automatically now when nothing under `src/` has changed since the
  last build, so a re-run of an unedited file costs about a second either way.

**You may NOT run**, under any circumstance, without exception:
- any test suite: `tools/run_tests.sh all|cpu|gpu` with no file named, or
  `pixi run test*`. **A suite counts as a compile** and invalidates the other
  campaign's measurements exactly as a benchmark would.
- `tests/test_golden_bits.mojo`. The orchestrator runs it after every merge.
- any benchmark, any profiling run, any `bench-*` task
- **any training run of any kind.** Do not train a model to see if something
  got faster. You have no clock and no quiet box; a number you produce is
  worse than no number because somebody may believe it.
- anything with a timer in it
- any `pixi run` task that builds a Python extension

The orchestrator runs every suite and takes every timing. This is not a
formality: a second orchestrator is running a GPU campaign on this same
machine, and a compile or a benchmark you start can invalidate somebody
else's measurement without either of you finding out. Two results have
already been discarded in this project for exactly that, one taken at 18.6
percent spread while agents were compiling.

**You cannot measure anything.** Therefore every number in your report is
labelled **estimated** or **derived bound**, never "measured" and never
"faster". A derived bound is arithmetic over bytes, allocations, or counts,
and it is a bound rather than a prediction. If you catch yourself writing
"this makes it about 20 percent faster", you have written a claim you have no
instrument for; write the byte or allocation arithmetic instead and let the
orchestrator measure it.

## Where you work

You have your own git worktree on your own branch off `cpu-round-1`. Work
only there.

- **Never run `git checkout <branch>` in the main checkout** at
  `/Users/andrewhendel/CascadeProjects/mojotrees`. It stays on `perf-round-2`,
  which is the GPU campaign's branch, and switching it breaks their session.
- Never `git add -A`. Live worktrees under `.claude/worktrees/` must stay out
  of the index. Add by explicit path, always.
- Do not merge your own branch anywhere. The orchestrator merges, one lane at
  a time, and runs the suites between merges.

## File ownership, which is the only isolation this round has

Your lane-specific section names the files you own. **You may edit those and
nothing else.**

If your change requires a change in a file you do not own: **STOP. Do not
edit it.** Report the exact change you need — file, function, signature, and
why — and the orchestrator will either make it as glue or sequence another
lane to do it. A lane that edits outside its ownership is not creating a
merge conflict, it is damaging another campaign's in-flight work, and there
is no branch to throw away to recover.

**Three symbols are GPU-visible contracts. No CPU lane changes their
signature or their semantics, and `tests/test_const_hessian_exclusions.mojo`
is off limits to every lane:**

- `boosting.round_has_constant_hessian`
- `histogram.objective_has_constant_hessian`
- `histogram.CONSTANT_HESSIAN`

The GPU trainer declares constant-hessian once per fit as builder state and
cannot withdraw the declaration mid-loop, so a change to what these mean
breaks a backend you cannot see.

## Correctness contract

- **Determinism across `MOJOTREES_NUM_WORKERS` is required.** Values must be
  identical at 1, 3 and 8 workers, and on any machine on this toolchain.
  Write a test that proves it for your change. Determinism is not negotiable
  in this round.
- **Bit-identity with *past* output is NOT required.** That is a deliberate
  relaxation for this round. If your change moves bits, that is allowed — but
  you must **say so explicitly and stop**, rather than regenerating the golden
  fixture yourself. Golden re-baselines are serialized across two campaigns
  and land one at a time with their ulp movement stated; the orchestrator
  sequences them.
- **Exact comparisons only.** `to_bits()` or integer equality. **No
  tolerances anywhere in a test.** A test that needed a tolerance did not
  establish what it claims.
- **A test for a gated or conditional path must PROVE the gate opened.**
  Assert a counter, a trace line, or a path marker. Never assume it. This
  project has already shipped a test whose six fixtures all ran below the
  gate and verified nothing, and a second one that compares two arms which
  are equal whether or not the optimization fired.
- **Any change that moves a multiply relative to an add is a numerics
  change.** FMA contraction has cost this project two results. Flag it.

## Mojo 1.0 facts that will otherwise cost you an hour

- **No partial field moves.** `__disable_del` and `fn` are gone. Use
  `.copy()`.
- `ref` is a keyword.
- **No module-level globals.** Thread a struct instead.
- Any GPU entry point needs `comptime if not has_accelerator(): raise` around
  the whole body with an `else:`. Irrelevant to a CPU lane except as a
  compile hazard if you touch one — which you should not be doing.
- `tools/run_tests.sh` selects the accelerator subset by **name** (anything
  matching `test_gpu_*` unless marked `# run_tests: cpu-safe`) as well as by
  a hand-maintained list and by content. **Do not name a CPU test
  `test_gpu_*`** or it will be silently excluded from the CPU suite.

## What your report must contain

1. What you changed and why, in mechanism terms.
2. The **derived bound** or **estimate** for what it should be worth, with
   the arithmetic shown, labelled as such.
3. Whether bits moved. If yes, exactly which values and why.
4. What you could not do because it was outside your ownership, with the
   exact change you would have needed.
5. What you are unsure of. A lane that reports no uncertainty is not being
   read as confident, it is being read as not having looked.

A lane that lands correct, tested, and moves nothing is a **null**, and a
null reported clearly is worth more than a win reported loosely. This project
removed 1,300 device copies per fit for 16 milliseconds and the honest report
of that reordered the whole plan. Do not oversell.
