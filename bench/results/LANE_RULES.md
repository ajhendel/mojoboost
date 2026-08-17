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

**3. Moves bits: TAKE THE REAL-DATA GATE.** `bench/real_data`, before the
change is believed.

**The gate is OUR OWN accuracy, and it stopped being a peer comparison on
2026-08-17.** The old rule was "within 1 percent relative of the better of
CatBoost as shipped and LightGBM stock+det". It was replaced because it fails
in BOTH directions, and the second one is why it was unsafe rather than merely
awkward.

- It blocked changes that cost nothing. The leaf-wise arm improved from 1.043 s
  to 0.839 s that day on bit-identical work, and the frontier table would not
  rank the arm at all, because it sits 1.42 percent behind CatBoost.
- **It permitted real accuracy loss whenever we were ahead.** The symmetric arm
  beats CatBoost at 799k, 0.303271 against 0.303468, so under a peer-anchored
  1 percent bar it could give away about 1.07 percent of its OWN accuracy and
  still pass. A rule that lets you lose accuracy because a competitor is weak
  is not an accuracy rule.
- The bar moved when somebody else shipped. A CatBoost release could fail our
  gate with no change to our code.

So: **our own accuracy is the gate, and competitors are the scoreboard.** A
change that moves bits is judged by what it costs US, on the same arm and the
same scenario, against a recorded reference. Peer numbers are reported beside
every result, always, and gate nothing. We still want to know we are 1.42
percent behind CatBoost. That is a column, not a verdict.

**The reference is an ABSOLUTE ANCHOR, not the previous run**, and that
distinction is the whole design rather than a detail. Anchoring on the previous
run gives a ratchet: lose 0.9 percent ten times, pass every time, end up nine
percent worse with no gate ever firing. The anchor is recorded per arm and per
scenario, it lives in a file, and it moves only by a deliberate act that shows
up in a diff, so drift accumulates against a fixed point.

Two consequences worth stating because they are easy to get wrong. A
bit-identical change costs zero of our accuracy by construction, so it passes
this gate trivially and rule 5 flips its default on measurement. And the
tolerance here is a DIFFERENT quantity from the old peer-relative 1 percent,
so reusing that number without an argument for it would be a mistake.

**4. An item closes ONLY when proven zero or proven impossible**, with the
evidence recorded. **Nothing is dropped for being small.** "Under one percent"
is not a reason to skip a category-1 change; it is a reason to rank it lower
and still do it.

**5. A SWITCH IS A TEMPORARY STATE, NOT A RESTING PLACE.** Added 2026-08-17
after this rule set was audited against what it actually produced. Rule 2 says
a trade is A/B'd in the window "before it becomes a default", and nothing in
this document ever made that second half happen, so the tree accumulated
measured, proven wins that shipped to nobody. On the day this rule was written
the library carried a 4.5 percent scan rewrite, an exact-integer sibling
subtraction worth 1.78x, a skipped last-level build and a hoisted noise copy,
all measured, all bit-identical, all default off.

So: **when an A/B resolves faster and identity holds, the default flips in the
same session as the measurement.** The switch then inverts, surviving one round
as an escape hatch to turn the new behavior OFF, and is deleted after that. A
switch that outlives a positive measurement is a defect and is reported as one.

The reasoning is worth stating because it removes the thing that felt like
caution. **A bit-identical change cannot alter any user's output.** Not the
model, not a prediction, not a digest. Flipping its default changes the clock
and nothing else, so there is no risk to weigh, and "measured faster, identity
proven, default off" is not a conservative position, it is an unshipped one.
For a change that MOVES bits, rule 3 stands unchanged and the accuracy budget
is the gate, because that one is a real trade.

**6. A DECLINE MUST CARRY A PRICE.** Also 2026-08-17, and this is the rule that
would have caught the biggest defect in the codebase. A comment that declines an
optimization must state what the decline costs, with the arithmetic, or be
labeled as ASSERTED. **An asserted decline is an open item under rule 4, not a
closed one**, however confidently it is written. Two declines were falsified on
one day: the oblivious level batch declined sibling subtraction to save about
1.1 ms of launches while paying about 125 ms of doubled traffic, an error of
roughly a hundred to one, and it framed launch count and subtraction as
exclusive when a fused subtraction costing no launch already existed thirty
lines away. Separately, the last level's histograms were built and never read
because skipping them was said to move a copy-back cost, which confused command
buffers with work. Both read as settled engineering. **When you decline
something, price it, and when you find a decline you cannot price, that is a
finding and not a footnote.**

**8. THE IDENTITY BUG: WHEN NOTHING FAILS, SUSPECT THE KEY.** Added 2026-08-17,
after a single day produced eleven instances of one defect. Every one was
internally consistent code with a wrong IDENTITY, and not one of them failed a
test. They come in two mirrored forms.

**An identity carrying a dimension it should not.** `frontier.py` put the device
inside the arm id, so a cpu cell and its gpu cell were different arms and a
351-job run produced ZERO oracle cells while every accelerator row reported "no
cpu twin". `verify.py` keyed a cell on the engine rather than the arm, so forty
arms collapsed into one verdict comparing whichever two records were written
last, at different tree counts. Four device and host tables were sized from
`num_leaves`, which does not bind under oblivious growth, so a depth-6 symmetric
tree met a table built for 31 leaves; two of those raised and two under-reserved
in silence.

**An identity missing a dimension it needs.** A role test compared an
ENGINE-name list against an arm id, matched nothing, and emitted no lines at
all, so the accuracy scoreboard simply vanished from every `--arms` run. An arm
recorded without the parameters that make it that arm is the same shape, which
is why resolved parameters travel in the record.

The rule that follows. **A gate that emits nothing is indistinguishable from a
gate that passes**, so silence is a finding and not a clean bill. When a check
is green, confirm it compared the things you meant, on the rows you meant, at
the identity you meant. `verify.py::check_coverage` is the mechanical form of
this: every subject cell must be NAMED by at least one gate, and a cell no gate
mentions is a WARN. Prefer that shape wherever a gate can be given one.

And when you find one instance, sweep for the rest, because this defect breeds.
Of the eleven, one was found by looking and ten were found by asking where else
the same identity was assembled.

**7. A STALE CLAIM IS A DEFECT, AND FINDING ONE OBLIGATES YOU TO FIX IT.**
Added 2026-08-17 (Andrew: "we need to always update if we find something
wrong"). This codebase documents itself heavily, which is a strength and is
exactly why a wrong comment is expensive: it reads as settled engineering and
gets believed instead of checked. Several of the largest wins of that day came
from falsifying a confident comment, and several near-misses came from
believing one.

So: **if you find a claim that is wrong, correcting it is part of your lane, not
a note for somebody else.** If the claim is in a file you own, fix it in the
same pass. If it is in a file another lane owns, quote the exact replacement
text in your report so the orchestrator can apply it verbatim, and say plainly
that it is a CORRECTION rather than a suggestion, so it is not triaged as an
improvement and deferred. A corrected claim records what it used to say and
when it changed, rather than quietly reading as though it always said the new
thing, because the next reader needs to know a belief moved.

This applies to measured RESULTS as much as to prose. A recorded number whose
arm never reached the code is not a small number, it is a NULL, and leaving it
filed as a result means it will be believed again. On the day this rule was
written, `MOJOTREES_CPU_LAYOUT_BY_NODE` was recorded as measuring "neutral" on
the symmetric CPU grower, and the switch had never reached that grower at all.
The conclusion drawn from it, that per-node layout is not where the symmetric
CPU cost lives, was unsupported for weeks.

**VERIFY BEFORE YOU CORRECT, because a correction applied on a stale premise
installs a new false claim.** An audit lane that day was handed four confident
corrections and found one of them already fixed in the working tree by another
lane hours earlier. Applying it would have re-asserted a condition the code no
longer has, and would have narrowed a finding that in fact still stands. A
correction is a claim like any other and takes the same standard of proof.

**CITE BY NAME, NOT BY LINE NUMBER.** This is the single most common form of rot
in this repository and it is a mechanical consequence of how we work: many lanes
edit the same files on the same day, so a line number is stale within hours
while the substance stays true. Every drifted citation found in that audit had
correct content and a wrong pointer, which is the worst combination because the
reader concludes the claim is wrong rather than the pointer. So anchor on
`file.function`, or quote the line of code you mean, and keep line numbers only
as a convenience beside a durable anchor. A document whose numbers have already
drifted should say so at the top rather than be half renumbered, because a
partly refreshed table is less trustworthy than a uniformly stale one.

The CPU path is the correctness ORACLE and the portability floor. It is not
optimized (2026-08-17 ruling). `device_agreement` and `backend_proof` in
`bench/real_data/verify.py` are built on it actually running, and on
2026-08-17 `device_agreement` caught a live noise-hash divergence in the
shipped defaults, so it earns its keep as a gate rather than as a product
claim. It is also genuinely faster than the device below roughly 150,000 rows,
and it is the only backend our CI can execute at all. What changed is where
engineering effort goes, not whether the path exists: speed lanes are GPU
lanes, and a CPU speed lane needs a reason beyond the number being improvable.

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
- **Run `source tools/lane_env.sh` once, before anything else.** A worktree
  contains `pixi.toml`, so a bare `pixi run` treats it as its own project and
  installs a second complete copy of the environment into `<worktree>/.pixi`,
  about 1.1 GB, before it compiles a line. It also gives you your own empty
  Mojo compile cache, because `MODULAR_HOME` follows the environment.
  Measured 2026-08-16: 46 lane worktrees had done this, **49 GB** of
  duplicated environments, with caches of 1.1 MB to 243 MB against the main
  checkout's 8.7 GB. Every lane was starting cold and sharing nothing.
  Sourcing that file points `PATH` and `MODULAR_HOME` at the main checkout's
  environment, copies nothing, and declines with a message if your
  `pixi.toml` differs from the main one. `tools/run_tests.sh` already does it
  for you; the commands that need you to do it yourself are `pixi run
  build-pkg` and any bare `mojo build`.

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
