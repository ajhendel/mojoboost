# The machine lock, and the two things it is not about

Agreed 2026-08-16 between the GPU campaign orchestrator and the CPU campaign
orchestrator, both working this repository concurrently, and ratified by Andrew.
This file exists in its own right rather than as a section of
`PROFILE_PROTOCOL.md` because three sessions are writing to that document at
once and a coordination rule that causes merge conflicts is not much of a
coordination rule.

## Two problems that look like one

**Code in parallel is a working-tree problem, and it is already solved.**
Separate branches, separate worktrees, agreed file ownership. Nothing about it
needs a lock.

**Timing in parallel is a machine problem, and it cannot be solved by any of
that.** A Mojo compile running *anywhere* on this box makes a benchmark number
garbage. So does accumulated heat. Neither is affected by which directory the
compile happens in.

Conflating them is what led to the lock being dropped once and reinstated in a
narrower form the same night.

## The evidence that this is not a theoretical concern

- Two results were discarded this week for being taken while lanes compiled, one
  of them at 18.6 percent spread with four agents building.
- The row-unroll A/B, **same command, same shape, same code**, measured
  *indistinguishable* in a busy window (8.1 percent delta against a 14.1 percent
  noise floor) and *resolved* in a quiet one (10.8 percent against 2.1). The
  verdict flipped on machine state alone. A null taken in a bad window is not
  evidence of absence.
- LightGBM drifted from 2.80 to 3.50 seconds across five back-to-back runs from
  heat alone, while the GPU arm held 2.7 percent spread over the same sequence.
  Ten CPU cores throttle harder than the GPU does, so a CPU-versus-CPU campaign
  is more exposed to this than a GPU one.

## The rule

**Working tree**

- The main checkout stays on `perf-round-2` and **nobody runs `git checkout
  <branch>` in it**, ever.
- Each campaign has its own integration branch: `perf-round-2` for GPU,
  `cpu-round-1` for CPU.
- Every lane works in **its own worktree**. Worktrees live under
  `.claude/worktrees/` and are gitignored; they are real working copies of this
  repository, never content of it. Six of them were committed as embedded
  gitlinks once by a `git add -A`; that is what the ignore rule prevents.
- File ownership is agreed between orchestrators before lanes launch, and a lane
  that believes it must edit outside its ownership **stops and reports** instead.

**Timing**

- `/tmp/mojotrees-bench.lock`, `mode: timing` **only**. There is no `mode:
  lanes`. Compiles proceed freely at all other times, which is almost all of the
  time.
- Short, exclusive, announced. The holder writes their session name (the one the
  other session can actually address via `ListAgents`, not an internal id), what
  they are timing, and an ETA.
- Both parties check it before starting any timing run. Neither times against a
  lock they do not hold.
- The holder deletes it when done. A lock whose ETA has long passed may be
  queried by message before being assumed stale.
- Whoever wants the box for timing asks; the other stops launching new lanes and
  reports when in-flight compiles have drained. **A compile cannot be killed
  mid-flight, so there is a drain interval and it gets reported rather than
  papered over.**
- **A test suite counts as a compile.** Neither campaign runs a suite during the
  other's announced timing window, and neither times while running its own. This
  is not a new lock or a new message, it is the existing rule saying what already
  followed from it: `tools/run_tests.sh` builds every test file and has run at
  533 to 874 percent CPU for four to eleven minutes on this machine. It is the
  single heaviest thing either campaign does, heavier than most lanes, and it was
  not obviously covered by a rule phrased around "lanes".
- The same applies to a session's own commits when the other holds the box: the
  pre-commit hook runs the gate scripts. Small, but free to defer.
- Every results header records whether the box was verifiably quiet, per M1.

## The instrument problem, stated plainly

`PROFILE_PROTOCOL.md` instructs every session to "capture thermal state into the
results header with `bench/apple/thermal_capture.sh`". **That instruction has
never been followable. One session followed it anyway, and the result is filed
as though it were a measurement.**

The CPU orchestrator read the script on 2026-08-16 and reports that it measures
nothing: it is a plan printer, its own header says it "starts no sampler, runs no
privileged command, fits no model, and writes no record", and `--execute` is
parsed and deliberately refused with exit code 3 because the plan it prints
includes `sudo powermetrics`.

This paragraph originally added "and no session has ever followed it". The
instruction audit refuted that from a committed artifact:
`bench/results/profile_2026-08-15/header.txt` has a `=== thermal before ===`
section holding the script's plan output, `run id thermal-PENDING` and all, and
it has been there since `24e5330` on 2026-08-15. Read every regime label in
that results directory accordingly.

**The pointer to the real commands was also dangling.** This paragraph used to
end by sending the reader to `handoffs/performance_17_thermal_energy.md`. That
file was deleted on 2026-08-14 in `21ff9fa`, two days before this document was
written; `bench/apple/thermal_capture.sh --self-check` fails on its absence with
exit code 4. Until it is restored, use `docs/APPLE_THERMAL_ENERGY.md`, or
recover the original with

    git show 21ff9fa^:handoffs/performance_17_thermal_energy.md

The commands are reproduced verbatim in
`bench/results/INSTRUCTION_AUDIT.md` section 10, with the privileged ones
marked. Do not run those from a session.

`pmset -g therm` was tried on this machine and returned "No thermal warning level
has been recorded", which is the expected answer on Apple silicon.

So the best instrument either campaign currently has is:

    uptime
    ps -Ao pcpu,comm | sort -rn | head

captured before and after every arm, into the results file. That is what both
orchestrators have agreed to use, and the instruction audit found the agreement
ahead of the practice: exactly one results directory records `uptime` at both
ends (`apple_m4_unified_memory_2026-08-15/env_start.txt` and `env_end.txt`), and
**no results file anywhere records the `ps` line at all.** It is two seconds of
work and it is still being skipped. Everything either of us knows about this machine's fast
and slow regimes was inferred **from effect** — both arms rising together by 65
percent — and not read from any instrument. One attribution was made and
retracted on that basis (a slow pair was blamed on `mobileassetd` at 100 percent
CPU; after waiting it out and confirming the box idle the numbers stayed high, so
it was regime, not contention).

Until someone runs the privileged commands by hand, "fast window" and "slow
window" in this repository's results are defined by their effect on the numbers
and not by a reading. Every result labelled by regime should be read with that in
mind.

## The gap in the narrow lock, measured

Holding `mode: timing` does not make the box quiet. It stops the other session
*timing*; it does not stop them compiling, by design.

**Measured by the CPU campaign, 2026-08-16, while it held the lock and the GPU
campaign compiled six lanes:** the same configuration measured 6.115, then
9.655, then 13.047 seconds across three pairs twelve minutes apart, with load
climbing 4.6 to 6.3. A **2.1x degradation of the arm under an uncontested
timing lock.**

So the lock is necessary and not sufficient. The patch, agreed between both
campaigns and preferred to reinstating a lanes mode: **a courtesy pause on
request.** Whoever needs a genuinely clean window asks, names the length, and
the other launches nothing and reports when in-flight compiles have drained. A
compile cannot be killed mid-flight, so the drain interval is real and gets
reported rather than papered over.

And the corollary, which is cheaper than any protocol: **do not take a
measurement while your own lanes are running either.** Both campaigns have now
lost numbers to their own builds, not just to each other's.

### The salvageable half, which is the strongest argument for pairing

Across that 58 percent shift in level, the **within-pair ratio held**: +17.2
percent and +15.7 percent. Adjacent pairing survives a regime change; absolute
levels do not.

That is the in-process interleaving rule arrived at from the failure side, and
it is why the rule below is stated as a preference in the protocol and should be
read as close to mandatory. A ratio taken from adjacent arms is worth having
even from a contaminated window. An absolute second from the same window is not.

### Superseded in part by the canary, and only in part

**Partly superseded, and only partly.** `bench/canary.mojo` is now a reading:
two fixed probes, one CPU and one GPU, run first and last in every
`bench_train_gpu.mojo` run and reported as a ratio against a recorded baseline
(protocol amendment A5, and `bench/README.md`, "The regime canary"). It needs no
privileged command, because it does not ask the machine what its thermal state
is -- it asks what the machine delivers, which is the quantity that actually
matters here. What it still cannot do is name a *cause*, so the `uptime` and
top-process capture above stays: that one says what was running, the canary says
whether it mattered. **And the canary's baselines have not been measured yet**,
so as of this writing every regime label in the tree is still an inference from
effect. Establishing them is
`pixi run mojo run -I src bench/bench_canary.mojo 7` on a box quiet by the
standard this file sets, and until that has happened nothing above is stale.

## What to do when you could not get a quiet box

Do not silently publish. Two options, in order of preference:

1. **Switch to in-process interleaved arms.** An alternating-*process* A/B breaks
   under contention, because arm A runs during someone's build and arm B runs
   after it, and the difference is the build. An in-process A/B samples both arms
   repeatedly inside one window, so contention and drift hit both alike. The
   harness supports this (`bench-train-gpu rows feats obj N arm1,arm2`) and it is
   the right default regardless of the lock.
2. **Label it.** "Box not verifiably quiet" in the header, and no promotion of
   the number to a summary table until retaken.

## Identifiers get read back before they are sent, and here is the mechanism

Both orchestrators fabricated a commit SHA on 2026-08-16, within an hour of each
other, and one of them did it in the message immediately after warning the other
about it. That makes it a process fault rather than a slip, and the mechanism is
specific enough to fix:

**Composing a message in the same step that produces the commit.** The SHA is
predicted from what the commit is about to be rather than read from what it
turned out to be. It looks right, it is the right length, and it is wrong.

The rule: **produce the commit, read the identifier back, then send.** Never in
one step. `git cat-file -t <sha>` and, for a merge,
`git merge-base --is-ancestor <branch> HEAD` cost nothing and settle it.

This generalizes past SHAs. Every fabricated number this week -- the "0.4s ideal
roofline", the "3.2 us/row" slope, both of which existed in no file and no commit
-- came from the same act of stating a value in the same breath as the reasoning
that motivated it, rather than reading it from the thing it describes.

## A merge that is not only a merge cannot be bisected through

Recorded because it will be met by whoever rebases. On `cpu-round-1`, commit
`201218f` is simultaneously the `perf-round-2` merge, an orphan-module deletion
and the `device='auto'` patches. The conflicted merge could not be committed
while the pre-commit gate refused a stale generated artifact, so `MERGE_HEAD`
stayed live and the next commit folded everything together.

The message names all three, so nothing is hidden. But a reviewer expecting a
clean merge commit will not find one, and **a bisect landing on that commit gets
three changes at once**. Not worth rewriting history on a shared branch; worth
knowing before rebasing onto it, and worth avoiding next time by committing the
merge alone before anything else touches the tree.

## Cost

None in throughput. Timing windows are minutes; lanes run for hours. The lock
costs one message and buys the only thing that makes any of the numbers
believable.
