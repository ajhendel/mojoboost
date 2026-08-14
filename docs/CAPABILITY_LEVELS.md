# Capability levels

Vocabulary version: 1
Written: 2026-08-14

One word, "supported", has to carry too much. A capability can be written
and never called, called and never reachable, reachable and never tested,
tested on one machine and claimed on every machine, or true in the source
tree and absent from the wheel a user installs. Collapsing those into a
single yes or no is how a status table drifts away from the repository
without anyone lying.

This file defines seven independent levels. `docs/LIGHTGBM_PARITY.md`
scores the contested capabilities against them, `README.md` and release
notes cite them by name, and `tools/check_parity.py` checks that the
contract's level table uses these names and no others.

## The seven levels

| Level | A capability has it when | How it is proved | What it does not imply |
|---|---|---|---|
| implemented | Code exists in this repository that performs the behavior | A named module, struct, or function | That anything calls it |
| integrated | A shipping code path calls it, so some entry point behaves differently because it exists | A caller outside the module's own tests and outside `tests/` | That a user can ask for it by name |
| publicly reachable | A user can invoke it through a surface section 2 of `docs/COMPATIBILITY_POLICY.md` lists as public | The public name, and the file it is exported from | That it is correct, or fast |
| focused-tested | A test in this repository exercises this behavior specifically, and a pixi task CI runs executes that test | The test file, and the task that runs it | That it agrees with LightGBM, or with anything outside this repository |
| differential-tested | Its output is checked against an independent implementation: LightGBM, scikit-learn, or an arithmetic reference computed by hand | The comparison script or the reference test | That the comparison runs automatically |
| hardware-validated | It has run on the physical device class the claim is about, and a record of that run is in this repository | The record, with a date and a device | That it ran on any other device class |
| release-packaged | It is present in the artifact a user installs, and something checks that it is | The packaging manifest or the artifact validator | That the artifact has been published |

## Rules

**The levels are not a ladder.** A capability can be focused-tested and
not integrated (an isolated module with its own passing suite), or
integrated and not focused-tested (a call site nothing exercises
directly). Two implications do hold, because they are definitional:
integrated implies implemented, and publicly reachable implies
integrated. Nothing else is entailed, so nothing else may be assumed.

**`n/a` is a real answer and needs a reason.** Hardware validation is
`n/a` for a capability with no device dependence. Differential testing is
`n/a` where LightGBM has no counterpart to compare against, which is the
usual case for a mojoboost extension.

**Skipped is not passed.** A test that runs its process and prints
"skipped: no accelerator" has not exercised the behavior. A suite that
CI never invokes, or that needs an optional dependency the test
environment does not declare, is not focused-tested. Say where it does
run instead.

**One machine is one machine.** Hardware validation names the device.
"Validated on Apple M4" never widens to "validated on GPUs".

**Levels are claims about today.** They are re-derived by reading the
repository, not carried forward from a handoff. A handoff describes what
its author built; whether it is integrated is a property of the tree
after everything landed, and only the tree can answer that.

## How the levels map onto the parity statuses

`docs/LIGHTGBM_PARITY.md` keeps its five status words, which are a
user-facing summary. The levels are what the summary has to be consistent
with:

| Status | Requires | Forbids |
|---|---|---|
| `supported` | implemented, integrated, publicly reachable, focused-tested | any of those being `no` |
| `partial` | implemented, and at least one of the four above being `no` or narrower than LightGBM's surface | claiming the missing level anywhere else |
| `different` | implemented and publicly reachable, by a deliberately different design | being used to describe an absence |
| `deferred` | nothing, and it is the right status for an implemented but unintegrated module | publicly reachable being `yes` |
| `unsupported` | nothing, and a stated reason not to build it | publicly reachable being `yes` |

The last two lines are the ones that rot: a module lands, a later change
wires it up, and the row still says `deferred`. `tools/check_parity.py`
watches the named public symbols behind those rows for exactly that, and
it watches symbols rather than files, because a file existing is not
support.

## Citing this file

> mojoboost states capabilities at seven levels (implemented,
> integrated, publicly reachable, focused-tested, differential-tested,
> hardware-validated, release-packaged). See docs/CAPABILITY_LEVELS.md.
