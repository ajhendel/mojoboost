# The instruction audit

Every executable instruction in `PROFILE_PROTOCOL.md`, `MACHINE_LOCK.md`,
`SESSION_QUEUE.md`, `bench/README.md` and `handoffs/`, with a verdict and the
evidence the verdict rests on.

Written 2026-08-16 on `lane/instruction-audit`, branched from `perf-round-2` at
`cbff73b`. No benchmark, timing run, or training run was taken: another
orchestrator holds `/tmp/mojotrees-bench.lock` in `mode: timing` for the whole
window this was written in. Everything labelled **verified by execution** below
is a non-timing execution — a self-check, a `--help`, a grep, a gate, a plan
print, or reading a live file.

## Why this file exists

`PROFILE_PROTOCOL.md` section A1 found that the protocol had been instructing
every session to "capture thermal state into the results header with
`bench/apple/thermal_capture.sh`", and that the script measures nothing. The
lesson it drew was that inherited instructions nobody executes are the failure
mode to watch. That is a correct diagnosis and it is a mood, not a control. A
mood does not tell you which of the other instructions are in the same state.

This file is the control. It enumerates them and settles each one against
evidence.

**The audit found the A1 finding itself to be half wrong, and wrong in the
direction that matters.** A1 says "No session has ever done this and none could
have." A session did. `bench/results/profile_2026-08-15/header.txt` line 9 opens
a section headed `=== thermal before ===` and pastes the script's output into
it. What is pasted is a plan, carrying `run id thermal-PENDING` and `would write
.../thermal-PENDING.json`. So the results header of the first stage-level
profile this repository ever recorded carries a plan printer's plan under a
heading that reads as a capture, and it has been committed and quoted since
2026-08-15 (`24e5330`).

That is strictly worse than the failure A1 described. An instruction nobody
follows leaves a gap. An instruction somebody follows, which returns a plausible
block of text that is not a measurement, leaves a **record** — and the record is
what the next reader trusts.

## Provenance vocabulary

Per `PROFILE_PROTOCOL.md`'s Sweep II vocabulary: **measured** / **fitted** /
**derived bound** / **estimated**, plus, for this lane, **verified by
execution** — I ran the thing myself in this session and report what happened.

## Verdicts

- **RUN** — demonstrably executed, with the results file, commit, or output
  named.
- **NEVER RUN** — no evidence anywhere in the repository that it has been
  executed.
- **NOT EXECUTABLE** — it cannot work as documented. Why is stated.
- **REQUIRES HUMAN ACTION** — it is executable but needs privileges a session
  does not have. Not the same as broken. The exact command is given so a person
  can run it deliberately.

## The scoreboard

Named instructions in the four documents plus `handoffs/`, **deduplicated**: an
instruction cited by three documents is counted once, and a block of six sweep
commands that stand or fall together is counted once. The 68 environment
variables are counted separately below, because they are a different kind of
object and would otherwise swamp the total.

| verdict | count | where the rows are |
|---|---|---|
| **RUN** | 46 | sections 3, 4, 5, 7, 9h |
| **NEVER RUN** | 23 | sections 3, 5, 7, 9h |
| **NOT EXECUTABLE** | 7 | sections 1, 1a, 2, 9b, 9d, 9h |
| **REQUIRES HUMAN ACTION** | 5 privileged, 1 manual | section 1b, section 10 |

Environment variables, from `compatibility/api_snapshot.json`'s
`environment.observed`, section 9. These four are exhaustive and sum to 68.

| verdict | count |
|---|---|
| **RUN** with a nameable results file | 5 |
| **RUN** but the artifact does not name the variable (section 9c) | 4 |
| **NEVER RUN** | 58 |
| **NOT EXECUTABLE** (read by no code at all) | 1 |

The RUN column is a **lower bound** and NEVER RUN an **upper bound**, for the
reason in section 11: a measurement can be taken and not filed, and section 9c
shows four cases where one was filed without naming what produced it.

### The three worst offenders

1. **`handoffs/performance_17_thermal_energy.md` does not exist**, and eight
   live places delegate to it, including the script whose self-check fails on
   its absence and the two documents that cite it as the fix for the thermal
   problem. It is the repository's only complete listing of its privileged
   commands. Section 1a. Recoverable in one command; see section 10.
2. **`SESSION_QUEUE.md` opens with "Everything below is pending" and every
   measurement item in it has been run.** Six A/Bs, all reported in
   `session3_2026-08-16/RESULTS.md`, and the queue records none of it. A
   session reading it today would re-run completed work and would read S1 as
   undecided when it has been closed. Section 5.
3. **"Record the canary ratio at both ends", added three hours ago in the
   commit whose purpose was to put the instructions that get read at the top of
   the protocol.** There is no canary anywhere in the repository. Section 2.
   Third place rather than first only because it is new enough that nobody has
   yet failed to follow it.

Dishonourable mention, because it is the exact defect A1 named and it is worse
than A1 thought: `profile_2026-08-15/header.txt` carries a plan printer's plan
under a heading reading `=== thermal before ===`, and has since 2026-08-15.

---

## 1. The worked example: `bench/apple/thermal_capture.sh`

**Verdict: NOT EXECUTABLE as an instrument. Partially RUN as a plan printer,
and its output was filed as though it were a measurement.**

Cited at `PROFILE_PROTOCOL.md:79-80` (session conditions), `:709-710` and
`:727` (A1), and `MACHINE_LOCK.md:73-82`.

The script's own header, `bench/apple/thermal_capture.sh:9-19`:

> This script does not measure anything. It starts no sampler, runs no
> privileged command, fits no model, and writes no record. It validates its
> arguments and prints a plan. That is the whole of it.
>
> The reason it is built this way is that the commands a thermal run wants to
> issue include `sudo powermetrics`, and a script that can be talked into
> running that by a typo is a script that eventually will.

Exit code 3 is documented as "`--execute` refused; this version has no
measurement path".

**Verified by execution**, three ways:

1. `bash bench/apple/thermal_capture.sh --execute --phase cold_fit` exits **3**
   and prints "Nothing was measured and no file was written." The refusal is
   real, not aspirational.
2. `bash bench/apple/thermal_capture.sh --print-plan --phase sustained` prints
   JSON containing `"executed": false`, `"run_id": "thermal-PENDING"`, and
   `"measurement_path": "absent; no fit driver exists for these phases"`.
3. `bash bench/apple/thermal_capture.sh --self-check` **FAILS**, exit code
   **4**, on one problem:

   ```
   FAIL missing handoffs/performance_17_thermal_energy.md
   ```

Finding 3 is new and is not recorded anywhere in the four documents.

### 1a. The replacement pointer is itself dangling

**Verdict: NOT EXECUTABLE.**

Both `PROFILE_PROTOCOL.md:727-729` and `MACHINE_LOCK.md:82` resolve the thermal
problem by pointing the reader somewhere else:

> The script stays where it is, because the plan it prints is a real plan and
> `handoffs/performance_17_thermal_energy.md` lists the privileged commands for
> a human

> The real commands are listed for a human to run by hand in
> `handoffs/performance_17_thermal_energy.md`.

**That file does not exist.** `ls handoffs/` returns five files and it is not
among them. It was deleted on 2026-08-14 in `21ff9fa`, "Remove obsolete handoff
documents", a bulk removal of some ninety handoffs — two days *before* both
documents were written pointing at it. `bench/apple/thermal_capture.sh` checks
for it and fails its self-check on its absence, and `--execute`'s own refusal
message directs the reader to it.

So the fix A1 wrote for a dangling instruction is a dangling instruction. Three
separate places send a reader to a file that has not existed since Aug 14.

### 1b. What the privileged commands actually are

**Verdict: REQUIRES HUMAN ACTION.** Recovered from `docs/APPLE_THERMAL_ENERGY.md`
lines 167-173, which survived the handoff deletion and is the only surviving
listing.

| command | privilege | what it yields |
|---|---|---|
| `sudo powermetrics --samplers cpu_power,gpu_power` | root | package / CPU / GPU / ANE power series, machine-wide |
| `sudo powermetrics --samplers thermal` | root | thermal pressure level (a coarse level, not a temperature) |
| `sudo powermetrics --samplers smc` | root | fan speed, and on some Macs temperatures; fields vary by model |
| `sudo powermetrics --samplers tasks --show-process-energy` | root | per-process energy |

Do not run these from a session. They are listed so a person can.

Two caveats travel with them, both from `docs/APPLE_THERMAL_ENERGY.md:458-462`
and both **estimated**: whether `pmset -g therm` moves at all on Apple silicon
under sustained load is unknown to this repository, and the `powermetrics`
parser in `bench/apple/suite.py` has never seen real output.

---

## 2. The newest instruction in the protocol is already unfollowable

**`PROFILE_PROTOCOL.md:13` — "record the **canary ratio** at both ends".**

**Verdict: NOT EXECUTABLE.**

**Verified by execution**: `grep -rn "canary" --exclude-dir=.git .` over the
whole repository returns exactly one hit, and it is the instruction itself.

There is no canary. No script defines one, no results file records one, no
source module computes one, and neither `MACHINE_LOCK.md` nor `SESSION_QUEUE.md`
nor any results file explains what ratio of what to what is meant. A session
told to "record the canary ratio at both ends" has nothing to run.

It was added three hours before this audit, in `cbff73b`, "Put a ten-line 'how
to take a number' checklist at the top" — the commit whose stated purpose is to
put the instructions that will actually be read at the top of the file. The
checklist has ten items and one of them cannot be followed.

This is A1's failure mode reproducing inside the amendment that diagnosed it,
on the same day, in the same file. It is the strongest argument in this audit
that the mood was not a control.

---

## 3. `PROFILE_PROTOCOL.md`

| what | verdict | evidence |
|---|---|---|
| Take `/tmp/mojotrees-bench.lock`, `mode: timing` (`:8-10`) | **RUN** | **Verified by execution**: the lock exists right now, 419 bytes, and reads `holder: cascadeprojects-3b (CPU campaign orchestrator...)`, `mode: timing`, `since: 2026-08-16 03:20 local`, with a `what` and an `eta`. The format matches `MACHINE_LOCK.md:55-64` exactly. This lane is honoring it. |
| `uptime` before first arm and after last (`:12`, `:722`) | **RUN**, partially | `bench/results/apple_m4_unified_memory_2026-08-15/env_start.txt:2` and `env_end.txt:2` carry it at both ends. **No other results directory does.** `profile_2026-08-15/header.txt`, `sweep2_2026-08-15/header.txt` and `metal_timeline_2026-08-15/header.txt` have no `uptime` line; `session3_2026-08-16/RESULTS.md:237` mentions "load 2.15" in prose without recording the command's output. **Verified by execution**: `uptime` runs and returns `load averages: 4.56 3.66 3.01`. |
| `ps -Ao pcpu,comm \| sort -rn \| head` (`:12`, `:723`) | **NEVER RUN** | No file under `bench/results/**` contains its output, in any directory, at either end. The `uptime` half of the pair has one instance; the `ps` half has none. **Verified by execution**: the command works and is instant. |
| Record the **canary ratio** (`:13`) | **NOT EXECUTABLE** | Section 2 above. No referent anywhere in the repository. |
| `bench/apple/thermal_capture.sh` before and after (`:79-80`) | **NOT EXECUTABLE** | Section 1 above. |
| `bench-train-gpu rows feats reg N arm1,arm2` interleaved (`:83`) | **RUN** | `bench/results/session3_2026-08-16/RESULTS.md` M2.3, M2.4 and "The margin, measured properly"; `bench/results/lgbm_1m.json`, `lgbm_1m_t10.json`, `unroll_1m.json` are its `MOJOTREES_BENCH_JSON` artifacts. |
| R1 / C1.2: our CPU at `MOJOTREES_NUM_WORKERS=1` vs LightGBM `num_threads=1` (`:118`, `:507`) | **RUN** | `bench/results/profile_2026-08-15/r1_ours_1w.txt` and `r1_lightgbm_1t.txt`. Result quoted at `bench/README.md:52` as 15.96s against 8.82s, **1.81x**. C2 (`:521`) schedules it re-taken; the re-take has **not** happened. |
| R4 / C1.5 / C4: `MOJOTREES_CPU_CORE_POOL=performance` against the default (`:167`, `:514`, `:598`) | **NEVER RUN** as a measurement | Read by code and exercised for correctness by `tests/test_cpu_dispatch.mojo`, `tests/test_cpu_feature_group.mojo`, `tests/test_const_hessian.mojo`. **No results file anywhere records a timing for it**, at any shape. It is scheduled by three separate rules and has never produced a number. |
| Sweep II arm `MOJOTREES_GPU_TREE_RESIDENT=1` (`:245`) | **RUN** | `session3_2026-08-16/RESULTS.md` M2.2 and S1; `sweep2_2026-08-15/resident_ab.txt`, `resident_1m.txt`. |
| Sweep II arm `grow_policy=depthwise` (`:246`) | **RUN** | `sweep2_2026-08-15/1m_depth.txt`; `bench/README.md:71` table. |
| Sweep II arm LightGBM at 10 threads (`:247`) | **RUN** | `sweep2_2026-08-15/lightgbm.txt`; `session3_2026-08-16/RESULTS.md` M2.4. |
| Sweep II shapes including 2,000,000 and 5,000,000 rows (`:251`) | **RUN at 2,000,000; NEVER RUN at 5,000,000** | `bench/README.md:72` has a 2,000,000 row. No results file anywhere has a 5,000,000 row point. The superlinearity question the sweep existed to answer (`:219-235`) is therefore still open, and no document says so. |
| C1.1: baseline, our CPU against LightGBM at 10 threads, in one process, at 1M / 250k / 50k | **NEVER RUN** | The `cpu` and `lightgbm` arms exist (`bench/bench_train_gpu.mojo`), and `session3` ran `gpu-device,lightgbm` at 1M. The **CPU** arm against LightGBM interleaved at three shapes is the CPU round's Phase 0 and no results file records it. The live lock says it is being taken right now. |
| C1.3 / C5: `bench/bench_profile.mojo` at 1,000,000 x 50, serial against auto (`:509`, `:610`) | **NEVER RUN** for a record | `bench/README.md:177` states it outright: "**No numbers are recorded here yet.**" `bench/results/profile_2026-08-15/` is *not* this instrument — it is the **GPU** phase profile via `MOJOTREES_PHASE_PROFILE`, as `phase_gpu_1m.txt` shows (`phase_profile begin label=train_gpu`). The naming collision is a trap: "the first stage-level profile ever recorded" (`profile_2026-08-15/RESULTS.md:3`) is a GPU profile, and the CPU stage profile the protocol keeps scheduling has still never produced a number. |
| C1.4 / C3: LightGBM `force_col_wise` against `force_row_wise`, both pinned, 1M x 50 | **NEVER RUN** | `session3_2026-08-16/RESULTS.md:385-387` says so directly: "**Unresolved. The CPU campaign is measuring `force_row_wise` against `force_col_wise`... Until that lands, every margin in this file carries this caveat.**" No results file contains a `force_col_wise` figure. This is the single largest open caveat on every LightGBM margin in the repository. |
| The comparator rule: "LightGBM's better pinned builder at each shape" (`:37-40`) | **NEVER RUN** | Same as above. The rule was registered in `cbff73b`, three hours ago, and no shape has had its builder measured. The rule is honest about this at `:42-44`. |
| `pmset -g therm` (`:733`) | **RUN**, and it returns nothing useful | **Verified by execution**: returns "No thermal warning level has been recorded / No performance warning level has been recorded / No CPU power status has been recorded". The protocol's claim that it "returns nothing useful on Apple silicon" is correct. It is recorded as a thermal condition in `sweep2_2026-08-15/header.txt:3` and `sweep2_2026-08-15/RESULTS.md:5` ("No thermal warning recorded"), where it reads as a thermal measurement and is not one. |
| S1's node-identity condition, `tests/test_gpu_tree_resident.mojo` (`:281`) | **RUN** | File exists; `session3_2026-08-16/RESULTS.md:280` records it satisfied with no tolerance. |
| C6: determinism across `MOJOTREES_NUM_WORKERS` (1, 3, 8) (`:638`) | **RUN**, as a contract | Eight test files pin `MOJOTREES_NUM_WORKERS`: `test_cpu_feature_group`, `test_round_overhead`, `test_cpu_dispatch`, `test_cpu_parallel`, `test_binning`, `test_const_hessian`, `test_histogram_reference`, `test_sparse`. |
| `handoffs/performance_17_thermal_energy.md` (`:727`) | **NOT EXECUTABLE** | Section 1a. Does not exist. |
| C-ops: "`/tmp/mojotrees-bench.lock` is deleted and neither session recreates it" (`:465-472`) | **CONTRADICTED BY FACT** | The lock exists and is held. `MACHINE_LOCK.md` and checklist item 1 both instruct sessions to take it. See section 6. |

---

## 4. `MACHINE_LOCK.md`

| what | verdict | evidence |
|---|---|---|
| `/tmp/mojotrees-bench.lock`, `mode: timing` only (`:55-64`) | **RUN** | **Verified by execution**: read the live lock. Its fields are exactly the ones this section specifies — holder as an addressable `ListAgents` name, mode, since, what, eta, and the rule restated. This is the best-followed instruction in the four documents. |
| Every lane in its own worktree under `.claude/worktrees/`, gitignored (`:46-49`) | **RUN** | `.gitignore:45-46` carries the rule with the reason. **Verified by execution**: this lane is running in `/Users/andrewhendel/CascadeProjects/mojotrees/.claude/worktrees/agent-a3a5f318d4b5043e7`, and `git worktree list` shows some forty lanes doing the same. |
| `bench/apple/thermal_capture.sh` "has never been followable and no session has ever followed it" (`:73-82`) | **FACTUALLY WRONG** in its second half | Section 1. `profile_2026-08-15/header.txt:9-20` is a session having followed it. The script cannot measure — that half stands — but the claim that nobody ran it is refuted by a committed artifact. |
| `handoffs/performance_17_thermal_energy.md` (`:82`) | **NOT EXECUTABLE** | Section 1a. |
| `pmset -g therm` (`:84-85`) | **RUN** | **Verified by execution**. The recorded answer matches. |
| `uptime` and `ps -Ao pcpu,comm \| sort -rn \| head` "captured before and after every arm... That is what both orchestrators now use" (`:87-91`) | **`uptime` partially RUN; `ps` NEVER RUN** | Section 3. The claim "that is what both orchestrators now use" is not evidenced in any results file written since it was made. |
| `bench-train-gpu rows feats obj N arm1,arm2` as the contention answer (`:113`) | **RUN** | Section 3. |
| Addressable session name via `ListAgents` (`:58`) | **RUN** | The live lock's `holder:` field carries one (`cascadeprojects-3b`). Note that `ListAgents` is a property of the agent harness, not of this repository, so a human reader following this file has nothing to invoke. Worth a parenthesis in the doc. |
| Nobody runs `git checkout <branch>` in the main checkout (`:42-43`) | **RUN**, holding | **Verified by execution**: `git worktree list` shows `/Users/andrewhendel/CascadeProjects/mojotrees` on `perf-round-2`, as specified. |

---

## 5. `SESSION_QUEUE.md`

**The headline finding for this file: its first line says "Everything below is
pending." Every measurement item in it has since been run.** `SESSION_QUEUE.md`
describes a queue that was drained on 2026-08-16 and records none of it. A
session reading it today would re-run five completed A/Bs and would read S1 as
undecided when it has been closed.

| what | verdict | evidence |
|---|---|---|
| "Everything below is pending" (`:3`) | **STALE** | `session3_2026-08-16/RESULTS.md` reports M2.1, M2.2, M2.3, M2.4, M2.5 and S1 all measured. |
| S1: `MOJOTREES_GPU_TREE_RESIDENT` ON/OFF at 250,000 and 50,000 (`:41-44`) | **RUN** | `session3_2026-08-16/RESULTS.md:235-284`. 250k: 1.126 vs 2.010, 44%, resolved. 50k: 0.789 vs 1.724, 2.2x, resolved. The queue's status table (`:31-34`) says "**250k has never been measured at all**" and "**never measured**" for 50k. Both are now false. |
| S1's conclusion, "the plane stays opt-in" (`:36-37`) | **SUPERSEDED, and the code did not follow** | `session3` declares "**All three hold. Under S1 the resident plane becomes the default GPU plane.**" **Verified by execution**: `src/mojotrees/gpu_resident_round.mojo:329` still reads `getenv("MOJOTREES_GPU_TREE_RESIDENT") == "1"`, so the plane is still off by default. A decision was recorded and not shipped. Belongs to `src/`; flagged for the orchestrator. |
| M2.1: `MOJOTREES_GPU_TABLE_RESET=0 MOJOTREES_GPU_PACKED_DOWNLOAD=0` off arm vs on arm (`:128-131`) | **RUN** | `session3_2026-08-16/RESULTS.md:16-36`. Five pairs. **REFUTED**: 0.016s against a registered 0.64s prediction. The queue still labels it "**Blocked on merge**" (`:73`). |
| M2.2: resident plane re-taken at 1M (`:143-144`) | **RUN** | `session3_2026-08-16/RESULTS.md:38-66`. Six pairs, resolved, 24% fast regime / 8% slow. |
| M2.3: `bench-train-gpu ... row-unroll-on,row-unroll-off` (`:162`) | **RUN, twice** | `session3_2026-08-16/RESULTS.md:95-134`. Indistinguishable in a slow window, resolved at 10.8% in a fast one. The queue says "**Unblocked.** The arms are wired and compile; nothing has been run" (`:149`). |
| M2.3: `bench-hist 1000000 50 20` (`:163`) | **RUN** | `session3_2026-08-16/RESULTS.md:136-152`. 1.488ms vs 3.178ms, 2.14x, resolved. |
| `gpu-unroll,gpu-nounroll` "still parses and resolves to the same pair" (`:176-177`) | **TRUE**, verified | **Verified by execution**: `bench/bench_train_gpu.mojo:441-444` maps `"row-unroll-on" or "gpu-unroll"` and `"row-unroll-off" or "gpu-nounroll"` to the same two arms. |
| M2.4: `bench-train-gpu 1000000 50 reg 5 gpu-device,lightgbm` (`:189`) | **RUN** | `session3_2026-08-16/RESULTS.md:173-193`, and `bench/results/lgbm_1m.json` / `lgbm_1m_t10.json`. LightGBM's repeat spread measured for the first time: 3.7% stable, 12.9% unstable. |
| M2.4: the same at **250,000** (`:190`) | **NEVER RUN** | `session3` reports the 1,000,000 shape only. The 250,000 line of this pair has no result anywhere. |
| M2.5: `MOJOTREES_PARALLEL_MIN_OPS=32768` against the default at 50,000 (`:215-216`) | **RUN** | `session3_2026-08-16/RESULTS.md:157-171`. Three alternating pairs, 1.700/1.703/1.714 against 1.710/1.704/1.722. **NULL**, indistinguishable, and the docstring's refusal to move the grain without measurement is vindicated. This is the model of what a RUN verdict looks like: a named results file, the arms, the repeats in order, and a verdict. Note that `session3`'s own "Still open" list (`:403`) contradicts its own section 157 by calling M2.5 "untaken". |
| `MOJOTREES_GPU_TREE_RESIDENT_TRACE=1` as the gate proof (`:269-274` in `session3`) | **RUN** | Reported output: `plane=device-resident status=budget_spent commits=30 leaves=31 root_rows=50000`. This is the repository's best instance of the C6 rule that a gated path's test must prove the gate opened. |
| `compatibility/api_snapshot.json` "is stale and was already stale before this round"; regenerate once at merge with `--write` (`:247-254`) | **DONE, and the bullet is now wrong** | **Verified by execution**: `python3 tools/api_snapshot.py --check` returns `ok`. All five names the bullet lists as missing — `MOJOTREES_CONST_HESSIAN`, `MOJOTREES_GPU_TREE_RESIDENT`, its trace variables, `MOJOTREES_GPU_TABLE_RESET`, `MOJOTREES_GPU_PACKED_DOWNLOAD` — are present in `environment.observed`. |
| `gpu_gradient_stream.HostGradientStage` "has **no callers anywhere** in `src/`, `bench/` or `tests/`" (`:278-281`) | **TRUE**, verified | **Verified by execution**: `grep -rn HostGradientStage src/ bench/ tests/` returns its own definition (`gpu_gradient_stream.mojo:508`), a docstring at `:36`, and two mentions in `gpu_fused_round.mojo` that are a docstring line and an error-message string. No call site. The queue's claim stands and the deletion it proposes is safe. |

---

## 6. The lock, and a protocol that contradicts itself in two places

**Verdict on the instruction: RUN. Verdict on the document: internally
inconsistent.**

Three passages in my file boundary describe the machine lock and they do not
agree.

- `PROFILE_PROTOCOL.md:8-10` (checklist item 1, added `cbff73b`): **take the
  lock.**
- `MACHINE_LOCK.md:38-69` (agreed and ratified 2026-08-16): **the lock is in
  force**, `mode: timing` only, both parties check it.
- `PROFILE_PROTOCOL.md:465-472` (C-ops, CPU round 1): "**There is no measurement
  lock.** One was negotiated between the two orchestrators and then overruled by
  Andrew... `/tmp/mojotrees-bench.lock` is deleted and neither session
  recreates it."

The observable fact settles it: **verified by execution**, the lock exists, is
held by the CPU campaign orchestrator, and is formatted to `MACHINE_LOCK.md`'s
specification. C-ops is the stale passage — `MACHINE_LOCK.md:22` records exactly
this history, "the lock being dropped once and reinstated in a narrower form the
same night", and C-ops was written during the dropped interval.

The irony is worth stating rather than left implicit: C-ops' own justification
for recording the absence was that "a protocol that describes a control which is
not in force is worse than one that admits it has none". It is now a protocol
passage describing the *absence* of a control that **is** in force, which is the
same error with the sign flipped. Corrected in this branch.

---

## 7. `bench/README.md`

Every `pixi run` task named in this file exists. **Verified by execution**: a
script parsed `pixi.toml`'s `[tasks]` and `[feature.*.tasks]` and cross-checked
all sixteen task names the four documents invoke. Zero missing. (The one
apparent miss, `pixi run mojo run -I src ...` at `:711` and `:893`, is pixi's
passthrough form and is correct.)

So the defect class here is not dangling commands. It is **drivers that exist,
compile, and have never produced a recorded number** — and the file is unusually
honest about most of them, which is why they are marked NEVER RUN rather than
NOT EXECUTABLE.

| what | verdict | evidence |
|---|---|---|
| `pixi run bench [rows] [features] [obj] [seed]` (`:23-27`) | **RUN** | The 100,000 x 100 baseline table at `:37-44`. Explicitly labelled a pre-multicore single-threaded baseline. |
| `pixi run -e bench bench-lgbm ...` (`:26-27`, `:210-212`) | **RUN** | Same table; `profile_2026-08-15/r1_lightgbm_1t.txt`. |
| `pixi run -e bench bench-lgbm --repeats 5` (`:329`) | **NEVER RUN** | The `--repeats` flag exists — **verified by execution**, `bench/bench_lightgbm.py:396-407` — but no separate-process LightGBM run with a measured spread is recorded anywhere. The in-process spread was measured instead (`session3` M2.4). |
| `pixi run bench-hist` (`:28`, `:382`) | **RUN** | `session3_2026-08-16/RESULTS.md:136-152`. |
| `pixi run bench-profile` (`:168-169`) | **NEVER RUN** for a record | The file says so at `:177`: "**No numbers are recorded here yet.**" Still true. Scheduled by `PROFILE_PROTOCOL.md` C1.3 and C5, which is the round's stage-selection rule, so the CPU round's lane ordering currently rests on an instrument that has never been read. |
| "Sweeping `TASKS_PER_CORE` in `src/mojotrees/parallel.mojo` and rerunning this is how that constant should be settled; it is currently an unmeasured starting value of 4" (`:172-174`) | **NEVER RUN**, and the instruction is out of date | **Verified by execution**: `src/mojotrees/parallel.mojo:174` still has `comptime TASKS_PER_CORE = DEFAULT_TASKS_PER_CORE`, unmeasured as stated. But `:172` of that same source file names `MOJOTREES_CPU_TASKS_PER_CORE` as the way to sweep it, so editing the source is no longer necessary. The README instruction predates the env var. Corrected in this branch. |
| `MOJOTREES_NUM_WORKERS=1` four-run matched-thread comparison (`:209-212`) | **RUN** at a different shape | `profile_2026-08-15/r1_*.txt` at 1,000,000 x 50. The README's own 100,000 x 100 version has "**No post-optimization numbers are recorded here yet**" (`:217`). The file already warns at `:220-224` that this four-process form is the protocol it forbids elsewhere. |
| `pixi run -e bench bench-train-gpu ... gpu-device-depth,lightgbm` (`:236`) | **RUN** | `sweep2_2026-08-15/`; `bench/README.md:68-72`. |
| `MOJOTREES_LGBM_THREADS=10 ... cpu,lightgbm` (`:237`) | **NEVER RUN** | No results file records a `cpu,lightgbm` interleaved pair. This is CPU round Phase 0 item 1 and the live lock says it is being taken now. |
| `MOJOTREES_BENCH_JSON=<path>` (`:285-286`) | **RUN** | **Verified by execution**: `bench/results/lgbm_1m.json`, `lgbm_1m_t10.json` and `unroll_1m.json` all parse and carry `arms`, `comparisons`, and (for the first two) a `lightgbm` block. The mechanism works and produced committed artifacts. |
| `pixi run bench-hist-scaling` (`:428-429`) | **RUN** | Results table at `:449-453`, three shapes, `reps=30`, with the honest note that the machine was not idle. |
| `MOJOTREES_GPU_HIST_STRATEGY` / `MOJOTREES_GPU_ROW_TILE` / `MOJOTREES_GPU_BLOCK_THREADS` "sweep the tiling by hand, **which is how the defaults in that module were chosen**" (`:432-435`) | **OVERSTATED** | The same file says at `:474-478` "Not yet measured: a sweep of `MOJOTREES_GPU_ROW_TILE` at the largest shape, which is what would confirm or move `TARGET_BLOCKS_PER_SM`". Both cannot be true. **Verified by execution**: `src/mojotrees/gpu_tiling.mojo:114-121` says a small footprint "is evidence that 8 blocks are not excluded, **not** evidence that more than 8 are resident", i.e. `TARGET_BLOCKS_PER_SM = 8` is a derived bound, not a measured choice. `MOJOTREES_GPU_HIST_STRATEGY` did produce the `:449-453` table. Corrected in this branch. |
| `pixi run bench-launch-cost` (`:489-490`) | **RUN** | `:501-502` quotes 20us launch, 126us wait, 280us per split; `:504-513` records the independent Metal-trace confirmation. One of the best-evidenced entries in the file. |
| `pixi run bench-unified-memory` and `MOJOTREES_UM_MODE=resident` (`:544-546`) | **RUN** | `bench/results/apple_m4_unified_memory_2026-08-15.md` and its 90-file artifact directory, run UM-2026-08-15-M4-01. |
| `MOJOTREES_UM_LADDER=1` | **NEVER RUN** | `apple_m4_unified_memory_2026-08-15.md:24` says so: "Not run: the size ladder (`MOJOTREES_UM_LADDER=1`), because the machine..." Honest, and out of `bench/README.md`'s scope. |
| `pixi run gpu-validate` (`:569-570`) | **RUN on Apple Metal only** | `:576-578` states the record is Apple Metal only and no NVIDIA or AMD device has executed it. `docs/GPU_VALIDATION.md:456` still has a `<paste the full gpu-validate output>` placeholder, so no committed capture exists under `bench/results/`. Verdict is RUN-but-unfiled. |
| `pixi run bench-train-gpu 250000 100 reg 5 gpu-host,gpu-device` etc., the dense crossover ladder (`:646-670`) | **PARTIALLY RUN** | Run at 50 features (`profile_2026-08-15`, `sweep2_2026-08-15`). The documented ladder is at **100** features and the three-seed / `binary` repeats at `:668-670` have no results anywhere. `crossover_rules()` carries one dense rule; `:596-602` records that it cannot fire through `resolve_device` at all. |
| `pixi run bench-train-gpu-sparse ...`, six commands (`:694-711`) | **NEVER RUN** | `:764` states it: "**The sparse sweep has no numbers yet**, and `crossover_rules()` carries no sparse rule for that reason." Six commands, zero runs. |
| `pixi run bench-custom` (`:778-779`) | **NEVER RUN** | `:793-794`: "no native-interface numbers are recorded here yet; run `pixi run bench-custom` and add them with the machine". |
| `pixi run -e bench bench-custom-py` (`:788-789`) | **RUN** | Table at `:797-800`, best of 3, with the not-idle caveat stated. |
| `pixi run bench-goss` (`:822-824`) | **NEVER RUN** | `:838-842`: "**No numbers are recorded here yet.**" |
| `pixi run -e bench compare-ranking` (`:862-863`) | **NEVER RUN** as a filed result | The prose describes expected behavior ("Expect around 1e-9") rather than an observed run, and no results file records it. Distinguish from `compare-missing` below, which reports an outcome. |
| `pixi run -e bench compare-missing` (`:876`) | **RUN** | `:879-882` reports an outcome against a named version: "As of LightGBM 4.7 every routing decision matches", with the `lambda_l2` difference explained. |
| `pixi run bench-sparse` (`:892-893`) | **RUN** | Table at `:901-910`, two configurations, with a repeat quoted at `:922-924` (2.22s / 1.28s) as the honesty check. |
| `sysctl hw.perflevel0.physicalcpu` (`:631`) | **RUN** | **Verified by execution**: returns `4` on this machine. |

### Tasks that exist and no document in scope mentions

`bench-threshold`, `bench-hybrid-costs`, `compare-categorical`, `lgbm-differential`.
`bench-hybrid-costs` has results (`apple_m4_hybrid_costs_2026-08-14.md`,
`apple_m4_hybrid_costs_2026-08-15.md`) and is the only one of the four with a
filed number. Not a defect in the four documents, recorded so the next audit
starts from a list.

---

## 8. `handoffs/`

Swept in section 9h, after the environment surface, because the two share a
finding. The one result that outranks everything else in this directory is
negative: **`handoffs/performance_17_thermal_energy.md` is cited by eight live
places and does not exist** (section 1a).

---

## 9. The environment surface

The orchestrator suggested `compatibility/api_snapshot.json` as the starting
index for documented environment knobs. It is the right starting point and it is
**incomplete in two ways that matter**, both **verified by execution**.

1. **It does not scan the benchmark harness.** `tools/api_snapshot.py:99` sets
   `ENV_SCAN_DIRS = ("src", "bindings", "python", "capi", "cli")`. `bench/` and
   `tools/` are deliberately excluded. So every knob that only the benchmark
   drivers read is invisible to it, including `MOJOTREES_BENCH_JSON`,
   `MOJOTREES_LGBM_THREADS`, `MOJOTREES_UM_MODE`, `MOJOTREES_UM_CONTEND`,
   `MOJOTREES_UM_HOLD_MIB`, `MOJOTREES_UM_LADDER`, `MOJOTREES_UM_LADDER_MAX_MIB`,
   `MOJOTREES_UM_LADDER_PCT` and `MOJOTREES_PIXI_MANIFEST` — nine names, all of
   them documented in files inside this audit's scope, none of them in the
   snapshot's 68.

2. **`observed` is a string-literal scan, not a read scan.**
   `tools/api_snapshot.py:839` states it: "`observed` is the double-quoted
   literal scan and not a scan of `getenv`". The snapshot's own
   `read_directly` list has **37** entries against `observed`'s **68**. A name
   in `observed` but not in `read_directly` may be read through a wrapper, or
   may appear only in a docstring. Membership in `observed` is therefore **not**
   evidence that a variable does anything.

The consequence for this audit: `observed` is a list of names the repository
*talks about*, and the interesting question — which of them has ever had a
measured effect — is answered below and is answered "almost none".

### 9a. Five of 68 environment variables have a measured effect on file

Each of the 68 names in `environment.observed` was checked three ways: is it
read by code, and where; does any file under `bench/results/**` or
`docs/METAL_TIMELINE.md` show it used in an actual measurement; is it exercised
by a test under `tests/`.

| category | count | verdict |
|---|---|---|
| read by **no code at all** | **1** | **NOT EXECUTABLE** |
| read, but no results file and no test | **32** | **NEVER RUN** |
| read and tested, but no measurement ever filed | **26** | **NEVER RUN** as a measurement |
| **measured**, with a nameable results file | **5** | **RUN** |
| measured, but the artifact does not name the variable | 4 more | **RUN**, untraceable — see 9c |

**Read by no code at all — `MOJOTREES_STARTUP_REPORT_FD`.** **NOT EXECUTABLE.**
`src/mojotrees/initialization.mojo:114` says so in as many words: "reserved,
unread here." It appears in `compatibility/api_snapshot.json` (twice),
`compatibility/DRIFT_REPORT.md:218`, and
`python/mojotrees/diagnostics.py:225` — the last inside a tuple whose own
comment reads "Listed, never interpreted". It is a documented, snapshotted,
drift-reported environment variable with no implementation, no test, and no
measurement. Nothing reads it; setting it does nothing. **Recommendation:
delete it from the snapshot and from `DRIFT_REPORT.md`, or implement it. It is
in `src/` and `compatibility/`, so it is left for the orchestrator.**

**The five with genuine measured evidence:**

| variable | evidence |
|---|---|
| `MOJOTREES_GPU_CLASS_BATCH` | `bench/results/profile_2026-08-15/RESULTS.md:202`, a results-table row: `mojotrees GPU, MOJOTREES_GPU_CLASS_BATCH=7 \| 15.45 \| 0.8%` |
| `MOJOTREES_GPU_SPLIT_STRATEGY` | `bench/results/sweep2_2026-08-15/RESULTS.md:98`, the forced-gate A/B at 20,000 rows with `=device` |
| `MOJOTREES_HYBRID_TRACE` | `bench/results/apple_m4_hybrid_costs_2026-08-15.md:91` |
| `MOJOTREES_HYBRID_GUARD_TRANSFER` | `bench/results/apple_m4_hybrid_costs_2026-08-15.md:135`, the `=1` guard arm |
| `MOJOTREES_GPU_TREE_RESIDENT_TRACE` | `bench/results/session3_2026-08-16/RESULTS.md:269`, output quoted to prove the gate was open |

That is **7 percent**. The remaining 93 percent are knobs the repository talks
about and has never demonstrated the effect of.

### 9b. `MOJOTREES_STARTUP_TRACE`, settled

**Verdict: NEVER RUN. The variable works; every documented way to exercise it
does not.**

- It **is** read: `src/mojotrees/initialization.mojo:284`,
  `return StartupTrace(getenv("MOJOTREES_STARTUP_TRACE") == "1")`. It is not
  inert, unlike its `_REPORT_FD` sibling.
- It has **no test** under `tests/` and **no results file** anywhere.
- Every documented invocation routes through a task that does not exist.
  `docs/STARTUP_LATENCY.md:260`, `:261`, `:262` and `:280` give four commands,
  all of the form `MOJOTREES_STARTUP_TRACE=1 ... pixi run bench-startup`.
  **Verified by execution**: `bench-startup` is not defined in `pixi.toml`'s
  `[tasks]` or any `[feature.*.tasks]`. The document self-flags this at its own
  `:265`, so it is honest rather than misleading, but the effect is the same —
  a variable whose only documented exercise is unrunnable.
- `python/mojotrees/diagnostics.py:651` promises "(untimed; set
  `MOJOTREES_STARTUP_TRACE=1` for durations)", which is the one reachable path,
  and no artifact shows it taken.

**Recommendation: fix the document.** Either add a `bench-startup` task, or
change `docs/STARTUP_LATENCY.md` to give the reachable invocation
(`MOJOTREES_STARTUP_TRACE=1 pixi run bench-train-gpu ...`, which
`:272` already does). Both are outside this lane's boundary; left for the
orchestrator with the exact lines named.

### 9c. Results files record arm labels, not the variables that produced them

This is a systematic traceability defect and it affects the four best-evidenced
performance knobs in the repository.

`SESSION_QUEUE.md:128` prescribes `MOJOTREES_GPU_TABLE_RESET=0
MOJOTREES_GPU_PACKED_DOWNLOAD=0` as the OFF arm of M2.1, and
`session3_2026-08-16/RESULTS.md:16-36` reports five pairs of it. But the results
table's columns are headed "collapse ON" and "collapse OFF". **Neither variable
name appears anywhere in any results file.** The same holds for
`MOJOTREES_GPU_TREE_RESIDENT`, prescribed at `SESSION_QUEUE.md:41-44` and
`:143-144` and measured six times, whose artifacts say "resident ON / resident
OFF" and never name it; only its `_TRACE` sibling is quoted.

So these four are **RUN** — the measurement happened and the results file exists
— but the artifact alone cannot prove which knob produced which arm. A reader
who wants to reproduce M2.1 must go to the queue document for the command, and
the queue document is the thing that goes stale.

**Recommendation: results files record the resolved environment of each arm,
not just its label.** `bench_train_gpu.mojo` already emits a `json_summary`
record; adding the `MOJOTREES_*` values it resolved would close this at one
site. `src/` and `bench/`, so left for the orchestrator.

### 9d. Two more inert or unreachable knobs, found outside the snapshot

- **`MOJOTREES_BUILD_LOCK` — NOT EXECUTABLE.** The emitted script from
  `tools/validation_plan.py` exports it, and **`tools/with_build_lock.sh:7`
  does not read it**: it opens the hard-coded `/tmp/mojotrees-build.lock`
  instead. **Verified by execution**: `python3 tools/validation_plan.py
  --self-check` prints the note itself — "`MOJOTREES_BUILD_LOCK` is exported by
  the emitted script and `tools/with_build_lock.sh` does not read it". Setting
  it does nothing. `handoffs/remaining_14_validation_plan.md:56` carries the
  patch (P1) and it has not been applied. Note the lock file **is** used in
  anger: `apple_m4_unified_memory_2026-08-15.md:32` records runs serialized
  under `tools/with_build_lock.sh`. Only the variable is inert.
- **`MOJOTREES_BINNING_SELECT_MIN_ROWS` — read but undocumented.**
  `src/mojotrees/binning.mojo:357`. `handoffs/remaining_14_validation_plan.md:238`
  (P5) is a patch to document it in `README.md` and has not been applied;
  **verified by execution**, the name appears nowhere in `README.md`.

### 9e. Where the snapshot's coverage stops, the evidence is best

The single sharpest illustration of section 9's point. The variables with the
**most** rigorous measured evidence in this repository —
`MOJOTREES_UM_MODE`, `MOJOTREES_UM_CONTEND`, `MOJOTREES_UM_HOLD_MIB`,
`MOJOTREES_UM_LADDER`, `MOJOTREES_UM_LADDER_MAX_MIB`, `MOJOTREES_UM_LADDER_PCT`,
read at `bench/apple/unified_memory.mojo:1424-1440` and `:1585-1586` — are the
arm definitions of `apple_m4_unified_memory_2026-08-15.md:39-40`, with a
quantified result at its `:160` ("`MOJOTREES_UM_CONTEND=1` cost 3% to 6% on
every route") and a 90-file artifact directory behind it. **None of the six is
in the snapshot**, because `ENV_SCAN_DIRS` excludes `bench/`.

An index that omits the best-evidenced knobs and includes one that nothing
reads is not, on its own, a map of what has been established. It is a map of
what `src/` mentions.

### 9f. The eight newest names, as the orchestrator asked

All eight landed in **one commit**, `7a73a64`, "Regenerate the API snapshot
once, for all six lanes at once", 2026-08-15 23:34. Its own message records that
these are **backfilled** names from earlier rounds, not newly implemented
behavior.

| variable | measured? | tested? |
|---|---|---|
| `MOJOTREES_CONST_HESSIAN` | no | `tests/test_const_hessian.mojo`, `tests/test_gpu_hist_row_unroll.mojo` |
| `MOJOTREES_CONST_HESSIAN_VERIFY` | no | `tests/test_const_hessian.mojo` |
| `MOJOTREES_GPU_PACKED_DOWNLOAD` | RUN via M2.1, artifact does not name it (9c) | **no test** |
| `MOJOTREES_GPU_SPLIT_TABLE_PACK` | no | **no test** |
| `MOJOTREES_GPU_TABLE_RESET` | RUN via M2.1, artifact does not name it (9c) | **no test** |
| `MOJOTREES_GPU_TREE_RESIDENT` | RUN via M2.2/S1, artifact does not name it (9c) | `tests/test_gpu_tree_resident.mojo` |
| `MOJOTREES_GPU_TREE_RESIDENT_TRACE` | **yes**, `session3:269` | `tests/test_gpu_tree_resident.mojo` |
| `MOJOTREES_GPU_TREE_RESIDENT_TRACE_STEPS` | no | `tests/test_gpu_tree_resident.mojo` |

One of eight has evidence that survives without the queue document. Two have no
test at all.

### 9g. The distributed surface: ten variables, no coverage of any kind

`MOJOTREES_DIST_MODE`, `DIST_RANK`, `DIST_WORLD_SIZE`, `DIST_MACHINES`,
`DIST_JOB_ID`, `DIST_TIMEOUT_S`, `DIST_RESTART_EPOCH`,
`DISTRIBUTED_PROVIDER`, `DISTRIBUTED_BASE_PORT`,
`DISTRIBUTED_CONNECT_TIMEOUT`. All ten are read
(`src/mojotrees/distributed_transport.mojo:2879-2911`,
`python/mojotrees/_dask_runtime.py`). **None is exercised by any test under
`tests/` and none appears in any results file.** `tests/test_distributed_transport.mojo`
exists and sets no environment variable.

**NEVER RUN**, all ten. This is outside the four documents' scope and is the
largest untested configuration surface the audit found. Recorded so the next
round starts from a list.

---

## 9h. `handoffs/`

The five surviving handoffs contain **no privileged instruction of any kind** —
grepped for `sudo`, `powermetrics`, `xcrun`, `xctrace`, `sysctl`, `pmset` and
Instruments, zero hits. Every privileged command in this repository lives in
`bench/apple/` and in the handoff that was deleted, which is what makes the
deletion invisible from `handoffs/` alone.

| what | verdict | evidence |
|---|---|---|
| `handoffs/performance_17_thermal_energy.md` | **NOT EXECUTABLE** — deleted | Section 1a. Cited by eight live places: `thermal_capture.sh:19,46,291,609`, `MACHINE_LOCK.md:82`, `PROFILE_PROTOCOL.md:727`, `docs/APPLE_THERMAL_ENERGY.md:448`, and `validation/manifests/handoffs.toml:123`. **The last one is why nobody noticed**: it lists the file in the `[archive]` retired table, so the validation planner does not flag it as missing while four other files still delegate to it. **Recoverable — see section 10.** |
| `tools/connectivity_audit.py` (`INDEX.md:5`) | **RUN** | **Verified by execution**: 12 findings, 7 EXPERIMENTAL, 5 PENDING, 0 DEAD. Wired into CI at `.github/workflows/ci.yml:124`. Note the drift: `consolidation_round.md:532` claims 6 findings and its table at `:490-496` logs 292 → 136 → 45. It is 12 today. |
| `tools/audit_integration.py` (`INDEX.md:7`) | **RUN** | **Verified by execution**: "docs/INTEGRATION_INVENTORY.md agrees with the tree". CI `ci.yml:126`. |
| `tools/api_snapshot.py --check` (`INDEX.md:25`) | **RUN** | **Verified by execution**: `ok`. CI `ci.yml:102`, pixi `check-api`. |
| `pixi run check-parity` (`INDEX.md:26`) | **RUN** | **Verified by execution**: `ok`, 148 rows. CI `ci.yml:80`. |
| `python3 tools/validation_plan.py --self-check` (`remaining_14:41`, called "The one command to run first") | **RUN** | **Verified by execution**: reports and exits `ok` — 9 tiers, 121 jobs, 46 subsystems, 6 gaps, 42 handoffs, 15 lanes. The document's own registered worry at `:41-43`, that it might raise rather than report, is discharged. |
| `tools/model_fixture_manifest.py --check` (`connect_22:38`) | **NOT EXECUTABLE** as documented, deliberately | **Verified by execution**: **49 problems**, ending "`compatibility/fixtures/checksums.json` does not exist … Until then this check fails, which is the correct state". `compatibility/fixtures/` holds only `README.md` and `manifest.toml`. The handoff's claim is confirmed and the failure is intended. |
| `tools/audit_python_compat.py` (`connect_22:40`) | **RUN** | **Verified by execution**: "no contradictions found in 3 note(s)". Not in CI, not a pixi task. |
| `tools/inspect_startup_artifacts.py --strict` (`connect_22:41`) | **NEVER RUN** | No CI job, no pixi task, no artifact. Not invoked here: it reads Mach-O and it is unclear whether it needs a build. |
| P1: patch `tools/with_build_lock.sh` to honor `MOJOTREES_BUILD_LOCK` (`remaining_14:56`) | **NEVER RUN / not applied** | Section 9d. `with_build_lock.sh:7` still hard-codes the path, and `validation_plan.py --self-check` still emits the note whose disappearance the handoff defined as the check. |
| P2: add `scipy` to `[feature.pytest.dependencies]` (`remaining_14:110`) | **ALREADY APPLIED** — the handoff is stale | `pixi.toml:139`, with a better rationale comment at `:131-138` than the patch proposed. A reader working the list top to bottom would regress it. |
| P3: add `pixi run test-c` to CI (`remaining_14:149`) | **NEVER RUN / not applied** | No `test-c` anywhere in `.github/workflows/ci.yml`. `validation/manifests/jobs.toml:565` still carries `provenance = "documented"` and "no CI job runs it". The task exists at `pixi.toml:92`, not `:35` as the handoff says. |
| P4: insert `tests/parallel/test_gpu_split_policy.mojo` into the `test` chain (`remaining_14:206`) | **SUPERSEDED, and discharged** | Obsolete twice over: the path is now `tests/test_gpu_split_policy.mojo`, and there is no `&&` chain to insert into — `pixi.toml:21` is `bash tools/run_tests.sh all`, glob discovery. `pixi.toml:20` names this exact file as the reason the glob exists. `jobs.toml:659-660` now reads `provenance = "ci"`, "First run 2026-08-15: 6 passed". |
| P5: document `MOJOTREES_BINNING_SELECT_MIN_ROWS` in README (`remaining_14:238`) | **NEVER RUN / not applied** | Section 9d. |
| `git show 21ff9fa^:handoffs/migration_20_device_policy.md` (`migration_20:6`) | **RUN**, works | **Verified by execution**: the commit exists and the blob prints. The only recovery command in any handoff, and it points at the same commit that ate `performance_17`. |
| "Run `tests/test_device.mojo` once that lane lands" (`migration_20:43`) | **DISCHARGED**, doc stale | The file exists and `tools/run_tests.sh` discovers it by glob, so it runs under `pixi run test`. The blocker it waited on is long past. |
| `connect_22:17`, "Nothing else under `tools/` is run by CI or by a pixi task" | **FACTUALLY WRONG today** | CI now runs four of them (`ci.yml:102,113,124,126`) and pixi defines six gates (`pixi.toml:40,46,47,48,51,56`). This is the document's central claim. |
| `connect_22:52`, "Recommendation for the coordinator (**not implemented**)" | **IMPLEMENTED, essentially verbatim** | `.github/workflows/ci.yml:118-126`, including the `continue-on-error: true` and the comment. |
| `connect_22:37` / `remaining_14:283`, "`compatibility/api_snapshot.json` … is not in the tree" | **FACTUALLY WRONG today** | The file exists and the gate is green. |
| `MOJOTREES_ABI_VERSION` (`connect_22:37`) | **NOT AN ENVIRONMENT VARIABLE** | It is a C macro, `capi/mojotrees.h:84`, `#define MOJOTREES_ABI_VERSION 3`. `consolidation_round.md:129` records the reclassification (commit `8e214a2`, "stop calling a C macro an env var"). The handoff that still lists it as an env var did not get the correction. |

### Two cross-cutting defects in `handoffs/`

1. **Every `tests/parallel/...` path is dead.** The directory was flattened.
   Affected: `connect_22:39`, `migration_20:139`, `remaining_14:195,203,206,230`,
   `consolidation_round:365,558,584`. The trap is that `python/tests/parallel/`
   **does** still exist and is live, so the name is dead only on the Mojo side.
2. **Every `pixi.toml:NN` line citation in the handoffs is wrong**, because
   `f2644e8` restructured the file. `connect_22:14` says `:18` (it is `:43`);
   `remaining_14:153` says `:35` (it is `:92`).

Neither is in this lane's file boundary. Both are recorded for the orchestrator.



---

## 10. What to do about it

### Executed in this session

| what | what happened |
|---|---|
| `bash bench/apple/thermal_capture.sh --self-check` | **FAILED**, exit 4, on `missing handoffs/performance_17_thermal_energy.md`. New finding; not recorded anywhere before this file. |
| `bash bench/apple/thermal_capture.sh --execute --phase cold_fit` | Refused, exit **3**, as documented. The refusal is real. |
| `bash bench/apple/thermal_capture.sh --print-plan --phase sustained` | Printed a plan with `"executed": false` and `"measurement_path": "absent"`. Confirms it is a plan printer. |
| `bash bench/apple/metal_capture.sh --self-check` | **PASSED**. All eleven checks green: `xctrace` 16.0 present, the `Metal System Trace` template installed, `python3` 3.14.6, the reader imports the standard library only, "no privileged command anywhere in this script". |
| `bash bench/apple/metal_capture.sh --print-plan` | Printed the four commands it would run. The third is byte-for-byte the `xcrun xctrace record --template 'Metal System Trace' --no-prompt --target-stdout - --launch --` command recorded as the Method in `bench/results/metal_timeline_2026-08-15/header.txt`. So the wrapper reproduces the procedure that was actually executed. |
| `python3 bench/apple/metal_timeline.py --help` | Works. Argument parser and full docstring print. |
| `python3 tools/api_snapshot.py --check` | **`ok`**. The snapshot is not stale, contradicting `SESSION_QUEUE.md:247`. |
| `uptime`, `ps -Ao pcpu,comm \| sort -rn \| head`, `pmset -g therm`, `sysctl hw.perflevel0.physicalcpu` | All four work. `pmset -g therm` returns the documented null. |
| `grep -rn "canary"` over the repository | One hit, the instruction itself. No referent. |
| `grep -rn HostGradientStage src/ bench/ tests/` | Confirms the dead-code claim in `SESSION_QUEUE.md:278`. |
| pixi task cross-check of all four documents | Sixteen task names, zero missing. |
| `python3 tools/connectivity_audit.py` | 12 findings, 7 EXPERIMENTAL, 5 PENDING, 0 DEAD. `consolidation_round.md:532` claims 6; it has drifted up. |
| `python3 tools/audit_integration.py` | Clean: "docs/INTEGRATION_INVENTORY.md agrees with the tree". |
| `python3 tools/check_parity.py` | `ok`, 148 rows. |
| `python3 tools/validation_plan.py --self-check` | `ok`. 9 tiers, 121 jobs, 46 subsystems, 6 gaps, 42 handoffs, 15 lanes. Discharges the registered worry at `remaining_14:41`. Still emits the P1 note, which is how P1 is known to be unapplied. |
| `python3 tools/model_fixture_manifest.py --check` | **49 problems**, correctly — `compatibility/fixtures/checksums.json` does not exist and the script says the failure is the correct state. |
| `python3 tools/audit_python_compat.py` | "no contradictions found in 3 note(s)". |
| `git show 21ff9fa^:handoffs/performance_17_thermal_energy.md` | **Recovers the deleted file, all 452 lines, with the privileged commands intact.** This is the remedy for section 1a. |

Nothing was timed. No benchmark, training run, or `sudo` command was executed.

### Fixed in this branch, inside the file boundary

Thirteen edits across the four documents. Every one replaces a claim the audit
found false, or deletes an instruction with no referent. No claim was weakened
and no rule was relaxed.

**`PROFILE_PROTOCOL.md`**

1. Checklist item 2 — **the canary ratio deleted.** It has no referent and a
   ten-line checklist cannot afford an item nobody can perform. Replaced by the
   accurate statement of what `uptime` and `ps` do and do not establish: they
   settle contention, and the regime label still comes from the arms moving
   together, per A3.
2. A1 — corrected from "No session has ever done this and none could have" to
   what actually happened, naming `profile_2026-08-15/header.txt` and its
   `thermal-PENDING` plan, and stating why a followed-but-inert instruction is
   worse than an unfollowed one.
3. A1 — the dangling `handoffs/performance_17_thermal_energy.md` pointer
   replaced by `docs/APPLE_THERMAL_ENERGY.md`, the `git show` recovery command,
   and this file, with the deletion commit and the failing self-check named.
4. A1 — the "both campaigns now do this" claim qualified to what the results
   corpus actually shows.
5. C-ops — the "There is no measurement lock" bullet corrected. The lock was
   reinstated the same night, is in force, and `MACHINE_LOCK.md` is the
   authority. The bullet's own justification is turned on itself rather than
   deleted, because the error is symmetric.

**`MACHINE_LOCK.md`**

6. The instrument section — "no session has ever followed it" corrected the same
   way as (2), with the artifact named and a warning to read that directory's
   regime labels accordingly.
7. The same section — the dangling handoff pointer replaced as in (3).
8. The `uptime` / `ps` prescription — "that is what both orchestrators now use"
   qualified: one directory has the `uptime` half at both ends, and no file
   anywhere has the `ps` half.

**`SESSION_QUEUE.md`**

9. A status banner at the top recording that the Session III queue was drained
   on 2026-08-16, with each item's verdict and where it is written up, and the
   one item (M2.4 at 250,000) that was not run. The commands are kept, because
   they are how the items get re-taken.
10. The S1 status table — 250,000 and 50,000 are measured, S1 is **closed**, and
    the surviving gap is recorded as a **code** gap: the plane is the default by
    decision and opt-in in source.
11. The `api_snapshot.json` bullet — the gate passes, all five named variables
    are present, and the commit that did it is named. The bullet's second
    question, whether these knobs get declared, is the part that is still open
    and is now the part that stands, with the two limitations of the snapshot
    stated so the next reader does not over-trust it.

**`bench/README.md`**

12. The `TASKS_PER_CORE` instruction — updated to name
    `MOJOTREES_CPU_TASKS_PER_CORE`, so the sweep is a loop over an environment
    variable rather than a rebuild per point, and marked as never run.
13. "which is how the defaults in that module were chosen" — corrected to say
    which of the three knobs produced a filed table and which default is still a
    derived bound, resolving a contradiction with a note twenty lines below it.

### Requires human action, with the exact commands

Nothing here should be attempted by a session. These are recovered verbatim
from the deleted `handoffs/performance_17_thermal_energy.md`, lines 335-396,
via `git show 21ff9fa^:handoffs/performance_17_thermal_energy.md`. They are the
only complete listing that exists, and they exist only in git history.

First, the unprivileged question that should be answered before any of the
rest — whether `CPU_Speed_Limit` moves at all on this machine under load. If it
does not, then every "fast window" and "slow window" label in
`bench/results/**` remains an inference from effect, exactly as
`PROFILE_PROTOCOL.md` A1 and `MACHINE_LOCK.md:99-103` admit, and no amount of
`pmset` will fix it:

```sh
while :; do
    printf '%s ' "$(date -u +%FT%TZ)"
    pmset -g therm | tr '\n' ' '
    printf '\n'
    sleep 1
done | tee /tmp/therm.log
```

Then the privileged ones. All need root, all report the whole machine, and
their output names every running process — read the privacy section of
`docs/APPLE_THERMAL_ENERGY.md` before publishing any of it:

```sh
# 60 s idle baseline at 200 ms, 300 samples, to a file.
sudo powermetrics --samplers cpu_power,gpu_power,thermal \
    -i 200 -n 300 -f plist -o /tmp/idle_baseline.plist

# The same sampler around a workload, for the same window length.
sudo powermetrics --samplers cpu_power,gpu_power,thermal \
    -i 200 -n 300 -f plist -o /tmp/workload.plist

# Which process dominated a window. Unitless scores, never joules.
sudo powermetrics --samplers tasks --show-process-energy \
    -i 1000 -n 60 -o /tmp/tasks.txt

# Around a long run, to stop the display and the disks from sleeping.
caffeinate -dimsu <the command>
```

One caveat travels with all three `powermetrics` lines, and it is the deleted
handoff's own, at its line 388: the parser in `bench/apple/suite.py` **has
never seen output from a real machine**, and its documented failure mode is an
empty key list with `available: true` rather than a plausible zero. Check the
key names it matched against the raw plist before publishing a joule.

Two things a person running these would settle that nothing else can, both
currently **estimated** in this repository and quoted as if better:

- whether `pmset -g therm` ever moves on this machine under sustained load. If
  it does not, then every "fast window" and "slow window" label in
  `bench/results/**` remains an inference from effect, exactly as
  `PROFILE_PROTOCOL.md` A1 and `MACHINE_LOCK.md:99-103` admit.
- whether the two-to-threefold drift is the GPU performance state. The Metal
  traces make this the leading candidate (`metal_timeline_2026-08-15/header.txt`
  records two captures at Minimum and two at Maximum), and only a power sampler
  can confirm it.

### Left for the orchestrator, outside this file boundary

Ordered by how much a reader is currently misled.

| what | where | why it is not mine |
|---|---|---|
| **Restore `handoffs/performance_17_thermal_energy.md`.** One command recovers it intact: `git show 21ff9fa^:handoffs/performance_17_thermal_energy.md > handoffs/performance_17_thermal_energy.md`. That fixes `thermal_capture.sh`'s self-check, the `--execute` refusal message, and three documents' dangling pointers at once, and it restores the repository's only complete listing of its privileged commands. If it is instead meant to stay deleted, then `validation/manifests/handoffs.toml:123` must stop archiving it silently and the four delegating citations must be rewritten | `handoffs/`, `validation/manifests/handoffs.toml` | outside my boundary; the highest-value single fix in this audit |
| `validation/manifests/handoffs.toml:123` lists the deleted file in the `[archive]` retired table, which is **why nobody noticed**: the validation planner treats it as intentionally gone while four live artifacts still delegate to it | `validation/manifests/handoffs.toml` | outside my boundary |
| The resident plane is still `getenv(...) == "1"`, opt-in, after `session3` closed S1 and declared it the default | `src/mojotrees/gpu_resident_round.mojo:329` | `src/`, three lanes working there |
| `bench/apple/thermal_capture.sh --self-check` fails on the missing handoff (`:320`). Restoring the file per row 1 fixes it; failing that, repoint `HANDOFF_PATH` (`:46`) and the two refusal messages (`:19`, `:291`, `:609`) at `docs/APPLE_THERMAL_ENERGY.md` | `bench/apple/thermal_capture.sh` | a script, not a doc |
| `MOJOTREES_STARTUP_REPORT_FD` is read by nothing. Delete it or implement it | `compatibility/api_snapshot.json`, `compatibility/DRIFT_REPORT.md:218`, `python/mojotrees/diagnostics.py:225` | `src/`-adjacent |
| `MOJOTREES_BUILD_LOCK` is inert: `tools/with_build_lock.sh:7` hard-codes the path. Patch P1 in `handoffs/remaining_14_validation_plan.md:56` is written and unapplied | `tools/with_build_lock.sh` | a script |
| `docs/STARTUP_LATENCY.md:260,261,262,280` invoke `pixi run bench-startup`, which is not a task. Either define it or rewrite the four commands onto the reachable form the same file already uses at `:272` | `docs/`, `pixi.toml` | outside my boundary |
| Results files record arm labels, never the environment that produced them (section 9c). Emitting the resolved `MOJOTREES_*` values into `json_summary` would close it at one site | `bench/bench_train_gpu.mojo` | `bench/` driver, not a doc |
| `handoffs/connect_22_audit.md:17` ("Nothing else under `tools/` is run by CI or by a pixi task") and `:52` ("not implemented") are both false today; `:37` and `remaining_14:283` say the API snapshot is not in the tree, and it is | `handoffs/` | outside my boundary |
| `handoffs/remaining_14_validation_plan.md` P2 is already applied with a better rationale than the patch proposes. A reader working the list top to bottom would regress it | `handoffs/` | outside my boundary |
| Every `tests/parallel/...` path in `handoffs/` is dead (the Mojo side was flattened; `python/tests/parallel/` still lives, which is the trap). Every `pixi.toml:NN` citation in `handoffs/` is off, because `f2644e8` restructured the file | `handoffs/` | outside my boundary |
| Ten distributed environment variables are read and have no test and no measurement (section 9g) | `src/mojotrees/distributed_transport.mojo`, `python/mojotrees/_dask_runtime.py` | `src/` |
| `session3_2026-08-16/RESULTS.md:403` lists M2.5 as "untaken" while `:157-171` of the same file reports its three pairs and its null verdict | `bench/results/session3_2026-08-16/RESULTS.md` | outside the four documents I may edit |
| `sweep2_2026-08-15/RESULTS.md:5` records "No thermal warning recorded" as a machine condition. That is `pmset -g therm`'s null answer on Apple silicon and means nothing; it reads as a thermal measurement | `bench/results/sweep2_2026-08-15/RESULTS.md` | same |
| `docs/GPU_VALIDATION.md:456` still carries `<paste the full gpu-validate output>`, so the Apple Metal validation run has no filed artifact | `docs/` | same |
| `docs/STARTUP_LATENCY.md:260-280` gives four `pixi run bench-startup` commands. **Verified by execution**: `bench-startup` is not a task in `pixi.toml`. The document admits this at its own `:265` ("A startup benchmark under `bench/` and the `bench-startup` pixi task **do** ..."), so it is self-flagged rather than misleading, but four commands in a document cannot be run | `docs/` | same |

---

## 11. What this audit could not establish

Stated plainly, because the point of the exercise is to stop reading silence as
confirmation.

- **Whether `bench/apple/metal_capture.sh` itself was ever executed end to end,
  as opposed to the raw `xctrace` command it wraps.** The evidence is
  ambiguous and I could not settle it without taking a capture, which is a
  timing run. In favor of the wrapper: `metal_timeline_2026-08-15/header.txt:5-6`
  names both scripts as "the capture and reduction tools". Against it: the
  header's Method block records the bare `xcrun xctrace record` line, and the
  reduction output at `metal_200000_50_reg_gpu.txt:1` names a trace at
  `.../scratchpad/traces/mst1.trace`, which is not the
  `bench/apple/metal_traces/` path the wrapper defaults to. Most likely reading,
  labelled **estimated**: the capture was taken by hand and reduced with
  `metal_timeline.py`, possibly through `--analyze-only`. What is certain, and
  **verified by execution**, is that the wrapper's self-check passes today, its
  plan matches the recorded procedure exactly, and it needs no privileges.
- **Whether a variable absent from `bench/results/**` was truly never
  measured.** Absence is strong evidence and is not proof: a measurement could
  have been taken and not filed. Section 9c shows the gap is real in the other
  direction too — four knobs *were* measured and their artifacts do not name
  them. So the counts in the scoreboard are a lower bound on RUN and an upper
  bound on NEVER RUN. This is precisely what `PROFILE_PROTOCOL.md`'s "write the
  results file before interpreting any of it" exists to remove, and it half
  works: for every knob with a filed result the file is nameable, but the file
  names an arm rather than a knob.
- **Whether `tools/inspect_startup_artifacts.py --strict` works.** Not
  invoked: it reads Mach-O and it was not clear whether it needs a build, and a
  build on a box under a timing lock is not mine to start. Recorded as NEVER
  RUN on the CI and pixi evidence alone.
- **The five suites and six Python suites that `handoffs/consolidation_round.md`
  counts as evidence.** `validation_plan.py --self-check` reports that no
  manifest job runs 33 of the Mojo suites and six of the pytest files. Whether
  they pass today needs a build. The handoff's recorded counts are from runs
  nobody can currently reproduce cheaply.
- **Whether the four documents' *prose* claims hold.** This audit covers
  executable instructions. Claims like "our marginal cost per row now equals
  LightGBM's" (`bench/README.md:118`) are fitted quantities with their own
  provenance and are out of scope here.
