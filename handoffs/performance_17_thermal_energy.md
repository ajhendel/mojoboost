# Handoff: MacBook thermal, energy, and sustained performance (task 17)

Lane 17 of the parallel performance round. Protocol, schema, and an inert
planner. No measurement was taken, no sampler was started, no privileged
command was run, and no thermal record exists. Nothing outside the five
assigned paths was touched.

## Files this lane owns

| Path | What it is |
|---|---|
| `docs/APPLE_THERMAL_ENERGY.md` | The protocol. Definitions, the eight phases, instruments, conditions and contrasts, attribution rules, invalid runs, publication rules, prohibitions, privacy, open items. |
| `bench/apple/thermal_schema.json` | JSON Schema 2020-12 for a thermal record. Every measured quantity is nullable, every null carries a reason, and every record links to a commit, a chip variant, a memory size, an OS build, a toolchain, a workload, a quality result, and a benchmark record. |
| `bench/apple/thermal_capture.sh` | The planner. Validates arguments and prints the commands a run would issue. Takes no sample. `--execute` is parsed and refused. |
| `launch/APPLE_BENCHMARK_REQUEST.txt` | The post asking owners of M1 through M5 Macs to contribute runs, with the privacy rules a contributor sees before running anything. |
| `handoffs/performance_17_thermal_energy.md` | This file. |

`bench/apple/` is shared. Lane A8 owns `suite.py` and `schema.json`, lane 18
owns `unified_memory.mojo`, lanes 13 and 14 own `cpu_plan.json` and
`histogram_plan.json`. Nothing here touches any of them, and the two schemas
are siblings that share no `$ref`.

One thing about the shared checkout worth knowing. Partway through this lane,
commit `9a9c8d1` ("Prepare packaging and parallel optimization work"), made by
another lane, swept up a mid-flight snapshot of four of these five files. What
is in that commit is not what this lane produced. The finished versions are the
working tree, uncommitted, as the brief requires. Whoever commits next should
take the working tree rather than assume `9a9c8d1` already has this lane's
work, and should expect the differences described under "Defects found by
static review and fixed", all of which were made after that commit.

## What this lane deliberately did not build

A measurement path. The script plans a run and cannot take one, on purpose.
The commands a thermal run wants to issue include `sudo powermetrics`, and a
script that can be talked into that by a mistyped flag is a script that
eventually will run it on someone else's laptop. The plan is printed so a
person can read it, decide, and run the parts they want deliberately.

The file is left non-executable, mode 0644, for the same reason. Every
documented invocation is `bash bench/apple/thermal_capture.sh ...`, so nothing
is lost, and a script about thermal measurement that cannot be started by
habit from a shell history entry is slightly harder to start by accident. Set
the executable bit if a future version gains a measurement path and that path
has been reviewed.

The consequence is that this lane produces no number and enables no claim. That
is the correct state. The alternative was a harness nobody has watched, writing
records nobody can check, which is how the timings currently in
`bench/README.md` came to be labeled unquotable in the first place.

## Validation actually run

Static only. No Mojo, no pixi, no Python, no pytest, no build, no benchmark, no
`powermetrics`, no `pmset` sampling, no profiler, no CI, no background job.

Running `thermal_capture.sh` does not violate that. The script executes nothing
it prints; the self-check contains a grep that proves it, described below.

```
$ /bin/bash -n bench/apple/thermal_capture.sh          # bash 3.2, the macOS default
$ bash -n bench/apple/thermal_capture.sh               # bash 5.2
$ jq empty bench/apple/thermal_schema.json             # schema parses
$ bash bench/apple/thermal_capture.sh --self-check     # 6 checks, all pass
$ bash bench/apple/thermal_capture.sh --list-phases
$ bash bench/apple/thermal_capture.sh --version
$ bash bench/apple/thermal_capture.sh --help
$ bash bench/apple/thermal_capture.sh                  # default plan, human readable
$ bash bench/apple/thermal_capture.sh --device gpu --threads 4 --energy \
      --process-energy --phase repeat_series --phase idle_baseline \
      --phase background_control
$ bash bench/apple/thermal_capture.sh --print-plan --all-phases | jq .   # plan parses
$ bash bench/apple/thermal_capture.sh --execute        # exits 3, refuses
```

Eleven argument-validation cases were exercised and each produced the intended
exit 2 with a specific message: unknown phase, unknown option, a duration below
one bucket, a duration above the ceiling, a sampler interval below 50 ms, a
non-numeric thread count, a run id with a space, a missing option value, a
non-JSON `--link-record`, `--process-energy` without `--energy`, and `--energy`
without the `idle_baseline` phase.

What `--self-check` checks:

1. The protocol, the parent protocol, the schema, and this handoff all exist.
   A plan that points at a document nobody can read is not a plan.
2. The three parallel arrays of the phase catalog are the same length, so a
   window is never printed against the wrong phase.
3. Every phase id the script knows appears in the schema's phase enum. This is
   the check that keeps the script and the schema from drifting apart, and it
   is the reason both files can be reviewed independently.
4. The schema contains no `estimated` value anywhere. The protocol's central
   prohibition is checked in the file rather than trusted.
5. The schema parses, when `jq` is installed. Its absence is reported rather
   than quietly counted as a pass.
6. No sampler or privileged command appears in command position anywhere in the
   script. `sudo`, `powermetrics`, `pmset`, `caffeinate`, and `yes` are searched
   for at the start of a line or immediately after a separator, which is where
   a command would have to be; everything the script prints goes through
   `emit_cmd` or `emit_priv` and can never be in that position.

The self-check runs on Linux as well as macOS, because it reads files and never
touches a power tool. That matters for the CI recommendation below.

## Defects found by static review and fixed

All five were in this lane's own files. Each is listed because each would have
cost something.

1. **The privileged-command guard matched its own failure message.** Check 6
   originally searched for the bare word `sudo` outside comments, and the line
   that reports the failure contains that word. The self-check could therefore
   never pass, which trains whoever hits it to ignore the check. It now
   searches for command position.
2. **`--duration` widened the idle baseline.** A single duration flag applied to
   every timed phase, so `--duration 900` turned a 60 s baseline into a 900 s
   one and silently changed what the baseline was a baseline of. There are now
   separate `--baseline-seconds` and `--cooldown-ceiling` options, and
   `--duration` applies to the load phases only.
3. **Phases were planned in the order they were typed.** `--phase sustained
   --phase idle_baseline` planned the baseline after the workload, on a machine
   the run had just heated. That is the same defect lane A8 had to fix in
   `suite.py`, and it is now impossible here: the selection is canonicalized
   into protocol order before anything is printed.
4. **The printed teardown for the background control was `pkill -f yes`,** which
   pattern-matches the command line of every process on the machine. The plan
   now prints a pid-file form that kills exactly the load it started.
5. **`printf %q` rendered `--samplers cpu_power,gpu_power,thermal` with escaped
   commas.** Correct shell, hard to read, and a plan nobody reads carefully is
   the failure mode this whole lane is built against. Quoting is now
   shlex-style: safe words bare, everything else single-quoted.

## The measurement path

The phases need a driver that fits models in a loop and scores them. It does
not exist. Two candidate homes, and neither belongs to this lane.

**Recommended: a mode inside `bench/apple/suite.py`.** That file already has
everything the thermal phases need and none of it should be written twice: the
synthetic workload generators with their `data_digest`, the engine adapters,
the scorers, the prediction digests, the idle gate with the ancestor filter,
the `pmset -g therm` reader, and the `powermetrics` parser. What would be added
is a phase loop, a bucketed throughput accumulator, and a writer that emits
`thermal_schema.json` instead of `schema.json`. Concretely:

- a `--thermal` mode with `--phase` repeatable, reusing the existing `--energy`,
  `--interval`, and `--allow-busy` plumbing,
- a `run_phase(phase_id, ...)` that loops fits until its stopping rule and
  records each fit with its quality value and prediction digest,
- reuse of `PowerSampler` around each phase, with `window_definition` set to
  `phase`, and a second use around the idle baseline with
  `window_definition` set to `idle_baseline`,
- a `benchmark_link` filled from the suite run in the same session, which is
  what makes `same_session` true and therefore what makes the two records
  mergeable at all.

That is an addition to a file this lane does not own, so it is described here
rather than made. It is perhaps two hundred lines and no change to either
schema.

**Alternative: a standalone `bench/apple/thermal_runner.py`.** Cleaner
ownership, and it duplicates the generators, the scorers, and the gate. The
duplication is the objection: two idle gates drift, and the one nobody looks at
is the one that stops working.

A Mojo driver was considered and rejected for the loop itself, because the loop
is orchestration and scoring rather than computation. The Mojo-first rule bites
in the other direction here and is respected: the fits, the training, and any
future per-fit phase counters stay native. Shell and Python collect operating
system telemetry around them and must not change what the trainer does. Nothing
in this lane's files runs inside a fit.

## Future integration points

| Lane or file | What it would give this protocol | What happens without it |
|---|---|---|
| `bench/apple/suite.py` (lane A8) | The fit driver above, and a benchmark record to link to in the same session | Every phase stays unimplemented; the protocol is a document and the script is a planner |
| Startup and first-use latency lane | Native per-fit phase counters, which would decompose cold minus warm into compile, context creation, and first allocation | Cold minus warm stays a single opaque number, which is honest but not actionable |
| Persistent GPU runtime lane | Whether a warm fit reuses a device context at all, which is the mechanism the warm and repeat phases are measuring | The phases still measure the gap; they cannot attribute it |
| Apple CPU and GPU policy lanes (13, 14) | A sustained ratio per policy, which is the only way to see that a policy that wins a thirty second benchmark loses a twenty minute one | Policy decisions are made on burst timings alone |
| Device selection lane | Crossover rules that hold under sustained load rather than only at the first fit | Selection rules stay conservative, which is correct with no data |
| `docs/HARDWARE_CONTRIBUTORS.md` (lane 11) | The submission flow, issue template, and attribution terms, which this lane defers to rather than duplicating | The launch post has to carry submission mechanics that will drift from the authoritative copy |
| `docs/PLATFORM_MATRIX.md` | Where a status lives. A thermal record must never move a platform status on its own; it describes conditions, not support | No effect; stated so nobody tries |

## External edits required, not made here

Every one of these is in a file this lane does not own.

### 1. pixi tasks

Under `[feature.bench.tasks]` in `pixi.toml`:

```toml
# Static validation of the thermal protocol's script and schema. No sampler,
# no privileged command, no measurement. Standard tools only.
thermal-check = "bash bench/apple/thermal_capture.sh --self-check"

# The plan, for reading before committing an afternoon and a laptop.
thermal-plan = "bash bench/apple/thermal_capture.sh --print-plan --all-phases"
```

`thermal-check` is the only one that belongs in CI, in the same spirit as
`check-parity` and `bench-apple-check`. It needs bash, grep, and optionally
`jq`, runs on Linux and macOS alike, and takes well under a second. It must not
be confused with a run: a green `thermal-check` says the script and the schema
agree, and says nothing about any machine.

Nothing else from this lane may ever run in CI. A GitHub runner is a shared,
virtualized, thermally uncontrolled machine, which makes it the one environment
where every number this protocol produces is meaningless.

### 2. Results directory

`bench/apple/thermal_results/` is where a future run would write. It does not
exist and this lane did not create it. Decide before the first run whether
records are committed. The recommendation is yes, committed, for the same
reason as the benchmark records: the protocol requires publishing the record
alongside any table drawn from it, and they are small. If not, add
`bench/apple/thermal_results/` to `.gitignore`, which this lane did not touch.

### 3. Cross-references

- `docs/APPLE_GPU_BENCHMARK_PROTOCOL.md`, owned by lane A8, should gain a line
  pointing at `docs/APPLE_THERMAL_ENERGY.md` and saying which document owns
  which question. The thermal document already points back.
- `bench/README.md` should point at both.
- `launch/README.txt` should index `APPLE_BENCHMARK_REQUEST.txt` with its
  ordering caveat: post it only after the benchmark suite has completed one
  smoke run somewhere, so the first volunteer is not the person who discovers
  the harness is broken.
- `docs/HARDWARE_CONTRIBUTORS.md` could accept an optional thermal record
  attachment on the same issue template. Optional is the right word; a
  contributor who runs only the benchmark suite has contributed fully.
- `README.md` gets nothing. Nothing in this lane produces a number.

## Publication rules

Repeated here in the form a maintainer needs at the moment of writing a
sentence, rather than at the moment of reading a protocol.

1. Publish the thermal record and the benchmark record it links to, alongside
   any table, curve, or sentence drawn from either.
2. Name the chip variant, memory size, OS build, commit, and condition set in
   the same place as the number. "M4" is not a machine.
3. State the window length beside every average power figure and the bucket
   size beside every throughput figure.
4. Quote a sustained ratio only from a phase that ran its full declared window.
5. Never write a sentence whose subject is a process and whose object is a
   joule. Apple's tooling reports the machine, and the energy impact score is a
   unitless heuristic that may support "mojoboost dominated this window" and
   nothing arithmetic.
6. Never publish an energy comparison between engines that reached different
   held-out quality without stating both losses.
7. Never compare across chip variants, OS versions, ambient conditions, or
   protocol versions. Within one record, within a declared contrast, or not at
   all.
8. Never publish from a record whose `validity.quotable` is false, with or
   without a caveat. Rerun it.
9. A phase that did not run appears as not run in any table containing the
   other phases.
10. Read the record for machine identifiers and paths before it leaves the
    machine.

## Invalid-run criteria

Enumerated in `validity.invalid_reasons` so the judgment travels inside the
record. `idle_gate_failed`, `background_process_detected`,
`background_control_not_rejected`, `dirty_tree`, `no_quality_metric`,
`benchmark_link_missing`, `workload_mismatch`, `energy_without_baseline`,
`sampler_keys_empty`, `phase_aborted`, `record_edited`, `condition_undeclared`,
`not_executed`.

Two of those deserve a note.

`background_control_not_rejected` invalidates the record because the negative
control is the only evidence in it that the idle gate works at all. A gate
nobody has watched fail is a gate nobody knows about. If the control passes the
gate, every other phase in that record was measured under a gate with no
demonstrated teeth.

`condition_undeclared` covers the case that will actually happen: someone plugs
in the charger during a battery run, or moves the machine from a desk to a lap.
The record has no way to represent a condition that changed mid-run, on
purpose, because averaging across a condition change is exactly the thing this
protocol exists to prevent.

What is deliberately not invalid: a throttled run. Throttling invalidates a
timing under the parent protocol and is the measurement under this one.

## Privacy risks in system reports

The protocol has the operator-facing version. This is the maintainer-facing
one, including the risks that come from the tooling rather than from the
record.

- `system_profiler SPHardwareDataType` carries the serial number and hardware
  UUID; `SPSoftwareDataType` carries the computer name and the logged-in user
  name, which is commonly a person's real name. Neither is read by this
  protocol and neither has a field in the schema. If a future harness adds
  either, it is adding personal data to a public file.
- `uname -a` includes the host name. The plan prints `uname -srm` for exactly
  that reason.
- `powermetrics --samplers tasks` lists every running process on the machine,
  which is far broader than the idle gate's filtered list, and with `-o` it
  writes them to a file. That output is the single most disclosive thing this
  protocol touches. The schema keeps only a target process, a share, and a
  short list of other top consumers, and says to redact by replacing a name
  rather than deleting an entry so the count stays honest.
- `git status --porcelain` in a dirty tree lists file names, which can name an
  unreleased feature. It is run to set `git_dirty`, and a dirty run is not
  quotable anyway.
- `invocation.argv` and `invocation.cwd` contain absolute paths and therefore a
  home directory and a user name. So does the plan the script prints, which is
  worth knowing before pasting a plan into a public issue.
- `ioreg -rn AppleSmartBattery` and `nvram` carry serial numbers. Not read, and
  not to be attached to an issue.
- `sysdiagnose` collects system logs, network names, and an installed software
  inventory. Never required by this protocol. A maintainer who asks a
  contributor for one is asking for something much larger than a benchmark.
- Ambient temperature notes and room descriptions are a person's whereabouts.
  Optional, and left to whoever runs it.

For contributed records, the reviewing maintainer reads the file before merging
it, not after. A record is small and this takes a minute.

## Exact commands a human may deliberately run later

None of these have been run. They are listed here rather than in the protocol
so that the protocol cannot be copied out of context into a terminal. Run them
after reading the privacy section, and read what they write before publishing
it.

Unprivileged, safe to run at any time:

```sh
# Machine identity and toolchain.
sysctl -n machdep.cpu.brand_string hw.model hw.memsize \
    hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu hw.logicalcpu
sw_vers -productVersion
sw_vers -buildVersion
uname -srm
system_profiler SPDisplaysDataType -json

# Provenance.
git rev-parse HEAD
git status --porcelain

# The idle gate, by hand.
sysctl -n vm.loadavg
pmset -g therm
pmset -g batt
pmset -g
ps -Ao comm= | sort | uniq -c | sort -rn | head -20
```

The one question worth answering before anything else is whether
`CPU_Speed_Limit` moves at all on the machine in question. A one second poll,
written to a file, while something else loads the machine:

```sh
while :; do
    printf '%s ' "$(date -u +%FT%TZ)"
    pmset -g therm | tr '\n' ' '
    printf '\n'
    sleep 1
done | tee /tmp/therm.log
```

Privileged, and therefore a deliberate decision each time. `powermetrics`
requires root, reports the whole machine, and its output names every process
running:

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
```

Before publishing a joule from any of that, check the key names the parser
matched against the raw plist. Lane A8's `powermetrics` parser has never seen
output from a real machine, and its documented failure mode is an empty key
list with `available: true` rather than a plausible zero.

Around a long run, to stop the display and the disks from sleeping:

```sh
caffeinate -dimsu <the command>
```

The negative control, which should make the idle gate fail:

```sh
sh -c 'yes > /dev/null & echo $! > /tmp/mojoboost_thermal_load.pid'
# re-run the gate here; it must fail
kill "$(cat /tmp/mojoboost_thermal_load.pid)" && rm -f /tmp/mojoboost_thermal_load.pid
```

## Risks

1. **Nothing has been validated against a machine.** The protocol, the schema,
   and the plan are all reasoning from documented tool behavior. The first
   real run will find something wrong, and the first thing to check is whether
   `pmset -g therm` reports anything but 100.
2. **Both throttle signals may be flat.** If `CPU_Speed_Limit` never moves and
   `powermetrics` thermal pressure stays nominal through a run whose throughput
   halves, this protocol cannot attribute the decline to thermals. The honest
   response is to record the sustained ratio, record that both instruments
   stayed flat, and say the mechanism is unestablished. It is not to infer
   throttling from the timings, which is the one thing the definitions section
   forbids.
3. **The sampler is a load.** `powermetrics` at 200 ms wakes the machine
   repeatedly, and so does a one second `pmset` poll. The idle baseline is
   sampled with the same instrument at the same interval, so the subtraction
   cancels most of it, but only if both really do run the same sampler. A
   baseline taken without the sampler that the workload phase used is not a
   baseline, and the 50 ms floor on `--interval-ms` exists for the same reason.
4. **A phase can outlast its window.** Ten fits of a 200,000 by 100 workload on
   a base chip may take longer than the sustained window it sits next to, and
   `throttle_probe` at its ceiling is an hour. Read the plan before starting,
   and expect an afternoon.
5. **`powermetrics` parsing is inherited risk.** This protocol depends on lane
   A8's parser, which its own handoff flags as unverified, including its
   assumption about which keys are milliwatts and which are watts.
6. **Bash 3.2.** The script is written to run under the macOS system bash and
   was parse-checked under both 3.2 and 5.2. It has not been run under 3.2,
   only parsed, because the machine here has a newer bash first on `PATH`.
7. **Contributed records cannot be verified.** A record from a stranger's laptop
   is a report. It is published as a report, with the contributor named if they
   want to be, and it never becomes the sole evidence for a claim in the
   README.
8. **The protocol is long enough that people will skim it.** The mitigations are
   that the script refuses the combinations that produce unusable records at
   plan time, and that the schema carries the prohibitions as fields, including
   `process_attribution.joules_claimable`, which is a constant false so the
   rule travels inside every record rather than only in a document.

## State of the numbers

None. No thermal record exists. No cell of the status table in
`docs/APPLE_THERMAL_ENERGY.md` is filled. Nothing in this lane authorizes a
statement about mojoboost's power consumption, thermal behavior, sustained
throughput, battery behavior, or energy efficiency on any Mac, and the existing
timings in `bench/README.md` and `docs/GPU_VALIDATION.md` remain what they say
they are, taken on a loaded development machine.
