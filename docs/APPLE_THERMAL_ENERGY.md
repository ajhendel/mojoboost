# MacBook thermal, energy, and sustained performance

Thermal protocol version 1.0.0. Record schema `bench/apple/thermal_schema.json`,
version 1.0.0. Capture script `bench/apple/thermal_capture.sh`, version 1.0.0.
Parent protocol `docs/APPLE_GPU_BENCHMARK_PROTOCOL.md`, version 1.0.0.

A laptop is not a server. The number a MacBook produces in the first thirty
seconds of a fit is not the number it produces in the twentieth minute, the
number it produces on battery is not the number it produces on AC, and the
number it produces on a desk is not the number it produces on a duvet. This
document defines how mojoboost measures that, what each measurement is allowed
to be called afterward, and which quantities this repository refuses to
estimate.

It is a procedure, not a result. No run has been performed under it.

## Status

| Quantity | Machines run | State |
|---|---|---|
| Cold fit | 0 | not run |
| Warm fit | 0 | not run |
| Ten repeated fits | 0 | not run |
| Sustained throughput | 0 | not run |
| Throttle onset | 0 | not run |
| Machine energy per fit | 0 | not run |
| Average power over a window | 0 | not run |
| Battery against AC | 0 | not run |
| Lid and surface contrasts | 0 | not run |
| Cooldown recovery | 0 | not run |

Nothing here has been executed. `bench/apple/thermal_capture.sh` has never
taken a sample, and in this version it cannot. No thermal record file exists.
No sentence about mojoboost's power consumption, thermal behavior, or sustained
throughput on any Mac is currently supported by this repository.

## Relationship to the parent protocol

`docs/APPLE_GPU_BENCHMARK_PROTOCOL.md` answers "how fast, against what, at what
quality". This document answers "and what happens after ten minutes, and what
did it cost the battery". It inherits from the parent rather than restating it.

Inherited unchanged, and not repeated here.

- The idle gate. Load average, competing builds, Low Power Mode, and a clean
  `pmset -g therm` before the run.
- The rule that a throughput number without a quality number next to it is not
  a result.
- The rule that a run from a dirty tree is exploration and is not quotable.
- The prohibition on fabricating, estimating, extrapolating, or interpolating
  any number in a record file.
- Machine identity by `machdep.cpu.brand_string` plus `hw.model`, never by a
  marketing name.

Added here.

- Time as a first-class axis. The parent measures repetitions with cooldown
  between them, which is designed to keep thermal state out of the numbers.
  This protocol deliberately removes the cooldown and measures what the parent
  is protecting against.
- Conditions as a declared variable. Battery, lid, and surface are recorded
  fields, and two records that differ in exactly one of them are a contrast.
- An explicit boundary between machine-wide telemetry and per-process claims.

A thermal record does not replace a benchmark record and is not a source of
comparative timings. It links to a benchmark record and describes the state the
machine was in.

## The boundary that matters most

Apple's power tooling reports the machine. It does not report mojoboost.

`powermetrics` samples package, CPU, GPU, and ANE power for the whole system.
Every process running contributes to every joule it reports, including the
sampler, the window server, and whatever the operating system decided to index
that minute. This is why an idle baseline is mandatory and why the quantity
worth quoting is energy above idle rather than absolute joules.

`powermetrics --samplers tasks --show-process-energy` does print a per-process
number, and that number is not joules. It is an energy impact score, a weighted
composite of CPU time, wakeups, disk and GPU activity, computed by a model
whose weights Apple does not publish and which changes across releases. The
schema has a field for it, the field is labeled as a heuristic, and it carries
a `joules_claimable` flag that is always false. It may be used to see whether
mojoboost was in fact the dominant consumer during a window. It may never be
converted to energy, compared against another machine, or printed with a unit.

So there are exactly three legitimate energy statements this protocol can
support, and one it cannot.

1. "On this machine, in this window, the whole system drew a mean of X watts
   while mojoboost fit this model, against Y watts idle immediately before."
2. "Energy above idle for this fit was Z joules on this machine, under an idle
   gate that recorded no competing process."
3. "At matched held-out quality, the GPU path used less energy above idle than
   the matched-thread CPU path on the same machine in the same session."
4. Not supportable. "mojoboost used Z joules." No public Apple silicon tooling
   attributes energy to a process, and this repository does not pretend
   otherwise.

Statement 3 is the interesting one, and it is the one a laptop makes worth
asking. A path that is slower on wall clock and materially cheaper in energy at
the same loss is a real result on a machine that runs on a battery.

## Definitions

These words are used precisely everywhere in this repository from here on.

**Cold fit.** The first fit of a model in a freshly started process, on a
machine that has been idle for at least twenty minutes and whose thermal sample
is clean. It contains everything a user pays once, which on the GPU path
includes device context creation, kernel or shader compilation, and first
allocation. Cold is a property of both the process and the machine, and this
protocol records both.

There is a third kind of cold, a cold page cache and a cold kernel, which is
only reachable across a reboot. This protocol does not claim it, does not
require a reboot, and records `machine_idle_minutes_before` so a reader knows
what kind of cold was actually achieved.

**Warm fit.** A fit in a process that has already completed at least one fit of
the same shape, taken immediately, with no cooldown. It is the number a user in
a notebook session experiences on the second cell execution.

**Repeat series.** Ten fits of the same shape back to back in one process, no
sleeping between them, all timed and all scored. Fit one is the cold fit. The
headline pair is fit one against the median of fits six through ten. This is
deliberately a different shape from the parent protocol's `w7_repeated_fit`,
which performs five fits with the parent's cooldown discipline around the
measurement. Ten fits with no cooldown is a thermal measurement. Five with
cooldown is a steady-state timing measurement. They are not interchangeable and
a record from one must never be quoted as the other.

**Sustained throughput.** Identical fits repeated back to back for a declared
wall-clock window, default twenty minutes, bucketed into sixty-second
intervals. Reported as fits per minute per bucket. The two derived numbers are
the sustained ratio, the last complete bucket divided by the first, and the
time to first throttle. A sustained ratio near 1.0 means the machine held its
clock. A ratio of 0.6 means the twentieth minute ran at sixty percent of the
first, which is what a user doing repeated training actually gets.

**Throttling.** A drop in the machine's available performance imposed by
firmware in response to thermal or power state. It is detected, never inferred
from timings. A timing that fell is evidence of nothing on its own, because
background load falls and rises too. Throttling is claimed only when a thermal
sample recorded it.

**Average power over an interval.** Mean of the sampled power series over a
window whose length is recorded in the same block. Average power without a
window length is not a number, and the schema makes the window length a
required companion of every mean.

**Energy over an interval.** Mean power times window length, from measured
samples only. Never derived from CPU time, core count, thread count, a TDP
figure, a battery percentage delta, or another machine's measurement.

**Energy above idle.** Energy over the interval minus the idle baseline mean
power times the same interval. Null whenever no baseline was taken in the same
session on the same machine.

## Instruments

Everything this protocol reads, what it actually gives, and what it costs.

| Command | Privilege | What it gives | Limits |
|---|---|---|---|
| `pmset -g therm` | none | `CPU_Speed_Limit`, `CPU_Scheduler_Limit`, `CPU_Available_CPUs` | Populated by the power management stack. On some Apple silicon machines these have been observed to sit at 100 through loads that clearly changed behavior. Whether they move on a given machine is one of the first things a run establishes. |
| `pmset -g batt` | none | Power source, charge percent, charging state | Text format, parsed loosely, recorded raw. |
| `pmset -g` | none | Low Power Mode and other power mode settings | Setting names differ by machine and release. Recorded verbatim, never normalized into a claim. |
| `powermetrics --samplers cpu_power,gpu_power` | root | Package, CPU, GPU, ANE power series | Machine-wide. Key names and units move across releases and chips, which is why the parser records the keys it matched. |
| `powermetrics --samplers thermal` | root | Thermal pressure level | A coarse level, not a temperature. |
| `powermetrics --samplers smc` | root | Fan speed, and on some Macs temperatures | On Apple silicon the fields present here vary by model and release, and a fanless machine reports no fan. Recorded raw and flagged unverified. |
| `sysctl` | none | Chip brand string, model identifier, core counts by performance level, memory size | Authoritative machine identity. |
| `system_profiler SPDisplaysDataType` | none | GPU core count and name | The core count key is absent on some macOS versions. Null then, never guessed from the chip name. |
| `sw_vers`, `uname` | none | OS version, build, kernel | |

Not used, and why.

- **Die temperature.** There is no public, documented, stable way to read CPU or
  GPU die temperature on Apple silicon. Third-party tools that display one read
  undocumented SMC keys. This protocol does not read them, does not ship a
  table of them, and treats any temperature in degrees Celsius appearing in a
  mojoboost record as a defect unless it came from a thermometer the operator
  names in `ambient_source`.
- **Battery charge deltas as energy.** Charge percentage is quantized, is
  temperature dependent, and is reported through a gauge that is itself
  modeled. A percentage drop across a run is not a joule count and is recorded
  only as context.
- **`sysdiagnose`.** It collects far more about the machine and its user than a
  benchmark needs. See the privacy section.

## Conditions and contrasts

A thermal record describes one condition set. The conditions are recorded
fields, not prose.

| Field | Values | Notes |
|---|---|---|
| `power_source` | `ac`, `battery`, `unknown` | The parent protocol requires AC. This protocol permits battery, in a record that declares it. |
| `battery_percent` | number or null | Context for a battery run. Not an energy measurement. |
| `low_power_mode` | true, false, null | |
| `power_mode_raw` | string or null | Verbatim from `pmset -g`, because the available modes differ by machine. |
| `lid_state` | `open`, `closed_clamshell`, `unknown` | A closed lid on a machine with an external display and power is a different thermal system from an open one. A closed lid with neither is a sleeping machine, not a measurement. |
| `surface` | `hard_desk`, `stand`, `soft`, `lap`, `unknown` | A MacBook Air is fanless and dissipates through the chassis, so the surface is part of the cooling system. `soft` is measured to document the penalty, not to be avoided in reporting. |
| `external_display_connected` | true, false, null | Drives GPU work that is not the workload. |
| `ambient_temp_c` and `ambient_source` | number or null, string or null | Recorded only when an actual thermometer was read, with the instrument named. Null otherwise. Never estimated from the season or the room. |
| `machine_idle_minutes_before` | number or null | What kind of cold the cold fit actually was. |

A **contrast** is two records that are identical in commit, machine, OS,
toolchain, workload, engine, threads, and every condition field except exactly
one, which is named in `contrast.declared_variable`. Battery against AC is a
contrast on `power_source`. Desk against duvet is a contrast on `surface`.

Two records that differ in more than one field are not a contrast and the
difference between them is not attributable. This is not a formality. A battery
run taken on a warmer afternoon, on a machine that had just finished an AC run,
differs in at least three ways, and the schema is built to make that visible
rather than to make it easy to publish.

## The phases

Eight, in this order. A record may contain any subset, and each phase carries
its own status, so a phase that was skipped is present and says so.

| Phase id | What it does | Duration |
|---|---|---|
| `idle_baseline` | Samples the idle machine with the same sampler and interval the workload phases will use. Nothing else runs. | 60 s default |
| `cold_fit` | One fit, fresh process, on the machine the gate has just checked. | one fit |
| `warm_fit` | One more fit in the same process, immediately. | one fit |
| `repeat_series` | Ten fits back to back in one process, all timed, all scored. | ten fits |
| `sustained` | Identical fits back to back for the declared window, bucketed per minute. | 1200 s default |
| `throttle_probe` | Continues the sustained load until a thermal sample shows a limit or the declared ceiling is reached, recording the offset at which it happened. | up to 3600 s |
| `cooldown_recovery` | Machine left idle, sampled at the same interval, until the thermal sample returns to its pre-run value or the ceiling is reached. Records the recovery time and whether it recovered at all. | up to 1800 s |
| `background_control` | The negative control. A declared competing load is started deliberately, and the harness must reject the run. A gate that passes here is a broken gate. | 120 s |

`background_control` earns its place. Every other gate in this repository is
believed rather than tested, and a gate nobody has watched fail is a gate
nobody knows works. The phase runs a named load, expects
`conditions.idle_gate_passed` to come back false, and sets
`conditions.background_load_rejected` to whether the harness in fact refused.
A record where the control passed the gate invalidates the gate, not the run
that follows it.

The load used for the control is declared in the record, never assumed.
`yes > /dev/null` on one core is enough to move a histogram timing and is the
recommended one, because it is trivially stoppable and does no input or output.

## What every phase records

**Fits.** Index, whether it is the cold fit, wall-clock seconds, and the quality
metric from that same fit. A fit without a quality value is recorded with
quality unavailable and a reason, and no throughput number derived from it is
quotable. This is the parent protocol's rule and it is not relaxed here.

**Thermal series.** A sample at the declared interval through the phase, each
with its offset in seconds from the phase start. Derived per phase are the
minimum speed limit seen, whether any sample was below 100 or above nominal
pressure, and the offset of the first such sample.

**Throughput buckets.** For `sustained` and `throttle_probe`, fits completed per
sixty-second bucket, with the bucket boundaries recorded so a partial trailing
bucket is visible as partial rather than counted as a slow minute.

**Energy.** Optional. Present with `available: false` and a reason whenever
`powermetrics` was not run, was not permitted, or produced nothing the parser
recognized. Never present with a filled-in number that was not sampled.

**Process attribution.** Optional, heuristic, and labeled as such, per the
boundary section above.

## Energy capture is optional

The three ways it is absent, all of which leave the rest of the record valid.

1. **No root.** `powermetrics` requires it. A run without `sudo` records
   `available: false, reason: "powermetrics needs root"` and continues. Every
   timing, throughput, and thermal quantity in that record remains usable.
2. **No tool, or no recognized keys.** `powermetrics` key names and units differ
   across macOS releases and chips. The parser records the keys it matched.
   Available true with an empty key list is a parser defect to fix, not a
   machine that used no power.
3. **No counters.** Some quantities are simply not exposed on some parts. The
   field stays null with a reason.

In none of those three cases is a number substituted. There is no estimator in
this protocol, the schema has no `estimated` method value, and adding one is a
change that must be argued for in a pull request rather than made quietly in a
parser.

The rule holds in the other direction too. If energy is available, it is
recorded whether or not it flatters mojoboost, and a run whose energy numbers
are unfavorable is published on the same terms as one whose numbers are good.

## Result attribution

What a given number in a record is permitted to be about.

| Number | Attributable to | Not attributable to |
|---|---|---|
| `fit_s` in any phase | This machine, this commit, this workload, this engine, this thread count, this condition set, at this point in the thermal history of the run | Apple silicon in general, another chip variant, another OS version, another surface |
| Sustained ratio | The machine's ability to hold clocks under this specific load | The library, unless a matched contrast on the same machine shows a different ratio for a different engine |
| Time to first throttle | This chassis under this load in this ambient environment | Any other ambient environment, since ambient is usually unrecorded and always uncontrolled |
| Mean power over a window | The whole machine during that window | mojoboost |
| Energy above idle | The workload plus whatever else was running, bounded by what the idle gate saw | A process |
| Energy impact score | Which process dominated the window | Joules, watts, or any comparison across machines |
| Cold minus warm | One-time cost paid in this process on this machine, including compilation and device setup | A breakdown of that cost, which needs the native phase counters described in `docs/STARTUP_LATENCY.md` and `handoffs/apple_a8_benchmarks.md` |

A throughput decline is attributed to thermal state only when all three hold.
The thermal series showed a limit change, the idle gate stayed clean through
the phase, and the decline persisted across the bucket where the limit
appeared. Two out of three is a finding worth recording and not a thermal
claim.

## Invalid runs

A record with any of these is written, kept, marked, and never quoted. The
schema carries them as an enumerated list on `validity.invalid_reasons` so a
reader does not have to reconstruct the judgment.

- `idle_gate_failed`. The gate failed and the run proceeded anyway.
- `background_process_detected`. A competing build or heavy process was seen
  during a measured phase, outside `background_control`.
- `background_control_not_rejected`. The negative control passed the gate,
  which means the gate in this record proves nothing.
- `dirty_tree`. Uncommitted changes. The commit field cannot identify what ran.
- `no_quality_metric`. Throughput without a loss next to it.
- `benchmark_link_missing`. No linked benchmark record and no explicit workload
  specification, so nobody can tell what was fit.
- `workload_mismatch`. The linked benchmark record's workload does not match the
  workload in this record.
- `energy_without_baseline`. Energy above idle present with no idle baseline
  phase in the same record.
- `sampler_keys_empty`. Energy marked available with no recognized power keys.
- `phase_aborted`. A measured phase ended early for a reason other than its own
  declared stopping rule.
- `record_edited`. Any hand edit after the run. Rerun instead and keep both.
- `condition_undeclared`. A condition known to have changed during the run, for
  example a charger plugged in mid-phase, that the record does not carry.

Warnings, which do not invalidate but travel with any table drawn from the
record, include an unrecorded ambient temperature, a null GPU core count, an
unverified fan reading, absent energy, and a short idle period before a cold
fit.

Note what is deliberately not an invalid reason. A throttled run is not invalid
here. Throttling invalidates a timing under the parent protocol, and it is the
measurement under this one.

## Publication rules

- Publish the thermal record file alongside any table, curve, or sentence drawn
  from it, and publish the benchmark record it links to. A sustained throughput
  curve without its record is not reviewable.
- Name the chip variant, the memory size, the OS build, the commit, and the
  condition set in the same place as any thermal or energy number. "M4" is not
  a machine. "M4 Pro, 24 GB, macOS 15.6, lid open, hard desk, AC" is.
- State the window length beside every average power figure and the bucket size
  beside every throughput figure.
- Never present a machine-wide power figure as mojoboost's power consumption,
  in a README, a chart axis, a slide, or a forum post. If the sentence has a
  process as its subject and joules as its object, it is wrong.
- Never publish an energy comparison between two engines that reached different
  held-out quality without stating both losses.
- Never compare a thermal number across chip variants, OS versions, ambient
  conditions, or protocol versions. Within one record, or within a declared
  contrast, or not at all.
- Never publish a curve from a record whose `validity.quotable` is false, even
  with a caveat. Rerun it.
- A phase that was not run is reported as not run in any table that includes
  the other phases. Dropping the row makes the machine look better than it is.

## Prohibited

- Any energy, power, or temperature value that was not sampled from the machine
  it is attributed to, in the run it is attributed to.
- Deriving energy from CPU time, core count, thread count, TDP, battery
  percentage, or another machine's measurement.
- Per-process joules, in any form, including a machine-wide figure scaled by an
  energy impact score.
- Presenting a `pmset` speed limit as a temperature, or a thermal pressure level
  as a temperature.
- Quoting a sustained ratio from a run shorter than the declared window.
- Filling a null field by hand.
- Editing a record after the fact.
- Publishing a record without first reading it for the machine identifiers
  described in the privacy section.

## Privacy

A benchmark record is a hardware report, and hardware reports on macOS carry
identity. Before any record leaves a machine, the person who ran it is
responsible for reading it. The capture script prints the commands it would run
so that this can be checked before anything is collected rather than after.

- `system_profiler SPHardwareDataType` includes the serial number and the
  hardware UUID. This protocol reads only `SPDisplaysDataType`, for the GPU core
  count and name, and the schema has no field for a serial number or a UUID.
- `system_profiler SPSoftwareDataType` includes the computer name and the logged
  in user name, both of which are commonly a person's real name. Not read.
- The competing process list in the idle gate contains process names from the
  machine, which can reveal employer software, unreleased projects, and
  personal applications. It is recorded because the gate is worthless without
  it. Review it before publishing, and redact by replacing a name with
  `redacted` rather than by deleting the entry, so the count stays honest.
- Paths are identity. `invocation.cwd` and any record path typically contain a
  home directory and therefore a user name. Review before publishing.
- `ioreg` battery output and `nvram` contain serial numbers. Not read by this
  protocol, and not to be pasted into an issue.
- `sysdiagnose` collects logs, network names, and installed software inventory
  across the whole system. It is never required by this protocol and must not
  be attached to a mojoboost issue or pull request.
- Ambient temperature and location notes are a person's whereabouts. Room
  descriptions belong in a record only when the person who ran it chose to put
  them there.

Contributed records from outside the project are covered by
`launch/APPLE_BENCHMARK_REQUEST.txt`, which repeats these rules in the form a
contributor sees before running anything.

## Running it

Nothing in this version takes a sample. The script is a planner.

```sh
bash bench/apple/thermal_capture.sh --list-phases
bash bench/apple/thermal_capture.sh --self-check
bash bench/apple/thermal_capture.sh --phase cold_fit --phase warm_fit
bash bench/apple/thermal_capture.sh --phase sustained --duration 1200 --energy
bash bench/apple/thermal_capture.sh --print-plan --phase sustained > plan.json
```

Each of those prints the exact commands a run would use, shell-quoted, in
order, with the privileged ones marked. `--execute` is parsed and refused, with
an exit code and a message, because a script that can be talked into running
`sudo powermetrics` by a typo is a script that will eventually do it.

What a human may deliberately run later, once the measurement path exists and
has been reviewed, is listed with exact command lines in
`handoffs/performance_17_thermal_energy.md`. They are kept there rather than
here so that this document cannot be copied out of context into a terminal.

## Open items

- No measurement path exists. The capture script plans a run and cannot take
  one. Nothing in this repository has sampled power or thermal state.
- The fit loops the phases describe are not implemented. They need either an
  addition to `bench/apple/suite.py`, which this lane does not own, or a small
  Mojo driver. The handoff describes both and recommends the first.
- Whether `pmset -g therm` moves at all on Apple silicon under sustained load
  is unknown to this repository. If it does not, `powermetrics --samplers
  thermal` pressure becomes the only throttle signal, and the honest response is
  to say so in the record rather than to infer throttling from timings.
- The `powermetrics` parser in `bench/apple/suite.py` has never seen output from
  a real machine, which its own lane already flags. This protocol depends on it
  and inherits that risk.
- Per-fit energy requires starting and stopping the sampler around each fit,
  which at a 200 ms interval is a small number of samples for a fast fit. For
  short fits the honest unit is energy per phase divided by the fit count, and
  the schema records both the window and the count so a reader can see which
  one they are looking at.
- Ambient temperature is uncontrolled in every run this protocol will realistically
  produce. Time to first throttle is therefore the least portable number here,
  and it is reported with that caveat attached rather than dropped.
