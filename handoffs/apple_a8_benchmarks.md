# Handoff: Apple benchmark and energy story (task A8)

Lane A8 of the parallel Apple round. Design and harness only. No benchmark
was executed, no result file exists, and nothing outside the four assigned
paths was touched.

## Files this lane owns

| Path | What it is |
|---|---|
| `bench/apple/suite.py` | The runner. Eight workloads times up to five engines, one child process per measurement, JSON record validated before it is written. |
| `bench/apple/schema.json` | JSON Schema 2020-12 for a run record. Every measured quantity is nullable and every null carries a reason. |
| `docs/APPLE_GPU_BENCHMARK_PROTOCOL.md` | The M1 to M5 validation protocol: fleet, workloads, engines, thread matching, idle gate, cooldown, what counts as a result, reporting rules, prohibitions. |
| `handoffs/apple_a8_benchmarks.md` | This file. |

`bench/apple/` is a new directory. Lane A7 owns
`bench/apple/unified_memory.mojo` in the same directory; nothing here
touches it and the two do not share code.

## Validation actually run

Static only, as the lane's brief requires. No Mojo, no pixi, no pytest, no
build, no benchmark, no power tool.

```
$ /usr/bin/python3 -m py_compile bench/apple/suite.py            # clean
$ /usr/bin/python3 -c "import json; json.load(open('bench/apple/schema.json'))"
schema parses
$ /usr/bin/python3 bench/apple/suite.py --self-check
self-check ok: 8 workloads, 5 engines, schema 1.0.0, protocol 1.0.0. No measurement was taken.
$ /usr/bin/python3 bench/apple/suite.py --plan --scale smoke --workloads w1_small_dense --engines mojoboost_gpu --threads 1
[plan printed]
$ /usr/bin/python3 bench/apple/suite.py --list --scale smoke
[catalog printed]
$ git diff --check -- bench/apple/suite.py bench/apple/schema.json \
      docs/APPLE_GPU_BENCHMARK_PROTOCOL.md handoffs/apple_a8_benchmarks.md
[no output]
```

`--self-check` checks that the schema parses, that every `$ref` resolves,
that the workload ids and engine names in the runner match the schema's
enums, that each catalog entry validates as a workload, that every engine
produces an objective for every workload, that the `powermetrics` parser
turns a synthetic two-sample output into a block the schema accepts and
that `energy_above_idle_j` is computed from a baseline, and that a
structurally complete all-null template record validates. The template is
marked `is_template` so it can never be read as a run, and the synthetic
power sample is parser input only; nothing derived from it reaches a
record.

The energy check earns its place. Before it existed the parser emitted
`ane_power_w_max` and `ane_energy_j`, fields the schema has no room for, so
every energy-bearing record would have failed validation on the first run
with `--energy`. That class of defect is now caught statically.

What was not run, and therefore what is not claimed: the measurement path.
`--run` has never executed. The engine adapters have never imported
LightGBM, XGBoost, or mojoboost. Nothing here is known to fit a model.

## Defects found by static review and fixed

Since the measurement path has never executed, the runner was read against
the APIs it calls (`python/mojoboost/basic.py` for `Dataset`, `train`, and
`Booster.predict`, whose signatures were confirmed) and five defects were
fixed in this lane's own file. They are listed because each would have cost
a run.

1. `device_resolved` was read from the wrong dict and was always null, so a
   GPU row could not have been checked against the backend that actually
   ran. It now comes from the engine's own report.
2. The energy parser emitted fields the schema does not define, which would
   have failed validation on every `--energy` run. Emission is now
   restricted to a declared table and covered by the self-check.
3. The idle power baseline was sampled after the workloads rather than
   before, on a machine the run had just heated, and was never subtracted
   from anything. It is now taken before the first measurement, on the
   machine the gate has just checked, and `energy_above_idle_j` is filled
   per measurement.
4. The idle gate counted its own launcher as a competing build, so a suite
   started with `pixi run` would always have failed the gate and would have
   trained whoever hit it to pass `--allow-busy`. Process matches are now
   filtered against this process and its ancestors.
5. Multiclass scoring accepted any prediction shape and would have
   silently scored a flat array. It now raises with the shape it got.

`--threads` values are also deduplicated, since on a single performance-core
machine the two required thread points coincide and running the plan twice
would not make it two measurements.

## Central wiring required, not done here

### 1. pixi tasks

`pixi.toml` is a shared hotspot, so these are described rather than added.
The suite needs the `bench` environment plus a built extension.

```toml
[feature.bench.tasks]
# Static validation. No measurement, no dependencies beyond the standard
# library, safe in CI.
bench-apple-check = "python bench/apple/suite.py --self-check"

# The catalog and the plan, for reading before committing an afternoon.
bench-apple-plan = "python bench/apple/suite.py --plan"

# The run itself. Depends on the extension because the suite measures the
# working copy, not an installed wheel.
bench-apple = { cmd = "python bench/apple/suite.py --run", depends-on = ["build-python"] }
```

`bench-apple-check` is the only one of the three that belongs in CI. It is
standard library only and takes well under a second, in the same spirit as
`check-parity`. The other two must never run in CI: a GitHub runner is a
shared, throttled, virtualized machine and its timings are meaningless.

### 2. Dependencies

`[feature.bench.dependencies]` currently has `python`, `numpy`, and
`lightgbm`. The suite additionally needs:

- `scipy`, for `w4_sparse`. Without it that workload errors for every
  engine rather than silently degrading.
- `xgboost`, for the `xgboost_cpu` engine. Absent, the engine records
  `unsupported: xgboost unavailable` and the run continues, so this is
  optional but leaves a hole in the comparison.

`jsonschema` is deliberately not required. The suite validates with it when
it is installed and with the small validator in `suite.py` when it is not,
because a benchmarking machine should carry the benchmark's dependencies
and nothing else.

### 3. Results directory

`bench/apple/results/` is created at run time and holds one JSON file per
run. Decide before the first run whether records are committed. The
recommendation is yes, committed, because the protocol requires publishing
the record alongside any table drawn from it, and they are small. If they
are not committed, add `bench/apple/results/` to `.gitignore`, which this
lane did not touch.

### 4. Documentation pointers

`README.md`, `bench/README.md`, and `docs/GPU_VALIDATION.md` are all shared
or owned elsewhere. Once this lands, someone should add:

- `bench/README.md`, a section pointing at the protocol and saying plainly
  that the numbers currently in that file were taken on a loaded machine
  and are not results under it.
- `docs/GPU_VALIDATION.md`, a cross-reference. That document covers
  cross-vendor correctness; this one covers Apple performance and energy.
  They overlap on the rule that a timing without a loss next to it is not a
  result, which is stated in both on purpose.
- `README.md`, nothing yet. Nothing in this lane produces a number worth
  putting in a README.

## Dependencies on other lanes

| Lane | What A8 needs from it | What happens without it |
|---|---|---|
| A5, persistent GPU runtime | Phase counters (compile, allocation, transfer, kernel, synchronization, cleanup) exported through the binding, so `device_phases.source` can become `instrumentation`. | `device_phases` stays `unavailable` on every measurement. The suite still records totals and the warmup gap. |
| A6, Apple tuning policy | The resolved launch geometry (strategy, row tiles, block threads, threadgroups) for the shape being fitted, so a timing can be tied to the geometry that produced it. | `device_phases.launch_geometry` stays null. A timing cannot be attributed to a tiling decision. |
| A9, device selection | The crossover table this suite is supposed to fill. A9's rules must stay conservative and versioned until a run under this protocol exists. | A9 defaults to CPU and says why, which is correct behavior with no data. |
| A7, unified memory | Whatever it concludes about shared buffers changes what the transfer phases mean, and possibly removes some. | Nothing breaks; the phase fields are already nullable. |
| A10, five-minute demo | Its CPU/GPU timing table must be filled only from a record produced by this protocol. | The table stays empty, which is the correct state today. |

### The instrumentation hook, concretely

The smallest thing that would let `device_phases.source` become
`instrumentation`:

1. A per-fit phase accumulator in the GPU trainer or in A5's session type,
   summing seconds per phase for the fit and resetting at its start.
2. One binding function, for example `last_fit_phases() -> PythonObject`,
   returning a dict whose keys are the field names in
   `schema.json` under `$defs/device_phases`. Those names were chosen to
   match what `bench/bench_gpu_validation.mojo` already prints, so nothing
   new has to be named.
3. In `suite.py`, `MojoBoostEngine.fit` calls it after training when the
   attribute exists and fills the block with `source: "instrumentation"`.
   That is roughly six lines in this file and no change to the schema.

Nobody should add the binding solely for this. It is worth doing when A5
needs the same counters anyway.

## Design decisions worth reviewing

- **One process per measurement.** It is what makes peak resident memory
  attributable, keeps the thread environment clean (`MOJOBOOST_NUM_WORKERS`
  and `OMP_NUM_THREADS` are read at import or at first use, so they must be
  set before the library loads), and keeps library initialization and
  kernel compilation out of whichever engine happened to run first. The
  cost is process startup per measurement, which is recorded as `import_s`
  and excluded from the comparison.
- **Threads matched at 1 and the performance-core count**, not the logical
  count. Efficiency cores change the ratio between engines rather than
  scaling it. A third point including them is allowed as its own row.
- **`resolved_threads` is null everywhere.** None of the three libraries
  reports what it actually used, so thread matching is requested and not
  verified, and the schema says so rather than implying otherwise.
- **Energy is measured or absent.** No derivation from CPU time or TDP.
  `powermetrics` needs root and reports the whole machine, so an idle
  baseline is recorded and `energy_above_idle_j` is the number worth
  quoting.
- **XGBoost GPU is a permanent `unsupported` row**, because its GPU tree
  method targets CUDA and there is no Metal backend. The row exists so a
  reader sees the statement rather than a blank.
- **Comparability notes travel with the record.** Leaf-wise growth
  approximated in XGBoost, LightGBM's `min_data_in_bin`, feature bundling
  forced off, different categorical split rules. These cannot be
  parameter-matched away and the record refuses to hide them.

## Risks

1. **The `powermetrics` parser is unverified against a real machine.** It
   handles both the plist and the plain-text shapes, records `keys_seen`,
   and is exercised by the self-check on a synthetic sample, but no output
   from any macOS version has been through it. On the first energy run,
   check `keys_seen` before trusting a joule. The failure mode is designed
   to be an empty `keys_seen` with `available: true`, which is detectable,
   rather than a plausible zero. The plist branch in particular assumes the
   `processor` dictionary reports milliwatts for the per-domain keys and
   watts for `package_watts`; confirm that against one real sample before
   any energy number is published.
2. **The engine adapters have never run.** Parameter names for the three
   libraries were taken from their documented APIs and from this
   repository's existing drivers, and the mojoboost path follows
   `python/mojoboost/basic.py` (`Dataset` plus `train`, with `device` in
   the params dict, which `_Config` forwards to the estimator). The first
   `--run --scale smoke` will find whatever is wrong. Do that before the
   first full run and treat any parameter fix as a protocol-neutral change.
3. **`w3_large_dense` at 1,000,000 x 50 may not fit comfortably on an 8 GB
   machine** once four engines have each held a copy at some point in their
   own process. The processes are sequential, so the peak is per engine
   rather than summed, but a base-configuration Mac should run the smoke
   scale first.
4. **Sparse generation goes through `scipy.sparse.coo_matrix` and
   `sum_duplicates`,** so the realized nonzero count is slightly below
   `rows * nonzeros_per_row` where the same column is drawn twice for a
   row. This is deterministic and identical for every engine, and the
   `data_digest` proves it, but the density in the record is the requested
   one and not the realized one. Fix by recording realized `nnz` if the
   difference turns out to matter.
5. **Full-scale runtime is hours.** Eight workloads times four engines
   times two thread counts times five repetitions plus a minute of cooldown
   each. The protocol says run it once properly, and that is a real
   scheduling constraint, not a figure of speech.
6. **The suite cannot see everything the idle gate needs.** Time Machine,
   Spotlight, and display sleep are checked by hand from the protocol's
   list. A run that fails the automatic gate is still written and marked
   unquotable, which is the intended pressure.

## State of the numbers

None. No cell of the M1 to M5 status table is filled. No result file
exists. Nothing in this lane authorizes a performance or energy claim about
mojoboost on Apple silicon, and the existing numbers in `bench/README.md`
and `docs/GPU_VALIDATION.md` remain what they say they are, timings taken
on a loaded development machine.
