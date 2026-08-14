# Performance 15: first-use, import, context, and compile latency

Lane 15. Files added, all new, no existing file touched:

- `src/mojoboost/initialization.mojo`
- `python/mojoboost/diagnostics.py`
- `tools/inspect_startup_artifacts.py`
- `docs/STARTUP_LATENCY.md`
- `handoffs/performance_15_startup.md` (this file)

Nothing in the repository imports `initialization` or `diagnostics` yet, so
this lane changes no behavior on its own. Everything below marked
**integration** is a central edit this lane was not permitted to make.

**Nothing was run. Nothing was measured.** The lane forbade executing Mojo,
pixi, Python, builds, import timing, benchmarks, and CI, so
`initialization.mojo` has never been compiled, `diagnostics.py` has never
been imported, and `inspect_startup_artifacts.py` has never parsed a file.
No performance claim appears anywhere in the four files or in this one.

## The headline finding

The `KernelRegistry` docstring in `gpu_runtime.mojo` says:

> The registry does not hold device function handles: binding those needs
> an API this module has not verified against the toolchain in use.

**That API has now been verified and it exists.**
`DeviceContext.compile_function[kernel]()` returns a `DeviceFunction`, and
the MAX documentation states its purpose directly: compiling as a separate
step lets you "execute the same compiled kernel on the same device multiple
times", which "avoids the overhead of compiling the kernel each time it's
executed". `DeviceFunction` is `Copyable` and `Movable`, so it can be held
in a struct field.

There is one hard constraint on holding it, and it is why nothing in this
lane caches a handle:

```
struct DeviceFunction[func_type, //, func, declared_arg_types, *, target, ...]
```

The struct is parameterized on the kernel function itself, so **every
kernel has a distinct type** and `List[DeviceFunction]` does not exist. A
handle cache is one typed field per kernel on whichever struct owns the
context, which means editing `GpuHistogramBuilder` or `GpuSession`. That is
central and belongs to the integration lane.

Whoever takes that edit should also correct the `KernelRegistry` docstring,
which is now wrong.

### What is not available

- **No persistent, cross-process kernel cache**, and the toolchain
  documents none. `DeviceContext.__deinit__` is documented as releasing "any
  cached memory buffers and compiled device functions" at refcount zero, so
  the cache that exists is in-memory, per context, and dies with the
  context.
- `MODULAR_MAX_CACHE_DIR` and `MODULAR_CACHE_DIR` are the MAX **model**
  cache and filesystem cache. `max warm-cache` and
  `max warm-interpreter-cache` preload models and interpreter op targets.
  None of that is a `compile_function` cache. Do not cite them as one.
- `DeviceContext.load_function` does load precompiled device code, but it
  takes PTX or SASS, which is an NVIDIA path. The only device this project
  has ever run on is an Apple M4 through Metal.

The full audit, with what each claim rests on, is the "What the toolchain
actually offers" section of `docs/STARTUP_LATENCY.md`.

## What the module provides

`src/mojoboost/initialization.mojo`, no device dependency, no global state:

| Piece | Answers |
| --- | --- |
| `PHASE_*` / `phase_name` / `phase_origin` | which ten phases, and who can time each |
| `StartupTrace` | per-phase calls, total, and first occurrence |
| `PhaseRecord` | one phase as a value a reader can hold |
| `FitLatency` | is this fit the cold one or a warm one |
| `WarmupPlan` / `env_warmup_level` | which kernels to create up front, and what each cost |
| `BuildIdentity` / `CACHE_KEY_VERSION` | what a cache *would* have to key on |

The ten phases, in the order a cold process pays them: `py_import`,
`ext_load`, `runtime_load`, `device_discovery`, `context_create`,
`kernel_create`, `first_alloc`, `first_transfer`, `first_fit`, `warm_fit`.

The first three are **supplied**: they are over before any Mojo code runs.
`StartupTrace.supply` takes them from a host harness and raises if asked to
supply a phase that is natively measurable, so the two paths cannot be
crossed by accident.

### Environment contract

Follows the `MOJOBOOST_` convention in `parallel.mojo`, `gpu_tiling.mojo`,
and `gpu_runtime.mojo`:

- `MOJOBOOST_STARTUP_TRACE=1` enables startup timing. Off by default, and
  off means no clock read: `clock()` returns 0 and `record` treats a zero
  start as "count it, do not time it". Call counts are always kept, which
  is what a test can assert on without becoming machine dependent.
- `MOJOBOOST_GPU_WARMUP` is `off` (default), `train`, or `all`. `off`
  reproduces today's behavior exactly, where every kernel is created on the
  launch that first needs it. An unrecognized value is `off`, so a typo
  never silently front-loads work.
- `MOJOBOOST_STARTUP_REPORT_FD` is named in the module docstring as
  reserved and is **not read**. Emitting the report is a call-site
  decision.

### No global state

Deliberately no module-level trace, no lazily-initialized singleton, no
exit hook. A `StartupTrace` is a value its owner holds, so two estimators
keep two, an estimator can be dropped without coordinating with anything,
and interpreter shutdown frees the last one by the ordinary rules. `merge`
is how a caller that wants one number across owners asks for it.

The same rule is why nothing here caches a `DeviceContext`. The fix for
`context_create` is **not** "one context per process"; it is "one per
estimator, reused across that estimator's fits", which is what `GpuSession`
already implements.

## Integration: required changes

None of these were made. Each names the owning file and what the edit has
to preserve. Items 1 to 7 are call-site changes; item 8 is a boundary with
another lane that has already landed overlapping work.

**Symbols, not line numbers.** Five or more sessions are editing this
checkout and `train_gpu.mojo` grew by roughly a thousand lines while this
handoff was being written. Every reference below names a function or a
struct; grep for it.

### 1. `src/mojoboost/__init__.mojo` (export)

`initialization` is not exported, so nothing can reach it and it is not
compiled by the `__init__` import graph. Add, in the alphabetical position
the file uses:

```mojo
from .initialization import (
    CACHE_KEY_VERSION,
    PHASE_CONTEXT_CREATE,
    PHASE_DEVICE_DISCOVERY,
    PHASE_EXT_LOAD,
    PHASE_FIRST_ALLOC,
    PHASE_FIRST_FIT,
    PHASE_FIRST_TRANSFER,
    PHASE_KERNEL_CREATE,
    PHASE_PY_IMPORT,
    PHASE_RUNTIME_LOAD,
    PHASE_WARM_FIT,
    WARMUP_ALL,
    WARMUP_OFF,
    WARMUP_TRAIN,
    BuildIdentity,
    FitLatency,
    PhaseRecord,
    StartupTrace,
    WarmupPlan,
    env_warmup_level,
    phase_name,
    phase_origin,
)
```

Export and test wiring should land in the same commit: `KNOWN_UNWIRED_TESTS`
in `tools/check_parity.py` is empty and must stay that way, so a
`tests/parallel/test_initialization.mojo` has to be added to the `test`
chain in `pixi.toml` at the same time it is written.

### 2. `src/mojoboost/gpu_runtime.mojo` (owner: A5 / runtime lane)

`GpuSession` is the natural owner of a `StartupTrace`, because it is
already the per-estimator object that outlives a fit.

- Add `var startup: StartupTrace` to `GpuSession`, initialized from
  `StartupTrace.from_env()` in `GpuSession.__init__`.
- Bracket the existing `self.ctx = DeviceContext()` and
  `self.caps = query_device_caps(self.ctx)` separately: the first is
  `PHASE_CONTEXT_CREATE`, the second is `PHASE_DEVICE_DISCOVERY`. Today
  both sit inside one `PHASE_ALLOC` record, which is the wrong phase for
  either.
- In `GpuSession.note_kernel`, when `mark_warm` returns True, also record
  `PHASE_KERNEL_CREATE` on the startup trace. The existing `PHASE_COMPILE`
  attribution stays; the two answer different questions (within-fit versus
  one-time) and neither replaces the other.
- Add `WarmupPlan` alongside `KernelRegistry`, populated with the same
  `KERNEL_*` ids. `WarmupPlan` stores caller-supplied ids precisely so it
  does not fork that numbering.
- Extend `trace()` to append `self.startup.report()`, so one call yields
  both the per-round and the startup view.

**Correctness constraint:** none of this may add a clock read on a disabled
path. `StartupTrace.clock()` already returns 0 when disabled; keep the
`clock()` / `record` pairing and do not call `perf_counter_ns` directly.

### 3. `src/mojoboost/histogram_gpu.mojo` (owner: kernel lane)

- `GpuHistogramBuilder` has three constructors and the `(ctx, caps, data,
  strategy)` one is where the other two land, so it is the only place to
  edit. The `enqueue_create_buffer` calls in it are `PHASE_FIRST_ALLOC`;
  the binned-matrix upload and the `ctx.synchronize()` after it are
  `PHASE_FIRST_TRANSFER`.
- Only the `GpuSession`-borrowing constructor has a trace to record into.
  The `(data, strategy)` form opens its own `DeviceContext` and therefore
  pays `context_create` again; that is exactly the waste worth measuring,
  so it should construct a `StartupTrace.from_env()` of its own rather than
  going unrecorded.
- `N_KERNELS` in `gpu_runtime.mojo` is 5 and does not cover the kernels in
  `gpu_active_rows.mojo` (9 launch sites), `gpu_objectives_native.mojo`
  (7), `gpu_predict.mojo` (5), or `gpu_split_search.mojo` (3). The
  inventory needs extending before `kernel_create` counts mean anything.
  This lane did not extend it, because the ids belong to `gpu_runtime`.

### 4. `src/mojoboost/train_gpu.mojo` (owner: GPU training lane)

Every entry point that constructs a `GpuHistogramBuilder` is a site:
`train_gpu`, `train_custom_gpu`, `train_multiclass_gpu`, and
`train_gpu_with_valid` as of this writing, and the list is still growing.
Grep for `GpuHistogramBuilder(data)` rather than trusting that list. Each is
where `FitLatency` goes:

```mojo
var latency = FitLatency()          # held by the estimator, not by train_gpu
var started = trace.clock()
# ... existing training ...
latency.note_fit(trace, started)
```

`FitLatency` must outlive a single `train_gpu` call or every fit reports as
cold. It belongs on whatever holds the `GpuSession`, which today is nothing:
`train_gpu` constructs a builder per call. That is the same structural
problem `GpuSession` exists to solve, and the two changes should land
together.

### 5. `bindings/_mojoboost.mojo` (owner: Python API lane)

- The extension entry point is `PyInit__mojoboost`. It is the last native
  code that runs
  during `ext_load`, so it is the only place a native timestamp can be
  taken that is comparable with the host's `dlopen` measurement. A
  `startup_epoch_ns()` binding returning `perf_counter_ns()` captured at
  `PyInit` time would let the harness compute `ext_load` without guessing.
- A `startup_report()` binding returning `StartupTrace.report()` as a
  Python string is what `diagnostics.parse_trace` consumes. Without it, the
  native phases are unreachable from Python and only a Mojo bench can read
  them.
- `def_function` caps at 6 arguments; both of these take none, so neither
  is affected.

### 6. `python/mojoboost/__init__.py` (owner: Python API lane)

- The extension is imported at module scope (`from . import _arrays, _eval,
  _mojoboost, ...`), so `import mojoboost` always pays `ext_load` and
  `runtime_load` even for a
  caller that only wants `mojoboost.device_selection` or
  `mojoboost.diagnostics`. Whether to defer it is a real API decision with a
  real cost either way, and it is the single largest lever on `py_import`.
  This lane did not take it.
- `diagnostics` is not imported by `__init__.py` and should stay that way:
  a diagnostics module that costs import time to have available is
  self-defeating. `from mojoboost import diagnostics` is the intended
  spelling and works without a change.
- `diagnostics.environment_snapshot()` deliberately does not import the
  extension, so it stays useful when importing it is what fails.

### 7. `pixi.toml` and `bench/` (owner: benchmark lane)

`bench/bench_startup.mojo` and a `bench-startup` task **do not exist**.
What the bench has to do, in order:

1. open a `GpuSession`, recording `device_discovery` and `context_create`
2. fit once on a small fixed matrix, recording `first_fit`
3. fit four more times on the same matrix, recording `warm_fit`
4. print `trace.report()` and `plan.report()`

Small and fixed matters: the point is the one-time cost, and a matrix large
enough for the fits to dominate hides it. Use the same counter-based
splitmix64 generator `bench/bench_train.mojo` uses so the data is
reproducible.

Also add `check-startup-artifacts = "python3 tools/inspect_startup_artifacts.py --strict"`
as a pixi task. It builds nothing and imports nothing, so it is as cheap as
`check-parity` and belongs in the same CI job.

### 8. Overlap with the release lane

The release lane landed `packaging/matrix/validate_artifact.py`,
`packaging/macos/inspect_wheel.py`, `packaging/linux/inspect_wheel.py`, and
`packaging/linux/inspect_elf.sh` in the working tree while this lane was
writing. Two things follow:

- `tools/inspect_startup_artifacts.py` **does not duplicate their Mach-O
  parser.** It loads `macho_info` from `packaging/matrix/validate_artifact.py`
  by path, honoring the rule that file's docstring states: one such parser
  in the repository, one place to fix it. If the release lane moves or
  renames that function, this script degrades to size, digest, install
  kind, and ELF only, and says so, rather than raising.
- The four questions are genuinely different and none of them subsumes
  another. `validate_artifact.py` validates a wheel against the matrix.
  `inspect_wheel.py` checks the release-only rules on a wheel.
  `inspect_elf.sh` asks whether a real Linux loader is satisfied on a real
  Linux box. This one describes the *installed* package and what its first
  import will cost. An integration lane consolidating them should keep all
  four questions, not merge the scripts.

## Measurement commands

All future work. None has been run. The full protocol, including the
five-fresh-process minimum and the page-cache caveat, is in
`docs/STARTUP_LATENCY.md`.

```sh
# Phase 0: the Python module graph
python -X importtime -c "import mojoboost" 2>&1 | tail -30

# Phases 1 and 2: the extension alone
python -X importtime -c "import mojoboost._mojoboost" 2>&1 | tail -5
DYLD_PRINT_STATISTICS=1 python -c "import mojoboost._mojoboost" 2>&1   # macOS
LD_DEBUG=statistics     python -c "import mojoboost._mojoboost" 2>&1   # Linux

# What the loader has to open, without loading it
python3 tools/inspect_startup_artifacts.py
python3 tools/inspect_startup_artifacts.py --strict
python3 tools/inspect_startup_artifacts.py --json > startup-artifacts.json

# Phases 3 to 9, once bench-startup exists
MOJOBOOST_STARTUP_TRACE=1 pixi run bench-startup
MOJOBOOST_STARTUP_TRACE=1 MOJOBOOST_GPU_WARMUP=train pixi run bench-startup
MOJOBOOST_STARTUP_TRACE=1 MOJOBOOST_DISABLE_GPU=1  pixi run bench-startup

# Startup and per-round attribution together
MOJOBOOST_STARTUP_TRACE=1 MOJOBOOST_GPU_TRACE=1 pixi run bench-train-gpu

# Any GPU phase attribution needs synchronous execution to mean anything
MODULAR_DEBUG=device-sync-mode MOJOBOOST_STARTUP_TRACE=1 pixi run bench-startup
```

Wrap every compiling command in `nice -n 19 tools/with_build_lock.sh`, per
the shared-checkout discipline, and set `MOJOBOOST_NUM_WORKERS=1` for
anything whose CPU parallelism would otherwise vary between runs.

## Expected output schema

`StartupTrace.report()`, one line per phase, all ten always present:

```
startup.<phase> <calls> <total_ns> <first_ns> <origin> <observed>
startup.cold_ns <ns>
startup.warm_ns <ns>
```

`origin` is `supplied` or `native`. `observed` is `1` or `0` and is the
field that separates "did not happen" from "took no measurable time": a
CPU-only run opens no context, and reporting that as zero nanoseconds says
the opposite of the truth.

### Cold, placeholders not values

```
startup.py_import        1 <ns> <ns> supplied 1
startup.ext_load         1 <ns> <ns> supplied 1
startup.runtime_load     1 <ns> <ns> supplied 1
startup.device_discovery 1  <ns> <ns> native   1
startup.context_create   1  <ns> <ns> native   1
startup.kernel_create    <n> <ns> <ns> native   1
startup.first_alloc      17 <ns> <ns> native   1
startup.first_transfer   1  <ns> <ns> native   1
startup.first_fit        1  <ns> <ns> native   1
startup.warm_fit         0  0    0    native   0
startup.cold_ns <ns>
startup.warm_ns 0
```

`first_alloc` at 17 is derived, not measured: `GpuHistogramBuilder`
allocates 6 device and 3 pinned host buffers, and the `GpuActiveRows` it
constructs allocates 5 device and 3 pinned host. `kernel_create` has no
expected value, for the reason in integration point 3.

### Warm, same process after four more fits

```
startup.context_create   1 <ns> <ns> native   1
startup.first_fit        1 <ns> <ns> native   1
startup.warm_fit         4 <ns> <ns> native   1
startup.cold_ns <ns>
startup.warm_ns <ns>
```

**The finding to watch for is a one-time phase whose `calls` climbs past
1.** `context_create` reaching 2 means a second `DeviceContext` was opened,
which is precisely the waste `GpuSession` exists to remove and precisely
what today's `train_gpu` does on every call.

`WarmupPlan.report()` is separate and additive:

```
warmup.level train
warmup.planned <n>
warmup.created <n>
warmup.total_ns <ns>
warmup.kernel <id> <0|1> <ns>
```

`python/mojoboost/diagnostics.py` parses the `startup.` lines and
recomputes both summary lines rather than trusting them, so a report and
its summary cannot disagree. It is the only supported parser; renaming a
phase string breaks it and `docs/STARTUP_LATENCY.md` together, which is
intended.

## Unverified assumptions

Every one of these is load-bearing and unchecked.

1. **`initialization.mojo` has never been compiled.** It follows the
   idioms in `gpu_runtime.mojo` closely (parallel `List` fields rather than
   a `List` of structs, `def ... raises ->` ordering, `byte_length()` over
   `len` on `String`, `@fieldwise_init` on the value types), but no
   compiler has seen it.
2. **`diagnostics.py` has never been imported** and
   `inspect_startup_artifacts.py` has never parsed a file. Its ELF reader
   was written from the format definition; its Mach-O path is an adapter
   over `macho_info` in `packaging/matrix/validate_artifact.py` and
   assumes that function's dict keys (`cputype`, `rpaths`, `dylibs`,
   `minos`, `signed`). That file belongs to the release lane, is
   uncommitted at the time of writing, and its own docstring says it has
   never been executed either.
3. **The `ext_load` / `runtime_load` split is guesswork.** Whether MAX
   runtime initialization runs as a library initializer during `dlopen` or
   is deferred to first use is unknown. If deferred, part of
   `runtime_load` will surface inside `context_create` instead.
4. **`enqueue_function`'s caching behavior is unknown.** The fused overload
   may hit the context's compiled-function cache after the first launch, in
   which case `compile_function` buys nothing. The documentation recommends
   `compile_function` for repeated execution, which suggests but does not
   state that the fused form recompiles. **Measure before building the
   handle cache.**
5. **Kernel compile cost on Metal is unmeasured.** The entire case for
   warm-up rests on it being non-trivial. It may be microseconds, in which
   case `MOJOBOOST_GPU_WARMUP` should stay `off` forever and the knob is
   just documentation.
6. **`DeviceContext()` cost is unmeasured** and may be dominated by driver
   context creation that no session reuse avoids on the first one.
7. **`kernel_create` = 5 and `first_alloc` = 7** are read off `N_KERNELS`
   and the `GpuHistogramBuilder` constructor. The real kernel count is
   almost certainly higher (see integration point 3).
8. **No non-Apple device has ever run this code**, so every statement about
   context creation, kernel creation, and driver behavior is a statement
   about Metal on one M4.
9. **`-X importtime` accounting is assumed additive**, and whether the
   `dlopen` lands entirely inside the `mojoboost._mojoboost` row has not
   been checked.
10. **The five-fresh-process minimum is unvalidated.** Startup variance on
    this machine has never been characterized.

## Not done, and why

- **No test file.** The lane forbade writing or running tests. A
  `tests/parallel/test_initialization.mojo` is needed and must land in the
  same commit as the `pixi.toml` wiring, because `KNOWN_UNWIRED_TESTS` in
  `tools/check_parity.py` is empty and has to stay that way.
- **No call sites wired.** Every file that would call in belongs to another
  lane.
- **No handle cache.** It requires editing `GpuHistogramBuilder` or
  `GpuSession`, and assumption 4 says it should be measured first anyway.
- **No persistent cache.** None is proposed. `BuildIdentity` specifies the
  key one would need; `docs/STARTUP_LATENCY.md` specifies the location,
  permissions, concurrency, and failure recovery. All of it is a
  specification and none of it is an implementation.
- **No `MOJOBOOST_STARTUP_REPORT_FD` implementation.** The name is reserved
  in the module docstring and read nowhere. Where a report is emitted is a
  call-site decision, and a module that writes to a file descriptor on its
  own is a module that writes to somebody's stdout in production.
