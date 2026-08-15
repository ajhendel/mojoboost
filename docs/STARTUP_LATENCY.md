# First-use and compile latency

What a cold process pays before mojotrees answers the first question, how
that cost is divided, and how each division would be measured.

The rule this document is built around:

> Nothing here is a measurement. Not one number in this document was
> observed. The startup contract has been specified and instrumented; it
> has never been run.

That is a deliberate state, not an oversight. The lane that wrote this was
not permitted to execute Mojo, pixi, Python, a build, or a benchmark, so
what exists is the vocabulary, the structures, the artifact inspector, and
the commands. Filling in the tables is the next lane's work, and
[the handoff](../handoffs/performance_15_startup.md) says exactly how.

## Status

| Phase | Instrumented | Measured | Wired into a call site |
|---|---|---|---|
| `py_import` | schema only (supplied) | no | no |
| `ext_load` | schema only (supplied) | no | no |
| `runtime_load` | schema only (supplied) | no | no |
| `device_discovery` | `StartupTrace` | no | no |
| `context_create` | `StartupTrace` | no | no |
| `kernel_create` | `StartupTrace` + `WarmupPlan` | no | no |
| `first_alloc` | `StartupTrace` | no | no |
| `first_transfer` | `StartupTrace` | no | no |
| `first_fit` | `StartupTrace` + `FitLatency` | no | no |
| `warm_fit` | `StartupTrace` + `FitLatency` | no | no |

"Wired into a call site" is `no` for every row because the files that would
call in (`gpu_runtime.mojo`, `train_gpu.mojo`, `histogram_gpu.mojo`,
`bindings/_mojotrees.mojo`, `python/mojotrees/__init__.py`) belong to other
lanes. The instrumentation exists, imports nothing from them, and changes
no behavior on its own.

## Why the split is where it is

A slow steady-state fit and a slow first fit are different problems with
different fixes, and today the repository cannot tell them apart.
`bench/bench_train_gpu.mojo` times whole fits. `PhaseCounters` in
`src/mojotrees/gpu_runtime.mojo` attributes time *within* a fit across
compile, alloc, transfer, kernel, sync, and cleanup. Neither separates the
cost of arriving at a usable trainer from the cost of using one.

The separation matters most for the users who never reach steady state at
all: a CLI invocation, a serving process that starts per request, a CI job,
a notebook cell run once. For them the first fit is the entire experience,
and every one-time cost is paid in full.

Ten phases, in the order a cold process pays them. Each is chosen so that
it has one plausible owner and one plausible fix.

| # | Phase | The cost is | Fixed by | Measured by |
|---|---|---|---|---|
| 0 | `py_import` | executing `mojotrees/__init__.py` and its module graph | deferring imports | host harness |
| 1 | `ext_load` | `dlopen` of `_mojotrees.so` and its dependency closure | fewer or smaller dylibs, better rpaths | host harness |
| 2 | `runtime_load` | MAX async runtime initialization during that load | nothing local; a toolchain property | host harness |
| 3 | `device_discovery` | enumerating accelerators | not opening a device you will not use | native |
| 4 | `context_create` | `DeviceContext()` and driver context setup | one session per process, not per fit | native |
| 5 | `kernel_create` | first launch of each kernel | reusing `DeviceFunction` handles | native |
| 6 | `first_alloc` | first device buffer of each role | `PoolLedger` | native |
| 7 | `first_transfer` | first host to device copy of the binned matrix | `ResidencyLedger` | native |
| 8 | `first_fit` | the first complete fit, phases above excluded | all of the above | native |
| 9 | `warm_fit` | a repeated fit on a live process | steady-state work | native |

Phases 0 to 2 are **supplied**. They are over before any Mojo code runs, so
nothing native can time them; a host harness measures them and hands the
values in through `StartupTrace.supply`. `phase_origin` in
`src/mojotrees/initialization.mojo` is the authority on which is which, and
every report marks supplied values so a reader never mistakes a number
somebody typed for a number a process observed.

The headline is `first_fit - warm_fit`. The number that says whether a fix
is worth building is how much of that difference lands in one phase.

## The output schema

`StartupTrace.report()` emits one line per phase, in declaration order,
always all ten, whether or not the phase happened:

```
startup.<phase> <calls> <total_ns> <first_ns> <origin> <observed>
```

| Field | Meaning |
|---|---|
| `calls` | how many times the phase was recorded |
| `total_ns` | summed duration, 0 when tracing was off |
| `first_ns` | the first occurrence's duration |
| `origin` | `supplied` or `native`, from `phase_origin` |
| `observed` | `1` when the phase happened at all, `0` when it did not |

`observed` is the field that separates "did not happen" from "took no
measurable time". A CPU-only run opens no device context, and reporting
that as zero nanoseconds says the opposite of the truth.

Two summary lines follow:

```
startup.cold_ns <ns>
startup.warm_ns <ns>
```

`cold_ns` is the first occurrence of every one-time phase, summed;
`warm_ns` is the mean recorded `warm_fit`. `python/mojotrees/diagnostics.py`
recomputes both from the per-phase lines rather than trusting them, so a
report and its summary cannot disagree.

### Cold run, expected shape

Placeholders, not values. Every `<ns>` below is a number nobody has
observed.

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

`first_alloc` at 17 is structural rather than measured, and it is worth
deriving rather than taking on faith. A cold `GpuHistogramBuilder`
allocates 6 device buffers and 3 pinned host buffers itself, and the
`GpuActiveRows` it constructs allocates 5 more device buffers and 3 more
pinned host buffers: 11 device, 6 pinned, 17 total. Anything else is a real
discrepancy worth chasing.

**Note for whoever reads `gpu_runtime.mojo` next:** its module docstring
says the builder "allocates seven device buffers and three pinned host
buffers". That was true before the active-row compaction landed and is
stale now. Counted from the constructors, not from the docstring.

`kernel_create` is left as `<n>` on purpose. `N_KERNELS` is 5, but that is
the size of `KernelRegistry`'s inventory, not the number of distinct
kernels the GPU path launches, which is at least the 24 launch sites across
`gpu_active_rows`, `gpu_objectives_native`, `gpu_predict`, and
`gpu_split_search`. The inventory has to be extended before this count
means anything; see the handoff.

### Warm run, expected shape

The same process, after further fits. Every one-time phase keeps
`calls = 1` and an unchanged `first_ns`; only `warm_fit` moves.

```
startup.context_create   1 <ns> <ns> native   1
startup.first_fit        1 <ns> <ns> native   1
startup.warm_fit         4 <ns> <ns> native   1
startup.cold_ns <ns>
startup.warm_ns <ns>
```

A one-time phase whose `calls` climbs past 1 is the finding, not the noise.
`context_create` reaching 2 means a second `DeviceContext` was opened,
which is the exact waste `GpuSession` exists to remove.

### A CPU-only run

Every GPU phase reads `observed = 0`, and `cold_ns` is the sum of the four
phases that did happen:

```
startup.device_discovery 0 0 0 native 0
startup.context_create   0 0 0 native 0
startup.kernel_create    0 0 0 native 0
startup.first_transfer   0 0 0 native 0
```

## Measuring it

Every command below is future work. None has been run.

Run each timed command on an otherwise idle machine, wrapped in
`nice -n 19 tools/with_build_lock.sh` if anything else in this checkout may
be compiling, and repeat it five times in **fresh processes**, reporting the
minimum. Minimum rather than mean: a cold start is dominated by one-time
costs, and the mean is polluted by however warm the filesystem cache
happened to be. Note in the record whether the page cache was warm, because
the first run after a reboot and the fifth run in a row are different
measurements of different things.

### Phase 0, `py_import`

```sh
python -X importtime -c "import mojotrees" 2>&1 | tail -30
```

`-X importtime` prints cumulative microseconds per module, so the
`mojotrees` line is the phase and the lines above it say which submodule
owns it. The extension is imported by `python/mojotrees/__init__.py` at
module scope, so this number *contains* phases 1 and 2; subtract them.

### Phases 1 and 2, `ext_load` and `runtime_load`

```sh
python -X importtime -c "import mojotrees._mojotrees" 2>&1 | tail -5
```

That isolates the extension from the rest of the package's module graph.
Splitting it further into the `dlopen` itself and the MAX runtime
initialization inside it needs the loader's own accounting:

```sh
# macOS
DYLD_PRINT_STATISTICS=1 python -c "import mojotrees._mojotrees" 2>&1

# Linux
LD_DEBUG=statistics python -c "import mojotrees._mojotrees" 2>&1
```

**The split between phases 1 and 2 is unverified.** Loader statistics
attribute time to mapping, rebasing, and initializers; whether MAX's runtime
initialization appears as a library initializer or as work deferred to the
first call is not known, and if it is deferred then part of `runtime_load`
will show up inside `context_create` instead. Record what the loader
actually prints before assigning the phases.

### What the loader has to open

```sh
python3 tools/inspect_startup_artifacts.py
python3 tools/inspect_startup_artifacts.py --strict
python3 tools/inspect_startup_artifacts.py --json > startup-artifacts.json
```

Reads the Mach-O or ELF headers of the extension and of anything bundled
beside it and prints the dependency list, the search paths and whether each
exists, the platform minimum, the code signature status, and the sizes. It
imports nothing and needs no toolchain, so it works on the machine where
the import failed. `--strict` exits 1 on a broken search path, a missing
bundled library, an unsigned arm64 image, or a wheel that still points at
the environment that built it.

It does not own a Mach-O parser. `packaging/matrix/validate_artifact.py`
does, and this script loads `macho_info` from it, so there is one such
reader in the repository and one place to fix it. The two ask different
questions: that one validates a *wheel* against
`packaging/matrix/platform_matrix.toml`, this one describes an *installed*
package and what its first import will cost. Run both before publishing
anything, and run `packaging/linux/inspect_elf.sh` on a real Linux box as
well, since only a real loader can say whether a real loader is satisfied.

### Phases 3 to 9

These need call sites that do not exist yet. Once the integration lane has
wired `StartupTrace` in per the handoff:

```sh
MOJOTREES_STARTUP_TRACE=1 pixi run bench-startup
MOJOTREES_STARTUP_TRACE=1 MOJOTREES_GPU_WARMUP=train pixi run bench-startup
MOJOTREES_STARTUP_TRACE=1 MOJOTREES_DISABLE_GPU=1 pixi run bench-startup
```

`bench/bench_startup.mojo` and the `bench-startup` pixi task **do not
exist**. They belong to the benchmark lane; the handoff specifies what they
have to do.

Pair the startup trace with the per-fit trace, which does exist:

```sh
MOJOTREES_STARTUP_TRACE=1 MOJOTREES_GPU_TRACE=1 pixi run bench-train-gpu
```

For any GPU phase attribution to mean anything, force synchronous
execution first. Enqueues return immediately, so an unsynchronized "kernel
create" measurement times the enqueue and not the compile:

```sh
MODULAR_DEBUG=device-sync-mode MOJOTREES_STARTUP_TRACE=1 pixi run bench-startup
```

That changes what is being measured (it serializes the queue), so record it
as a separate row rather than as a correction to the asynchronous one.

## What the toolchain actually offers

This section is the audit the task demanded before anything proposes
precompiled kernels or a persistent cache. It is drawn from the MAX
`DeviceContext` reference and the GPU fundamentals guide at
docs.modular.com, read against the `max >=26.5,<27` and `mojo >=1.0` pins in
`pixi.toml`.

**There is no persistent kernel cache in this repository, and the toolchain
documents none.** Nothing below should be read as saying otherwise.

### What exists

1. **`DeviceContext.compile_function[kernel]()`** returns a `DeviceFunction`
   handle that `enqueue_function` accepts. The documentation states the
   reason plainly: compiling as a separate step lets you "execute the same
   compiled kernel on the same device multiple times", which "avoids the
   overhead of compiling the kernel each time it's executed".

2. **The fused `ctx.enqueue_function[kernel](...)`** compiles and enqueues
   in one step. Every one of the 24 kernel launch sites in this repository
   uses this form, and `compile_function` appears nowhere: 9 in
   `gpu_active_rows.mojo`, 7 in `gpu_objectives_native.mojo`, 5 in
   `gpu_predict.mojo`, 3 in `gpu_split_search.mojo`, and none in
   `histogram_gpu.mojo`, which since the active-row integration launches
   through `gpu_active_rows`. No `DeviceFunction` is held anywhere.

3. **A per-context cache.** `DeviceContext.__deinit__` is documented as
   releasing, at refcount zero, "any cached memory buffers and compiled
   device functions". So a context does cache compiled functions, and that
   cache dies with the context. It is in-memory, in-process, and per
   context: a second `DeviceContext` in the same process does not inherit
   it, and neither does a second process.

4. **`DeviceContext.load_function`** loads precompiled device code, but it
   takes vendor assembly (PTX or SASS) as a `DeviceExternalFunction`. That
   is an NVIDIA path. The only accelerator this project has ever run on is
   an Apple M4 through Metal, so the one documented precompiled-kernel
   mechanism does not cover the one backend with evidence behind it.

5. **`DeviceFunction.get_attribute`** and
   **`occupancy_max_active_blocks_per_multiprocessor`** exist on the handle,
   which is worth knowing because they are only reachable if the handle is
   kept.

### What does not exist

- **No documented on-disk cache for Mojo GPU kernels.**
  `MODULAR_MAX_CACHE_DIR` and `MODULAR_CACHE_DIR` are documented as the MAX
  *model* cache and the MAX filesystem cache. `max warm-cache` and
  `max warm-interpreter-cache` preload and compile *models* and *interpreter
  op targets*. None of that is a `DeviceContext.compile_function` cache, and
  assuming it is would be a guess about the internals of a closed component.
- **No claim, here or anywhere in this repository, that kernel compilation
  is cached across processes.** If it turns out to be, `kernel_create` will
  simply be small, and that is a measurement, not an assumption to build on.

### What this means for a fix

The reachable improvement is item 1, and it is in-process only:
`compile_function` once per session, keep the handles, launch through them.
That moves per-launch compile cost, if any, into `kernel_create`, where
`WarmupPlan` can attribute it.

There is a hard constraint on how those handles can be stored, and it is
the reason `WarmupPlan` holds none. `DeviceFunction` is parameterized on the
kernel function itself:

```
struct DeviceFunction[func_type, //, func, declared_arg_types, *, target, ...]
```

so every kernel has a **distinct type**. A `List[DeviceFunction]` does not
exist. A handle cache is one typed field per kernel on whichever struct owns
the context, which means editing `GpuHistogramBuilder` or `GpuSession`. That
edit is central and belongs to the integration lane.

The handles are `Copyable` and `Movable` refcounted values, so storing them
in a struct field is fine once the fields are declared.

## Wheel versus source install

The install kinds have different first-import costs and different failure
modes, and `tools/inspect_startup_artifacts.py` tells them apart without
loading anything: a staged runtime directory next to the extension means a
wheel, and which directory it is names the builder.

| | Source build | Wheel, `dylibs` | Wheel, `libs` |
|---|---|---|---|
| Produced by | `bindings/build.sh` | `packaging/build_wheel.sh` | `packaging/linux/build_wheel_linux.sh` |
| Platform | any | macOS arm64 | Linux |
| MAX runtime | in the pixi environment | four dylibs in `mojotrees/.dylibs` | the ELF closure in `mojotrees/.libs`, staged by soname |
| Search path | absolute rpath to `$CONDA_PREFIX/lib` | `@loader_path/.dylibs` | `$ORIGIN/.libs` |
| Signature | as linked | re-signed ad hoc after `install_name_tool` | not applicable |
| Breaks when | the pixi environment moves or is removed | a bundled dylib is stripped by a repacker | a closure member is missed at staging time |

The two wheel layouts are not interchangeable and must not be checked
against each other. The macOS bundle is a known set of four, so a missing
member is detectable by name. The Linux bundle is whatever the closure
turned out to be, so there is no list to check it against, and
`--strict` says so rather than inventing one. That is what
`packaging/linux/check_metadata_ready.py` and a real loader
(`packaging/linux/inspect_elf.sh`) are for.

Consequences for the startup contract:

- **`ext_load` is not the same phase in both.** The wheel opens four
  libraries from one directory the loader has already touched. The source
  build opens them from a conda prefix that may be on a different
  filesystem and holds thousands of other files. Measure both; do not
  quote one as the other.
- **The absolute rpath is a correctness issue before it is a latency
  issue.** A wheel that still carries `$CONDA_PREFIX/lib` imports fine on
  the machine that built it and fails everywhere else. That is
  `--strict` check 4.
- **An unsigned arm64 image is killed, not diagnosed.**
  `install_name_tool` invalidates the signature, `build_wheel.sh` re-signs
  with `codesign --force --sign -`, and if that step is skipped the symptom
  is `SIGKILL` on import with no Python traceback. That is `--strict`
  check 5, and it is worth running on a wheel before publishing one.
- **The platform minimum is baked in.** The wheel's `plat_name` is pinned
  to the Mojo toolchain's `LC_BUILD_VERSION` minimum, which
  `inspect_startup_artifacts.py` prints as `minimum os`. A wheel tagged
  above the interpreter's platform is not installable, which is a first-use
  failure that never reaches any of the ten phases.
- **No sdist.** `docs/PLATFORM_MATRIX.md` lists `sdist` as unsupported, so
  there is no third install kind whose startup cost has to be characterized.

## If a persistent cache is ever built

It is not built, and this document does not propose building it. What
follows is the specification any such cache would have to satisfy, written
down now because the failure mode of getting it wrong is a silently stale
kernel producing wrong numbers, and that is worth more than a paragraph of
warning later. The struct is `BuildIdentity` in
`src/mojotrees/initialization.mojo`.

### The key

Every field, all of them, whole-key equality with no partial credit:

| Field | Why | Where from |
|---|---|---|
| `toolchain` | two toolchains emit different code from identical source | build time |
| `target_arch` | the host architecture the extension targets | build time |
| `device_api` | `metal`, `cuda`, `hip`; same source, different device code | runtime |
| `driver_version` | a driver upgrade changes what compiles | `DeviceContext.get_api_version()` |
| `kernel_source_hash` | the sources that went in | build time |
| `compile_options` | a compilation input like any other |  `compile_function` argument |

`CACHE_KEY_VERSION` versions the *meaning* of that list, so an artifact
written under an older layout is rejected rather than reinterpreted.

The field most likely to be forgotten is `driver_version`, because nothing
about the package changes when a driver is upgraded, and the resulting bug
looks like hardware flakiness.

Deliberately absent: anything about the data. A kernel does not depend on
row count, feature count, or bin count, and keying on them yields a cache
that never hits.

`BuildIdentity.is_complete()` exists because an empty string is a missing
value, not a value. An incomplete identity must never be written or looked
up; if it were, every incomplete identity would collide with every other.

### Location, permissions, and failure

- **Location.** Follow the toolchain's own precedence rather than inventing
  one: `$MODULAR_HOME/cache`, else `$HOME/.modular`, else
  `$XDG_CACHE_HOME/modular`, else `$HOME/.cache/modular`. A mojotrees
  subdirectory under whichever wins. Never inside the installed package: a
  wheel directory may be read-only, may be shared between users, and may be
  on a filesystem that is remounted read-only in production.
- **Permissions.** Owner-only, `0700` on the directory and `0600` on
  entries. A world-writable cache of compiled code is a way to execute
  somebody else's code, and a shared machine is exactly where a cache looks
  most attractive.
- **Concurrency.** Write to a temporary file in the same directory and
  `rename` into place, so a reader never sees a partial artifact. Never
  hold a lock across a compile; two processes compiling the same kernel and
  one of them discarding its result is cheaper than either of them waiting,
  and far cheaper than debugging a stale lock in CI.
- **Failure recovery.** Every failure is a cache miss, never an error. A
  missing directory, a read-only filesystem, a corrupt entry, a permission
  denial, a full disk: compile and continue. A cache that can fail a
  training run is a worse cache than no cache. Log at most once per process,
  and only when tracing is on.
- **Eviction.** Entries are small and keyed by an identity that changes on
  every upgrade, so stale entries accumulate slowly and harmlessly.
  Deleting the directory must always be safe and must never need a tool.

## Global state and shutdown

Nothing in `initialization.mojo` is process-global. There is no
module-level trace, no lazily-initialized singleton, and no exit hook. A
`StartupTrace` is a value its owner holds, so:

- two estimators in one process keep two traces and neither coordinates
  with the other,
- an estimator can be dropped without draining anything shared,
- interpreter shutdown frees the last one by the ordinary rules, with no
  finalizer running at an unspecified point,
- `merge` is how a caller that wants one number across several owners asks
  for it, explicitly.

The same rule is why nothing caches a `DeviceContext` process-wide. A
process-wide context would outlive the estimator that opened it and would
have to be torn down by a hook running during interpreter finalization,
which is where CPython destroys modules in an order nothing may depend on.
`GpuSession` in `gpu_runtime.mojo` is the per-estimator owner and its
`close()` is the supported teardown.

This is also a constraint on the fix for `context_create`. "Open one
context per process" is the wrong shape. "Open one per estimator, and reuse
it across that estimator's fits" is the right one, and it is what
`GpuSession` already implements.

## Instrumentation cost

Off by default, and off means off. `StartupTrace.clock()` returns 0 without
reading a clock, and `record` treats a zero start as "count it, do not time
it". Call counts are always kept, following `PhaseCounters`: an integer add
is cheap enough to always pay, and counts are what a test asserts on
without becoming machine dependent.

This matters because the intended call sites (buffer allocation, kernel
launch) also run in steady state. Instrumentation that costs something when
disabled is instrumentation that gets deleted by the next person running a
benchmark.

## Unverified assumptions

Every one of these is a claim this document depends on and nobody has
checked. They are the first thing to test.

1. **The phase 1 / phase 2 split is guesswork.** Whether MAX runtime
   initialization runs as a library initializer during `dlopen` or is
   deferred to first use is unknown. If deferred, part of `runtime_load`
   will appear inside `context_create`.
2. **`enqueue_function`'s caching behavior is unknown.** The fused overload
   may hit the context's compiled-function cache on every launch after the
   first, in which case `compile_function` buys nothing and
   `kernel_create` will be a single small number. The documentation
   recommends `compile_function` for repeated execution, which suggests but
   does not state that the fused form recompiles.
3. **Kernel compile cost on Metal is unmeasured.** The whole case for
   warm-up rests on it being non-trivial. It may be microseconds.
4. **`DeviceContext()` cost is unmeasured**, and may be dominated by driver
   context creation that no amount of session reuse avoids on the first
   one.
5. **`first_alloc` = 17 is counted from constructors, not observed**, and
   `kernel_create` has no expected value at all, because `N_KERNELS` = 5
   is the registry's inventory rather than the number of kernels the path
   launches. Extending that inventory is a prerequisite, not a follow-up.
6. **The five-run minimum protocol is unvalidated.** Startup variance on
   this machine has never been characterized; five may be far too few.
7. **No non-Apple device has ever run this code.** Every statement about
   `context_create`, `kernel_create`, and driver behavior is a statement
   about Metal on one M4, and Metal is the backend with the least in common
   with CUDA in exactly these phases.
8. **`-X importtime` accounting is assumed additive.** It attributes
   cumulative time per module; whether the extension's `dlopen` lands
   entirely inside the `mojotrees._mojotrees` row has not been checked.
9. **`inspect_startup_artifacts.py` has never been run.** Its ELF reader
   was written from the format definition and has parsed no file, and its
   adapter over `macho_info` assumes that function's dict keys
   (`cputype`, `rpaths`, `dylibs`, `minos`, `signed`), which are
   themselves from a lane whose own script states it has never been
   executed.

## Related

- [`src/mojotrees/initialization.mojo`](../src/mojotrees/initialization.mojo)
  is the contract in code.
- [`python/mojotrees/diagnostics.py`](../python/mojotrees/diagnostics.py)
  formats a report from measured values.
- [`tools/inspect_startup_artifacts.py`](../tools/inspect_startup_artifacts.py)
  reads the artifacts that fix `ext_load`.
- [`handoffs/performance_15_startup.md`](../handoffs/performance_15_startup.md)
  is the integration and measurement plan.
- [`docs/PLATFORM_MATRIX.md`](PLATFORM_MATRIX.md) is where install kinds and
  their evidence live.
- [`docs/GPU_VALIDATION.md`](GPU_VALIDATION.md) is why every GPU statement
  here is about one Apple M4.
