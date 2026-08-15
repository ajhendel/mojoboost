"""First-use cost: the phases a cold process pays for before the first
prediction comes back, and the structures that let them be attributed.

The question this module exists to make answerable is "why did the first
`fit` take so much longer than the second one, and which of those seconds
would a second process pay again". Today nothing in the repository can
answer it. `bench/bench_train_gpu.mojo` times whole fits, `PhaseCounters`
in gpu_runtime.mojo attributes *within* a fit (compile / alloc / transfer /
kernel / sync / cleanup), and neither separates the one-time cost of
arriving at a usable trainer from the steady-state cost of using one.

Those are different costs with different fixes. A slow steady-state fit is
a kernel or a memory-traffic problem. A slow first fit is a loader, driver,
or compiler problem, and it is the one a user meets first: it is the whole
of the experience for someone who imports the package, scores one row, and
exits, which is exactly the shape of a CLI invocation, a serving process
that starts per request, and a CI job.

The contract
------------

Ten phases, in the order a cold process pays them. The split is chosen so
that each phase has exactly one plausible owner and one plausible fix:

| Phase | Cost is | Fixed by |
| --- | --- | --- |
| `py_import` | executing `mojotrees/__init__.py` and its module graph | deferring imports |
| `ext_load` | `dlopen` of `_mojotrees.so` and its dependency closure | fewer/smaller dylibs, better rpaths |
| `runtime_load` | MAX async runtime initialization inside that load | nothing local; a toolchain property |
| `device_discovery` | enumerating accelerators | not opening a device you will not use |
| `context_create` | `DeviceContext()` and driver context setup | one session per process, not per fit |
| `kernel_create` | first `compile_function` / first fused `enqueue_function` per kernel | reusing `DeviceFunction` handles |
| `first_alloc` | first device buffer of each role | pooling (`PoolLedger` in gpu_runtime.mojo) |
| `first_transfer` | first host-to-device copy of the binned matrix | residency (`ResidencyLedger`) |
| `first_fit` | the first complete `fit`, everything above excluded | all of the above |
| `warm_fit` | a repeated `fit` on a live process | the steady-state work |

The number that matters is `first_fit - warm_fit`, and the number that
tells you whether a fix is worth building is how much of that difference
lands in one phase.

Three of the ten (`py_import`, `ext_load`, `runtime_load`) cannot be
measured from inside Mojo, because by the time Mojo code runs they have
already happened. They are *supplied*: a host harness measures them and
hands the value in through `supply`. `phase_origin` says which is which,
and a report marks a supplied phase so a reader never mistakes a value the
process observed for a value somebody typed in.

Instrumentation cost
--------------------

Disabled by default and disabled means disabled: `clock()` returns 0
without reading a clock, and `record` on a zero start does no arithmetic
beyond the call count. That matters because the intended wiring puts
`record` calls on paths that also run in steady state (buffer allocation,
kernel launch), and instrumentation that costs something when off is
instrumentation that gets removed later by someone benchmarking.

Call counts are always kept, following `PhaseCounters`: an integer add is
cheap enough to always pay, and counts are what a test can assert on
without turning timing on and becoming machine dependent.

No global state
---------------

There is deliberately no module-level trace, no lazily-initialized
singleton, and no `atexit`-style hook. A `StartupTrace` is a value its
owner holds, so two estimators in one process keep two, an estimator can
be dropped without coordinating with anything, and interpreter shutdown
frees the last one by the ordinary rules. `merge` is how a caller that
wants one number across several owners gets it, explicitly.

The same rule is why nothing here caches a `DeviceContext`: a process-wide
context would outlive the estimator that opened it and would have to be
torn down by a hook that runs at an unspecified point during interpreter
finalization. `GpuSession` in gpu_runtime.mojo is the per-estimator owner,
and its `close()` is the supported teardown.

What this module does not do
----------------------------

It opens no device, queries no capability, and decides nothing about where
training runs. Device policy is src/mojotrees/device_policy.mojo and the
tiling policy is src/mojotrees/gpu_tiling.mojo; duplicating either here
would give the two layers a way to disagree. Everything here is a plain
value type with no device dependency, so the whole contract is exercisable
on a machine with no accelerator.

The one thing this module *hands* the device policy is `SessionState`: two
booleans and a warm-up level saying how much of the one-time cost a process
has already paid. It travels as data because `decide_device` is pure by
contract, and the dependency runs one way only, device_policy importing this
module and never the reverse, so this file stays a leaf that imports nothing
but the standard library. What the policy is allowed to do with it is
bounded at the struct: report it, and warn on a cold session. Selecting a
backend on it would be a performance rule, and there is no measurement here
to put behind one.

It also does not cache compiled kernels, and nothing in this repository
does. See `BuildIdentity` for what such a cache would have to key on and
docs/STARTUP_LATENCY.md for the audit of what the toolchain actually
offers, which is a per-context in-memory handle (`compile_function`) and
no documented on-disk cache at all.

Environment contract, matching the `MOJOTREES_` convention in
parallel.mojo, gpu_tiling.mojo, and gpu_runtime.mojo:

- `MOJOTREES_STARTUP_TRACE=1` turns on startup timing. Off by default, so
  an untraced process reads no clock.
- `MOJOTREES_GPU_WARMUP`: `off` (default), `train`, or `all`. Selects how
  much kernel creation a caller front-loads before the first fit. `off`
  reproduces today's behavior exactly, where every kernel is created on
  the launch that first needs it.
- `MOJOTREES_STARTUP_REPORT_FD`: reserved, unread here. See the handoff;
  emitting the report is a call-site decision, not this module's.
"""

from std.os import getenv
from std.time import perf_counter_ns


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

comptime PHASE_PY_IMPORT = 0
comptime PHASE_EXT_LOAD = 1
comptime PHASE_RUNTIME_LOAD = 2
comptime PHASE_DEVICE_DISCOVERY = 3
comptime PHASE_CONTEXT_CREATE = 4
comptime PHASE_KERNEL_CREATE = 5
comptime PHASE_FIRST_ALLOC = 6
comptime PHASE_FIRST_TRANSFER = 7
comptime PHASE_FIRST_FIT = 8
comptime PHASE_WARM_FIT = 9
comptime N_STARTUP_PHASES = 10

# Where a phase's value can come from.
comptime ORIGIN_SUPPLIED = 0
"""Measurable only outside the native library: it is already over by the
time any Mojo code runs. The host harness measures it and calls `supply`."""

comptime ORIGIN_NATIVE = 1
"""Measurable in place, by bracketing the work with `clock()`/`record`."""


def phase_name(phase: Int) -> String:
    """The stable report key for a phase. These strings are the schema:
    docs/STARTUP_LATENCY.md and python/mojotrees/diagnostics.py both
    depend on them, so renaming one is a breaking change to both."""
    if phase == PHASE_PY_IMPORT:
        return String("py_import")
    if phase == PHASE_EXT_LOAD:
        return String("ext_load")
    if phase == PHASE_RUNTIME_LOAD:
        return String("runtime_load")
    if phase == PHASE_DEVICE_DISCOVERY:
        return String("device_discovery")
    if phase == PHASE_CONTEXT_CREATE:
        return String("context_create")
    if phase == PHASE_KERNEL_CREATE:
        return String("kernel_create")
    if phase == PHASE_FIRST_ALLOC:
        return String("first_alloc")
    if phase == PHASE_FIRST_TRANSFER:
        return String("first_transfer")
    if phase == PHASE_FIRST_FIT:
        return String("first_fit")
    if phase == PHASE_WARM_FIT:
        return String("warm_fit")
    return String("unknown")


def phase_origin(phase: Int) -> Int:
    """`ORIGIN_SUPPLIED` for the three phases that are over before Mojo
    code runs, `ORIGIN_NATIVE` for the rest.

    A phase's origin is a property of when it happens, not of who wants
    the number, which is why it is a function of the phase alone.
    """
    if phase == PHASE_PY_IMPORT:
        return ORIGIN_SUPPLIED
    if phase == PHASE_EXT_LOAD:
        return ORIGIN_SUPPLIED
    if phase == PHASE_RUNTIME_LOAD:
        return ORIGIN_SUPPLIED
    return ORIGIN_NATIVE


def startup_origin_name(origin: Int) -> String:
    if origin == ORIGIN_SUPPLIED:
        return String("supplied")
    if origin == ORIGIN_NATIVE:
        return String("native")
    return String("unknown")


def phase_is_one_time(phase: Int) -> Bool:
    """True for the phases a *second* fit in the same process should not
    pay again. `warm_fit` is the only phase that is not one-time, and
    `first_fit` is one-time in the sense that only the first one is
    recorded under it.

    This is the definition the cold total is built on, so a phase added
    later has to answer this question before it can be summed.
    """
    return phase >= 0 and phase < N_STARTUP_PHASES and phase != PHASE_WARM_FIT


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


@fieldwise_init
struct PhaseRecord(Copyable, Movable):
    """One phase's accumulated observations, as a value a reader can hold.

    `first_nanos` is kept separately from `nanos` because for every phase
    except `warm_fit` the first occurrence is the interesting one: the
    second `context_create` in a process is a different event from the
    first even though it runs the same code, and averaging them hides the
    thing being measured.
    """

    var calls: Int
    var nanos: Int
    var first_nanos: Int
    var supplied: Bool
    """A value handed in through `supply` rather than observed by a clock
    this process read. Never silently mixed with observed values."""

    def is_empty(self) -> Bool:
        return self.calls == 0

    def mean_nanos(self) -> Int:
        """Total over calls, truncated, or 0 when nothing was recorded.
        Only meaningful for `warm_fit`; every other phase should be read
        through `first_nanos`."""
        if self.calls < 1:
            return 0
        return self.nanos // self.calls


struct StartupTrace(Copyable, Movable):
    """Per-phase call counts and elapsed nanoseconds for one owner.

    Parallel lists rather than a list of `PhaseRecord`, matching
    `PhaseCounters` in gpu_runtime.mojo: the counters are read and written
    far more often than they are inspected, and `record_of` builds the
    struct on the way out for whoever wants one.

    Disabled is the default. A disabled trace never reads a clock, so the
    `clock()`/`record` pair can sit on a path that also runs in steady
    state without costing that path anything.
    """

    var calls: List[Int]
    var nanos: List[Int]
    var first_nanos: List[Int]
    var supplied: List[Bool]
    var enabled: Bool

    def __init__(out self, enabled: Bool = False):
        self.calls = List[Int](capacity=N_STARTUP_PHASES)
        self.nanos = List[Int](capacity=N_STARTUP_PHASES)
        self.first_nanos = List[Int](capacity=N_STARTUP_PHASES)
        self.supplied = List[Bool](capacity=N_STARTUP_PHASES)
        for _ in range(N_STARTUP_PHASES):
            self.calls.append(0)
            self.nanos.append(0)
            self.first_nanos.append(0)
            self.supplied.append(False)
        self.enabled = enabled

    @staticmethod
    def from_env() -> StartupTrace:
        """A trace configured by `MOJOTREES_STARTUP_TRACE`.

        Deliberately not a singleton and deliberately not cached: the
        caller owns the value, so two estimators get two traces and
        neither has to be torn down by a hook.
        """
        return StartupTrace(getenv("MOJOTREES_STARTUP_TRACE") == "1")

    # -- measurement ------------------------------------------------------

    def clock(self) -> Int:
        """A start timestamp, or 0 when tracing is off.

        The zero is the whole point: `record` treats a zero start as
        "count it, do not time it", so a disabled trace performs no clock
        read at either end.
        """
        if not self.enabled:
            return 0
        return Int(perf_counter_ns())

    def record(mut self, phase: Int, started: Int) raises:
        """Count one occurrence of `phase`, and time it when `started` came
        from `clock()` on an enabled trace."""
        if phase < 0 or phase >= N_STARTUP_PHASES:
            raise Error("unknown startup phase ", phase)
        var first = self.calls[phase] == 0
        self.calls[phase] += 1
        if not self.enabled or started <= 0:
            return
        var elapsed = Int(perf_counter_ns()) - started
        if elapsed <= 0:
            return
        self.nanos[phase] += elapsed
        if first:
            self.first_nanos[phase] = elapsed

    def supply(mut self, phase: Int, nanos: Int) raises:
        """Record a duration this process did not observe.

        Used for the three phases that are over before any Mojo code runs.
        The value is stored whether or not tracing is enabled, because the
        caller has already paid to measure it and dropping it would only
        make the report harder to read; it is flagged `supplied` so no
        reader mistakes it for something this process timed.
        """
        if phase < 0 or phase >= N_STARTUP_PHASES:
            raise Error("unknown startup phase ", phase)
        if nanos < 0:
            raise Error("a supplied phase duration cannot be negative")
        if phase_origin(phase) != ORIGIN_SUPPLIED:
            raise Error(
                "phase ",
                phase_name(phase),
                " is measurable natively; use clock()/record instead of"
                " supplying a value for it",
            )
        var first = self.calls[phase] == 0
        self.calls[phase] += 1
        self.nanos[phase] += nanos
        self.supplied[phase] = True
        if first:
            self.first_nanos[phase] = nanos

    # -- reading ----------------------------------------------------------

    def record_of(self, phase: Int) -> PhaseRecord:
        """One phase as a value. An out-of-range phase reads as empty
        rather than raising, so a report loop cannot fail on a phase it
        does not know about."""
        if phase < 0 or phase >= N_STARTUP_PHASES:
            return PhaseRecord(0, 0, 0, False)
        return PhaseRecord(
            self.calls[phase],
            self.nanos[phase],
            self.first_nanos[phase],
            self.supplied[phase],
        )

    def calls_of(self, phase: Int) -> Int:
        if phase < 0 or phase >= N_STARTUP_PHASES:
            return 0
        return self.calls[phase]

    def nanos_of(self, phase: Int) -> Int:
        if phase < 0 or phase >= N_STARTUP_PHASES:
            return 0
        return self.nanos[phase]

    def first_nanos_of(self, phase: Int) -> Int:
        if phase < 0 or phase >= N_STARTUP_PHASES:
            return 0
        return self.first_nanos[phase]

    def observed(self, phase: Int) -> Bool:
        """Whether anything at all was recorded for `phase`. A phase with
        no observation is reported as absent, never as zero: a GPU phase
        on a CPU-only run did not take no time, it did not happen."""
        return self.calls_of(phase) > 0

    def has_timings(self) -> Bool:
        """True when at least one phase carries a duration. False for a
        trace that only counted, which is what an untraced run produces,
        and a report of such a trace is not a measurement."""
        for p in range(N_STARTUP_PHASES):
            if self.nanos[p] > 0:
                return True
        return False

    def cold_nanos(self) -> Int:
        """First-occurrence total over every one-time phase.

        This is the "what a fresh process pays" number. `warm_fit` is
        excluded by `phase_is_one_time`, and `first_fit` is included, so
        the total is end to end: interpreter start through the first
        prediction.
        """
        var total = 0
        for p in range(N_STARTUP_PHASES):
            if phase_is_one_time(p):
                total += self.first_nanos[p]
        return total

    def warm_nanos(self) -> Int:
        """Mean recorded `warm_fit`, or 0 when none was recorded."""
        return self.record_of(PHASE_WARM_FIT).mean_nanos()

    def overhead_nanos(self) -> Int:
        """`first_fit - warm_fit`: the part of the first fit that a warm
        fit does not repeat, and therefore the part worth attacking.

        `-1` when there is nothing to subtract, which is any run that
        fitted once. Returning the first fit's own duration in that case
        would report the entire fit as overhead, which is the single most
        misleading number this module could produce.

        Otherwise clamped at zero. A negative difference is real (a warm
        fit can be slower than the first, from a different bagging draw or
        plain noise) but it is not an overhead, and reporting a negative
        one invites somebody to subtract it from a total.
        """
        if self.calls[PHASE_WARM_FIT] < 1:
            return -1
        if self.calls[PHASE_FIRST_FIT] < 1:
            return -1
        var d = self.first_nanos[PHASE_FIRST_FIT] - self.warm_nanos()
        if d < 0:
            return 0
        return d

    def merge(mut self, other: StartupTrace):
        """Fold another owner's trace into this one.

        The explicit alternative to a process-wide singleton: a caller
        that wants one number across several estimators asks for it.

        A phase's first occurrence is taken from whichever trace has a
        non-zero one, `self` winning a tie. That is merge order, not
        chronological order: nothing here records wall-clock timestamps,
        so which owner genuinely went first is not knowable, and a caller
        that needs it has to merge in the order the owners ran.
        """
        for p in range(N_STARTUP_PHASES):
            self.calls[p] += other.calls[p]
            self.nanos[p] += other.nanos[p]
            if self.first_nanos[p] == 0:
                self.first_nanos[p] = other.first_nanos[p]
            if other.supplied[p]:
                self.supplied[p] = True

    def reset(mut self):
        """Drop every observation, keeping the enabled flag. Lets one
        owner measure a cold sequence and then a warm one without
        constructing a second trace."""
        for p in range(N_STARTUP_PHASES):
            self.calls[p] = 0
            self.nanos[p] = 0
            self.first_nanos[p] = 0
            self.supplied[p] = False

    def report(self) -> String:
        """The wire format, one phase per line:

        ```
        startup.<phase> <calls> <total_ns> <first_ns> <origin> <observed>
        ```

        Six fields, space separated, phases always in declaration order,
        and every phase always present so a consumer can index rather than
        search. `origin` is `supplied` or `native`; `observed` is `1` or
        `0` and is the field that distinguishes "did not happen" from
        "took no measurable time".

        Two trailing summary lines, `startup.cold_ns` and
        `startup.warm_ns`, so a reader that wants only the headline does
        not have to reimplement `phase_is_one_time`.

        Intended for a harness, a benchmark, or a debug print.
        python/mojotrees/diagnostics.py parses exactly this and is the
        only supported parser.
        """
        var out = String("")
        for p in range(N_STARTUP_PHASES):
            out += "startup." + phase_name(p)
            out += " " + String(self.calls[p])
            out += " " + String(self.nanos[p])
            out += " " + String(self.first_nanos[p])
            out += " " + startup_origin_name(phase_origin(p))
            if self.calls[p] > 0:
                out += " 1\n"
            else:
                out += " 0\n"
        out += "startup.cold_ns " + String(self.cold_nanos()) + "\n"
        out += "startup.warm_ns " + String(self.warm_nanos()) + "\n"
        return out


# ---------------------------------------------------------------------------
# First fit versus warm fit
# ---------------------------------------------------------------------------


struct FitLatency(Copyable, Movable):
    """Routes each fit to `first_fit` or `warm_fit` for one owner.

    The distinction is positional, not temporal: the first fit an owner
    performs is the cold one regardless of how much wall clock passed
    before it, because "cold" here means "before the one-time costs were
    paid", and they are paid by that fit.

    Held separately from `StartupTrace` so that an owner can hand a trace
    to something else (a merge, a report) without that thing being able to
    change which fit counts as first.
    """

    var fits: Int

    def __init__(out self):
        self.fits = 0

    def phase_for_next(self) -> Int:
        """Which phase the next fit will be recorded under. Lets a caller
        decide whether to bother timing at all before it starts."""
        if self.fits == 0:
            return PHASE_FIRST_FIT
        return PHASE_WARM_FIT

    def note_fit(mut self, mut trace: StartupTrace, started: Int) raises:
        """Close a fit measurement started with `trace.clock()`."""
        var phase = self.phase_for_next()
        self.fits += 1
        trace.record(phase, started)

    def is_cold(self) -> Bool:
        return self.fits == 0


# ---------------------------------------------------------------------------
# Kernel creation
# ---------------------------------------------------------------------------

comptime WARMUP_OFF = 0
"""Every kernel is created on the launch that first needs it. Today's
behavior, and the default."""

comptime WARMUP_TRAIN = 1
"""Create the kernels a training run is certain to launch, before the
first round, so their creation cost lands in `kernel_create` rather than
inside the first round's `first_fit`."""

comptime WARMUP_ALL = 2
"""Also create the kernels only some workloads launch (prediction,
device-side split search). Costs creation time a run may never use."""


def env_warmup_level() -> Int:
    """`MOJOTREES_GPU_WARMUP` as a level. Unset or unrecognized is
    `WARMUP_OFF`, so an unknown value never silently front-loads work."""
    var s = getenv("MOJOTREES_GPU_WARMUP")
    if s == "train":
        return WARMUP_TRAIN
    if s == "all":
        return WARMUP_ALL
    return WARMUP_OFF


def warmup_level_name(level: Int) -> String:
    if level == WARMUP_OFF:
        return String("off")
    if level == WARMUP_TRAIN:
        return String("train")
    if level == WARMUP_ALL:
        return String("all")
    return String("unknown")


# ---------------------------------------------------------------------------
# What a device decision needs to know about startup
# ---------------------------------------------------------------------------


@fieldwise_init
struct SessionState(Copyable, Movable):
    """How much of the one-time cost this process has already paid.

    The one thing the device policy needs from this module, and the reason
    it is a value rather than a probe: `device_policy.decide_device` is pure
    by contract, so anything it accounts for has to arrive as data a caller
    could have serialized.

    Why the policy cares. Two of the ten phases above (`context_create` and
    `kernel_create`) are paid by whichever run first reaches the device, so
    the *same* workload costs a cold process strictly more than a warm one.
    A decision report that does not say which of the two it is describing
    invites a reader to compare a cold GPU run against a warm CPU one.

    What the policy may do with it is deliberately bounded: report it, and
    warn on a cold session. It may not select the GPU because a session is
    warm, and it may not refuse the GPU because one is cold. Either would be
    a performance rule, and a performance rule needs a measurement behind it
    (`crossover_rules()` in device_policy.mojo, which is empty).

    - `context_open`: a `DeviceContext` is already open and will be reused,
      so `context_create` is already paid. `GpuSession` in gpu_runtime.mojo
      is the owner that can answer this truthfully.
    - `kernels_ready`: the kernels this run will launch have already been
      created, so `kernel_create` is already paid.
    - `warmup_level`: the `WARMUP_*` level in effect, which is what decides
      whether kernel creation lands in `kernel_create` or inside the first
      round's `first_fit`.
    """

    var context_open: Bool
    var kernels_ready: Bool
    var warmup_level: Int

    @staticmethod
    def cold() -> SessionState:
        """A process that has not opened a device. The conservative answer,
        and the correct one for any caller that does not own a session."""
        return SessionState(False, False, WARMUP_OFF)

    @staticmethod
    def from_env() -> SessionState:
        """Cold, with the warm-up level the environment asks for.

        Still cold: `MOJOTREES_GPU_WARMUP` says what a caller *intends* to
        front-load, not what it has already done, and reading an intention as
        an accomplishment is how `kernel_create` gets reported as paid on a
        process that never opened a device.
        """
        return SessionState(False, False, env_warmup_level())

    def is_cold(self) -> Bool:
        return not self.context_open

    def paid_nothing(self) -> Bool:
        return not self.context_open and not self.kernels_ready

    def describe(self) -> String:
        var out = String("context_open=")
        if self.context_open:
            out += "true"
        else:
            out += "false"
        out += " kernels_ready="
        if self.kernels_ready:
            out += "true"
        else:
            out += "false"
        out += " warmup=" + warmup_level_name(self.warmup_level)
        return out^


def session_state_from_trace(
    trace: StartupTrace, warmup_level: Int = WARMUP_OFF
) -> SessionState:
    """Read a `SessionState` off what a trace has actually observed.

    `context_create` and `kernel_create` are counted whether or not timing is
    on (see `StartupTrace.record`), so this answers correctly on an untraced
    run too, which is the run a device decision is usually being made on.

    Counts, not durations. A phase that was observed in zero measurable time
    still happened, and `observed` is the field that carries that distinction
    through the whole contract.
    """
    return SessionState(
        trace.observed(PHASE_CONTEXT_CREATE),
        trace.observed(PHASE_KERNEL_CREATE),
        warmup_level,
    )


struct WarmupPlan(Copyable, Movable):
    """Which kernels an owner intends to create up front, and what each
    one cost when it was created.

    Kernel ids are the caller's, not this module's: `gpu_runtime.mojo`
    already numbers them (`KERNEL_HIST_ATOMIC` and friends) and a second
    numbering here would be a second thing to keep in sync. The plan
    stores whatever ids it is given, so it also survives that inventory
    growing.

    This is bookkeeping and attribution, not a handle cache. It holds no
    `DeviceFunction`, and it cannot: `DeviceFunction` is parameterized on
    the kernel function itself, so every kernel has a distinct type and a
    homogeneous list of them does not exist. A real handle cache is one
    typed field per kernel on whichever struct owns the context, which is
    a change to `GpuHistogramBuilder` or `GpuSession` and therefore not
    this lane's to make. See handoffs/performance_15_startup.md.
    """

    var level: Int
    var kernel_ids: List[Int]
    var created: List[Bool]
    var nanos: List[Int]

    def __init__(out self, level: Int = WARMUP_OFF) raises:
        if level < WARMUP_OFF or level > WARMUP_ALL:
            raise Error("unknown warm-up level ", level)
        self.level = level
        self.kernel_ids = List[Int]()
        self.created = List[Bool]()
        self.nanos = List[Int]()

    @staticmethod
    def from_env() raises -> WarmupPlan:
        return WarmupPlan(env_warmup_level())

    def _index_of(self, kernel_id: Int) -> Int:
        for i in range(len(self.kernel_ids)):
            if self.kernel_ids[i] == kernel_id:
                return i
        return -1

    def include(mut self, kernel_id: Int) -> Bool:
        """Add `kernel_id` to the plan. True when it was not already in
        it, so a caller can assemble a plan from overlapping sets without
        checking first.

        Does not raise: a duplicate is the normal case this returns False
        for, and there is no other way to be wrong. A caller building a
        plan from several overlapping sets should not need a `try`.
        """
        if self._index_of(kernel_id) >= 0:
            return False
        self.kernel_ids.append(kernel_id)
        self.created.append(False)
        self.nanos.append(0)
        return True

    def planned(self) -> Int:
        return len(self.kernel_ids)

    def is_planned(self, kernel_id: Int) -> Bool:
        return self._index_of(kernel_id) >= 0

    def note_created(mut self, kernel_id: Int, nanos: Int) raises:
        """Record that `kernel_id` was created and what it cost. `nanos`
        is 0 when the owning trace is disabled, which is why creation is
        tracked by the boolean and not by a non-zero duration."""
        var i = self._index_of(kernel_id)
        if i < 0:
            raise Error("kernel ", kernel_id, " is not in this warm-up plan")
        if nanos < 0:
            raise Error("a kernel creation cannot take negative time")
        self.created[i] = True
        self.nanos[i] = nanos

    def created_count(self) -> Int:
        var n = 0
        for i in range(len(self.created)):
            if self.created[i]:
                n += 1
        return n

    def pending_count(self) -> Int:
        return self.planned() - self.created_count()

    def total_nanos(self) -> Int:
        var total = 0
        for i in range(len(self.nanos)):
            total += self.nanos[i]
        return total

    def slowest_kernel(self) -> Int:
        """The id whose creation cost the most, or -1 when nothing was
        created or nothing was timed. The one number that says whether
        warm-up is worth doing at all is dominated by this kernel."""
        var best = -1
        var best_nanos = 0
        for i in range(len(self.nanos)):
            if self.created[i] and self.nanos[i] > best_nanos:
                best_nanos = self.nanos[i]
                best = self.kernel_ids[i]
        return best

    def report(self) -> String:
        """`warmup.level`, `warmup.planned`, `warmup.created`, and
        `warmup.total_ns`, plus one `warmup.kernel <id> <created> <ns>`
        line per planned kernel."""
        var out = String("warmup.level ")
        out += warmup_level_name(self.level) + "\n"
        out += "warmup.planned " + String(self.planned()) + "\n"
        out += "warmup.created " + String(self.created_count()) + "\n"
        out += "warmup.total_ns " + String(self.total_nanos()) + "\n"
        for i in range(len(self.kernel_ids)):
            out += "warmup.kernel " + String(self.kernel_ids[i])
            if self.created[i]:
                out += " 1"
            else:
                out += " 0"
            out += " " + String(self.nanos[i]) + "\n"
        return out


# ---------------------------------------------------------------------------
# What a persistent cache would have to key on
# ---------------------------------------------------------------------------
#
# Nothing here writes, reads, or implies a cache. This repository has no
# persistent compilation cache and the toolchain does not document one:
# `DeviceContext.compile_function` returns a handle whose lifetime is the
# context's, and the only documented way to load precompiled device code is
# `DeviceContext.load_function`, which takes vendor assembly (PTX or SASS)
# and so does not cover the one backend this project has ever run on. The
# audit is written out in docs/STARTUP_LATENCY.md.
#
# What follows is therefore a specification, not an implementation: the
# identity a cached artifact would have to carry for reuse to be safe. It
# lives here rather than in the handoff because the failure mode of getting
# it wrong is a silently stale kernel, and a struct that has to be
# constructed is harder to leave a field out of than a bullet list.

comptime CACHE_KEY_VERSION = 1
"""Bumped whenever the *meaning* of any field below changes, so an artifact
written by an older layout is rejected rather than reinterpreted. Distinct
from the toolchain version: this versions the key, not the compiler."""


@fieldwise_init
struct BuildIdentity(Copyable, Movable):
    """Everything that has to match for compiled device code to be reusable.

    Every field is supplied by the caller. Nothing is probed here, because
    a probe would make this module depend on a device and because several
    of these values are build-time facts a running process cannot
    rediscover.

    - `toolchain`: the Mojo/MAX version that produced the code. Two
      toolchains can emit different code from identical source, so this is
      part of the key even when nothing else changed.
    - `target_arch`: the host architecture the extension was built for.
    - `device_api`: `metal`, `cuda`, or `hip`. The same source compiles to
      different device code per backend.
    - `driver_version`: `DeviceContext.get_api_version()`. A driver upgrade
      can change what the same input compiles to, and is the invalidation
      most likely to be forgotten, because nothing about the package
      changed.
    - `kernel_source_hash`: a digest of the kernel sources that went in.
      Supplied by whatever builds the artifact; recomputing it at runtime
      would mean shipping the sources.
    - `compile_options`: the option string passed to `compile_function`,
      which is a compilation input like any other.

    Deliberately absent: anything about the *data*. A kernel does not
    depend on row count, feature count, or bin count, and keying on them
    would produce a cache that never hits.
    """

    var toolchain: String
    var target_arch: String
    var device_api: String
    var driver_version: Int
    var kernel_source_hash: UInt64
    var compile_options: String

    def key(self) -> String:
        """The identity as one string, for use as a filename or a map key.

        Fields are joined with `|`, which none of them may contain; a
        caller that might have one should reject it before constructing an
        identity, since a separator inside a field lets two different
        builds produce one key.
        """
        var out = String("v") + String(CACHE_KEY_VERSION)
        out += "|" + self.toolchain
        out += "|" + self.target_arch
        out += "|" + self.device_api
        out += "|" + String(self.driver_version)
        out += "|" + String(self.kernel_source_hash)
        out += "|" + self.compile_options
        return out

    def matches(self, other: BuildIdentity) -> Bool:
        """Whether an artifact built under `other` may be used here.

        Whole-key equality, with no partial credit. There is no field here
        that is safe to ignore on a mismatch, and a cache that guesses
        which mismatches are benign is a cache that eventually serves a
        kernel compiled for a different device.
        """
        if self.toolchain != other.toolchain:
            return False
        if self.target_arch != other.target_arch:
            return False
        if self.device_api != other.device_api:
            return False
        if self.driver_version != other.driver_version:
            return False
        if self.kernel_source_hash != other.kernel_source_hash:
            return False
        if self.compile_options != other.compile_options:
            return False
        return True

    def is_complete(self) -> Bool:
        """Whether every field that identifies a build was actually filled
        in. An identity with an unknown field must never be written to or
        looked up in a cache: an empty string is not a value, it is a
        missing value, and treating it as one makes every incomplete
        identity collide with every other."""
        if self.toolchain.byte_length() == 0:
            return False
        if self.target_arch.byte_length() == 0:
            return False
        if self.device_api.byte_length() == 0:
            return False
        if self.driver_version <= 0:
            return False
        if self.kernel_source_hash == UInt64(0):
            return False
        return True
