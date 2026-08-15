"""Readable startup reports, from values somebody else measured.

The native contract lives in `src/mojotrees/initialization.mojo`: ten
phases, in the order a cold process pays them, and a `StartupTrace` that
records them. This module is the presentation half. It takes phase
durations that were already measured and turns them into a table, a
sentence, or a JSON-serializable dict.

    from mojotrees import diagnostics

    report = diagnostics.report_from_trace(trace_text)
    print(report)

It measures nothing. That is a deliberate limitation, not an omission:
three of the ten phases (`py_import`, `ext_load`, `runtime_load`) are over
before any code in this package runs, so a function here that tried to
time them would be timing its own second import and reporting a number
that is wrong by the entire cost of the first one. The commands that
produce honest values for those phases are in
[docs/STARTUP_LATENCY.md](../../docs/STARTUP_LATENCY.md), and they all
involve a fresh interpreter.

For the same reason nothing here imports the compiled extension.
`environment_snapshot()` reads the filesystem and `sys.modules`, so asking
what the install looks like does not itself load it. `extension_loaded()`
answers "has something else already imported it" without being the
something else.

What this module deliberately does not decide
---------------------------------------------
Anything about devices. Whether an accelerator is available, whether
`auto` resolves to it, what backend it is, how much memory it has, and
whether a workload is covered are all `mojotrees.device_selection`, which
is a policy with reasons and a report of its own. A second opinion here
would be a second place for that policy to live and a first place for it
to be wrong. `environment_snapshot()` lists the `MOJOTREES_*` variables
that are set, because which variables are in effect is a fact about the
process, but it does not interpret the device ones.

Install kinds
-------------
`describe_install()` reports which artifact is answering, because the two
kinds have different first-use costs and different failure modes:

- **source build**: `bindings/build.sh` wrote `_mojotrees.so` next to this
  file, and the extension resolves the MAX runtime through an absolute
  rpath into the pixi environment that built it. Moving or deleting that
  environment breaks the import.
- **wheel, `dylibs` layout**: `packaging/build_wheel.sh` bundled the four
  MAX runtime dylibs into `mojotrees/.dylibs` and rewrote the rpath to
  `@loader_path/.dylibs`, so the loader stays inside the package.
- **wheel, `libs` layout**: `packaging/linux/build_wheel_linux.sh` staged
  the ELF closure into `mojotrees/.libs` by soname and set the RPATH to
  `$ORIGIN/.libs`. The staged set is whatever the closure turned out to
  be, not a fixed list, which is why `missing_runtime_libs` applies only
  to the macOS layout.

The distinction is visible without loading anything: a `.dylibs` or
`.libs` directory next to the extension means the wheel layout, and which
one it is names the builder. What that implies for load time, and what
`tools/inspect_startup_artifacts.py` reports about it, is in the startup
document.
"""

import os
import platform
import sys

__all__ = [
    "BUNDLED_RUNTIME_LIBS",
    "PHASES",
    "PHASE_NAMES",
    "WATCHED_ENV",
    "InstallDescription",
    "PhaseTiming",
    "StartupReport",
    "WarmupSummary",
    "describe_install",
    "environment_snapshot",
    "extension_loaded",
    "format_duration",
    "parse_trace",
    "parse_warmup",
    "report_from_trace",
    "report_from_values",
]


class _Phase:
    """One phase of the startup contract, as documentation."""

    __slots__ = ("name", "index", "origin", "one_time", "summary")

    def __init__(self, name, index, origin, one_time, summary):
        self.name = name
        self.index = index
        self.origin = origin
        self.one_time = one_time
        self.summary = summary

    def __repr__(self):
        return "<phase %s %s>" % (self.name, self.origin)


#: The phases, in the order `StartupTrace.report()` emits them. The index,
#: the name, and the origin must match `phase_name` and `phase_origin` in
#: src/mojotrees/initialization.mojo; they are one contract in two files
#: and the startup document is where the pairing is written down.
PHASES = (
    _Phase(
        "py_import",
        0,
        "supplied",
        True,
        "executing mojotrees/__init__.py and the modules it pulls in",
    ),
    _Phase(
        "ext_load",
        1,
        "supplied",
        True,
        "dlopen of _mojotrees.so and its dependency closure",
    ),
    _Phase(
        "runtime_load",
        2,
        "supplied",
        True,
        "MAX async runtime initialization during that load",
    ),
    _Phase(
        "device_discovery",
        3,
        "native",
        True,
        "enumerating accelerators",
    ),
    _Phase(
        "context_create",
        4,
        "native",
        True,
        "opening a DeviceContext and its driver-side context",
    ),
    _Phase(
        "kernel_create",
        5,
        "native",
        True,
        "creating each device function the first time it is launched",
    ),
    _Phase(
        "first_alloc",
        6,
        "native",
        True,
        "the first device buffer of each role",
    ),
    _Phase(
        "first_transfer",
        7,
        "native",
        True,
        "the first host-to-device copy of the binned matrix",
    ),
    _Phase(
        "first_fit",
        8,
        "native",
        True,
        "the first complete fit, with the phases above excluded",
    ),
    _Phase(
        "warm_fit",
        9,
        "native",
        False,
        "a repeated fit on a process that has already paid the rest",
    ),
)

#: Phase names in contract order.
PHASE_NAMES = tuple(phase.name for phase in PHASES)

_BY_NAME = {phase.name: phase for phase in PHASES}

#: Environment variables that change what a startup measurement means.
#: Listed, never interpreted: the device ones belong to
#: `mojotrees.device_selection` and the parallelism ones to
#: `src/mojotrees/parallel.mojo`.
WATCHED_ENV = (
    "MOJOTREES_STARTUP_TRACE",
    "MOJOTREES_STARTUP_REPORT_FD",
    "MOJOTREES_GPU_WARMUP",
    "MOJOTREES_GPU_TRANSFER",
    "MOJOTREES_GPU_TRANSFER_UNPROVEN",
    "MOJOTREES_GPU_TRACE",
    "MOJOTREES_GPU_STAGING_SLOTS",
    "MOJOTREES_GPU_HIST_STRATEGY",
    "MOJOTREES_GPU_SPLIT_STRATEGY",
    "MOJOTREES_GPU_BLOCK_THREADS",
    "MOJOTREES_GPU_VERIFY_ROWS",
    "MOJOTREES_DISABLE_GPU",
    "MOJOTREES_AUTO_MIN_CELLS",
    "MOJOTREES_GPU_BACKEND",
    "MOJOTREES_NUM_WORKERS",
    "MOJOTREES_PARALLEL_MIN_OPS",
    "MODULAR_HOME",
    "MODULAR_CACHE_DIR",
    "MODULAR_ENABLE_PROFILING",
    "MODULAR_DEBUG",
)

#: The four MAX runtime libraries the extension links, in the order
#: `packaging/build_wheel.sh` bundles them. Named here so a report can say
#: which of them a macOS wheel is missing, which is the failure that turns
#: into an unexplained ImportError.
#:
#: macOS only. The Linux builder stages an ELF closure by soname rather
#: than this fixed set, so `missing_runtime_libs` applies this list to the
#: `dylibs` layout and to nothing else.
BUNDLED_RUNTIME_LIBS = (
    "libKGENCompilerRTShared",
    "libAsyncRTMojoBindings",
    "libMSupportGlobals",
    "libAsyncRTRuntimeGlobals",
)


def format_duration(nanos):
    """Nanoseconds as a short human string, or "-" for None.

    Three significant figures and a unit, because a startup report is read
    to find the big phase, not to do arithmetic on. The exact integers are
    always in `to_dict()`.
    """
    if nanos is None:
        return "-"
    n = float(nanos)
    if n < 1e3:
        return "%d ns" % int(n)
    if n < 1e6:
        return "%.3g us" % (n / 1e3)
    if n < 1e9:
        return "%.3g ms" % (n / 1e6)
    return "%.3g s" % (n / 1e9)


class PhaseTiming:
    """One phase's observations.

    `observed` is the field that separates "did not happen" from "took no
    measurable time". A CPU-only run does not open a device context, and
    reporting that as zero nanoseconds would say the opposite of the
    truth.

    `supplied` marks a value the reporting process did not time itself.
    Every `py_import`, `ext_load`, and `runtime_load` value is supplied by
    construction; a native phase should never be.
    """

    __slots__ = ("name", "calls", "total_ns", "first_ns", "origin", "observed")

    def __init__(
        self, name, calls=0, total_ns=0, first_ns=0, origin=None, observed=None
    ):
        phase = _BY_NAME.get(name)
        self.name = name
        self.calls = int(calls)
        self.total_ns = int(total_ns)
        self.first_ns = int(first_ns)
        if origin is None:
            origin = phase.origin if phase is not None else "unknown"
        self.origin = origin
        self.observed = (self.calls > 0) if observed is None else bool(observed)

    @property
    def supplied(self):
        return self.origin == "supplied"

    @property
    def mean_ns(self):
        """Total over calls, or None when nothing was recorded. Only
        meaningful for `warm_fit`; read every other phase through
        `first_ns`."""
        if self.calls < 1:
            return None
        return self.total_ns // self.calls

    def to_dict(self):
        return {
            "name": self.name,
            "calls": self.calls,
            "total_ns": self.total_ns,
            "first_ns": self.first_ns,
            "origin": self.origin,
            "observed": self.observed,
        }

    def __repr__(self):
        return "<PhaseTiming %s calls=%d first=%s>" % (
            self.name,
            self.calls,
            format_duration(self.first_ns if self.observed else None),
        )


class InstallDescription:
    """Which artifact is answering, and what it depends on to load.

    `kind` is one of:

    - `"wheel"`: a staged runtime directory sits next to the extension, so
      the runtime libraries travel with the package.
    - `"source"`: the extension is present with no bundled runtime, which
      is what `bindings/build.sh` produces. It resolves its dependencies
      through an absolute rpath into the environment that built it.
    - `"absent"`: no extension. Every native phase is unmeasurable and
      `import mojotrees` fails.

    `runtime_layout` distinguishes the two builders, which stage different
    things and must not be checked against each other:

    - `"dylibs"`, from `packaging/build_wheel.sh` on macOS, holds exactly
      the four MAX runtime dylibs.
    - `"libs"`, from `packaging/linux/build_wheel_linux.sh`, holds the ELF
      closure staged by soname, so the names carry version suffixes and
      the set is whatever the closure turned out to be.

    Nothing here loads the extension to decide, so the description is
    valid even when importing it would fail, which is precisely when
    somebody needs it.
    """

    __slots__ = (
        "kind",
        "package_dir",
        "extension",
        "runtime_dir",
        "runtime_layout",
        "bundled",
    )

    def __init__(
        self,
        kind,
        package_dir,
        extension,
        runtime_dir,
        bundled,
        runtime_layout=None,
    ):
        self.kind = kind
        self.package_dir = package_dir
        self.extension = extension
        self.runtime_dir = runtime_dir
        self.runtime_layout = runtime_layout
        self.bundled = tuple(bundled)

    @property
    def missing_runtime_libs(self):
        """MAX runtime libraries absent from a macOS bundle.

        Empty for anything but the `dylibs` layout. The Linux builder
        stages an ELF closure rather than a fixed list of four, so
        applying this list to it would report every correct Linux wheel as
        broken.
        """
        if self.runtime_layout != "dylibs":
            return ()
        have = {_library_stem(name) for name in self.bundled}
        return tuple(
            name for name in BUNDLED_RUNTIME_LIBS if name not in have
        )

    def to_dict(self):
        return {
            "kind": self.kind,
            "package_dir": self.package_dir,
            "extension": self.extension,
            "runtime_dir": self.runtime_dir,
            "runtime_layout": self.runtime_layout,
            "bundled": list(self.bundled),
            "missing_runtime_libs": list(self.missing_runtime_libs),
        }

    def __repr__(self):
        return "<InstallDescription %s>" % self.kind


def _library_stem(filename):
    """`libfoo` from `libfoo.dylib`, `libfoo.so`, or `libfoo.so.1.2`.

    `os.path.splitext` is wrong for the ELF case: it turns `libfoo.so.1`
    into `libfoo.so`, so a name comparison against it never matches.
    """
    for marker in (".dylib", ".so"):
        at = filename.find(marker)
        if at > 0:
            return filename[:at]
    return filename


def _extension_candidates(package_dir):
    """Extension filenames to look for, most specific first. The build
    scripts produce a plain `_mojotrees.so` on every platform, but an
    installed wheel may carry an ABI tag, so both spellings are checked."""
    names = []
    try:
        entries = sorted(os.listdir(package_dir))
    except OSError:
        return names
    for entry in entries:
        if not entry.startswith("_mojotrees"):
            continue
        if entry.endswith((".so", ".pyd", ".dylib")):
            names.append(entry)
    return names


def describe_install(package_dir=None):
    """Describe the installed artifact without importing it.

    `package_dir` defaults to the directory holding this file, which is
    the package that would answer an `import mojotrees`.
    """
    if package_dir is None:
        package_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = _extension_candidates(package_dir)
    extension = (
        os.path.join(package_dir, candidates[0]) if candidates else None
    )
    runtime_dir = None
    runtime_layout = None
    bundled = []
    for name, layout in ((".dylibs", "dylibs"), (".libs", "libs")):
        path = os.path.join(package_dir, name)
        if os.path.isdir(path):
            runtime_dir = path
            runtime_layout = layout
            try:
                bundled = sorted(os.listdir(path))
            except OSError:
                bundled = []
            break
    if extension is None:
        kind = "absent"
    elif runtime_dir is not None:
        kind = "wheel"
    else:
        kind = "source"
    return InstallDescription(
        kind, package_dir, extension, runtime_dir, bundled, runtime_layout
    )


def extension_loaded():
    """Whether the compiled extension is already in `sys.modules`.

    Answered by looking, not by importing, so calling this does not change
    the answer. A startup harness uses it to assert that the interpreter
    it is about to measure is genuinely cold.
    """
    return "mojotrees._mojotrees" in sys.modules


def environment_snapshot(environ=None, package_dir=None):
    """Facts about this process that change what a timing means.

    Returns a plain dict, JSON-serializable, with no interpretation in it.
    The `env` entry holds only the watched variables that are actually
    set, so an empty `env` means the defaults were in effect rather than
    that nothing was checked.
    """
    environ = os.environ if environ is None else environ
    install = describe_install(package_dir)
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "executable": sys.executable,
        "system": platform.system(),
        "machine": platform.machine(),
        "install": install.to_dict(),
        "extension_already_loaded": extension_loaded(),
        "env": {
            name: environ[name]
            for name in WATCHED_ENV
            if name in environ and environ[name] != ""
        },
    }


def parse_trace(text):
    """Parse `StartupTrace.report()` into `{phase_name: PhaseTiming}`.

    The wire format, one line per phase in contract order:

        startup.<phase> <calls> <total_ns> <first_ns> <origin> <observed>

    plus the two summary lines `startup.cold_ns` and `startup.warm_ns`,
    which are recomputed here rather than trusted, so a report and its
    summary cannot disagree. Lines that are not `startup.` prefixed are
    ignored, so a trace may be pasted in with surrounding log output.

    Raises `ValueError` on a `startup.` line that names an unknown phase
    or does not carry six fields: a schema that has drifted should be
    loud, because the alternative is a report that silently omits a phase.
    """
    timings = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("startup."):
            continue
        parts = line.split()
        key = parts[0][len("startup.") :]
        if key in ("cold_ns", "warm_ns"):
            continue
        if key not in _BY_NAME:
            raise ValueError("unknown startup phase %r in trace" % key)
        if len(parts) != 6:
            raise ValueError(
                "startup.%s has %d fields, expected 6" % (key, len(parts))
            )
        timings[key] = PhaseTiming(
            key,
            calls=int(parts[1]),
            total_ns=int(parts[2]),
            first_ns=int(parts[3]),
            origin=parts[4],
            observed=parts[5] == "1",
        )
    return timings


class WarmupSummary:
    """What a run front-loaded before its first fit, and what that cost.

    The other half of what `src/mojotrees/initialization.mojo` emits.
    `WarmupPlan.report()` has been writing these lines since the module was
    written and nothing parsed them, so a run with `MOJOTREES_GPU_WARMUP=train`
    produced a record that no report could show. This is the parser.

    `planned` and `created` differ when the run did not reach every kernel it
    intended to create, which is the case a reader wants: a plan of six with
    two created is not a warm-up, it is a warm-up that was interrupted.

    `total_ns` is zero on an untraced run even when kernels were created,
    because `WarmupPlan.note_created` records a duration only when the owning
    trace is enabled. Creation is tracked by the flag, never by a nonzero
    duration, and this class keeps that distinction.
    """

    __slots__ = ("level", "planned", "created", "total_ns", "kernels")

    def __init__(self, level="off", planned=0, created=0, total_ns=0, kernels=()):
        self.level = level
        self.planned = int(planned)
        self.created = int(created)
        self.total_ns = int(total_ns)
        #: `(kernel_id, created, nanos)` per planned kernel, in plan order.
        self.kernels = tuple(kernels)

    @property
    def pending(self):
        """Kernels planned but never created."""
        return max(0, self.planned - self.created)

    @property
    def is_off(self):
        """True for a run that front-loaded nothing, which is the default
        and reproduces the behavior where every kernel is created on the
        launch that first needs it."""
        return self.level == "off" and self.planned == 0

    def slowest_kernel(self):
        """`(kernel_id, nanos)` for the costliest creation, or None when
        nothing was created or nothing was timed. Whether warm-up is worth
        doing at all is dominated by this one."""
        best = None
        for kernel_id, created, nanos in self.kernels:
            if not created or nanos <= 0:
                continue
            if best is None or nanos > best[1]:
                best = (kernel_id, nanos)
        return best

    def to_dict(self):
        return {
            "level": self.level,
            "planned": self.planned,
            "created": self.created,
            "pending": self.pending,
            "total_ns": self.total_ns,
            "kernels": [
                {"id": k, "created": c, "nanos": n} for k, c, n in self.kernels
            ],
        }

    def describe(self):
        if self.is_off:
            return (
                "off: every kernel is created on the launch that first needs"
                " it"
            )
        text = "%s: %d of %d kernels created" % (
            self.level,
            self.created,
            self.planned,
        )
        if self.total_ns > 0:
            text += " in %s" % format_duration(self.total_ns)
            slowest = self.slowest_kernel()
            if slowest is not None:
                text += " (slowest kernel %d, %s)" % (
                    slowest[0],
                    format_duration(slowest[1]),
                )
        else:
            text += " (untimed; set MOJOTREES_STARTUP_TRACE=1 for durations)"
        if self.pending:
            text += "; %d planned kernel(s) never created" % self.pending
        return text

    def __repr__(self):
        return "<WarmupSummary %s %d/%d>" % (
            self.level,
            self.created,
            self.planned,
        )


def parse_warmup(text):
    """Parse `WarmupPlan.report()` output into a `WarmupSummary`.

    The wire format, four scalar lines and one line per planned kernel:

        warmup.level <off|train|all>
        warmup.planned <n>
        warmup.created <n>
        warmup.total_ns <n>
        warmup.kernel <id> <created> <ns>

    Lines that are not `warmup.` prefixed are ignored, so a plan report and
    a startup trace can be pasted in together, which is how they are emitted.
    A text with no `warmup.` lines yields the default off summary rather than
    raising: a run that front-loaded nothing has nothing to report, and that
    is not a schema error.

    Raises `ValueError` on a malformed `warmup.kernel` line, for the same
    reason `parse_trace` does: a schema that has drifted should be loud.
    """
    level = "off"
    planned = 0
    created = 0
    total_ns = 0
    kernels = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("warmup."):
            continue
        parts = line.split()
        key = parts[0][len("warmup.") :]
        if key == "kernel":
            if len(parts) != 4:
                raise ValueError(
                    "warmup.kernel has %d fields, expected 4" % len(parts)
                )
            kernels.append((int(parts[1]), parts[2] == "1", int(parts[3])))
        elif key == "level" and len(parts) == 2:
            level = parts[1]
        elif key == "planned" and len(parts) == 2:
            planned = int(parts[1])
        elif key == "created" and len(parts) == 2:
            created = int(parts[1])
        elif key == "total_ns" and len(parts) == 2:
            total_ns = int(parts[1])
    return WarmupSummary(level, planned, created, total_ns, kernels)


class StartupReport:
    """A startup measurement, formatted.

    Holds one `PhaseTiming` per phase in `PHASES`, a `WarmupSummary`, an
    environment snapshot, and any notes the caller wants carried along.
    Phases the source did not mention are present and unobserved, so a
    consumer can iterate `PHASES` and always find an entry.

    `warmup` defaults to the off summary rather than to None, for the same
    reason: a report that front-loaded nothing has an answer, and it is
    "nothing", not "unknown".
    """

    def __init__(self, timings, environment=None, notes=None, warmup=None):
        self.timings = {}
        for phase in PHASES:
            supplied = timings.get(phase.name)
            if supplied is None:
                supplied = PhaseTiming(phase.name)
            self.timings[phase.name] = supplied
        self.environment = environment or {}
        self.notes = list(notes or ())
        self.warmup = WarmupSummary() if warmup is None else warmup

    # -- derived numbers --------------------------------------------------

    def phase(self, name):
        return self.timings[name]

    @property
    def cold_ns(self):
        """What a fresh process pays end to end: the first occurrence of
        every one-time phase, summed. `warm_fit` is excluded."""
        return sum(
            self.timings[p.name].first_ns for p in PHASES if p.one_time
        )

    @property
    def warm_ns(self):
        """Mean recorded `warm_fit`, or None when none was recorded."""
        return self.timings["warm_fit"].mean_ns

    @property
    def overhead_ns(self):
        """`first_fit - warm_fit`, clamped at zero, or None when either
        side is missing. The part of the first fit that a warm fit does
        not repeat, and therefore the part worth attacking."""
        warm = self.warm_ns
        first = self.timings["first_fit"]
        if warm is None or not first.observed:
            return None
        return max(0, first.first_ns - warm)

    @property
    def has_timings(self):
        """False for a report whose phases were only counted. Such a
        report is a record of what happened, not a measurement, and
        `explanation` says so rather than printing zeros."""
        return any(t.total_ns > 0 for t in self.timings.values())

    def dominant_phase(self):
        """The one-time phase with the largest first occurrence, or None
        when nothing was timed. The first question a startup report is
        opened to answer."""
        best = None
        for p in PHASES:
            if not p.one_time:
                continue
            timing = self.timings[p.name]
            if not timing.observed or timing.first_ns <= 0:
                continue
            if best is None or timing.first_ns > best.first_ns:
                best = timing
        return best

    # -- rendering --------------------------------------------------------

    def table(self):
        """The phases as a fixed-width table, contract order preserved.

        Order is the contract's, not sorted by duration: the point of the
        table is that a reader can see where in the sequence the time went,
        and re-sorting destroys that.
        """
        header = "%-18s %-9s %6s %12s %12s" % (
            "phase",
            "origin",
            "calls",
            "first",
            "total",
        )
        lines = [header, "-" * len(header)]
        for p in PHASES:
            t = self.timings[p.name]
            if not t.observed:
                lines.append(
                    "%-18s %-9s %6s %12s %12s"
                    % (p.name, t.origin, "-", "not run", "not run")
                )
                continue
            lines.append(
                "%-18s %-9s %6d %12s %12s"
                % (
                    p.name,
                    t.origin,
                    t.calls,
                    format_duration(t.first_ns),
                    format_duration(t.total_ns),
                )
            )
        return "\n".join(lines)

    @property
    def explanation(self):
        """The table, the headline numbers, and the caveats, as prose."""
        parts = [self.table(), ""]
        if not self.has_timings:
            parts.append(
                "No durations were recorded. The phases above were counted"
                " but not timed, which is what a run without"
                " MOJOTREES_STARTUP_TRACE=1 produces. This is not a"
                " measurement."
            )
        else:
            parts.append("cold start:      %s" % format_duration(self.cold_ns))
            parts.append("warm fit:        %s" % format_duration(self.warm_ns))
            parts.append(
                "first-fit extra: %s" % format_duration(self.overhead_ns)
            )
            dominant = self.dominant_phase()
            if dominant is not None:
                parts.append(
                    "largest one-time phase: %s (%s)"
                    % (dominant.name, format_duration(dominant.first_ns))
                )
        supplied = [
            t.name
            for t in self.timings.values()
            if t.origin == "supplied" and t.observed
        ]
        if supplied:
            parts.append(
                "supplied by the harness, not timed by this process: %s"
                % ", ".join(sorted(supplied))
            )
        parts.append("kernel warm-up: %s" % self.warmup.describe())
        missing = [p.name for p in PHASES if not self.timings[p.name].observed]
        if missing:
            parts.append(
                "not run in this process: %s" % ", ".join(missing)
            )
        env = self.environment.get("env") if self.environment else None
        if env:
            parts.append(
                "environment: %s"
                % ", ".join("%s=%s" % kv for kv in sorted(env.items()))
            )
        install = (
            self.environment.get("install") if self.environment else None
        )
        if install:
            parts.append("install kind: %s" % install.get("kind"))
            if install.get("missing_runtime_libs"):
                parts.append(
                    "bundled runtime libraries missing: %s"
                    % ", ".join(install["missing_runtime_libs"])
                )
        parts.append(
            "device availability and backend are not decided here; see"
            " mojotrees.device_selection."
        )
        for note in self.notes:
            parts.append(note)
        return "\n".join(parts)

    def to_dict(self):
        """JSON-serializable, which is what a CI log or a support ticket
        wants. Durations stay integer nanoseconds; the formatting in
        `explanation` is for eyes only."""
        return {
            "phases": [self.timings[p.name].to_dict() for p in PHASES],
            "cold_ns": self.cold_ns,
            "warm_ns": self.warm_ns,
            "overhead_ns": self.overhead_ns,
            "has_timings": self.has_timings,
            "warmup": self.warmup.to_dict(),
            "environment": self.environment,
            "notes": list(self.notes),
        }

    def __str__(self):
        return self.explanation

    def __repr__(self):
        return "<StartupReport cold=%s warm=%s>" % (
            format_duration(self.cold_ns),
            format_duration(self.warm_ns),
        )


def report_from_trace(text, environment=None, notes=None):
    """A report from `StartupTrace.report()` output.

    `text` may also carry the `warmup.` lines `WarmupPlan.report()` emits;
    they are picked up when present and ignored when not, so the two reports
    can be concatenated by whatever emits them and pasted in together.

    `environment` defaults to `environment_snapshot()`. Pass one captured
    in the measured process instead when the trace came from elsewhere:
    the snapshot describes whichever process calls it, and describing the
    wrong one is worse than describing none.
    """
    env = environment_snapshot() if environment is None else environment
    return StartupReport(parse_trace(text), env, notes, parse_warmup(text))


def report_from_values(values, environment=None, notes=None, warmup=None):
    """A report from phase durations measured by a harness.

    `values` maps phase names to either an integer nanosecond duration or
    a mapping with any of `calls`, `total_ns`, `first_ns`, `origin`, and
    `observed`. A bare integer is read as one observed occurrence, which
    is what a harness that timed a phase once has.

    This is the entry point for the three phases nothing native can see:
    a cold-start harness times the interpreter, the extension load, and
    the runtime initialization, and hands the numbers in here.

    Raises `ValueError` for an unknown phase name rather than dropping it,
    for the same reason `parse_trace` does.
    """
    timings = {}
    for name, value in values.items():
        if name not in _BY_NAME:
            raise ValueError("unknown startup phase %r" % name)
        if isinstance(value, dict):
            first = value.get("first_ns", value.get("total_ns", 0))
            timings[name] = PhaseTiming(
                name,
                calls=value.get("calls", 1),
                total_ns=value.get("total_ns", first),
                first_ns=first,
                origin=value.get("origin"),
                observed=value.get("observed"),
            )
        else:
            nanos = int(value)
            timings[name] = PhaseTiming(
                name, calls=1, total_ns=nanos, first_ns=nanos, observed=True
            )
    env = environment_snapshot() if environment is None else environment
    return StartupReport(timings, env, notes, warmup)
