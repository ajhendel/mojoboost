"""A regime canary: a fixed, tiny arm that reads the machine, not the code.

This machine drifts. `bench/results/session3_2026-08-16/RESULTS.md` measured
the device-resident tree plane at 24 percent in one window and 8 percent in
another, both resolved and both correct, and measured the histogram row
unroll as *indistinguishable* (8.1 percent against a 14.1 percent floor) in
one window and *resolved* (10.8 against 2.1) in another, from the same
command and the same code. So the regime is not noise to average out. It is
part of every result this repository records, and until now it has been
inferred **by hand, from effect** -- somebody noticing both arms rise
together -- which is an inference that was made and retracted the same night.

There has never been an instrument. `bench/apple/thermal_capture.sh` prints a
plan and refuses `--execute` with exit code 3, so the protocol step that told
every session to capture thermal state with it was never followable and no
session ever followed it (`bench/results/PROFILE_PROTOCOL.md`, A1).
`pmset -g therm` returns nothing useful on Apple silicon. The A1 replacement,
`uptime` plus the top processes, is real and worth keeping, but Session III
recorded the trap that makes it insufficient on its own: **the slow regime
showed a quiet box and slow results at the same time.** Load average said
nothing was wrong while a comparator moved by a factor of 2.2.

This module is the missing instrument, and it is deliberately the dumbest one
that could work: run a fixed amount of arithmetic, see how long it takes, and
divide by how long that same arithmetic took in a window somebody certified as
quiet. If the ratio is near 1.0 the machine is delivering what it delivered
then. If it is 1.5 the machine is not, and every arm timed in that session is
a number taken on a slower computer than the one the baseline was taken on.

Four properties are what make it an instrument rather than another arm.

**It measures the machine, not this repository.** Both probes are
self-contained loops written in this file. Neither calls into
`src/mojotrees/` and neither touches a dataset, a thread count, an
environment variable, a bin count, or anything else a session varies. That is
the whole point: if the probe went through the training code, then a kernel
optimization landing in another lane would move the canary, and a code change
would be indistinguishable from a regime change -- which is precisely the
confusion the canary exists to end. The cost of a probe here can only change
if the machine changes or if the constants below change, and the constants
carry `CANARY_PROBE_VERSION` so that changing them invalidates the baseline
loudly rather than silently comparing two different amounts of work.

**Two probes, reported separately, never averaged.** Session III measured ten
CPU cores degrading by a factor of 2.2 across a regime change while the GPU
arm degraded by 1.5 in the same window. A single-engine canary would have
mislabelled that window whichever engine it picked, and an average of the two
would have reported 1.85, a number describing neither engine. The two engines
throttle differently and that difference is itself a finding, so it is
reported as two numbers.

**It runs first and last.** A regime shift *during* a session is exactly what
bit this project: a six-pair sequence straddled one, and pairs 5 and 6 were a
different machine from pairs 1 through 4. A canary taken only at the start
would have certified that session. Both readings are reported, and when they
disagree by more than `CANARY_DRIFT_THRESHOLD_PCT` the output says so in
capitals, because the consequence is not that the numbers are noisy: it is
that **the arms in between are not comparable to each other**, which is the
one assumption the interleaved harness is built on.

**It reports a ratio and the raw milliseconds.** The ratio is the answer, but
a ratio is a quotient of two numbers, one of which is a measurement somebody
might later correct. So the raw figures go into the record beside it and a
future reader can re-derive without re-running.

## The baselines are not in this file and were not measured here

`bench/canary_baseline.json` ships with its values **absent**, and everything
below handles that case by reporting raw milliseconds and saying the ratio is
unavailable. That is not an oversight to be tidied up by somebody filling in a
plausible number. A fabricated baseline in a regime detector does not fail
loudly; it silently mislabels every future session, and it mislabels them in a
way that looks like data. Having no detector is better. The command that
establishes them is in `bench/bench_canary.mojo` and it must be run on a box
that is verifiably quiet, by somebody who can say so in the record.
"""

from std.gpu import global_idx
from std.os import getenv
from std.sys import has_accelerator
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext


# Bumped whenever any probe constant below changes, or either probe's inner
# loop is rewritten. A baseline records the version it was measured under, and
# a mismatch refuses to produce a ratio. Without this, retuning the iteration
# count to hit a nicer wall time would leave a stale baseline quietly dividing
# two different amounts of work and reporting the quotient as a regime.
comptime CANARY_PROBE_VERSION = 1

# --- CPU probe constants -------------------------------------------------
#
# A dependent chain of splitmix64 mixing steps, one thread, fixed trip count.
#
# Shape, and why this one. The chain is *serial*: each round's input is the
# previous round's output, so the loop cannot be vectorized, cannot be
# reordered, and cannot overlap with itself. Its wall time is therefore
# (rounds x latency-in-cycles) / core-frequency and essentially nothing else.
# That makes it a clock probe and only a clock probe: it does not touch memory
# beyond a register, so no cache state, no DRAM clock, and no other process's
# working set can move it, and it uses one thread, so core availability cannot
# move it either.
#
# That isolation is the argument for single-threaded and it is also the
# argument against. A multi-threaded probe over the standard worker pool would
# be closer to what a real arm experiences -- a real arm is memory-bound and
# runs on ten cores -- but it would fold clock, core availability, and whatever
# else is running on the box into one number, and the canary would then be
# unable to say which of them moved. Session III's finding is the case for
# isolating: the interesting fact was that CPU and GPU moved by *different*
# factors, and you only get facts of that shape from probes that each measure
# one thing. The blind spot is stated rather than argued away: a window in
# which the memory system throttles but core clock holds would read as fast
# here and would not be. The GPU probe does not cover that gap, because it is
# a different memory system. If such a window is ever identified, the fix is a
# third probe with its own baseline, not a redefinition of this one.
#
# The round count is **estimated**, not measured, and nothing depends on it
# being right. Roughly 13 dependent cycles per round (add, then two
# shift-xor-multiply groups at about three cycles per 64-bit multiply, then a
# final shift-xor) at a nominal 4.4 GHz gives about 3 ns per round, so 2^26
# rounds is about 200 ms. If the real figure lands anywhere from 100 to 400 ms
# the probe is doing its job, because the ratio does not care what the
# denominator is as long as it is the same denominator every time. Change this
# constant only with a reason, and bump CANARY_PROBE_VERSION when you do.
comptime CPU_PROBE_ROUNDS = 67_108_864

# Fixed, so the checksum is reproducible: the same probe on the same build
# must print the same `cpu_checksum` forever, and a differing checksum means
# the probe is not doing the work the baseline was taken over. The digits of
# pi, which is to say: arbitrary, and chosen to be visibly arbitrary.
#
# A fixed compile-time seed is in theory foldable -- a sufficiently determined
# optimizer could evaluate the chain at compile time and leave an empty loop.
# In practice no production optimizer will interpret 67 million iterations of
# a multiply chain; the loop-evaluation limits are three orders of magnitude
# below that. The guard is cheap and is implemented rather than assumed:
# `take_reading` warns when the CPU probe reads implausibly fast, which is
# what folding would look like. The alternative -- seeding from a runtime
# value to defeat folding by construction -- was rejected because it destroys
# the reproducible checksum, and the checksum is the only evidence a reader
# has that two runs did the same work.
comptime CPU_PROBE_SEED = 0x243F6A8885A308D3

# --- GPU probe constants -------------------------------------------------
#
# The same idea on the other engine: a fixed grid of threads, each running a
# serial 32-bit mixing chain of fixed length, launched a fixed number of times
# with one synchronization at the end.
#
# 32-bit rather than 64-bit because 64-bit integer multiply is emulated on
# Apple GPUs and its cost is a property of the shader compiler's expansion
# rather than of the hardware clock, which is exactly the kind of thing that
# should not be inside an instrument.
#
# The grid is sized to **saturate**, unlike the CPU probe which is sized to
# isolate. The asymmetry is deliberate. On the CPU, "how many cores am I
# getting" is contaminated by every other process on the box, so folding it in
# would make the probe measure the machine's visitors rather than the machine.
# The GPU is not shared that way during a benchmark session, and Apple silicon
# GPU throttling presents as a clock reduction across the whole device, so a
# saturating grid measures what an arm would actually receive. A quarter of a
# million threads is far more than an M4's ten GPU cores need; a machine with
# a larger GPU will read differently, which is fine, because a baseline is per
# machine and the machine is recorded in the baseline file.
comptime GPU_PROBE_GROUPS = 1024
comptime GPU_PROBE_BLOCK = 256
comptime GPU_PROBE_THREADS = GPU_PROBE_GROUPS * GPU_PROBE_BLOCK
comptime GPU_PROBE_ITERS = 32768

# Few launches, one synchronization, and no copy back. Each of those three is
# a decision:
#
# Few launches, because `enqueue_function` is a *host* cost. Every launch
# inside the timed region imports a slice of the CPU's regime into the GPU
# number, which is the one contamination this probe cannot tolerate given that
# its whole purpose is to be compared against the CPU probe. Four launches at
# the order of 150 us apiece (**measured** shape, from `bench_launch_cost.mojo`
# on this machine, though not re-measured here) is a **derived bound** of about
# 0.6 ms of host time inside a probe estimated at tens of milliseconds. More
# than one launch, rather than a single very long kernel, only because a
# multi-second kernel is a driver-watchdog risk and buys nothing.
#
# One synchronization, at the end, for the same reason: a synchronize per
# launch would multiply the host round-trip cost by four.
#
# No copy back at all. On Metal every `enqueue_copy` is a synchronous drain of
# the whole queue, so a probe that reads its results back is timing a kernel
# plus a drain plus a transfer, and the transfer is a memory-system
# measurement wearing a compute label. The kernel writes to the device buffer
# and the buffer is never read. Nothing downstream needs the values; the
# writes exist so the arithmetic is not dead code.
comptime GPU_PROBE_LAUNCHES = 4

# Whether the device open is inside the number. It is **not**, and this is the
# decision most worth arguing, because both answers are defensible and they
# answer different questions.
#
# "Device open plus kernel" answers *what does a session pay to touch the GPU
# at all*, which is a real quantity and is part of every `train_gpu` call's
# first repeat. "Kernel alone" answers *what is this GPU delivering right
# now*, which is the regime question.
#
# The regime question wins here on a structural argument, not a philosophical
# one. The canary runs first and last in a session and compares the two, and
# by the time the last one runs the driver has been initialized, contexts have
# been created and destroyed, and shader pipelines are cached. A probe that
# included device open would therefore read systematically higher at the start
# than at the end **for a reason that has nothing to do with the machine's
# regime**, and requirement 4 -- shout when start and end disagree -- would
# fire on every single run and be ignored within a week. An alarm that always
# fires is not an alarm.
#
# The open cost is still worth having, so it is measured and reported beside
# the probe as `gpu_open_ms`, excluded from the ratio, and labelled. Read the
# start-of-session one as the cold figure and the end-of-session one as the
# warm figure; the gap between them is roughly what the first GPU arm of a run
# pays and no other arm does.

# How far the start and end readings may differ before the session is called
# unstable, as a percentage of the smaller reading. **Chosen, not measured.**
# Five percent because the smallest effect this repository has ever resolved
# was 10.8 percent against a 2.1 percent noise floor: a machine that moved
# five percent between the first arm and the last is already moving by half
# the size of the effects being hunted, and a pair straddling that is not a
# pair. It is not derived from any distribution of canary readings, because no
# canary reading has ever been taken. Revisit it once a few sessions of
# start/end pairs exist, and say so when you do.
comptime CANARY_DRIFT_THRESHOLD_PCT = 5.0

# Where the baseline lives, relative to the bench directory. Resolved through
# MOJOTREES_BENCH_DIR exactly as the LightGBM arm resolves its module, because
# the pixi tasks run from the repository root and a hand invocation might not.
comptime CANARY_BASELINE_FILE = "canary_baseline.json"


struct CanaryReading(Copyable, Movable):
    """One start-of-session or end-of-session reading, in milliseconds.

    `gpu_ms` is -1.0 when there is no accelerator in this build or the device
    could not be opened, which is a legitimate state and not an error: a
    CPU-only build still gets a CPU regime reading, and half a canary is worth
    more than none. `gpu_open_ms` is -1.0 under the same conditions.
    """

    var cpu_ms: Float64
    var cpu_checksum: UInt64
    var gpu_ms: Float64
    var gpu_open_ms: Float64
    var gpu_note: String

    def __init__(
        out self,
        cpu_ms: Float64,
        cpu_checksum: UInt64,
        gpu_ms: Float64,
        gpu_open_ms: Float64,
        gpu_note: String,
    ):
        self.cpu_ms = cpu_ms
        self.cpu_checksum = cpu_checksum
        self.gpu_ms = gpu_ms
        self.gpu_open_ms = gpu_open_ms
        self.gpu_note = gpu_note.copy()

    def has_gpu(self) -> Bool:
        return self.gpu_ms >= 0.0


struct CanaryBaseline(Copyable, Movable):
    """What `bench/canary_baseline.json` says, and whether it says anything.

    `status` is the single word a reader needs and every consumer branches on
    it rather than re-deriving the condition:

      `ok`                      both values present and the version matches
      `partial`                 one engine recorded, the other not
      `none-recorded`           the file is there, the values are absent
      `probe-version-mismatch`  measured under a different probe; unusable
      `file-missing`            no baseline file at the resolved path

    Only `ok` and `partial` produce ratios, and `partial` produces one.
    """

    var status: String
    var path: String
    var probe_version: Int
    var machine: String
    var measured_on: String
    var cpu_ms: Float64
    var gpu_ms: Float64

    def __init__(
        out self,
        status: String,
        path: String,
        probe_version: Int,
        machine: String,
        measured_on: String,
        cpu_ms: Float64,
        gpu_ms: Float64,
    ):
        self.status = status.copy()
        self.path = path.copy()
        self.probe_version = probe_version
        self.machine = machine.copy()
        self.measured_on = measured_on.copy()
        self.cpu_ms = cpu_ms
        self.gpu_ms = gpu_ms

    def has_cpu(self) -> Bool:
        return self.cpu_ms > 0.0

    def has_gpu(self) -> Bool:
        return self.gpu_ms > 0.0


struct _JsonNumber(Copyable, Movable):
    """A parsed number, or the fact that there wasn't one. See `_number_at`."""

    var present: Bool
    var value: Float64

    def __init__(out self, present: Bool, value: Float64):
        self.present = present
        self.value = value


def _mix64(x: UInt64) -> UInt64:
    """One splitmix64 finalizer round. The CPU probe's unit of work.

    Copied into this file rather than imported, deliberately. Everything in
    `src/mojotrees/` is subject to change by another lane, and a probe whose
    cost can be moved by somebody else's optimization is not a probe. The
    duplication is the feature.
    """
    var z = x + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _canary_gpu_kernel(
    out_buf: MutPointer[UInt32, MutAnyOrigin], n_threads: Int32, iters: Int32
):
    """A serial 32-bit mixing chain per thread, then one store.

    The chain is seeded from the thread index, which is a runtime value, so
    the loop cannot be constant-folded however aggressive the shader compiler
    is, and the store keeps it from being eliminated as dead. `iters` is a
    runtime argument so the same kernel can be used for the cheap warm-up
    launches and the timed ones; a separate warm-up kernel would warm a
    different pipeline than the one being timed, which is the mistake
    `bench_launch_cost.mojo` documents having made once already.
    """
    var i = global_idx.x
    if i < Int(n_threads):
        var x = UInt32(i) * 2654435761 + 12345
        for _ in range(Int(iters)):
            x = x * 1664525 + 1013904223
            x = x ^ (x >> 15)
        out_buf[unsafe_offset=i] = x


struct CpuProbe(Copyable, Movable):
    """The CPU probe's two outputs: what it cost and what it computed.

    Both come out of one loop. Recomputing the checksum in a second pass would
    double the probe's cost, and a checksum from a different pass than the
    timed one would not be evidence about the timed one.
    """

    var ms: Float64
    var checksum: UInt64

    def __init__(out self, ms: Float64, checksum: UInt64):
        self.ms = ms
        self.checksum = checksum


def cpu_probe() -> CpuProbe:
    """Run the CPU probe once. Fixed work, one thread, no allocation.

    Timed with `perf_counter_ns`, the same clock every arm in `bench/` is
    timed by, so no number here carries a different clock's offset.
    """
    var z = UInt64(CPU_PROBE_SEED)
    var t0 = perf_counter_ns()
    for _ in range(CPU_PROBE_ROUNDS):
        z = _mix64(z)
    var t1 = perf_counter_ns()
    return CpuProbe(Float64(t1 - t0) / 1e6, z)


struct GpuProbe(Copyable, Movable):
    """The GPU probe's outputs, including the excluded device-open cost.

    `ms` is -1.0 when the probe could not run, and `note` says why in one
    phrase suitable for printing beside it.
    """

    var ms: Float64
    var open_ms: Float64
    var note: String

    def __init__(out self, ms: Float64, open_ms: Float64, note: String):
        self.ms = ms
        self.open_ms = open_ms
        self.note = note.copy()


def gpu_probe() -> GpuProbe:
    """Run the GPU probe once, or report why it could not run.

    The whole accelerator-touching body sits inside the `else` of a
    `comptime if not has_accelerator()`. That is not a style preference: a
    CPU-only build that elaborates any of this dies at compile time on
    "Unknown GPU architecture", and a machine with an accelerator never
    reproduces it, so the failure lands on CI and on other people's laptops.

    Degrading rather than raising is the other half. A canary that aborts a
    benchmark session because the box has no GPU has made the CPU numbers
    unobtainable in order to protect the GPU numbers that were never going to
    exist. Unavailable is a state, and it is reported as one.
    """
    comptime if not has_accelerator():
        return GpuProbe(
            -1.0, -1.0, String("no accelerator in this build")
        )
    else:
        # Every device call in here can raise, and none of them may take a
        # benchmark session down with it. A box whose GPU is busy, absent, or
        # held by another process still has a CPU regime worth reading, and
        # the caller is written to report half a canary.
        try:
            var open_t0 = perf_counter_ns()
            var ctx = DeviceContext()
            var buf = ctx.enqueue_create_buffer[DType.uint32](
                GPU_PROBE_THREADS
            )
            ctx.synchronize()
            var open_t1 = perf_counter_ns()
            var open_ms = Float64(open_t1 - open_t0) / 1e6

            # Warm the pipeline this probe will time, with the same kernel at
            # a trivial iteration count. The first launch of a kernel pays its
            # one-time setup and it must not land inside the measurement.
            for _ in range(2):
                ctx.enqueue_function[_canary_gpu_kernel](
                    buf.unsafe_ptr(),
                    Int32(GPU_PROBE_THREADS),
                    Int32(64),
                    grid_dim=GPU_PROBE_GROUPS,
                    block_dim=GPU_PROBE_BLOCK,
                )
            ctx.synchronize()

            var t0 = perf_counter_ns()
            for _ in range(GPU_PROBE_LAUNCHES):
                ctx.enqueue_function[_canary_gpu_kernel](
                    buf.unsafe_ptr(),
                    Int32(GPU_PROBE_THREADS),
                    Int32(GPU_PROBE_ITERS),
                    grid_dim=GPU_PROBE_GROUPS,
                    block_dim=GPU_PROBE_BLOCK,
                )
            # One synchronization, no copy back. See GPU_PROBE_LAUNCHES.
            ctx.synchronize()
            var t1 = perf_counter_ns()
            return GpuProbe(
                Float64(t1 - t0) / 1e6, open_ms, String("ok")
            )
        except e:
            return GpuProbe(
                -1.0, -1.0, String("device unavailable: ", String(e))
            )


def take_reading() -> CanaryReading:
    """Both probes, once each, CPU first.

    CPU first and GPU second on every reading, so the two readings of a
    session are the same sequence and any ordering effect cancels between
    them rather than being attributed to the machine.
    """
    var c = cpu_probe()
    var g = gpu_probe()
    return CanaryReading(c.ms, c.checksum, g.ms, g.open_ms, g.note)


def enabled() -> Bool:
    """Whether the canary runs. On unless MOJOTREES_CANARY is 0, off, or false.

    An escape hatch exists for one reason: results recorded before the canary
    was wired in were taken by a process that did not open a device before its
    first arm, and reproducing one of those exactly requires being able to
    turn this off. Every consumer prints the resolved state either way, so the
    off case is a declared condition in the record rather than a silent one.
    """
    var s = getenv("MOJOTREES_CANARY")
    if s == "0" or s == "off" or s == "false" or s == "no":
        return False
    return True


def _bench_dir() -> String:
    var d = getenv("MOJOTREES_BENCH_DIR")
    if d.byte_length() == 0:
        return String("bench")
    return d^


def baseline_path() -> String:
    return String(_bench_dir(), "/", CANARY_BASELINE_FILE)


def _skip_space(text: String, start: Int) -> Int:
    var i = start
    var n = text.byte_length()
    while i < n:
        var c = String(text[byte=i : i + 1])
        if c == " " or c == "\t" or c == "\n" or c == "\r":
            i += 1
        else:
            break
    return i


def _value_token(text: String, key: String) -> String:
    """The raw token following `"key":`, or the empty string if absent.

    This is not a JSON parser and is not trying to be. The baseline file is
    written by `bench/bench_canary.mojo` and edited by hand, it is a flat
    object of four scalars and some prose, and it is read once per session.
    A real parser would be a hundred lines of surface area in a benchmark
    harness for no gain. What this does instead is find the key, step past the
    colon, and take everything up to the next delimiter -- which means a
    malformed file yields a token that fails to parse and is reported as
    absent, rather than yielding a wrong number. Absent is safe here: absent
    means "no ratio", and no ratio is the state this whole mechanism is
    designed to survive.
    """
    var needle = String("\"", key, "\"")
    var n = text.byte_length()
    var from_ = 0
    while from_ < n:
        var rest = String(text[byte=from_:n])
        var rel = rest.find(needle)
        if rel < 0:
            return String("")
        var at = from_ + rel
        var i = _skip_space(text, at + needle.byte_length())
        if i >= n or String(text[byte=i : i + 1]) != ":":
            # The same word inside a prose value. Keep looking, so that a note
            # field mentioning a key by name cannot shadow the key itself.
            from_ = at + needle.byte_length()
            continue
        i = _skip_space(text, i + 1)
        if i >= n:
            return String("")
        if String(text[byte=i : i + 1]) == "\"":
            var s = i + 1
            var j = s
            while j < n and String(text[byte=j : j + 1]) != "\"":
                j += 1
            return String(text[byte=s:j])
        var start = i
        while i < n:
            var c = String(text[byte=i : i + 1])
            if (
                c == ","
                or c == "}"
                or c == "]"
                or c == " "
                or c == "\n"
                or c == "\r"
                or c == "\t"
            ):
                break
            i += 1
        var token = String(text[byte=start:i])
        # A JSON null is the absence of a value, and every caller here treats
        # absence as "no baseline", so it is normalized to the empty string
        # rather than handed back as the four-letter word `null`.
        if token == "null":
            return String("")
        return token^
    return String("")


def _number_at(text: String, key: String) -> _JsonNumber:
    """`_value_token` parsed as a number. `null`, missing, or junk is absent."""
    var token = _value_token(text, key)
    if token.byte_length() == 0 or token == "null":
        return _JsonNumber(False, -1.0)
    try:
        return _JsonNumber(True, Float64(token))
    except:
        return _JsonNumber(False, -1.0)


def load_baseline() -> CanaryBaseline:
    """Read `bench/canary_baseline.json`, tolerating every way it can be empty.

    Four of the five outcomes are non-fatal by design and none of them raises.
    A benchmark session must not abort because a calibration file is missing;
    it must print raw milliseconds and say the ratio is unavailable. The one
    outcome that is loud is `probe-version-mismatch`, because that is the case
    where a plausible-looking number exists and dividing by it would be wrong.
    """
    var path = baseline_path()
    var text: String
    try:
        text = open(path, "r").read()
    except:
        return CanaryBaseline(
            String("file-missing"),
            path,
            -1,
            String(""),
            String(""),
            -1.0,
            -1.0,
        )

    var machine = _value_token(text, "machine")
    var measured_on = _value_token(text, "measured_on")
    var ver = _number_at(text, "probe_version")
    var probe_version = Int(ver.value) if ver.present else -1
    var cpu = _number_at(text, "cpu_ms")
    var gpu = _number_at(text, "gpu_ms")

    if ver.present and probe_version != CANARY_PROBE_VERSION:
        return CanaryBaseline(
            String("probe-version-mismatch"),
            path,
            probe_version,
            machine,
            measured_on,
            -1.0,
            -1.0,
        )

    var cpu_ms = cpu.value if cpu.present else -1.0
    var gpu_ms = gpu.value if gpu.present else -1.0
    var status = String("none-recorded")
    if cpu.present and gpu.present:
        status = String("ok")
    elif cpu.present or gpu.present:
        status = String("partial")
    return CanaryBaseline(
        status, path, probe_version, machine, measured_on, cpu_ms, gpu_ms
    )


def _ratio(reading_ms: Float64, baseline_ms: Float64) -> Float64:
    """The regime ratio, or -1.0 when either side is absent.

    Reading over baseline, so **larger is slower** and 1.0 is the window the
    baseline was taken in. That orientation rather than its inverse because
    the number a reader wants is "how much longer is everything taking", which
    is the factor they would multiply a duration by.
    """
    if reading_ms < 0.0 or baseline_ms <= 0.0:
        return -1.0
    return reading_ms / baseline_ms


def _fmt_ratio(r: Float64) -> String:
    """A ratio for a human, rounded to three decimals, or `unavailable`.

    Rounded on the printed line and **not** in the JSON record, which carries
    the full quotient. Three decimals is already finer than the quantity it
    describes -- a regime is a factor, and the difference between 1.070 and
    1.0701754 is not a thing anyone will act on -- and a headline number that
    a reader has to visually truncate before comparing to the next one is a
    headline number that gets misread. The raw milliseconds on the lines above
    are what a re-derivation uses.
    """
    if r < 0.0:
        return String("unavailable")
    var scaled = r * 1000.0
    return String(Float64(Int(scaled + 0.5)) / 1000.0)


def _fmt_pct(p: Float64) -> String:
    """A drift percentage, or `unavailable` when one side could not be read."""
    if p < 0.0:
        return String("unavailable")
    return String(p)


def _pct_change(a: Float64, b: Float64) -> Float64:
    """|b - a| as a percentage of the smaller, or -1.0 if either is absent."""
    if a <= 0.0 or b <= 0.0:
        return -1.0
    var lo = a if a < b else b
    var hi = a if a > b else b
    var frac = (hi - lo) / lo * 100.0
    var scaled = frac * 10.0
    return Float64(Int(scaled + 0.5)) / 10.0


def print_baseline(b: CanaryBaseline):
    """One block saying what the ratios below are divided by, before any are.

    Printed even when there is nothing -- especially when there is nothing.
    The failure this repository keeps having is a number quoted without the
    conditions it was taken under, and a ratio against an unstated baseline is
    that failure in its purest form.
    """
    print("canary_probe_version:", CANARY_PROBE_VERSION)
    print("canary_baseline_path:", b.path)
    print("canary_baseline_status:", b.status)
    if b.status == "file-missing":
        print(
            "canary_baseline_note: no baseline file at that path; readings"
            " below are raw milliseconds and no ratio is computed"
        )
        return
    if b.status == "probe-version-mismatch":
        print(
            "canary_baseline_note: BASELINE UNUSABLE. It was measured under"
            " probe version",
            b.probe_version,
            "and this build is probe version",
            CANARY_PROBE_VERSION,
            "so the two are not the same amount of work. No ratio is"
            " computed. Re-measure with bench/bench_canary.mojo.",
        )
        return
    print(
        "canary_baseline_machine:",
        b.machine if b.machine.byte_length() > 0 else String("unrecorded"),
    )
    print(
        "canary_baseline_measured_on:",
        b.measured_on if b.measured_on.byte_length()
        > 0 else String("unrecorded"),
    )
    if b.has_cpu():
        print("canary_baseline_cpu_ms:", b.cpu_ms)
    else:
        print("canary_baseline_cpu_ms: none-recorded")
    if b.has_gpu():
        print("canary_baseline_gpu_ms:", b.gpu_ms)
    else:
        print("canary_baseline_gpu_ms: none-recorded")
    if b.status != "ok":
        print(
            "canary_baseline_note: at least one engine has no recorded"
            " baseline; its reading is reported raw and its ratio is"
            " unavailable. Establish it with bench/bench_canary.mojo in a"
            " quiet window -- do not fill in a number that was not measured."
        )


def print_reading(phase: String, r: CanaryReading, b: CanaryBaseline):
    """One reading, raw and as a ratio, under a `start` or `end` label.

    The raw milliseconds are printed whether or not a ratio exists, so that a
    session run before any baseline was established still produces a record
    somebody can compute ratios from later if the baseline is established
    afterwards. That is worth the two extra lines: the alternative is a
    session's regime being unrecoverable because the calibration happened to
    land a day later.
    """
    print("canary_" + phase + "_cpu_ms:", r.cpu_ms)
    print("canary_" + phase + "_cpu_checksum:", r.cpu_checksum)
    print(
        "canary_" + phase + "_cpu_ratio:", _fmt_ratio(_ratio(r.cpu_ms, b.cpu_ms))
    )
    if r.has_gpu():
        print("canary_" + phase + "_gpu_ms:", r.gpu_ms)
        print(
            "canary_" + phase + "_gpu_ratio:",
            _fmt_ratio(_ratio(r.gpu_ms, b.gpu_ms)),
        )
        # Excluded from the ratio on purpose; see the GPU_PROBE_LAUNCHES block.
        print("canary_" + phase + "_gpu_open_ms_excluded:", r.gpu_open_ms)
    else:
        print("canary_" + phase + "_gpu_ms: unavailable", "--", r.gpu_note)
        print("canary_" + phase + "_gpu_ratio: unavailable")
    if r.cpu_ms >= 0.0 and r.cpu_ms < 1.0:
        # 200 ms is the estimate; 1 ms would mean the chain is not running.
        print(
            "canary_warning: the CPU probe read under a millisecond, which is"
            " two orders of magnitude below its estimate. Suspect the loop was"
            " optimized away and treat this reading as void."
        )


def _worse(a: Float64, b: Float64) -> Float64:
    """The larger of two ratios, ignoring absences. -1.0 if both are absent.

    Larger is slower, so this is the worst regime the session was in.
    """
    if a < 0.0:
        return b
    if b < 0.0:
        return a
    return a if a > b else b


def print_session_verdict(
    start: CanaryReading, end: CanaryReading, b: CanaryBaseline
) -> String:
    """The headline ratios and whether the session held together. Returns the
    regime word that goes into the JSON record.

    Two headline ratios, `canary_cpu_ratio` and `canary_gpu_ratio`, each
    defined as **the worse of the session's two readings**, not the mean and
    not the first. The question a headline ratio answers is "how much should I
    discount what this session measured", and the answer has to be the
    slowest regime any arm in it might have run under. Averaging a fast start
    with a slow end produces a number no arm experienced.

    The drift check is the part that matters more. It compares start against
    end per engine and, past `CANARY_DRIFT_THRESHOLD_PCT`, says in capitals
    that the arms are not comparable to each other -- which is a stronger and
    more actionable statement than "the machine was slow", because a slow but
    *stable* session still yields valid A/Bs at a stretched scale, and a
    session that moved underneath its own arms does not.
    """
    var cpu_r = _worse(
        _ratio(start.cpu_ms, b.cpu_ms), _ratio(end.cpu_ms, b.cpu_ms)
    )
    var gpu_r = _worse(
        _ratio(start.gpu_ms, b.gpu_ms), _ratio(end.gpu_ms, b.gpu_ms)
    )
    print("canary_cpu_ratio:", _fmt_ratio(cpu_r))
    print("canary_gpu_ratio:", _fmt_ratio(gpu_r))
    print(
        "canary_ratio_note: worse (slower) of the start and end readings;"
        " 1.0 is the baseline window, larger is slower. The two engines are"
        " never averaged."
    )

    var cpu_drift = _pct_change(start.cpu_ms, end.cpu_ms)
    var gpu_drift = _pct_change(start.gpu_ms, end.gpu_ms)
    print("canary_drift_cpu_pct:", _fmt_pct(cpu_drift))
    print("canary_drift_gpu_pct:", _fmt_pct(gpu_drift))
    print("canary_drift_threshold_pct:", CANARY_DRIFT_THRESHOLD_PCT)

    var shifted = (
        cpu_drift > CANARY_DRIFT_THRESHOLD_PCT
        or gpu_drift > CANARY_DRIFT_THRESHOLD_PCT
    )
    if shifted:
        print(
            "canary_regime: SHIFTED-DURING-SESSION",
        )
        print(
            "canary_warning: THE MACHINE CHANGED UNDERNEATH THIS RUN. CPU"
            " probe",
            start.cpu_ms,
            "ms at the start and",
            end.cpu_ms,
            "ms at the end; GPU probe",
            start.gpu_ms,
            "ms and",
            end.gpu_ms,
            "ms. The arms between these two readings were not all taken on"
            " the same machine, so the comparisons below are between arms in"
            " different regimes and are not usable as A/Bs. Session III"
            " discarded a six-pair sequence for exactly this. Re-take the"
            " run; do not correct it.",
        )
        return String("shifted")
    print("canary_regime: stable")
    return String("stable")


def _json_num(v: Float64) -> String:
    """A number, or `null` when the sentinel says it is absent."""
    if v < 0.0:
        return String("null")
    return String(v)


def _json_str(s: String) -> String:
    return String("\"", s.replace("\\", "\\\\").replace("\"", "\\\""), "\"")


def _json_str_or_null(s: String) -> String:
    """An empty provenance field is `null`, not `""`.

    The two are different claims. `""` says a machine name was recorded and it
    was blank; `null` says none was recorded. This record's whole job is to
    keep absence distinguishable from a value.
    """
    if s.byte_length() == 0:
        return String("null")
    return _json_str(s)


def _reading_json(r: CanaryReading) -> String:
    return String(
        "{\"cpu_ms\":",
        _json_num(r.cpu_ms),
        ",\"cpu_checksum\":",
        _json_str(String(r.cpu_checksum)),
        ",\"gpu_ms\":",
        _json_num(r.gpu_ms),
        ",\"gpu_open_ms_excluded\":",
        _json_num(r.gpu_open_ms),
        ",\"gpu_note\":",
        _json_str(r.gpu_note),
        "}",
    )


def canary_json(
    start: CanaryReading,
    end: CanaryReading,
    b: CanaryBaseline,
    regime: String,
) -> String:
    """The `canary` object for a harness's `json_summary` record.

    Additive: it is one new key on an existing record and it renames nothing,
    because other sessions and committed results under `bench/results/` read
    the keys that are already there.

    Every absent value is `null` and never zero, following the record's
    existing convention for `noise_floor_pct` at one repeat: a null is the
    absence of a measurement and a zero is a measurement of zero, and this
    file exists because that distinction got lost once already.
    """
    var cpu_r = _worse(
        _ratio(start.cpu_ms, b.cpu_ms), _ratio(end.cpu_ms, b.cpu_ms)
    )
    var gpu_r = _worse(
        _ratio(start.gpu_ms, b.gpu_ms), _ratio(end.gpu_ms, b.gpu_ms)
    )
    return String(
        "{\"probe_version\":",
        CANARY_PROBE_VERSION,
        ",\"enabled\":true",
        ",\"baseline\":{\"path\":",
        _json_str(b.path),
        ",\"status\":",
        _json_str(b.status),
        ",\"machine\":",
        _json_str_or_null(b.machine),
        ",\"measured_on\":",
        _json_str_or_null(b.measured_on),
        ",\"probe_version\":",
        String(b.probe_version) if b.probe_version >= 0 else String("null"),
        ",\"cpu_ms\":",
        _json_num(b.cpu_ms),
        ",\"gpu_ms\":",
        _json_num(b.gpu_ms),
        "},\"start\":",
        _reading_json(start),
        ",\"end\":",
        _reading_json(end),
        ",\"cpu_ratio\":",
        _json_num(cpu_r),
        ",\"gpu_ratio\":",
        _json_num(gpu_r),
        ",\"ratio_definition\":",
        _json_str(
            String(
                "reading_ms / baseline_ms, worse (larger, slower) of start and"
                " end; engines never averaged"
            )
        ),
        ",\"drift_cpu_pct\":",
        _json_num(_pct_change(start.cpu_ms, end.cpu_ms)),
        ",\"drift_gpu_pct\":",
        _json_num(_pct_change(start.gpu_ms, end.gpu_ms)),
        ",\"drift_threshold_pct\":",
        CANARY_DRIFT_THRESHOLD_PCT,
        ",\"regime\":",
        _json_str(regime),
        "}",
    )


def canary_disabled_json() -> String:
    """The `canary` object when MOJOTREES_CANARY turned it off.

    Present rather than omitted, so that "this run has no regime reading" is
    a recorded condition instead of an absence a reader has to notice.
    """
    return String(
        "{\"probe_version\":",
        CANARY_PROBE_VERSION,
        ",\"enabled\":false,\"regime\":\"unmeasured\",\"note\":",
        _json_str(
            String(
                "MOJOTREES_CANARY disabled the regime canary for this run; no"
                " regime reading exists and the arm timings below carry no"
                " window label"
            )
        ),
        "}",
    )
