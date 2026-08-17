"""Establish, or re-read, the regime canary's baselines.

Two jobs, and they are the same command.

**Calibrate.** Run this on a box that is verifiably quiet, in a window
somebody is willing to certify as fast, and paste the block it prints into
`bench/canary_baseline.json`. From then on every `bench_train_gpu.mojo` run
reports its own regime as a ratio against those two numbers instead of
leaving it to be inferred by hand from effect sizes, which is how this
repository has done it until now and which produced an attribution that was
made and retracted the same night.

**Check.** Run it any time to ask what regime the machine is in right now,
without paying for a training run. Once a baseline exists this is a
two-probe, sub-second answer to the question that has been unanswerable.

    pixi run mojo run -I src bench/bench_canary.mojo [repeats]

Defaults to 5 repeats. See `bench/canary.mojo` for what the two probes do and
why they are shaped the way they are; nothing about the measurement is decided
in this file, which only runs it, reduces it, and formats it.

## The reduction is the minimum, and the calibration protocol is the point

Each probe's baseline is the **minimum** across the repeats, matching every
other benchmark in this directory: the minimum is the sample least
contaminated by drift, and the spread beside it is what says whether it can be
trusted at all. For a *baseline* the argument is stronger than usual. The
baseline is supposed to represent the machine at its best, since the ratio it
anchors is meant to read 1.0 in a fast window and above 1.0 otherwise. A
median baseline taken in a mediocre window would put fast windows below 1.0
and read as though the machine had improved.

So the spread is the gate, and this file refuses nothing but says so plainly:
**if the repeats disagree by more than a few percent, the window was not
quiet and the baseline should not be recorded from it.** A baseline is not
something to get out of the way. It is the denominator of every regime
statement this project will make afterwards.

## What this deliberately will not do

It will not write `bench/canary_baseline.json` itself. It prints the file body
with `FILL-IN` markers where the date, the toolchain, the machine, and the
evidence that the box was quiet belong, and a human puts them in. Those
fields are the difference between a measured baseline and a number, and the
person who can attest that nothing else was running is not this program.
"""

from std.sys import argv, has_accelerator

from canary import (
    CANARY_DRIFT_THRESHOLD_PCT,
    CANARY_PROBE_VERSION,
    CPU_PROBE_ROUNDS,
    GPU_PROBE_ITERS,
    GPU_PROBE_LAUNCHES,
    GPU_PROBE_THREADS,
    baseline_path,
    cpu_probe,
    gpu_probe,
    load_baseline,
    print_baseline,
)


def _min_of(values: List[Float64]) -> Float64:
    var m = values[0]
    for i in range(1, len(values)):
        if values[i] < m:
            m = values[i]
    return m


def _max_of(values: List[Float64]) -> Float64:
    var m = values[0]
    for i in range(1, len(values)):
        if values[i] > m:
            m = values[i]
    return m


def _median_of(values: List[Float64]) -> Float64:
    var s = List[Float64](capacity=len(values))
    for i in range(len(values)):
        s.append(values[i])
    for i in range(1, len(s)):
        var v = s[i]
        var j = i - 1
        while j >= 0 and s[j] > v:
            s[j + 1] = s[j]
            j -= 1
        s[j + 1] = v
    var n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def _pct(fraction: Float64) -> Float64:
    var scaled = fraction * 1000.0
    if scaled >= 0.0:
        return Float64(Int(scaled + 0.5)) / 10.0
    return Float64(Int(scaled - 0.5)) / 10.0


def _join(values: List[Float64]) -> String:
    var out = String("")
    for i in range(len(values)):
        if i > 0:
            out += " "
        out += String(values[i])
    return out^


def main() raises:
    var repeats = 5
    var args = argv()
    if len(args) > 1:
        repeats = Int(String(args[1]))
    if repeats < 1:
        raise Error("repeats must be at least 1")

    print("mojotrees regime canary")
    print("canary_probe_version:", CANARY_PROBE_VERSION)
    print("cpu_probe_rounds:", CPU_PROBE_ROUNDS)
    print("gpu_probe_threads:", GPU_PROBE_THREADS)
    print("gpu_probe_iters_per_thread:", GPU_PROBE_ITERS)
    print("gpu_probe_launches:", GPU_PROBE_LAUNCHES)
    print("repeats:", repeats)
    comptime if not has_accelerator():
        print(
            "accelerator: none in this build; the GPU probe reports"
            " unavailable and only a CPU baseline can be established here"
        )
    else:
        print("accelerator: present")

    # What is already recorded, printed before anything is measured, so that a
    # transcript of a re-calibration shows both the old and the new figure and
    # a reader can see how far the machine or the toolchain moved.
    var existing = load_baseline()
    print_baseline(existing)

    # Interleaved, CPU then GPU, repeat by repeat, for the reason
    # `bench_train_gpu.mojo` interleaves its arms: on this machine only
    # adjacent samples are comparable, and running all the CPU repeats and
    # then all the GPU repeats would calibrate the two engines in two
    # different windows and bake that difference into the baselines.
    var cpu_ms = List[Float64](capacity=repeats)
    var gpu_ms = List[Float64](capacity=repeats)
    var gpu_open_ms = List[Float64](capacity=repeats)
    var checksum = UInt64(0)
    var gpu_note = String("")
    var have_gpu = False
    for rep in range(repeats):
        var c = cpu_probe()
        var g = gpu_probe()
        cpu_ms.append(c.ms)
        checksum = c.checksum
        gpu_note = g.note.copy()
        if g.ms >= 0.0:
            have_gpu = True
            gpu_ms.append(g.ms)
            gpu_open_ms.append(g.open_ms)
            print(
                "run",
                rep + 1,
                "cpu_ms:",
                c.ms,
                "gpu_ms:",
                g.ms,
                "gpu_open_ms:",
                g.open_ms,
            )
        else:
            print("run", rep + 1, "cpu_ms:", c.ms, "gpu_ms: unavailable")

    var cpu_lo = _min_of(cpu_ms)
    var cpu_spread = (_max_of(cpu_ms) - cpu_lo) / cpu_lo
    print("cpu_ms_samples:", _join(cpu_ms))
    print("cpu_ms:", cpu_lo)
    print("cpu_ms_median:", _median_of(cpu_ms))
    print("cpu_spread_pct:", _pct(cpu_spread))
    print("cpu_checksum:", checksum)

    var gpu_lo = -1.0
    var gpu_spread = -1.0
    if have_gpu:
        gpu_lo = _min_of(gpu_ms)
        gpu_spread = (_max_of(gpu_ms) - gpu_lo) / gpu_lo
        print("gpu_ms_samples:", _join(gpu_ms))
        print("gpu_ms:", gpu_lo)
        print("gpu_ms_median:", _median_of(gpu_ms))
        print("gpu_spread_pct:", _pct(gpu_spread))
        # Excluded from the baseline on purpose: see the device-open argument
        # in canary.mojo. Recorded because the first repeat's figure is the
        # cold one and the rest are warm, and that gap is what the first GPU
        # arm of a training run pays.
        print("gpu_open_ms_samples_excluded:", _join(gpu_open_ms))
    else:
        print("gpu_ms: unavailable --", gpu_note)

    # Two shape checks on the probes themselves, which are not about the
    # machine at all. `CPU_PROBE_ROUNDS` and the GPU grid were sized from an
    # **estimate** -- a nominal clock and published instruction latencies --
    # and no one has ever run them, so the first person to calibrate is also
    # the first person to find out whether the estimate was any good.
    #
    # The low guard is the one that matters. A serial mixing chain of 2^26
    # rounds cannot finish in a millisecond on any hardware that exists, so a
    # reading that fast means the loop was eliminated and the "baseline" would
    # be a measurement of nothing.
    if cpu_lo < 1.0:
        print(
            "calibration_error: the CPU probe read under a millisecond, which"
            " is impossible for the work it claims to do. The loop was almost"
            " certainly optimized away. Do not record this. Fix the probe"
            " first."
        )
    elif cpu_lo < 50.0 or cpu_lo > 800.0:
        print(
            "calibration_note: the CPU probe is far from its 200 ms estimate."
            " That is not an error -- a ratio does not care what its"
            " denominator is, only that it never changes -- but if the probe"
            " is inconveniently short or long, retune CPU_PROBE_ROUNDS in"
            " bench/canary.mojo, bump CANARY_PROBE_VERSION in the same commit,"
            " and calibrate again. Never retune it after a baseline exists"
            " without bumping the version."
        )
    if have_gpu and gpu_lo >= 0.0 and gpu_lo < 10.0:
        # Four launches at the order of 150 us apiece is a **derived bound**
        # from bench_launch_cost.mojo's shape on this machine, so about 0.6 ms
        # of host enqueue sits inside every GPU reading. Under 10 ms that is
        # more than six percent of the probe, which puts the CPU's regime
        # inside the GPU's number -- the one contamination this probe exists
        # to avoid.
        print(
            "calibration_note: the GPU probe is short enough that host enqueue"
            " cost is a material share of it, which imports CPU regime into"
            " the GPU reading. Raise GPU_PROBE_ITERS in bench/canary.mojo and"
            " bump CANARY_PROBE_VERSION."
        )

    # The one thing this file asserts. A baseline taken across repeats that
    # disagree is a baseline taken in a window that was not the quiet one it
    # claims to be, and it will make every subsequent ratio wrong in a
    # direction nobody can see.
    #
    # RAISED FROM 3 TO 5 PERCENT BY ANDREW, 2026-08-17, AND THE REASON IS THAT
    # 3 PERCENT WAS UNREACHABLE ON THIS MACHINE RATHER THAN THAT 5 IS BETTER.
    # The box is a 10-core M4 laptop that the person reading these numbers also
    # works on, so WindowServer, an editor and an agent session are ambient and
    # waiting does not remove them. Measured the same morning, seven repeats:
    # 15.9 percent CPU with a browser and a chat client open, and 3.2 percent
    # CPU with 3.9 percent GPU once those were quit, against a bar of 3. The
    # second window was the quietest this machine gets while remaining usable,
    # and it still refused itself, so no baseline could ever be recorded and
    # every regime ratio stayed unavailable. A bar nothing can pass is not a
    # strict bar, it is an absent one.
    #
    # WHAT THE WIDER BAR COSTS, STATED SO A LATER READER DOES NOT HAVE TO
    # REDERIVE IT. A 5 percent window cannot resolve a difference smaller than
    # about 5 percent, so any pair that close is `indistinguishable` under M0
    # and must be reported as such rather than ranked. That was checked against
    # the run this change was made for and not assumed: the closest pair in the
    # 2026-08-17 dense decision row is CatBoost at 3.224 s against our GPU at
    # 3.616 s, 12.2 percent apart, so every comparison in that table survives
    # the wider bar. A future table with two arms inside 5 percent of each other
    # gets no verdict from this box at this bar, and the honest response is to
    # say so rather than to narrow the bar back to a number nothing can meet.
    #
    # This value is the CALIBRATION bar, which is a different question from the
    # `drift_threshold_pct` written into the baseline file below. That one asks
    # whether a later window has drifted from the recorded baseline and has been
    # 5.0 since the file existed. The two now agree, which they did not before,
    # and there was never a reason for the entry gate to be stricter than the
    # drift gate it feeds.
    #
    # `PROFILE_PROTOCOL.md` registers the quiet-box precondition and still says
    # 3 percent. It is the registered rule and this file is only its
    # instrument, so the document is the thing that has to be amended for this
    # change to be legitimate rather than a local override.
    comptime CALIBRATION_SPREAD_BAR = 0.05
    var noisy = (
        cpu_spread > CALIBRATION_SPREAD_BAR
        or (have_gpu and gpu_spread > CALIBRATION_SPREAD_BAR)
    )
    if repeats < 3:
        print(
            "calibration_warning: fewer than 3 repeats cannot show whether"
            " the window was quiet. Do not record a baseline from this run."
        )
    elif noisy:
        print(
            "calibration_warning: the repeats disagree by more than 5 percent,"
            " so this window is not the quiet one a baseline has to come from."
            " Do not record these numbers. Find out what else is running and"
            " take it again."
        )
    else:
        print(
            "calibration_note: repeat spread is under 5 percent, which is"
            " consistent with a quiet window but does not prove one, and which"
            " is wide enough that two readings inside 5 percent of each other"
            " are indistinguishable rather than ranked. The evidence that"
            " nothing else was running belongs in the file below, written by"
            " whoever checked."
        )

    # A ready-to-paste file body. Values that were measured are filled in;
    # values that describe the conditions are left as FILL-IN because this
    # program cannot attest to them.
    print("")
    print("--- paste into", baseline_path(), "---")
    print("{")
    print("  \"schema\": \"mojotrees-canary-baseline/1\",")
    print("  \"probe_version\":", String(CANARY_PROBE_VERSION) + ",")
    print("  \"machine\": \"FILL-IN e.g. Apple M4 MacBook, 10 cores\",")
    print("  \"toolchain\": \"FILL-IN output of `mojo --version`\",")
    print("  \"measured_on\": \"FILL-IN YYYY-MM-DD\",")
    print("  \"provenance\": \"measured\",")
    print(
        "  \"quiet_window_evidence\": \"FILL-IN what was checked, e.g. uptime"
        " load and the top processes, no lane or build or agent running\","
    )
    print("  \"repeats\":", String(repeats) + ",")
    print("  \"cpu_ms\":", String(cpu_lo) + ",")
    print("  \"cpu_spread_pct\":", String(_pct(cpu_spread)) + ",")
    print("  \"cpu_checksum\": \"" + String(checksum) + "\",")
    if have_gpu:
        print("  \"gpu_ms\":", String(gpu_lo) + ",")
        print("  \"gpu_spread_pct\":", String(_pct(gpu_spread)) + ",")
    else:
        print("  \"gpu_ms\": null,")
        print("  \"gpu_spread_pct\": null,")
    print(
        "  \"drift_threshold_pct\":",
        String(CANARY_DRIFT_THRESHOLD_PCT) + ",",
    )
    print(
        "  \"note\": \"Minimum of the repeats above. Reading over baseline is"
        " the regime ratio; 1.0 is this window and larger is slower.\""
    )
    print("}")
    print("--- end ---")
