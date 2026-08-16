"""Does this machine's GPU clock state move, and does keeping it busy hold it up?

Bucket C: an instrument. It ships nothing and changes no arithmetic; its only
output is a ratio, and a ratio near one closes a question this campaign has been
unable to control or record.

Why it exists
-------------
The Metal System Trace in `docs/METAL_TIMELINE.md` found runs sitting at the
**Minimum** GPU performance state, with the same kernel roughly **2.8x faster at
Maximum**. Nothing in this campaign controls that, records it, or can tell after
the fact which state a number was taken in. Every "fast window" and "slow window"
label in `bench/results/` is inferred from effect -- both arms moving together --
and one such inference was made and retracted the same night.

`pmset -g therm` returns nothing useful on Apple silicon, and `powermetrics`
needs privileges a session does not have. So this probe does not ask the machine
what state it is in. It asks what the machine **delivers**, under three
conditions it controls, and reports the ratios between them. That is the same
posture as `bench/canary.mojo`, and deliberately so.

The three conditions
--------------------
- **cold** -- after an idle gap long enough for a downclock to plausibly have
  happened. This is the state a benchmark's first repeat is in, and therefore
  the state that contaminates a median if the rest of the repeats are not.
- **saturated** -- immediately after a burst that fills the command queue, with
  no gap. This is the state the *middle* of a real fit is in.
- **warm** -- after a few milliseconds of burst, then the measurement. This is
  the state a deliberate warm-up would put a benchmark in.

What the ratios decide, registered before the data
--------------------------------------------------
- **saturated/cold and warm/cold both about 1.0**: the clock is not moving on
  this machine at these workloads, the question closes **with a number**, and no
  lane follows. This is a perfectly good outcome and the most likely one to be
  dismissed for being boring.
- **saturated or warm beats cold outside the spread**: then idle gaps cost real
  time, and that is **the second and larger payoff of `launch-fusion`,
  `trip-count` and K2's speculation** -- all three reduce the host stalls that
  create the gaps, and would be buying clock state on top of the work they
  remove. A warm-up burst then becomes a rule-(2) switch: a trade, shipped behind
  a flag and A/B'd before it becomes a default.

The reference kernel is `bench/canary.mojo`'s GPU probe, reused rather than
reinvented, so a reading here is comparable with the canary line that already
sits in every window header. Changing the kernel would break that comparison for
no gain.

Nothing here is timed by a lane. The orchestrator runs it in the window.
"""

from std.sys import argv
from std.time import perf_counter_ns, sleep

from canary import gpu_probe, GPU_PROBE_LAUNCHES, CANARY_PROBE_VERSION


comptime DEFAULT_REPS = 7
"""Repeats per condition. Seven so a median has something either side of it and
a single contaminated sample cannot carry the result."""

comptime IDLE_SECONDS = 2.0
"""The gap before a `cold` reading.

Chosen, not measured: long enough that a downclock is plausible, short enough
that seven of them are two minutes rather than ten. If the probe reports no
effect, this is the first constant to doubt -- a machine that downclocks after
five seconds would read as flat here. Say so rather than concluding the clock is
fixed.
"""

comptime WARM_BURSTS = 12
"""Bursts before a `warm` reading. Each burst is the reference probe itself, so
the warm-up is the same work the measurement is, which keeps the comparison
honest: `warm` differs from `saturated` only in that the queue has drained
between the burst and the reading."""


struct Condition(Copyable, Movable):
    var name: String
    var samples: List[Float64]

    def __init__(out self, name: String):
        self.name = name.copy()
        self.samples = List[Float64]()

    def median(self) -> Float64:
        var s = self.samples.copy()
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                if s[j] < s[i]:
                    var t = s[i]
                    s[i] = s[j]
                    s[j] = t
        if len(s) == 0:
            return -1.0
        return s[len(s) // 2]

    def spread_pct(self) -> Float64:
        if len(self.samples) < 2:
            return -1.0
        var lo = self.samples[0]
        var hi = self.samples[0]
        for v in self.samples:
            if v < lo:
                lo = v
            if v > hi:
                hi = v
        var med = self.median()
        if med <= 0.0:
            return -1.0
        return 100.0 * (hi - lo) / med


def _one_reading() raises -> Float64:
    """One reference-kernel timing, or -1.0 if the probe is unavailable."""
    var p = gpu_probe()
    return p.ms


def _burst(n: Int) raises:
    """Fill the queue with the reference kernel and do not wait between."""
    for _ in range(n):
        _ = gpu_probe()


def _report(c: Condition):
    """One condition's samples, median and spread, on three lines."""
    print(c.name + "_samples_ms:", end="")
    for v in c.samples:
        print("", v, end="")
    print("")
    print(c.name + "_median_ms:", c.median())
    print(c.name + "_spread_pct:", c.spread_pct())


def main() raises:
    var reps = DEFAULT_REPS
    var args = argv()
    if len(args) > 1:
        reps = Int(args[1])

    print("clock_probe_version:", CANARY_PROBE_VERSION)
    print("reference_kernel: canary gpu_probe,", GPU_PROBE_LAUNCHES, "launches")
    print("reps_per_condition:", reps)
    print("idle_seconds_before_cold:", IDLE_SECONDS)
    print("warm_bursts:", WARM_BURSTS)

    var probe = gpu_probe()
    if probe.ms < 0.0:
        print("clock_probe.status: unavailable")
        print("clock_probe.detail:", probe.note)
        return

    var cold = Condition("cold")
    var saturated = Condition("saturated")
    var warm = Condition("warm")

    # Interleaved, not blocked. A blocked design would put every `cold` sample
    # at the start of the run and every `warm` at the end, so a thermal drift
    # across the run would land entirely on one condition and read as the
    # effect being measured. This project has already lost numbers to exactly
    # that; see `bench/results/MACHINE_LOCK.md`.
    for _ in range(reps):
        sleep(IDLE_SECONDS)
        cold.samples.append(_one_reading())

        _burst(WARM_BURSTS)
        saturated.samples.append(_one_reading())

        _burst(WARM_BURSTS)
        _ = _one_reading()  # let the queue drain, discard
        warm.samples.append(_one_reading())

    # Reported one at a time rather than over a list: `Condition` owns a
    # `List` and is not implicitly copyable, which is the language keeping a
    # silent copy of a result out of a print loop.
    _report(cold)
    _report(saturated)
    _report(warm)

    var cm = cold.median()
    var sm = saturated.median()
    var wm = warm.median()
    print("saturated_over_cold:", cm / sm if sm > 0.0 else -1.0)
    print("warm_over_cold:", cm / wm if wm > 0.0 else -1.0)

    # The verdict rule, stated here rather than left to a reader, and matching
    # `PROFILE_PROTOCOL.md` M0 as amended: an effect inside the wider
    # condition's own spread is not resolved.
    var band = cold.spread_pct()
    if saturated.spread_pct() > band:
        band = saturated.spread_pct()
    if warm.spread_pct() > band:
        band = warm.spread_pct()
    var best = sm if sm < wm else wm
    var gain_pct = 100.0 * (cm - best) / cm if cm > 0.0 else 0.0
    print("band_pct:", band)
    print("best_gain_over_cold_pct:", gain_pct)
    if gain_pct > band:
        print(
            "verdict: resolved -- keeping the device busy holds the clock up."
            " Idle gaps cost real time, so launch-fusion, trip-count and the"
            " speculative prebuild each buy clock state on top of the work"
            " they remove, and a warm-up burst becomes a rule-(2) switch."
        )
    else:
        print(
            "verdict: indistinguishable -- no clock effect at these workloads."
            " Close the question with this number. Doubt IDLE_SECONDS before"
            " concluding the clock is fixed: a machine that downclocks on a"
            " shorter gap than this one would read as flat here."
        )
