"""What share of the histogram phase are the three shared atomics?

**The second arm of this probe builds a WRONG histogram on purpose.** Nothing
it prints is a model, a loss, or an accuracy. Its whole output is one ratio.

Run it, do not read about it:

```
pixi run mojo run -I src tools/probe_hist_atomic_fraction.mojo 1000000 50 256 15
pixi run mojo run -I src tools/probe_hist_atomic_fraction.mojo  250000 50 256 25
pixi run mojo run -I src tools/probe_hist_atomic_fraction.mojo   50000 50 256 40
```

Arguments are `n_rows n_features n_bins reps`, all optional, in that order.

Why this file exists
--------------------
Two lanes in this campaign independently revised memory-side estimates
downward and landed on the same sentence. The layout lane: the three shared
atomics are untouched by any address or arrangement change, so they bound the
win from above, and nobody has measured their share. The bin-layout lane: the
atomics cancel in a comparison but not in a speedup, so every number in that
brief is an upper bound on wall-clock effect. **Every memory-side estimate in
the batch is therefore bounded above by a quantity nobody has measured.** This
is the thirty-line measurement of that quantity, and it decides whether the
next kernel lane attacks atomics -- per-node rescaling to unblock int16
packing, or threadgroup privatization -- or attacks addresses.

The two arms
------------
Both build the root histogram of the same dataset through
`GpuHistogramBuilder.enqueue_leaf(0)`, the same kernel at the same launch
geometry, differing in one runtime flag:

- `off`: the shipping arm. Three `Atomic.fetch_add` per (row, feature).
- `probe`: `GpuActiveRows.set_histogram_atomic_probe`. The same row-index
  load, the same quantized gradient load, the same bin gather, the same
  shared-cell address arithmetic, and no atomics at all; the values fold into
  a per-thread sink that is written to threadgroup memory once per thread per
  tile. `_hist_rows_step` lists what is kept and what is dropped, and records
  how the kept work was shown not to have been optimized away -- by reading
  optimized target assembly of a reduced replica, with a negative control in
  which deleting the terminal store collapses the whole loop to `ret`.

The arms alternate inside one process, which on this machine is not a style
preference: device timings here drift several-fold between time windows, so
two runs minutes apart are incomparable and only adjacent samples resolve
anything. That is the same protocol `bench/bench_histogram.mojo` uses for its
own A/Bs and the reduction below is deliberately the same shape as that
file's, so the two can be read side by side.

Which way the number errs
-------------------------
`atomic_fraction` below is `1 - median(probe) / median(off)`. It is a **lower
bound** on what removing atomic contention could buy, and an upper bound on
nothing. Four reasons, every one of them in the same direction:

1. The probe keeps the shared-cell index `lift + bin`, which a real atomics
   change would also keep, so it does not credit itself with work such a
   change would not remove either.
2. The probe pays for its own sink -- one XOR and one or two integer adds per
   (row, feature) -- which lands in the residual and makes the atomics' share
   come out smaller than it is.
3. Removing the atomics would also remove the shared-plane zeroing and the
   flush's dependence on the count plane; the probe removes neither.
4. The bins here are uniform over `n_bins`, which is the least contended
   arrangement at that bin count. Real features are skewed and contend more.
   Sweeping `n_bins` downward is the cheap way to see how much of the number
   is contention rather than atomic-unit throughput, and it is why `n_bins` is
   an argument rather than a constant.

What the number licenses: a large fraction sends the next kernel lane at the
atomics and says the layout estimates in this batch are the small numbers they
look like. A small fraction says no atomics change can be worth its risk and
sends the next lane at addresses. What it does not license: quoting
`1 / (1 - fraction)` as a speedup. Nothing here builds an atomic-free
histogram that is *correct*, so nothing here has measured what one costs.

This is a probe, not a test and not a benchmark. It is referenced by no pixi
task and no CI job on purpose: it needs a device, one of its arms is wrong by
construction, and a suite that ran it would be asserting a property of a
machine. `tests/test_gpu_hist_atomic_probe.mojo` is the part that belongs in a
suite -- it asserts that the probe is off by default, refuses to be turned on
without an acknowledgment, refuses the two entry points through which a
histogram becomes a tree, and really does change the histogram and leave no
residue. Run that before trusting anything here.
"""

from std.sys import argv, has_accelerator
from std.time import perf_counter_ns

from mojotrees.binning import BinnedMatrix
from mojotrees.histogram_gpu import GpuHistogramBuilder


def _splitmix64(state: UInt64) -> UInt64:
    var z = state + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _make_data(
    n_rows: Int, n_features: Int, n_bins: Int
) raises -> BinnedMatrix:
    """A feature-major binned matrix of pseudorandom bins.

    Built directly rather than through `bin_equal_width` on Float64 features,
    because at 1,000,000 x 50 the Float64 source would be 400 MB of host
    memory to produce 50 MB of bins that the binning would have made uniform
    anyway. The distribution is what matters to atomic contention and it is
    uniform either way; see the note on which way the number errs.
    """
    var bins = List[UInt8](capacity=n_rows * n_features)
    for f in range(n_features):
        for r in range(n_rows):
            var v = _splitmix64(UInt64(f * n_rows + r) + 0x51ED2701)
            bins.append(UInt8(Int(v % UInt64(n_bins))))
    return BinnedMatrix(bins^, n_rows, n_features, n_bins)


def _quiet_band(times: List[Float64]) -> Float64:
    """How far this arm's median sits above its own best run, in percent.

    The min-to-max spread is not a noise floor: one descheduled run inflates
    the maximum threefold and would bury any real difference. The band between
    the minimum and the median is what the arm reproduces, so that is what a
    difference has to clear. Lifted from `bench/bench_histogram.mojo` so the
    two files decide "resolved" the same way.
    """
    var lo = times[0]
    var med = times[len(times) // 2]
    if lo <= 0.0:
        return 0.0
    return 100.0 * (med - lo) / lo


def _join_ms(times: List[Float64]) -> String:
    var out = String("")
    for i in range(len(times)):
        if i > 0:
            out += " "
        out += String(times[i] * 1e3)
    return out^


def main() raises:
    var n_rows = 1_000_000
    var n_features = 50
    var n_bins = 256
    var reps = 15
    var args = argv()
    if len(args) > 1:
        n_rows = Int(String(args[1]))
    if len(args) > 2:
        n_features = Int(String(args[2]))
    if len(args) > 3:
        n_bins = Int(String(args[3]))
    if len(args) > 4:
        reps = Int(String(args[4]))

    print("hist atomic fraction probe")
    print("WARNING: the probe arm builds a WRONG histogram by design")
    print("shape:", n_rows, "rows x", n_features, "features,", n_bins, "bins")
    print("reps:", reps, "(arms interleaved in one process)")

    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var data = _make_data(n_rows, n_features, n_bins)
        var grad = List[Float64](capacity=n_rows)
        var hess = List[Float64](capacity=n_rows)
        var base = UInt64(n_rows) * UInt64(n_features)
        for r in range(n_rows):
            grad.append(
                Float64(Int(_splitmix64(base + UInt64(r)) % 2000)) * 0.001
                - 1.0
            )
            hess.append(1.0)

        var builder = GpuHistogramBuilder(data)
        # One untimed full build pays context setup, the binned upload, and
        # the first compilation of every kernel this probe will launch.
        _ = builder.build(grad, hess)
        builder.begin_tree()
        print("feature_group:", builder.feature_group())

        # Both arms are one kernel instantiation reading a runtime flag, so
        # neither pays a compilation the other does not; the untimed pairs
        # below are still run so the first timed sample of each arm is not the
        # first launch of the session.
        for _ in range(2):
            builder.rows.set_histogram_atomic_probe(False, True)
            builder.enqueue_leaf(0)
            builder.synchronize()
            builder.rows.set_histogram_atomic_probe(True, True)
            builder.enqueue_leaf(0)
            builder.synchronize()

        var off = List[Float64](capacity=reps)
        var on = List[Float64](capacity=reps)
        for _ in range(reps):
            builder.rows.set_histogram_atomic_probe(False, True)
            var a0 = perf_counter_ns()
            builder.enqueue_leaf(0)
            builder.synchronize()
            var a1 = perf_counter_ns()
            off.append(Float64(a1 - a0) / 1e9)

            builder.rows.set_histogram_atomic_probe(True, True)
            var b0 = perf_counter_ns()
            builder.enqueue_leaf(0)
            builder.synchronize()
            var b1 = perf_counter_ns()
            on.append(Float64(b1 - b0) / 1e9)
        # Leave the builder on the correct arm rather than on whichever ran
        # last. A probe that exits with the wrong histogram still selected is
        # a trap for whatever runs next in the same process.
        builder.rows.set_histogram_atomic_probe(False, True)

        print("atomics_on_samples_ms:", _join_ms(off))
        print("atomics_off_probe_samples_ms:", _join_ms(on))
        sort(off)
        sort(on)
        var band_off = _quiet_band(off)
        var band_on = _quiet_band(on)
        var floor = band_off if band_off > band_on else band_on
        var med_off = off[len(off) // 2]
        var med_on = on[len(on) // 2]
        print("atomics_on_min_ms:", off[0] * 1e3)
        print("atomics_on_median_ms:", med_off * 1e3)
        print("atomics_off_probe_min_ms:", on[0] * 1e3)
        print("atomics_off_probe_median_ms:", med_on * 1e3)
        # The number this file exists for. MEASURED, on the medians, and a
        # LOWER bound on the share for the four reasons in the module
        # docstring.
        var fraction = 0.0
        if med_off > 0.0:
            fraction = 1.0 - med_on / med_off
        print("atomic_fraction_lower_bound:", fraction)
        var delta = 100.0 * (med_off / med_on - 1.0) if med_on > 0.0 else 0.0
        if abs(delta) > floor:
            print("verdict: resolved, band_pct", floor)
        else:
            print(
                "verdict: indistinguishable, band_pct",
                floor,
                "-- the atomics are not separable from noise at this shape",
            )
