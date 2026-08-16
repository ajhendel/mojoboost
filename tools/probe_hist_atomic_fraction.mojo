"""What is the histogram phase actually made of? Four arms of one kernel.

**Three of the four arms build a WRONG histogram on purpose.** Nothing this
file prints is a model, a loss, or an accuracy. Its whole output is three
ratios.

Run it, do not read about it:

```
pixi run mojo run -I src tools/probe_hist_atomic_fraction.mojo 1000000 50 256 15
pixi run mojo run -I src tools/probe_hist_atomic_fraction.mojo  250000 50 256 25
pixi run mojo run -I src tools/probe_hist_atomic_fraction.mojo   50000 50 256 40
```

Arguments are `n_rows n_features n_bins reps`, all optional, in that order.
All four arms come out of one invocation at one launch geometry, interleaved
inside one process, because on this machine device timings drift several-fold
between time windows and only adjacent samples resolve anything.

Why this file exists
--------------------
It began as a two-arm probe of one question and it answered it: at
1,000,000 x 50 x 256 bins the three shared atomics are **9.8%** of the
histogram phase, resolved against a 4.5% band. A second lane closed the other
question everyone was asking, showing the feature-blocked bin layout is a
no-op at the shipping configuration and that the bin gather is not broken,
because a GPU coalesces across threads rather than across one thread's own
accesses.

Between them that is about a tenth of the phase, and **every kernel proposal
on the table was aimed at one of those two small parts.** Roughly ninety
percent of the histogram phase was unattributed. These two further arms cut
the remainder along the only two seams that are left: what the loop's memory
traffic costs, and what a launch costs before any thread does anything.

The four arms
-------------
All four build the root histogram of the same dataset through
`GpuHistogramBuilder.enqueue_leaf(0)`: the same kernel, the same grid, the
same threadgroup shape, differing in one runtime flag.

1. `full` -- the shipping arm and the denominator. Nothing is removed.
2. `no_atomics` -- `GpuActiveRows.set_histogram_atomic_probe`. Keeps the row
   load, the quantized gradient load, the bin gather and the shared-cell
   index; drops the three `Atomic.fetch_add`; folds into a per-thread XOR
   sink written once per thread per tile. Built by the atomic-fraction lane
   and reused unchanged.
3. `no_gather` -- `HIST_PROBE_NO_GATHER`. The reverse cut. Drops all three
   global reads and keeps the atomics, the trip count, the shared zeroing,
   the barrier and the whole flush. The bin comes from a per-thread rolling
   counter, seeded at `tid % n_bins`, so a warp still spreads over the bins
   the way uniform random bins spread it; a single constant bin would have
   serialized the threadgroup on one cell and made the arm slower than the
   kernel it is a floor for.
4. `empty` -- `HIST_PROBE_EMPTY`. The kernel returns as soon as its
   threadgroup planes exist. Launch, grid and occupancy, and nothing else.

`_hist_rows_step`, `_hist_probe_empty_mark` and
`GpuActiveRows.set_histogram_probe_mode` are where each arm's kept-and-
dropped list lives, where the assembly reading that shows the kept work was
not optimized away is recorded, and where the negative control for each is
written down. Read those before quoting any number from here.

Which way each number errs
--------------------------
Not one of the three is two-sided, and they do not err the same way. In
short, with the full argument at the arm:

- `atomic_fraction` is a **lower bound** on what removing atomic contention
  could buy. The probe keeps the shared-cell index a real change would keep,
  it pays for its own sink, it removes neither the shared zeroing nor the
  flush's dependence on the count plane, and uniform bins are the least
  contended arrangement at a given bin count.
- `gather_fraction` is **bracketed rather than one-signed**. The rolling
  counter's three ALU operations per visit push it down; the friendlier
  consecutive-bin address stream pushes it up. The second term is the one to
  rule out first, by re-running at a lower `n_bins`.
- `launch_fraction` bounds **launch overhead proper from above** (the empty
  arm still pays six threadgroup stores, three loads and a branch per thread)
  and **fixed per-launch cost from below**, by a lot: the zeroing, both
  barriers and the entire flush are excluded and land in the residual.

**The four do not partition the phase and nothing here claims they sum to
one.** Each arm removes a part; what is left over is a residual containing
everything every arm keeps. And no arm here builds a histogram that is
*correct*, so nothing here has measured what any real change would cost.
Quoting `1 / (1 - fraction)` as a speedup is the one thing this instrument
makes easy and it is wrong.

What each result sends the next lane at
---------------------------------------
- **gather-bound** (`gather_fraction` large): bit-packed bins, `ellpack`
  style at `ceil(log2(n_bins))` bits per feature, and Float16 or packed
  gradient staging.
- **reduce-bound** (all three small, residual large): tile count and
  reduction shape. Those arms already exist. Note that an earlier 80-tile
  experiment measured 22% slower at 50 features and 36% at 100, with 12.3 MB
  of partials per node histogram, linear in tile count, as the registered
  explanation.
- **launch-bound** (`launch_fraction` large): fusion, the speculative
  prebuild that is already built and unmeasured, and fewer launches per
  round.

This is a probe, not a test and not a benchmark. It is referenced by no pixi
task and no CI job on purpose: it needs a device, three of its arms are wrong
by construction, and a suite that ran it would be asserting a property of a
machine. `tests/test_gpu_hist_atomic_probe.mojo` and
`tests/test_gpu_hist_decomposition.mojo` are the parts that belong in a
suite -- between them they assert that every arm is off by default, refuses
to be turned on without an acknowledgment, refuses the two entry points
through which a histogram becomes a tree, refuses to be combined with
another arm, and really does change the histogram and leave no residue. Run
those before trusting anything here.
"""

from std.sys import argv, has_accelerator
from std.time import perf_counter_ns

from mojotrees.binning import BinnedMatrix
from mojotrees.gpu_active_rows import (
    HIST_PROBE_EMPTY,
    HIST_PROBE_NO_GATHER,
    HIST_PROBE_OFF,
)
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
    uniform either way; see the note on which way the numbers err.
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


def _verdict(med_full: Float64, med_arm: Float64, band: Float64) -> String:
    """Whether this arm's distance from the full kernel clears the noise.

    The band is the larger of the two arms' own quiet bands, which is the
    same rule `bench/bench_histogram.mojo` applies, so an arm that cannot
    reproduce itself cannot resolve anything against another either.
    """
    var delta = 0.0
    if med_arm > 0.0:
        delta = 100.0 * (med_full / med_arm - 1.0)
    if abs(delta) > band:
        return String("resolved")
    return String("INDISTINGUISHABLE")


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

    print("hist decomposition probe: four arms, one launch geometry")
    print("WARNING: three of the four arms build a WRONG histogram by design")
    print("shape:", n_rows, "rows x", n_features, "features,", n_bins, "bins")
    print("reps:", reps, "(all four arms interleaved in one process)")

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

        # All four arms are one kernel instantiation reading two runtime
        # flags, so none pays a compilation the others do not; the untimed
        # rounds below are still run so the first timed sample of each arm is
        # not the first launch of the session.
        #
        # The two flags are exclusive by construction -- each setter refuses
        # while the other is live -- so every arm below clears the previous
        # one before selecting itself, and there is no ordering in which two
        # arms are on at once.
        for _ in range(2):
            builder.rows.set_histogram_probe_mode(HIST_PROBE_OFF, True)
            builder.rows.set_histogram_atomic_probe(False, True)
            builder.enqueue_leaf(0)
            builder.synchronize()
            builder.rows.set_histogram_atomic_probe(True, True)
            builder.enqueue_leaf(0)
            builder.synchronize()
            builder.rows.set_histogram_atomic_probe(False, True)
            builder.rows.set_histogram_probe_mode(HIST_PROBE_NO_GATHER, True)
            builder.enqueue_leaf(0)
            builder.synchronize()
            builder.rows.set_histogram_probe_mode(HIST_PROBE_EMPTY, True)
            builder.enqueue_leaf(0)
            builder.synchronize()

        var full = List[Float64](capacity=reps)
        var no_atomics = List[Float64](capacity=reps)
        var no_gather = List[Float64](capacity=reps)
        var empty = List[Float64](capacity=reps)
        for _ in range(reps):
            builder.rows.set_histogram_probe_mode(HIST_PROBE_OFF, True)
            builder.rows.set_histogram_atomic_probe(False, True)
            var a0 = perf_counter_ns()
            builder.enqueue_leaf(0)
            builder.synchronize()
            var a1 = perf_counter_ns()
            full.append(Float64(a1 - a0) / 1e9)

            builder.rows.set_histogram_atomic_probe(True, True)
            var b0 = perf_counter_ns()
            builder.enqueue_leaf(0)
            builder.synchronize()
            var b1 = perf_counter_ns()
            no_atomics.append(Float64(b1 - b0) / 1e9)
            builder.rows.set_histogram_atomic_probe(False, True)

            builder.rows.set_histogram_probe_mode(HIST_PROBE_NO_GATHER, True)
            var c0 = perf_counter_ns()
            builder.enqueue_leaf(0)
            builder.synchronize()
            var c1 = perf_counter_ns()
            no_gather.append(Float64(c1 - c0) / 1e9)

            builder.rows.set_histogram_probe_mode(HIST_PROBE_EMPTY, True)
            var d0 = perf_counter_ns()
            builder.enqueue_leaf(0)
            builder.synchronize()
            var d1 = perf_counter_ns()
            empty.append(Float64(d1 - d0) / 1e9)
        # Leave the builder on the correct arm rather than on whichever ran
        # last. A probe that exits with a wrong histogram still selected is a
        # trap for whatever runs next in the same process.
        builder.rows.set_histogram_probe_mode(HIST_PROBE_OFF, True)
        builder.rows.set_histogram_atomic_probe(False, True)

        print("full_samples_ms:", _join_ms(full))
        print("no_atomics_samples_ms:", _join_ms(no_atomics))
        print("no_gather_samples_ms:", _join_ms(no_gather))
        print("empty_samples_ms:", _join_ms(empty))
        sort(full)
        sort(no_atomics)
        sort(no_gather)
        sort(empty)
        var band_full = _quiet_band(full)
        var band_na = _quiet_band(no_atomics)
        var band_ng = _quiet_band(no_gather)
        var band_em = _quiet_band(empty)
        var med_full = full[len(full) // 2]
        var med_na = no_atomics[len(no_atomics) // 2]
        var med_ng = no_gather[len(no_gather) // 2]
        var med_em = empty[len(empty) // 2]
        print("full_min_ms:", full[0] * 1e3)
        print("full_median_ms:", med_full * 1e3, "band_pct", band_full)
        print("no_atomics_min_ms:", no_atomics[0] * 1e3)
        print("no_atomics_median_ms:", med_na * 1e3, "band_pct", band_na)
        print("no_gather_min_ms:", no_gather[0] * 1e3)
        print("no_gather_median_ms:", med_ng * 1e3, "band_pct", band_ng)
        print("empty_min_ms:", empty[0] * 1e3)
        print("empty_median_ms:", med_em * 1e3, "band_pct", band_em)

        # The three numbers this file exists for. MEASURED, on the medians.
        # Each one is `1 - median(arm) / median(full)`, and each errs in the
        # direction its own docstring states; they are not a partition.
        var f_atomic = 0.0
        var f_gather = 0.0
        var f_launch = 0.0
        if med_full > 0.0:
            f_atomic = 1.0 - med_na / med_full
            f_gather = 1.0 - med_ng / med_full
            f_launch = med_em / med_full
        print("atomic_fraction_lower_bound:", f_atomic)
        print("gather_fraction_bracketed:", f_gather)
        # Note the different form: the empty arm's share IS the launch floor,
        # not the complement of it, because nothing was removed from it.
        print("launch_fraction_of_full:", f_launch)
        # What is left after all three cuts. It is not "the reduction": it
        # contains the shared zeroing, both barriers, the flush, and every
        # part each arm keeps. A large residual is what sends the next lane
        # at the tiling and the reduction shape rather than at the row loop.
        print("unattributed_residual:", 1.0 - f_atomic - f_gather - f_launch)

        var band_a = band_full if band_full > band_na else band_na
        var band_g = band_full if band_full > band_ng else band_ng
        var band_e = band_full if band_full > band_em else band_em
        print(
            "verdict_atomics:",
            _verdict(med_full, med_na, band_a),
            "band_pct",
            band_a,
        )
        print(
            "verdict_gather:",
            _verdict(med_full, med_ng, band_g),
            "band_pct",
            band_g,
        )
        print(
            "verdict_launch:",
            _verdict(med_full, med_em, band_e),
            "band_pct",
            band_e,
        )
