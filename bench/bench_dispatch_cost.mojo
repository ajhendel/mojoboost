"""What one `sync_parallelize` costs on this machine, with no work in it.

The CPU counterpart of `bench_launch_cost.mojo`, and it exists for the same
reason. A tree grower pays a fixed cost per node that has nothing to do with
the node: a fixed number of `parallel.dispatch_*` fan-outs, each one a pool
wake and a barrier. Whether it is worth restructuring the grower to make
fewer of them depends entirely on what one of them costs, and that is a
property of the machine and the runtime, not of this repository. Nothing in
this project had ever measured it, so every argument about dispatch overhead
here was arithmetic over an unknown constant.

One quantity, of a dispatch whose body is a single store, so the body is not
what is timed:

  per_dispatch_us    R dispatches of N tasks, divided by R. What one
                     `sync_parallelize(f, N)` costs end to end: enqueue, the
                     wake of whatever workers were parked, the run of N
                     one-store bodies, and the join.

The `tasks=0` row is the control: the same R iterations calling the same body
`max(WIDTHS)` times in a plain loop, with no `sync_parallelize` at all. It
prices the body so a reader can confirm the body is not what is being timed.

Each task writes to its own slot, strided by `SLOT_STRIDE` `Int`s so that two
tasks never share a cache line. Without that padding the barrier would be
timed against a false-sharing storm and the number would be the coherence
protocol's, not the runtime's.

Several trials are run per width and the min, median and max are printed.
The min is the number to quote: this is a shared box and every other sample
is min plus somebody else's interference. The spread is printed so a reader
can see how much of that there was, and a run whose spread is wide is a run
to discard rather than to average.

Usage: mojo run -I src bench/bench_dispatch_cost.mojo [reps] [trials]

Defaults: 2000 dispatches per sample, 5 trials.
"""

from max.algorithm import sync_parallelize
from std.sys import argv
from std.sys.info import (
    num_logical_cores,
    num_performance_cores,
    num_physical_cores,
)
from std.time import perf_counter_ns

from mojotrees.parallel import DispatchSettings

# `Int`s between two tasks' slots. 8 * 8 bytes is one 64-byte line on both
# targets this project builds for; anything smaller times false sharing.
comptime SLOT_STRIDE = 8


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
    var sorted = values.copy()
    for i in range(1, len(sorted)):
        var v = sorted[i]
        var j = i - 1
        while j >= 0 and sorted[j] > v:
            sorted[j + 1] = sorted[j]
            j -= 1
        sorted[j + 1] = v
    return sorted[len(sorted) // 2]


def _round2(x: Float64) -> Float64:
    var scaled = x * 100.0
    if scaled >= 0.0:
        return Float64(Int(scaled + 0.5)) / 100.0
    return Float64(Int(scaled - 0.5)) / 100.0


def _row(label: String, samples: List[Float64], note: String) -> String:
    return String(
        label,
        " | ",
        _round2(_min_of(samples)),
        " | ",
        _round2(_median_of(samples)),
        " | ",
        _round2(_max_of(samples)),
        note,
    )


def main() raises:
    var reps = 2000
    var trials = 5
    var args = argv()
    if len(args) > 1:
        reps = Int(String(args[1]))
    if len(args) > 2:
        trials = Int(String(args[2]))

    # Widths swept. `1` is the runtime's own degenerate fan-out, which
    # `parallel._run_feature_ranges` never asks for (it calls the body
    # directly), and it is here to separate the cost of the machinery from
    # the cost of waking N workers. The rest bracket what the grower actually
    # asks for: 10 is `dispatch_cores` on the development machine, 25 is the
    # histogram's group count at 50 features and interleave width 2, 40 is
    # `max_auto_tasks`.
    var widths: List[Int] = [1, 2, 4, 8, 10, 16, 25, 32, 40]

    var max_width = 0
    for i in range(len(widths)):
        if widths[i] > max_width:
            max_width = widths[i]

    var slots = List[Int](capacity=max_width * SLOT_STRIDE)
    slots.resize(max_width * SLOT_STRIDE, 0)
    var p = slots.unsafe_ptr()

    def touch(w: Int) {imm}:
        p.unsafe_store(w * SLOT_STRIDE, w)

    print("dispatch cost:", reps, "dispatches per sample,", trials, "trials")
    print(
        "machine: physical",
        num_physical_cores(),
        "logical",
        num_logical_cores(),
        "performance",
        num_performance_cores(),
    )
    print("settings:", DispatchSettings.resolve().describe())
    print("tasks | per_dispatch_us min | median | max")

    # Warm the pool once, off the clock, so the first timed sample does not
    # pay whatever one-time cost the runtime has.
    for _ in range(64):
        sync_parallelize(touch, max_width)

    # The control: the same body, the same number of calls, no fan-out.
    var control = List[Float64]()
    for _ in range(trials):
        var t0 = perf_counter_ns()
        for _ in range(reps):
            for w in range(max_width):
                touch(w)
        var t1 = perf_counter_ns()
        control.append(Float64(t1 - t0) / 1e3 / Float64(reps))
    print(_row("    0", control, "   (control: no sync_parallelize)"))

    for i in range(len(widths)):
        var n_tasks = widths[i]
        var samples = List[Float64]()
        for _ in range(trials):
            var t0 = perf_counter_ns()
            for _ in range(reps):
                sync_parallelize(touch, n_tasks)
            var t1 = perf_counter_ns()
            samples.append(Float64(t1 - t0) / 1e3 / Float64(reps))
        print(_row(String(n_tasks), samples, ""))

    # Keep the stores from being dead.
    var sink = 0
    for w in range(max_width):
        sink += slots[w * SLOT_STRIDE]
    print("sink", sink)
