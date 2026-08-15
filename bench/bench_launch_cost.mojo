"""What one kernel launch and one host synchronization cost on this device.

The GPU trainer's device-resident grower spends a fixed cost per split that
has nothing to do with the split: a fixed number of kernel launches, and one
host wait. Whether any given optimization is worth building depends on which
of the two dominates, and that is a property of the device, not of this
repository. Guessing it wrong wastes a lane -- the split-search scan kernel
was reshaped, tested, measured inside noise, and reverted on the hypothesis
that occupancy was the problem, when the per-split cost was mostly enqueue
and wait. So measure it, and quote the measurement.

Two quantities, both of an empty kernel so the work is excluded:

  per_launch_us        N launches, then one synchronize, divided by N. What
                       one `enqueue_function` costs to submit.
  per_launch_wait_us   N (launch, synchronize) pairs, divided by N. One
                       round trip to the device and back.

A bare synchronization is the difference between them. It is reported that
way rather than measured alone because a synchronize on an empty queue is
not the thing the trainer pays; the trainer always waits on work it just
enqueued.

The arms alternate inside one process, as `bench_train_gpu.mojo` does and
for the same reason: this machine's device timings drift several-fold
between time windows, so only adjacent samples compare. Each arm reports its
own spread, and the summary prices a split against it.

Usage: mojo run -I src bench/bench_launch_cost.mojo [reps] [trials]

Defaults: 200 launches per sample, 5 trials.

    pixi run bench-launch-cost
    pixi run bench-launch-cost 500 5
"""

from std.gpu import global_idx
from std.sys import argv, has_accelerator
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

# What a device-resident split enqueues, from `_device_search_resident` in
# train_gpu.mojo: four for the row partition (flag scan, block-sum scan,
# scatter, copy back), two or three for the histogram (a conditional zeroing,
# then either the atomic kernel or the partial and reduce pair), and two for
# the split search (per-slot scan, per-record reduce). The sibling
# subtraction is not among them: it rides inside the histogram kernel.
comptime SPLIT_LAUNCHES = 8
comptime SPLIT_WAITS = 1


def _touch_kernel(out_buf: MutPointer[Int32, MutAnyOrigin], n: Int32):
    """As little work as a kernel can do and still be a kernel. The point is
    to time the submission, so the body must not be what is measured."""
    var i = global_idx.x
    if i < Int(n):
        out_buf[unsafe_offset=i] = Int32(i)


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


def _pct(fraction: Float64) -> Float64:
    var scaled = fraction * 1000.0
    if scaled >= 0.0:
        return Float64(Int(scaled + 0.5)) / 10.0
    return Float64(Int(scaled - 0.5)) / 10.0


def main() raises:
    comptime if not has_accelerator():
        print("no accelerator present; launch cost benchmark skipped")
    else:
        var reps = 200
        var trials = 5
        var args = argv()
        if len(args) > 1:
            reps = Int(String(args[1]))
        if len(args) > 2:
            trials = Int(String(args[2]))
        if reps < 1 or trials < 1:
            raise Error("reps and trials must both be at least 1")

        var ctx = DeviceContext()
        var n = 1024
        var buf = ctx.enqueue_create_buffer[DType.int32](n)

        # The first launch of a kernel pays its one-time setup, and the first
        # round trip pays more of it than the first launch does -- an
        # unwarmed wait arm read 235us against a settled 148us. Warm both
        # shapes, not just the cheaper one.
        for _ in range(20):
            ctx.enqueue_function[_touch_kernel](
                buf.unsafe_ptr(), Int32(n), grid_dim=4, block_dim=256
            )
        ctx.synchronize()
        for _ in range(20):
            ctx.enqueue_function[_touch_kernel](
                buf.unsafe_ptr(), Int32(n), grid_dim=4, block_dim=256
            )
            ctx.synchronize()

        print("mojotrees launch cost bench: reps", reps, "trials", trials)
        var launch_us = List[Float64](capacity=trials)
        var wait_us = List[Float64](capacity=trials)
        for t in range(trials):
            var t0 = perf_counter_ns()
            for _ in range(reps):
                ctx.enqueue_function[_touch_kernel](
                    buf.unsafe_ptr(), Int32(n), grid_dim=4, block_dim=256
                )
            ctx.synchronize()
            var t1 = perf_counter_ns()
            launch_us.append(Float64(t1 - t0) / 1e3 / Float64(reps))

            var t2 = perf_counter_ns()
            for _ in range(reps):
                ctx.enqueue_function[_touch_kernel](
                    buf.unsafe_ptr(), Int32(n), grid_dim=4, block_dim=256
                )
                ctx.synchronize()
            var t3 = perf_counter_ns()
            wait_us.append(Float64(t3 - t2) / 1e3 / Float64(reps))
            print(
                "trial",
                t + 1,
                "per_launch_us:",
                launch_us[t],
                "per_launch_wait_us:",
                wait_us[t],
            )

        # The minimum leads, as in bench_train_gpu.mojo: it is the sample
        # least contaminated by drift, and the spread beside it says whether
        # it can be trusted.
        var launch = _min_of(launch_us)
        var wait = _min_of(wait_us)
        print("per_launch_us:", launch)
        print(
            "per_launch_spread_pct:",
            _pct((_max_of(launch_us) - launch) / launch),
        )
        print("per_launch_wait_us:", wait)
        print(
            "per_launch_wait_spread_pct:",
            _pct((_max_of(wait_us) - wait) / wait),
        )
        print("sync_us:", wait - launch)

        # What the two numbers mean for the grower, which is the only reason
        # to measure them. A split's fixed cost is what an optimization that
        # removes a launch, or a wait, is bidding against.
        var per_split = Float64(SPLIT_LAUNCHES) * launch + Float64(
            SPLIT_WAITS
        ) * (wait - launch)
        print("split_fixed_us:", per_split)
        print("split_launch_share_us:", Float64(SPLIT_LAUNCHES) * launch)
        print("split_wait_share_us:", Float64(SPLIT_WAITS) * (wait - launch))
        # A default run is 100 rounds of a 31-leaf tree, so 30 splits a tree.
        print("fixed_cost_s_per_100x31_run:", per_split * 3000.0 / 1e6)
        print("one_launch_removed_s_per_100x31_run:", launch * 3000.0 / 1e6)
