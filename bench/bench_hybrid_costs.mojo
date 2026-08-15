"""Calibrate the eight `HybridCosts` coefficients on this machine.

Experiment E1 of docs/design/HYBRID_TRAINING.md §9: the hybrid leaf
scheduler refuses to place a single leaf until the coefficients its cost
model compares have been measured, and this is the measurement. It times
each term of the device histogram path (launch, transfer, synchronize,
fixed-point conversion, accumulation) and of the host replica path
(zeroing, accumulation, partition mirror) separately, on one dataset shape,
and prints the coefficients in the exact form
`HybridCosts.__init__` takes, with per-unit rates in nanoseconds per
thousand units as that struct defines them.

The trials interleave inside one process, as `bench_train_gpu.mojo` does
and for the same reason: device timings on this machine drift several-fold
between time windows, so only adjacent samples compare. The minimum across
trials is reported, because every quantity here is a fixed cost plus noise
and the calibration wants the fixed cost.

Two builders run: the main shape times the scaling terms, and a tiny
(1024-row) builder isolates the launch cost, whose accumulation is beneath
measurement at that size.

Usage: mojo run -I src bench/bench_hybrid_costs.mojo [rows] [feats] [trials]

Defaults: 500000 rows, 50 features, 255 bins, 5 trials.

    pixi run bench-hybrid-costs
"""

from std.sys import argv, has_accelerator
from std.time import perf_counter_ns

from mojotrees.binning import BinnedMatrix
from mojotrees.gpu_active_rows import (
    LeafRange,
    RowRouting,
    partition_range_host,
)
from mojotrees.histogram import Histogram, build_histogram_subset_into
from mojotrees.histogram_gpu import GpuHistogramBuilder


def _splitmix64(state: UInt64) -> UInt64:
    var z = state + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _uniform(counter: UInt64) -> Float64:
    return Float64(_splitmix64(counter) >> 11) * (1.0 / 9007199254740992.0)


def _make_bins(n_rows: Int, n_features: Int, n_bins: Int) -> List[UInt8]:
    var bins = List[UInt8](capacity=n_rows * n_features)
    for k in range(n_rows * n_features):
        bins.append(UInt8(_splitmix64(UInt64(k)) % UInt64(n_bins)))
    return bins^


def _make_grads(n_rows: Int) -> List[Float64]:
    var g = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        g.append(2.0 * _uniform(UInt64(r) + 0x51AB) - 1.0)
    return g^


def _min_of(values: List[Float64]) -> Float64:
    var m = values[0]
    for i in range(1, len(values)):
        if values[i] < m:
            m = values[i]
    return m


def _per_unit(nanos: Float64, units: Float64) -> Int:
    if units <= 0.0:
        return 0
    var rate = nanos / units
    if rate < 0.0:
        return 0
    return Int(rate + 0.5)


def main() raises:
    comptime if not has_accelerator():
        print("no accelerator; nothing to calibrate")
        return
    else:
        var n_rows = 500_000
        var n_features = 50
        var trials = 5
        var args = argv()
        if len(args) > 1:
            n_rows = Int(args[1])
        if len(args) > 2:
            n_features = Int(args[2])
        if len(args) > 3:
            trials = Int(args[3])
        var n_bins = 255
        comptime REPS = 50

        print(
            "hybrid cost calibration:",
            n_rows,
            "rows x",
            n_features,
            "features x",
            n_bins,
            "bins,",
            trials,
            "trials of",
            REPS,
            "reps",
        )

        var data = BinnedMatrix(
            _make_bins(n_rows, n_features, n_bins), n_rows, n_features, n_bins
        )
        var builder = GpuHistogramBuilder(data)
        var grad = _make_grads(n_rows)
        var hess = List[Float64](capacity=n_rows)
        for _ in range(n_rows):
            hess.append(1.0)
        builder.upload_gradients(grad, hess)
        builder.begin_tree()

        # The tiny builder isolates the launch: at 1024 rows the kernel's
        # accumulation is beneath what the enqueue itself costs.
        var tiny_rows = 1024
        var tiny = BinnedMatrix(
            _make_bins(tiny_rows, n_features, n_bins),
            tiny_rows,
            n_features,
            n_bins,
        )
        var tiny_builder = GpuHistogramBuilder(tiny)
        var tiny_grad = List[Float64](capacity=tiny_rows)
        var tiny_hess = List[Float64](capacity=tiny_rows)
        for r in range(tiny_rows):
            tiny_grad.append(grad[r])
            tiny_hess.append(1.0)
        tiny_builder.upload_gradients(tiny_grad, tiny_hess)
        tiny_builder.begin_tree()

        # Warm both paths once before any timer starts.
        _ = builder.build_leaf(0)
        _ = tiny_builder.build_leaf(0)
        var host_hist = Histogram.zeroed(n_features, n_bins)
        var fixed = List[Int32]()
        var host_rows = List[Int32](capacity=n_rows)
        for r in range(n_rows):
            host_rows.append(Int32(r))
        var host_scratch = List[Int32](capacity=n_rows)
        for _ in range(n_rows):
            host_scratch.append(Int32(0))
        var host_rows_int = List[Int](capacity=n_rows)
        for r in range(n_rows):
            host_rows_int.append(r)
        var snapshot = List[Int32]()
        builder.build_leaf_host_replica(
            host_hist, fixed, data, host_rows, 0, n_rows
        )

        var sync_ns = List[Float64]()
        var launch_ns = List[Float64]()
        var transfer_ns = List[Float64]()
        var convert_ns = List[Float64]()
        var device_build_ns = List[Float64]()
        var host_build_ns = List[Float64]()
        var host_f64_ns = List[Float64]()
        var zero_ns = List[Float64]()
        var partition_ns = List[Float64]()

        for _ in range(trials):
            # A bare synchronize on this queue.
            var t0 = perf_counter_ns()
            for _ in range(REPS):
                builder.ctx.synchronize()
            sync_ns.append(Float64(perf_counter_ns() - t0) / Float64(REPS))

            # Launch: submit REPS tiny-node histogram kernels, wait once.
            t0 = perf_counter_ns()
            for _ in range(REPS):
                tiny_builder.enqueue_leaf(0)
            tiny_builder.ctx.synchronize()
            launch_ns.append(Float64(perf_counter_ns() - t0) / Float64(REPS))

            # The transfer rate, from the largest copy the design makes: the
            # whole-permutation snapshot (4 * n_rows bytes). A rate taken
            # from the small histogram download would fold the fixed enqueue
            # overhead into the per-KiB term and misprice the per-tree
            # snapshot by an order of magnitude.
            t0 = perf_counter_ns()
            for _ in range(10):
                builder.snapshot_rows(snapshot)
            transfer_ns.append(Float64(perf_counter_ns() - t0) / 10.0)

            # Fixed-point to Float64 conversion, host side.
            t0 = perf_counter_ns()
            for _ in range(REPS):
                _ = builder.histogram_from_host()
            convert_ns.append(Float64(perf_counter_ns() - t0) / Float64(REPS))

            # The whole device path for the root.
            t0 = perf_counter_ns()
            for _ in range(5):
                _ = builder.build_leaf(0)
            device_build_ns.append(Float64(perf_counter_ns() - t0) / 5.0)

            # The whole host replica path for the same root.
            t0 = perf_counter_ns()
            for _ in range(5):
                builder.build_leaf_host_replica(
                    host_hist, fixed, data, host_rows, 0, n_rows
                )
            host_build_ns.append(Float64(perf_counter_ns() - t0) / 5.0)

            # The Float64 host builder on the same node, as a reference: if
            # the replica is far slower than this, the replica loop and not
            # the host is what needs work.
            t0 = perf_counter_ns()
            for _ in range(5):
                build_histogram_subset_into(
                    host_hist, data, grad, hess, host_rows_int, 0, n_rows
                )
            host_f64_ns.append(Float64(perf_counter_ns() - t0) / 5.0)

            # The zeroing pass alone.
            t0 = perf_counter_ns()
            for _ in range(REPS):
                host_hist.reset()
            zero_ns.append(Float64(perf_counter_ns() - t0) / Float64(REPS))

            # The host partition mirror over the whole root range. Each rep
            # re-partitions the (already partitioned) permutation; the cost
            # per row is the same whatever the order.
            var routing = RowRouting.numerical(0, n_bins // 2)
            t0 = perf_counter_ns()
            for _ in range(5):
                _ = partition_range_host(
                    host_rows, host_scratch, data, LeafRange(0, n_rows), routing
                )
            partition_ns.append(Float64(perf_counter_ns() - t0) / 5.0)

        var sync = _min_of(sync_ns)
        var launch = _min_of(launch_ns) - sync / Float64(REPS)
        var transfer = _min_of(transfer_ns) - sync
        var convert = _min_of(convert_ns)
        var device_total = _min_of(device_build_ns)
        var host_total = _min_of(host_build_ns)
        var zero = _min_of(zero_ns)
        var partition = _min_of(partition_ns)

        var host_f64 = _min_of(host_f64_ns)
        var kcells = Float64(n_features * n_bins) / 1000.0
        var row_kib = Float64(4 * n_rows) / 1024.0
        var hist_kib = Float64(12 * n_features * n_bins) / 1024.0
        var krow_slots = Float64(n_rows * n_features) / 1000.0
        var krows = Float64(n_rows) / 1000.0

        var transfer_rate = _per_unit(transfer, row_kib)
        var device_accum = (
            device_total
            - launch
            - Float64(transfer_rate) * hist_kib
            - sync
            - convert
        )
        var host_accum = host_total - zero

        print("")
        print("raw minima (nanoseconds):")
        print("  synchronize        ", Int(sync))
        print("  launch             ", Int(launch))
        print("  row readback       ", Int(transfer), "(", row_kib, "KiB )")
        print("  convert            ", Int(convert))
        print("  device build_leaf  ", Int(device_total))
        print("  host replica build ", Int(host_total))
        print("  host float64 build ", Int(host_f64))
        print("  host zero pass     ", Int(zero))
        print("  host partition     ", Int(partition))
        print("")
        print("HybridCosts coefficients (nanos per thousand units):")
        print("  launch_nanos                =", Int(launch))
        print("  sync_nanos                  =", Int(sync))
        print("  transfer_nanos_per_kib      =", transfer_rate)
        print(
            "  device_nanos_per_krow_slot  =",
            _per_unit(device_accum, krow_slots),
        )
        print(
            "  host_nanos_per_krow_slot    =",
            _per_unit(host_accum, krow_slots),
        )
        print(
            "  host_partition_nanos_per_krow =", _per_unit(partition, krows)
        )
        print("  host_zero_nanos_per_kcell   =", _per_unit(zero, kcells))
        print("  convert_nanos_per_kcell     =", _per_unit(convert, kcells))
