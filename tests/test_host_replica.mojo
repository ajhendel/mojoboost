"""The host fixed-point replica builder: the oracle the GPU path is checked
against.

This file was `test_hybrid_replica.mojo` until 2026-08-16, when the hybrid
CPU/GPU leaf scheduler was deleted. The scheduler was one caller of the
replica; it was never the reason the replica exists. docs/ARCHITECTURE.md
names the CPU trainer as "the reference implementation the GPU path is
verified against", and `build_histogram_subset_replica_into` is the only
thing in the package that replays the device's histogram pipeline on the
host bit for bit — same Float32 inputs, same scale, Int32 accumulation,
same dequantization. A trace says what the device did; only this says
whether it was right. So the replica and its tests outlive the scheduler
and the fit-equivalence case that went with it.

Three claims, in the order they build on each other. The replica builder
reproduces its own specification on the CPU (shape, feature subsetting,
determinism). The replica reproduces the device's histograms bit for bit on
this hardware — experiment E2 of docs/design/HYBRID_TRAINING.md §9. And the
same holds over device-produced gradients once `stage_from_device` has read
them back, which is the path the unbagged built-in objectives take.

GPU cases skip (passing) when no accelerator is present.
"""

from std.sys import has_accelerator
from std.testing import assert_equal, assert_true, TestSuite

from mojotrees.binning import BinnedMatrix
from mojotrees.boosting import SQUARED_ERROR
from mojotrees.histogram import (
    Histogram,
    build_histogram_subset_into,
    build_histogram_subset_replica_into,
)
from mojotrees.histogram_gpu import GpuHistogramBuilder
from support import _splitmix64, _uniform


def _make_bins(n_rows: Int, n_features: Int, n_bins: Int) -> List[UInt8]:
    var bins = List[UInt8](capacity=n_rows * n_features)
    for k in range(n_rows * n_features):
        bins.append(UInt8(_splitmix64(UInt64(k)) % UInt64(n_bins)))
    return bins^


def _adversarial_grads(n_rows: Int) -> List[Float64]:
    """Gradients aimed at the rounding: ordinary values, near-tie values,
    tiny values a Float32 barely resolves, sign flips, and exact powers of
    two, so a multiply-and-round that differs between host and device has
    somewhere to differ."""
    var g = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        var m = r % 7
        var u = 2.0 * _uniform(UInt64(r) + 0xE2) - 1.0
        if m == 0:
            g.append(u)
        elif m == 1:
            g.append(u * 1e-6)
        elif m == 2:
            g.append(0.5 if u >= 0.0 else -0.5)
        elif m == 3:
            g.append(u * 0.001953125)  # 2^-9 scale
        elif m == 4:
            g.append(1.0 / 3.0 * u)
        elif m == 5:
            g.append(u * 1e-30)
        else:
            g.append(-u)
    return g^


def _histograms_equal(a: Histogram, b: Histogram) raises:
    assert_equal(a.n_features, b.n_features)
    assert_equal(a.n_bins, b.n_bins)
    for i in range(a.n_features * a.n_bins):
        assert_equal(a.grad_at(i), b.grad_at(i))
        assert_equal(a.hess_at(i), b.hess_at(i))
        assert_equal(a.count_at(i), b.count_at(i))


def test_replica_builder_cpu_contract() raises:
    """Counts match the Float64 builder exactly; excluded features stay
    zero; a second run reproduces the first bit for bit."""
    var n_rows = 3000
    var n_features = 6
    var n_bins = 31
    var data = BinnedMatrix(
        _make_bins(n_rows, n_features, n_bins), n_rows, n_features, n_bins
    )
    var grad = _adversarial_grads(n_rows)
    var hess = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        hess.append(0.5 + _uniform(UInt64(r) + 0x77))

    # Float32 stand-ins for the staged gradients, exactly as
    # `stage_gradients` converts them.
    var g32 = List[Float32](capacity=n_rows)
    var h32 = List[Float32](capacity=n_rows)
    for r in range(n_rows):
        g32.append(Float32(grad[r]))
        h32.append(Float32(hess[r]))

    # A node owning an arbitrary window of an arbitrary permutation.
    var rows = List[Int32](capacity=n_rows)
    for r in range(n_rows):
        rows.append(Int32((r * 2654435761) % n_rows))
    var row_start = 100
    var row_count = 1700

    var g_scale = 214321.0
    var h_scale = 391007.0
    var features: List[Int] = [0, 2, 5]

    var replica = Histogram.zeroed(n_features, n_bins)
    var fixed = List[Int32]()
    build_histogram_subset_replica_into(
        replica, fixed, data, g32, h32, rows, row_start, row_count,
        g_scale, h_scale, features,
    )

    var rows_int = List[Int](capacity=row_count)
    for j in range(row_start, row_start + row_count):
        rows_int.append(Int(rows[j]))
    var reference = Histogram.zeroed(n_features, n_bins)
    build_histogram_subset_into(
        reference, data, grad, hess, rows_int, 0, row_count, features
    )

    # Counts are integers on both paths and must agree exactly; gradient
    # sums differ only by the fixed-point quantization, bounded by half a
    # unit per row.
    var tol = 0.5 * Float64(row_count) / g_scale + 1e-9
    for i in range(n_features * n_bins):
        assert_equal(replica.count_at(i), reference.count_at(i))
        assert_true(abs(replica.grad_at(i) - reference.grad_at(i)) <= tol)

    # Excluded features' slices are zero.
    for b in range(n_bins):
        assert_equal(replica.count_at(1 * n_bins + b), 0)
        assert_equal(replica.grad_at(3 * n_bins + b), 0.0)

    var again = Histogram.zeroed(n_features, n_bins)
    var fixed2 = List[Int32]()
    build_histogram_subset_replica_into(
        again, fixed2, data, g32, h32, rows, row_start, row_count,
        g_scale, h_scale, features,
    )
    _histograms_equal(replica, again)


def test_replica_matches_device_bitwise() raises:
    """E2: the host replica of a node reproduces the device's fixed-point
    histogram exactly, over gradients aimed at the rounding."""
    comptime if not has_accelerator():
        pass
    else:
        var n_rows = 20_000
        var n_features = 9
        var n_bins = 63
        var data = BinnedMatrix(
            _make_bins(n_rows, n_features, n_bins), n_rows, n_features, n_bins
        )
        var builder = GpuHistogramBuilder(data)
        var grad = _adversarial_grads(n_rows)
        var hess = List[Float64](capacity=n_rows)
        for r in range(n_rows):
            hess.append(0.25 + _uniform(UInt64(r) + 0x33))
        builder.upload_gradients(grad, hess)
        builder.begin_tree()

        var device = builder.build_leaf(0)
        var host = Histogram.zeroed(n_features, n_bins)
        var fixed = List[Int32]()
        var rows = List[Int32](capacity=n_rows)
        for r in range(n_rows):
            rows.append(Int32(r))
        builder.build_leaf_host_replica(host, fixed, data, rows, 0, n_rows)
        _histograms_equal(device, host)

        # The snapshot is the root's identity permutation before any split.
        var snapshot = List[Int32]()
        builder.snapshot_rows(snapshot)
        assert_equal(len(snapshot), n_rows)
        for r in range(n_rows):
            assert_equal(Int(snapshot[r]), r)

        # The per-range readback is that same permutation, one window at a
        # time: exactly `count` rows, in buffer order, from any offset; an
        # empty window is legal and moves nothing; a window past the buffer
        # is refused.
        var window = List[Int32]()
        builder.readback_range(1234, 777, window)
        assert_equal(len(window), 777)
        for j in range(777):
            assert_equal(window[j], snapshot[1234 + j])
        builder.readback_range(n_rows - 5, 5, window)
        assert_equal(len(window), 5)
        assert_equal(Int(window[4]), n_rows - 1)
        builder.readback_range(40, 0, window)
        assert_equal(len(window), 0)
        var refused = False
        try:
            builder.readback_range(n_rows - 4, 5, window)
        except:
            refused = True
        assert_true(refused)


def test_replica_reaches_device_gradients() raises:
    """The unbagged built-in objective computes its gradients on the device;
    `stage_from_device` reads the exact Float32 back, so the replica has the
    same inputs as the kernels and reproduces the device build over
    device-made gradients. Before the readback the replica must refuse
    rather than accumulate a stale round."""
    comptime if not has_accelerator():
        pass
    else:
        var n_rows = 6000
        var n_features = 8
        var n_bins = 63
        var data = BinnedMatrix(
            _make_bins(n_rows, n_features, n_bins), n_rows, n_features, n_bins
        )
        var target = List[Float64](capacity=n_rows)
        for r in range(n_rows):
            var b0 = Float64(data.bin_at(r, 0))
            var b1 = Float64(data.bin_at(r, 1))
            target.append(
                0.15 * b0 - 0.07 * b1 + _uniform(UInt64(r) + 0x991) * 0.1
            )

        # The builder-level contract: a device fill leaves the gradients
        # device-only, the readback makes them host-side, and the replica
        # then reproduces the device build over device-made gradients.
        var builder = GpuHistogramBuilder(data)
        var state = builder.objective_state(target, [], 1, 64)
        state.init_raw(builder.ctx, [0.0])
        builder.fill_gradients_device(state, SQUARED_ERROR, 0.0)
        assert_true(not builder.gradients_host)
        var refused = False
        var rows = List[Int32](capacity=n_rows)
        for r in range(n_rows):
            rows.append(Int32(r))
        var host = Histogram.zeroed(n_features, n_bins)
        var fixed = List[Int32]()
        try:
            builder.build_leaf_host_replica(host, fixed, data, rows, 0, n_rows)
        except:
            refused = True
        assert_true(refused)
        builder.stage_from_device()
        assert_true(builder.gradients_host)
        builder.begin_tree()
        var device = builder.build_leaf(0)
        builder.build_leaf_host_replica(host, fixed, data, rows, 0, n_rows)
        _histograms_equal(device, host)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
