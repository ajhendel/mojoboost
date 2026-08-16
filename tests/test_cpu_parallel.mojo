"""Equivalence and scheduling contracts for the multicore CPU stages.

Three things are checked here, for the stages added or reworked by the CPU
maturity pass (row partitioning, gradient generation, and the scratch-buffer
histogram builders):

1. Every stage is bit-identical between the forced-serial path and the
   parallel path. That is a design property, not a tolerance: feature-parallel
   accumulation keeps each feature's sum inside one task, and row-block
   parallelism is used only where the work is elementwise or where blocks
   write disjoint output ranges in ascending order. Nothing is reassociated,
   so `assert_equal` on Float64 is the right assertion.

2. The `_into` builders agree with the allocating ones, including when a
   buffer is reused, which is what would break if `Histogram.reset` missed
   bins.

3. `plan_tasks` obeys its documented rules: the grain floor, the per-core cap,
   and the `MOJOTREES_NUM_WORKERS` override.

Shapes are deliberately odd against every plausible SIMD width (2, 4, 8, 16,
32 lanes) so the vector loops leave a scalar tail on NEON, AVX2, and AVX-512
alike; CI runs this file on x86-64 and ARM64, which is where that coverage
comes from. The detected ISA is printed so a failing run says which path it
was on.
"""

from std.os import setenv
from std.sys.info import CompilationTarget, simd_width_of
from std.testing import assert_equal, assert_true, TestSuite

from mojotrees.binning import BinnedMatrix, bin_equal_width, fit_bins
from mojotrees.boosting import (
    BINARY_LOGISTIC,
    HUBER,
    L1,
    POISSON,
    QUANTILE,
    SQUARED_ERROR,
    fill_grad_hess,
)
from mojotrees.histogram import (
    Histogram,
    SIMD_LANES,
    build_histogram,
    build_histogram_into,
    build_histogram_subset,
    build_histogram_subset_into,
    subtract_histogram,
    subtract_histogram_into,
)
from mojotrees.apple_cpu_policy import cpu_profile
from mojotrees.parallel import (
    PARALLEL_MIN_OPS,
    TASKS_PER_CORE,
    plan_row_blocks,
    plan_tasks,
)
from mojotrees.split import SplitInfo, find_best_split
from mojotrees.tree import partition_split_rows
from support import _uniform


def _serial():
    _ = setenv("MOJOTREES_NUM_WORKERS", "1")


def _forced_parallel():
    # Workers > 1 forces the parallel path whatever the size, which is the
    # only way to exercise it on a shape small enough to keep a test fast.
    _ = setenv("MOJOTREES_NUM_WORKERS", "4")


def _auto():
    _ = setenv("MOJOTREES_NUM_WORKERS", "")


def _make_data(
    n_rows: Int, n_features: Int, n_bins: Int, seed: UInt64
) raises -> BinnedMatrix:
    var features = List[Float64](capacity=n_rows * n_features)
    for k in range(n_rows * n_features):
        features.append(_uniform(seed + UInt64(k)))
    return bin_equal_width(features, n_rows, n_features, n_bins)


def _grads(n_rows: Int, seed: UInt64) -> List[Float64]:
    var g = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        g.append(2.0 * _uniform(seed + UInt64(r)) - 1.0)
    return g^


def _hessians(n_rows: Int, seed: UInt64) -> List[Float64]:
    var h = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        h.append(_uniform(seed + UInt64(r)) + 0.01)
    return h^


def _assert_same_hist(a: Histogram, b: Histogram) raises:
    assert_equal(a.n_features, b.n_features)
    assert_equal(a.n_bins, b.n_bins)
    for i in range(a.n_features * a.n_bins):
        assert_equal(a.grad[i], b.grad[i])
        assert_equal(a.hess[i], b.hess[i])
        assert_equal(a.count[i], b.count[i])


def test_reports_detected_simd_target() raises:
    """Not an assertion so much as a label on the rest of the file: the SIMD
    tails below only mean something once you know which vector width ran."""
    var isa = String("scalar/unknown")
    if CompilationTarget.has_avx512f():
        isa = String("avx512f")
    elif CompilationTarget.has_avx2():
        isa = String("avx2")
    elif CompilationTarget.has_neon():
        isa = String("neon")
    print(
        "  simd target:",
        isa,
        "float64 width",
        simd_width_of[DType.float64](),
        "kernel lanes",
        SIMD_LANES,
    )
    # The kernels index by SIMD_LANES, so a nonsensical width would corrupt
    # every vector loop rather than fail visibly.
    assert_true(SIMD_LANES >= 1)
    assert_true(simd_width_of[DType.float64]() >= 1)


def test_plan_tasks_respects_grain_and_cap() raises:
    _auto()
    # Below one grain of work the loop stays serial no matter how many items.
    assert_equal(plan_tasks(1_000_000, PARALLEL_MIN_OPS - 1), 1)
    # A single item is never worth a task.
    assert_equal(plan_tasks(1, 1_000_000_000), 1)
    # Exactly two grains of work buys two tasks by the grain alone, but a loop
    # that has cleared the crossover is fanned out over at least one task per
    # core: having paid for the barrier, running on two cores out of ten is
    # eight idle cores. The grain still binds above the core count, and the
    # per-core ceiling still binds above that. See "The core floor" in
    # parallel.mojo, and test_cpu_dispatch.mojo for the per-caller table.
    var two_grains = plan_tasks(1_000_000, 2 * PARALLEL_MIN_OPS)
    var cores = cpu_profile().dispatch_cores()
    assert_true(two_grains >= 2)
    assert_equal(two_grains, cores if cores > 2 else 2)
    assert_true(two_grains <= TASKS_PER_CORE * cores)
    # Plenty of work: capped by cores, never exceeding the item count.
    var big = plan_tasks(1_000_000, 10_000 * PARALLEL_MIN_OPS)
    assert_true(big > 1)
    assert_true(big <= TASKS_PER_CORE * 1024)
    assert_equal(plan_tasks(3, 10_000 * PARALLEL_MIN_OPS), 3)

    # An explicit worker count overrides both the grain floor and the cap,
    # which is what lets the tests above force a path.
    _serial()
    assert_equal(plan_tasks(1_000_000, 10_000 * PARALLEL_MIN_OPS), 1)
    _forced_parallel()
    assert_equal(plan_tasks(1_000_000, 1), 4)
    assert_equal(plan_tasks(2, 1), 2)
    _auto()


def test_row_blocks_cover_the_range_exactly() raises:
    _forced_parallel()
    # 997 is prime, so the ceiling-divided chunk never lands evenly and the
    # last block is short: the case where an off-by-one drops or duplicates
    # rows.
    var blocks = plan_row_blocks(997, 1_000_000_000)
    assert_true(blocks.n_blocks > 1)
    var seen = 0
    for b in range(blocks.n_blocks):
        assert_equal(blocks.start(b), seen)
        assert_true(blocks.end(b) > blocks.start(b))
        seen = blocks.end(b)
    assert_equal(seen, 997)

    _serial()
    var one = plan_row_blocks(997, 1_000_000_000)
    assert_equal(one.n_blocks, 1)
    assert_equal(one.start(0), 0)
    assert_equal(one.end(0), 997)

    # An empty range asks for no blocks at all rather than one empty one.
    _auto()
    assert_equal(plan_row_blocks(0, 1_000_000_000).n_blocks, 0)


def test_partition_matches_serial_reference() raises:
    var n_rows = 1009
    var n_features = 7
    var data = _make_data(n_rows, n_features, 17, UInt64(21))
    var grad = _grads(n_rows, UInt64(1_000_000))
    var hess = _hessians(n_rows, UInt64(2_000_000))

    var hist = build_histogram(data, grad, hess)
    var split = find_best_split(hist)
    assert_true(split.found)
    var missing = data.missing_bin[split.feature]

    # Every row, and an odd strided subset, so the block boundaries fall in
    # different places relative to the data.
    var all_rows = List[Int](capacity=n_rows)
    for r in range(n_rows):
        all_rows.append(r)
    var strided = List[Int]()
    for r in range(1, n_rows, 3):
        strided.append(r)

    var row_sets = [all_rows^, strided^]
    for s in range(len(row_sets)):
        var rows = row_sets[s].copy()

        # Plain single-pass reference: append in row order.
        var ref_left = List[Int]()
        var ref_right = List[Int]()
        for i in range(len(rows)):
            var r = rows[i]
            var bin = data.bin_at(r, split.feature)
            var go_left: Bool
            if split.is_categorical:
                go_left = split.goes_left(bin)
            elif bin == missing:
                go_left = split.default_left
            else:
                go_left = bin <= split.bin
            if go_left:
                ref_left.append(r)
            else:
                ref_right.append(r)

        _serial()
        var serial = partition_split_rows(data, rows, split, missing)
        _forced_parallel()
        var parallel = partition_split_rows(data, rows, split, missing)
        _auto()

        assert_equal(len(serial.left) + len(serial.right), len(rows))
        assert_equal(len(serial.left), len(ref_left))
        assert_equal(len(parallel.left), len(ref_left))
        for i in range(len(ref_left)):
            assert_equal(serial.left[i], ref_left[i])
            assert_equal(parallel.left[i], ref_left[i])
        for i in range(len(ref_right)):
            assert_equal(serial.right[i], ref_right[i])
            assert_equal(parallel.right[i], ref_right[i])


def test_grad_hess_matches_serial_across_objectives() raises:
    var n = 1531
    var target = List[Float64](capacity=n)
    var raw = List[Float64](capacity=n)
    var weights = List[Float64](capacity=n)
    for r in range(n):
        # Poisson needs nonnegative targets; keeping every objective on the
        # same target keeps this one loop honest for all of them.
        target.append(_uniform(UInt64(3_000_000 + r)) * 3.0)
        raw.append(2.0 * _uniform(UInt64(4_000_000 + r)) - 1.0)
        weights.append(_uniform(UInt64(5_000_000 + r)) + 0.1)

    var objectives = [SQUARED_ERROR, BINARY_LOGISTIC, POISSON, HUBER, QUANTILE, L1]
    var no_weights = List[Float64]()
    for k in range(len(objectives)):
        var obj = objectives[k]
        for weighted in range(2):
            var w = weights.copy() if weighted == 1 else no_weights.copy()

            _serial()
            var g_ser = List[Float64]()
            var h_ser = List[Float64]()
            fill_grad_hess(raw, target, obj, w, 0.7, g_ser, h_ser)

            _forced_parallel()
            var g_par = List[Float64]()
            var h_par = List[Float64]()
            fill_grad_hess(raw, target, obj, w, 0.7, g_par, h_par)
            _auto()

            assert_equal(len(g_ser), n)
            assert_equal(len(h_ser), n)
            for r in range(n):
                assert_equal(g_ser[r], g_par[r])
                assert_equal(h_ser[r], h_par[r])

    # Reusing the same buffers must fully overwrite them, not append: this is
    # what the boosting loop does every round.
    _auto()
    var g = List[Float64]()
    var h = List[Float64]()
    fill_grad_hess(raw, target, SQUARED_ERROR, no_weights, 0.7, g, h)
    fill_grad_hess(raw, target, SQUARED_ERROR, no_weights, 0.7, g, h)
    assert_equal(len(g), n)
    assert_equal(len(h), n)
    for r in range(n):
        assert_equal(g[r], raw[r] - target[r])


def test_into_builders_match_allocating_and_survive_reuse() raises:
    # 3 features x 17 bins = 51 entries: odd against 2/4/8/16/32 lanes, so
    # `reset` and the subtraction sweep both leave a scalar tail.
    var n_rows = 907
    var n_features = 3
    var data = _make_data(n_rows, n_features, 17, UInt64(31))
    var grad = _grads(n_rows, UInt64(6_000_000))
    var hess = _hessians(n_rows, UInt64(7_000_000))

    var half = List[Int]()
    for r in range(0, n_rows, 2):
        half.append(r)

    var expect_full = build_histogram(data, grad, hess)
    var expect_half = build_histogram_subset(data, grad, hess, half)
    var expect_sub = subtract_histogram(expect_full, expect_half)

    var buf = Histogram.zeroed(n_features, data.n_bins)

    # A buffer that already holds the subset result must come back holding
    # exactly the full result, which fails if `reset` misses the tail.
    build_histogram_subset_into(
        buf, data, grad, hess, half, 0, len(half)
    )
    _assert_same_hist(expect_half, buf)
    build_histogram_into(buf, data, grad, hess)
    _assert_same_hist(expect_full, buf)

    # The row window is the whole point of the subset `_into` form: a window
    # into a longer list must equal building from that slice alone.
    var padded = List[Int]()
    for r in range(0, n_rows, 3):
        padded.append(r)
    var window_start = 5
    var window_count = len(padded) - 9
    var slice = List[Int]()
    for i in range(window_start, window_start + window_count):
        slice.append(padded[i])
    build_histogram_subset_into(
        buf, data, grad, hess, padded, window_start, window_count
    )
    _assert_same_hist(build_histogram_subset(data, grad, hess, slice), buf)

    var sub = Histogram.zeroed(n_features, data.n_bins)
    subtract_histogram_into(sub, expect_full, expect_half)
    _assert_same_hist(expect_sub, sub)

    # Shape mismatches are rejected rather than writing out of bounds.
    var wrong = Histogram.zeroed(n_features + 1, data.n_bins)
    var raised = False
    try:
        build_histogram_into(wrong, data, grad, hess)
    except:
        raised = True
    assert_true(raised)

    raised = False
    try:
        build_histogram_subset_into(
            buf, data, grad, hess, half, 1, len(half)
        )
    except:
        raised = True
    assert_true(raised)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
