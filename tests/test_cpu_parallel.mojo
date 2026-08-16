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

4. `dispatch_regions` covers several independent unit spaces under one
   fan-out without changing any of them. Every unit runs exactly once, a
   fused run is bit-identical to one dispatch per region at every worker
   count, and fusing never lowers the width a region would have been given
   alone. The fan-out is *proved* rather than assumed: the coverage assertion
   passes identically on the serial path, so the tests that force it also
   count the emitted ranges and pin the number against the split arithmetic.
   The one thing fusing does change is the go/no-go decision, since the
   estimate is now the union's, and that has its own test rather than being
   left to a reader of the docstring.

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
    dispatch_feature_ranges,
    dispatch_regions,
    plan_row_blocks,
    plan_tasks,
    region_units,
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
        assert_equal(a.grad_at(i), b.grad_at(i))
        assert_equal(a.hess_at(i), b.hess_at(i))
        assert_equal(a.count_at(i), b.count_at(i))


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


# ---------------------------------------------------------------------------
# dispatch_regions: several independent unit spaces under one fan-out
# ---------------------------------------------------------------------------


def _region_offsets(sizes: List[Int]) -> List[Int]:
    """The prefix offsets `_run_regions` builds, recomputed here from the
    documented rule rather than imported, so the test would notice the
    implementation changing its mind about what a non-positive size means."""
    var offs = List[Int](capacity=len(sizes) + 1)
    offs.append(0)
    var run = 0
    for r in range(len(sizes)):
        if sizes[r] > 0:
            run += sizes[r]
        offs.append(run)
    return offs^


def test_regions_tile_every_unit_exactly_once_and_the_fan_out_happened(
) raises:
    # Seven units in region 0 and five in region 1, over four forced tasks.
    # `MOJOTREES_NUM_WORKERS=4` makes the task count 4 on every machine, so
    # the arithmetic below is not a property of the development box.
    var sizes: List[Int] = [7, 5]
    var total = region_units(sizes)
    assert_equal(total, 12)
    var offs = _region_offsets(sizes)

    var visits = List[Int]()
    visits.resize(total, 0)
    var starts = List[Int]()
    starts.resize(total, 0)
    var vp = visits.unsafe_ptr()
    var sp = starts.unsafe_ptr()
    var op = offs.unsafe_ptr()

    # Every flat unit index is owned by exactly one task, so both stores are
    # to storage no other task touches and neither is a race.
    def body(r: Int, s: Int, e: Int) {imm}:
        var base = op.unsafe_load(r)
        sp.unsafe_store(base + s, sp.unsafe_load(base + s) + 1)
        for u in range(s, e):
            vp.unsafe_store(base + u, vp.unsafe_load(base + u) + 1)

    _forced_parallel()
    # Asserted while the forced setting is still live: `_auto()` restores the
    # size-driven rule, under which a twelve-unit loop with an op estimate of
    # 1 is serial and this would say the opposite of what it means.
    assert_equal(plan_tasks(total, 1), 4)
    dispatch_regions(body, sizes, 1)
    _auto()

    for u in range(total):
        assert_equal(visits[u], 1)

    var n_calls = 0
    for u in range(total):
        n_calls += starts[u]
    # Four tasks over twelve units take [0,3) [3,6) [6,9) [9,12). Region 0 is
    # flat [0,7) and region 1 is flat [7,12), so the third task straddles the
    # boundary and is cut in two: the calls are (0,[0,3)) (0,[3,6)) (0,[6,7))
    # (1,[0,2)) (1,[2,5)). Five, against the two a serial run makes.
    #
    # This assertion is the path marker. A test that only checked coverage
    # would pass identically on the serial path and would have established
    # nothing about the fan-out.
    assert_equal(n_calls, 5)


def test_the_serial_region_path_is_one_call_per_non_empty_region() raises:
    # Zero-size regions keep their index and are never called, which is what
    # lets a caller pass a fixed-length vector with a region switched off.
    var sizes: List[Int] = [0, 4, 0, 6]
    var total = region_units(sizes)
    assert_equal(total, 10)
    var offs = _region_offsets(sizes)

    var visits = List[Int]()
    visits.resize(total, 0)
    var per_region = List[Int]()
    per_region.resize(len(sizes), 0)
    var vp = visits.unsafe_ptr()
    var rp = per_region.unsafe_ptr()
    var op = offs.unsafe_ptr()

    def body(r: Int, s: Int, e: Int) {imm}:
        rp.unsafe_store(r, rp.unsafe_load(r) + 1)
        var base = op.unsafe_load(r)
        for u in range(s, e):
            vp.unsafe_store(base + u, vp.unsafe_load(base + u) + 1)

    _serial()
    dispatch_regions(body, sizes, 1)
    _auto()

    for u in range(total):
        assert_equal(visits[u], 1)
    assert_equal(per_region[0], 0)
    assert_equal(per_region[1], 1)
    assert_equal(per_region[2], 0)
    assert_equal(per_region[3], 1)


def _fill_region_sums(
    mut out: List[Float64], sizes: List[Int], fused: Bool
) raises:
    """Sum sixty-four terms per unit, in ascending term order, into that
    unit's own slot -- either through one fused `dispatch_regions` or through
    one `dispatch_feature_ranges` per region.

    The per-unit sum is a fixed sequence of Float64 additions in a fixed
    order, so anything that moved a bit here would be the dispatcher
    reassociating something, which is exactly the property under test.
    """
    var offs = _region_offsets(sizes)
    var op = offs.unsafe_ptr()
    var outp = out.unsafe_ptr()

    if fused:

        def body(r: Int, s: Int, e: Int) {imm}:
            var base = op.unsafe_load(r)
            for u in range(s, e):
                var acc = Float64(0.0)
                for k in range(64):
                    acc += _uniform(UInt64(1_000_003 * (base + u) + k))
                outp.unsafe_store(base + u, acc)

        dispatch_regions(body, sizes, 8 * region_units(sizes))
        return

    for r in range(len(sizes)):
        if sizes[r] <= 0:
            continue
        var base = offs[r]

        def region_range(s: Int, e: Int) {imm}:
            for u in range(s, e):
                var acc = Float64(0.0)
                for k in range(64):
                    acc += _uniform(UInt64(1_000_003 * (base + u) + k))
                outp.unsafe_store(base + u, acc)

        dispatch_feature_ranges(region_range, sizes[r], 8 * sizes[r])


def test_a_fused_dispatch_equals_separate_ones_at_every_worker_count(
) raises:
    var sizes: List[Int] = [13, 11, 7]
    var total = region_units(sizes)

    var baseline = List[Float64]()
    baseline.resize(total, 0.0)
    _serial()
    _fill_region_sums(baseline, sizes, False)

    var settings: List[String] = ["1", "3", "8", "4", ""]
    for i in range(len(settings)):
        _ = setenv("MOJOTREES_NUM_WORKERS", settings[i])
        var fused = List[Float64]()
        fused.resize(total, 0.0)
        _fill_region_sums(fused, sizes, True)
        var separate = List[Float64]()
        separate.resize(total, 0.0)
        _fill_region_sums(separate, sizes, False)
        for u in range(total):
            # Exact, no tolerance: the fan-out is a scheduling decision and
            # every unit's summation order is fixed by the kernel.
            assert_equal(fused[u], baseline[u])
            assert_equal(separate[u], baseline[u])
    _auto()


def test_the_crossover_is_tested_against_the_union() raises:
    _auto()
    _ = setenv("MOJOTREES_PARALLEL_MIN_OPS", "")
    _ = setenv("MOJOTREES_PARALLEL_MIN_TASK_OPS", "")
    _ = setenv("MOJOTREES_CPU_TASK_FLOOR", "")

    # Three quarters of the crossover each: below it alone, above it together.
    var ops_each = (PARALLEL_MIN_OPS * 3) // 4
    assert_true(ops_each < PARALLEL_MIN_OPS)
    assert_true(2 * ops_each >= PARALLEL_MIN_OPS)

    # Separately, each region stays serial. Fused, the pair clears the
    # crossover and fans out. That is a change in the answer, and it is the
    # documented consequence of pricing one scheduling event against the total
    # work behind it.
    assert_equal(plan_tasks(50, ops_each), 1)
    assert_true(plan_tasks(100, 2 * ops_each) > 1)

    var sizes: List[Int] = [50, 50]
    var total = region_units(sizes)
    var starts = List[Int]()
    starts.resize(total, 0)
    var visits = List[Int]()
    visits.resize(total, 0)
    var offs = _region_offsets(sizes)
    var sp = starts.unsafe_ptr()
    var vp = visits.unsafe_ptr()
    var op = offs.unsafe_ptr()

    def body(r: Int, s: Int, e: Int) {imm}:
        var base = op.unsafe_load(r)
        sp.unsafe_store(base + s, sp.unsafe_load(base + s) + 1)
        for u in range(s, e):
            vp.unsafe_store(base + u, vp.unsafe_load(base + u) + 1)

    dispatch_regions(body, sizes, 2 * ops_each)
    for u in range(total):
        assert_equal(visits[u], 1)
    var n_calls = 0
    for u in range(total):
        n_calls += starts[u]
    assert_true(n_calls > len(sizes))


def test_fusing_never_lowers_the_width_a_region_would_have_had() raises:
    _auto()
    _ = setenv("MOJOTREES_PARALLEL_MIN_OPS", "")
    _ = setenv("MOJOTREES_PARALLEL_MIN_TASK_OPS", "")
    _ = setenv("MOJOTREES_CPU_TASK_FLOOR", "")

    # The shape this primitive exists for: the two children of one split,
    # each a fifty-feature scan. `split.split_scan_ops(50, 255, ...)` is
    # 216,750 for a two-sided scan; the exact value does not matter here,
    # only that the fused call is never given fewer tasks than either half
    # would have been given alone.
    var per_child: List[Int] = [50, 50, 216_750, 12_000, 70_000, 5_000_000]
    for i in range(2, len(per_child)):
        var ops = per_child[i]
        var alone = plan_tasks(50, ops)
        var fused = plan_tasks(100, 2 * ops)
        assert_true(fused >= alone)

    # And the width is never more than the machine would have chosen for one
    # of them either, because `max_auto_tasks` binds last and binds
    # absolutely. Fusing buys one fan-out, not a wider one.
    var wide = plan_tasks(100, 2 * 216_750)
    assert_true(wide <= cpu_profile().max_auto_tasks())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
