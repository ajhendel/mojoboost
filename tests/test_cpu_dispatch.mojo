"""Contracts for the CPU dispatch rule, the resolved settings snapshot, and
the parallel split scan.

Three things are pinned here, all of them properties rather than timings.
Nothing in this file measures anything, and nothing in it may be read as
evidence that any of the changes it covers is faster; the machine that would
settle that is not this one and the harness is bench/, not tests/.

1. **The split chosen is the split the serial scan chose, bit for bit,
   including the tie-break.** `find_best_split` now scans features in
   parallel. Its tie-break was a strict `>` over an ascending feature order,
   which selects the lowest-indexed feature among candidates of exactly equal
   gain, and the fold that replaced the inline comparison is the same strict
   `>` over the same ascending order. The histograms below are built so that
   several features have *identical* slices, which makes their gains equal in
   the strongest sense available: the same additions in the same order on the
   same values, so the comparison is a genuine tie and not a near miss that
   a different rounding would break. The forced-serial and forced-parallel
   paths must then agree on feature, bin, gain, and missing direction.

2. **The dispatch grain gives a loop above the crossover more than one
   task, at every size above it rather than at some sizes above it.** The
   rule used to answer 1 for any total between one grain and two grains,
   which is the serial path with a scheduling event attached. The sweep below
   walks both sides of the crossover, both sides of the per-task floor, and
   the row counts either side of the point where an elementwise stage's
   estimate clears the crossover, including one million rows, which is the
   shape at which the gradient fill used to fall serial.

3. **A resolved snapshot answers what a live read answers.** `plan_tasks_with`
   against `DispatchSettings.resolve()` must equal `plan_tasks`, and
   `ResolvedCpuPolicy` must equal `CpuProfile.detect()` field for field and
   answer `max_auto_tasks` and the accumulation plan identically, under every
   setting of the variables involved. The snapshot is checked to be a
   snapshot too: it does not observe a `setenv` that happens after it is
   taken, and the unresolved functions do, which is the whole of the
   invalidation contract.
"""

from std.os import setenv
from std.testing import assert_equal, assert_true, TestSuite

from mojotrees.apple_cpu_policy import (
    CORE_POOL_ALL,
    CORE_POOL_PERFORMANCE,
    CpuProfile,
    ResolvedCpuPolicy,
    cpu_profile,
    derive_accumulation_plan,
    derive_accumulation_plan_with,
    env_core_pool,
    env_tasks_per_core,
    plan_feature_group,
    split_scan_ops,
)
from mojotrees.histogram import Histogram
from mojotrees.parallel import (
    DEFAULT_MIN_TASK_OPS,
    DispatchSettings,
    ELEMENTWISE_ROW_COST_DEN,
    MIN_TASKS_ABOVE_GRAIN,
    PARALLEL_MIN_OPS,
    elementwise_row_ops,
    plan_row_blocks,
    plan_tasks,
    plan_tasks_with,
)
from mojotrees.split import SplitInfo, find_best_split


def _serial():
    _ = setenv("MOJOTREES_NUM_WORKERS", "1")


def _forced_parallel():
    # Workers > 1 forces the parallel path whatever the size, which is the
    # only way to exercise the fan-out on a shape small enough to keep a test
    # fast. Four rather than one per core so the count does not depend on the
    # machine the suite runs on.
    _ = setenv("MOJOTREES_NUM_WORKERS", "4")


def _auto():
    _ = setenv("MOJOTREES_NUM_WORKERS", "")


def _clear_grain():
    _ = setenv("MOJOTREES_PARALLEL_MIN_TASK_OPS", "")


# --------------------------------------------------------------------------
# 1. The split scan: parallel across features, identical to serial.
# --------------------------------------------------------------------------


def _tied_histogram(
    n_features: Int, n_bins: Int, n_distinct: Int
) raises -> Histogram:
    """A histogram whose features repeat `n_distinct` distinct slices.

    Feature f gets slice `f % n_distinct`, so every slice is present at least
    twice whenever `n_features > n_distinct` and the features sharing a slice
    have exactly equal gains at exactly equal bins: identical inputs through
    identical arithmetic, not merely close ones. Whichever slice scores
    highest is therefore a tie among all the features carrying it, and the
    tie-break is the only thing that decides the winner.
    """
    var h = Histogram.zeroed(n_features, n_bins)
    for f in range(n_features):
        var kind = f % n_distinct
        for b in range(n_bins):
            # A step in the gradient partway along the bins, at a position
            # that depends on the slice, so different slices have different
            # best gains and different best bins.
            var left = b <= (n_bins * (kind + 1)) // (n_distinct + 2)
            var g = -1.0 - Float64(kind) if left else 1.0
            h.grad[f * n_bins + b] = g
            h.hess[f * n_bins + b] = 1.0 + 0.25 * Float64(b % 3)
            h.count[f * n_bins + b] = 4 + (b % 5)
    return h^


def _assert_same_split(a: SplitInfo, b: SplitInfo) raises:
    assert_equal(a.found, b.found)
    assert_equal(a.feature, b.feature)
    assert_equal(a.bin, b.bin)
    assert_equal(a.gain, b.gain)
    assert_equal(a.default_left, b.default_left)
    assert_equal(a.is_categorical, b.is_categorical)
    for w in range(4):
        assert_equal(a.cat_bitset[w], b.cat_bitset[w])


def test_tied_features_break_toward_the_lower_index() raises:
    """Equal gains resolve to the lowest-indexed feature, serially and in
    parallel.

    This is the assertion the parallel scan exists to not break. With eleven
    features over three distinct slices, the winning slice is carried by four
    features, and the answer must be the first of them in scan order.
    """
    var h = _tied_histogram(11, 23, 3)

    _serial()
    var serial = find_best_split(h, lambda_reg=1.0, min_child_hess=1e-3)
    assert_true(serial.found)
    # The winner is the first feature carrying its slice: the slice index is
    # `feature % 3`, so a tie broken toward the lower index can only land on
    # a feature below 3.
    assert_true(serial.feature < 3)

    _forced_parallel()
    var parallel = find_best_split(h, lambda_reg=1.0, min_child_hess=1e-3)
    _assert_same_split(serial, parallel)

    _auto()
    var auto = find_best_split(h, lambda_reg=1.0, min_child_hess=1e-3)
    _assert_same_split(serial, auto)


def test_tie_break_holds_across_widths() raises:
    """The same tie, at feature counts that fall on either side of every task
    boundary the fan-out can choose.

    A parallel fold that resolved ties by arrival rather than by index would
    pass at one width and fail at another, because which task holds the tied
    features changes with the split. Sweeping the width is what makes the
    assertion about the rule rather than about one schedule.
    """
    var widths = [2, 3, 4, 5, 7, 8, 9, 16, 17, 33]
    for i in range(len(widths)):
        var n_features = widths[i]
        var h = _tied_histogram(n_features, 19, 2)
        _serial()
        var serial = find_best_split(h, lambda_reg=1.0, min_child_hess=1e-3)
        _forced_parallel()
        var parallel = find_best_split(h, lambda_reg=1.0, min_child_hess=1e-3)
        _assert_same_split(serial, parallel)
        assert_true(serial.found)
        # Two distinct slices, so the winner is feature 0 or feature 1.
        assert_true(serial.feature < 2)
    _auto()


def test_tie_break_holds_under_a_feature_subset() raises:
    """With a feature list the scan order is the list's order, and the
    tie-break follows it rather than the raw feature id.

    The list is ascending, which `find_best_split` requires, so the two
    coincide; the point of the case is that the fold indexes slots by
    position in the list and still resolves toward the first position.
    """
    var h = _tied_histogram(12, 21, 3)
    var subset = List[Int]()
    for f in range(12):
        if f % 3 != 0:
            subset.append(f)
    _serial()
    var serial = find_best_split(
        h, lambda_reg=1.0, min_child_hess=1e-3, features=subset
    )
    _forced_parallel()
    var parallel = find_best_split(
        h, lambda_reg=1.0, min_child_hess=1e-3, features=subset
    )
    _assert_same_split(serial, parallel)
    assert_true(serial.found)
    # Feature 0 is not in the list, so the first carrier of slice 0 that the
    # scan can see is feature 3 -- which is also not in the list -- and the
    # winner must be a listed feature whichever slice wins.
    var listed = False
    for i in range(len(subset)):
        if subset[i] == serial.feature:
            listed = True
    assert_true(listed)
    _auto()


def test_missing_direction_survives_the_parallel_scan() raises:
    """A feature with a reserved missing bin reports the same direction from
    either path.

    The two-sided scan is the part of a feature's body with the most state
    carried across bins, and `default_left` is the field a fold that dropped
    a flag would silently get wrong on every node.
    """
    var n_features = 6
    var n_bins = 17
    var h = _tied_histogram(n_features, n_bins, 2)
    # Reserve the last bin of every feature as its missing bin and put rows
    # in it, so both sides of every threshold are scored.
    var missing = List[Int]()
    for f in range(n_features):
        missing.append(n_bins - 1)
        h.grad[f * n_bins + n_bins - 1] = -3.0
        h.hess[f * n_bins + n_bins - 1] = 2.0
        h.count[f * n_bins + n_bins - 1] = 9

    _serial()
    var serial = find_best_split(
        h, lambda_reg=1.0, min_child_hess=1e-3, missing_bins=missing
    )
    _forced_parallel()
    var parallel = find_best_split(
        h, lambda_reg=1.0, min_child_hess=1e-3, missing_bins=missing
    )
    _assert_same_split(serial, parallel)
    assert_true(serial.found)
    _auto()


def test_split_scan_error_reports_from_the_lowest_failing_feature() raises:
    """A feature whose scan raises still raises, from inside a task.

    A task cannot raise, so the scan records the failure and the fold re-runs
    that feature serially to surface the message. The path is exercised here
    because a silently swallowed error would look exactly like a node with no
    split.
    """
    var h = _tied_histogram(5, 13, 2)
    var missing = List[Int]()
    for _ in range(5):
        missing.append(2)
    # Out of range for this histogram, which the per-feature scan rejects.
    missing[3] = 99
    _forced_parallel()
    var raised = False
    try:
        _ = find_best_split(
            h, lambda_reg=1.0, min_child_hess=1e-3, missing_bins=missing
        )
    except:
        raised = True
    assert_true(raised)
    _auto()


def test_split_scan_ops_scales_with_the_work() raises:
    """The scan's work estimate counts candidates, not features.

    It only ever feeds the threshold, so what matters is that it grows with
    both dimensions and that a two-sided feature is dearer than a one-sided
    one; the constant inside it is an unmeasured reading of an instruction
    mix and this asserts nothing about its value.
    """
    assert_true(split_scan_ops(50, 255, False) > split_scan_ops(50, 128, False))
    assert_true(split_scan_ops(50, 255, False) > split_scan_ops(25, 255, False))
    assert_true(split_scan_ops(50, 255, True) > split_scan_ops(50, 255, False))
    assert_equal(split_scan_ops(0, 255, True), 0)
    assert_equal(split_scan_ops(50, 0, True), 0)


# --------------------------------------------------------------------------
# 2. The grain rule: parallelism at every size above the crossover.
# --------------------------------------------------------------------------


def test_crossing_the_crossover_buys_at_least_two_tasks() raises:
    """Above the crossover the answer is never 1.

    One task is the serial path plus a scheduling event, so a rule that
    clears its own threshold and then answers 1 has decided nothing. The
    window between one grain and two grains is exactly where the old rule did
    that, and it is walked bin by bin here.
    """
    _auto()
    _clear_grain()
    # Below: serial, as it always was.
    assert_equal(plan_tasks(1_000_000, PARALLEL_MIN_OPS - 1), 1)
    assert_equal(plan_tasks(1_000_000, 0), 1)
    # At and just above: parallel, and never one task.
    assert_true(plan_tasks(1_000_000, PARALLEL_MIN_OPS) >= MIN_TASKS_ABOVE_GRAIN)
    var probes = [
        PARALLEL_MIN_OPS,
        PARALLEL_MIN_OPS + 1,
        PARALLEL_MIN_OPS + PARALLEL_MIN_OPS // 3,
        2 * PARALLEL_MIN_OPS - 1,
        2 * PARALLEL_MIN_OPS,
        7 * PARALLEL_MIN_OPS,
        1000 * PARALLEL_MIN_OPS,
    ]
    for i in range(len(probes)):
        var n = plan_tasks(1_000_000, probes[i])
        assert_true(n >= MIN_TASKS_ABOVE_GRAIN)
    # The established slope is untouched where it already answered more than
    # one: two grains still buys exactly two tasks.
    assert_equal(plan_tasks(1_000_000, 2 * PARALLEL_MIN_OPS), 2)
    # And the item count still caps everything.
    assert_equal(plan_tasks(1, 1000 * PARALLEL_MIN_OPS), 1)
    assert_equal(plan_tasks(2, 1000 * PARALLEL_MIN_OPS), 2)


def test_task_count_never_falls_as_work_grows() raises:
    """The fan-out is monotone in the work estimate.

    A rule assembled from a threshold, a floor, and a ceiling can easily be
    non-monotone at the seams, and a non-monotone rule is one that answers
    "serial" for a shape larger than one it answered "parallel" for. The
    sweep steps through both seams.
    """
    _auto()
    _clear_grain()
    var previous = 1
    var ops = PARALLEL_MIN_OPS // 4
    while ops < 4096 * PARALLEL_MIN_OPS:
        var n = plan_tasks(4_000_000, ops)
        assert_true(n >= previous)
        previous = n
        ops = ops + ops // 3 + 1


def test_elementwise_rows_reach_the_parallel_path() raises:
    """An elementwise stage fans out above a stated row count and below it
    stays serial, and one million rows is above it.

    The row count is `PARALLEL_MIN_OPS * ELEMENTWISE_ROW_COST_DEN`, which is
    derived here rather than written down so that changing either constant
    moves the assertion with it. What is asserted is the shape of the rule:
    serial below, parallel at and above, monotone across. No timing is
    implied, and the 100k-row finding that motivated the cost weight in the
    first place is checked to still hold.
    """
    _auto()
    _clear_grain()
    var boundary = PARALLEL_MIN_OPS * ELEMENTWISE_ROW_COST_DEN

    # The measured finding: 100k rows of elementwise work stays serial.
    assert_equal(plan_tasks(100_000, elementwise_row_ops(100_000)), 1)

    # Either side of the boundary.
    assert_equal(plan_tasks(boundary - 1, elementwise_row_ops(boundary - 1)), 1)
    assert_true(
        plan_tasks(boundary, elementwise_row_ops(boundary))
        >= MIN_TASKS_ABOVE_GRAIN
    )

    # A sweep of row counts, including the shape the gradient fill used to
    # fall serial at.
    var rows = [
        1,
        1_000,
        100_000,
        boundary - 1,
        boundary,
        boundary + 1,
        500_000,
        1_000_000,
        4_000_000,
    ]
    var previous = 1
    for i in range(len(rows)):
        var n = plan_tasks(rows[i], elementwise_row_ops(rows[i]))
        assert_true(n >= previous)
        previous = n
    assert_true(
        plan_tasks(1_000_000, elementwise_row_ops(1_000_000))
        >= MIN_TASKS_ABOVE_GRAIN
    )
    # And the row blocks that follow from it cover the range exactly.
    var blocks = plan_row_blocks(1_000_000, elementwise_row_ops(1_000_000))
    assert_true(blocks.n_blocks >= MIN_TASKS_ABOVE_GRAIN)
    var seen = 0
    for b in range(blocks.n_blocks):
        assert_equal(blocks.start(b), seen)
        assert_true(blocks.end(b) > blocks.start(b))
        seen = blocks.end(b)
    assert_equal(seen, 1_000_000)


def test_elementwise_row_ops_is_a_scaled_count() raises:
    """The estimate is the row count over a named denominator, and a nonempty
    range is never estimated at zero work."""
    assert_equal(elementwise_row_ops(0), 0)
    assert_equal(elementwise_row_ops(-5), 0)
    assert_equal(elementwise_row_ops(1), 1)
    assert_equal(
        elementwise_row_ops(1_000_000), 1_000_000 // ELEMENTWISE_ROW_COST_DEN
    )


def test_per_task_floor_is_overridable() raises:
    """`MOJOTREES_PARALLEL_MIN_TASK_OPS` moves the fan-out and nothing else.

    Its default equals the crossover, which is what this module applied to
    both questions before they were separated, so an unset variable must
    reproduce the default exactly. Lowering it must not lower the crossover:
    a loop below the crossover stays serial however small the per-task floor
    is set, which is the property that keeps the two questions separate.
    """
    _auto()
    _clear_grain()
    var default_tasks = plan_tasks(1_000_000, 8 * PARALLEL_MIN_OPS)
    assert_equal(DEFAULT_MIN_TASK_OPS, PARALLEL_MIN_OPS)

    _ = setenv("MOJOTREES_PARALLEL_MIN_TASK_OPS", String(PARALLEL_MIN_OPS))
    assert_equal(plan_tasks(1_000_000, 8 * PARALLEL_MIN_OPS), default_tasks)

    _ = setenv("MOJOTREES_PARALLEL_MIN_TASK_OPS", String(PARALLEL_MIN_OPS // 8))
    assert_true(plan_tasks(1_000_000, 8 * PARALLEL_MIN_OPS) >= default_tasks)
    # The crossover is untouched by it.
    assert_equal(plan_tasks(1_000_000, PARALLEL_MIN_OPS - 1), 1)

    # Nonsense falls back to the default rather than to zero or a division by
    # zero.
    _ = setenv("MOJOTREES_PARALLEL_MIN_TASK_OPS", "0")
    assert_equal(plan_tasks(1_000_000, 8 * PARALLEL_MIN_OPS), default_tasks)
    _ = setenv("MOJOTREES_PARALLEL_MIN_TASK_OPS", "not-a-number")
    assert_equal(plan_tasks(1_000_000, 8 * PARALLEL_MIN_OPS), default_tasks)
    _clear_grain()


# --------------------------------------------------------------------------
# 3. The resolved snapshot answers what a live read answers.
# --------------------------------------------------------------------------


def _assert_same_profile(a: CpuProfile, b: CpuProfile) raises:
    assert_equal(a.physical_cores, b.physical_cores)
    assert_equal(a.logical_cores, b.logical_cores)
    assert_equal(a.performance_cores, b.performance_cores)
    assert_equal(a.float64_lanes, b.float64_lanes)
    assert_equal(a.has_neon, b.has_neon)
    assert_equal(a.cache_line_bytes, b.cache_line_bytes)
    assert_equal(a.l1d_bytes, b.l1d_bytes)


def test_resolved_policy_matches_a_fresh_detection() raises:
    """`ResolvedCpuPolicy.resolve()` reports the machine `CpuProfile.detect()`
    reports, and the four knobs the live readers read."""
    _auto()
    _clear_grain()
    var resolved = ResolvedCpuPolicy.resolve()
    var fresh = CpuProfile.detect()
    _assert_same_profile(resolved.profile, fresh)
    _assert_same_profile(resolved.profile, cpu_profile())
    assert_equal(resolved.tasks_per_core, env_tasks_per_core())
    assert_equal(resolved.core_pool, env_core_pool())
    assert_equal(resolved.dispatch_cores(), fresh.dispatch_cores())
    assert_equal(resolved.max_auto_tasks(), fresh.max_auto_tasks())
    var bin_counts = [1, 15, 32, 255, 256, 4096, 100_000]
    for i in range(len(bin_counts)):
        assert_equal(
            resolved.feature_group_for(bin_counts[i]),
            plan_feature_group(fresh, bin_counts[i]),
        )


def test_resolved_policy_tracks_every_pool_and_fan_out_setting() raises:
    """Under each documented setting of the two scheduling knobs, the
    resolved and the live answers agree.

    Including `performance`, which is the only way to confirm the pool is
    reachable at all rather than a spelling nobody has ever exercised.
    """
    _auto()
    _clear_grain()
    var pools = ["", "all", "performance", "PERFORMANCE", "p", "nonsense"]
    var fan_outs = ["", "1", "2", "4", "16", "0", "junk"]
    for i in range(len(pools)):
        _ = setenv("MOJOTREES_CPU_CORE_POOL", pools[i])
        for j in range(len(fan_outs)):
            _ = setenv("MOJOTREES_CPU_TASKS_PER_CORE", fan_outs[j])
            var resolved = ResolvedCpuPolicy.resolve()
            var fresh = CpuProfile.detect()
            assert_equal(resolved.core_pool, env_core_pool())
            assert_equal(resolved.dispatch_cores(), fresh.dispatch_cores())
            assert_equal(resolved.max_auto_tasks(), fresh.max_auto_tasks())
    _ = setenv("MOJOTREES_CPU_CORE_POOL", "")
    _ = setenv("MOJOTREES_CPU_TASKS_PER_CORE", "")


def test_performance_pool_is_reachable_and_no_larger_than_all() raises:
    """The `performance` pool selects the reported performance cores.

    On a symmetric machine the two pools coincide, which is why this asserts
    an inequality rather than a difference: the CI runners are symmetric and
    an assertion that the pools differ would fail there for the right reason.
    Nothing here says which pool is faster; nobody has measured that, and the
    default is deliberately unchanged.
    """
    _auto()
    _ = setenv("MOJOTREES_CPU_CORE_POOL", "")
    var all_pool = ResolvedCpuPolicy.resolve()
    assert_equal(all_pool.core_pool, CORE_POOL_ALL)
    _ = setenv("MOJOTREES_CPU_CORE_POOL", "performance")
    var perf_pool = ResolvedCpuPolicy.resolve()
    assert_equal(perf_pool.core_pool, CORE_POOL_PERFORMANCE)
    assert_true(perf_pool.dispatch_cores() >= 1)
    assert_true(perf_pool.dispatch_cores() <= all_pool.dispatch_cores())
    assert_equal(perf_pool.dispatch_cores(), perf_pool.profile.performance_cores)
    assert_equal(all_pool.dispatch_cores(), all_pool.profile.physical_cores)
    _ = setenv("MOJOTREES_CPU_CORE_POOL", "")


def test_resolved_dispatch_settings_plan_what_a_live_read_plans() raises:
    """`plan_tasks_with` equals `plan_tasks` across the sweep and under every
    worker setting."""
    _clear_grain()
    var settings_for = ["", "1", "3", "0"]
    var items = [1, 2, 3, 40, 1_000_000]
    var ops = [
        0,
        PARALLEL_MIN_OPS - 1,
        PARALLEL_MIN_OPS,
        2 * PARALLEL_MIN_OPS,
        99 * PARALLEL_MIN_OPS,
        100_000 * PARALLEL_MIN_OPS,
    ]
    for w in range(len(settings_for)):
        _ = setenv("MOJOTREES_NUM_WORKERS", settings_for[w])
        var snapshot = DispatchSettings.resolve()
        for i in range(len(items)):
            for j in range(len(ops)):
                assert_equal(
                    plan_tasks_with(snapshot, items[i], ops[j]),
                    plan_tasks(items[i], ops[j]),
                )
    _auto()


def test_resolved_accumulation_plan_matches_the_live_one() raises:
    """The histogram build's plan is the same plan from either form."""
    _auto()
    var resolved = ResolvedCpuPolicy.resolve()
    var fresh = CpuProfile.detect()
    var rows = [0, 1, 255, 256, 4096, 1_000_000]
    var actives = [0, 1, 2, 7, 50]
    for i in range(len(rows)):
        for j in range(len(actives)):
            for k in range(2):
                var indirect = k == 1
                var a = derive_accumulation_plan_with(
                    resolved, 50, actives[j], 255, rows[i], indirect
                )
                var b = derive_accumulation_plan(
                    fresh, 50, actives[j], 255, rows[i], indirect
                )
                assert_equal(a.group_width, b.group_width)
                assert_equal(a.compact_rows, b.compact_rows)
                assert_equal(a.active_ops, b.active_ops)
                assert_equal(a.excluded_ops, b.excluded_ops)
                assert_equal(a.gather_ops, b.gather_ops)


def test_a_snapshot_is_a_snapshot_and_a_live_read_is_live() raises:
    """The invalidation contract, stated as an assertion.

    A resolved value does not observe a later `setenv`; the unresolved
    functions do. That is what lets the existing tests flip
    `MOJOTREES_NUM_WORKERS` mid-process and see the flip, and it is why
    anything that holds a snapshot for the length of a fit must resolve again
    to see a change rather than expect a cache to notice one.
    """
    _serial()
    var snapshot = DispatchSettings.resolve()
    assert_equal(snapshot.num_workers, 1)
    assert_equal(plan_tasks_with(snapshot, 1_000_000, 99 * PARALLEL_MIN_OPS), 1)

    _forced_parallel()
    # The snapshot still says what it said.
    assert_equal(snapshot.num_workers, 1)
    assert_equal(plan_tasks_with(snapshot, 1_000_000, 99 * PARALLEL_MIN_OPS), 1)
    # The live read has moved.
    assert_equal(plan_tasks(1_000_000, 99 * PARALLEL_MIN_OPS), 4)

    # Resolving again is the whole of the invalidation story.
    var again = DispatchSettings.resolve()
    assert_equal(again.num_workers, 4)
    assert_equal(plan_tasks_with(again, 1_000_000, 99 * PARALLEL_MIN_OPS), 4)
    _auto()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
