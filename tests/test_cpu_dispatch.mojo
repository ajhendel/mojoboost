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

4. **A snapshot handed to a builder makes that builder read nothing.** This
   is the one the other three cannot establish. Sections 1 and 3 compare
   answers, and the answers were equal before any of this was wired -- which
   is exactly how the snapshot mechanism passed a whole round of tests
   without a single caller in `src/`. Equality proves nothing about *when*
   the environment was read.

   So section 4 counts the reads instead, and it counts them by poisoning.
   `MOJOTREES_CPU_FEATURE_GROUP=3` is off the ladder, and
   `apple_cpu_policy.env_feature_group` *raises* on it rather than rounding
   it, which makes one environment read observable as an exception. Set the
   poison, then drive many node-sized histogram builds through a snapshot
   resolved before it: every build that read the variable would raise, so a
   run that completes read it zero times across all of them. The control is
   the same workload on the same poisoned environment with no snapshot, which
   must raise on the first build. Zero reads over N nodes against at least
   one read per node is the O(1)-versus-O(nodes) claim, asserted rather than
   argued.

   What this does and does not establish is worth being exact about, because
   the whole point of the section is not to accept a proxy. It establishes,
   exactly, that the histogram builders perform **no** read of
   `MOJOTREES_CPU_FEATURE_GROUP` and **no** call of the live
   `derive_accumulation_plan` (and therefore no `CpuProfile.detect()` from
   that site) when handed a snapshot. It does not establish the same for
   `MOJOTREES_NUM_WORKERS` and the four other scheduling variables, because
   those are read by functions that swallow a bad value by design and so
   cannot be poisoned. For those, what is asserted is the divergence
   property directly on `plan_tasks_with`: after a `setenv` of all five, a
   snapshot still answers what it answered, which is only possible if it read
   none of them.
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
    subtract_ops,
    subtract_ops_for_planes,
)
from mojotrees.binning import BinnedMatrix
from mojotrees.histogram import (
    Histogram,
    build_histogram_into,
    build_histogram_subset_into_scratch,
    subtract_histogram_into,
)
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
    # Two grains buys two tasks by the grain alone; the core floor raises that
    # to one task per core, and the item count and the per-core ceiling still
    # cap it. `test_the_core_floor_moves_only_the_idle_shapes` is where that
    # rule is pinned per caller.
    var cores = cpu_profile().dispatch_cores()
    assert_equal(
        plan_tasks(1_000_000, 2 * PARALLEL_MIN_OPS),
        cores if cores > MIN_TASKS_ABOVE_GRAIN else MIN_TASKS_ABOVE_GRAIN,
    )
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


def _rule_before_the_core_floor(n_items: Int, total_ops: Int) raises -> Int:
    """The auto-mode fan-out rule exactly as it stood before the core floor.

    A frozen literal copy of the previous `plan_tasks` / `_cap_tasks`, so that
    "this caller's task count moved" and "this caller's task count did not"
    are assertions against the old behavior rather than against a restatement
    of the new one. If this ever needs updating, the change under it was not
    a pure scheduling change.
    """
    if n_items <= 1:
        return 1
    if total_ops < PARALLEL_MIN_OPS:
        return 1
    var max_auto = cpu_profile().max_auto_tasks()
    var by_grain = total_ops // DEFAULT_MIN_TASK_OPS
    if by_grain < MIN_TASKS_ABOVE_GRAIN:
        by_grain = MIN_TASKS_ABOVE_GRAIN
    var n = by_grain if by_grain < max_auto else max_auto
    if n > n_items:
        n = n_items
    return n


def test_the_core_floor_moves_only_the_idle_shapes() raises:
    """Every caller shape of one profiled fit, old task count against new.

    The fit is the one the round's scaling profile was taken on: 1,000,000
    rows, 50 features, 255 bins, 31 leaves, feature-group width 2 (so 25
    accumulation groups). Every `(n_items, total_ops)` pair below is the pair
    its call site actually passes, computed here from the same estimator
    functions the call site calls, so a change to an estimator moves this
    table with it instead of leaving it stale.

    Three things are asserted, and none of them is a timing:

    1. **No caller lost tasks.** `new >= old` for every shape. The core floor
       only ever raises `by_grain`, and the crossover is tested before it, so
       nothing that was parallel became serial and nothing that was serial
       became parallel.
    2. **The shapes the profile reported as losing never fanned out at all**,
       before or after. Sibling subtraction at 50 x 255 cells and the row
       partition at small and tiny nodes are all below the crossover, on any
       machine, so no fan-out cost can be charged to them and no floor change
       can recover their time.
    3. **The shape that was leaving cores idle now fills them.** The split
       scan's estimate does not depend on the node's row count at all, so the
       old rule answered 3 for every node in the fit whatever the machine;
       that constant is asserted directly, and the new answer is at least one
       task per core.
    """
    _auto()
    _clear_grain()
    var cores = cpu_profile().dispatch_cores()
    var ceiling = cpu_profile().max_auto_tasks()
    print("  dispatch_cores:", cores, "max_auto_tasks:", ceiling)

    comptime N_FEATURES = 50
    comptime N_BINS = 255
    comptime N_GROUPS = 25  # 50 features at the width-2 group the policy picks
    comptime N_LEAVES = 31
    comptime CELLS = N_FEATURES * N_BINS
    comptime ROOT = 1_000_000

    # (name, n_items, total_ops), one row per dispatch the fit makes.
    var names = [
        String("subtract 3-plane"),          # histogram.mojo:1611
        String("subtract 2-plane const-h"),  # histogram.mojo:1611
        String("partition tiny 1953"),       # tree.mojo:673
        String("partition small 15625"),
        String("partition medium 50000"),
        String("partition large 300000"),
        String("partition root 1000000"),
        String("histogram root"),            # histogram.mojo:842/1230
        String("histogram medium 50000"),
        String("histogram small 5000"),
        String("histogram tiny 1953"),
        String("histogram tiny 1000"),
        String("zeroing, nothing excluded"), # histogram.mojo:489
        String("gather compact root"),       # histogram.mojo:1014
        String("split scan two-sided"),      # split.mojo:773
        String("split scan one-sided"),
        String("grad fill root"),            # boosting.mojo:602
        String("grad fill 100000"),
        String("score update traversal"),    # boosting.mojo:1222
        String("score update by leaf"),      # boosting.mojo:1290
    ]
    var items = [
        CELLS, CELLS,
        1_953, 15_625, 50_000, 300_000, ROOT,
        N_GROUPS, N_GROUPS, N_GROUPS, N_GROUPS, N_GROUPS,
        N_FEATURES,
        ROOT,
        N_FEATURES, N_FEATURES,
        ROOT, 100_000,
        ROOT, N_LEAVES,
    ]
    var ops = [
        subtract_ops(CELLS),
        subtract_ops_for_planes(CELLS, 2),
        3 * 1_953, 3 * 15_625, 3 * 50_000, 3 * 300_000, 3 * ROOT,
        N_FEATURES * (N_BINS + ROOT),
        N_FEATURES * (N_BINS + 50_000),
        N_FEATURES * (N_BINS + 5_000),
        N_FEATURES * (N_BINS + 1_953),
        N_FEATURES * (N_BINS + 1_000),
        0,
        2 * ROOT,
        split_scan_ops(N_FEATURES, N_BINS, True),
        split_scan_ops(N_FEATURES, N_BINS, False),
        elementwise_row_ops(ROOT),
        elementwise_row_ops(100_000),
        8 * ROOT,
        2 * ROOT,
    ]
    assert_equal(len(items), len(ops))
    assert_equal(len(items), len(names))

    for i in range(len(items)):
        var old = _rule_before_the_core_floor(items[i], ops[i])
        var new = plan_tasks(items[i], ops[i])
        print("   ", names[i], "ops", ops[i], "items", items[i], ":", old, "->", new)
        # 1. No caller lost tasks, and the go/no-go decision did not move.
        assert_true(new >= old)
        assert_equal(new == 1, old == 1)
        assert_equal(new == 1, ops[i] < PARALLEL_MIN_OPS or items[i] <= 1)
        # 3. Anything that fans out fills the machine, up to what it has.
        if new > 1:
            var reachable = cores if cores < items[i] else items[i]
            if reachable > ceiling:
                reachable = ceiling
            assert_true(new >= reachable)

    # 2. The two phases the profile reported as slower in parallel are serial
    # on both sides of this change, on every machine, so no fan-out cost was
    # ever charged to them. These are absolute, not machine-relative.
    assert_equal(plan_tasks(CELLS, subtract_ops(CELLS)), 1)
    assert_equal(plan_tasks(CELLS, subtract_ops_for_planes(CELLS, 2)), 1)
    assert_equal(plan_row_blocks(15_625, 3 * 15_625).n_blocks, 1)
    assert_equal(plan_row_blocks(1_953, 3 * 1_953).n_blocks, 1)
    # ...and the first node size at which the partition does fan out, so the
    # boundary is pinned rather than implied. 3n >= 65536 at n = 21846.
    assert_equal(plan_row_blocks(21_845, 3 * 21_845).n_blocks, 1)
    assert_true(plan_row_blocks(21_846, 3 * 21_846).n_blocks > 1)

    # 3, stated as the constant it was. The split scan's estimate carries no
    # row count, so the old rule fanned every node in the fit three ways.
    var scan_ops = split_scan_ops(N_FEATURES, N_BINS, True)
    assert_equal(scan_ops // DEFAULT_MIN_TASK_OPS, 3)
    assert_equal(_rule_before_the_core_floor(N_FEATURES, scan_ops), 3)
    var scan_now = plan_tasks(N_FEATURES, scan_ops)
    var scan_reachable = cores if cores < N_FEATURES else N_FEATURES
    assert_true(scan_now >= scan_reachable)
    assert_true(scan_now >= 3)

    # The root histogram is the shape that must not move, and the reason it
    # cannot is arithmetic rather than luck: 50 * (255 + 1000000) is 50012750
    # ops, which is 763 whole grains, so the grain answer is already past both
    # the group count and any ceiling this hardware family reports and the
    # floor has nothing left to raise.
    var root_hist_ops = N_FEATURES * (N_BINS + ROOT)
    assert_equal(root_hist_ops // DEFAULT_MIN_TASK_OPS, 763)
    assert_equal(
        plan_tasks(N_GROUPS, root_hist_ops),
        _rule_before_the_core_floor(N_GROUPS, root_hist_ops),
    )
    # Same for the medium node, which is the regression the round could least
    # afford: 50 * (255 + 50000) is 2512750 ops, 38 whole grains, above the
    # 25 groups the dispatch has to give out.
    var med_hist_ops = N_FEATURES * (N_BINS + 50_000)
    assert_equal(med_hist_ops // DEFAULT_MIN_TASK_OPS, 38)
    assert_equal(
        plan_tasks(N_GROUPS, med_hist_ops),
        _rule_before_the_core_floor(N_GROUPS, med_hist_ops),
    )
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
    # The width rule now takes the active feature count too, because the
    # balance clamp bounds it by how many groups the dispatch can spread
    # over the cores. Both counts are swept: the snapshot and the live path
    # must agree at every (bins, active) pair, since a disagreement would
    # size the accumulation kernel and the plan that allocated its buffers
    # differently and nothing else checks those two against each other.
    var bin_counts = [1, 15, 32, 255, 256, 4096, 100_000]
    var active_counts = [1, 2, 7, 19, 50, 200, 1000]
    for i in range(len(bin_counts)):
        for j in range(len(active_counts)):
            assert_equal(
                resolved.feature_group_for(bin_counts[i], active_counts[j]),
                plan_feature_group(fresh, bin_counts[i], active_counts[j]),
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


# --------------------------------------------------------------------------
# 4. The wiring: a snapshot handed down makes the per-node path read nothing.
# --------------------------------------------------------------------------

# The off-ladder width. `env_feature_group` refuses it rather than rounding
# it, which is what turns one environment read into an observable event. 3 is
# used rather than a word so the refusal is the ladder check and not the
# integer parse.
comptime _POISON = "3"

# Nodes driven through the builders under the poison. Large enough that "the
# reads did not scale with the node count" is a statement about a trend and
# not about a single call, and small enough that the file stays a test.
comptime _POISON_NODES = 200


def _poison():
    _ = setenv("MOJOTREES_CPU_FEATURE_GROUP", _POISON)


def _unpoison():
    _ = setenv("MOJOTREES_CPU_FEATURE_GROUP", "")


def _matrix(n_rows: Int, n_features: Int, n_bins: Int) raises -> BinnedMatrix:
    """A deterministic binned matrix, column-major as `BinnedMatrix` stores
    it. The bin ids are a fixed mixing of (row, feature) so every feature has
    a different occupancy and no two runs of this file differ."""
    var bins = List[UInt8](capacity=n_rows * n_features)
    bins.resize(n_rows * n_features, UInt8(0))
    for f in range(n_features):
        for r in range(n_rows):
            var v = (r * 7 + f * 13 + (r % 5) * (f % 3)) % n_bins
            bins[f * n_rows + r] = UInt8(v)
    return BinnedMatrix(bins^, n_rows, n_features, n_bins)


def _grad(n_rows: Int) -> List[Float64]:
    var g = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        g.append(0.5 - Float64((r * 11) % 17) / 8.0)
    return g^


def _hess(n_rows: Int) -> List[Float64]:
    var h = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        h.append(1.0 + Float64((r * 5) % 9) / 16.0)
    return h^


def _node_rows(n_rows: Int) -> List[Int]:
    """Row ids of a node: every third row, which is the indirect,
    non-contiguous shape a real node has."""
    var rows = List[Int]()
    for r in range(n_rows):
        if r % 3 != 2:
            rows.append(r)
    return rows^


def _assert_same_histogram(a: Histogram, b: Histogram) raises:
    """Bit equality on both float planes and integer equality on the count
    plane. `to_bits()` rather than `==` so a NaN or a signed zero would be
    caught, and no tolerance anywhere: a scheduling change that moved a bit
    would be a defect in the change, not a rounding difference to absorb."""
    assert_equal(a.n_features, b.n_features)
    assert_equal(a.n_bins, b.n_bins)
    assert_equal(len(a.grad), len(b.grad))
    for i in range(len(a.grad)):
        assert_equal(a.grad[i].to_bits(), b.grad[i].to_bits())
        assert_equal(a.hess[i].to_bits(), b.hess[i].to_bits())
        assert_equal(a.count[i], b.count[i])


def test_resolving_the_snapshot_is_where_a_bad_width_is_refused() raises:
    """The poison is real, and it is reported at the snapshot.

    The positive control for everything below it. If `resolve()` stopped
    raising here, the tests that follow would pass by reading nothing that
    could ever have failed, which is the failure mode this section exists to
    avoid.
    """
    _auto()
    _poison()
    var raised = False
    try:
        _ = DispatchSettings.resolve()
    except:
        raised = True
    _unpoison()
    assert_true(raised)
    # And with the poison cleared it resolves, so the refusal is the value
    # and not the act of resolving.
    var ok = DispatchSettings.resolve()
    assert_true(ok.resolved)


def test_a_snapshot_makes_node_histograms_read_no_environment() raises:
    """Zero reads of the poisoned variable over 200 node builds, against at
    least one read on the very first build without a snapshot.

    This is the assertion the lane exists for, and it is a count rather than
    a comparison. The snapshot is taken while the environment is clean; the
    poison is set afterwards. Any `getenv` of `MOJOTREES_CPU_FEATURE_GROUP`
    on the per-node path would then raise, so completing the loop is a proof
    that the count over 200 nodes is exactly zero.

    Both builders are driven, because they plan separately: the full-dataset
    build and the subset build each call the accumulation planner once, and a
    wiring that reached one and not the other would otherwise pass.
    """
    _auto()
    _unpoison()
    var n_rows = 400
    var n_features = 6
    var n_bins = 17
    var data = _matrix(n_rows, n_features, n_bins)
    var grad = _grad(n_rows)
    var hess = _hess(n_rows)
    var rows = _node_rows(n_rows)

    var settings = DispatchSettings.resolve()
    assert_true(settings.resolved)

    _poison()
    var out = Histogram.zeroed(n_features, n_bins)
    var pairs = List[Float64]()
    var parent = Histogram.zeroed(n_features, n_bins)
    var sibling = Histogram.zeroed(n_features, n_bins)
    # Not one call: the claim is that the read count does not grow with the
    # node count, so the loop is what makes the claim testable at all.
    for _ in range(_POISON_NODES):
        build_histogram_subset_into_scratch(
            out, pairs, data, grad, hess, rows, 0, len(rows), [], False,
            settings,
        )
        build_histogram_into(parent, data, grad, hess, [], False, settings)
        subtract_histogram_into(sibling, parent, out, False, settings)
        _ = find_best_split(
            out, lambda_reg=1.0, min_child_hess=1e-3, settings=settings
        )

    # The control, on the same poisoned environment: one build with no
    # snapshot must raise, which is what makes the zero above meaningful.
    var raised_full = False
    try:
        build_histogram_into(parent, data, grad, hess)
    except:
        raised_full = True
    var raised_subset = False
    try:
        build_histogram_subset_into_scratch(
            out, pairs, data, grad, hess, rows, 0, len(rows)
        )
    except:
        raised_subset = True
    _unpoison()
    assert_true(raised_full)
    assert_true(raised_subset)


def test_the_sentinel_default_is_the_pre_change_path() raises:
    """An unwired call site behaves exactly as it did, reads included.

    Two halves. The values: the sentinel plans what a live read plans, at
    every shape in the sweep. The reads: under the poison, a build with the
    sentinel raises exactly as a build with no argument at all does, which is
    what says the default did not quietly acquire a snapshot's silence.
    """
    _auto()
    _unpoison()
    var unresolved = ResolvedCpuPolicy.unresolved()
    assert_true(not unresolved.resolved)
    var fresh = CpuProfile.detect()
    assert_equal(unresolved.dispatch_cores(), fresh.dispatch_cores())
    assert_equal(unresolved.max_auto_tasks(), fresh.max_auto_tasks())

    var rows = [0, 1, 255, 256, 4096, 1_000_000]
    var actives = [0, 1, 2, 7, 50]
    for i in range(len(rows)):
        for j in range(len(actives)):
            for k in range(2):
                var indirect = k == 1
                var a = derive_accumulation_plan_with(
                    unresolved, 50, actives[j], 255, rows[i], indirect
                )
                var b = derive_accumulation_plan(
                    fresh, 50, actives[j], 255, rows[i], indirect
                )
                assert_equal(a.group_width, b.group_width)
                assert_equal(a.group_count, b.group_count)
                assert_equal(a.compact_rows, b.compact_rows)
                assert_equal(a.active_ops, b.active_ops)
                assert_equal(a.excluded_ops, b.excluded_ops)
                assert_equal(a.gather_ops, b.gather_ops)

    var sentinel = DispatchSettings.unresolved()
    assert_true(not sentinel.resolved)
    var items = [1, 2, 40, 1_000_000]
    var ops = [0, PARALLEL_MIN_OPS - 1, PARALLEL_MIN_OPS, 99 * PARALLEL_MIN_OPS]
    for i in range(len(items)):
        for j in range(len(ops)):
            assert_equal(
                plan_tasks_with(sentinel, items[i], ops[j]),
                plan_tasks(items[i], ops[j]),
            )

    var n_rows = 200
    var data = _matrix(n_rows, 5, 13)
    var grad = _grad(n_rows)
    var hess = _hess(n_rows)
    var out = Histogram.zeroed(5, 13)
    _poison()
    var raised = False
    try:
        build_histogram_into(out, data, grad, hess, [], False, sentinel)
    except:
        raised = True
    _unpoison()
    assert_true(raised)


def test_a_snapshot_ignores_a_setenv_of_every_scheduling_variable() raises:
    """After the snapshot is taken, all five variables move and it does not.

    The poison covers one variable exactly; this covers the other five by the
    only instrument they admit, which is that their effect on the plan is
    visible and a snapshot's plan does not move when they do. It is a weaker
    statement than a count -- it proves the snapshot did not re-read them at
    the moment `plan_tasks_with` was called, which is the same thing said
    from the other side.
    """
    _serial()
    _clear_grain()
    _ = setenv("MOJOTREES_CPU_TASKS_PER_CORE", "1")
    _ = setenv("MOJOTREES_CPU_CORE_POOL", "")
    var snapshot = DispatchSettings.resolve()
    var before = plan_tasks_with(snapshot, 1_000_000, 99 * PARALLEL_MIN_OPS)
    assert_equal(before, 1)

    _ = setenv("MOJOTREES_NUM_WORKERS", "8")
    _ = setenv("MOJOTREES_PARALLEL_MIN_OPS", "1")
    _ = setenv("MOJOTREES_PARALLEL_MIN_TASK_OPS", "1")
    _ = setenv("MOJOTREES_CPU_TASKS_PER_CORE", "16")
    _ = setenv("MOJOTREES_CPU_CORE_POOL", "performance")

    assert_equal(
        plan_tasks_with(snapshot, 1_000_000, 99 * PARALLEL_MIN_OPS), before
    )
    assert_equal(snapshot.num_workers, 1)
    assert_equal(snapshot.min_ops, PARALLEL_MIN_OPS)
    assert_equal(snapshot.policy.tasks_per_core, 1)
    assert_equal(snapshot.policy.core_pool, CORE_POOL_ALL)
    # The live read moved, which is what says the flips landed at all.
    assert_equal(plan_tasks(1_000_000, 99 * PARALLEL_MIN_OPS), 8)

    _ = setenv("MOJOTREES_PARALLEL_MIN_OPS", "")
    _ = setenv("MOJOTREES_CPU_TASKS_PER_CORE", "")
    _ = setenv("MOJOTREES_CPU_CORE_POOL", "")
    _clear_grain()
    _auto()


def _build_all(
    settings: DispatchSettings,
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    rows: List[Int],
    n_features: Int,
    n_bins: Int,
) raises -> Histogram:
    """One node's worth of the wired path, ending in a histogram that carries
    every stage's output: the subset build, the full build, and the sibling
    subtraction, folded so that a single comparison covers all three."""
    var node = Histogram.zeroed(n_features, n_bins)
    var pairs = List[Float64]()
    build_histogram_subset_into_scratch(
        node, pairs, data, grad, hess, rows, 0, len(rows), [], False, settings
    )
    var parent = Histogram.zeroed(n_features, n_bins)
    build_histogram_into(parent, data, grad, hess, [], False, settings)
    var sibling = Histogram.zeroed(n_features, n_bins)
    subtract_histogram_into(sibling, parent, node, False, settings)
    # Fold the three into one value so the comparison cannot miss a plane:
    # the sibling carries the subtraction, and the counts carry both builds.
    for i in range(len(node.grad)):
        node.grad[i] = node.grad[i] + sibling.grad[i]
        node.hess[i] = node.hess[i] + sibling.hess[i]
        node.count[i] = node.count[i] + sibling.count[i]
    return node^


def test_the_wired_path_is_identical_at_one_three_and_eight_workers() raises:
    """Determinism across `MOJOTREES_NUM_WORKERS`, on the paths this lane
    rewired, with the snapshot resolved under each setting so the task counts
    genuinely differ.

    Exact comparison on every plane and no tolerance anywhere. The lane moves
    no arithmetic at all -- it removes environment reads and core detections
    and nothing else -- so anything but bit equality here is a defect in the
    change rather than a rounding difference to accept.
    """
    _unpoison()
    _clear_grain()
    var n_rows = 500
    var n_features = 7
    var n_bins = 19
    var data = _matrix(n_rows, n_features, n_bins)
    var grad = _grad(n_rows)
    var hess = _hess(n_rows)
    var rows = _node_rows(n_rows)

    _serial()
    var one = _build_all(
        DispatchSettings.resolve(), data, grad, hess, rows, n_features, n_bins
    )
    var workers = ["3", "8"]
    for i in range(len(workers)):
        _ = setenv("MOJOTREES_NUM_WORKERS", workers[i])
        var settings = DispatchSettings.resolve()
        assert_equal(settings.num_workers, Int(workers[i]))
        _assert_same_histogram(
            one,
            _build_all(settings, data, grad, hess, rows, n_features, n_bins),
        )
        # And the unwired default, at the same worker count, agrees with the
        # snapshot: the two differ in when they read, never in what they
        # compute.
        _assert_same_histogram(
            one,
            _build_all(
                DispatchSettings.unresolved(),
                data, grad, hess, rows, n_features, n_bins,
            ),
        )
    _auto()


def test_the_split_scan_is_identical_with_and_without_a_snapshot() raises:
    """`find_best_split` chooses the same split from a snapshot as from a
    live read, at every worker count, including the tie-break.

    The scan is the other per-node dispatch this lane rewired, and its output
    is a choice rather than an array, so it gets its own comparison: a fold
    that resolved ties by arrival would be exposed by a task count that
    changed, and the snapshot is what changes it here.
    """
    _unpoison()
    _clear_grain()
    var h = _tied_histogram(13, 23, 3)
    _serial()
    var serial = find_best_split(
        h, lambda_reg=1.0, min_child_hess=1e-3,
        settings=DispatchSettings.resolve(),
    )
    assert_true(serial.found)
    var workers = ["1", "3", "8", ""]
    for i in range(len(workers)):
        _ = setenv("MOJOTREES_NUM_WORKERS", workers[i])
        var settings = DispatchSettings.resolve()
        _assert_same_split(
            serial,
            find_best_split(
                h, lambda_reg=1.0, min_child_hess=1e-3, settings=settings
            ),
        )
        _assert_same_split(
            serial, find_best_split(h, lambda_reg=1.0, min_child_hess=1e-3)
        )
    _auto()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
