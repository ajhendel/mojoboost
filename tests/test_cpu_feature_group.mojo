"""The CPU accumulation ladder: one histogram, whatever the interleave width.

`histogram.mojo` accumulates `GROUP` features in one walk of a node's rows,
with `GROUP` a comptime rung of the ladder 1, 2, 4, 8, 16 and the rung chosen
by `apple_cpu_policy.plan_feature_group`. Widening an interleave is exactly
the kind of change that quietly reassociates a Float64 sum, so the central
assertion of this file is that it does not: every rung must produce the
*identical bytes*, compared with `assert_equal` on Float64 rather than with a
tolerance, and the same at every worker count and on either side of the
gradient-gather decision.

Why that is a design property rather than a hope is argued in
`histogram.mojo`'s docstring: a feature belongs to one group, a group belongs
to one task, and the task walks the node's rows in ascending order, so the
sequence of additions into a cell `(f, b)` is the node's rows in ascending
order filtered to bin `b` whatever the width. This file is the guard on that
argument, not a substitute for it.

The shapes are chosen to break the width rather than to flatter it:

- Feature counts 1, 3, 8, 17 and 50. Every one of them leaves a ragged tail at
  some rung (17 at width 8 is two full groups and a group of one; 50 at width
  16 is three full groups and a group of two), and 1 is the degenerate case
  where the tail group *is* the only group.
- Bin counts 4, 7, 32 and 255, so the zeroing loop leaves a scalar tail
  against every plausible `SIMD_LANES`, and so the L1 estimate resolves to
  different rungs across the sweep.
- Row counts on both sides of `DEFAULT_COMPACT_MIN_ROWS`, which is what
  selects between the gathered pair buffer and the two indirect loads. Before
  this ladder the interleave required the gather; now the two are independent
  and both have to agree.
- Gradients spanning sixteen orders of magnitude, alternating per row. A sum
  of such terms is worth several ULP of difference under reassociation, so a
  width that reordered anything would fail loudly instead of coincidentally
  agreeing.
- A strict row window, `rows[3 : len - 4]`, because a node in a real tree is
  a window into a shared arena and the window offset is applied once, outside
  the group loop.

Nothing here measures anything. The ladder's justification is memory traffic
and that is a benchmark's question; this file only establishes that choosing
among the rungs is free of numerical consequence, which is what makes the
benchmark worth running.
"""

from std.os import setenv
from std.testing import assert_equal, assert_raises, assert_true, TestSuite

from mojotrees.apple_cpu_policy import (
    CpuProfile,
    DEFAULT_FEATURE_GROUP,
    FEATURE_GROUP_LADDER,
    MAX_FEATURE_GROUP,
    TASK_BALANCE_FACTOR,
    cache_feature_group,
    cpu_profile,
    private_cell_bytes,
    derive_accumulation_plan,
    describe_cpu_policy,
    dispatch_rounds,
    dispatch_utilization_percent,
    env_feature_group,
    feature_group_count,
    feature_group_floor,
    is_feature_group_width,
    plan_feature_group,
    schedule_feature_group,
)
from mojotrees.binning import BinnedMatrix, bin_equal_width
from mojotrees.histogram import (
    Histogram,
    build_histogram,
    build_histogram_subset,
)

from support import _uniform


def _rungs() -> List[Int]:
    """Every width a kernel is instantiated at, spelled out rather than
    derived from `FEATURE_GROUP_LADDER`, so that a change to the ladder has to
    be made in two places and is noticed here."""
    var out: List[Int] = [1, 2, 4, 8, 16]
    return out^


def _set_group(group: Int):
    _ = setenv("MOJOTREES_CPU_FEATURE_GROUP", String(group))


def _unset_group():
    _ = setenv("MOJOTREES_CPU_FEATURE_GROUP", "")


def _set_workers(n: String):
    _ = setenv("MOJOTREES_NUM_WORKERS", n)


def _set_compact_min_rows(n: String):
    _ = setenv("MOJOTREES_CPU_COMPACT_MIN_ROWS", n)


def _reset_env():
    _unset_group()
    _set_workers("")
    _set_compact_min_rows("")
    _ = setenv("MOJOTREES_CPU_CORE_POOL", "")


def _make_data(
    n_rows: Int, n_features: Int, n_bins: Int, seed: UInt64
) raises -> BinnedMatrix:
    var values = List[Float64](capacity=n_rows * n_features)
    for k in range(n_rows * n_features):
        values.append(_uniform(seed + UInt64(k)))
    return bin_equal_width(values, n_rows, n_features, n_bins)


def _grads(n_rows: Int, seed: UInt64) -> List[Float64]:
    """Signed gradients whose magnitudes alternate over sixteen decades.

    Summing 1e8 and 1e-8 terms in a different order changes the result in the
    low bits, which is precisely the failure a wider interleave could
    introduce and precisely what a uniform [0, 1) sample would hide.
    """
    var g = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        var u = 2.0 * _uniform(seed + UInt64(r)) - 1.0
        var scale = 1.0e8 if r % 3 == 0 else 1.0e-8
        g.append(u * scale)
    return g^


def _hessians(n_rows: Int, seed: UInt64) -> List[Float64]:
    var h = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        var u = _uniform(seed + UInt64(r)) + 0.01
        var scale = 1.0e-7 if r % 2 == 0 else 1.0e7
        h.append(u * scale)
    return h^


def _feature_ids(n_features: Int) -> List[Int]:
    var f = List[Int](capacity=n_features)
    for i in range(n_features):
        f.append(i)
    return f^


def _row_ids(n_rows: Int) -> List[Int]:
    var rows = List[Int](capacity=n_rows)
    for r in range(n_rows):
        rows.append(r)
    return rows^


def _assert_identical(a: Histogram, b: Histogram, label: String) raises:
    """Exact equality on all three planes. No tolerance: the contract is
    bit-identity, and a tolerance here would pass exactly the regression this
    file exists to catch."""
    assert_equal(a.n_features, b.n_features)
    assert_equal(a.n_bins, b.n_bins)
    for i in range(a.n_features * a.n_bins):
        if a.grad_at(i) != b.grad_at(i) or a.hess_at(i) != b.hess_at(i):
            print("  mismatch at cell", i, "for", label)
        assert_equal(a.grad_at(i), b.grad_at(i))
        assert_equal(a.hess_at(i), b.hess_at(i))
        assert_equal(a.count_at(i), b.count_at(i))


# --------------------------------------------------------------------------
# The ladder itself
# --------------------------------------------------------------------------


def test_ladder_membership() raises:
    """Only the five rungs are widths, because only the five have a kernel."""
    var rungs = _rungs()
    _reset_env()
    assert_equal(FEATURE_GROUP_LADDER, 5)
    assert_equal(MAX_FEATURE_GROUP, 16)
    assert_equal(DEFAULT_FEATURE_GROUP, MAX_FEATURE_GROUP)
    for i in range(len(rungs)):
        assert_true(is_feature_group_width(rungs[i]))
    var refused: List[Int] = [-4, -1, 0, 3, 5, 6, 7, 9, 12, 15, 17, 24, 32]
    for i in range(len(refused)):
        assert_true(not is_feature_group_width(refused[i]))


def test_floor_rounds_down_never_up() raises:
    """A clamp expresses a bound, so the rung it resolves to must respect the
    bound. Rounding up would widen past the cache estimate that produced it."""
    _reset_env()
    assert_equal(feature_group_floor(-1), 1)
    assert_equal(feature_group_floor(0), 1)
    assert_equal(feature_group_floor(1), 1)
    assert_equal(feature_group_floor(2), 2)
    assert_equal(feature_group_floor(3), 2)
    # 32768 / 6120 = 5 slices fit at 255 bins, and 5 is not a rung.
    assert_equal(feature_group_floor(5), 4)
    assert_equal(feature_group_floor(7), 4)
    assert_equal(feature_group_floor(8), 8)
    assert_equal(feature_group_floor(15), 8)
    assert_equal(feature_group_floor(16), 16)
    assert_equal(feature_group_floor(1000), MAX_FEATURE_GROUP)


def test_group_count_covers_the_ragged_tail() raises:
    _reset_env()
    assert_equal(feature_group_count(0, 4), 0)
    assert_equal(feature_group_count(50, 1), 50)
    assert_equal(feature_group_count(50, 2), 25)
    assert_equal(feature_group_count(50, 4), 13)
    assert_equal(feature_group_count(50, 8), 7)
    assert_equal(feature_group_count(50, 16), 4)
    assert_equal(feature_group_count(17, 8), 3)
    assert_equal(feature_group_count(1, 16), 1)


def test_env_override_reaches_every_rung() raises:
    """The knob's whole purpose is to reach widths the estimate declines to
    choose, so it must bypass the derived clamps and land exactly."""
    var rungs = _rungs()
    _reset_env()
    var machine = CpuProfile.synthetic(10, 4)
    for i in range(len(rungs)):
        var rung = rungs[i]
        _set_group(rung)
        assert_equal(env_feature_group(), rung)
        # 255 bins: the cache estimate alone would cap this at 4.
        assert_equal(plan_feature_group(machine, 255, 50), rung)
        # And it reaches past the active feature count too, where the
        # kernel's tail group owns everything there is.
        assert_equal(plan_feature_group(machine, 255, 3), rung)
        var plan = derive_accumulation_plan(machine, 50, 50, 255, 4096, True)
        assert_equal(plan.group_width, rung)
        assert_equal(plan.group_count, feature_group_count(50, rung))
    _reset_env()
    assert_equal(env_feature_group(), 0)


def test_env_override_refuses_an_off_ladder_value() raises:
    """Refused, not rounded. A benchmark arm that asked for 3 and silently got
    2 has recorded a result under the wrong label, which has happened in this
    repository before."""
    _reset_env()
    var machine = CpuProfile.synthetic(10, 4)
    var refused: List[String] = [
        "0", "3", "5", "6", "7", "9", "17", "32", "-2", "wide", "2.0",
    ]
    for i in range(len(refused)):
        _ = setenv("MOJOTREES_CPU_FEATURE_GROUP", refused[i])
        with assert_raises():
            _ = env_feature_group()
        with assert_raises():
            _ = plan_feature_group(machine, 255, 50)
        with assert_raises():
            _ = derive_accumulation_plan(machine, 50, 50, 255, 4096, True)

    # Surrounding whitespace is not a refusal, because `Int` trims it and a
    # shell that exported `" 2"` asked for 2. Asserted rather than left
    # unstated, so the boundary between "refused" and "accepted" is written
    # down where the next person reading the error message will look.
    _ = setenv("MOJOTREES_CPU_FEATURE_GROUP", " 2")
    assert_equal(env_feature_group(), 2)
    _reset_env()


def test_derived_width_is_always_a_rung() raises:
    """Whatever the shape, the plan names a width a kernel exists at."""
    _reset_env()
    var machine = CpuProfile.synthetic(10, 4)
    var bins: List[Int] = [1, 2, 4, 7, 32, 63, 128, 255, 256, 1024, 100000]
    var actives: List[Int] = [0, 1, 2, 3, 8, 17, 50, 100, 1000]
    for b in range(len(bins)):
        for a in range(len(actives)):
            var group = plan_feature_group(machine, bins[b], actives[a])
            assert_true(is_feature_group_width(group))
            assert_true(group <= MAX_FEATURE_GROUP)
            assert_true(group <= cache_feature_group(machine, bins[b]))
            assert_true(group <= schedule_feature_group(machine, actives[a]))


def test_the_balance_rule_binds_before_the_cache_estimate() raises:
    """The width on an **unblocked** node, pinned with both of the bounds that
    could have chosen it, so a later edit to `ASSUMED_L1D_BYTES`,
    `L1_GROUP_DIVISOR` or `TASK_BALANCE_FACTOR` has to come here and say so.

    At 255 bins a feature's slice is 255 * 24 = 6120 bytes, the assumed budget
    is 65536 / 2 = 32768, five slices fit, and the widest rung at or below
    five is 4. The balance rule is narrower: ten cores at
    `TASK_BALANCE_FACTOR` 2 want 20 dispatch units, and on one block those
    units can only be groups, so `50 // 20 = 2` and the width is 2 rather than
    the 4 the cache estimate would have allowed. That is the correction that
    keeps a byte count from choosing alone.

    Every call below leaves `row_blocks` at its default of 1, which is the
    only shape on which the balance rule still binds first.
    `test_the_balance_rule_counts_block_group_units` is the blocked case, and
    on it this bound stops binding entirely.
    """
    _reset_env()
    var machine = CpuProfile.synthetic(10, 4)
    assert_equal(TASK_BALANCE_FACTOR, 2)
    assert_equal(cache_feature_group(machine, 255), 4)
    assert_equal(schedule_feature_group(machine, 50), 2)
    assert_equal(plan_feature_group(machine, 255, 50), 2)
    # Low-cardinality data does not change the answer, because the cache
    # estimate was never what bound: 32 bins is 768 bytes a slice, so the
    # cache allows 16 and the balance rule still says 2.
    assert_equal(cache_feature_group(machine, 32), 16)
    assert_equal(plan_feature_group(machine, 32, 50), 2)
    # The rungs unlock at width * TASK_BALANCE_FACTOR * cores features.
    assert_equal(plan_feature_group(machine, 32, 39), 1)
    assert_equal(plan_feature_group(machine, 32, 40), 2)
    assert_equal(plan_feature_group(machine, 32, 79), 2)
    assert_equal(plan_feature_group(machine, 32, 80), 4)
    assert_equal(plan_feature_group(machine, 32, 160), 8)
    assert_equal(plan_feature_group(machine, 32, 320), 16)
    # At 255 bins the cache estimate takes over above width 4, so a very wide
    # matrix stops at 4 however many features it has.
    assert_equal(plan_feature_group(machine, 255, 320), 4)


def test_the_cache_clamp_sizes_the_private_cell_not_the_output_cell() raises:
    """The clamp protects the cell the accumulate kernels read and write, and
    under a constant hessian that is two Float64 rather than three.

    This is a regression test for a defect, so it asserts both arms by
    literal rung rather than asserting a relation between them. The clamp
    used to size every fit with `HISTOGRAM_BYTES_PER_CELL`, which is 24 and
    describes the *output* `Histogram`; the private cell in
    `histogram._accumulate_blocked_at` is `ROW_BLOCK_CELL_FLOATS_CONST_H`
    floats, which is 16 bytes, whenever the objective guarantees a constant
    hessian. Squared error does, so the default objective was running an
    interleave one rung narrower than its own working set allows, and the
    width divides the number of times the accumulate re-walks the node's
    row-id list and its gathered derivatives.

    The three-plane arm is asserted too, and not as an afterthought: it is
    the arm every caller without an objective still gets, and a fix that
    silently widened it as well would have widened it past the working set
    it really has.
    """
    _reset_env()
    var machine = CpuProfile.synthetic(10, 4)
    assert_equal(private_cell_bytes(False), 24)
    assert_equal(private_cell_bytes(True), 16)
    # 65536 / 2 = 32768 budget. 32768 / (255 * 24 = 6120) = 5 -> rung 4.
    assert_equal(cache_feature_group(machine, 255, False), 4)
    # 32768 / (255 * 16 = 4080) = 8 -> rung 8.
    assert_equal(cache_feature_group(machine, 255, True), 8)
    # It reaches the planner, with enough blocks that the balance rule does
    # not bind first and enough features that the active floor does not
    # either. This is the assertion that would have failed before the fix.
    assert_equal(plan_feature_group(machine, 255, 100, 27, False), 4)
    assert_equal(plan_feature_group(machine, 255, 100, 27, True), 8)
    # At low cardinality the cache estimate was never what bound, so the
    # correction changes nothing there and the two arms agree.
    assert_equal(cache_feature_group(machine, 32, False), 16)
    assert_equal(cache_feature_group(machine, 32, True), 16)
    # An explicit request still bypasses the clamp at either cell size; the
    # knob exists to test the estimate, so the estimate must not bound it.
    _ = setenv("MOJOTREES_CPU_FEATURE_GROUP", "16")
    assert_equal(plan_feature_group(machine, 255, 100, 27, False), 16)
    assert_equal(plan_feature_group(machine, 255, 100, 27, True), 16)
    _reset_env()


def test_the_balance_rule_counts_block_group_units() raises:
    """The clamp counts `(block, group)` pairs, which is what the kernel
    dispatches over, and not feature groups alone.

    `histogram._accumulate_blocked_at` runs `row_blocks * group_count` units
    and indexes them `block * group_count + group`. Counting groups alone made
    the width *fall* as the machine got bigger while the second axis stood
    idle: at 50 features on 14 cores the old rule wanted 28 groups out of 50
    features, `50 // 28 = 1`, so the shipped width was **1** and the gradient
    stream was walked 50 times per block. Every number below is arithmetic on
    the constants, not a measurement.

    The three shapes that matter, all at 50 features and 255 bins, where the
    L1 estimate allows 4:

    - 1,000,000 rows blocks 54 ways (245 by the 4,080-row minimum, capped by
      the 16 MB scratch budget at `16777216 / (50 * 255 * 3 * 8) = 54`).
      `ceil(28 / 54) = 1` group per block, `50 // 1 = 50`, floored to the top
      rung 16 -- so the balance rule stops binding and the **L1 clamp of 4
      governs**.
    - 250,000 rows blocks 54 ways too, for the same ceiling, so it is the same
      answer: **L1 clamp, width 4**.
    - 8,000 rows does not block at all (4,080 is the minimum, so one block),
      and there the balance rule still governs and still returns 1. Most nodes
      deep in a tree are this shape and this change does not touch them.

    Nothing here moves a bit; `test_blocked_bits_identical_across_feature_
    group_widths` in `test_histogram_reference.mojo` is the guard on that.
    """
    _reset_env()
    var m14 = CpuProfile.synthetic(14, 14)
    var m10 = CpuProfile.synthetic(10, 10)

    # One block is the old arithmetic to the integer, on both machines.
    assert_equal(schedule_feature_group(m14, 50), 1)
    assert_equal(schedule_feature_group(m10, 50), 2)
    assert_equal(plan_feature_group(m14, 255, 50), 1)

    # 54 blocks covers the demand with one group each, so the schedule bound
    # goes to the top of the ladder and the L1 estimate decides.
    assert_equal(schedule_feature_group(m14, 50, 54), MAX_FEATURE_GROUP)
    assert_equal(schedule_feature_group(m10, 50, 54), MAX_FEATURE_GROUP)
    assert_equal(plan_feature_group(m14, 255, 50, 54), 4)

    # And through the planner, which is what the histogram actually calls.
    var p1m = derive_accumulation_plan(m14, 50, 50, 255, 1_000_000, True)
    assert_equal(p1m.row_blocks, 54)
    assert_equal(p1m.group_width, 4)
    assert_equal(p1m.group_count, feature_group_count(50, 4))
    var p250k = derive_accumulation_plan(m14, 50, 50, 255, 250_000, True)
    assert_equal(p250k.row_blocks, 54)
    assert_equal(p250k.group_width, 4)

    # A node too small to block keeps the old answer, which is the right one:
    # with one block there really are only `ceil(50 / width)` units.
    var small = derive_accumulation_plan(m14, 50, 50, 255, 8_000, True)
    assert_equal(small.row_blocks, 1)
    assert_equal(small.group_width, 1)

    # Narrow data, so the clamp direction is visible in the other regime. Ten
    # features block 64 ways at a million rows (`MAX_ROW_BLOCKS`), so the
    # schedule bound is `50 // 1` again; what is left is the L1 estimate at
    # 255 bins (4) and the active-feature floor at 32 bins, where L1 allows 16
    # and ten features floor to 8.
    var narrow255 = derive_accumulation_plan(m14, 10, 10, 255, 1_000_000, True)
    assert_equal(narrow255.row_blocks, 64)
    assert_equal(cache_feature_group(m14, 255), 4)
    assert_equal(narrow255.group_width, 4)
    var narrow32 = derive_accumulation_plan(m14, 10, 10, 32, 1_000_000, True)
    assert_equal(narrow32.row_blocks, 64)
    assert_equal(cache_feature_group(m14, 32), 16)
    assert_equal(feature_group_floor(10), 8)
    assert_equal(narrow32.group_width, 8)

    # Monotone in the block count: more blocks lowers the per-block demand, so
    # it may widen the interleave and can never narrow it. A rule that was not
    # monotone here would make a bigger node plan a narrower walk.
    var prev = 0
    for blocks in range(1, 65):
        var w = schedule_feature_group(m14, 50, blocks)
        assert_true(w >= prev)
        prev = w
    assert_equal(prev, MAX_FEATURE_GROUP)

    # An explicit width still bypasses all of it, blocked or not.
    _set_group(2)
    assert_equal(plan_feature_group(m14, 255, 50, 54), 2)
    _reset_env()


def test_utilization_is_the_number_the_byte_table_cannot_see() raises:
    """The cost side of the width, computed the way the balance rule prices
    it. 13 tasks on 10 cores is two rounds with seven cores idle through the
    second; that is what width 4 would have bought at 50 features, and it is
    why the balance rule exists."""
    _reset_env()
    assert_equal(dispatch_rounds(0, 10), 0)
    assert_equal(dispatch_rounds(13, 10), 2)
    assert_equal(dispatch_rounds(25, 10), 3)
    assert_equal(dispatch_rounds(40, 10), 4)
    assert_equal(dispatch_utilization_percent(0, 10), 0)
    assert_equal(dispatch_utilization_percent(50, 10), 100)
    assert_equal(dispatch_utilization_percent(40, 10), 100)
    assert_equal(dispatch_utilization_percent(25, 10), 83)
    assert_equal(dispatch_utilization_percent(13, 10), 65)
    assert_equal(dispatch_utilization_percent(7, 10), 70)
    assert_equal(dispatch_utilization_percent(4, 10), 40)
    # The guaranteed floor for `n_tasks >= F * cores` is at `F * cores + 1`,
    # one task past a whole number of rounds. On ten cores that is 70% at
    # F = 2 and 82% at F = 4: what the factor buys is real but shallow, and
    # weaker than the 83% the chosen shape happens to get.
    assert_equal(
        dispatch_utilization_percent(TASK_BALANCE_FACTOR * 10 + 1, 10), 70
    )
    assert_equal(dispatch_utilization_percent(41, 10), 82)


def test_the_core_pool_still_moves_the_width_but_says_so() raises:
    """A correction to an earlier claim in this file.

    `MOJOTREES_CPU_CORE_POOL=performance` used to change the interleave as a
    silent side effect, because it changed the task count and a task holding
    one feature could not pair. It still changes the width, now through the
    balance rule and in the stated direction: fewer dispatch cores means fewer
    groups are needed, so a wider group is affordable. Four performance cores
    want 8 groups, `50 // 8 = 6`, which floors to 4, against 2 on all ten
    cores. That is a documented consequence of the rule rather than an
    artefact of the splitter, but it is still a confound in a core-pool A/B,
    and the way to hold the width fixed across that A/B is to set
    `MOJOTREES_CPU_FEATURE_GROUP` explicitly for both arms.
    """
    _reset_env()
    var machine = CpuProfile.synthetic(10, 4)
    assert_equal(plan_feature_group(machine, 32, 50), 2)
    _ = setenv("MOJOTREES_CPU_CORE_POOL", "performance")
    assert_equal(schedule_feature_group(machine, 50), 4)
    assert_equal(plan_feature_group(machine, 32, 50), 4)
    # Pinning the knob holds the width across both arms, which is what a
    # core-pool sweep has to do to be measuring one thing.
    _set_group(2)
    var perf_pinned = plan_feature_group(machine, 32, 50)
    _ = setenv("MOJOTREES_CPU_CORE_POOL", "")
    assert_equal(plan_feature_group(machine, 32, 50), perf_pinned)
    _reset_env()


def test_prints_the_running_machine_plan() raises:
    """A label on the rest of the file rather than an assertion: a width is
    only interesting once you know the profile it was derived against."""
    _reset_env()
    var machine = cpu_profile()
    var plan = derive_accumulation_plan(machine, 50, 50, 255, 1000000, True)
    print("  ", describe_cpu_policy(machine, plan))
    assert_true(is_feature_group_width(plan.group_width))
    assert_equal(plan.group_count, feature_group_count(50, plan.group_width))


# --------------------------------------------------------------------------
# Bit-identity across the ladder
# --------------------------------------------------------------------------


def _sweep_shape(
    n_rows: Int, n_features: Int, n_bins: Int, seed: UInt64
) raises:
    """Every rung, both builders, a full and a strict window, against the
    width-1 reference for the same shape."""
    var rungs = _rungs()
    var data = _make_data(n_rows, n_features, n_bins, seed)
    var grad = _grads(n_rows, seed + 977)
    var hess = _hessians(n_rows, seed + 1213)
    var rows = _row_ids(n_rows)
    var start = 3 if n_rows > 8 else 0
    var count = (n_rows - 7) if n_rows > 8 else n_rows
    var window = List[Int](capacity=count)
    for i in range(start, start + count):
        window.append(rows[i])

    _set_group(1)
    var ref_full = build_histogram(data, grad, hess)
    var ref_all = build_histogram_subset(data, grad, hess, rows)
    var ref_window = build_histogram_subset(data, grad, hess, window)

    var label = String(
        n_rows, "x", n_features, " bins=", n_bins, " width="
    )
    for i in range(len(rungs)):
        _set_group(rungs[i])
        var tag = String(label, rungs[i])
        _assert_identical(
            build_histogram(data, grad, hess), ref_full, String(tag, " full")
        )
        _assert_identical(
            build_histogram_subset(data, grad, hess, rows),
            ref_all,
            String(tag, " subset"),
        )
        _assert_identical(
            build_histogram_subset(data, grad, hess, window),
            ref_window,
            String(tag, " window"),
        )
    _unset_group()


def test_identical_across_the_ladder_on_a_gathered_node() raises:
    """Above `DEFAULT_COMPACT_MIN_ROWS`, so a multi-feature subset build takes
    the gathered pair buffer."""
    _reset_env()
    var features: List[Int] = [1, 3, 8, 17, 50]
    var bins: List[Int] = [4, 7, 32, 255]
    for f in range(len(features)):
        for b in range(len(bins)):
            _sweep_shape(
                311, features[f], bins[b], UInt64(1000 * f + 17 * b + 1)
            )
    _reset_env()


def test_identical_across_the_ladder_on_a_small_node() raises:
    """Below `DEFAULT_COMPACT_MIN_ROWS`, so the gather is skipped and the
    gradients are read through the row ids. That path used to run one feature
    at a time regardless of the policy; it is now interleaved like the other,
    and has to agree with it."""
    _reset_env()
    var features: List[Int] = [1, 3, 8, 17, 50]
    var bins: List[Int] = [4, 7, 32, 255]
    for f in range(len(features)):
        for b in range(len(bins)):
            _sweep_shape(
                37, features[f], bins[b], UInt64(7000 + 100 * f + 13 * b)
            )
    _reset_env()


def test_identical_with_the_gather_forced_either_way() raises:
    """The gather decision and the interleave width are independent now, so
    the four combinations of (gather, width) must all agree."""
    var rungs = _rungs()
    _reset_env()
    var n_rows = 401
    var data = _make_data(n_rows, 17, 63, 4242)
    var grad = _grads(n_rows, 555)
    var hess = _hessians(n_rows, 666)
    var rows = _row_ids(n_rows)

    _set_group(1)
    _set_compact_min_rows("1000000000")
    var reference = build_histogram_subset(data, grad, hess, rows)

    for i in range(len(rungs)):
        _set_group(rungs[i])
        _set_compact_min_rows("1000000000")
        _assert_identical(
            build_histogram_subset(data, grad, hess, rows),
            reference,
            String("no gather width=", rungs[i]),
        )
        _set_compact_min_rows("1")
        _assert_identical(
            build_histogram_subset(data, grad, hess, rows),
            reference,
            String("gathered width=", rungs[i]),
        )
    _reset_env()


def test_identical_at_every_worker_count() raises:
    """The task split is over groups now rather than over features. A task
    still holds whole groups at every worker count, so the summation order is
    still the row order and the bytes still match."""
    var rungs = _rungs()
    _reset_env()
    var n_rows = 257
    var data = _make_data(n_rows, 50, 32, 31337)
    var grad = _grads(n_rows, 8080)
    var hess = _hessians(n_rows, 9090)
    var rows = _row_ids(n_rows)

    _set_group(1)
    _set_workers("1")
    var reference = build_histogram_subset(data, grad, hess, rows)

    var workers: List[String] = ["1", "2", "3", "7", "64", ""]
    for i in range(len(rungs)):
        for w in range(len(workers)):
            _set_group(rungs[i])
            _set_workers(workers[w])
            _assert_identical(
                build_histogram_subset(data, grad, hess, rows),
                reference,
                String("width=", rungs[i], " workers=", workers[w]),
            )
            _assert_identical(
                build_histogram(data, grad, hess),
                reference,
                String("full width=", rungs[i], " workers=", workers[w]),
            )
    _reset_env()


def test_identical_under_feature_subsampling() raises:
    """A feature subset changes which slices are accumulated and which are
    zeroed by the excluded pass, and it makes the group's members
    non-consecutive. Neither may move a value, and the excluded slices must
    still come back zero at every width."""
    var rungs = _rungs()
    _reset_env()
    var n_rows = 289
    var n_features = 23
    var data = _make_data(n_rows, n_features, 17, 5150)
    var grad = _grads(n_rows, 4004)
    var hess = _hessians(n_rows, 3003)
    var rows = _row_ids(n_rows)
    # Non-consecutive, unsorted, and a strict subset: 11 of 23.
    var chosen: List[Int] = [22, 0, 7, 3, 19, 4, 12, 1, 20, 8, 15]

    _set_group(1)
    var reference = build_histogram_subset(data, grad, hess, rows, chosen)
    var reference_full = build_histogram(data, grad, hess, chosen)

    for i in range(len(rungs)):
        _set_group(rungs[i])
        _assert_identical(
            build_histogram_subset(data, grad, hess, rows, chosen),
            reference,
            String("subsampled width=", rungs[i]),
        )
        _assert_identical(
            build_histogram(data, grad, hess, chosen),
            reference_full,
            String("subsampled full width=", rungs[i]),
        )

    # The excluded slices are zero, not stale.
    var active = List[UInt8](capacity=n_features)
    active.resize(n_features, UInt8(0))
    for i in range(len(chosen)):
        active[chosen[i]] = UInt8(1)
    for f in range(n_features):
        if active[f] == UInt8(0):
            for b in range(17):
                assert_equal(reference.grad_at(f * 17 + b), 0.0)
                assert_equal(reference.hess_at(f * 17 + b), 0.0)
                assert_equal(reference.count_at(f * 17 + b), 0)
    _reset_env()


def test_totals_are_conserved_at_every_width() raises:
    """A cross-check that does not depend on the reference build: every
    feature's bins must sum to the node's row count, at every width. This
    catches a tail group that dropped a feature or counted one twice, which
    an all-widths-agree comparison alone would miss if the ladder dispatch
    were wrong in the same way everywhere."""
    var rungs = _rungs()
    _reset_env()
    var n_rows = 133
    var n_features = 17
    var n_bins = 9
    var data = _make_data(n_rows, n_features, n_bins, 99)
    var grad = _grads(n_rows, 12)
    var hess = _hessians(n_rows, 13)
    var rows = _row_ids(n_rows)
    for i in range(len(rungs)):
        _set_group(rungs[i])
        var hist = build_histogram_subset(data, grad, hess, rows)
        for f in range(n_features):
            var total = 0
            for b in range(n_bins):
                total += hist.count_at(f * n_bins + b)
            assert_equal(total, n_rows)
    _reset_env()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
