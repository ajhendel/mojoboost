"""`MOJOTREES_CPU_ROW_BLOCK_AMORTIZE`: that the knob takes, and what it costs.

The block-size floor `row_block_min_rows(bins) = 2 * ROW_BLOCK_AMORTIZE *
bins` is 4,080 rows at 255 bins, so a node blocks only above 8,160 rows. That
constant has never been measured. This file is the switch that lets it be
swept and the gate that proves a sweep arm actually moved the plan.

**Read this before reading the assertions: which invariant is tested and
which is not.**

*Tested.* At one fixed setting of the ratio, the built histogram is
`to_bits()` identical across `MOJOTREES_NUM_WORKERS` 1, 3 and 8, across two
`MOJOTREES_CPU_TASKS_PER_CORE` values an order of magnitude apart, and across
a grain setting that forces the dispatch serial and one that maximizes its
fan-out. The block count is derived from `rows`, `n_bins`, `n_active` and the
ratio and from nothing about the machine, so every one of those arms folds
the same blocks in the same ascending order and must produce the same bytes.
That property is the one the round requires and it is what this knob was
allowed to exist on condition of preserving.

*Deliberately NOT tested, because it is false.* That two different settings
of the ratio agree with each other. They do not and they cannot. The fold
sums block partials in ascending block order, so changing the block **count**
changes the **association** of a Float64 sum, and Float64 addition is not
associative. `test_the_ratio_is_observable_in_the_output` asserts the
opposite -- that two settings differ somewhere on a fixture built to expose
it -- precisely so that the determinism arms above cannot pass vacuously by
running a shape where nothing was ever going to move. Neither order is more
accurate than the other; `histogram.mojo`'s module docstring argues that.

So: bits move **between settings** and never move **within one**. A fit run
at a non-default setting is not comparable cell for cell with a fit that was
not, exactly as `MOJOTREES_CPU_ROW_BLOCKS` already is.

Every comparison here is `to_bits()` or integer equality. There is no
tolerance anywhere in this file.

Every expected number is hand-derived in the comment beside it from
`floor(2 * num * bins / den)`, the byte budget
`MAX_ROW_BLOCK_SCRATCH_BYTES // (active * bins * 3 * 8)`, and
`MAX_ROW_BLOCKS`, so a change to any of the three fails here with the
arithmetic on screen rather than silently re-planning both sides at once.
"""

from std.os import setenv
from std.testing import assert_equal, assert_not_equal, assert_true, TestSuite

from mojotrees.apple_cpu_policy import (
    AccumulationPlan,
    CpuProfile,
    MAX_ROW_BLOCKS,
    ROW_BLOCK_AMORTIZE,
    ResolvedCpuPolicy,
    RowBlockAmortize,
    derive_accumulation_plan,
    env_row_block_amortize,
    max_row_blocks_for_cells,
    parse_row_block_amortize,
    row_block_min_rows,
    row_block_min_rows_at,
)
from mojotrees.binning import bin_equal_width, BinnedMatrix
from mojotrees.histogram import build_histogram_subset, Histogram
from support import _uniform


comptime AMORTIZE_ENV = "MOJOTREES_CPU_ROW_BLOCK_AMORTIZE"


def _clear_env() raises:
    _ = setenv(AMORTIZE_ENV, "")
    _ = setenv("MOJOTREES_CPU_ROW_BLOCKS", "")
    _ = setenv("MOJOTREES_NUM_WORKERS", "")
    _ = setenv("MOJOTREES_PARALLEL_MIN_OPS", "")
    _ = setenv("MOJOTREES_PARALLEL_MIN_TASK_OPS", "")
    _ = setenv("MOJOTREES_CPU_TASKS_PER_CORE", "")
    _ = setenv("MOJOTREES_CPU_CORE_POOL", "")
    _ = setenv("MOJOTREES_CPU_FEATURE_GROUP", "")
    _ = setenv("MOJOTREES_CONST_HESSIAN", "")


def _plan(
    profile: CpuProfile, n_features: Int, n_bins: Int, rows: Int
) raises -> AccumulationPlan:
    """A whole-feature subset build's plan, the call `histogram` makes."""
    return derive_accumulation_plan(
        profile, n_features, n_features, n_bins, rows, True
    )


def _blocks_at(spec: String, n_features: Int, n_bins: Int, rows: Int) raises -> Int:
    """The resolved block count at one setting of the ratio."""
    _ = setenv(AMORTIZE_ENV, spec)
    var b = _plan(CpuProfile.detect(), n_features, n_bins, rows).row_blocks
    _ = setenv(AMORTIZE_ENV, "")
    return b


# ---------------------------------------------------------------------------
# Unset is what shipped
# ---------------------------------------------------------------------------


def test_unset_reproduces_the_shipped_rule_exactly() raises:
    """Default-unset must be byte-identical to today, and this is the proof.

    Every literal below is the number this repository planned before the knob
    existed; `tests/test_histogram_reference.mojo` asserts the same 4,080, the
    same 54, and the same 9-blocks-of-4445, which is the cross-check that this
    file did not simply restate its own arithmetic.
    """
    _clear_env()
    var m = CpuProfile.detect()

    # 2 * 8 * bins, floored, at three bin counts.
    assert_equal(row_block_min_rows(255), 4080)
    assert_equal(row_block_min_rows(32), 512)
    assert_equal(row_block_min_rows(2), 32)
    # A degenerate bin count still cannot divide by zero downstream.
    assert_equal(row_block_min_rows(0), 1)

    # The resolved ratio unset is 8/1, and reads as recognized.
    var a = env_row_block_amortize()
    assert_equal(a.num, ROW_BLOCK_AMORTIZE)
    assert_equal(a.den, 1)
    assert_true(a.recognized)
    assert_equal(row_block_min_rows_at(a, 255), 4080)
    assert_equal(row_block_min_rows_at(RowBlockAmortize.default(), 255), 4080)

    # And the plans, at the four things that can bind, unchanged.
    assert_equal(_plan(m, 50, 255, 8159).row_blocks, 1)  # 8159 // 4080 == 1
    var two = _plan(m, 50, 255, 8160)  # 8160 // 4080 == 2
    assert_equal(two.row_blocks, 2)
    assert_equal(two.block_rows, 4080)
    var med = _plan(m, 50, 255, 40_000)  # 40000 // 4080 == 9
    assert_equal(med.row_blocks, 9)
    assert_equal(med.block_rows, 4445)  # ceil(40000 / 9)
    # 50 * 255 = 12750 cells, 12750 * 24 = 306000 bytes a block,
    # 16777216 // 306000 == 54, against a floor asking for 1000000 // 4080 == 245.
    assert_equal(_plan(m, 50, 255, 1_000_000).row_blocks, 54)
    # 4 features of 8 bins is 32 cells, 768 bytes: the byte budget allows
    # 21845 and the floor asks 4000000 // 128 == 31250, so the hard ceiling.
    assert_equal(_plan(m, 4, 8, 4_000_000).row_blocks, MAX_ROW_BLOCKS)


def test_setting_the_default_explicitly_changes_nothing() raises:
    """`8` and `8/1` are the shipped ratio spelled out, and must plan
    identically to unset. Without this, "the default is unchanged" would rest
    on the parser never having been exercised on the default path."""
    _clear_env()
    var m = CpuProfile.detect()
    var rows_l = [8_159, 8_160, 40_000, 50_000, 1_000_000]
    for k in range(len(rows_l)):
        _ = setenv(AMORTIZE_ENV, "")
        var unset = _plan(m, 50, 255, rows_l[k])
        _ = setenv(AMORTIZE_ENV, "8")
        var eight = _plan(m, 50, 255, rows_l[k])
        _ = setenv(AMORTIZE_ENV, "8/1")
        var ratio = _plan(m, 50, 255, rows_l[k])
        assert_equal(unset.row_blocks, eight.row_blocks)
        assert_equal(unset.row_blocks, ratio.row_blocks)
        assert_equal(unset.block_rows, eight.block_rows)
        assert_equal(unset.block_rows, ratio.block_rows)
        assert_equal(unset.block_cells, eight.block_cells)
        assert_equal(unset.block_scratch_floats(), ratio.block_scratch_floats())
    _clear_env()


# ---------------------------------------------------------------------------
# The sweep table. This is the gate that proves the knob took.
# ---------------------------------------------------------------------------


def test_the_knob_changes_the_resolved_block_count() raises:
    """The sweep, at the two shapes the measurement window will run.

    50 features, 255 bins, so `cells = 12750` and the scratch budget allows
    `16777216 // (12750 * 24) == 54` blocks; `MAX_ROW_BLOCKS` is 64 and never
    binds here.

    At 50,000 rows the ratio is live across the whole sweep, 1 block to 54.
    At 1,000,000 rows it is **not**: the byte budget answers 54 for every
    ratio at or below 32, so the amortization floor is not what is choosing
    there. That asymmetry is the point of asserting both shapes and it is the
    single most useful thing this table says.
    """
    _clear_env()

    var specs = ["1/8", "1", "2", "4", "8", "16", "32", "64", "128"]
    #  min_rows = floor(2 * num * 255 / den):
    #      1/8 -> 63, 1 -> 510, 2 -> 1020, 4 -> 2040, 8 -> 4080,
    #      16 -> 8160, 32 -> 16320, 64 -> 32640, 128 -> 65280
    #
    #  50,000 rows: floor asks 793, 98, 49, 24, 12, 6, 3, 1, 0
    #               capped at 54, so:
    var at_50k = [54, 54, 49, 24, 12, 6, 3, 1, 1]
    #  1,000,000 rows: floor asks 15873, 1960, 980, 490, 245, 122, 61, 30, 15
    #                  capped at 54, so:
    var at_1m = [54, 54, 54, 54, 54, 54, 54, 30, 15]

    for k in range(len(specs)):
        assert_equal(_blocks_at(specs[k], 50, 255, 50_000), at_50k[k])
        assert_equal(_blocks_at(specs[k], 50, 255, 1_000_000), at_1m[k])

    # The knob demonstrably took: adjacent arms at 50k are different plans.
    # Stated as inequalities as well as literals, because a table of literals
    # that all happened to be equal would still pass its own assertions.
    assert_not_equal(
        _blocks_at("4", 50, 255, 50_000), _blocks_at("8", 50, 255, 50_000)
    )
    assert_not_equal(
        _blocks_at("8", 50, 255, 50_000), _blocks_at("16", 50, 255, 50_000)
    )
    assert_not_equal(
        _blocks_at("32", 50, 255, 50_000), _blocks_at("64", 50, 255, 50_000)
    )
    # And at 1M the byte budget, not the ratio, is what answers: a 128-fold
    # sweep of the ratio moves nothing at all until the ratio passes 32.
    assert_equal(
        _blocks_at("1/8", 50, 255, 1_000_000),
        _blocks_at("32", 50, 255, 1_000_000),
    )
    assert_equal(max_row_blocks_for_cells(50 * 255), 54)
    _clear_env()


def test_the_sweep_moves_scratch_and_fold_bytes() raises:
    """What each setting actually buys or spends, as bytes rather than words.

    Private scratch is `blocks * cells * 3 planes * 8 bytes` and the fold
    reads every partial cell once and writes the output once, three planes
    each, so its traffic is `(blocks + 1) * cells * 24`. Both are linear in
    the block count, which is why the sweep is a trade and not a free win: the
    accumulate work is `active * rows` at every setting and does not move.
    """
    _clear_env()
    var m = CpuProfile.detect()
    var cells = 50 * 255  # 12750

    _ = setenv(AMORTIZE_ENV, "8")
    var default_50k = _plan(m, 50, 255, 50_000)
    assert_equal(default_50k.row_blocks, 12)
    assert_equal(default_50k.block_cells, cells)
    # 12 * 12750 * 3 = 459,000 Float64 = 3,672,000 bytes of private scratch.
    assert_equal(default_50k.block_scratch_floats(), 459_000)
    # fold_ops = 3 * (12 + 1) * 12750 = 497,250 cell-plane touches.
    assert_equal(default_50k.fold_ops, 497_250)

    _ = setenv(AMORTIZE_ENV, "32")
    var coarse = _plan(m, 50, 255, 50_000)
    assert_equal(coarse.row_blocks, 3)
    assert_equal(coarse.block_scratch_floats(), 114_750)  # 3 * 12750 * 3
    assert_equal(coarse.fold_ops, 153_000)  # 3 * 4 * 12750

    _ = setenv(AMORTIZE_ENV, "1")
    var fine = _plan(m, 50, 255, 50_000)
    assert_equal(fine.row_blocks, 54)
    assert_equal(fine.block_scratch_floats(), 2_065_500)  # 54 * 12750 * 3
    assert_equal(fine.fold_ops, 2_103_750)  # 3 * 55 * 12750

    # The accumulate term is the same at all three: only the overhead moved.
    # block_ops = active * rows + blocks * cells, so subtracting the zeroing
    # term leaves an identical 50 * 50000.
    assert_equal(default_50k.block_ops - 12 * cells, 50 * 50_000)
    assert_equal(coarse.block_ops - 3 * cells, 50 * 50_000)
    assert_equal(fine.block_ops - 54 * cells, 50 * 50_000)
    _clear_env()


# ---------------------------------------------------------------------------
# The property the knob had to preserve: workload-only
# ---------------------------------------------------------------------------


def test_block_count_is_workload_only_at_every_setting() raises:
    """No setting of the ratio may make the block count machine-dependent.

    Two synthetic profiles a hundred cores apart, and the four environment
    variables that change the schedule, at every arm of the sweep. The block
    count, the chunk and the cell count must be identical throughout. The
    feature-group width is *allowed* to move with the machine and is asserted
    to, so this is not vacuously true of everything the planner returns.
    """
    _clear_env()
    var one_core = CpuProfile.synthetic(1, 1)
    var many = CpuProfile.synthetic(128, 96)
    var specs = ["1/8", "1", "2", "4", "8", "16", "32", "64"]

    for k in range(len(specs)):
        _ = setenv(AMORTIZE_ENV, specs[k])
        var a = _plan(one_core, 50, 255, 200_000)
        var b = _plan(many, 50, 255, 200_000)
        assert_equal(a.row_blocks, b.row_blocks)
        assert_equal(a.block_rows, b.block_rows)
        assert_equal(a.block_cells, b.block_cells)
        # The width is machine-dependent and the block count is not. Checked
        # on a node of 60 rows, which is below the smallest block minimum any
        # swept ratio produces and so blocks 1 way at every arm: with many
        # blocks the balance rule stops binding and both machines land on the
        # L1 clamp, which would make this vacuous for the opposite reason.
        var a1 = _plan(one_core, 50, 255, 60)
        var b1 = _plan(many, 50, 255, 60)
        assert_equal(a1.row_blocks, 1)
        assert_equal(b1.row_blocks, 1)
        assert_not_equal(a1.group_width, b1.group_width)

    # And the schedule variables cannot reach it either. One arm is enough to
    # be worth stating; it is checked at a setting that blocks 24 ways so a
    # collapse to 1 could not hide inside it.
    _ = setenv(AMORTIZE_ENV, "4")
    var base = _plan(CpuProfile.detect(), 50, 255, 50_000)
    assert_equal(base.row_blocks, 24)
    var workers = ["1", "3", "8", "", "", ""]
    var per_core = ["", "", "", "1", "16", ""]
    var pool = ["", "", "", "", "", "performance"]
    for k in range(len(workers)):
        _ = setenv("MOJOTREES_NUM_WORKERS", workers[k])
        _ = setenv("MOJOTREES_CPU_TASKS_PER_CORE", per_core[k])
        _ = setenv("MOJOTREES_CPU_CORE_POOL", pool[k])
        var arm = _plan(CpuProfile.detect(), 50, 255, 50_000)
        assert_equal(arm.row_blocks, base.row_blocks)
        assert_equal(arm.block_rows, base.block_rows)
    _clear_env()


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def _planning_raises(spec: String) raises -> Bool:
    _ = setenv(AMORTIZE_ENV, spec)
    var raised = False
    try:
        _ = _plan(CpuProfile.detect(), 50, 255, 50_000)
    except:
        raised = True
    _ = setenv(AMORTIZE_ENV, "")
    return raised


def test_a_refused_value_stops_the_plan_rather_than_relabelling_it() raises:
    """An arm that asked for something unparsable must fail, not run the
    default under the requested label. This project has already discarded two
    results to a silently substituted setting."""
    _clear_env()
    var bad = ["abc", "0", "-2", "8/0", "8/-1", "1/x", "/", "8/", "1/2/3", ""]
    #  The last entry is empty, which is *valid* (it means unset), so it is
    #  the negative control for this list.
    for k in range(len(bad) - 1):
        assert_true(_planning_raises(bad[k]))
        assert_true(not parse_row_block_amortize(bad[k]).recognized)
    assert_true(not _planning_raises(bad[len(bad) - 1]))

    var good = ["1", "8", "64", "1/8", "3/4"]
    for k in range(len(good)):
        assert_true(parse_row_block_amortize(good[k]).recognized)
        assert_true(not _planning_raises(good[k]))

    # The snapshot refuses at the one point it is taken, too.
    _ = setenv(AMORTIZE_ENV, "nonsense")
    var snapshot_raised = False
    try:
        _ = ResolvedCpuPolicy.resolve()
    except:
        snapshot_raised = True
    assert_true(snapshot_raised)
    _clear_env()

    # A refused value carries the default, so a non-raising caller still
    # plans the shipped rule rather than a zero-row block.
    var refused = parse_row_block_amortize(String("abc"))
    assert_equal(row_block_min_rows_at(refused, 255), 4080)


def test_the_ratio_reaches_the_region_the_question_is_about() raises:
    """LightGBM's comparable floor at 255 bins is 77 rows and ours is 4,080.

    The two are not the same quantity -- theirs is a work-stealing chunk whose
    private histogram count comes from the thread count, ours is the private
    histogram count itself -- so it is not adopted. But the sweep has to be
    able to *reach* that region, or it cannot say which side of it is right.
    An integer-only ratio bottoms out at 510 rows and could not.
    """
    _clear_env()
    var eighth = parse_row_block_amortize(String("1/8"))
    assert_true(eighth.recognized)
    assert_equal(eighth.num, 1)
    assert_equal(eighth.den, 8)
    assert_equal(row_block_min_rows_at(eighth, 255), 63)  # (2 * 255) // 8
    # The integer floor, for contrast: the smallest integer ratio is 510 rows.
    assert_equal(
        row_block_min_rows_at(parse_row_block_amortize(String("1")), 255), 510
    )
    # Rounding is toward zero, so a smaller ratio never yields a larger block.
    assert_equal(row_block_min_rows_at(eighth, 3), 1)  # (2 * 3) // 8 == 0 -> 1
    _clear_env()


# ---------------------------------------------------------------------------
# The output: identical within a setting, different between settings
# ---------------------------------------------------------------------------
#
# One fixture. 30,000 rows of 12 features at 32 bins, accumulating a 15,000
# row subset (every second row):
#
#   cells         = 12 * 32                     = 384
#   byte budget   = 16777216 // (384 * 24)      = 1820, never binds
#   MAX_ROW_BLOCKS= 64,                           binds only below ratio 4
#
#   ratio  8: min_rows = 512,  15000 // 512  = 29  -> 29 blocks, chunk 518
#   ratio 32: min_rows = 2048, 15000 // 2048 = 7   ->  7 blocks, chunk 2143
#
# Both block, and 29 != 7, so the two settings are two different summation
# orders on the same rows.

comptime BLK_ROWS = 30_000
comptime BLK_FEATURES = 12
comptime BLK_BINS = 32
comptime BLK_SUB = 15_000
comptime DEFAULT_BLOCKS = 29
comptime COARSE_BLOCKS = 7


def _pow2(e: Int) -> Float64:
    """`2 ** e` by exact doublings, so no library rounding enters."""
    var v = 1.0
    if e >= 0:
        for _ in range(e):
            v *= 2.0
    else:
        for _ in range(-e):
            v *= 0.5
    return v


def _grads_spread(n_rows: Int, seed: UInt64) -> List[Float64]:
    """Gradients spanning forty-one binary orders of magnitude.

    A per-row derivative is rounded to Float32 before it is accumulated, so a
    bin's Float64 sum of a few hundred uniform values is frequently *exact*
    and reassociating it cannot change it. A fixture built from uniforms would
    make `test_the_ratio_is_observable_in_the_output` quietly vacuous. Scaling
    each row by a power of two cycling over `2^-20 .. 2^20` forces the running
    sum past 53 bits of dynamic range and makes the grouping visible again.
    Every value is still exactly representable.
    """
    var g = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        g.append(
            (2.0 * _uniform(seed + UInt64(r)) - 1.0) * _pow2(-20 + (r % 41))
        )
    return g^


def _hessians(n_rows: Int, seed: UInt64) -> List[Float64]:
    var h = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        h.append(_uniform(seed + UInt64(r)) + 0.01)
    return h^


def _fixture() raises -> BinnedMatrix:
    var features = List[Float64](capacity=BLK_ROWS * BLK_FEATURES)
    for k in range(BLK_ROWS * BLK_FEATURES):
        features.append(_uniform(UInt64(97) + UInt64(k)))
    return bin_equal_width(features, BLK_ROWS, BLK_FEATURES, BLK_BINS)


def _even_rows() -> List[Int]:
    var rows = List[Int]()
    for r in range(0, BLK_ROWS, 2):
        rows.append(r)
    return rows^


def _assert_same(a: Histogram, b: Histogram) raises:
    assert_equal(a.n_features, b.n_features)
    assert_equal(a.n_bins, b.n_bins)
    for i in range(a.n_features * a.n_bins):
        assert_equal(a.grad_at(i).to_bits(), b.grad_at(i).to_bits())
        assert_equal(a.hess_at(i).to_bits(), b.hess_at(i).to_bits())
        assert_equal(a.count_at(i), b.count_at(i))


def _differs_somewhere(a: Histogram, b: Histogram) raises -> Bool:
    for i in range(a.n_features * a.n_bins):
        if a.grad_at(i).to_bits() != b.grad_at(i).to_bits():
            return True
    return False


def test_the_ratio_is_observable_in_the_output() raises:
    """The gate. Two settings, two block counts, two different histograms.

    Without this, the determinism test below could pass on a shape where the
    ratio never changed anything, and would then be asserting nothing. Note
    what is asserted and what is not: that the settings *differ*, never that
    they agree. They cannot agree, because a fold over 29 blocks and a fold
    over 7 is a different association of the same Float64 addends.

    The count plane is integer addition, which is associative, so it must be
    identical across the two settings. That is asserted too, and it is what
    makes the gradient difference a statement about associativity rather than
    about the two arms having accumulated different rows.
    """
    _clear_env()
    var data = _fixture()
    var grad = _grads_spread(BLK_ROWS, UInt64(51_000_000))
    var hess = _hessians(BLK_ROWS, UInt64(52_000_000))
    var rows = _even_rows()
    assert_equal(len(rows), BLK_SUB)
    var m = CpuProfile.detect()

    _ = setenv(AMORTIZE_ENV, "8")
    var plan_default = _plan(m, BLK_FEATURES, BLK_BINS, BLK_SUB)
    assert_true(plan_default.blocked())
    assert_equal(plan_default.row_blocks, DEFAULT_BLOCKS)
    assert_equal(plan_default.block_rows, 518)  # ceil(15000 / 29)
    var built_default = build_histogram_subset(data, grad, hess, rows)

    _ = setenv(AMORTIZE_ENV, "32")
    var plan_coarse = _plan(m, BLK_FEATURES, BLK_BINS, BLK_SUB)
    assert_true(plan_coarse.blocked())
    assert_equal(plan_coarse.row_blocks, COARSE_BLOCKS)
    assert_equal(plan_coarse.block_rows, 2143)  # ceil(15000 / 7)
    var built_coarse = build_histogram_subset(data, grad, hess, rows)

    # Bits moved between the two settings. This is the invariant this file
    # asserts *is violated* across settings, and it is deliberate.
    assert_true(_differs_somewhere(built_default, built_coarse))

    # The count plane did not move, because integer addition is associative.
    for i in range(BLK_FEATURES * BLK_BINS):
        assert_equal(built_default.count_at(i), built_coarse.count_at(i))
    _clear_env()


def test_bits_identical_across_workers_at_a_fixed_setting() raises:
    """The invariant this knob had to preserve, at a non-default setting.

    Six arms at ratio 32: three worker counts, two fan-out ceilings an order
    of magnitude apart, and a grain that forces the dispatch serial. Every one
    must be bit-identical, because none of them reaches the block count -- the
    ratio and the workload shape decide that between them and the machine is
    never asked.

    Two gates in front of the arms, so a pass cannot be vacuous:
    the plan must actually block at 7, and the 7-block build must differ from
    the 29-block one on this fixture.
    """
    _clear_env()
    var data = _fixture()
    var grad = _grads_spread(BLK_ROWS, UInt64(53_000_000))
    var hess = _hessians(BLK_ROWS, UInt64(54_000_000))
    var rows = _even_rows()

    _ = setenv(AMORTIZE_ENV, "8")
    var at_default = build_histogram_subset(data, grad, hess, rows)

    _ = setenv(AMORTIZE_ENV, "32")
    var plan = _plan(CpuProfile.detect(), BLK_FEATURES, BLK_BINS, BLK_SUB)
    assert_true(plan.blocked())
    assert_equal(plan.row_blocks, COARSE_BLOCKS)
    var expected = build_histogram_subset(data, grad, hess, rows)
    assert_true(_differs_somewhere(at_default, expected))

    #   workers 1 / 3 / 8            the three the round requires
    #   tasks_per_core 1 and 16      the fan-out ceiling moved 16x
    #   min_ops 1e9                  auto, forced back to serial
    var workers = ["1", "3", "8", "", "", ""]
    var per_core = ["", "", "", "1", "16", ""]
    var min_ops = ["", "", "", "", "", "1000000000"]
    for k in range(len(workers)):
        _ = setenv("MOJOTREES_NUM_WORKERS", workers[k])
        _ = setenv("MOJOTREES_CPU_TASKS_PER_CORE", per_core[k])
        _ = setenv("MOJOTREES_PARALLEL_MIN_OPS", min_ops[k])
        # The block count is unmoved by any of them, and the bytes with it.
        var arm_plan = _plan(CpuProfile.detect(), BLK_FEATURES, BLK_BINS, BLK_SUB)
        assert_equal(arm_plan.row_blocks, COARSE_BLOCKS)
        _assert_same(expected, build_histogram_subset(data, grad, hess, rows))
    _clear_env()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
