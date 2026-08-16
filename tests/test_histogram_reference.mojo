"""Serial-reference equivalence for the CPU histogram kernels.

The optimized builders (pointer stores, SIMD subtraction, per-feature
multicore) must be bit-identical to a plain List-indexing serial reference.
There are now two references, because there are two accumulation kernels, and
which one a build takes is a *value* decision rather than a scheduling one:

- The **feature-partition** kernel sums each feature's bins over the node's
  rows in one ascending walk inside one task. `_reference_subset_flat` is that
  order written out with `List` indexing.
- The **row-blocked** kernel cuts the node's rows into contiguous ascending
  blocks, gives each block a private histogram, and folds the partials in
  ascending block order. `_reference_subset_blocked` is *that* order written
  out, block partials first and the fold started from block 0 so the leading
  `0.0 +` of a naive accumulation cannot hide a signed zero.

`apple_cpu_policy.derive_accumulation_plan` decides between them, from the row
count, the bin count and the active feature count alone, so `_reference_subset`
asks it the same question the builder asks and follows the same answer. That
is deliberate: a reference that always modelled the flat order would have
turned every blocked build into a failure, and a reference that always
modelled the blocked one would have stopped checking the kernel that runs on
every small node.

Shapes are chosen to be odd with respect to any plausible SIMD width
(NEON, AVX2, AVX-512) and to fall on both sides of the multicore
threshold, so both the serial and sync_parallelize paths are covered on
every architecture CI runs on. The row-blocked tests additionally state their
expected block count and chunk as literals, computed by hand from
`row_block_min_rows`, so a change to that rule fails here with the arithmetic
in front of the reader instead of silently re-planning both sides at once.

Every comparison is `to_bits()` or integer equality. There is no tolerance
anywhere in this file.
"""

from std.os import setenv
from std.testing import assert_equal, assert_not_equal, assert_true, TestSuite

from mojotrees.apple_cpu_policy import (
    AccumulationPlan,
    CpuProfile,
    MAX_ROW_BLOCKS,
    derive_accumulation_plan,
    row_block_min_rows,
)
from mojotrees.binning import bin_equal_width, BinnedMatrix
from mojotrees.histogram import (
    build_histogram,
    build_histogram_into_scratch,
    build_histogram_subset,
    build_histogram_subset_into_scratch,
    cnt_factor,
    CONSTANT_HESSIAN,
    derived_count,
    score_t,
    subtract_histogram,
    Histogram,
)
from mojotrees.parallel import PARALLEL_MIN_OPS
from support import _uniform


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


def _pow2(e: Int) -> Float64:
    """`2 ** e`, built by exact doublings so no library rounding enters."""
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

    `_grads` is uniform in [-1, 1] and, now that a per-row derivative is
    rounded to Float32 before it is accumulated, a bin's Float64 sum of a few
    hundred such values is frequently *exact*: 24 significand bits plus the
    handful of carry bits a few hundred addends need still fits inside 53, so
    reassociating the sum cannot change it. That is a real and welcome
    property of the Float32 narrowing, and it is also a hazard for any test
    whose point is that two summation orders differ -- the guard would go
    quietly vacuous rather than fail.

    So the fixtures that must *distinguish* orders use this instead: each row
    is scaled by a power of two cycling over `2^-20 .. 2^20`, which forces the
    running sum past 53 bits of dynamic range and makes the grouping visible
    again. Every value is still exactly representable, so nothing here depends
    on a rounding this file cannot state.
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


def _constant(n_rows: Int, value: Float64) -> List[Float64]:
    var h = List[Float64](capacity=n_rows)
    h.resize(n_rows, value)
    return h^


def _stride_rows(n_rows: Int, stride: Int) -> List[Int]:
    var rows = List[Int]()
    for r in range(0, n_rows, stride):
        rows.append(r)
    return rows^


def _subset_plan(data: BinnedMatrix, n_sub: Int) raises -> AccumulationPlan:
    """The plan a whole-feature subset build of `n_sub` rows would take.

    The same call `histogram._accumulate_subset` makes, through the same
    unresolved-sentinel path, so the reference below and the builder cannot
    disagree about the block count. Note what it is *not* given: no worker
    count, no task count. The block count is derived from the shape.
    """
    return derive_accumulation_plan(
        CpuProfile.detect(),
        data.n_features,
        data.n_features,
        data.n_bins,
        n_sub,
        True,
    )


def _full_plan(data: BinnedMatrix) raises -> AccumulationPlan:
    """The plan a whole-dataset build takes. `rows_are_indirect` is False, and
    since both builders block on one rule that no longer changes the block
    count -- which is the point."""
    return derive_accumulation_plan(
        CpuProfile.detect(),
        data.n_features,
        data.n_features,
        data.n_bins,
        data.n_rows,
        False,
    )


def _all_rows(n_rows: Int) -> List[Int]:
    var rows = List[Int](capacity=n_rows)
    for r in range(n_rows):
        rows.append(r)
    return rows^


def _reference_full(
    data: BinnedMatrix, grad: List[Float64], hess: List[Float64]
) raises -> Histogram:
    """Whichever order the planner selects for a whole-dataset build.

    The whole-dataset builder used to be excluded from row blocking, and this
    reference used to model the flat order unconditionally. Both changed
    together: the two builders now block on one plan, so growing on a bag and
    growing on the dataset of those rows are the same sequence of Float64
    additions again.
    """
    var plan = _full_plan(data)
    if plan.blocked():
        return _reference_subset_blocked(
            data,
            grad,
            hess,
            _all_rows(data.n_rows),
            plan.row_blocks,
            plan.block_rows,
        )
    return _reference_full_flat(data, grad, hess)


def _reference_full_flat(
    data: BinnedMatrix, grad: List[Float64], hess: List[Float64]
) raises -> Histogram:
    var size = data.n_features * data.n_bins
    var g = List[Float64](capacity=size)
    g.resize(size, 0.0)
    var h = List[Float64](capacity=size)
    h.resize(size, 0.0)
    var c = List[Int](capacity=size)
    c.resize(size, 0)
    for f in range(data.n_features):
        for r in range(data.n_rows):
            var b = f * data.n_bins + data.bin_at(r, f)
            g[b] += score_t(grad[r])
            h[b] += score_t(hess[r])
            c[b] += 1
    return Histogram.from_planes(g^, h^, c^, data.n_features, data.n_bins)


def _reference_subset_flat(
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    rows: List[Int],
) raises -> Histogram:
    """The unblocked order: one ascending walk of the node's rows per
    feature."""
    var size = data.n_features * data.n_bins
    var g = List[Float64](capacity=size)
    g.resize(size, 0.0)
    var h = List[Float64](capacity=size)
    h.resize(size, 0.0)
    var c = List[Int](capacity=size)
    c.resize(size, 0)
    for f in range(data.n_features):
        for i in range(len(rows)):
            var r = rows[i]
            var b = f * data.n_bins + data.bin_at(r, f)
            g[b] += score_t(grad[r])
            h[b] += score_t(hess[r])
            c[b] += 1
    return Histogram.from_planes(g^, h^, c^, data.n_features, data.n_bins)


def _reference_subset_blocked(
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    rows: List[Int],
    blocks: Int,
    chunk: Int,
) raises -> Histogram:
    """The blocked order: per-block partials, then a fold in ascending block
    order started from block 0.

    Started from block 0 rather than from a zero accumulator on purpose. The
    kernel's fold loads block 0 and adds blocks 1.. onto it, and `0.0 + x` is
    `x` for every Float64 except `-0.0`, so a reference that began at zero
    would agree everywhere except on a cell whose first block summed to
    negative zero -- which is precisely the kind of disagreement a reference is
    supposed to catch rather than to smooth over.
    """
    var nb = data.n_bins
    var nf = data.n_features
    var size = nf * nb
    var g = List[Float64](capacity=size)
    g.resize(size, 0.0)
    var h = List[Float64](capacity=size)
    h.resize(size, 0.0)
    var c = List[Int](capacity=size)
    c.resize(size, 0)

    var pg = List[Float64](capacity=blocks * nb)
    var ph = List[Float64](capacity=blocks * nb)
    var pc = List[Int](capacity=blocks * nb)
    pg.resize(blocks * nb, 0.0)
    ph.resize(blocks * nb, 0.0)
    pc.resize(blocks * nb, 0)

    for f in range(nf):
        for i in range(blocks * nb):
            pg[i] = 0.0
            ph[i] = 0.0
            pc[i] = 0
        for blk in range(blocks):
            var r0 = blk * chunk
            var r1 = r0 + chunk
            if r1 > len(rows):
                r1 = len(rows)
            for i in range(r0, r1):
                var r = rows[i]
                var b = blk * nb + data.bin_at(r, f)
                pg[b] += score_t(grad[r])
                ph[b] += score_t(hess[r])
                pc[b] += 1
        for b in range(nb):
            var sg = pg[b]
            var sh = ph[b]
            var sc = pc[b]
            for blk in range(1, blocks):
                sg += pg[blk * nb + b]
                sh += ph[blk * nb + b]
                sc += pc[blk * nb + b]
            g[f * nb + b] = sg
            h[f * nb + b] = sh
            c[f * nb + b] = sc
    return Histogram.from_planes(g^, h^, c^, nf, nb)


def _reference_subset(
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    rows: List[Int],
) raises -> Histogram:
    """Whichever order the planner selects for this shape."""
    var plan = _subset_plan(data, len(rows))
    if plan.blocked():
        return _reference_subset_blocked(
            data, grad, hess, rows, plan.row_blocks, plan.block_rows
        )
    return _reference_subset_flat(data, grad, hess, rows)


def _assert_same(a: Histogram, b: Histogram) raises:
    assert_equal(a.n_features, b.n_features)
    assert_equal(a.n_bins, b.n_bins)
    for i in range(a.n_features * a.n_bins):
        assert_equal(a.grad_at(i).to_bits(), b.grad_at(i).to_bits())
        assert_equal(a.hess_at(i).to_bits(), b.hess_at(i).to_bits())
        assert_equal(a.count_at(i), b.count_at(i))


def _differs_somewhere(a: Histogram, b: Histogram) raises -> Bool:
    """Whether two histograms disagree in any gradient cell, by bits.

    Used to prove a fixture actually distinguishes two summation orders. A
    test that asserts "the blocked build equals the blocked reference" proves
    nothing unless the blocked reference is known to differ from the flat one
    on this data, because otherwise both references agree and the assertion
    passes whichever kernel ran.
    """
    for i in range(a.n_features * a.n_bins):
        if a.grad_at(i).to_bits() != b.grad_at(i).to_bits():
            return True
    return False


def _clear_env() raises:
    _ = setenv("MOJOTREES_NUM_WORKERS", "")
    _ = setenv("MOJOTREES_PARALLEL_MIN_OPS", "")
    _ = setenv("MOJOTREES_PARALLEL_MIN_TASK_OPS", "")
    _ = setenv("MOJOTREES_CPU_TASKS_PER_CORE", "")
    _ = setenv("MOJOTREES_CPU_FEATURE_GROUP", "")
    _ = setenv("MOJOTREES_CPU_ROW_BLOCKS", "")
    _ = setenv("MOJOTREES_CPU_ROW_BLOCK_AMORTIZE", "")
    _ = setenv("MOJOTREES_CONST_HESSIAN", "")


# ---------------------------------------------------------------------------
# The pre-existing contract: both kernels against a plain serial reference
# ---------------------------------------------------------------------------


def test_full_matches_reference_serial_odd_shapes() raises:
    var rows_l = [101, 997, 1023, 511]
    var feats_l = [1, 3, 5, 7]
    var bins_l = [2, 2, 17, 3]
    for k in range(len(rows_l)):
        var n_rows = rows_l[k]
        var n_features = feats_l[k]
        # Guard: these shapes must exercise the serial path.
        assert_true(n_features * n_rows < PARALLEL_MIN_OPS)
        var data = _make_data(n_rows, n_features, bins_l[k], UInt64(1000 * k))
        var grad = _grads(n_rows, UInt64(7_000_000 + 1000 * k))
        var hess = _hessians(n_rows, UInt64(8_000_000 + 1000 * k))
        _assert_same(
            _reference_full(data, grad, hess),
            build_histogram(data, grad, hess),
        )


def test_full_matches_reference_parallel_odd_shape() raises:
    var n_rows = 6247
    var n_features = 21
    # Guard: this shape must exercise the sync_parallelize path.
    assert_true(n_features * n_rows >= PARALLEL_MIN_OPS)
    var data = _make_data(n_rows, n_features, 255, UInt64(1))
    var grad = _grads(n_rows, UInt64(9_000_000))
    var hess = _hessians(n_rows, UInt64(10_000_000))
    _assert_same(
        _reference_full(data, grad, hess),
        build_histogram(data, grad, hess),
    )


def test_both_builders_block_on_the_same_plan() raises:
    """The whole-dataset builder and the subset builder must plan the same
    blocks for the same rows.

    This is a correctness property and not a speed one. While only the subset
    builder blocked, `test_bagged_tree_equals_tree_on_subset_dataset` was
    false by four ulp on a leaf value: the bagged tree reached the subset
    builder and folded row blocks while its reference summed flat. Asserted on
    the planner, which is where the decision is made.
    """
    _clear_env()
    var m = CpuProfile.detect()
    var indirect = derive_accumulation_plan(m, 50, 50, 255, 1_000_000, True)
    var direct = derive_accumulation_plan(m, 50, 50, 255, 1_000_000, False)
    assert_true(indirect.row_blocks > 1)
    assert_equal(direct.row_blocks, indirect.row_blocks)
    assert_equal(direct.block_rows, indirect.block_rows)
    assert_equal(direct.block_cells, indirect.block_cells)
    assert_equal(direct.fold_ops, indirect.fold_ops)


def test_full_and_subset_builders_agree_bit_for_bit() raises:
    """The same rows through both builders, to the bit.

    The fixture is chosen so the plan blocks -- asserted, not assumed -- and
    so the blocked order is *distinguishable* from the flat one on this data,
    which `_differs_somewhere` establishes. Without that second check the
    assertion would pass whichever kernel ran.
    """
    _clear_env()
    var n_rows = 20_000
    var n_features = 6
    var data = _make_data(n_rows, n_features, 32, UInt64(4242))
    var grad = _grads_spread(n_rows, UInt64(20_000_001))
    var hess = _hessians(n_rows, UInt64(20_000_002))
    var plan = _full_plan(data)
    assert_true(plan.blocked())
    assert_equal(plan.row_blocks, _subset_plan(data, n_rows).row_blocks)

    var rows = _all_rows(n_rows)
    var flat = _reference_subset_flat(data, grad, hess, rows)
    var blocked = _reference_subset_blocked(
        data, grad, hess, rows, plan.row_blocks, plan.block_rows
    )
    assert_true(_differs_somewhere(flat, blocked))

    var full = build_histogram(data, grad, hess)
    var subset = build_histogram_subset(data, grad, hess, rows)
    _assert_same(full, subset)
    _assert_same(full, blocked)


def test_subset_matches_reference_serial_and_parallel() raises:
    var n_rows = 12_000
    var n_features = 25
    var data = _make_data(n_rows, n_features, 64, UInt64(2))
    var grad = _grads(n_rows, UInt64(11_000_000))
    var hess = _hessians(n_rows, UInt64(12_000_000))

    # Tiny odd subset: serial path, SIMD-width-free row count, and far below
    # the row-block floor.
    var tiny = List[Int]()
    for i in range(7):
        tiny.append(3 * i + 1)
    assert_equal(_subset_plan(data, len(tiny)).row_blocks, 1)
    _assert_same(
        _reference_subset(data, grad, hess, tiny),
        build_histogram_subset(data, grad, hess, tiny),
    )

    # Strides derived from the threshold so these cases keep exercising the
    # serial and parallel paths at whatever value PARALLEL_MIN_OPS is tuned
    # to; the guards fail loudly if a future value breaks that.
    var below_stride = n_features * n_rows // (PARALLEL_MIN_OPS - 1) + 1
    var below = _stride_rows(n_rows, below_stride)
    assert_true(n_features * len(below) < PARALLEL_MIN_OPS)
    _assert_same(
        _reference_subset(data, grad, hess, below),
        build_histogram_subset(data, grad, hess, below),
    )

    var above_stride = n_features * n_rows // (2 * PARALLEL_MIN_OPS)
    if above_stride < 1:
        above_stride = 1
    var above = _stride_rows(n_rows, above_stride)
    assert_true(n_features * len(above) >= PARALLEL_MIN_OPS)
    _assert_same(
        _reference_subset(data, grad, hess, above),
        build_histogram_subset(data, grad, hess, above),
    )


def test_subtract_matches_reference_odd_tail() raises:
    # 3 features x 17 bins = 51 entries: odd against 8/16/32-lane widths,
    # so the SIMD loop leaves a scalar tail on every architecture.
    var n_rows = 907
    var n_features = 3
    var data = _make_data(n_rows, n_features, 17, UInt64(3))
    var grad = _grads(n_rows, UInt64(13_000_000))
    var hess = _hessians(n_rows, UInt64(14_000_000))

    var left = _stride_rows(n_rows, 2)

    var parent = build_histogram(data, grad, hess)
    var child = build_histogram_subset(data, grad, hess, left)
    var sibling = subtract_histogram(parent, child)

    var size = n_features * parent.n_bins
    for i in range(size):
        assert_equal(
            sibling.grad_at(i).to_bits(),
            (parent.grad_at(i) - child.grad_at(i)).to_bits(),
        )
        assert_equal(
            sibling.hess_at(i).to_bits(),
            (parent.hess_at(i) - child.hess_at(i)).to_bits(),
        )
        assert_equal(
            sibling.count_at(i), parent.count_at(i) - child.count_at(i)
        )

    # The two children partition the parent's rows.
    var total = 0
    for i in range(size):
        total += child.count_at(i) + sibling.count_at(i)
    assert_equal(total, n_features * n_rows)


def test_env_worker_and_threshold_overrides() raises:
    # One test owns all env mutation for the unblocked shape so no other test
    # sees a dirty environment regardless of suite ordering; empty string
    # means unset.
    var n_rows = 997
    var n_features = 3
    var data = _make_data(n_rows, n_features, 17, UInt64(4))
    var grad = _grads(n_rows, UInt64(15_000_000))
    var hess = _hessians(n_rows, UInt64(16_000_000))
    var expected = _reference_full(data, grad, hess)

    # Lowered threshold forces the per-feature parallel path on a shape
    # that would otherwise run serial.
    _ = setenv("MOJOTREES_PARALLEL_MIN_OPS", "1")
    _assert_same(expected, build_histogram(data, grad, hess))

    # Explicit worker count takes the chunked path (2 tasks over 3 features).
    _ = setenv("MOJOTREES_NUM_WORKERS", "2")
    _assert_same(expected, build_histogram(data, grad, hess))

    # More workers than features clamps to one feature per task.
    _ = setenv("MOJOTREES_NUM_WORKERS", "8")
    _assert_same(expected, build_histogram(data, grad, hess))

    # workers=1 forces serial even with the threshold floored.
    _ = setenv("MOJOTREES_NUM_WORKERS", "1")
    _assert_same(expected, build_histogram(data, grad, hess))

    _clear_env()


# ---------------------------------------------------------------------------
# The row-block rule, as arithmetic
# ---------------------------------------------------------------------------


def test_row_block_rule_arithmetic() raises:
    """The block count, hand-computed, at each of the four things that can
    bind it.

    Every expected number below is derived in the comment next to it from
    `row_block_min_rows(bins) = 2 * ROW_BLOCK_AMORTIZE * bins` and the byte
    budget, so a change to either constant fails here with its own arithmetic
    on screen rather than silently re-planning the reference too.
    """
    _clear_env()
    var m = CpuProfile.detect()

    # 255 bins: the floor is 2 * 8 * 255 = 4080 rows per block.
    assert_equal(row_block_min_rows(255), 4080)

    # Below two blocks' worth of rows: unblocked. 8159 // 4080 == 1.
    assert_equal(
        derive_accumulation_plan(m, 50, 50, 255, 8159, True).row_blocks, 1
    )
    # Exactly two blocks' worth: blocked, two blocks of 4080.
    var two = derive_accumulation_plan(m, 50, 50, 255, 8160, True)
    assert_equal(two.row_blocks, 2)
    assert_equal(two.block_rows, 4080)
    assert_equal(two.block_cells, 50 * 255)

    # The amortization floor binding: 40000 // 4080 == 9.
    var med = derive_accumulation_plan(m, 50, 50, 255, 40_000, True)
    assert_equal(med.row_blocks, 9)
    # chunk = ceil(40000 / 9) = 4445, and ceil(40000 / 4445) is 9 again.
    assert_equal(med.block_rows, 4445)

    # The byte budget binding: 50 * 255 = 12750 cells, 12750 * 3 * 8 = 306000
    # bytes a block, and 16777216 // 306000 == 54. The floor alone would have
    # asked for 1000000 // 4080 == 245.
    var big = derive_accumulation_plan(m, 50, 50, 255, 1_000_000, True)
    assert_equal(big.row_blocks, 54)

    # `MAX_ROW_BLOCKS` binding: 8 bins over 4 features is 32 cells, 768 bytes
    # a block, so the byte budget would allow 21845. The floor asks for
    # 4000000 // 128 == 31250. The absolute ceiling is what answers.
    var narrow = derive_accumulation_plan(m, 4, 4, 8, 4_000_000, True)
    assert_equal(narrow.row_blocks, MAX_ROW_BLOCKS)

    # The full-dataset builder plans the identical count, and a degenerate
    # shape.
    assert_equal(
        derive_accumulation_plan(m, 50, 50, 255, 1_000_000, False).row_blocks,
        54,
    )
    assert_equal(derive_accumulation_plan(m, 50, 50, 0, 100, True).row_blocks, 1)
    assert_equal(derive_accumulation_plan(m, 50, 50, 255, 0, True).row_blocks, 1)


def test_row_block_count_ignores_the_machine() raises:
    """The same shape must plan the same block count on any core layout.

    `derive_accumulation_plan` takes a `CpuProfile`, and a synthetic one lets
    this assert the independence directly rather than inferring it from the
    source. The group width is allowed to move with the machine and does; the
    block count is not and must not.
    """
    _clear_env()
    var one_core = CpuProfile.synthetic(1, 1)
    var many = CpuProfile.synthetic(128, 96)
    var a = derive_accumulation_plan(one_core, 50, 50, 255, 200_000, True)
    var b = derive_accumulation_plan(many, 50, 50, 255, 200_000, True)
    assert_equal(a.row_blocks, b.row_blocks)
    assert_equal(a.block_rows, b.block_rows)
    assert_equal(a.block_cells, b.block_cells)
    assert_true(a.row_blocks > 1)
    # And the width is machine-dependent, so this is not vacuously true of
    # everything the planner returns.
    assert_not_equal(a.group_width, b.group_width)


def test_row_block_env_override() raises:
    _clear_env()
    var m = CpuProfile.detect()
    # 1 is the off switch: the shape would otherwise block 9 ways.
    _ = setenv("MOJOTREES_CPU_ROW_BLOCKS", "1")
    assert_equal(
        derive_accumulation_plan(m, 50, 50, 255, 40_000, True).row_blocks, 1
    )
    # An explicit count bypasses the amortization floor.
    _ = setenv("MOJOTREES_CPU_ROW_BLOCKS", "4")
    var forced = derive_accumulation_plan(m, 50, 50, 255, 1000, True)
    assert_equal(forced.row_blocks, 4)
    assert_equal(forced.block_rows, 250)
    # And is clamped by the row count: a block cannot be shorter than a row.
    _ = setenv("MOJOTREES_CPU_ROW_BLOCKS", "40")
    assert_equal(
        derive_accumulation_plan(m, 50, 50, 255, 7, True).row_blocks, 7
    )
    _clear_env()


# ---------------------------------------------------------------------------
# The row-blocked kernel: that it runs, that it is right, that it is
# deterministic
# ---------------------------------------------------------------------------
#
# One fixture serves all of them. 15,000 rows of a 30,000-row matrix, 12
# features, 32 bins:
#
#   row_block_min_rows(32) = 2 * 8 * 32          = 512
#   blocks (floor)         = 15000 // 512        = 29
#   cells                  = 12 * 32             = 384
#   byte budget            = 16777216 // (384*24) = 1820, so it does not bind
#   MAX_ROW_BLOCKS         = 64,                   so it does not bind
#   chunk                  = ceil(15000 / 29)    = 518
#   recount                = ceil(15000 / 518)   = 29
#
# All of those are literals in the assertions below.

comptime BLK_ROWS = 30_000
comptime BLK_FEATURES = 12
comptime BLK_BINS = 32
comptime BLK_SUB = 15_000
comptime BLK_BLOCKS = 29
comptime BLK_CHUNK = 518


def _blocked_fixture() raises -> BinnedMatrix:
    return _make_data(BLK_ROWS, BLK_FEATURES, BLK_BINS, UInt64(21))


def test_blocked_path_runs_and_matches_the_blocked_reference() raises:
    """The marker. Three assertions, and the third is the one that proves a
    gate opened rather than assuming it.

    1. The planner selects 29 blocks of 518 rows for this shape, with the
       arithmetic above.
    2. The build is bit-identical to the blocked reference.
    3. The blocked reference is bit-*different* from the flat reference on
       this data, and the build under `MOJOTREES_CPU_ROW_BLOCKS=1` is
       bit-identical to the flat one.

    Without (3) the pair (1, 2) is gate-blind: if the two references agreed,
    (2) would pass whichever kernel ran. With (3) the two orders are known to
    produce different bytes here, so a build that matches the blocked one and
    not the flat one can only have come from the blocked kernel.
    """
    _clear_env()
    var data = _blocked_fixture()
    var grad = _grads_spread(BLK_ROWS, UInt64(21_000_000))
    var hess = _hessians(BLK_ROWS, UInt64(22_000_000))
    var rows = _stride_rows(BLK_ROWS, 2)
    assert_equal(len(rows), BLK_SUB)

    var plan = _subset_plan(data, BLK_SUB)
    assert_true(plan.blocked())
    assert_equal(plan.row_blocks, BLK_BLOCKS)
    assert_equal(plan.block_rows, BLK_CHUNK)
    assert_equal(plan.block_cells, BLK_FEATURES * BLK_BINS)

    var flat = _reference_subset_flat(data, grad, hess, rows)
    var blocked = _reference_subset_blocked(
        data, grad, hess, rows, BLK_BLOCKS, BLK_CHUNK
    )
    # The fixture distinguishes the two orders. If this ever stops holding,
    # every other assertion in this test stops meaning anything, so it is
    # checked and not assumed.
    assert_true(_differs_somewhere(flat, blocked))

    _assert_same(blocked, build_histogram_subset(data, grad, hess, rows))

    _ = setenv("MOJOTREES_CPU_ROW_BLOCKS", "1")
    assert_equal(_subset_plan(data, BLK_SUB).row_blocks, 1)
    _assert_same(flat, build_histogram_subset(data, grad, hess, rows))
    _clear_env()


def test_blocked_counts_are_exact_and_planes_agree() raises:
    """The count plane is integer addition, which is associative, so blocking
    cannot move it. Asserted against the flat reference, not the blocked one,
    which is the whole point: the gradient plane moved and the count plane did
    not."""
    _clear_env()
    var data = _blocked_fixture()
    var grad = _grads_spread(BLK_ROWS, UInt64(23_000_000))
    var hess = _hessians(BLK_ROWS, UInt64(24_000_000))
    var rows = _stride_rows(BLK_ROWS, 2)
    var flat = _reference_subset_flat(data, grad, hess, rows)
    var built = build_histogram_subset(data, grad, hess, rows)
    assert_true(_subset_plan(data, len(rows)).blocked())

    var moved = 0
    for i in range(BLK_FEATURES * BLK_BINS):
        assert_equal(built.count_at(i), flat.count_at(i))
        if built.grad_at(i).to_bits() != flat.grad_at(i).to_bits():
            moved += 1
    # And the gradient plane did move somewhere, so the count assertion above
    # is a statement about associativity rather than about nothing happening.
    assert_true(moved > 0)


def test_blocked_bits_identical_across_workers_and_task_counts() raises:
    """The central assertion of this lane.

    Nine arms. Three worker counts, three grain settings that change the task
    count by more than an order of magnitude, and a fan-out ceiling. Every one
    must produce the identical bytes, because none of them reaches the block
    count.
    """
    _clear_env()
    var data = _blocked_fixture()
    var grad = _grads_spread(BLK_ROWS, UInt64(25_000_000))
    var hess = _hessians(BLK_ROWS, UInt64(26_000_000))
    var rows = _stride_rows(BLK_ROWS, 2)

    # Gate: the arms below are only worth running if this shape blocks.
    assert_true(_subset_plan(data, len(rows)).blocked())

    var expected = build_histogram_subset(data, grad, hess, rows)
    # And the arms are only worth running if the blocking is observable, so
    # the same fixture check as the marker test.
    assert_true(
        _differs_somewhere(
            _reference_subset_flat(data, grad, hess, rows), expected
        )
    )

    # Nine arms, as four parallel lists rather than a list of tuples: the
    # index is the arm and an empty string is "unset".
    #
    #   workers 1 / 3 / 8            the three the round requires
    #   min_ops 1 with min_task 1    auto, maximum fan-out
    #   min_ops 1 alone              auto, crossover cleared, default grain
    #   min_ops 1e9                  auto, forced back to serial
    #   tasks_per_core 1 and 16      the fan-out ceiling moved 16x
    #   min_task 97                  an odd grain, so the split lands oddly
    var workers = ["1", "3", "8", "", "", "", "", "", ""]
    var min_ops = ["", "", "", "1", "1", "1000000000", "1", "1", "1"]
    var min_task = ["", "", "", "1", "", "", "1", "1", "97"]
    var per_core = ["", "", "", "", "", "", "1", "16", ""]

    for k in range(len(workers)):
        _ = setenv("MOJOTREES_NUM_WORKERS", workers[k])
        _ = setenv("MOJOTREES_PARALLEL_MIN_OPS", min_ops[k])
        _ = setenv("MOJOTREES_PARALLEL_MIN_TASK_OPS", min_task[k])
        _ = setenv("MOJOTREES_CPU_TASKS_PER_CORE", per_core[k])
        # Still blocked, still the same block count, under every arm.
        var p = _subset_plan(data, len(rows))
        assert_equal(p.row_blocks, BLK_BLOCKS)
        assert_equal(p.block_rows, BLK_CHUNK)
        _assert_same(expected, build_histogram_subset(data, grad, hess, rows))

    _clear_env()


def test_blocked_bits_identical_across_feature_group_widths() raises:
    """The interleave width is scheduling-only and stays scheduling-only under
    blocking. This also walks every rung of the ladder the blocked kernel is
    instantiated at, so all five arms are compiled and run."""
    _clear_env()
    var data = _blocked_fixture()
    var grad = _grads(BLK_ROWS, UInt64(27_000_000))
    var hess = _hessians(BLK_ROWS, UInt64(28_000_000))
    var rows = _stride_rows(BLK_ROWS, 2)
    assert_true(_subset_plan(data, len(rows)).blocked())

    _ = setenv("MOJOTREES_CPU_FEATURE_GROUP", "1")
    var expected = build_histogram_subset(data, grad, hess, rows)
    var rungs = ["1", "2", "4", "8", "16"]
    for k in range(len(rungs)):
        _ = setenv("MOJOTREES_CPU_FEATURE_GROUP", rungs[k])
        var p = _subset_plan(data, len(rows))
        assert_equal(p.group_width, Int(rungs[k]))
        assert_equal(p.row_blocks, BLK_BLOCKS)
        _assert_same(expected, build_histogram_subset(data, grad, hess, rows))
    _clear_env()


def test_blocked_constant_hessian_elides_and_is_exact() raises:
    """Two claims, and the second is what proves the elision actually fired.

    1. With a genuinely constant hessian of 1.0, the declared and undeclared
       builds are bit-identical, which is the contract.
    2. With a hessian of 2.0 and the declaration made anyway -- a deliberate
       lie, which `const_hessian_verify` exists to catch and which is off
       here -- the declared build writes `Float64(count)` and the undeclared
       one writes twice that. They differ, and they differ in exactly the way
       the elision predicts, so (1) is a statement about a path that ran.
    """
    _clear_env()
    var data = _blocked_fixture()
    var grad = _grads(BLK_ROWS, UInt64(29_000_000))
    var rows = _stride_rows(BLK_ROWS, 2)
    assert_true(_subset_plan(data, len(rows)).blocked())

    var ones = _constant(BLK_ROWS, 1.0)
    var plain = build_histogram_subset(data, grad, ones, rows, [], False)
    var elided = build_histogram_subset(data, grad, ones, rows, [], True)
    _assert_same(plain, elided)
    # And the reference agrees with both, so this is not two wrong answers
    # agreeing with each other.
    _assert_same(_reference_subset(data, grad, ones, rows), plain)

    var twos = _constant(BLK_ROWS, 2.0)
    var honest = build_histogram_subset(data, grad, twos, rows, [], False)
    var lied = build_histogram_subset(data, grad, twos, rows, [], True)
    var seen = 0
    for i in range(BLK_FEATURES * BLK_BINS):
        var n = honest.count_at(i)
        assert_equal(lied.count_at(i), n)
        # The elided plane is the count, exactly.
        assert_equal(lied.hess_at(i).to_bits(), Float64(n).to_bits())
        # The three-plane path summed 2.0 n times, which for these n is
        # exactly 2n.
        assert_equal(honest.hess_at(i).to_bits(), Float64(2 * n).to_bits())
        if n > 0:
            seen += 1
    assert_true(seen > 0)

    # The off switch reaches the blocked kernel too.
    _ = setenv("MOJOTREES_CONST_HESSIAN", "0")
    _assert_same(honest, build_histogram_subset(data, grad, twos, rows, [], True))
    _clear_env()


def test_blocked_feature_subset_indexes_two_spaces_correctly() raises:
    """The blocked kernel indexes its private partials by *active slot* and
    the output by *feature id*, and those are two different numbers under
    feature subsampling. Nothing in the whole-feature tests above can tell the
    two apart, because there slot equals id.

    The reference is the all-features blocked build with the excluded slices
    zeroed, which is what every builder in the module promises to produce.
    """
    _clear_env()
    var data = _blocked_fixture()
    var grad = _grads(BLK_ROWS, UInt64(33_000_000))
    var hess = _hessians(BLK_ROWS, UInt64(34_000_000))
    var rows = _stride_rows(BLK_ROWS, 2)
    var features: List[Int] = [1, 4, 7, 9, 11]

    var plan = derive_accumulation_plan(
        CpuProfile.detect(),
        BLK_FEATURES,
        len(features),
        BLK_BINS,
        len(rows),
        True,
    )
    assert_true(plan.blocked())
    assert_equal(plan.row_blocks, BLK_BLOCKS)
    assert_equal(plan.block_cells, len(features) * BLK_BINS)

    var expected = _reference_subset_blocked(
        data, grad, hess, rows, BLK_BLOCKS, BLK_CHUNK
    )
    var active = List[Bool]()
    active.resize(BLK_FEATURES, False)
    for k in range(len(features)):
        active[features[k]] = True
    for f in range(BLK_FEATURES):
        if not active[f]:
            for b in range(BLK_BINS):
                expected.set_grad_at(f * BLK_BINS + b, 0.0)
                expected.set_hess_at(f * BLK_BINS + b, 0.0)
                expected.set_count_at(f * BLK_BINS + b, 0)

    _assert_same(
        expected, build_histogram_subset(data, grad, hess, rows, features)
    )


def test_blocked_single_feature_takes_the_ungathered_loops() raises:
    """One active feature: the gather cannot pay for itself, so `compact` is
    False and the blocked kernel runs its indirect-load row loops. Those are a
    separate pair of loops from the gathered ones and nothing else in this
    file reaches them at a size that blocks."""
    _clear_env()
    var data = _blocked_fixture()
    var grad = _grads(BLK_ROWS, UInt64(35_000_000))
    var hess = _hessians(BLK_ROWS, UInt64(36_000_000))
    var rows = _stride_rows(BLK_ROWS, 2)
    var features: List[Int] = [5]

    var plan = derive_accumulation_plan(
        CpuProfile.detect(), BLK_FEATURES, 1, BLK_BINS, len(rows), True
    )
    assert_true(plan.blocked())
    assert_true(not plan.compact_rows)
    assert_equal(plan.group_count, 1)

    var expected = _reference_subset_blocked(
        data, grad, hess, rows, BLK_BLOCKS, BLK_CHUNK
    )
    for f in range(BLK_FEATURES):
        if f != 5:
            for b in range(BLK_BINS):
                expected.set_grad_at(f * BLK_BINS + b, 0.0)
                expected.set_hess_at(f * BLK_BINS + b, 0.0)
                expected.set_count_at(f * BLK_BINS + b, 0)

    _assert_same(
        expected, build_histogram_subset(data, grad, hess, rows, features)
    )


def test_blocked_sibling_subtraction_stays_exact() raises:
    """What "exact" means for a sibling subtraction, under blocking.

    The parent and the two children have different row counts, so they get
    *different* block counts -- 58, 19 and 39 here -- and there is no way to
    give a subtraction "the same blocking on both operands" when the operands
    are different row sets. What survives that, and is asserted here, is
    everything that was ever exact:

    - The count plane. Integer addition is associative, so a blocked count is
      the same integer at any block count, and `parent - child` is the
      sibling's own count exactly.
    - The hessian plane under the constant-hessian specialization. Both
      operands hold `Float64(count)`, both are exactly representable integers
      below 2^53, and so is their difference.

    The gradient plane's `parent - child` was never bit-equal to a direct
    build of the sibling and is not asserted to be. That is a property of
    Float64 subtraction, not of blocking, and it is unchanged.
    """
    _clear_env()
    var n_rows = 30_000
    var nf = 8
    var nb = 32
    var data = _make_data(n_rows, nf, nb, UInt64(31))
    var grad = _grads(n_rows, UInt64(31_000_000))
    var ones = _constant(n_rows, 1.0)

    var all_rows = List[Int]()
    var left = List[Int]()
    var right = List[Int]()
    for r in range(n_rows):
        all_rows.append(r)
        if r % 3 == 0:
            left.append(r)
        else:
            right.append(r)

    # row_block_min_rows(32) = 512.
    #   parent 30000 // 512 = 58, chunk ceil(30000/58) = 518, recount 58
    #   left   10000 // 512 = 19, chunk ceil(10000/19) = 527, recount 19
    #   right  20000 // 512 = 39, chunk ceil(20000/39) = 513, recount 39
    var pp = _subset_plan(data, len(all_rows))
    var lp = _subset_plan(data, len(left))
    var rp = _subset_plan(data, len(right))
    assert_equal(pp.row_blocks, 58)
    assert_equal(lp.row_blocks, 19)
    assert_equal(rp.row_blocks, 39)

    var parent = build_histogram_subset(data, grad, ones, all_rows, [], True)
    var child = build_histogram_subset(data, grad, ones, left, [], True)
    var direct = build_histogram_subset(data, grad, ones, right, [], True)
    var derived = subtract_histogram(parent, child, True)

    for i in range(nf * nb):
        # Counts: exact, across three different block counts.
        assert_equal(derived.count_at(i), direct.count_at(i))
        assert_equal(
            derived.count_at(i), parent.count_at(i) - child.count_at(i)
        )
        # Constant hessian: exact, and equal to the direct build's plane.
        assert_equal(
            derived.hess_at(i).to_bits(),
            Float64(direct.count_at(i)).to_bits(),
        )
        assert_equal(
            derived.hess_at(i).to_bits(), direct.hess_at(i).to_bits()
        )

    # The three-plane subtraction agrees with the elided one on this data,
    # which is the claim `_subtract_histogram_arrays` makes.
    var derived3 = subtract_histogram(parent, child, False)
    _assert_same(derived, derived3)
    _clear_env()


# ---------------------------------------------------------------------------
# The LightGBM cell: interleaved, Float32 derivatives, and a derived count
# ---------------------------------------------------------------------------

comptime _POISON = -1.5e300
"""A value no accumulation can produce, written into every float of the
caller-owned scratch before a build so that "was this slot written" is an
observable fact rather than an assumption."""


def _poisoned(n: Int) -> List[Float64]:
    var p = List[Float64](capacity=n)
    p.resize(n, _POISON)
    return p^


def _blocked_subset_fixture() raises -> BinnedMatrix:
    return _make_data(20_000, 6, 32, UInt64(515_151))


def test_gathered_pair_buffer_holds_two_float32_per_row() raises:
    """The gather is one Float64 word per row holding two Float32, which is
    LightGBM's `score_t` precision and half the bytes the pair used to cost.

    Read back through the same bitcast the kernel uses. This is a layout
    assertion, not a value assertion: it fails if the buffer goes back to two
    Float64 per row even though every histogram cell would still be correct.
    """
    _clear_env()
    var data = _blocked_subset_fixture()
    var n_rows = data.n_rows
    var grad = _grads(n_rows, UInt64(31_000_001))
    var hess = _hessians(n_rows, UInt64(31_000_002))
    var rows = _stride_rows(n_rows, 1)
    var plan = _subset_plan(data, len(rows))
    assert_true(plan.blocked())

    var pairs = _poisoned(len(rows) + plan.block_scratch_floats())
    var out = Histogram.zeroed(data.n_features, data.n_bins)
    build_histogram_subset_into_scratch(
        out, pairs, data, grad, hess, rows, 0, len(rows)
    )

    var p32 = pairs.unsafe_ptr().unsafe_bitcast[Float32]()
    for i in range(len(rows)):
        var r = rows[i]
        assert_equal(p32.unsafe_load(2 * i), Float32(grad[r]))
        assert_equal(p32.unsafe_load(2 * i + 1), Float32(hess[r]))
        # And the word it lives in is no longer the poison, which is what
        # proves the gather ran at all.
        assert_not_equal(pairs[i].to_bits(), Float64(_POISON).to_bits())


def test_constant_hessian_private_cell_is_lightgbm_sixteen_bytes() raises:
    """The marker for the interleaved cell, and it is a *layout* marker.

    Under the constant-hessian specialization a private block cell is two
    Float64 -- gradient, then a count in the slot LightGBM aliases as
    `hist_cnt_t` -- so the kernel addresses two thirds of the three slots the
    scratch reserves per cell. The reserved third stays poisoned. Run the same
    fixture without the declaration and every slot is written instead, which
    is what makes this assertion distinguish the two strides rather than
    merely observe that some scratch went untouched.

    The scratch is reserved at three slots per cell either way, on purpose:
    the block count is derived from the byte budget and must not depend on
    `const_hessian`, or a fit that turned the specialization off would fold a
    different number of partials.
    """
    _clear_env()
    var data = _blocked_subset_fixture()
    var n_rows = data.n_rows
    var grad = _grads(n_rows, UInt64(32_000_001))
    var flat = _constant(n_rows, CONSTANT_HESSIAN)
    var rows = _stride_rows(n_rows, 1)
    var plan = _subset_plan(data, len(rows))
    assert_true(plan.blocked())

    var part_off = len(rows)
    var region = plan.block_cells * 3
    var total = part_off + plan.block_scratch_floats()
    assert_equal(plan.block_scratch_floats(), plan.row_blocks * region)

    var two = _poisoned(total)
    var out_two = Histogram.zeroed(data.n_features, data.n_bins)
    build_histogram_subset_into_scratch(
        out_two, two, data, grad, flat, rows, 0, len(rows), [], True
    )

    var three = _poisoned(total)
    var out_three = Histogram.zeroed(data.n_features, data.n_bins)
    build_histogram_subset_into_scratch(
        out_three, three, data, grad, flat, rows, 0, len(rows), [], False
    )

    var survived_two = 0
    var survived_three = 0
    var written_two = 0
    for blk in range(plan.row_blocks):
        var base = part_off + blk * region
        # Slots the two-float cell reaches.
        for i in range(base, base + plan.block_cells * 2):
            if two[i].to_bits() != Float64(_POISON).to_bits():
                written_two += 1
        # The third slot of every cell, which it does not.
        for i in range(base + plan.block_cells * 2, base + region):
            if two[i].to_bits() == Float64(_POISON).to_bits():
                survived_two += 1
            if three[i].to_bits() == Float64(_POISON).to_bits():
                survived_three += 1

    var reserved_third = plan.row_blocks * plan.block_cells
    assert_equal(survived_two, reserved_third)
    assert_equal(survived_three, 0)
    assert_true(written_two > 0)

    # And the two arms still produce the identical histogram, which is the
    # whole contract the specialization is allowed to keep.
    _assert_same(out_two, out_three)


def test_derived_count_is_lightgbm_round_int() raises:
    """`derived_count` is `Common::RoundInt`, which is `static_cast<int>(x +
    0.5f)` -- add a half, truncate toward zero -- and `cnt_factor` is
    `num_data / sum_hessian`. Stated as a table so the rule is readable
    without the C++ open next to it."""
    var xs = [0.0, 0.4, 0.5, 0.6, 1.0, 1.49, 1.5, 2.5, 9.5, 1000.0]
    for k in range(len(xs)):
        assert_equal(derived_count(xs[k], 1.0), Int(xs[k] + 0.5))
    # The factor multiplies before the round, as it does in LightGBM.
    assert_equal(derived_count(4.0, 2.5), 10)
    assert_equal(derived_count(3.0, 0.5), 2)
    assert_equal(cnt_factor(100, 50.0), 2.0)
    # Under the constant-hessian specialization the factor is exactly 1.0,
    # which is the whole reason the derived count is exact there.
    assert_equal(cnt_factor(7, CONSTANT_HESSIAN * 7.0), 1.0)
    assert_equal(cnt_factor(0, 0.0), 0.0)


def test_derived_count_is_exact_under_constant_hessian() raises:
    """A count recovered from a constant-hessian sum is the count, exactly,
    with no tolerance anywhere.

    The sum of `n` copies of 1.0 is `Float64(n)` because every partial is an
    exactly representable integer below 2^53, so the recovery is
    `Int(Float64(n) + 0.5)` and truncates back to `n`. Checked as arithmetic
    over a wide range of n, and then on a real blocked build whose counts must
    match the serial reference integer for integer.
    """
    _clear_env()
    var ns = [0, 1, 2, 3, 255, 4096, 100_003, 1 << 30]
    for k in range(len(ns)):
        var n = ns[k]
        assert_equal(derived_count(Float64(n), 1.0), n)

    var data = _blocked_subset_fixture()
    var n_rows = data.n_rows
    var grad = _grads(n_rows, UInt64(33_000_001))
    var flat = _constant(n_rows, CONSTANT_HESSIAN)
    var rows = _stride_rows(n_rows, 1)
    assert_true(_subset_plan(data, len(rows)).blocked())
    var built = build_histogram_subset(data, grad, flat, rows, [], True)
    var want = _reference_subset(data, grad, flat, rows)
    var total = 0
    for i in range(data.n_features * data.n_bins):
        assert_equal(built.count_at(i), want.count_at(i))
        # And the hessian plane is `Float64(count)` to the bit, which is what
        # the count slot held.
        assert_equal(
            built.hess_at(i).to_bits(), Float64(built.count_at(i)).to_bits()
        )
        total += built.count_at(i)
    assert_equal(total, data.n_features * len(rows))


def test_general_hessian_keeps_an_exact_count() raises:
    """Where the hessian varies per row, a count is not a function of the
    hessian sum, so this package keeps an exact one rather than taking
    LightGBM's estimate.

    The divergence is deliberate and this is what it buys: on a fixture whose
    hessians are drawn per row, the derived estimate is checked to be *wrong*
    somewhere -- otherwise the exact slot would be paying eight bytes a cell
    for nothing -- while the stored count matches the reference exactly.
    """
    _clear_env()
    var data = _blocked_subset_fixture()
    var n_rows = data.n_rows
    var grad = _grads(n_rows, UInt64(34_000_001))
    var hess = _hessians(n_rows, UInt64(34_000_002))
    var rows = _stride_rows(n_rows, 1)
    assert_true(_subset_plan(data, len(rows)).blocked())
    var built = build_histogram_subset(data, grad, hess, rows)
    var want = _reference_subset(data, grad, hess, rows)

    var sum_h = 0.0
    for i in range(data.n_bins):
        sum_h += built.hess_at(i)
    var factor = cnt_factor(len(rows), sum_h)

    var estimate_wrong = False
    for i in range(data.n_features * data.n_bins):
        assert_equal(built.count_at(i), want.count_at(i))
        if derived_count(built.hess_at(i), factor) != built.count_at(i):
            estimate_wrong = True
    assert_true(estimate_wrong)


def _bits_of(h: Histogram) -> List[UInt64]:
    var out = List[UInt64](capacity=3 * h.n_features * h.n_bins)
    for i in range(h.n_features * h.n_bins):
        out.append(UInt64(h.grad_at(i).to_bits()))
        out.append(UInt64(h.hess_at(i).to_bits()))
        out.append(UInt64(h.count_at(i)))
    return out^


def _assert_bits_equal(a: List[UInt64], b: List[UInt64]) raises:
    assert_equal(len(a), len(b))
    for i in range(len(a)):
        assert_equal(a[i], b[i])


def test_interleaved_cell_is_deterministic_across_workers() raises:
    """Identical bits at one, three and eight workers, and across the task
    counts those imply, on both builders and on both hessian arms.

    The blocked kernel's unit is a `(block, group)` pair and the fold is
    dispatched over active slots, so both dispatches are cut differently at
    every worker count; what cannot move is the block count, the block
    boundaries and the ascending fold order, which come from the shape alone.
    This is the central assertion of the lane.
    """
    _clear_env()
    var data = _blocked_subset_fixture()
    var n_rows = data.n_rows
    var grad = _grads(n_rows, UInt64(35_000_001))
    var hess = _hessians(n_rows, UInt64(35_000_002))
    var flat = _constant(n_rows, CONSTANT_HESSIAN)
    var rows = _stride_rows(n_rows, 1)
    assert_true(_subset_plan(data, len(rows)).blocked())
    assert_true(_full_plan(data).blocked())

    var workers = ["1", "3", "8"]
    var base_sub = List[UInt64]()
    var base_full = List[UInt64]()
    var base_ch = List[UInt64]()
    for k in range(len(workers)):
        _ = setenv("MOJOTREES_NUM_WORKERS", workers[k])
        # Also move the task grain, so this is not merely a worker-count
        # sweep at one fixed number of tasks.
        _ = setenv("MOJOTREES_CPU_TASKS_PER_CORE", "1" if k == 0 else "4")
        var sub = _bits_of(build_histogram_subset(data, grad, hess, rows))
        var full = _bits_of(build_histogram(data, grad, hess))
        var ch = _bits_of(
            build_histogram_subset(data, grad, flat, rows, [], True)
        )
        if k == 0:
            base_sub = sub^
            base_full = full^
            base_ch = ch^
        else:
            _assert_bits_equal(base_sub, sub)
            _assert_bits_equal(base_full, full)
            _assert_bits_equal(base_ch, ch)
    _clear_env()


def test_full_builder_scratch_is_reused_not_reallocated() raises:
    """`build_histogram_into_scratch` grows the caller's buffer once and
    returns the same histogram the allocating form returns.

    The whole-dataset builder blocks now, so it needs private histograms it
    used to have no home for. This is the home; `build_histogram_into` passes
    a fresh empty list, which is the allocate-per-call behaviour every
    unwired caller keeps.
    """
    _clear_env()
    var data = _blocked_subset_fixture()
    var n_rows = data.n_rows
    var grad = _grads(n_rows, UInt64(36_000_001))
    var hess = _hessians(n_rows, UInt64(36_000_002))
    var plan = _full_plan(data)
    assert_true(plan.blocked())

    var scratch = List[Float64]()
    var reused = Histogram.zeroed(data.n_features, data.n_bins)
    build_histogram_into_scratch(reused, scratch, data, grad, hess)
    var grown = len(scratch)
    assert_equal(grown, plan.block_scratch_floats())
    _assert_same(reused, build_histogram(data, grad, hess))

    # A second build reuses the allocation rather than growing it again.
    build_histogram_into_scratch(reused, scratch, data, grad, hess)
    assert_equal(len(scratch), grown)
    _assert_same(reused, build_histogram(data, grad, hess))
    _clear_env()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
