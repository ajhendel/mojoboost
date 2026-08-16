"""Proof that the constant-hessian elision actually ran.

Why this file exists beside `test_const_hessian.mojo`, which covers the same
specialization. That file asserts that the two-plane arm and the three-plane
arm are **the same bytes**, on data where the declaration is true. That is the
correctness claim and it is worth having. It is not a claim that the
optimization fired, and it cannot be made into one: the two arms are equal by
design, so every assertion in it holds whether the elided loop ran, whether it
was compiled out, or whether `_resolve_const_hessian` were rewritten to
`return False`. Its `hess == Float64(count)` checks do not close the gap
either, because a hessian array of all `1.0` makes the honest three-plane arm
produce `Float64(count)` too -- that is the whole exactness argument the
specialization rests on.

So this file inverts the logic. The elision is a **declaration**: nothing in
the builders infers it and nothing validates it unless
`MOJOTREES_CONST_HESSIAN_VERIFY` is on. Declare it on data where it is
**false** -- hessians that are genuinely not `CONSTANT_HESSIAN` -- with the
verifier off, and the two arms stop being equal:

  - the honest three-plane arm accumulates the hessians it was given, so a
    cell holds the sum of that cell's rows' hessians;
  - the two-plane arm never touches the hessian input at all and refills the
    plane from the count, so a cell holds `Float64(count)`.

Those differ, and they can only differ if the elided loop ran. Every
assertion below is of that shape: **divergence in exactly the case where only
the optimized path can produce it**. Un-wire the elision -- make
`_resolve_const_hessian` return False, delete the `const_h` branch in
`_accumulate_full_at` or `_accumulate_subset_at`, drop the `const_h` arm of
`_subtract_histogram_arrays` -- and every test in this file goes red, because
the two arms come back equal.

The control for each divergence claim is `MOJOTREES_CONST_HESSIAN=0`, which
withdraws permission at `const_hessian_allowed`. Under it the same false
declaration must produce **agreement** again. Divergence-with-permission and
agreement-without-permission together say the difference is the elision and
not some unrelated instability, and they say the off switch a bisection
depends on really reaches the loop.

A note on legitimacy, since a false declaration is a thing this package
otherwise treats as a defect. It is legitimate here for exactly one reason:
`const_hessian_verify()` is off, which is its default, and the last test in
the CPU section pins that turning it on turns this same fixture into a raise.
No trainer in `src/` declares a constant hessian on data like this; the
declaration is made in one place per backend, from the objective, through
`boosting.round_has_constant_hessian`. This file reaches the builders
directly, which is the only way to construct the negative control at all --
see the report note about what could not be done without editing `src/`.

Exactness. Every comparison here is `to_bits()` or an integer equality and
there is no tolerance anywhere. The reference values the tests compare
against are integers converted to Float64 (`Float64(count)`, `Float64(2 *
count)`), never a floating multiply, so the reference introduces no rounding
and no contraction site of the kind `docs/NUMERICS.md` catalogues.

Nothing here is timed and nothing here may be read as evidence that the
specialization is faster.
"""

from std.os import setenv
from std.testing import assert_equal, assert_true, TestSuite

from mojotrees.binning import BinnedMatrix
from mojotrees.histogram import (
    CONSTANT_HESSIAN,
    Histogram,
    build_histogram,
    build_histogram_subset,
    const_hessian_allowed,
    subtract_histogram,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

comptime _N_ROWS = 1_024
comptime _N_FEATURES = 17
"""Deliberately not a multiple of any rung of the interleave ladder, so the
width-16 instantiation grows a tail group owning one feature. The elision
branch lives inside the group body, so the tail is its own path through it."""
comptime _N_BINS = 13


def _matrix() raises -> BinnedMatrix:
    """A deterministic column-major binned matrix in which **every** (feature,
    bin) cell receives at least one row.

    That property is load-bearing rather than incidental: the divergence
    assertions below are made cell by cell, and an empty cell diverges under
    no arm, so an all-cells-differ claim would be silently weakened by a
    fixture with holes in it. `test_every_cell_of_the_fixture_is_occupied`
    pins it.
    """
    var bins = List[UInt8](capacity=_N_ROWS * _N_FEATURES)
    bins.resize(_N_ROWS * _N_FEATURES, UInt8(0))
    for f in range(_N_FEATURES):
        for r in range(_N_ROWS):
            var v = (r + f * 3) % _N_BINS
            if r % 11 == 0:
                v = (v + 5) % _N_BINS
            bins[f * _N_ROWS + r] = UInt8(v)
    return BinnedMatrix(bins^, _N_ROWS, _N_FEATURES, _N_BINS)


def _grad() -> List[Float64]:
    """Signed gradients with a wide magnitude spread, so a cell's sum is
    genuinely order-sensitive: the gradient plane is asserted **equal** across
    the two arms, and a fixture whose gradients were all alike would not
    notice a reassociation."""
    var g = List[Float64](capacity=_N_ROWS)
    for r in range(_N_ROWS):
        var scale = 1.0
        for _ in range(r % 8):
            scale *= 0.03125
        var sign = -1.0 if (r % 3 == 0) else 1.0
        g.append(sign * scale * (1.0 + Float64(r % 7)))
    return g^


def _hess_doubled() -> List[Float64]:
    """A hessian array of exactly 2.0 in every slot: the declaration
    `CONSTANT_HESSIAN` is false in every row, and falsely by an exact power of
    two, so the honest sum over a cell is exactly `Float64(2 * count)` and the
    reference needs no multiply and no tolerance.

    Uniformly wrong rather than sporadically wrong because it makes the
    divergence claim total -- **every** occupied cell must differ -- which is
    a far stronger statement than "some cell differed"."""
    var h = List[Float64](capacity=_N_ROWS)
    for _ in range(_N_ROWS):
        h.append(2.0)
    return h^


def _hess_varied() -> List[Float64]:
    """A hessian that is wrong in a different, ragged way: the integers 1, 2,
    3, 4 by row. Rows carrying exactly `CONSTANT_HESSIAN` are present, so this
    is not the degenerate "every row is wrong by the same amount" shape, and a
    reconstruction that happened to work by scaling rather than by counting
    would not survive it."""
    var h = List[Float64](capacity=_N_ROWS)
    for r in range(_N_ROWS):
        h.append(1.0 + Float64(r % 4))
    return h^


def _node_rows() -> List[Int]:
    """One node's rows: every third row dropped, which is the indirect,
    non-contiguous shape the subset builder is written for."""
    var rows = List[Int]()
    for r in range(_N_ROWS):
        if r % 3 != 2:
            rows.append(r)
    return rows^


def _bits(v: Float64) -> UInt64:
    return v.to_bits().cast[DType.uint64]()


def _reset_env():
    _ = setenv("MOJOTREES_CONST_HESSIAN", "")
    _ = setenv("MOJOTREES_CONST_HESSIAN_VERIFY", "")
    _ = setenv("MOJOTREES_CPU_FEATURE_GROUP", "")
    _ = setenv("MOJOTREES_CPU_COMPACT_MIN_ROWS", "")
    _ = setenv("MOJOTREES_NUM_WORKERS", "")
    _ = setenv("MOJOTREES_PARALLEL_MIN_OPS", "")


# ---------------------------------------------------------------------------
# Shared assertions
# ---------------------------------------------------------------------------

def _assert_untouched_planes_agree(a: Histogram, b: Histogram) raises:
    """The elision is allowed to change the hessian plane of a falsely
    declared build and nothing else. The gradient plane is compared as bits
    and the count plane as integers, so a two-plane arm that had reordered an
    addition or dropped a row would fail here rather than pass as a
    'divergence'."""
    assert_equal(a.n_features, b.n_features)
    assert_equal(a.n_bins, b.n_bins)
    for i in range(a.n_features * a.n_bins):
        assert_equal(_bits(a.grad[i]), _bits(b.grad[i]))
        assert_equal(a.count[i], b.count[i])


def _assert_reconstructed_from_count(h: Histogram) raises:
    """Every hessian cell of the two-plane arm is exactly `Float64(count)`.

    This is the positive half of the proof: not merely "the arms differ", but
    "the two-plane arm holds the specific value the reconstruction produces".
    A divergence with some third value in it would be a defect, not the
    elision, and this is what tells the two apart."""
    for i in range(h.n_features * h.n_bins):
        assert_equal(_bits(h.hess[i]), _bits(Float64(h.count[i])))


def _assert_honest_is_double_the_count(h: Histogram) raises:
    """Every hessian cell of the three-plane arm under `_hess_doubled` is
    exactly `Float64(2 * count)`: the sum of `count` copies of 2.0, every
    partial sum of which is an exactly representable even integer, so no
    rounding occurs anywhere in the series. Integer doubling then conversion,
    so the reference itself contains no floating multiply."""
    for i in range(h.n_features * h.n_bins):
        assert_equal(_bits(h.hess[i]), _bits(Float64(2 * h.count[i])))


def _count_diverging_cells(a: Histogram, b: Histogram) -> Int:
    var n = 0
    for i in range(a.n_features * a.n_bins):
        if _bits(a.hess[i]) != _bits(b.hess[i]):
            n += 1
    return n


def _occupied_cells(h: Histogram) -> Int:
    var n = 0
    for i in range(h.n_features * h.n_bins):
        if h.count[i] > 0:
            n += 1
    return n


# ---------------------------------------------------------------------------
# The fixture's own precondition
# ---------------------------------------------------------------------------

def test_every_cell_of_the_fixture_is_occupied() raises:
    """The all-cells-differ assertions elsewhere in this file are only as
    strong as this. An empty cell has hessian 0.0 on both arms, so a fixture
    that left cells empty would let `diverging == occupied` hold while most of
    the histogram proved nothing. Asserted here once rather than trusted."""
    _reset_env()
    var data = _matrix()
    var hist = build_histogram(data, _grad(), _hess_doubled(), [], False)
    assert_equal(_occupied_cells(hist), _N_FEATURES * _N_BINS)
    var total = 0
    for i in range(_N_FEATURES * _N_BINS):
        total += hist.count[i]
    assert_equal(total, _N_FEATURES * _N_ROWS)
    _reset_env()


# ---------------------------------------------------------------------------
# The full-dataset builder
# ---------------------------------------------------------------------------

def _full_diverges_at_group(group: String) raises:
    _ = setenv("MOJOTREES_CPU_FEATURE_GROUP", group)
    var data = _matrix()
    var grad = _grad()
    var hess = _hess_doubled()

    var three = build_histogram(data, grad, hess, [], False)
    var two = build_histogram(data, grad, hess, [], True)

    _assert_untouched_planes_agree(three, two)
    _assert_honest_is_double_the_count(three)
    _assert_reconstructed_from_count(two)
    assert_equal(
        _count_diverging_cells(three, two), _N_FEATURES * _N_BINS
    )


def test_full_build_elides_the_plane_at_every_group_width() raises:
    """The negative control on the full-dataset builder, at every rung of the
    interleave ladder.

    `_accumulate_full_at` is parametric on `GROUP`, so each rung is a separate
    instantiation of the accumulation body with its own copy of the elision
    branch, and 17 features gives the width-16 rung a one-feature tail group
    besides. A rung whose branch had been dropped would produce the honest
    plane and fail here; nothing in `test_const_hessian.mojo` could see that,
    because its arms agree on every rung either way.
    """
    _reset_env()
    var rungs: List[String] = ["1", "2", "4", "8", "16"]
    for i in range(len(rungs)):
        _full_diverges_at_group(rungs[i])
    _reset_env()


def test_full_build_reconstructs_from_count_on_a_ragged_hessian() raises:
    """The same claim on a hessian that is wrong row by row rather than
    uniformly, including rows that do carry `CONSTANT_HESSIAN`.

    The divergence count is asserted total here as well, and it is total for a
    reason that is a property of the fixture rather than of the arithmetic:
    every cell of `_matrix` holds at least 78 rows and the wrong hessians
    recur every four rows, so no cell can consist entirely of rows whose
    hessian happens to be 1.0. The two-plane arm still holds exactly
    `Float64(count)`, which is the statement that it read the count and not
    the hessian input."""
    _reset_env()
    var data = _matrix()
    var grad = _grad()
    var hess = _hess_varied()

    var three = build_histogram(data, grad, hess, [], False)
    var two = build_histogram(data, grad, hess, [], True)

    _assert_untouched_planes_agree(three, two)
    _assert_reconstructed_from_count(two)
    assert_equal(_count_diverging_cells(three, two), _N_FEATURES * _N_BINS)
    _reset_env()


def test_a_feature_subset_elides_only_the_selected_slices() raises:
    """Under feature subsampling the excluded features' slices are zeroed by a
    separate pass that the elision must not refill. An excluded slice has
    count 0, so a refill would write `Float64(0)` there and leave it looking
    correct; what would not look correct is an excluded slice that had somehow
    been accumulated, so both are checked. The selected slices must still
    diverge, which is what says the elision reached the pass that runs under a
    subset at all."""
    _reset_env()
    var data = _matrix()
    var grad = _grad()
    var hess = _hess_doubled()
    var features: List[Int] = [1, 4, 16]

    var three = build_histogram(data, grad, hess, features, False)
    var two = build_histogram(data, grad, hess, features, True)
    _assert_untouched_planes_agree(three, two)

    var diverged = 0
    for f in range(_N_FEATURES):
        var selected = f == 1 or f == 4 or f == 16
        for b in range(_N_BINS):
            var i = f * _N_BINS + b
            if selected:
                assert_true(two.count[i] > 0)
                assert_equal(_bits(two.hess[i]), _bits(Float64(two.count[i])))
                assert_equal(
                    _bits(three.hess[i]), _bits(Float64(2 * three.count[i]))
                )
                if _bits(three.hess[i]) != _bits(two.hess[i]):
                    diverged += 1
            else:
                assert_equal(two.count[i], 0)
                assert_equal(_bits(two.grad[i]), _bits(0.0))
                assert_equal(_bits(two.hess[i]), _bits(0.0))
                assert_equal(_bits(three.hess[i]), _bits(0.0))
    assert_equal(diverged, 3 * _N_BINS)
    _reset_env()


# ---------------------------------------------------------------------------
# The subset (tree node) builder
# ---------------------------------------------------------------------------

def _subset_diverges(compact_min_rows: String, group: String) raises:
    _ = setenv("MOJOTREES_CPU_COMPACT_MIN_ROWS", compact_min_rows)
    _ = setenv("MOJOTREES_CPU_FEATURE_GROUP", group)
    var data = _matrix()
    var grad = _grad()
    var hess = _hess_doubled()
    var rows = _node_rows()

    var three = build_histogram_subset(data, grad, hess, rows, [], False)
    var two = build_histogram_subset(data, grad, hess, rows, [], True)

    _assert_untouched_planes_agree(three, two)
    _assert_honest_is_double_the_count(three)
    _assert_reconstructed_from_count(two)
    assert_equal(_count_diverging_cells(three, two), _N_FEATURES * _N_BINS)

    var total = 0
    for i in range(_N_FEATURES * _N_BINS):
        total += two.count[i]
    assert_equal(total, _N_FEATURES * len(rows))


def test_subset_build_elides_the_plane_on_both_gather_arms() raises:
    """The node builder crosses the gather decision with the elision, so there
    are four row loops and each one has to be shown to elide.
    `MOJOTREES_CPU_COMPACT_MIN_ROWS=1` takes the gathered `(g, h)` pair buffer
    and a value above the row count reads the gradients through the row ids
    instead; the pair buffer is filled from `hess` on both arms, so a
    two-plane loop that had gone on reading it would produce the honest plane
    and fail here."""
    _reset_env()
    var groups: List[String] = ["1", "2", "4", "16"]
    for i in range(len(groups)):
        _subset_diverges("1", groups[i])
        _subset_diverges("1000000000", groups[i])
    _reset_env()


# ---------------------------------------------------------------------------
# Sibling subtraction
# ---------------------------------------------------------------------------

def test_sibling_subtraction_takes_the_hessian_from_the_counts() raises:
    """`_subtract_histogram_arrays` under `const_h` does not read either
    operand's hessian plane; it writes the integer count difference converted
    to Float64.

    Given two honest three-plane operands built on the doubled hessian, the
    general path computes `Float64(2 * pc) - Float64(2 * cc)`, which is
    exactly `Float64(2 * (pc - cc))`, while the elided path writes
    `Float64(pc - cc)`. Those differ in every cell where the sibling is
    non-empty, and the elided path cannot produce its value from the operands'
    hessian planes at all -- which is the whole proof. Both operands are built
    with the declaration OFF, so the divergence is attributable to the
    subtraction and to nothing upstream of it.
    """
    _reset_env()
    var data = _matrix()
    var grad = _grad()
    var hess = _hess_doubled()
    var rows = _node_rows()

    var parent = build_histogram(data, grad, hess, [], False)
    var child = build_histogram_subset(data, grad, hess, rows, [], False)

    var three = subtract_histogram(parent, child, False)
    var two = subtract_histogram(parent, child, True)

    _assert_untouched_planes_agree(three, two)
    _assert_reconstructed_from_count(two)
    var expected_diverging = 0
    for i in range(_N_FEATURES * _N_BINS):
        var sibling_count = parent.count[i] - child.count[i]
        assert_equal(two.count[i], sibling_count)
        assert_equal(_bits(two.hess[i]), _bits(Float64(sibling_count)))
        assert_equal(_bits(three.hess[i]), _bits(Float64(2 * sibling_count)))
        if sibling_count > 0:
            expected_diverging += 1
    assert_true(expected_diverging > 0)
    assert_equal(_count_diverging_cells(three, two), expected_diverging)
    _reset_env()


# ---------------------------------------------------------------------------
# The control: withdrawing permission puts the divergence back to zero
# ---------------------------------------------------------------------------

def test_the_off_switch_removes_the_divergence_everywhere() raises:
    """`MOJOTREES_CONST_HESSIAN=0` and the same false declaration on the same
    fixtures: the arms are the same bytes again, on the full build at every
    rung, on both gather arms of the subset build, and on the subtraction.

    This is the control that makes every divergence above mean what it claims.
    Divergence with permission and agreement without it is the elision; the
    same divergence under both would be some other instability wearing the
    elision's label. It also pins the off switch itself, which is what a
    bisection of a suspected constant-hessian defect turns first."""
    _reset_env()
    _ = setenv("MOJOTREES_CONST_HESSIAN", "0")
    assert_true(not const_hessian_allowed())

    var data = _matrix()
    var grad = _grad()
    var hess = _hess_doubled()
    var rows = _node_rows()

    var rungs: List[String] = ["1", "2", "4", "8", "16"]
    for i in range(len(rungs)):
        _ = setenv("MOJOTREES_CPU_FEATURE_GROUP", rungs[i])
        var three = build_histogram(data, grad, hess, [], False)
        var two = build_histogram(data, grad, hess, [], True)
        _assert_untouched_planes_agree(three, two)
        _assert_honest_is_double_the_count(two)
        assert_equal(_count_diverging_cells(three, two), 0)
    _ = setenv("MOJOTREES_CPU_FEATURE_GROUP", "")

    var compacts: List[String] = ["1", "1000000000"]
    for i in range(len(compacts)):
        _ = setenv("MOJOTREES_CPU_COMPACT_MIN_ROWS", compacts[i])
        var s3 = build_histogram_subset(data, grad, hess, rows, [], False)
        var s2 = build_histogram_subset(data, grad, hess, rows, [], True)
        _assert_untouched_planes_agree(s3, s2)
        _assert_honest_is_double_the_count(s2)
        assert_equal(_count_diverging_cells(s3, s2), 0)
    _ = setenv("MOJOTREES_CPU_COMPACT_MIN_ROWS", "")

    var parent = build_histogram(data, grad, hess, [], False)
    var child = build_histogram_subset(data, grad, hess, rows, [], False)
    var d3 = subtract_histogram(parent, child, False)
    var d2 = subtract_histogram(parent, child, True)
    _assert_untouched_planes_agree(d3, d2)
    assert_equal(_count_diverging_cells(d3, d2), 0)

    # Permission restored, and the divergence comes back: the switch is the
    # cause, not the order the tests happened to run in.
    _ = setenv("MOJOTREES_CONST_HESSIAN", "1")
    assert_true(const_hessian_allowed())
    var back3 = build_histogram(data, grad, hess, [], False)
    var back2 = build_histogram(data, grad, hess, [], True)
    assert_equal(
        _count_diverging_cells(back3, back2), _N_FEATURES * _N_BINS
    )
    _reset_env()


# ---------------------------------------------------------------------------
# Determinism, and what makes the negative control legitimate
# ---------------------------------------------------------------------------

def test_the_elided_plane_is_identical_at_every_worker_count() raises:
    """`MOJOTREES_NUM_WORKERS` is a scheduling knob and may not reach the
    arithmetic, and the refill is the part of the elision most able to break
    that: it is a second pass over a slice, run on the task that owns the
    slice, so a task boundary that split or duplicated it would show as a
    changed cell.

    Both arms are pinned at 1, 3 and 8 workers with the crossover forced to
    zero so the parallel path is actually taken, and the divergence count is
    pinned too -- a worker count that quietly stopped eliding would report
    zero divergences rather than a wrong number.
    """
    _reset_env()
    _ = setenv("MOJOTREES_PARALLEL_MIN_OPS", "1")
    var data = _matrix()
    var grad = _grad()
    var hess = _hess_doubled()
    var rows = _node_rows()

    _ = setenv("MOJOTREES_NUM_WORKERS", "1")
    var base_full = build_histogram(data, grad, hess, [], True)
    var base_sub = build_histogram_subset(data, grad, hess, rows, [], True)
    var base_three = build_histogram(data, grad, hess, [], False)
    assert_equal(
        _count_diverging_cells(base_three, base_full), _N_FEATURES * _N_BINS
    )

    var workers: List[String] = ["3", "8"]
    for i in range(len(workers)):
        _ = setenv("MOJOTREES_NUM_WORKERS", workers[i])
        var full = build_histogram(data, grad, hess, [], True)
        var sub = build_histogram_subset(data, grad, hess, rows, [], True)
        var three = build_histogram(data, grad, hess, [], False)
        for j in range(_N_FEATURES * _N_BINS):
            assert_equal(_bits(full.grad[j]), _bits(base_full.grad[j]))
            assert_equal(_bits(full.hess[j]), _bits(base_full.hess[j]))
            assert_equal(full.count[j], base_full.count[j])
            assert_equal(_bits(sub.grad[j]), _bits(base_sub.grad[j]))
            assert_equal(_bits(sub.hess[j]), _bits(base_sub.hess[j]))
            assert_equal(sub.count[j], base_sub.count[j])
            assert_equal(_bits(three.hess[j]), _bits(base_three.hess[j]))
        assert_equal(
            _count_diverging_cells(three, full), _N_FEATURES * _N_BINS
        )
    _reset_env()


def test_the_verifier_would_refuse_this_files_fixtures() raises:
    """What makes a false declaration a legitimate instrument rather than a
    latent defect: the verifier is off by default, and turning it on turns
    every fixture in this file into a raise at the build.

    Stated here so that a reader who reaches the divergence assertions above
    and wonders whether the package tolerates a false declaration has the
    answer next to them. It does not; it declines to pay for the check unless
    asked, which is the gap this file borrows.
    """
    _reset_env()
    var data = _matrix()
    var grad = _grad()
    var rows = _node_rows()
    _ = setenv("MOJOTREES_CONST_HESSIAN_VERIFY", "1")

    var arrays: List[List[Float64]] = [_hess_doubled(), _hess_varied()]
    for i in range(len(arrays)):
        var raised_full = False
        try:
            _ = build_histogram(data, grad, arrays[i], [], True)
        except:
            raised_full = True
        assert_true(raised_full)

        var raised_subset = False
        try:
            _ = build_histogram_subset(data, grad, arrays[i], rows, [], True)
        except:
            raised_subset = True
        assert_true(raised_subset)

        # Undeclared, so unchecked: the verifier checks a declaration and
        # never the data.
        _ = build_histogram(data, grad, arrays[i], [], False)
    _reset_env()


def test_a_true_declaration_still_agrees() raises:
    """The other side of the instrument, kept here so this file is
    self-contained about what it is claiming. On data where the declaration is
    TRUE the two arms are the same bytes, which is `test_const_hessian.mojo`'s
    claim and remains the correctness contract; this file adds that the
    agreement is produced by a loop that ran, not by a loop that was never
    there."""
    _reset_env()
    var data = _matrix()
    var grad = _grad()
    var hess = List[Float64](capacity=_N_ROWS)
    for _ in range(_N_ROWS):
        hess.append(CONSTANT_HESSIAN)

    var three = build_histogram(data, grad, hess, [], False)
    var two = build_histogram(data, grad, hess, [], True)
    _assert_untouched_planes_agree(three, two)
    assert_equal(_count_diverging_cells(three, two), 0)
    _assert_reconstructed_from_count(two)
    _assert_reconstructed_from_count(three)
    _reset_env()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
