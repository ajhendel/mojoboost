"""The row-major bin view, and the one claim it makes.

`BinnedMatrix.build_row_major` adds a second copy of the same bin ids laid out
by row instead of by feature, with features of at most 16 bins packed two to a
byte. `histogram.build_histogram_subset_row_major*` accumulates from it.

**The whole correctness claim is that this moves no bits**, and that is what
this file asserts. A histogram accumulated from either layout visits the same
rows in the same order and adds the same Float64 into the same cell; only the
address the bin id is loaded from changes. So the two histograms must be equal
*to the bit*, on every plane, with no tolerance anywhere -- `to_bits()` on the
Float64 planes and integer equality on the counts.

Nothing here is a timing and nothing here may be read as evidence that either
layout is faster. The lane that wrote it had no instrument and took no
measurement; `bench/` is where that question is settled.

What is covered, and why each case is here rather than assumed:

1. **The view is the same information.** `row_bin_at(r, f) == bin_at(r, f)`
   for every cell of several shapes, including a matrix where every feature
   packs, one where none does, and one that mixes them.

2. **Packing is lossless and its boundary is where it is claimed to be.** A
   feature realizing exactly 16 bins is packed (16 ids are 0..15, which is a
   nibble); a feature realizing 17 is not (id 16 does not fit). Both are
   asserted on the same matrix, and the realized `row_stride` is asserted
   against the arithmetic `unpacked + (packed + 1) / 2` rather than against a
   remembered number.

3. **Bit-identity, on both kernels.** The row-major path has two kernels and
   the plan picks between them, so a test that only ever hit one would be
   asserting half of it. The blocked case asserts `plan.blocked()` and the
   unblocked case asserts `not plan.blocked()` *before* comparing, so neither
   can silently degrade into the other and pass. Both are run over a node whose
   row list is a scattered subset, over a feature subset, and with the
   constant-hessian specialization on and off.

4. **Determinism across worker counts.** The row-major arm at 1, 3 and 8
   workers must produce one histogram, bit for bit, and it must be the
   feature-major one.

5. **The layout gate reports itself.** `build_histogram_subset_by_layout_into_
   scratch` returns the layout that ran, and the test asserts that return
   value. A request for a layout that does not exist degrades to feature-major,
   and the test asserts the degradation rather than the request -- this
   repository has already shipped a test comparing two arms that were equal
   whether or not the optimization fired.
"""

from std.os import setenv
from std.testing import assert_equal, assert_true, TestSuite

from mojotrees.apple_cpu_policy import (
    BIN_LAYOUT_FEATURE_MAJOR,
    BIN_LAYOUT_ROW_MAJOR,
    CpuProfile,
    derive_accumulation_plan,
)
from mojotrees.binning import BinnedMatrix
from mojotrees.histogram import (
    Histogram,
    build_histogram_subset,
    build_histogram_subset_by_layout_into_scratch,
    build_histogram_subset_row_major,
)
from mojotrees.parallel import DispatchSettings


def _matrix(n_rows: Int, widths: List[Int], n_bins: Int) raises -> BinnedMatrix:
    """A binned matrix whose feature f realizes exactly `widths[f]` bins.

    Every bin of every feature is hit, provided `n_rows >= widths[f]`: the row
    index sweeps the residues one at a time, so the *realized* width is the
    declared one and the packing decision below is about the number the test
    says it is about. A multiplier here instead of a stride of one would skip
    residues whenever it shared a factor with the width, and the realized width
    would quietly stop being the declared one. The pattern is arithmetic rather
    than random so the fixture is the same on every machine.
    """
    var nf = len(widths)
    var bins = List[UInt8](capacity=n_rows * nf)
    bins.resize(n_rows * nf, UInt8(0))
    for f in range(nf):
        var w = widths[f]
        for r in range(n_rows):
            bins[f * n_rows + r] = UInt8((r + 7 * f) % w)
    return BinnedMatrix(bins^, n_rows, nf, n_bins)


def _gradients(n_rows: Int) -> List[Float64]:
    """Gradients no two of which are equal, so a histogram cell is a genuine
    sum of distinct Float64 and a reassociation would show."""
    var g = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        g.append(1.0 / Float64(r + 3) - 0.25 * Float64((r % 7) + 1))
    return g^


def _hessians(n_rows: Int, constant: Bool) -> List[Float64]:
    var h = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        h.append(1.0 if constant else 0.5 + 1.0 / Float64((r % 11) + 2))
    return h^


def _every_k(n_rows: Int, k: Int, offset: Int) -> List[Int]:
    """A node's row ids: ascending, scattered, not the whole matrix."""
    var rows = List[Int]()
    var r = offset
    while r < n_rows:
        rows.append(r)
        r += k
    return rows^


def _all_rows(n_rows: Int) -> List[Int]:
    var rows = List[Int](capacity=n_rows)
    for r in range(n_rows):
        rows.append(r)
    return rows^


def _assert_same_bits(a: Histogram, b: Histogram, label: String) raises:
    """Equal to the bit on all three planes. No tolerance: a comparison that
    needed one would not establish what this file claims."""
    assert_equal(a.n_cells(), b.n_cells(), label)
    var nonzero = 0
    for i in range(a.n_cells()):
        assert_equal(
            a.grad_at(i).to_bits().cast[DType.uint64](),
            b.grad_at(i).to_bits().cast[DType.uint64](),
            String(label, " grad cell ", i),
        )
        assert_equal(
            a.hess_at(i).to_bits().cast[DType.uint64](),
            b.hess_at(i).to_bits().cast[DType.uint64](),
            String(label, " hess cell ", i),
        )
        assert_equal(
            a.count_at(i), b.count_at(i), String(label, " count cell ", i)
        )
        if a.count_at(i) != 0:
            nonzero += 1
    # A pair of all-zero histograms would satisfy every assertion above and
    # establish nothing at all.
    assert_true(nonzero > 0, String(label, " compared two empty histograms"))


def _mixed_widths(nf: Int) raises -> List[Int]:
    """Widths that put some features either side of the packing boundary."""
    var w = List[Int]()
    for f in range(nf):
        if f % 3 == 0:
            w.append(4)
        elif f % 3 == 1:
            w.append(16)
        else:
            w.append(40)
    return w^


def test_row_major_view_is_the_same_information() raises:
    """`row_bin_at` equals `bin_at` for every cell, in all three packing
    regimes: every feature packed, none packed, and mixed."""
    var cases = List[List[Int]]()
    cases.append([2, 5, 9, 16, 3, 16])
    cases.append([17, 40, 255, 200])
    cases.append(_mixed_widths(9))
    for c in range(len(cases)):
        ref widths = cases[c]
        var data = _matrix(300, widths, 255)
        assert_true(not data.has_row_major(), "view exists before it is built")
        data.build_row_major()
        assert_true(data.has_row_major(), "view missing after build")
        for f in range(data.n_features):
            assert_equal(
                data.feature_bins[f],
                widths[f],
                String("case ", c, " realized width of feature ", f),
            )
            for r in range(data.n_rows):
                assert_equal(
                    data.row_bin_at(r, f),
                    data.bin_at(r, f),
                    String("case ", c, " cell (", r, ", ", f, ")"),
                )


def test_packing_boundary_is_sixteen_bins() raises:
    """16 bins pack into a nibble, 17 do not, and the stride is the
    arithmetic rather than a remembered number."""
    var widths = List[Int]()
    widths.append(16)
    widths.append(17)
    widths.append(16)
    widths.append(255)
    widths.append(1)
    var data = _matrix(300, widths, 255)
    data.build_row_major()

    assert_equal(data.row_mask[0], 15, "16 bins should pack")
    assert_equal(data.row_mask[1], 255, "17 bins must not pack")
    assert_equal(data.row_mask[2], 15, "16 bins should pack")
    assert_equal(data.row_mask[3], 255, "255 bins must not pack")
    assert_equal(data.row_mask[4], 15, "1 bin should pack")
    assert_equal(data.packed_feature_count(), 3, "packed feature count")
    # Two whole bytes for the two unpacked features, plus ceil(3 / 2) for the
    # three packed ones.
    assert_equal(data.row_stride, 2 + 2, "row_stride")
    assert_equal(
        data.row_major_bytes(), 300 * 4, "row-major bytes for this shape"
    )
    # Two packed features must share a byte, and the third must start a new
    # one on its low nibble.
    assert_equal(data.row_byte[0], data.row_byte[2], "packed pair shares byte")
    assert_equal(data.row_shift[0], 0, "first of a pair is the low nibble")
    assert_equal(data.row_shift[2], 4, "second of a pair is the high nibble")
    assert_equal(data.row_shift[4], 0, "an odd packed feature starts a byte")
    assert_true(
        data.row_byte[4] != data.row_byte[0], "third packed feature's byte"
    )
    # Cumulative offsets are the compact histogram's boundaries.
    assert_equal(data.compact_bin_count(), 16 + 17 + 16 + 255 + 1, "compact")
    assert_equal(data.bin_offset[0], 0, "first offset")
    assert_equal(data.bin_offset[2], 33, "third offset")


def test_all_packable_halves_the_stride() raises:
    """Every feature under the boundary gives `(n + 1) / 2` bytes a row, which
    is the memory the packing recovers."""
    var widths = List[Int]()
    for _ in range(10):
        widths.append(9)
    var data = _matrix(48, widths, 16)
    data.build_row_major()
    assert_equal(data.packed_feature_count(), 10, "all ten pack")
    assert_equal(data.row_stride, 5, "ten packed features are five bytes")
    for f in range(10):
        for r in range(48):
            assert_equal(
                data.row_bin_at(r, f), data.bin_at(r, f), "packed round trip"
            )


def _plan_is_blocked(
    n_features: Int, n_active: Int, n_bins: Int, n_sub: Int
) raises -> Bool:
    """The plan the builders will derive for this node, so a fixture can assert
    which kernel it is about to exercise instead of hoping."""
    var plan = derive_accumulation_plan(
        CpuProfile.detect(), n_features, n_active, n_bins, n_sub, True
    )
    return plan.blocked()


def _compare_layouts(
    data: BinnedMatrix,
    rows: List[Int],
    features: List[Int],
    const_h: Bool,
    label: String,
) raises:
    var grad = _gradients(data.n_rows)
    var hess = _hessians(data.n_rows, const_h)
    var feature_major = build_histogram_subset(
        data, grad, hess, rows, features, const_h
    )
    var row_major = build_histogram_subset_row_major(
        data, grad, hess, rows, features, const_h
    )
    _assert_same_bits(feature_major, row_major, label)


def test_unblocked_node_is_bit_identical() raises:
    """A node the plan does not block, on the feature-partition kernel."""
    var widths = _mixed_widths(7)
    var data = _matrix(300, widths, 64)
    data.build_row_major()
    var rows = _every_k(300, 3, 1)
    assert_true(
        not _plan_is_blocked(7, 7, 64, len(rows)),
        "fixture meant to exercise the unblocked kernel is blocked",
    )
    _compare_layouts(data, rows, [], False, "unblocked all features")
    _compare_layouts(data, rows, [], True, "unblocked const hessian")
    _compare_layouts(data, rows, [1, 2, 5], False, "unblocked subset")
    _compare_layouts(data, rows, [4], False, "unblocked single feature")
    _compare_layouts(
        data, _all_rows(300), [], False, "unblocked whole matrix"
    )


def test_blocked_node_is_bit_identical() raises:
    """A node the plan blocks, on the row-blocked kernel with the compact
    line-padded private partials and the bin-range fold."""
    var widths = _mixed_widths(11)
    var data = _matrix(9000, widths, 40)
    data.build_row_major()
    var rows = _every_k(9000, 2, 0)
    assert_true(
        _plan_is_blocked(11, 11, 40, len(rows)),
        "fixture meant to exercise the blocked kernel is not blocked",
    )
    _compare_layouts(data, rows, [], False, "blocked all features")
    _compare_layouts(data, rows, [], True, "blocked const hessian")
    assert_true(
        _plan_is_blocked(11, 4, 40, len(rows)),
        "blocked subset fixture is not blocked",
    )
    _compare_layouts(data, rows, [0, 3, 7, 10], False, "blocked subset")
    _compare_layouts(
        data, _all_rows(9000), [], False, "blocked whole matrix"
    )


def test_wide_bins_are_bit_identical() raises:
    """The regime the headline shape is in: no feature packs, and the
    realized widths are close to the bin budget."""
    var widths = List[Int]()
    for f in range(6):
        widths.append(200 + 9 * f)
    var data = _matrix(4000, widths, 255)
    data.build_row_major()
    assert_equal(data.packed_feature_count(), 0, "nothing should pack here")
    assert_equal(data.row_stride, 6, "one byte a feature")
    var rows = _every_k(4000, 5, 2)
    _compare_layouts(data, rows, [], False, "wide bins")
    _compare_layouts(data, rows, [], True, "wide bins const hessian")


def _plan_compacts(
    n_features: Int, n_active: Int, n_bins: Int, n_sub: Int
) raises -> Bool:
    """Whether this node will gather its derivatives into the pair buffer."""
    var plan = derive_accumulation_plan(
        CpuProfile.detect(), n_features, n_active, n_bins, n_sub, True
    )
    return plan.compact_rows


def test_both_gather_regimes_agree() raises:
    """The two layouts must agree whether or not the node gathers.

    This is the axis that broke when the derivative precision changed under
    this lane. A gathered row reads its `(gradient, hessian)` as two Float32
    halves of one Float64 word; an ungathered row reads the Float64 arrays and
    narrows with `score_t`. Those two have to deliver the same Float64, or
    `compact_rows` -- a *policy* decision -- would change a cell, and no policy
    decision in the accumulation is allowed to.

    Both regimes are forced explicitly and the plan is asserted to have
    actually changed between them, because two arms that were secretly the
    same arm would pass this without establishing anything.
    """
    var widths = _mixed_widths(7)
    var data = _matrix(1200, widths, 64)
    data.build_row_major()
    var rows = _every_k(1200, 3, 1)
    var grad = _gradients(1200)
    var hess = _hessians(1200, False)
    assert_true(
        not _plan_is_blocked(7, 7, 64, len(rows)),
        "the gather regime only varies on an unblocked node; this one blocks",
    )

    _ = setenv("MOJOTREES_CPU_COMPACT_MIN_ROWS", "100000000")
    assert_true(
        not _plan_compacts(7, 7, 64, len(rows)), "the gather did not turn off"
    )
    _compare_layouts(data, rows, [], False, "ungathered")
    _compare_layouts(data, rows, [], True, "ungathered const hessian")
    var ungathered = build_histogram_subset(
        data, grad, hess, rows, [], False
    )

    _ = setenv("MOJOTREES_CPU_COMPACT_MIN_ROWS", "1")
    assert_true(
        _plan_compacts(7, 7, 64, len(rows)), "the gather did not turn on"
    )
    _compare_layouts(data, rows, [], False, "gathered")
    _compare_layouts(data, rows, [], True, "gathered const hessian")
    var gathered = build_histogram_subset(data, grad, hess, rows, [], False)

    _ = setenv("MOJOTREES_CPU_COMPACT_MIN_ROWS", "")
    _assert_same_bits(ungathered, gathered, "gather regimes")


def test_blocked_subset_under_constant_hessian() raises:
    """The blocked kernel's two specializations crossed with a feature subset.

    The compact private extent depends on which features are active and the
    addressed cell stride depends on `const_h`, so the two interact: a subset
    changes `bin_offset` and the constant-hessian arm changes the stride those
    offsets are scaled by.
    """
    var widths = _mixed_widths(11)
    var data = _matrix(9000, widths, 40)
    data.build_row_major()
    var rows = _every_k(9000, 2, 0)
    assert_true(
        _plan_is_blocked(11, 5, 40, len(rows)),
        "blocked constant-hessian subset fixture is not blocked",
    )
    _compare_layouts(
        data, rows, [1, 2, 4, 8, 9], True, "blocked subset const hessian"
    )
    _compare_layouts(
        data, rows, [0, 10], True, "blocked narrow subset const hessian"
    )


def test_row_major_is_deterministic_across_workers() raises:
    """One histogram at 1, 3 and 8 workers, and it is the feature-major one.

    Worker count is a schedule: it moves which core sums a cell and never the
    order the cells are summed in. The blocked fixture is the one that matters
    here, since that is the kernel with private partials and a fold.
    """
    var widths = _mixed_widths(9)
    var data = _matrix(9000, widths, 40)
    data.build_row_major()
    var rows = _every_k(9000, 2, 1)
    assert_true(
        _plan_is_blocked(9, 9, 40, len(rows)),
        "worker-count fixture is not blocked",
    )
    var grad = _gradients(data.n_rows)
    var hess = _hessians(data.n_rows, False)

    _ = setenv("MOJOTREES_NUM_WORKERS", "1")
    var reference = build_histogram_subset(data, grad, hess, rows, [], False)
    var workers = ["1", "3", "8", ""]
    for i in range(len(workers)):
        _ = setenv("MOJOTREES_NUM_WORKERS", workers[i])
        var got = build_histogram_subset_row_major(
            data, grad, hess, rows, [], False
        )
        _assert_same_bits(
            reference, got, String("row-major at workers=", workers[i])
        )
    _ = setenv("MOJOTREES_NUM_WORKERS", "")


def test_layout_gate_reports_which_kernel_ran() raises:
    """The return value is the observable, and the degradation is asserted
    rather than the request."""
    var widths = _mixed_widths(6)
    var data = _matrix(600, widths, 40)
    var rows = _every_k(600, 2, 0)
    var grad = _gradients(data.n_rows)
    var hess = _hessians(data.n_rows, False)
    var pairs = List[Float64]()
    var out = Histogram.zeroed(data.n_features, data.n_bins)

    # No view built: a request for the row-major layout must degrade, and say
    # so, rather than raise or silently claim it ran.
    var got = build_histogram_subset_by_layout_into_scratch(
        out, pairs, data, grad, hess, rows, 0, len(rows),
        BIN_LAYOUT_ROW_MAJOR, [], False,
    )
    assert_equal(
        got, BIN_LAYOUT_FEATURE_MAJOR, "row-major without a view must degrade"
    )
    var degraded = out.copy()

    data.build_row_major()
    var ran_row = build_histogram_subset_by_layout_into_scratch(
        out, pairs, data, grad, hess, rows, 0, len(rows),
        BIN_LAYOUT_ROW_MAJOR, [], False,
    )
    assert_equal(ran_row, BIN_LAYOUT_ROW_MAJOR, "row-major should have run")
    _assert_same_bits(degraded, out, "gate arms")

    var ran_feature = build_histogram_subset_by_layout_into_scratch(
        out, pairs, data, grad, hess, rows, 0, len(rows),
        BIN_LAYOUT_FEATURE_MAJOR, [], False,
    )
    assert_equal(
        ran_feature,
        BIN_LAYOUT_FEATURE_MAJOR,
        "an explicit feature-major request must not be overridden",
    )
    _assert_same_bits(degraded, out, "gate arms after explicit feature-major")


def test_row_major_builder_refuses_a_matrix_without_the_view() raises:
    """Reading the other layout silently is what makes a benchmark arm
    indistinguishable from its control, so the direct entry point raises."""
    var widths = _mixed_widths(4)
    var data = _matrix(200, widths, 40)
    var grad = _gradients(data.n_rows)
    var hess = _hessians(data.n_rows, False)
    var raised = False
    try:
        _ = build_histogram_subset_row_major(
            data, grad, hess, _all_rows(200), [], False
        )
    except:
        raised = True
    assert_true(raised, "the row-major builder accepted a matrix with no view")


def test_dropping_the_view_releases_it() raises:
    """`drop_row_major` is how a caller with a memory ceiling gets the bytes
    back, so it must actually leave the matrix in the un-built state."""
    var widths = _mixed_widths(5)
    var data = _matrix(128, widths, 40)
    data.build_row_major()
    assert_true(data.row_major_bytes() > 0, "view has no bytes")
    data.drop_row_major()
    assert_true(not data.has_row_major(), "view survived the drop")
    assert_equal(data.row_major_bytes(), 0, "bytes survived the drop")
    # And the feature-major matrix is untouched by any of it.
    for f in range(data.n_features):
        for r in range(data.n_rows):
            assert_equal(
                data.bin_at(r, f),
                Int(UInt8((r + 7 * f) % widths[f])),
                "feature-major matrix disturbed",
            )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
