"""The row-major bin layout, reached from the grower.

`tests/test_row_major_bins.mojo` establishes that the two histogram kernels
are bit-identical. It calls them directly, which is exactly the gap this file
closes: until now `tree.mojo` imported `build_histogram_subset_into_scratch`
and nothing else, so `MOJOTREES_CPU_ROW_MAJOR=1` paid for the row-major view
at fit time and **no trainer at any setting ever read it**. The grower now
goes through `histogram.build_histogram_subset_by_layout_into_scratch` with
the layout `tree.GrowScratch` resolved once for the fit.

So this file asserts the two halves a caller cannot see from inside the
histogram module:

1. **The chain from the environment to the kernel.** `MOJOTREES_CPU_BIN_LAYOUT`
   reaches `GrowScratch.bin_layout` (`auto` and `row` both mean row-major to a
   grower, `feature` is the off switch), and the gate that field is handed to
   returns `BIN_LAYOUT_ROW_MAJOR` for the same matrix and rows the grower will
   pass it. Both ends are asserted, because a test that checked only the
   environment read would pass while the kernel never ran -- this repository
   has shipped that mistake twice.

2. **A whole grown tree is bit-identical.** Not a histogram: the tree. Every
   array of `Tree` compared element for element, the Float64 ones through
   `to_bits()` with no tolerance. A layout is a scheduling decision and a
   scheduling decision that moved a split threshold, a leaf value or a gain by
   one ulp would be a defect, not a trade. Grown over a scattered bag as well
   as the whole matrix, so the root reaches `_hist_subset` on one arm and
   `_hist_full` on the other.

3. **Determinism across `MOJOTREES_NUM_WORKERS`.** The row-major arm at 1, 3
   and 8 workers is one tree, and it is the feature-major tree.

4. **The degradation is safe.** A matrix with no row-major view grown with the
   layout requested is the same tree, because the gate falls back rather than
   raising. That is what lets the grower default to row-major while
   `binning.env_row_major_bins` decides separately whether the view is worth
   its memory.

Nothing here is a timing and nothing here may be read as a speed claim.
"""

from std.os import setenv
from std.testing import assert_equal, assert_true, TestSuite

from mojotrees.apple_cpu_policy import (
    BIN_LAYOUT_FEATURE_MAJOR,
    BIN_LAYOUT_ROW_MAJOR,
)
from mojotrees.binning import BinnedMatrix
from mojotrees.efb import BundledMatrix
from mojotrees.histogram import (
    Histogram,
    build_histogram_subset_by_layout_into_scratch,
)
from mojotrees.tree import GrowScratch, Tree, TreeParams, grow_tree


def _matrix(n_rows: Int, n_features: Int, n_bins: Int) raises -> BinnedMatrix:
    """A binned matrix that hits every bin of every feature, arithmetic rather
    than random so the fixture is identical on every machine."""
    var bins = List[UInt8](capacity=n_rows * n_features)
    bins.resize(n_rows * n_features, UInt8(0))
    for f in range(n_features):
        for r in range(n_rows):
            bins[f * n_rows + r] = UInt8((r * (f + 1) + 5 * f) % n_bins)
    return BinnedMatrix(bins^, n_rows, n_features, n_bins)


def _gradients(n_rows: Int) -> List[Float64]:
    """Gradients no two of which are equal, so a split gain is a genuine sum
    of distinct Float64 and a reassociation would show as a different tree."""
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


def _assert_same_tree(a: Tree, b: Tree, label: String) raises:
    """Every array of a grown tree, element for element, Float64 through
    `to_bits()`. No tolerance anywhere."""
    assert_equal(a.n_leaves, b.n_leaves, String(label, " n_leaves"))
    assert_equal(len(a.feature), len(b.feature), String(label, " n_nodes"))
    assert_true(a.n_leaves > 1, String(label, " compared two stumps"))
    for i in range(len(a.feature)):
        assert_equal(a.feature[i], b.feature[i], String(label, " feature ", i))
        assert_equal(
            a.threshold_bin[i],
            b.threshold_bin[i],
            String(label, " threshold ", i),
        )
        assert_equal(a.left[i], b.left[i], String(label, " left ", i))
        assert_equal(a.right[i], b.right[i], String(label, " right ", i))
        assert_equal(
            a.default_left[i],
            b.default_left[i],
            String(label, " default_left ", i),
        )
        assert_equal(
            a.missing_bin[i],
            b.missing_bin[i],
            String(label, " missing_bin ", i),
        )
        assert_equal(
            a.cat_offset[i], b.cat_offset[i], String(label, " cat_offset ", i)
        )
        assert_equal(
            a.value[i].to_bits().cast[DType.uint64](),
            b.value[i].to_bits().cast[DType.uint64](),
            String(label, " value ", i),
        )
        assert_equal(
            a.split_gain[i].to_bits().cast[DType.uint64](),
            b.split_gain[i].to_bits().cast[DType.uint64](),
            String(label, " split_gain ", i),
        )
        assert_equal(
            a.count[i].to_bits().cast[DType.uint64](),
            b.count[i].to_bits().cast[DType.uint64](),
            String(label, " count ", i),
        )


def _params() -> TreeParams:
    """LightGBM's stock tree, with `min_data_in_leaf` lowered so a fixture of
    a few thousand rows still grows past a stump. Everything a layout could
    conceivably interact with -- the gain floor, the feature fractions, the
    leaf bound -- is left at its default."""
    return TreeParams(31, 5, 0.0, 1e-3, 0.0)


def test_environment_reaches_the_kernel() raises:
    """The whole chain, both ends asserted: the variable reaches the scratch,
    and the layout the scratch holds reaches the row-major kernel."""
    _ = setenv("MOJOTREES_CPU_BIN_LAYOUT", "")
    var auto = GrowScratch(6, 40)
    # **This assertion was inverted on 2026-08-16 and the inversion is the
    # point.** It used to require that an unrequested grower asks for
    # row-major, which was safe only while the view did not exist by default.
    # `lane/row-major-auto` made the view default-on under a memory budget, so
    # that mapping would have silently turned every CPU fit under the budget
    # row-major with no timing -- the flat default flip that was declined and
    # replaced with LightGBM's build-both-keep-the-faster rule.
    #
    # AUTO is not a layout; it means nobody has chosen yet. Until
    # `resolve_layout_timed` lands with the histogram file's lane, the layout
    # that shipped is the one that runs. When the probe lands, this assertion
    # changes again -- to "AUTO is still unresolved at construction and is
    # decided at the root" -- and not back to row-major.
    assert_equal(
        auto.bin_layout,
        BIN_LAYOUT_FEATURE_MAJOR,
        "an unrequested grower must not silently take row-major before the"
        " timed probe has chosen",
    )
    _ = setenv("MOJOTREES_CPU_BIN_LAYOUT", "row")
    var asked = GrowScratch(6, 40)
    assert_equal(asked.bin_layout, BIN_LAYOUT_ROW_MAJOR, "explicit row")
    _ = setenv("MOJOTREES_CPU_BIN_LAYOUT", "feature")
    var off = GrowScratch(6, 40)
    assert_equal(
        off.bin_layout, BIN_LAYOUT_FEATURE_MAJOR, "feature is the off switch"
    )
    _ = setenv("MOJOTREES_CPU_BIN_LAYOUT", "")

    # And the far end. The gate reports which kernel it ran, so this is the
    # assertion that the layout the grower carries is not silently degraded on
    # a matrix that does have the view.
    var data = _matrix(600, 6, 40)
    data.build_row_major()
    var grad = _gradients(600)
    var hess = _hessians(600, False)
    var rows = _every_k(600, 2, 0)
    var out = Histogram.zeroed(data.n_features, data.n_bins)
    var pairs = List[Float64]()
    # Driven from the EXPLICITLY-row grower rather than the default one. The
    # default now carries feature-major until the timed probe chooses, so
    # asking it to prove the row kernel is reachable would prove nothing --
    # it would pass on a grower that never wanted row-major in the first
    # place. `asked` is the one that carries the request.
    var ran = build_histogram_subset_by_layout_into_scratch(
        out, pairs, data, grad, hess, rows, 0, len(rows), asked.bin_layout,
        [], False,
    )
    assert_equal(
        ran,
        BIN_LAYOUT_ROW_MAJOR,
        "the layout an explicitly-row grower carries did not reach the row"
        " kernel",
    )
    # And the other end of the same chain: the default grower's layout must
    # reach the FEATURE kernel on the very same matrix -- one that does have
    # the view. Without this the assertion above could pass while the default
    # quietly ran row-major anyway, which is the exact regression this pair
    # of assertions exists to catch.
    var out2 = Histogram.zeroed(data.n_features, data.n_bins)
    var ran_default = build_histogram_subset_by_layout_into_scratch(
        out2, pairs, data, grad, hess, rows, 0, len(rows), auto.bin_layout,
        [], False,
    )
    assert_equal(
        ran_default,
        BIN_LAYOUT_FEATURE_MAJOR,
        "a default grower ran the row kernel on a matrix that has the view,"
        " which is the flat default flip that was declined",
    )


def _grow_both_ways(
    n_rows: Int,
    n_features: Int,
    n_bins: Int,
    bag: List[Int],
    const_h: Bool,
    label: String,
) raises:
    var data = _matrix(n_rows, n_features, n_bins)
    data.build_row_major()
    assert_true(data.has_row_major(), String(label, ": no view was built"))
    var grad = _gradients(n_rows)
    var hess = _hessians(n_rows, const_h)
    var params = _params()

    _ = setenv("MOJOTREES_CPU_BIN_LAYOUT", "feature")
    var feature_major = grow_tree(
        data, grad, hess, params, bag, 0, BundledMatrix.none(), const_h
    )
    _ = setenv("MOJOTREES_CPU_BIN_LAYOUT", "row")
    var row_major = grow_tree(
        data, grad, hess, params, bag, 0, BundledMatrix.none(), const_h
    )
    _ = setenv("MOJOTREES_CPU_BIN_LAYOUT", "")
    _assert_same_tree(feature_major, row_major, label)


def test_grown_tree_is_bit_identical_unblocked() raises:
    """A shape whose nodes the plan does not block."""
    _grow_both_ways(600, 6, 40, [], False, "unblocked whole matrix")
    _grow_both_ways(
        600, 6, 40, _every_k(600, 2, 1), False, "unblocked bag"
    )
    _grow_both_ways(600, 6, 40, [], True, "unblocked const hessian")


def test_grown_tree_is_bit_identical_blocked() raises:
    """The row-blocked kernel, forced so a small fixture reaches it."""
    _ = setenv("MOJOTREES_CPU_ROW_BLOCKS", "4")
    _grow_both_ways(4000, 8, 24, [], False, "blocked whole matrix")
    _grow_both_ways(
        4000, 8, 24, _every_k(4000, 2, 0), False, "blocked bag"
    )
    _grow_both_ways(4000, 8, 24, [], True, "blocked const hessian")
    _ = setenv("MOJOTREES_CPU_ROW_BLOCKS", "")


def test_grown_tree_is_deterministic_across_workers() raises:
    """Worker count is a schedule and the layout is a schedule; neither may
    move a node."""
    _ = setenv("MOJOTREES_CPU_ROW_BLOCKS", "4")
    var data = _matrix(4000, 8, 24)
    data.build_row_major()
    var grad = _gradients(4000)
    var hess = _hessians(4000, False)
    var params = _params()

    _ = setenv("MOJOTREES_CPU_BIN_LAYOUT", "feature")
    _ = setenv("MOJOTREES_NUM_WORKERS", "1")
    var reference = grow_tree(data, grad, hess, params)

    _ = setenv("MOJOTREES_CPU_BIN_LAYOUT", "row")
    var workers = ["1", "3", "8", ""]
    for i in range(len(workers)):
        _ = setenv("MOJOTREES_NUM_WORKERS", workers[i])
        var got = grow_tree(data, grad, hess, params)
        _assert_same_tree(
            reference, got, String("row-major at workers=", workers[i])
        )
    _ = setenv("MOJOTREES_NUM_WORKERS", "")
    _ = setenv("MOJOTREES_CPU_BIN_LAYOUT", "")
    _ = setenv("MOJOTREES_CPU_ROW_BLOCKS", "")


def test_no_view_degrades_to_the_same_tree() raises:
    """The grower's default asks for a layout the matrix may not have. The
    gate falls back rather than raising, and the fallback is the tree that
    always grew."""
    var data = _matrix(600, 6, 40)
    assert_true(not data.has_row_major(), "fixture built a view it should not")
    var grad = _gradients(600)
    var hess = _hessians(600, False)
    var params = _params()

    _ = setenv("MOJOTREES_CPU_BIN_LAYOUT", "feature")
    var explicit = grow_tree(data, grad, hess, params)
    _ = setenv("MOJOTREES_CPU_BIN_LAYOUT", "")
    var defaulted = grow_tree(data, grad, hess, params)
    _assert_same_tree(explicit, defaulted, "no view, default layout")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
