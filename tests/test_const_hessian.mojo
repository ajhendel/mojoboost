"""The constant-hessian specialization, on both backends.

Four of the built-in objectives write the same hessian into every row when
the fit carries no sample weights, and that value is exactly 1.0. Where that
holds, the hessian plane of a histogram cell is the count plane and storing
both is waste. `histogram.mojo` and the range histogram kernels in
`gpu_active_rows.mojo` can be told so, and then stop accumulating the plane
and reconstruct it at the end of the pass.

**Every assertion here is exact, and that is the whole point of the file.**
The specialization is only allowed to save work; it is not allowed to move a
value, not even in the last bit, because a histogram cell feeds a split gain
which feeds a tree which feeds the next round's gradients (see
`docs/NUMERICS.md` section 2). So the Float64 planes are compared through
`to_bits` as integers rather than to any tolerance, and the fixed-point device
words are compared as the Int32 they are.

The comparisons are all of the same shape: run the identical work twice on the
identical inputs, once with the specialization declared and once without, and
require the two results to be the same bytes. That is a stronger check than
comparing either arm against a hand-written reference, because it holds the
inputs, the schedule, the launch geometry, and the toolchain fixed and varies
only the thing under test.

What is NOT asserted here, and is asserted nowhere: that the specialization
is faster. No timing is taken in this file and none should be inferred from
it. The saving is structural (one plane fewer accumulated, zeroed, and, on the
tiled device path, written and reduced) and its wall-clock consequence is a
measurement this lane did not make.

The device half skips, passing, when there is no accelerator, so this file
belongs in the CPU set and enforces the host half of the contract on a
CPU-only runner.
"""

from std.os import setenv
from std.sys import has_accelerator
from std.testing import assert_equal, assert_true, TestSuite

from mojotrees.binning import BinnedMatrix, bin_equal_width
from mojotrees.gpu_tiling import STRATEGY_ATOMIC, STRATEGY_TILED
from mojotrees.histogram import (
    CONSTANT_HESSIAN,
    Histogram,
    build_histogram,
    build_histogram_subset,
    const_hessian_allowed,
    objective_has_constant_hessian,
    subtract_histogram,
)
from mojotrees.histogram_gpu import GpuHistogramBuilder
from mojotrees.objective_registry import (
    BINARY_LOGISTIC,
    CROSS_ENTROPY,
    CUSTOM,
    FAIR,
    GAMMA,
    HUBER,
    L1,
    LAMBDARANK,
    MAPE,
    MULTICLASS,
    POISSON,
    QUANTILE,
    SQUARED_ERROR,
    TWEEDIE,
)

from support import _uniform


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_data(
    n_rows: Int, n_features: Int, n_bins: Int, seed: UInt64
) raises -> BinnedMatrix:
    var values = List[Float64](capacity=n_rows * n_features)
    for k in range(n_rows * n_features):
        values.append(_uniform(seed + UInt64(k)))
    return bin_equal_width(values, n_rows, n_features, n_bins)


def _grads(n_rows: Int, seed: UInt64) -> List[Float64]:
    """Signed gradients with a wide magnitude spread, so that a cell's sum is
    genuinely order-sensitive and an accidental reassociation would show."""
    var g = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        var u = _uniform(seed + UInt64(r))
        var scale = 1.0
        var e = r % 8
        for _ in range(e):
            scale *= 0.03125
        g.append((2.0 * u - 1.0) * scale)
    return g^


def _ones(n_rows: Int) -> List[Float64]:
    """The hessian array a constant-hessian objective produces: the Float64
    literal 1.0 in every slot, which is what `boosting._fill_grad_hess_into`
    stores for `SQUARED_ERROR`, `L1`, `HUBER` and `QUANTILE` when there are no
    sample weights."""
    var h = List[Float64](capacity=n_rows)
    for _ in range(n_rows):
        h.append(CONSTANT_HESSIAN)
    return h^


def _every_other_row(n_rows: Int) -> List[Int]:
    var rows = List[Int](capacity=n_rows)
    for r in range(n_rows):
        if r % 2 == 0 or r % 7 == 3:
            rows.append(r)
    return rows^


def _bits(v: Float64) -> UInt64:
    """The IEEE-754 pattern, so a comparison is an integer comparison and a
    one-ulp move fails it. `docs/NUMERICS.md` section 7.1 is why every check
    in this file is written this way."""
    return v.to_bits().cast[DType.uint64]()


def _assert_same_bits(a: Histogram, b: Histogram) raises:
    """Two histograms are the same histogram: same shape, and every plane
    equal as bits rather than as numbers."""
    assert_equal(a.n_features, b.n_features)
    assert_equal(a.n_bins, b.n_bins)
    for i in range(a.n_features * a.n_bins):
        assert_equal(_bits(a.grad_at(i)), _bits(b.grad_at(i)))
        assert_equal(_bits(a.hess_at(i)), _bits(b.hess_at(i)))
        assert_equal(a.count_at(i), b.count_at(i))


def _reset_env():
    _ = setenv("MOJOTREES_CONST_HESSIAN", "")
    _ = setenv("MOJOTREES_CONST_HESSIAN_VERIFY", "")
    _ = setenv("MOJOTREES_CPU_FEATURE_GROUP", "")
    _ = setenv("MOJOTREES_CPU_COMPACT_MIN_ROWS", "")
    _ = setenv("MOJOTREES_NUM_WORKERS", "")
    _ = setenv("MOJOTREES_PARALLEL_MIN_OPS", "")


# ---------------------------------------------------------------------------
# Which objectives qualify
# ---------------------------------------------------------------------------

def test_constant_hessian_objectives() raises:
    """The predicate is a claim about second derivatives, and this is the
    truth table it has to satisfy.

    The four that qualify end their row body with `hp.unsafe_store(r, w)` in
    `boosting._fill_grad_hess_into`, where `w` is the Float64 literal 1.0
    when the fit has no weights. Everything else either varies the hessian
    with the raw score or the label, or comes from a caller.

    The weighted column is the one worth having a test for, because it is the
    exclusion a reader is most likely to think is over-cautious. It is not:
    the hessian *is* the weight for all four, so a weighted fit has a per-row
    hessian by construction, and this is exactly LightGBM's own
    `IsConstantHessian()`, which is `return !weights_`.
    """
    var yes: List[Int] = [SQUARED_ERROR, L1, HUBER, QUANTILE]
    for i in range(len(yes)):
        assert_true(objective_has_constant_hessian(yes[i], False))
        assert_true(not objective_has_constant_hessian(yes[i], True))

    var no: List[Int] = [
        BINARY_LOGISTIC,
        POISSON,
        GAMMA,
        TWEEDIE,
        MAPE,
        FAIR,
        CROSS_ENTROPY,
        MULTICLASS,
        CUSTOM,
        LAMBDARANK,
    ]
    for i in range(len(no)):
        assert_true(not objective_has_constant_hessian(no[i], False))
        assert_true(not objective_has_constant_hessian(no[i], True))


def test_mape_is_excluded_although_its_family_is_not() raises:
    """MAPE sits in the same regression family as the four that qualify and
    is the one that does not, because its hessian is `w * label_weight(y)`
    and varies with the label even unweighted. Called out on its own because
    it is the single most plausible mistake a future edit could make here,
    and because LightGBM excludes it the same way and for the same reason."""
    assert_true(not objective_has_constant_hessian(MAPE, False))


# ---------------------------------------------------------------------------
# CPU: the full-dataset builder
# ---------------------------------------------------------------------------

def _full_matches_at_group(group: String) raises:
    var n_rows = 4_001
    var n_features = 7
    var n_bins = 33
    var data = _make_data(n_rows, n_features, n_bins, UInt64(3))
    var grad = _grads(n_rows, UInt64(900_001))
    var hess = _ones(n_rows)

    _ = setenv("MOJOTREES_CPU_FEATURE_GROUP", group)
    var three = build_histogram(data, grad, hess, [], False)
    var two = build_histogram(data, grad, hess, [], True)
    _assert_same_bits(three, two)

    # And the elided plane really is the count, not merely equal to whatever
    # the other arm produced: a check that would fail if both arms were
    # broken the same way.
    for i in range(n_features * n_bins):
        assert_equal(_bits(two.hess_at(i)), _bits(Float64(two.count_at(i))))


def test_full_build_is_bit_identical_at_every_group_width() raises:
    """The interleave width and the elision are independent knobs and must
    stay so. Every rung of the ladder is a separate instantiation of the
    accumulation body, and the elision adds a branch inside each one, so a
    rung is exactly the granularity at which this could go wrong."""
    _reset_env()
    var rungs: List[String] = ["1", "2", "4", "8", "16"]
    for i in range(len(rungs)):
        _full_matches_at_group(rungs[i])
    _reset_env()


def test_full_build_is_bit_identical_at_every_worker_count() raises:
    """`MOJOTREES_NUM_WORKERS` is a scheduling knob that the package promises
    cannot reach the arithmetic (`docs/NUMERICS.md` section 1), and the
    elision must not become the exception. The refill runs on the task that
    owns the slice, so a task boundary can neither split it nor duplicate
    it; this is that claim under test rather than under argument."""
    _reset_env()
    _ = setenv("MOJOTREES_PARALLEL_MIN_OPS", "1")
    var workers: List[String] = ["1", "2", "3", "8"]
    for i in range(len(workers)):
        _ = setenv("MOJOTREES_NUM_WORKERS", workers[i])
        _full_matches_at_group("2")
    _reset_env()


def test_full_build_with_feature_subset_leaves_excluded_slices_zero() raises:
    """Under feature subsampling the excluded features' slices are zeroed by
    a separate pass, which the elision deliberately does not touch: an
    excluded feature's hessian cells are not refilled from anything, so they
    have to stay zeroed rather than be skipped."""
    _reset_env()
    var n_rows = 2_003
    var n_features = 6
    var n_bins = 17
    var data = _make_data(n_rows, n_features, n_bins, UInt64(19))
    var grad = _grads(n_rows, UInt64(910_001))
    var hess = _ones(n_rows)
    var features: List[Int] = [1, 4]

    var three = build_histogram(data, grad, hess, features, False)
    var two = build_histogram(data, grad, hess, features, True)
    _assert_same_bits(three, two)

    for f in range(n_features):
        if f == 1 or f == 4:
            continue
        for b in range(n_bins):
            var i = f * n_bins + b
            assert_equal(_bits(two.grad_at(i)), _bits(0.0))
            assert_equal(_bits(two.hess_at(i)), _bits(0.0))
            assert_equal(two.count_at(i), 0)
    _reset_env()


# ---------------------------------------------------------------------------
# CPU: the subset (tree node) builder
# ---------------------------------------------------------------------------

def _subset_matches(compact_min_rows: String, group: String) raises:
    var n_rows = 3_001
    var n_features = 5
    var n_bins = 29
    var data = _make_data(n_rows, n_features, n_bins, UInt64(23))
    var grad = _grads(n_rows, UInt64(920_001))
    var hess = _ones(n_rows)
    var rows = _every_other_row(n_rows)

    _ = setenv("MOJOTREES_CPU_COMPACT_MIN_ROWS", compact_min_rows)
    _ = setenv("MOJOTREES_CPU_FEATURE_GROUP", group)
    var three = build_histogram_subset(data, grad, hess, rows, [], False)
    var two = build_histogram_subset(data, grad, hess, rows, [], True)
    _assert_same_bits(three, two)

    var total = 0
    for i in range(n_features * n_bins):
        total += two.count_at(i)
    assert_equal(total, n_features * len(rows))


def test_subset_build_is_bit_identical_on_both_gather_arms() raises:
    """The subset builder has four row loops now rather than two: the gather
    decision crossed with the elision. A low `MOJOTREES_CPU_COMPACT_MIN_ROWS`
    takes the gathered `(g, h)` pair buffer, a high one reads the gradients
    through the row ids, and both have to be bit-identical to the three-plane
    path of the same arm."""
    _reset_env()
    var groups: List[String] = ["1", "2", "4", "16"]
    for i in range(len(groups)):
        _subset_matches("1", groups[i])
        _subset_matches("1000000000", groups[i])
    _reset_env()


# ---------------------------------------------------------------------------
# CPU: sibling subtraction
# ---------------------------------------------------------------------------

def test_sibling_subtraction_is_bit_identical() raises:
    """Deriving a sibling from the parent and the built child, with the
    hessian plane taken from the integer count difference instead of
    subtracted.

    Exact because both operands hold `Float64(count)` in that plane, so the
    difference the general path computes is `Float64(a) - Float64(b)` for two
    exactly representable integers, which is exactly `Float64(a - b)`. Under
    test rather than under argument because the general path is elementwise
    over SIMD blocks and the elided one folds the integer difference it just
    computed, and those are two different instruction sequences arriving at
    the same claim.
    """
    _reset_env()
    var n_rows = 2_501
    var n_features = 4
    var n_bins = 21
    var data = _make_data(n_rows, n_features, n_bins, UInt64(31))
    var grad = _grads(n_rows, UInt64(930_001))
    var hess = _ones(n_rows)

    var left = List[Int]()
    var right = List[Int]()
    for r in range(n_rows):
        if data.bin_at(r, 0) > n_bins // 2:
            right.append(r)
        else:
            left.append(r)
    assert_true(len(left) > 0 and len(right) > 0)

    var parent = build_histogram(data, grad, hess, [], True)
    var child = build_histogram_subset(data, grad, hess, left, [], True)
    var three = subtract_histogram(parent, child, False)
    var two = subtract_histogram(parent, child, True)
    _assert_same_bits(three, two)

    # The derived sibling is the right child, so its integer planes have to
    # be the right child's: the counts, and with them the reconstructed
    # hessian plane, which is what says the elision derived the sibling and
    # not merely something both arms agreed on.
    #
    # Only the integer planes. The gradient plane of a subtracted sibling is
    # NOT bit-equal to a direct build of the same rows, and it never was:
    # `parent - child` reassociates a Float64 sum, which is exactly the
    # property `histogram.mojo`'s module docstring says the accumulation
    # order controls. This test asserted the stronger thing on a first draft
    # and it failed, on the gradient plane, by nine units in the last place;
    # that is the subtraction trick behaving as documented and not a defect,
    # and the check is narrowed rather than loosened so that the distinction
    # stays visible.
    var direct = build_histogram_subset(data, grad, hess, right, [], True)
    for i in range(n_features * n_bins):
        assert_equal(direct.count_at(i), two.count_at(i))
        assert_equal(_bits(direct.hess_at(i)), _bits(two.hess_at(i)))
        assert_equal(_bits(two.hess_at(i)), _bits(Float64(two.count_at(i))))
    _reset_env()


# ---------------------------------------------------------------------------
# The gate and the verifier
# ---------------------------------------------------------------------------

def test_environment_can_force_the_specialization_off() raises:
    """`MOJOTREES_CONST_HESSIAN=0` withdraws permission, so a declaration is
    refused rather than honored. There is nothing to see in the output, which
    is the point: the two arms agree either way. What is observable is the
    permission itself, and a bisection needs that to be reliable."""
    _reset_env()
    assert_true(const_hessian_allowed())
    _ = setenv("MOJOTREES_CONST_HESSIAN", "0")
    assert_true(not const_hessian_allowed())

    var n_rows = 601
    var data = _make_data(n_rows, 3, 13, UInt64(37))
    var grad = _grads(n_rows, UInt64(940_001))
    var hess = _ones(n_rows)
    _assert_same_bits(
        build_histogram(data, grad, hess, [], False),
        build_histogram(data, grad, hess, [], True),
    )

    _ = setenv("MOJOTREES_CONST_HESSIAN", "1")
    assert_true(const_hessian_allowed())
    _reset_env()


def test_verifier_catches_a_false_declaration() raises:
    """`MOJOTREES_CONST_HESSIAN_VERIFY=1` turns a wrong declaration into an
    error at the build instead of a wrong hessian plane six rounds later.

    The array used here is what a weighted fit or a GOSS round produces: an
    objective whose hessian is constant, scaled per row by something else.
    That is precisely the case `objective_has_constant_hessian` excludes and
    cannot detect on its own, since the objective code is unchanged, so the
    verifier is the only mechanism that can catch a caller who declares it
    anyway.
    """
    _reset_env()
    var n_rows = 401
    var data = _make_data(n_rows, 3, 11, UInt64(41))
    var grad = _grads(n_rows, UInt64(950_001))
    var hess = _ones(n_rows)
    hess[n_rows - 1] = 2.0

    _ = setenv("MOJOTREES_CONST_HESSIAN_VERIFY", "1")
    var raised = False
    try:
        _ = build_histogram(data, grad, hess, [], True)
    except:
        raised = True
    assert_true(raised)

    # Not declared, so not checked, and the build still runs: the verifier is
    # a check on a declaration and never a check on the data.
    _ = build_histogram(data, grad, hess, [], False)
    _reset_env()


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------

def _gpu_leaf_matches(strategy: Int, narrow: Bool) raises:
    """One node's device histogram with and without the elision, compared as
    the fixed-point integers the kernels actually accumulate.

    Two builders rather than one builder driven twice, so that nothing --
    a cached quantized gradient buffer, a resolved tiling, a feature epoch --
    can carry state from the first arm into the second.

    `narrow` selects the active feature set, which is the branch that decides
    whether the output buffer is zeroed before a tiled build. Under the
    elision the reduction writes every active feature's slice of all three
    output planes, including the one it reconstructs, so the zeroing rule is
    unchanged; a narrowed set is where that would show if it were not.
    """
    # GUARDED 2026-08-17, and the mechanism is not the one it looks like.
    # Every CALLER of this helper is already wrapped in
    # `comptime if not has_accelerator()`, which ought to prune it, and does
    # not. `TestSuite.discover_tests[__functions_in_module()]()` enumerates
    # every function in this module, so each one is instantiated whether or
    # not a live call reaches it, and this helper builds a
    # `GpuHistogramBuilder`. On a CPU-only build that elaborates a GPU
    # kernel and the compile dies with `Unknown GPU architecture detected`,
    # taking the whole file with it.
    #
    # So in a test module the guard belongs on the HELPER and not only on
    # the tests that call it. Guarding the callers is what every other CPU
    # test in this suite does and it is sufficient there only because their
    # helpers touch no device API.
    comptime if not has_accelerator():
        raise Error("this helper needs an accelerator")
    else:
        var n_rows = 30_011
        var n_features = 6
        var n_bins = 64
        var data = _make_data(n_rows, n_features, n_bins, UInt64(53))
        var grad = _grads(n_rows, UInt64(960_001))
        var hess = _ones(n_rows)

        var three = GpuHistogramBuilder(data, strategy)
        var two = GpuHistogramBuilder(data, strategy)
        two.set_constant_hessian(True)
        assert_true(not three.constant_hessian())
        assert_true(two.constant_hessian())
        if narrow:
            var features: List[Int] = [0, 2, 5]
            three.set_features(features)
            two.set_features(features)

        three.upload_gradients(grad, hess)
        two.upload_gradients(grad, hess)
        three.begin_tree()
        two.begin_tree()

        var h3 = three.build_leaf(0)
        var h2 = two.build_leaf(0)
        _assert_same_bits(h3, h2)

        # A split, so the comparison also covers a node that owns a strict subset
        # of the rows and therefore a different launch geometry.
        three.apply_split(0, n_bins // 2, 0, 1, 2)
        two.apply_split(0, n_bins // 2, 0, 1, 2)
        _assert_same_bits(three.build_leaf(1), two.build_leaf(1))
        _assert_same_bits(three.build_leaf(2), two.build_leaf(2))


def test_gpu_atomic_strategy_is_bit_identical() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        _reset_env()
        _gpu_leaf_matches(STRATEGY_ATOMIC, False)
        _gpu_leaf_matches(STRATEGY_ATOMIC, True)
        _reset_env()


def test_gpu_tiled_strategy_is_bit_identical() raises:
    """The tiled path is the one a large fit runs, and it is where the
    elision changes a layout rather than only a loop: the partial buffer
    holds `[grad | count]` per tile instead of `[grad | hess | count]`, so
    the partial kernel and the reduction have to agree about which they are
    looking at."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        _reset_env()
        _gpu_leaf_matches(STRATEGY_TILED, False)
        _gpu_leaf_matches(STRATEGY_TILED, True)
        _reset_env()


def _gpu_feature_group_matches(strategy: Int, group: Int) raises:
    # Guarded for the same reason as `_gpu_leaf_matches`, and found by
    # `tools/check_gpu_guards.py` rather than by another CI cycle:
    # `TestSuite.discover_tests[__functions_in_module()]()` instantiates
    # EVERY function in this module, so guarding the tests that call this
    # helper does not prune the helper. It builds a `GpuHistogramBuilder`,
    # which elaborates a GPU kernel and fails a CPU-only compile.
    comptime if not has_accelerator():
        raise Error("this helper needs an accelerator")
    else:
        var n_rows = 20_003
        var n_features = 8
        var n_bins = 32
        var data = _make_data(n_rows, n_features, n_bins, UInt64(59))
        var grad = _grads(n_rows, UInt64(970_001))
        var hess = _ones(n_rows)

        var three = GpuHistogramBuilder(data, strategy)
        var two = GpuHistogramBuilder(data, strategy)
        three.set_feature_group(group)
        two.set_feature_group(group)
        two.set_constant_hessian(True)
        three.upload_gradients(grad, hess)
        two.upload_gradients(grad, hess)
        three.begin_tree()
        two.begin_tree()
        _assert_same_bits(three.build_leaf(0), two.build_leaf(0))


def test_gpu_feature_group_widths_are_bit_identical() raises:
    """The kernel family is instantiated per (group, bin capacity), and the
    elision is a runtime flag inside every one of them, so each rung is its
    own code path through the branch. Widths past 2 have never been launched
    on most devices, which is a reason to cover them here rather than a
    reason not to."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        _reset_env()
        var rungs: List[Int] = [1, 2, 4]
        for i in range(len(rungs)):
            _gpu_feature_group_matches(STRATEGY_TILED, rungs[i])
            _gpu_feature_group_matches(STRATEGY_ATOMIC, rungs[i])
        _reset_env()


def _gpu_fused_subtraction_matches(strategy: Int) raises:
    """The sibling subtraction folded into the build, with and without the
    elision, compared as raw device words across both slots.

    This is the path where the reconstruction has to be subtracted rather
    than written: the derived sibling's hessian cell is reduced by
    `hq_const * acc` instead of by the accumulated hessian. Both kernels fold
    the subtraction differently, so it is checked under both strategies, in
    the shape `test_gpu_strategies._subtraction_paths_agree` established.
    """
    # Guarded for the same reason as `_gpu_leaf_matches`, and found by
    # `tools/check_gpu_guards.py` rather than by another CI cycle:
    # `TestSuite.discover_tests[__functions_in_module()]()` instantiates
    # EVERY function in this module, so guarding the tests that call this
    # helper does not prune the helper. It builds a `GpuHistogramBuilder`,
    # which elaborates a GPU kernel and fails a CPU-only compile.
    comptime if not has_accelerator():
        raise Error("this helper needs an accelerator")
    else:
        var n_rows = 30_011
        var n_features = 5
        var n_bins = 64
        var data = _make_data(n_rows, n_features, n_bins, UInt64(61))
        var grad = _grads(n_rows, UInt64(980_001))
        var hess = _ones(n_rows)
        var cells = 3 * n_features * n_bins

        var three = GpuHistogramBuilder(data, strategy)
        var two = GpuHistogramBuilder(data, strategy)
        two.set_constant_hessian(True)
        three.upload_gradients(grad, hess)
        two.upload_gradients(grad, hess)
        assert_true(three.open_resident(4))
        assert_true(two.open_resident(4))
        three.begin_tree()
        two.begin_tree()

        var p3 = three.acquire_resident(0)
        var p2 = two.acquire_resident(0)
        assert_true(p3 >= 0 and p2 >= 0)
        three.enqueue_resident_leaf(0, p3)
        two.enqueue_resident_leaf(0, p2)
        three.apply_split(0, n_bins // 2, 0, 1, 2)
        two.apply_split(0, n_bins // 2, 0, 1, 2)

        var c3 = three.acquire_resident(1)
        var c2 = two.acquire_resident(1)
        assert_true(c3 >= 0 and c2 >= 0)
        three.enqueue_resident_leaf_subtracting(1, c3, p3)
        two.enqueue_resident_leaf_subtracting(1, c2, p2)

        var s3: List[Int] = [p3, c3]
        var s2: List[Int] = [p2, c2]
        var w3 = three.batcher[0].download_slots(s3)
        var w2 = two.batcher[0].download_slots(s2)
        assert_equal(len(w3), 2 * cells)
        assert_equal(len(w2), 2 * cells)
        for i in range(2 * cells):
            assert_equal(Int(w3[i]), Int(w2[i]))

        three.release_resident_all()
        two.release_resident_all()


def test_gpu_fused_subtraction_is_bit_identical_atomic() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        _reset_env()
        _gpu_fused_subtraction_matches(STRATEGY_ATOMIC)
        _reset_env()


def test_gpu_fused_subtraction_is_bit_identical_tiled() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        _reset_env()
        _gpu_fused_subtraction_matches(STRATEGY_TILED)
        _reset_env()


def test_gpu_float_plane_arm_is_bit_identical() raises:
    """The kernels have two gradient sources, the pre-quantized interleaved
    buffer and the two Float32 planes, and the elision adds a branch inside
    each. `MOJOTREES_GPU_QUANTIZED_GRADS=0` forces the second, which is the
    arm no default run takes and therefore the arm most likely to rot."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        _reset_env()
        _ = setenv("MOJOTREES_GPU_QUANTIZED_GRADS", "0")
        _gpu_leaf_matches(STRATEGY_TILED, False)
        _gpu_leaf_matches(STRATEGY_ATOMIC, False)
        _ = setenv("MOJOTREES_GPU_QUANTIZED_GRADS", "")
        _reset_env()


def test_gpu_environment_refuses_the_declaration() raises:
    """A builder constructed with `MOJOTREES_CONST_HESSIAN=0` refuses a later
    declaration, and says so. A benchmark arm that trusted its own request
    instead of reading this back would report the three-plane path's numbers
    under the two-plane path's label, which is the failure mode
    `set_feature_group`'s docstring records actually happening here once."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        _reset_env()
        _ = setenv("MOJOTREES_CONST_HESSIAN", "0")
        var data = _make_data(2_003, 3, 32, UInt64(67))
        var builder = GpuHistogramBuilder(data)
        builder.set_constant_hessian(True)
        assert_true(not builder.constant_hessian())
        _reset_env()

        var allowed = GpuHistogramBuilder(data)
        allowed.set_constant_hessian(True)
        assert_true(allowed.constant_hessian())
        allowed.set_constant_hessian(False)
        assert_true(not allowed.constant_hessian())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
