"""The grower's scratch buffers, and the one thing wiring them must not do.

`tree._hist_full` used to call `histogram.build_histogram_into`, which hands
the blocked kernel a **fresh, zero-filled** `List[Float64]` and frees it again.
At the root of a large fit that list is the whole
`apple_cpu_policy.MAX_ROW_BLOCK_SCRATCH_BYTES` budget, allocated, first-touched
and zeroed once per tree. It now hands over `tree.GrowScratch.pairs`, the same
list the subset builder has been reusing across nodes since it existed.

The hazard that swap creates, and the only one, is a **stale cell**: a reused
buffer arrives holding the previous build's partial sums where a fresh one
arrived holding zeros. So the property under test is not "the two builders
agree" -- they are the same function -- it is that the blocked kernel writes
every private cell it later reads, and therefore that a buffer arriving full
of poison produces the histogram a buffer arriving full of zeros produces,
**to the bit**.

`resize(wanted, 0.0)` inside `build_histogram_into_scratch` is what used to
supply the zeros, and it fires only when the list is *shorter* than wanted, so
a warm buffer skips it entirely. That is exactly the case these fixtures put
in front of it: every comparison here pre-grows the scratch past what the
build needs and fills it with a value no correct partial sum could be.

Nothing here is a timing and nothing here may be read as a speed claim. The
bound this change earns is a byte count, not a second.

Covered:

1. **Poison is not read, unblocked.** The plain feature-partition kernel never
   touches the scratch at all, which is asserted rather than assumed by
   checking `plan.blocked()` is false before comparing.

2. **Poison is not read, blocked.** The row-blocked kernel is the one with
   private partials, so this is the case that matters. Blocking is forced
   through `MOJOTREES_CPU_ROW_BLOCKS` so a small fixture reaches it, and the
   plan is asserted to have actually blocked.

3. **Across a sequence.** A single scratch carrying real partial sums from one
   build into the next, over several different feature subsets and both
   constant-hessian arms, against a fresh buffer each time. A kernel that
   zeroed only the slots of the *currently* active features would pass case 2
   and fail here.

4. **Determinism across workers.** The reused-scratch arm at 1, 3 and 8
   workers is one histogram, bit for bit, and it is the fresh-scratch one.
"""

from std.os import setenv
from std.testing import assert_equal, assert_true, TestSuite

from mojotrees.apple_cpu_policy import (
    CpuProfile,
    derive_accumulation_plan,
)
from mojotrees.binning import BinnedMatrix
from mojotrees.histogram import (
    Histogram,
    build_histogram_into,
    build_histogram_into_scratch,
)

comptime POISON = -1.0e300
"""A value no partial sum in these fixtures can produce, so a cell that
survives into the output is unmistakably a cell nothing wrote."""


def _matrix(n_rows: Int, n_features: Int, n_bins: Int) raises -> BinnedMatrix:
    """A binned matrix that hits every bin of every feature, arithmetic rather
    than random so the fixture is identical on every machine."""
    var bins = List[UInt8](capacity=n_rows * n_features)
    bins.resize(n_rows * n_features, UInt8(0))
    for f in range(n_features):
        for r in range(n_rows):
            bins[f * n_rows + r] = UInt8((r + 3 * f) % n_bins)
    return BinnedMatrix(bins^, n_rows, n_features, n_bins)


def _gradients(n_rows: Int) -> List[Float64]:
    """Gradients no two of which are equal, so a cell is a genuine sum of
    distinct Float64 and a reassociation would show."""
    var g = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        g.append(1.0 / Float64(r + 3) - 0.25 * Float64((r % 7) + 1))
    return g^


def _hessians(n_rows: Int, constant: Bool) -> List[Float64]:
    var h = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        h.append(1.0 if constant else 0.5 + 1.0 / Float64((r % 11) + 2))
    return h^


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


def _poisoned(n: Int) -> List[Float64]:
    var s = List[Float64](capacity=n)
    s.resize(n, POISON)
    return s^


def _plan_is_blocked(
    n_features: Int, n_bins: Int, n_rows: Int
) raises -> Bool:
    """The plan the whole-dataset builder will derive, so a fixture can assert
    which kernel it is about to exercise instead of hoping."""
    var plan = derive_accumulation_plan(
        CpuProfile.detect(), n_features, n_features, n_bins, n_rows, False
    )
    return plan.blocked()


def _compare(
    data: BinnedMatrix,
    features: List[Int],
    const_h: Bool,
    label: String,
    expect_touched: Bool,
) raises:
    """One build with the allocating form against one with a scratch that is
    already large and already wrong.

    `expect_touched` is the positive control, and it is the reason this file
    is worth anything. Two histograms that agree because the blocked kernel
    never ran would satisfy every bit comparison here and establish nothing --
    this repository has shipped that mistake twice. So a fixture that says it
    blocks asserts the builder actually wrote into the buffer, and a fixture
    that says it does not asserts the buffer came back untouched.
    """
    var grad = _gradients(data.n_rows)
    var hess = _hessians(data.n_rows, const_h)
    var fresh = Histogram.zeroed(data.n_features, data.n_bins)
    build_histogram_into(fresh, data, grad, hess, features, const_h)

    var reused = Histogram.zeroed(data.n_features, data.n_bins)
    # Generously larger than any plan for this shape can ask for, so the
    # `resize(wanted, 0.0)` inside the builder cannot fire and supply the
    # zeros the old caller got for free.
    var scratch = _poisoned(
        8 * data.n_features * data.n_bins * 4 + 4096
    )
    build_histogram_into_scratch(
        reused, scratch, data, grad, hess, features, const_h
    )
    var touched = False
    for i in range(len(scratch)):
        if scratch[i] != POISON:
            touched = True
            break
    assert_equal(
        touched,
        expect_touched,
        String(label, ": scratch touched=", touched),
    )
    _assert_same_bits(fresh, reused, label)


def test_unblocked_ignores_a_poisoned_scratch() raises:
    """The feature-partition kernel never touches the scratch, and the wiring
    must not have made it start."""
    var data = _matrix(300, 7, 32)
    assert_true(
        not _plan_is_blocked(7, 32, 300),
        "fixture meant to exercise the unblocked kernel is blocked",
    )
    _compare(data, [], False, "unblocked all features", False)
    _compare(data, [], True, "unblocked const hessian", False)
    _compare(data, [1, 2, 5], False, "unblocked subset", False)


def test_blocked_ignores_a_poisoned_scratch() raises:
    """The row-blocked kernel owns the scratch, so this is the case the wiring
    is about. Blocking is forced rather than hoped for."""
    _ = setenv("MOJOTREES_CPU_ROW_BLOCKS", "4")
    assert_true(
        _plan_is_blocked(9, 24, 2400),
        "blocking did not turn on; the fixture proves nothing",
    )
    var data = _matrix(2400, 9, 24)
    _compare(data, [], False, "blocked all features", True)
    _compare(data, [], True, "blocked const hessian", True)
    _compare(data, [0, 3, 7], False, "blocked subset", True)
    _compare(data, [0, 3, 7], True, "blocked subset const hessian", True)
    _ = setenv("MOJOTREES_CPU_ROW_BLOCKS", "")


def test_one_scratch_across_a_sequence_of_builds() raises:
    """The grower's actual usage: one list, many builds, none of which may see
    another's partial sums.

    The subsets differ between builds on purpose. A kernel that zeroed only
    the private slots belonging to the currently active features would leave
    a previous build's slots live, and the compact private layout indexes by
    *active slot* rather than by feature id, so a shorter feature list shifts
    which stored partials land under the slots this build reads.
    """
    _ = setenv("MOJOTREES_CPU_ROW_BLOCKS", "4")
    var data = _matrix(2400, 9, 24)
    var grad = _gradients(data.n_rows)
    var shared = _poisoned(64)
    var subsets = List[List[Int]]()
    subsets.append([])
    subsets.append([0, 1, 2, 3, 4, 5, 6, 7, 8])
    subsets.append([2, 5])
    subsets.append([])
    subsets.append([0, 8])
    subsets.append([1, 3, 6])
    for i in range(len(subsets)):
        var const_h = (i % 2) == 1
        var hess = _hessians(data.n_rows, const_h)
        ref feats = subsets[i]
        var fresh = Histogram.zeroed(data.n_features, data.n_bins)
        build_histogram_into(fresh, data, grad, hess, feats, const_h)
        var reused = Histogram.zeroed(data.n_features, data.n_bins)
        build_histogram_into_scratch(
            reused, shared, data, grad, hess, feats, const_h
        )
        _assert_same_bits(fresh, reused, String("sequence step ", i))
    _ = setenv("MOJOTREES_CPU_ROW_BLOCKS", "")


def test_reused_scratch_is_deterministic_across_workers() raises:
    """Worker count is a schedule: it moves which core sums a cell and never
    the order the cells are summed in, and a shared scratch must not have
    changed that."""
    _ = setenv("MOJOTREES_CPU_ROW_BLOCKS", "4")
    var data = _matrix(2400, 9, 24)
    var grad = _gradients(data.n_rows)
    var hess = _hessians(data.n_rows, False)
    assert_true(
        _plan_is_blocked(9, 24, 2400), "worker-count fixture is not blocked"
    )

    _ = setenv("MOJOTREES_NUM_WORKERS", "1")
    var reference = Histogram.zeroed(data.n_features, data.n_bins)
    build_histogram_into(reference, data, grad, hess, [], False)

    var shared = _poisoned(64)
    var workers = ["1", "3", "8", ""]
    for i in range(len(workers)):
        _ = setenv("MOJOTREES_NUM_WORKERS", workers[i])
        var got = Histogram.zeroed(data.n_features, data.n_bins)
        build_histogram_into_scratch(
            got, shared, data, grad, hess, [], False
        )
        _assert_same_bits(
            reference, got, String("reused scratch at workers=", workers[i])
        )
    _ = setenv("MOJOTREES_NUM_WORKERS", "")
    _ = setenv("MOJOTREES_CPU_ROW_BLOCKS", "")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
