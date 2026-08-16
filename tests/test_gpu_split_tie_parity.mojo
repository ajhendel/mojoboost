"""The device kernels on the constructed ties, on a machine that has one.

`tests/test_split_tie_parity.mojo` pins the *rule*: it holds
`gpu_split_search.reference_search`, which is the kernels' arithmetic run
serially on the host, to `split.find_best_split` on histograms whose
candidates have exactly equal gain. That file runs everywhere and is where
the reasoning lives.

This file closes the half of the chain the replica cannot reach. The
replica's cross-feature fold is one serial loop; the kernel's is
`_reduce_slots_block_kernel`, a `block.max` over the gains followed by a
`block.min` over the slots of the threads holding the maximum. Those are the
same rule only if the reduction really does hand every thread the block-wide
maximum, and only if a thread that owns several slots keeps the lowest of its
own ties. A tie is the only input that can tell the two apart, and before
this file no device test used one: `test_gpu_split_search`'s tie case calls
the replica only, and its device cases all have a unique winner.

So the shapes here are chosen for the reduction, not for the arithmetic:

- Two features tied at the same gain but winning at different bins, which
  is the smallest case where following the gain and following the slot give
  different answers.
- Ninety-six feature slots with the winning gain reached at slots 6, 70 and
  71. At `REDUCE_SLOT_THREADS` of 64, slots 6 and 70 land on one thread and
  71 on another, so the answer needs both halves of the rule: the intra-thread
  ascending walk to keep 6 over 70, and the cross-thread `block.min` to keep
  6 over 71. It also spans more than one warp, which is where a reduction
  that returned a per-warp partial rather than the block maximum would first
  be visible.
- A tie between the two missing directions of one bin, which the scan
  kernel rather than the reduction decides.

Every case is asserted against `find_best_split` directly, so what is pinned
is the host's decision and not the replica's agreement with itself.
"""

from std.sys import has_accelerator
from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_true,
    TestSuite,
)

from mojotrees.categorical import CategoricalParams, CategoricalSpec
from mojotrees.gpu_split_search import (
    GpuSplitParams,
    GpuSplitRecord,
    GpuSplitSearcher,
    reference_search,
)
from mojotrees.histogram import Histogram
from mojotrees.monotone import OutputBounds
from mojotrees.split import SplitInfo, find_best_split

comptime _TOL = 1e-4


# --- Histograms, shared with tests/test_split_tie_parity.mojo -------------
#
# Duplicated rather than imported because the two files are read together
# and the numbers are the argument: the patterns below are only ties because
# of what their bins hold, and a reader who has to follow an import to see
# that cannot check the claim.


def _zeroed(n: Int) -> List[Int32]:
    var out = List[Int32](capacity=n)
    out.resize(n, Int32(0))
    return out^


def _histogram_words(
    n_features: Int, n_bins: Int, g: List[Int], h: List[Int], c: List[Int]
) raises -> List[Int32]:
    var size = n_features * n_bins
    if len(g) != size or len(h) != size or len(c) != size:
        raise Error("plane length must equal n_features * n_bins")
    var words = _zeroed(3 * size)
    for i in range(size):
        words[i] = Int32(g[i])
        words[size + i] = Int32(h[i])
        words[2 * size + i] = Int32(c[i])
    return words^


def _as_histogram(
    words: List[Int32], n_features: Int, n_bins: Int
) -> Histogram:
    var size = n_features * n_bins
    var grad = List[Float64](capacity=size)
    var hess = List[Float64](capacity=size)
    var count = List[Int](capacity=size)
    for i in range(size):
        grad.append(Float64(words[i]))
        hess.append(Float64(words[size + i]))
        count.append(Int(words[2 * size + i]))
    return Histogram.from_planes(grad^, hess^, count^, n_features, n_bins)


def _params() -> GpuSplitParams:
    return GpuSplitParams(1.0, 0.0, 0.0, 0, CategoricalParams.default())


def _host(
    words: List[Int32],
    n_features: Int,
    n_bins: Int,
    missing_bins: List[Int] = [],
) raises -> SplitInfo:
    return find_best_split(
        _as_histogram(words, n_features, n_bins),
        lambda_reg=1.0,
        min_child_hess=0.0,
        min_data_in_leaf=0,
        lambda_l1=0.0,
        missing_bins=missing_bins,
    )


# The three four-bin patterns `tests/test_split_tie_parity.mojo` documents:
# 0 "early" wins gain 16 first at bin 0, 1 "late" wins the identical 16 first
# at bin 1, and 2 "weak" wins only 4 and must lose to either.


def _pattern_g(p: Int) -> List[Int]:
    if p == 0:
        return [-4, 0, 0, 4]
    if p == 1:
        return [0, -4, 0, 4]
    return [-2, 0, 0, 2]


def _pattern_h(p: Int) -> List[Int]:
    if p == 1:
        return [0, 1, 0, 1]
    return [1, 0, 0, 1]


def _pattern_c(p: Int) -> List[Int]:
    if p == 1:
        return [0, 10, 0, 10]
    return [10, 0, 0, 10]


def _lay_out(pick: List[Int]) raises -> List[Int32]:
    var n = len(pick)
    var g = List[Int](capacity=4 * n)
    var h = List[Int](capacity=4 * n)
    var c = List[Int](capacity=4 * n)
    for f in range(n):
        var pg = _pattern_g(pick[f])
        var ph = _pattern_h(pick[f])
        var pc = _pattern_c(pick[f])
        for b in range(4):
            g.append(pg[b])
            h.append(ph[b])
            c.append(pc[b])
    return _histogram_words(n, 4, g, h, c)


def _device_search(
    words: List[Int32],
    n_features: Int,
    n_bins: Int,
    missing_bins: List[Int] = [],
) raises -> GpuSplitRecord:
    # The comptime guard keeps the device instantiation out of CPU-only
    # builds: module-level helpers compile unconditionally, so without it a
    # machine with no accelerator fails the arch constraint at compile time
    # even though only guarded tests call this.
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var searcher = GpuSplitSearcher(
            n_features, n_bins, missing_bins, CategoricalSpec.none()
        )
        searcher.set_monotone([])
        searcher.set_allowed([])
        searcher.upload_histogram(words)
        return searcher.search(
            _params(), 1.0, 1.0, OutputBounds.unbounded()
        )


def _assert_device_matches_host(
    words: List[Int32],
    n_features: Int,
    n_bins: Int,
    missing_bins: List[Int] = [],
) raises -> GpuSplitRecord:
    """The kernels' decision, the host's decision, and the replica's, all on
    one histogram. Returns the device record so a caller can assert what the
    tie itself looked like."""
    var got = _device_search(words, n_features, n_bins, missing_bins)
    var want = _host(words, n_features, n_bins, missing_bins)
    var replica = reference_search(
        words, n_features, n_bins, 1.0, 1.0, _params(), [], [], missing_bins
    )
    assert_equal(got.found, want.found)
    assert_true(want.found)
    assert_equal(got.feature, want.feature)
    assert_equal(got.bin, want.bin)
    assert_equal(got.default_left, want.default_left)
    assert_equal(got.is_categorical, want.is_categorical)
    assert_almost_equal(got.gain, want.gain, atol=_TOL)
    # And the kernels agree with the replica, so a future divergence names
    # which of the two moved.
    assert_equal(got.feature, replica.feature)
    assert_equal(got.bin, replica.bin)
    assert_equal(got.default_left, replica.default_left)
    return got^


def test_device_tie_between_two_features_keeps_the_lower_slot() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        # Feature 0 wins gain 16 at bin 0, feature 1 wins the same 16 at bin
        # 1. A reduction that carried only the gain could return either.
        var early_first = _lay_out([0, 1])
        var rec = _assert_device_matches_host(early_first, 2, 4)
        assert_equal(rec.feature, 0)
        assert_equal(rec.bin, 0)
        assert_almost_equal(rec.runner_gain, 16.0, atol=_TOL)

        # Swapped, so the lower slot is the one whose winner sits at bin 1.
        # The answer follows the slot, not the bin.
        var late_first = _lay_out([1, 0])
        var rec2 = _assert_device_matches_host(late_first, 2, 4)
        assert_equal(rec2.feature, 0)
        assert_equal(rec2.bin, 1)


def test_device_tie_across_more_slots_than_one_reduce_thread_owns() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        # Slots 6, 70 and 71 tie at the top; every other slot scores 4. With
        # 64 reduce threads, slots 6 and 70 share a thread and 71 has its
        # own, and 6 and 71 are in different warps.
        var pick = List[Int](capacity=96)
        for f in range(96):
            pick.append(0 if (f == 6 or f == 70 or f == 71) else 2)
        var rec = _assert_device_matches_host(_lay_out(pick), 96, 4)
        assert_equal(rec.feature, 6)
        assert_equal(rec.bin, 0)
        # The two losing ties are genuine runners-up, so the margin is zero
        # and the record says so.
        assert_almost_equal(rec.runner_gain, 16.0, atol=_TOL)
        assert_almost_equal(rec.margin(), 0.0, atol=_TOL)
        assert_true(rec.is_near_tie())

        # One tied slot on its own thread and nothing else tied: the
        # cross-thread `block.min` alone has to decide it.
        var pair = List[Int](capacity=96)
        for f in range(96):
            pair.append(0 if (f == 3 or f == 90) else 2)
        var rec2 = _assert_device_matches_host(_lay_out(pair), 96, 4)
        assert_equal(rec2.feature, 3)

        # And a unique winner in the high half, so the case above is not
        # passing by always answering with a low slot.
        var single = List[Int](capacity=96)
        for f in range(96):
            single.append(0 if f == 90 else 2)
        var rec3 = _assert_device_matches_host(_lay_out(single), 96, 4)
        assert_equal(rec3.feature, 90)


def test_device_tie_between_the_two_missing_directions() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        # The missing bin carries a count but no gradient and no hessian, so
        # sending it left and sending it right score identically at every
        # threshold. Missing-left is scored first, so it keeps the tie.
        var words = _histogram_words(
            1, 3, [-4, 4, 0], [1, 1, 0], [10, 10, 5]
        )
        var missing: List[Int] = [2]
        var rec = _assert_device_matches_host(words, 1, 3, missing)
        assert_equal(rec.bin, 0)
        assert_true(rec.default_left)
        assert_almost_equal(rec.runner_gain, 16.0, atol=_TOL)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
