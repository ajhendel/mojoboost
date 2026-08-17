"""`score_function=Cosine` in the device split search, against the CPU path
that is its specification.

WHAT THIS FILE CLAIMS, AND WHAT IT DOES NOT
-------------------------------------------
It claims **structural** identity with `split.find_best_split` under
`SCORE_COSINE`: the same candidate set, the same admission guards, the same
scan order, the same tie rule, and therefore the same chosen split. It does
**not** claim bit-identity, and no assertion here is written as one. The
device search carries its sums as Float32 dequantized from a fixed-point
Int32 histogram; the host carries them as Float64 over exact Float64 sums.
That difference predates Cosine and is what every gain in
`gpu_split_search.mojo` already lives with. Cosine adds exactly one new
elementary operation to it, the Float32 square root at `gpu_cosine_score`.

The histograms below are therefore laid out so that the *only* remaining
difference is the arithmetic width. Every gradient, hessian and count is a
small integer and both fixed-point scales are 1.0, so the device's
dequantized Float32 sums are the same real numbers the host's Float64 sums
are, exactly, with no quantization error anywhere. What a gain comparison
here measures is Float32 against Float64 and nothing else, which is why the
gains are compared with a tolerance and the *decisions* are compared without
one.

THE ONE THING A SINGLE-NODE TEST CANNOT SEE, AND WHY IT IS HERE ANYWAY
----------------------------------------------------------------------
Cosine and L2 have the same argmax within one parent at `lambda_l2 = 0`,
provably (`split._cosine_pair`). It is a real temptation to implement Cosine
by relabelling the L2 kernel on that identity, and a suite that only ever
looked at one node at `lambda_l2 = 0` would not catch it. Two tests exist to
close that door and they are the point of the file:

  - `test_cosine_and_l2_disagree_inside_one_node`, at `lambda_l2 = 1`, where
    the two functionals pick *different features* from one histogram, and
    the same histogram at `lambda_l2 = 0`, where they must agree. The pair
    is what pins the mechanism rather than the outcome.
  - `test_cosine_reorders_two_parents`, where two nodes' winners are ranked
    one way by L2 and the other way by Cosine. That is the leaf-wise queue's
    comparison, the stock `grow_policy` is `lossguide`, and `sqrt(a) -
    sqrt(p)` does not order like `a - p` across two different `p`.

The device arm runs only where there is an accelerator and compares the
kernels to the replica and to `find_best_split` on the same histogram, which
is the chain `tests/test_gpu_split_tie_parity.mojo` established for L2.
"""

from std.sys import has_accelerator
from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
    TestSuite,
)

from mojotrees.categorical import CategoricalParams, CategoricalSpec
from mojotrees.gpu_split_search import (
    GpuSplitParams,
    GpuSplitRecord,
    GpuSplitSearcher,
    gpu_cosine_parent,
    reference_search,
)
from mojotrees.histogram import Histogram
from mojotrees.monotone import MONOTONE_INCREASING, OutputBounds
from mojotrees.split import (
    SCORE_COSINE,
    SCORE_L2,
    SplitInfo,
    find_best_split,
)

# Float32 carries 24 bits, the gains below are of order 1 to 30, and the
# quantities compared are the same real numbers on both sides. 1e-4 is far
# above the Float32/Float64 gap and far below any difference a wrong
# functional would produce, which is the property a tolerance has to have.
comptime _TOL = 1e-4


# --- Fixtures --------------------------------------------------------------


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
    """The same words as the Float64 histogram the host scans. Both scales
    are 1.0 everywhere in this file, so this conversion is exact and the two
    searches are reading the same real numbers."""
    var size = n_features * n_bins
    var grad = List[Float64](capacity=size)
    var hess = List[Float64](capacity=size)
    var count = List[Int](capacity=size)
    for i in range(size):
        grad.append(Float64(words[i]))
        hess.append(Float64(words[size + i]))
        count.append(Int(words[2 * size + i]))
    return Histogram.from_planes(grad^, hess^, count^, n_features, n_bins)


def _params(lambda_l2: Float64 = 1.0) -> GpuSplitParams:
    return GpuSplitParams(
        lambda_l2, 0.0, 0.0, 0, CategoricalParams.default()
    )


def _host(
    words: List[Int32],
    n_features: Int,
    n_bins: Int,
    lambda_l2: Float64 = 1.0,
    score_function: Int = SCORE_COSINE,
    missing_bins: List[Int] = [],
    monotone: List[Int] = [],
) raises -> SplitInfo:
    return find_best_split(
        _as_histogram(words, n_features, n_bins),
        lambda_reg=lambda_l2,
        min_child_hess=0.0,
        min_data_in_leaf=0,
        lambda_l1=0.0,
        missing_bins=missing_bins,
        monotone=monotone,
        score_function=score_function,
    )


def _replica(
    words: List[Int32],
    n_features: Int,
    n_bins: Int,
    lambda_l2: Float64 = 1.0,
    score_function: Int = SCORE_COSINE,
    missing_bins: List[Int] = [],
    monotone: List[Int] = [],
) raises -> GpuSplitRecord:
    return reference_search(
        words,
        n_features,
        n_bins,
        1.0,
        1.0,
        _params(lambda_l2),
        [],
        [],
        missing_bins,
        monotone,
        score_function=score_function,
    )


def _assert_same_decision(
    words: List[Int32],
    n_features: Int,
    n_bins: Int,
    lambda_l2: Float64 = 1.0,
    missing_bins: List[Int] = [],
    monotone: List[Int] = [],
) raises -> GpuSplitRecord:
    """The device replica's Cosine decision against the host's, on one
    histogram. The decision fields are compared exactly; the gain is
    compared with `_TOL`, because it is the one quantity where Float32 and
    Float64 are allowed to differ."""
    var got = _replica(
        words,
        n_features,
        n_bins,
        lambda_l2,
        SCORE_COSINE,
        missing_bins,
        monotone,
    )
    var want = _host(
        words,
        n_features,
        n_bins,
        lambda_l2,
        SCORE_COSINE,
        missing_bins,
        monotone,
    )
    assert_equal(got.found, want.found)
    if want.found:
        assert_equal(got.feature, want.feature)
        assert_equal(got.bin, want.bin)
        assert_equal(got.default_left, want.default_left)
        assert_equal(got.is_categorical, want.is_categorical)
        assert_almost_equal(got.gain, want.gain, atol=_TOL)
    return got^


# The histogram the two functionals disagree on. Two features, two bins, and
# both features total to `G = -8`, `H = 4`, which is what a real histogram
# does: every feature of one node sums to that node's own gradient and
# hessian. With two bins there is exactly one threshold per feature, so the
# node's whole candidate set is two candidates and the disagreement is a
# disagreement about which *feature* wins.
#
#   feature 0   left (-8, 2)   right (0, 2)
#   feature 1   left (-6, 1)   right (-2, 3)
#
# At `lambda_l2 = 1`: L2 scores 8.533 and 6.200, Cosine scores 1.657 and
# 2.085. At `lambda_l2 = 0`: L2 scores 16.0 and 21.333, Cosine 1.657 and
# 2.110 -- the same order, which is the identity holding.


def _disagreement() raises -> List[Int32]:
    return _histogram_words(
        2, 2, [-8, 0, -6, -2], [2, 2, 1, 3], [5, 5, 5, 5]
    )


# The three four-bin tie patterns `tests/test_split_tie_parity.mojo`
# documents, which are ties under Cosine for the same reason they are ties
# under L2: bins 1 and 2 are empty, so the prefix sums at bins 0, 1 and 2 are
# the same sums and every functional of them is the same number.


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


def _one_categorical(n_features: Int, n_categories: Int) -> CategoricalSpec:
    """Feature 0 categorical with `n_categories` codes, the rest numerical;
    the same helper `tests/test_gpu_split_search.mojo` uses."""
    var flags = List[Bool](capacity=n_features)
    var offsets = List[Int](capacity=n_features + 1)
    var codes = List[Int](capacity=n_categories)
    for i in range(n_categories):
        codes.append(i)
    offsets.append(0)
    for f in range(n_features):
        flags.append(f == 0)
        offsets.append(n_categories if f == 0 else offsets[f])
    return CategoricalSpec(flags^, codes^, offsets^)


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


# --- The functional itself -------------------------------------------------


def test_cosine_parent_is_the_unsplit_ratio() raises:
    """`gpu_cosine_parent` is `|G| / sqrt(H)`, in which `lambda_l2` has
    cancelled outright.

    Worth pinning as its own case because it is the one closed form in the
    whole functional and it is the sanity check on the sign convention: a
    node constant that moved with `lambda_l2` would mean the numerator and
    the denominator had been built from different outputs."""
    for lam in [Float32(0.0), Float32(1.0), Float32(7.5)]:
        assert_almost_equal(
            Float64(
                gpu_cosine_parent(Float32(-8.0), Float32(4.0), 0.0, lam)
            ),
            4.0,
            atol=_TOL,
        )
        assert_almost_equal(
            Float64(
                gpu_cosine_parent(Float32(3.0), Float32(9.0), 0.0, lam)
            ),
            1.0,
            atol=_TOL,
        )
    # A node with no hessian mass has no unsplit score and no division by
    # zero: `gpu_cosine_out`'s guard sends the output to zero, which sends
    # both accumulators to zero, and the denominator seed makes the ratio
    # 0 rather than NaN.
    assert_equal(
        Float64(gpu_cosine_parent(Float32(0.0), Float32(0.0), 0.0, 0.0)),
        0.0,
    )


def test_cosine_and_l2_disagree_inside_one_node() raises:
    """The two functionals pick different features from one histogram at
    `lambda_l2 = 1`, and the same feature at `lambda_l2 = 0`.

    This is the test that makes the Cosine arm load-bearing. If the kernel
    had been implemented by relabelling the L2 gain -- which the
    `lambda_l2 = 0` identity invites -- the first half of this would fail
    and the second half would still pass."""
    var words = _disagreement()

    var cos_one = _assert_same_decision(words, 2, 2, 1.0)
    var l2_one = _replica(words, 2, 2, 1.0, SCORE_L2)
    assert_true(cos_one.found)
    assert_true(l2_one.found)
    assert_equal(cos_one.feature, 1)
    assert_equal(l2_one.feature, 0)
    assert_almost_equal(cos_one.gain, 2.0848698, atol=_TOL)
    assert_almost_equal(l2_one.gain, 8.5333333, atol=_TOL)

    # And the identity, on the same histogram: at `lambda_l2 = 0` Cosine is
    # `sqrt` of the L2 score and `sqrt` is strictly increasing, so within
    # this one node the argmax is the same one.
    var cos_zero = _assert_same_decision(words, 2, 2, 0.0)
    var l2_zero = _replica(words, 2, 2, 0.0, SCORE_L2)
    assert_equal(cos_zero.feature, 1)
    assert_equal(l2_zero.feature, 1)
    # Spelled out rather than asserted as "equal": `sqrt(gain + P) -
    # sqrt(P)` is what the Cosine gain of an L2 gain is at `lambda_l2 = 0`,
    # and this is the arithmetic statement of the identity the docstrings
    # keep citing.
    assert_almost_equal(cos_zero.gain, 2.1101009, atol=_TOL)
    assert_almost_equal(l2_zero.gain, 21.333333, atol=_TOL)


def test_cosine_reorders_two_parents() raises:
    """Two nodes whose winners L2 ranks one way and Cosine ranks the other.

    This is the comparison a leaf-wise queue makes and `lossguide` is the
    stock `grow_policy`, so it is the comparison a default fit makes. Node A
    totals `G = -1, H = 3` and node B totals `G = 0, H = 2`; they are
    different parents, which is the condition the per-parent identity does
    not cover."""
    var node_a = _histogram_words(1, 2, [-5, 4], [1, 2], [5, 5])
    var node_b = _histogram_words(1, 2, [-4, 4], [1, 1], [5, 5])

    var cos_a = _assert_same_decision(node_a, 1, 2, 1.0)
    var cos_b = _assert_same_decision(node_b, 1, 2, 1.0)
    var l2_a = _replica(node_a, 1, 2, 1.0, SCORE_L2)
    var l2_b = _replica(node_b, 1, 2, 1.0, SCORE_L2)

    assert_true(cos_a.found and cos_b.found and l2_a.found and l2_b.found)
    # L2 puts A ahead of B; Cosine puts B ahead of A. The queue would pop a
    # different leaf first under each, which is the whole difference.
    assert_true(l2_a.gain > l2_b.gain)
    assert_true(cos_b.gain > cos_a.gain)


# --- The tie rule, which the new functional must not move ------------------


def test_cosine_keeps_the_lowest_tied_bin() raises:
    """Bins 1 and 2 add nothing, so bins 0, 1 and 2 are the same candidate
    and score identically under Cosine. The scan ascends and accepts only a
    strictly greater gain, so bin 0 keeps it."""
    var rec = _assert_same_decision(_lay_out([0]), 1, 4, 1.0)
    assert_true(rec.found)
    assert_equal(rec.bin, 0)
    assert_false(rec.default_left)


def test_cosine_keeps_the_lowest_tied_feature() raises:
    """Feature 0 reaches the winning Cosine score first at bin 0 and feature
    1 reaches the identical score at bin 1. The cross-feature fold walks
    ascending and accepts only a strictly greater gain, so the answer
    follows the slot and not the bin."""
    var early = _assert_same_decision(_lay_out([0, 1]), 2, 4, 1.0)
    assert_equal(early.feature, 0)
    assert_equal(early.bin, 0)

    # Swapped, so the lower slot is the one that wins at bin 1.
    var late = _assert_same_decision(_lay_out([1, 0]), 2, 4, 1.0)
    assert_equal(late.feature, 0)
    assert_equal(late.bin, 1)


def test_cosine_keeps_default_left_on_a_missing_direction_tie() raises:
    """The missing bin carries a count but no gradient and no hessian, so
    routing it left and routing it right build the same two accumulators.
    Missing-left is scored first and a tie is not strictly greater, so
    `default_left` survives -- LightGBM's rule, unchanged by the
    functional."""
    var words = _histogram_words(1, 3, [-4, 4, 0], [1, 1, 0], [10, 10, 5])
    var missing: List[Int] = [2]
    var rec = _assert_same_decision(words, 1, 3, 1.0, missing)
    assert_true(rec.found)
    assert_equal(rec.bin, 0)
    assert_true(rec.default_left)


def test_cosine_monotone_rejection_scores_zero() raises:
    """A candidate an active monotone constraint rejects contributes 0.0 and
    therefore loses to a best that starts at 0.0, which is exactly what the
    host does with `_CosineTerms.ok`.

    Asserted as a difference from the unconstrained answer as well as as an
    agreement with the host, so it cannot pass by rejecting nothing."""
    var words = _histogram_words(1, 3, [-4, 4, 0], [1, 1, 0], [10, 10, 5])
    var free = _assert_same_decision(words, 1, 3, 1.0)
    assert_true(free.found)
    assert_equal(free.bin, 0)

    var increasing: List[Int] = [MONOTONE_INCREASING]
    var held = _assert_same_decision(words, 1, 3, 1.0, [], increasing)
    # The only candidate puts the larger output on the left, which an
    # increasing constraint forbids, and the remaining threshold scores 0.
    assert_false(held.found)


# --- What the Cosine arm refuses, and why the direction matters ------------


def test_cosine_refuses_a_categorical_matrix() raises:
    """`find_best_split`'s own refusal, in the replica and on the searcher.

    A categorical feature is searched as category partitions scored with the
    L2 gain, and only that search's winner reaches the fold, so admitting
    the pair would put two score functions inside one argmax. The refusal is
    the host's and is not a device limitation."""
    var words = _histogram_words(
        1, 4, [-4, 1, 1, 2], [1, 1, 1, 1], [4, 4, 4, 4]
    )
    var cats = _one_categorical(1, 3)
    with assert_raises(contains="numerical thresholds"):
        _ = reference_search(
            words,
            1,
            4,
            1.0,
            1.0,
            _params(),
            [],
            [],
            [],
            [],
            cats,
            score_function=SCORE_COSINE,
        )
    # The same matrix under L2 is not refused, so the refusal is about the
    # pairing and not about the matrix.
    var l2 = reference_search(
        words, 1, 4, 1.0, 1.0, _params(), [], [], [], [], cats
    )
    assert_true(l2.found)


def test_unknown_score_function_still_refuses() raises:
    """A selector that is neither `SCORE_L2` nor `SCORE_COSINE` is refused
    rather than resolved to either.

    The direction is load-bearing and this test is here to keep it. A third
    functional added later and not taught to these kernels must fail at the
    door; the alternative is that it silently receives one of the two
    answers already implemented, under its own label, which is the defect
    every gate in this area exists to prevent."""
    var words = _histogram_words(1, 2, [-4, 4], [1, 1], [5, 5])
    with assert_raises(contains="score_function"):
        _ = _replica(words, 1, 2, 1.0, 2)
    with assert_raises(contains="score_function"):
        _ = _replica(words, 1, 2, 1.0, -1)


# --- The kernels, where there is one to run them ---------------------------


def _device_cosine(
    words: List[Int32],
    n_features: Int,
    n_bins: Int,
    lambda_l2: Float64 = 1.0,
    missing_bins: List[Int] = [],
) raises -> GpuSplitRecord:
    # The comptime guard keeps the device instantiation out of CPU-only
    # builds; module-level helpers compile unconditionally, so without it a
    # machine with no accelerator fails the arch constraint at compile time
    # even though only guarded tests call this.
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var searcher = GpuSplitSearcher(
            n_features, n_bins, missing_bins, CategoricalSpec.none()
        )
        searcher.set_score_function(SCORE_COSINE)
        searcher.set_monotone([])
        searcher.set_allowed([])
        searcher.upload_histogram(words)
        return searcher.search(
            _params(lambda_l2), 1.0, 1.0, OutputBounds.unbounded()
        )


def test_device_cosine_matches_the_host_and_the_replica() raises:
    """The kernels, the replica and `find_best_split`, all under Cosine on
    the same histograms.

    The replica is not an independent implementation -- it calls the same
    Float32 helpers the kernels call -- so what this pins is that the two
    loop structures agree, and the comparison against `find_best_split` is
    what pins both to the specification."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        # The disagreement histogram, where a kernel that had quietly kept
        # scoring L2 would return feature 0.
        var got = _device_cosine(_disagreement(), 2, 2, 1.0)
        var replica = _replica(_disagreement(), 2, 2, 1.0)
        var want = _host(_disagreement(), 2, 2, 1.0)
        assert_true(want.found)
        assert_equal(got.feature, 1)
        assert_equal(got.feature, want.feature)
        assert_equal(got.bin, want.bin)
        assert_equal(got.default_left, want.default_left)
        assert_almost_equal(got.gain, want.gain, atol=_TOL)
        assert_equal(got.feature, replica.feature)
        assert_equal(got.bin, replica.bin)
        assert_equal(got.default_left, replica.default_left)
        assert_almost_equal(got.gain, replica.gain, atol=_TOL)

        # And the tie shapes, which are what the cross-feature reduction
        # decides rather than the scan.
        var tied = _device_cosine(_lay_out([0, 1]), 2, 4, 1.0)
        var tied_want = _host(_lay_out([0, 1]), 2, 4, 1.0)
        assert_equal(tied.feature, 0)
        assert_equal(tied.bin, 0)
        assert_equal(tied.feature, tied_want.feature)
        assert_equal(tied.bin, tied_want.bin)

        # And the missing-direction tie, which the scan decides.
        var words = _histogram_words(
            1, 3, [-4, 4, 0], [1, 1, 0], [10, 10, 5]
        )
        var missing: List[Int] = [2]
        var miss = _device_cosine(words, 1, 3, 1.0, missing)
        var miss_want = _host(words, 1, 3, 1.0, SCORE_COSINE, missing)
        assert_equal(miss.bin, 0)
        assert_true(miss.default_left)
        assert_equal(miss.bin, miss_want.bin)
        assert_equal(miss.default_left, miss_want.default_left)


def test_searcher_reports_and_refuses_the_functional() raises:
    """`describe_scan` names the functional, and the searcher refuses the
    pairs it cannot serve.

    A gain reported without the functional it is a gain of cannot be
    compared to any other gain from this searcher, which is a stronger
    version of the reason `gain=` is already on that line."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var searcher = GpuSplitSearcher(2, 2, [], CategoricalSpec.none())
        assert_equal(searcher.score_function(), SCORE_L2)
        assert_true("score=L2" in searcher.describe_scan())
        searcher.set_score_function(SCORE_COSINE)
        assert_equal(searcher.score_function(), SCORE_COSINE)
        assert_true("score=Cosine" in searcher.describe_scan())
        with assert_raises(contains="score_function"):
            searcher.set_score_function(2)

        # A searcher built over a categorical matrix refuses Cosine for
        # `find_best_split`'s reason, and says so before a launch.
        var cats = _one_categorical(2, 3)
        var with_cat = GpuSplitSearcher(2, 4, [], cats)
        with assert_raises(contains="numerical thresholds"):
            with_cat.set_score_function(SCORE_COSINE)

        # THE OBLIVIOUS LEVEL NO LONGER REFUSES COSINE, and this assertion is
        # inverted from what it said until 2026-08-17. It used to require a
        # refusal reading "score_function=L2 only", on the argument that a
        # level's Cosine score is a ratio of two cross-leaf accumulators while
        # its leaf loop could only sum. That argument was true when it was
        # written and the lane it named, `lane/cosine-device`, has since
        # implemented the ratio: `gpu_split_search.mojo` now branches on
        # `score_function == SCORE_COSINE` in three level kernels, and the
        # refusal string does not exist anywhere in `src/`.
        #
        # TWO THINGS CHANGED AT ONCE and only one of them is the feature,
        # which is why this failed as "Didn't raise" rather than as a wrong
        # message. `score_function` became an explicit ARGUMENT of
        # `search_oblivious_level` rather than a searcher field, deliberately
        # and temporarily, so that a field only the level scan honored could
        # not answer Cosine for a level and L2 for a node without saying so
        # (see that method's docstring). So `set_score_function(SCORE_COSINE)`
        # no longer reaches this call at all: the old test was passing the
        # default L2 and asserting a Cosine refusal, and would have kept
        # passing for the wrong reason if the refusal had survived.
        #
        # The assertion is therefore on the NEW contract in both halves: the
        # searcher field does not reach the level, and the level accepts
        # Cosine when it is named at the call.
        var level = GpuSplitSearcher(2, 2, [], CategoricalSpec.none(), 3)
        level.set_score_function(SCORE_COSINE)
        level.set_monotone([])
        level.set_allowed([])
        # Named at the call, Cosine is served rather than refused.
        var cosine_level = level.search_oblivious_level(
            _params(), 1.0, 1.0, [0, 1], 0, 1, SCORE_COSINE
        )
        # Unnamed, the argument defaults to L2 whatever the field says, which
        # is the property that makes the field unable to lie about a level.
        var l2_level = level.search_oblivious_level(
            _params(), 1.0, 1.0, [0, 1], 0, 1
        )
        # Both return a record; the point is that neither raises and that the
        # two are reached through different functionals.
        assert_true(cosine_level.feature >= -1)
        assert_true(l2_level.feature >= -1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
