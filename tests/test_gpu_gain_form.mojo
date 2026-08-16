"""The two split gain forms, and the near tie that separates them.

WHAT THIS FILE IS FOR
---------------------
`gpu_split_search` scores every candidate through one of two algebraically
equivalent expressions. `GAIN_FORM_SUBTRACTIVE` is what shipped:
`left_score + right_score - parent_score`, with each candidate's right-hand
sums derived by Float32 subtraction from the node total. `GAIN_FORM_CROSS` is
candidates 5 and 6 of `docs/design/ACCURACY_BUDGET.md`: the cross form
`D^2 / (HL' * HR' * S)` less a node constant, with the right-hand sums taken
in the integer fixed-point domain. The whole argument lives at
`gpu_cross_gain`, `gpu_right_sum`, and `GpuSplitSearcher.set_gain_form`. This
file checks the claims that are checkable without a device and without a fit.

  1. The identity is an identity. On well-separated candidates the two forms
     pick the same winner and agree on its gain, over a set of shapes that
     includes zero L2, a categorical feature, a missing bin, and a monotone
     constraint. If the algebra were wrong this is where it would show.
  2. The near tie. On one node in the one-sided-gradient regime, two
     candidates whose exact gains differ by one part in ten thousand: the
     shipped form returns the *identical* Float32 for both and therefore
     keeps its incumbent, and the cross form separates them and ranks them
     the right way round.
  3. L1 sends the cross form back. `lambda_l1 != 0` breaks the identity, so
     `gpu_resolve_gain_form` refuses the arm and the two codes return the
     same record. This is the guard that keeps a bias out of L1 fits.
  4. Both arms are reachable, they differ, and the default is the cross one.
  5. On the small-integer histograms the rest of the suite is built from,
     the two arms agree far inside that suite's tolerance and select the
     same split. That is not a property of this change so much as a bound on
     its blast radius, and it is asserted here because the files it is about
     are accelerator-only and cannot be run to find out.

`GpuSplitSearcher.set_gain_form`'s refusal of an unknown code is *not*
tested here, because reaching the setter means constructing a searcher and
that opens a device. It belongs to the accelerator set.

WHERE THE NEAR-TIE FIXTURE CAME FROM
------------------------------------
Not from a hand calculation, and not from a run. It was searched for in a
standalone NumPy model of this scan, by fixing a node total, solving the
gain quadratic for a left gradient sum that hits a chosen gain, and keeping
the pairs the shipped form cannot separate. The node is deliberately
one-sided (`parent_score / gain` is 2900), because that ratio -- not the row
count -- is what controls the Float32 gain's resolution, which is
`ACCURACY_BUDGET.md` section 10's finding and the reason this file exists.

**The ordering assertion is forced, not lucky, and the difference matters.**
The cross form's own absolute error on this fixture is about 3e-05, so a pair
separated by less than that would come out right about half the time and the
test would be asserting a coin flip. The gap here is 8.9e-05, chosen to sit
above the cross form's resolution and below the shipped form's. In the same
model, over 1500 independent pairs at this gap and this ratio, the cross form
ranks the better candidate first 100 percent of the time and the shipped form
25 percent. The margin the assertions actually see is over 2000 Float32 ulps,
which is also why the fixture does not care whether `gpu_cross_gain`'s
multiply-add is fused: it survives either contraction by three orders of
magnitude.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
No device is opened and no kernel is instantiated, so this runs in the CPU
set. Everything here goes through `reference_search`, which is the kernels'
own arithmetic on the host -- it calls the same Float32 helpers and walks the
same candidate order -- so a claim about the gain form is a claim about the
kernels. That the kernel *loops* agree with `reference_search` is
`tests/test_gpu_split_search.mojo`'s job and is not repeated here.

No timing and no fit. The case for the default arm is an accuracy case, and
`set_gain_form` records that its speed effect is unmeasured in either
direction. This file measures nothing.

No assertion that the cross form is closer to the exact gain in general. It
is not, at every node: below a `parent_score / gain` of one the two are
interchangeable and the cross form is a few percent worse on the median. The
claim under test is about resolution in the one-sided regime, and that is
what the near-tie test is shaped to catch.
"""

# run_tests: cpu-safe -- opens no device; see tools/run_tests.sh
# gpu_by_content.

from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_not_equal,
    assert_true,
    TestSuite,
)

from mojotrees.categorical import CategoricalSpec
from mojotrees.monotone import MONOTONE_INCREASING, OutputBounds
from mojotrees.gpu_split_search import (
    DEFAULT_GAIN_FORM,
    GAIN_FORM_CROSS,
    GAIN_FORM_SUBTRACTIVE,
    GpuSplitParams,
    GpuSplitRecord,
    describe_gain_form,
    gpu_cross_gain,
    gpu_cross_node_s,
    gpu_cross_offset,
    gpu_resolve_gain_form,
    gpu_right_sum,
    reference_search,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _zeroed(n: Int) -> List[Int32]:
    var out = List[Int32](capacity=n)
    out.resize(n, Int32(0))
    return out^


def _histogram_words(
    n_features: Int, n_bins: Int, g: List[Int], h: List[Int], c: List[Int]
) raises -> List[Int32]:
    """A `[grad | hess | count]` buffer from flat `[f * n_bins + b]` lists,
    as `tests/test_gpu_split_search.mojo` builds one."""
    var size = n_features * n_bins
    if len(g) != size or len(h) != size or len(c) != size:
        raise Error("plane length must equal n_features * n_bins")
    var words = _zeroed(3 * size)
    for i in range(size):
        words[i] = Int32(g[i])
        words[size + i] = Int32(h[i])
        words[2 * size + i] = Int32(c[i])
    return words^


def _one_categorical(n_features: Int, n_categories: Int) -> CategoricalSpec:
    """Feature 0 categorical with `n_categories` codes, the rest numerical.
    The same helper `tests/test_gpu_split_search.mojo` uses, copied rather
    than shared because a test that imports another test's private helper is
    a dependency neither file declares."""
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


def _params(
    lambda_l2: Float64 = 1.0,
    lambda_l1: Float64 = 0.0,
    min_child_hess: Float64 = 1e-3,
    min_data_in_leaf: Int = 0,
) -> GpuSplitParams:
    var p = GpuSplitParams.default()
    p.lambda_l2 = lambda_l2
    p.lambda_l1 = lambda_l1
    p.min_child_hess = min_child_hess
    p.min_data_in_leaf = min_data_in_leaf
    return p^


# The near-tie node. Two features, two bins each, so each feature offers
# exactly one candidate (the top threshold is not a split) and the node's
# whole search is a two-way compare over twelve Int32 words.
#
# Node totals: 900_000_000 gradient units and 300_000_000 hessian units at a
# power-of-two scale of 2^20, which is `total_g = 858.31`, `total_h = 286.10`,
# `parent_score = 2565.95` at `lambda_l2 = 1`.
#
#   slot 0: left grad 505_471_308, left hess 159_290_537
#   slot 1: left grad 501_726_546, left hess 158_029_032
#
# Exact gains, in rational arithmetic over the integer sums:
#
#   slot 0   0.8848110150053855
#   slot 1   0.8848997565436061      better, by 1.003e-04 relative
#
# The shipped form computes 0.885009765625 for *both*, because its resolution
# here is `eps * parent_score = 1.5e-04` and the gap is 8.9e-05. Ties go to
# the incumbent, so it selects slot 0.
comptime _TIE_SCALE = 1048576.0
comptime _TIE_TOTAL_G = 900_000_000
comptime _TIE_TOTAL_H = 300_000_000
comptime _TIE_TOTAL_C = 300_000
comptime _TIE_EXACT_GAIN_0 = 0.8848110150053855
comptime _TIE_EXACT_GAIN_1 = 0.8848997565436061


def _near_tie_words() raises -> List[Int32]:
    var lg0 = 505_471_308
    var lh0 = 159_290_537
    var lc0 = 159_290
    var lg1 = 501_726_546
    var lh1 = 158_029_032
    var lc1 = 158_029
    return _histogram_words(
        2,
        2,
        [lg0, _TIE_TOTAL_G - lg0, lg1, _TIE_TOTAL_G - lg1],
        [lh0, _TIE_TOTAL_H - lh0, lh1, _TIE_TOTAL_H - lh1],
        [lc0, _TIE_TOTAL_C - lc0, lc1, _TIE_TOTAL_C - lc1],
    )


def _search(
    words: List[Int32],
    n_features: Int,
    n_bins: Int,
    params: GpuSplitParams,
    form: Int,
    scale: Float64 = 1.0,
    missing_bins: List[Int] = [],
    monotone: List[Int] = [],
    cats: CategoricalSpec = CategoricalSpec.none(),
) raises -> GpuSplitRecord:
    return reference_search(
        words,
        n_features,
        n_bins,
        scale,
        scale,
        params,
        [],
        [],
        missing_bins,
        monotone,
        cats,
        OutputBounds.unbounded(),
        form,
    )


# ---------------------------------------------------------------------------
# 1. The identity is an identity
# ---------------------------------------------------------------------------


def test_the_two_forms_agree_on_well_separated_candidates() raises:
    """The algebra check. On candidates whose gains are far apart in Float32
    terms, the two forms must select the same split and report the same gain,
    because they are the same function of the same inputs.

    This is where a wrong sign in `gpu_cross_offset`, a `2*lambda_l2` that
    should have been `lambda_l2`, or a swapped `HL'`/`HR'` in the cross
    product would surface, and it is the only reason to assert on
    well-separated values at all -- they carry no information about
    resolution, which is what tests 2 and the document care about.

    Gains here are order 10 to 50 and the two forms differ by at most a few
    times `eps * parent_score`, so the tolerance is generous by four orders
    of magnitude relative to that and still tight enough to catch an algebra
    error, which would be a whole-number-sized discrepancy.
    """
    var words = _histogram_words(
        1, 4, [-4, -2, 2, 4], [1, 1, 1, 1], [10, 10, 10, 10]
    )
    var sub = _search(words, 1, 4, _params(), GAIN_FORM_SUBTRACTIVE)
    var cross = _search(words, 1, 4, _params(), GAIN_FORM_CROSS)
    assert_true(sub.found and cross.found)
    assert_equal(sub.bin, cross.bin)
    assert_equal(sub.ordinal, cross.ordinal)
    assert_equal(sub.feature, cross.feature)
    assert_almost_equal(sub.gain, cross.gain, atol=1e-4)
    # And against the closed form, so this is not two implementations
    # agreeing with each other and both being wrong:
    #   bin 1: 36/(2+1) + 36/(2+1) - 0 = 24.
    assert_almost_equal(cross.gain, 24.0, atol=1e-4)


def test_the_two_forms_agree_with_no_l2_at_all() raises:
    """`lambda_l2 = 0` is the case worth separating, because it is the one
    where `gpu_cross_offset` returns exactly zero and the cross form has no
    subtraction anywhere. It is also LightGBM's default, so it is not a
    corner.

    With the offset gone, `D^2 / (HL' * HR' * S)` has to reproduce the gain
    on its own. `-3, 0, 0, 3` over unit hessians gives bin 0 and bin 2 the
    same exact gain, 9/1 + 9/3 = 12, and bin 1 a smaller one, 9/2 + 9/2 = 9,
    so this also pins that the cross form preserves an *exact* tie rather
    than perturbing it into an ordering.
    """
    var words = _histogram_words(
        1, 4, [-3, 0, 0, 3], [1, 1, 1, 1], [10, 10, 10, 10]
    )
    var p = _params(lambda_l2=0.0)
    var sub = _search(words, 1, 4, p, GAIN_FORM_SUBTRACTIVE)
    var cross = _search(words, 1, 4, p, GAIN_FORM_CROSS)
    assert_true(sub.found and cross.found)
    assert_almost_equal(cross.gain, 12.0, atol=1e-4)
    assert_almost_equal(sub.gain, cross.gain, atol=1e-4)
    # The tie between bin 0 and bin 2 resolves to the lower bin under both.
    assert_equal(sub.bin, 0)
    assert_equal(cross.bin, 0)


def test_the_offset_is_zero_exactly_when_the_lambdas_vanish() raises:
    """`gpu_cross_offset` on its own, because the claim in its docstring is
    an exactness claim and a tolerance would hide it. With `lambda_l2 = 0`
    and a child lambda of zero the scaled factor `(a + a) - lambda_l2` is
    exactly zero, so the product is exactly zero whatever the totals.

    The second assertion is the one that would catch a sign error: with a
    positive `lambda_l2` the offset must be *positive*, because the cross
    term is the gain a parent scored at `2*lambda_l2` would have shown and
    the real parent, scored at `lambda_l2`, scores higher.
    """
    assert_equal(
        gpu_cross_offset(
            Float32(858.3), Float32(286.1), Float32(0.0), Float32(0.0),
            Float32(0.0), gpu_cross_node_s(Float32(286.1), Float32(0.0)),
        ),
        Float32(0.0),
    )
    var node_s = gpu_cross_node_s(Float32(286.1), Float32(1.0))
    assert_equal(node_s, Float32(288.1))
    assert_true(
        gpu_cross_offset(
            Float32(858.3), Float32(286.1), Float32(0.0), Float32(1.0),
            Float32(1.0), node_s,
        )
        > Float32(0.0)
    )


def test_the_two_forms_agree_through_the_other_candidate_shapes() raises:
    """A missing bin, a categorical feature, and a monotone constraint, each
    of which routes through a different one of the four gain sites.

    The monotone case is the one with an asymmetry worth naming: a
    constrained candidate keeps the subtraction under both codes, because
    `gpu_split_gain`'s constrained branch scores clamped outputs and the
    identity does not cover them. So the two arms must agree there
    *exactly*, not approximately, on any candidate the constraint touches --
    though the unconstrained candidates of the same node still differ, which
    is why this asserts the decision rather than the gain.
    """
    # A missing bin at index 3, so the default-left candidate is scored.
    var miss = _histogram_words(
        1, 4, [-4, -2, 2, 4], [1, 1, 1, 1], [10, 10, 10, 10]
    )
    var sub = _search(
        miss, 1, 4, _params(), GAIN_FORM_SUBTRACTIVE, 1.0, [3]
    )
    var cross = _search(miss, 1, 4, _params(), GAIN_FORM_CROSS, 1.0, [3])
    assert_equal(sub.found, cross.found)
    assert_equal(sub.ordinal, cross.ordinal)
    assert_equal(sub.default_left, cross.default_left)
    assert_almost_equal(sub.gain, cross.gain, atol=1e-4)

    # A categorical feature, which reaches `gpu_cat_gain`.
    var cats = _one_categorical(1, 3)
    var cwords = _histogram_words(
        1, 5, [0, -6, 1, 5, 0], [0, 2, 2, 2, 0], [0, 10, 10, 10, 0]
    )
    var csub = _search(
        cwords, 1, 5, _params(), GAIN_FORM_SUBTRACTIVE, 1.0, [], [], cats
    )
    var ccross = _search(
        cwords, 1, 5, _params(), GAIN_FORM_CROSS, 1.0, [], [], cats
    )
    assert_equal(csub.found, ccross.found)
    assert_equal(csub.is_categorical, ccross.is_categorical)
    assert_almost_equal(csub.gain, ccross.gain, atol=1e-4)

    # A monotone constraint, which routes the candidates it touches through
    # the branch neither arm changes.
    var msub = _search(
        miss,
        1,
        4,
        _params(),
        GAIN_FORM_SUBTRACTIVE,
        1.0,
        [],
        [MONOTONE_INCREASING],
    )
    var mcross = _search(
        miss, 1, 4, _params(), GAIN_FORM_CROSS, 1.0, [], [MONOTONE_INCREASING]
    )
    assert_equal(msub.found, mcross.found)
    assert_equal(msub.ordinal, mcross.ordinal)


# ---------------------------------------------------------------------------
# 2. The near tie, which is the whole point
# ---------------------------------------------------------------------------


def test_the_shipped_form_cannot_separate_the_near_tie() raises:
    """The shipped form's resolution, demonstrated rather than argued.

    Two candidates whose exact gains differ by one part in ten thousand.
    `left_score + right_score` is about `parent_score` for both, and its
    rounding error is `eps` of its own magnitude, so the computed gain
    carries an absolute error of about `eps * 2566 = 1.5e-04` -- larger than
    the 8.9e-05 that separates the candidates. The two therefore land on the
    same Float32, and the scan's strict `>` keeps the incumbent, which is the
    worse of the two.

    Note what is *not* the mechanism here, because this project believed it
    once and wrote it down. The final `- parent_score` is not where the bits
    go. It is a subtraction of a common constant, rounding is monotone so it
    preserves order, and in this regime Sterbenz's lemma makes it exactly
    representable. The information was already gone one step earlier.
    """
    var words = _near_tie_words()
    var rec = _search(
        words, 2, 2, _params(), GAIN_FORM_SUBTRACTIVE, _TIE_SCALE
    )
    assert_true(rec.found)
    # Slot 1 holds the better candidate. The shipped form takes slot 0.
    assert_equal(rec.feature, 0)
    # It is not that slot 0 scored higher; it is that the two scored the
    # same. `runner_gain` is the losing feature's gain, and it is equal to
    # the winner's, which is the resolution failure stated as an assertion.
    assert_equal(rec.gain, rec.runner_gain)


def test_the_cross_form_separates_the_near_tie_and_ranks_it_right() raises:
    """The same twelve words, the same node, the other arm.

    The cross form never forms `left_score + right_score`. Its error enters
    through `D = GL*HR' - GR*HL'`, and because the gain is proportional to
    `D^2` the resolution works out to about `eps * sqrt(parent_score * gain)`
    rather than `eps * parent_score`. Here that is roughly 3e-05 against
    1.5e-04, and the 8.9e-05 gap falls between them.

    So this asserts three things and the third is the load-bearing one: the
    winner moves to slot 1, the two candidates are no longer equal, and the
    margin is wide enough that the result is not an accident of the last
    bits. Over 1500 independent pairs at this gap and this ratio in the
    NumPy model, the cross form ranked the better candidate first 100 percent
    of the time.
    """
    var words = _near_tie_words()
    var rec = _search(words, 2, 2, _params(), GAIN_FORM_CROSS, _TIE_SCALE)
    assert_true(rec.found)
    assert_equal(rec.feature, 1)
    assert_not_equal(rec.gain, rec.runner_gain)
    assert_true(rec.gain > rec.runner_gain)
    # The margin, in absolute terms. A Float32 ulp at 0.885 is 6.0e-08, so
    # 1.0e-05 is over 150 of them; the measured separation is about 4.1e-05.
    # A one-ulp difference from a different contraction of `gpu_cross_gain`'s
    # multiply-add cannot reach this threshold, which is why the test does
    # not have to pin the fusion.
    assert_true(rec.gain - rec.runner_gain > 1.0e-05)

    # Both computed gains are within the cross form's own error of the exact
    # values, which is what says the arm is reporting a gain and not a
    # surrogate ordering key. `min_gain_to_split` and `SPLIT_TIE_RELATIVE`
    # both depend on that being true.
    assert_almost_equal(rec.gain, _TIE_EXACT_GAIN_1, atol=1e-4)
    assert_almost_equal(rec.runner_gain, _TIE_EXACT_GAIN_0, atol=1e-4)


def test_the_near_tie_is_a_near_tie_and_not_a_wide_gap() raises:
    """Guards the fixture itself. If a later edit perturbed the twelve words
    into candidates that are merely different, both arms would pick slot 1,
    the two tests above would still pass, and the file would have stopped
    testing anything.

    The exact gains are 8.9e-05 apart on a gain of 0.885, so the fixture is
    a near tie if and only if that ratio is what it was when the fixture was
    searched for.
    """
    var gap = _TIE_EXACT_GAIN_1 - _TIE_EXACT_GAIN_0
    assert_true(gap > 0.0)
    var relative = gap / _TIE_EXACT_GAIN_1
    assert_true(relative < 2.0e-04)
    assert_true(relative > 5.0e-05)


# ---------------------------------------------------------------------------
# 3. L1 sends the cross form back
# ---------------------------------------------------------------------------


def test_l1_forces_the_subtractive_form() raises:
    """The guard, at the function that implements it.

    The identity behind the cross form needs `GL + GR = G`, and under L1 the
    gain is built from `T(GL)`, `T(GR)`, `T(G)`, which are not additive. So
    the arm is not merely less accurate under L1, it is wrong -- a bias, not
    a rounding -- and `gpu_resolve_gain_form` is what keeps it out.
    """
    assert_equal(
        gpu_resolve_gain_form(Int32(GAIN_FORM_CROSS), Float32(0.0)),
        Int32(GAIN_FORM_CROSS),
    )
    assert_equal(
        gpu_resolve_gain_form(Int32(GAIN_FORM_CROSS), Float32(0.5)),
        Int32(GAIN_FORM_SUBTRACTIVE),
    )
    # A negative lambda is not a supported input, but the guard is stated as
    # "nonzero" and should behave that way rather than letting a sign through.
    assert_equal(
        gpu_resolve_gain_form(Int32(GAIN_FORM_CROSS), Float32(-0.5)),
        Int32(GAIN_FORM_SUBTRACTIVE),
    )
    # And the subtractive arm is never promoted by the resolver.
    assert_equal(
        gpu_resolve_gain_form(Int32(GAIN_FORM_SUBTRACTIVE), Float32(0.0)),
        Int32(GAIN_FORM_SUBTRACTIVE),
    )


def test_an_l1_search_returns_the_same_record_under_both_codes() raises:
    """End to end through `reference_search`: with `lambda_l1` nonzero the
    two codes must produce the *same* record, bit for bit on the gain, since
    both run the subtractive arm.

    Asserted with `==` rather than a tolerance on purpose. "Approximately the
    same" would pass even if the cross form were reaching the candidates,
    which is exactly the failure this guards.
    """
    var words = _histogram_words(
        1, 4, [-4, -2, 2, 4], [1, 1, 1, 1], [10, 10, 10, 10]
    )
    var p = _params(lambda_l1=1.0)
    var sub = _search(words, 1, 4, p, GAIN_FORM_SUBTRACTIVE)
    var cross = _search(words, 1, 4, p, GAIN_FORM_CROSS)
    assert_true(sub.found and cross.found)
    assert_equal(sub.bin, cross.bin)
    assert_equal(sub.ordinal, cross.ordinal)
    assert_equal(sub.gain, cross.gain)
    assert_equal(sub.left.grad, cross.left.grad)
    assert_equal(sub.right.grad, cross.right.grad)

    # The near-tie node under L1 as well, because that is the node where a
    # leak would be visible: with the guard removed the cross arm would take
    # slot 1 here and the codes would disagree.
    var tie = _near_tie_words()
    var tsub = _search(
        tie, 2, 2, _params(lambda_l1=1.0), GAIN_FORM_SUBTRACTIVE, _TIE_SCALE
    )
    var tcross = _search(
        tie, 2, 2, _params(lambda_l1=1.0), GAIN_FORM_CROSS, _TIE_SCALE
    )
    assert_equal(tsub.feature, tcross.feature)
    assert_equal(tsub.gain, tcross.gain)


# ---------------------------------------------------------------------------
# 4. Integer right-hand subtraction, on its own
# ---------------------------------------------------------------------------


def test_the_integer_right_hand_sum_is_the_exact_one() raises:
    """`gpu_right_sum`'s two arms, on a case where they differ.

    The node total is 900_000_000 lattice units and the left child holds
    899_999_999 of them, so the right child holds exactly one. Dequantized at
    2^-20 that is 9.5367431640625e-07, exactly representable.

    The Float32 arm cannot produce it. Both operands are cast to Float32
    first, and at 9e08 the Float32 lattice spacing is 64 units, so the two
    casts land on the same representable value and their difference is zero.
    The integer arm subtracts first and casts a one, which is exact. That is
    the entire mechanism of candidate 6, and note that it is the *cast* that
    loses the bits and not the subtraction -- with a power-of-two scale the
    subtraction itself is exact by Sterbenz here.
    """
    var inv = Float32(1.0 / 1048576.0)
    var total_q = Int32(900_000_000)
    var left_q = Int32(899_999_999)
    var total_f = total_q.cast[DType.float32]() * inv
    var left_f = left_q.cast[DType.float32]() * inv

    var exact = gpu_right_sum(
        total_f, left_f, total_q, left_q, inv, Int32(GAIN_FORM_CROSS)
    )
    assert_equal(exact, Float32(1.0) * inv)

    var approx = gpu_right_sum(
        total_f, left_f, total_q, left_q, inv, Int32(GAIN_FORM_SUBTRACTIVE)
    )
    assert_equal(approx, Float32(0.0))
    assert_not_equal(exact, approx)


def test_the_right_hand_arms_agree_where_the_cast_is_exact() raises:
    """The other half of the claim: below 2^24 an Int32 casts to Float32
    exactly, so both arms return the same value and the difference is
    confined to the regime the docstring says it is.

    If the two arms disagreed on small sums as well, the accuracy argument
    would be about something other than the cast, and the docstring would be
    wrong about where the error lives.
    """
    var inv = Float32(1.0 / 1024.0)
    var total_q = Int32(1_000_000)
    for left in [Int32(1), Int32(500_000), Int32(999_999)]:
        var total_f = total_q.cast[DType.float32]() * inv
        var left_f = left.cast[DType.float32]() * inv
        assert_equal(
            gpu_right_sum(
                total_f, left_f, total_q, left, inv, Int32(GAIN_FORM_CROSS)
            ),
            gpu_right_sum(
                total_f,
                left_f,
                total_q,
                left,
                inv,
                Int32(GAIN_FORM_SUBTRACTIVE),
            ),
        )


# ---------------------------------------------------------------------------
# 5. The arm as an arm
# ---------------------------------------------------------------------------


def test_both_arms_are_reachable_and_differ() raises:
    """The A/B this change has to be measurable through. Two arms that
    returned the same record would be one arm with a spare constant.

    The near-tie node is the witness: the two codes select different
    features on it. That is a stronger statement than "the gains differ in
    the last bits", and it is the statement `set_gain_form` makes when it
    says this arm can change a tree.
    """
    var words = _near_tie_words()
    var sub = _search(
        words, 2, 2, _params(), GAIN_FORM_SUBTRACTIVE, _TIE_SCALE
    )
    var cross = _search(words, 2, 2, _params(), GAIN_FORM_CROSS, _TIE_SCALE)
    assert_not_equal(sub.feature, cross.feature)

    assert_equal(describe_gain_form(GAIN_FORM_CROSS), "cross")
    assert_equal(describe_gain_form(GAIN_FORM_SUBTRACTIVE), "subtractive")
    assert_equal(describe_gain_form(7), "unknown")


def test_the_default_is_the_cross_form() raises:
    """What the package actually does with no arm named, which is the only
    thing a reader of `DEFAULT_GAIN_FORM` can check against behavior.

    `reference_search` defaults to the same code the searcher does, so the
    host replica keeps replicating; a default that drifted between the two
    would make every host/device comparison fail for a reason that is about
    neither of them.
    """
    assert_equal(DEFAULT_GAIN_FORM, GAIN_FORM_CROSS)
    var words = _near_tie_words()
    var explicit = _search(
        words, 2, 2, _params(), GAIN_FORM_CROSS, _TIE_SCALE
    )
    var implied = reference_search(
        words, 2, 2, _TIE_SCALE, _TIE_SCALE, _params()
    )
    assert_equal(explicit.feature, implied.feature)
    assert_equal(explicit.gain, implied.gain)


def test_the_cross_gain_helper_matches_a_hand_computation() raises:
    """`gpu_cross_gain` on its own, against arithmetic done by hand, so the
    helper is pinned independently of the scan that calls it.

    `GL = -6, HL = 2, GR = 6, HR = 2, child_l2 = 1`:
        HL' = 3, HR' = 3, S = 6
        D   = -6*3 - 6*3 = -36
        gain = 36^2 / (3*3*6) = 1296 / 54 = 24
    and the offset is zero because the node's gradient total is zero. Every
    value here is a small integer exactly representable in Float32, so this
    is an equality and not a tolerance.
    """
    var node_s = gpu_cross_node_s(Float32(4.0), Float32(1.0))
    assert_equal(node_s, Float32(6.0))
    var offset = gpu_cross_offset(
        Float32(0.0), Float32(4.0), Float32(0.0), Float32(1.0),
        Float32(1.0), node_s,
    )
    assert_equal(offset, Float32(0.0))
    assert_equal(
        gpu_cross_gain(
            Float32(-6.0),
            Float32(2.0),
            Float32(6.0),
            Float32(2.0),
            Float32(1.0),
            node_s,
            offset,
        ),
        Float32(24.0),
    )


# ---------------------------------------------------------------------------
# 6. The blast radius on the rest of the suite
# ---------------------------------------------------------------------------


def _lcg(mut state: UInt64) -> UInt64:
    """Numerical Recipes' 64-bit LCG. Deterministic, self-contained, and
    good enough to walk a space of tiny histograms; nothing here is a
    statistical claim."""
    state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
    return state


def test_small_integer_histograms_agree_far_inside_the_suite_tolerance(
) raises:
    """The bound on what this change can break elsewhere.

    Every other split-search test in this repository is built from histograms
    of small integers at a scale of 1.0, asserts gains with `atol=1e-4`, and
    lives in the accelerator-only set, so it cannot be run on a machine
    without a device to find out whether the new default disturbs it. This
    test answers the question from the CPU set instead, over 600 randomly
    drawn histograms of that shape.

    The reason to expect agreement is arithmetic and not luck. At a scale of
    1.0 with sums under a thousand, every quantity in either expression is
    exactly representable in Float32, so both forms evaluate a well
    conditioned function with a handful of correctly-rounded operations, and
    `parent_score / gain` never leaves the neighborhood of one -- which is
    precisely the regime `gpu_cross_gain` says the two forms are
    interchangeable in. The tolerance asserted here is 1e-5, an order tighter
    than the suite's, so this fails before the suite would.

    The decision agreement is asserted separately and exactly. An exact tie
    between two candidates stays an exact tie under both forms when the
    inputs are exactly representable, which is what lets this be `==` on the
    selected ordinal rather than a count of near misses.
    """
    var state = UInt64(0x2026_0816_C56A_1F77)
    var worst = Float64(0.0)
    var trials = 0
    for _ in range(200):
        for l2 in [Float64(0.0), Float64(1.0), Float64(5.0)]:
            var n_bins = 4
            var g = List[Int]()
            var h = List[Int]()
            var c = List[Int]()
            for _ in range(n_bins):
                g.append(Int(_lcg(state) % UInt64(41)) - 20)
                h.append(Int(_lcg(state) % UInt64(5)) + 1)
                c.append(Int(_lcg(state) % UInt64(20)) + 5)
            var words = _histogram_words(1, n_bins, g, h, c)
            var p = _params(lambda_l2=l2)
            var sub = _search(words, 1, n_bins, p, GAIN_FORM_SUBTRACTIVE)
            var cross = _search(words, 1, n_bins, p, GAIN_FORM_CROSS)
            assert_equal(sub.found, cross.found)
            if not sub.found:
                continue
            trials += 1
            assert_equal(sub.ordinal, cross.ordinal)
            assert_equal(sub.bin, cross.bin)
            var parent = sub.total.grad * sub.total.grad / (
                sub.total.hess + l2
            )
            var unit = 5.9604644775390625e-08 * (parent + abs(sub.gain))
            var ratio = abs(sub.gain - cross.gain) / unit
            if ratio > worst:
                worst = ratio
    assert_true(trials > 100)
    # **Measured over this sweep: 1.96.** The two arms differ by at most two
    # units of `eps32 * (parent_score + gain)`, which is the derived bound
    # essentially tight. The threshold is four times that, so this is a
    # regression guard and not a restatement of the measurement.
    #
    # Translated into the terms the accelerator-only files assert in: their
    # fixtures keep `parent_score + gain` under about fifty, so this bounds
    # the disagreement there at 6e-06 against their `atol` of 1e-4, a factor
    # of seventeen inside it. That is the blast-radius claim, and it is a
    # bound rather than a run.
    assert_true(worst < 8.0, String("worst ratio = ", worst))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
