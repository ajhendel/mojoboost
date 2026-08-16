"""`score_function`: CatBoost's Cosine split score as an opt-in on the CPU.

`split.SCORE_L2` is the default and is the second-order gain the package has
always maximized. `split.SCORE_COSINE` is CatBoost's CPU default
(`docs/design/CATBOOST_CATALOG.md` A10, verified from source): a RATIO of two
cross-child accumulators rather than a sum.

What these tests are for, in the order they matter:

1. **The default did not move.** Every entry point called with no
   `score_function`, and called with `SCORE_L2` explicitly, returns the same
   split down to the bits of the gain. This is the property the golden
   fixtures depend on and the reason none of them had to be regenerated.

2. **The gate provably opens.** At `lambda_l2 = 3` a hand-built two-feature
   histogram makes L2 choose `(feature 1, bin 1)` and Cosine choose
   `(feature 0, bin 3)`. A test that only checked "cosine runs" would pass
   with the parameter ignored; this one cannot. The fixture's numbers are
   printed in `_diverging_*` below with the gain of every candidate under
   both score functions, and the winner clears its runner-up by more than 30
   percent in each arm, so the divergence is structural and not a tie-break.

3. **The headline result, which is a null at stock.** At `lambda_l2 = 0` --
   which is what mojotrees stock now uses -- Cosine's numerator and
   denominator are the same expression, so the score collapses to `sqrt` of
   the L2 score and the argmax cannot move. The SAME fixture that diverges at
   `lambda_l2 = 3` agrees at `lambda_l2 = 0`. That is the result this lane
   exists to establish and it is asserted rather than asserted-about.

   The identity is mathematical, not bitwise: Cosine spells the numerator
   `(G / (H + lambda)) * G` where L2 spells it `G * G / (H + lambda)`, which
   is one reassociated multiply, so two candidates within an ulp of each
   other could still swap. The fixtures below are chosen with wide margins
   for exactly that reason, and no assertion here compares a Cosine float to
   an L2 float.

4. **The oblivious level reduces to the node.** At a level of one leaf,
   `find_best_split_shared` under Cosine returns `find_best_split`'s answer to
   the bit -- the two-accumulator cross-leaf fold, the level parent term and
   the single ratio all have to be right for that to hold.

5. **The illegal-leaf rule.** A leaf that cannot take the level's candidate
   contributes its UNSPLIT terms under Cosine, which is the exact
   generalization of contributing 0.0 to a sum of differences under L2. The
   `SharedSplitAudit` counter is asserted in both arms, so the test fails if
   the rule ever becomes a veto and if nobody is ever illegal.

6. **Determinism.** Identical bits at `MOJOTREES_NUM_WORKERS` 1, 3 and 8 on
   both entry points under Cosine.

No tolerance appears anywhere below. Every float comparison is on
`to_bits()`; everything else is an `Int` or a `Bool`.
"""

from std.os import setenv
from std.testing import (
    assert_equal,
    assert_false,
    assert_not_equal,
    assert_raises,
    assert_true,
    TestSuite,
)

from mojotrees.growth_policy import SharedSplitAudit
from mojotrees.histogram import Histogram
from mojotrees.split import (
    SCORE_COSINE,
    SCORE_L2,
    SplitInfo,
    check_score_function,
    find_best_split,
    find_best_split_shared,
)


# --------------------------------------------------------------------- fixtures


def _hist(grad: List[Float64], hess: List[Float64], n_bins: Int) -> Histogram:
    """A histogram from two flat `[f * n_bins + b]` planes, counts taken as
    the rounded hessians so `min_data_in_leaf` never binds in these tests."""
    var n_features = len(grad) // n_bins
    var count = List[Int](capacity=len(grad))
    for i in range(len(grad)):
        count.append(Int(hess[i]))
    return Histogram.from_planes(
        grad.copy(), hess.copy(), count^, n_features, n_bins
    )


def _diverging_grad() -> List[Float64]:
    """Feature 0 then feature 1, five bins each, no missing bin.

    The whole fixture, with both features carrying the same node totals
    (G = -9.1, H = 30), which is what a real per-feature histogram of one
    node looks like. Gains per threshold bin 0..3:

        lambda = 3   f0  L2   0.357552  7.469837 12.827684  8.575919
                     f0  cos  0.145977  1.759994  2.740567  4.848654
                     f1  L2   3.991717 17.384221  8.624416  2.658788
                     f1  cos  1.454691  3.639793  1.960782  0.775454
          -> L2 picks (f1, bin 1); Cosine picks (f0, bin 3)

        lambda = 0   f0  L2   0.508189  8.970667 16.629093 41.015184
                     f0  cos  0.146481  1.763630  2.741917  4.954882
                     f1  L2   6.960083 25.405816 10.368000  3.190917
                     f1  cos  1.456333  3.645754  1.961879  0.778093
          -> both pick (f0, bin 3)

    Those numbers are the design of the fixture, not an assertion: what is
    asserted below is the chosen feature and bin, which are integers.
    """
    return [
        -0.7, 5.4, 0.3, -7.5, -6.6,
        -7.6, -6.2, -0.3, 3.1, 1.9,
    ]


def _diverging_hess() -> List[Float64]:
    return [
        8.0, 2.0, 9.0, 10.0, 1.0,
        6.0, 1.0, 11.0, 4.0, 8.0,
    ]


def _diverging() -> Histogram:
    return _hist(_diverging_grad(), _diverging_hess(), 5)


def _wide(
    n_features: Int, n_bins: Int, n_rows: Int, seed: UInt64 = 20260816
) -> Histogram:
    """A fixture wide enough for the feature dispatch to actually fan out.

    Built by ACCUMULATING ROWS rather than by filling cells independently,
    which matters and is not cosmetic. Every feature of a real node histogram
    sums to the same node totals, so every feature faces the same parent
    score; a cell-wise fixture gives each feature its own totals, and the
    `lambda_l2 = 0` identity then fails for a reason that has nothing to do
    with the code -- `sqrt(a) - sqrt(p1)` against `b - p2` is not the same
    comparison as `a - p1` against `b - p2` when `p1 != p2`. An earlier draft
    of this file used a cell-wise fixture and the null test failed on it. The
    identity is a statement about one node, and the fixture has to be one
    node.
    """
    var n = n_features * n_bins
    var grad = List[Float64](capacity=n)
    var hess = List[Float64](capacity=n)
    var count = List[Int](capacity=n)
    for _ in range(n):
        grad.append(0.0)
        hess.append(0.0)
        count.append(0)
    var state = seed
    for _ in range(n_rows):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float64(state >> 11) * (1.0 / 9007199254740992.0)
        var g = 4.0 * u - 2.0
        state = state * 6364136223846793005 + 1442695040888963407
        var v = Float64(state >> 11) * (1.0 / 9007199254740992.0)
        var hv = 0.5 + 2.0 * v
        for f in range(n_features):
            state = state * 6364136223846793005 + 1442695040888963407
            var w = Float64(state >> 11) * (1.0 / 9007199254740992.0)
            var b = Int(w * Float64(n_bins))
            if b >= n_bins:
                b = n_bins - 1
            grad[f * n_bins + b] += g
            hess[f * n_bins + b] += hv
            count[f * n_bins + b] += 1
    return Histogram.from_planes(grad^, hess^, count^, n_features, n_bins)


def _same_split(a: SplitInfo, b: SplitInfo) -> Bool:
    return (
        a.found == b.found
        and a.feature == b.feature
        and a.bin == b.bin
        and a.default_left == b.default_left
        and a.is_categorical == b.is_categorical
        and a.gain.to_bits() == b.gain.to_bits()
    )


def _assert_same(a: SplitInfo, b: SplitInfo) raises:
    assert_equal(a.found, b.found)
    assert_equal(a.feature, b.feature)
    assert_equal(a.bin, b.bin)
    assert_equal(a.default_left, b.default_left)
    assert_equal(a.is_categorical, b.is_categorical)
    assert_equal(Int(a.gain.to_bits()), Int(b.gain.to_bits()))


def _serial():
    _ = setenv("MOJOTREES_NUM_WORKERS", "1")


def _workers(n: Int):
    _ = setenv("MOJOTREES_NUM_WORKERS", String(n))


def _auto():
    _ = setenv("MOJOTREES_NUM_WORKERS", "")


# ------------------------------------------------------ 1. the default is intact


def test_default_selector_is_l2() raises:
    """Not passing `score_function` and passing `SCORE_L2` are the same call.

    This is the assertion the golden fixtures rest on: `find_best_split`'s
    default path was not modified, only branched around, and the branch is
    not taken at the default."""
    _serial()
    var h = _diverging()
    for lam in [0.0, 1.0, 3.0]:
        var implicit = find_best_split(h, lambda_reg=lam, min_child_hess=1e-3)
        var explicit = find_best_split(
            h, lambda_reg=lam, min_child_hess=1e-3, score_function=SCORE_L2
        )
        assert_true(implicit.found)
        _assert_same(implicit, explicit)
    _auto()


def test_default_selector_is_l2_shared() raises:
    _serial()
    var hists: List[Histogram] = [_diverging()]
    for lam in [0.0, 1.0, 3.0]:
        var a1 = SharedSplitAudit.none()
        var implicit = find_best_split_shared(
            a1, hists, lambda_reg=lam, min_child_hess=1e-3
        )
        var a2 = SharedSplitAudit.none()
        var explicit = find_best_split_shared(
            a2,
            hists,
            lambda_reg=lam,
            min_child_hess=1e-3,
            score_function=SCORE_L2,
        )
        assert_true(implicit.found)
        _assert_same(implicit, explicit)
        assert_equal(a1.n_leaves, a2.n_leaves)
        assert_equal(a1.n_illegal, a2.n_illegal)
        assert_equal(a1.n_scored, a2.n_scored)
    _auto()


def test_unknown_selector_is_refused() raises:
    """A selector nobody implemented must not fall through to L2 and hand the
    caller a plausible answer to a question they did not ask."""
    var h = _diverging()
    with assert_raises(contains="score_function must be"):
        _ = find_best_split(h, lambda_reg=3.0, score_function=2)
    with assert_raises(contains="score_function must be"):
        _ = find_best_split(h, lambda_reg=3.0, score_function=-1)
    var audit = SharedSplitAudit.none()
    var hists: List[Histogram] = [_diverging()]
    with assert_raises(contains="score_function must be"):
        _ = find_best_split_shared(
            audit, hists, lambda_reg=3.0, score_function=7
        )
    check_score_function(SCORE_L2)
    check_score_function(SCORE_COSINE)


# ------------------------------------------- 2. the gate provably opens at lambda > 0


def test_cosine_picks_a_different_split_at_positive_lambda() raises:
    """THE gate assertion. At `lambda_l2 = 3` the two score functions choose
    different features AND different bins on this fixture, so the parameter
    cannot be silently ignored and the test still pass."""
    _serial()
    var h = _diverging()
    var l2 = find_best_split(h, lambda_reg=3.0, min_child_hess=1e-3)
    var cos = find_best_split(
        h, lambda_reg=3.0, min_child_hess=1e-3, score_function=SCORE_COSINE
    )

    assert_true(l2.found)
    assert_true(cos.found)
    assert_equal(l2.feature, 1)
    assert_equal(l2.bin, 1)
    assert_equal(cos.feature, 0)
    assert_equal(cos.bin, 3)
    assert_not_equal(Int(l2.gain.to_bits()), Int(cos.gain.to_bits()))
    assert_true(cos.gain > 0.0)
    _auto()


def test_cosine_picks_a_different_split_at_positive_lambda_shared() raises:
    """The same divergence through the level search, at a level of one leaf.
    The cross-leaf accumulators, the level parent term and the single ratio
    all have to be right for this to reproduce."""
    _serial()
    var hists: List[Histogram] = [_diverging()]
    var a1 = SharedSplitAudit.none()
    var l2 = find_best_split_shared(
        a1, hists, lambda_reg=3.0, min_child_hess=1e-3
    )
    var a2 = SharedSplitAudit.none()
    var cos = find_best_split_shared(
        a2,
        hists,
        lambda_reg=3.0,
        min_child_hess=1e-3,
        score_function=SCORE_COSINE,
    )
    assert_equal(l2.feature, 1)
    assert_equal(l2.bin, 1)
    assert_equal(cos.feature, 0)
    assert_equal(cos.bin, 3)
    assert_equal(a2.n_leaves, 1)
    assert_equal(a2.n_illegal, 0)
    assert_equal(a2.n_scored, 1)
    _auto()


# ------------------------------- 3. the headline: a null at stock's lambda_l2 = 0


def test_cosine_is_a_no_op_at_zero_lambda() raises:
    """At `lambda_l2 = 0` -- mojotrees stock -- Cosine and L2 have the same
    argmax, provably: the numerator and the denominator become the same
    expression and the ratio collapses to `sqrt` of the L2 score.

    Asserted on the SAME fixture that diverges at `lambda_l2 = 3`, so this is
    not a fixture that happens to agree. Only the chosen split is compared;
    the gains are on different scales by construction and are never compared
    to each other."""
    _serial()
    var h = _diverging()
    var l2 = find_best_split(h, lambda_reg=0.0, min_child_hess=1e-3)
    var cos = find_best_split(
        h, lambda_reg=0.0, min_child_hess=1e-3, score_function=SCORE_COSINE
    )
    assert_true(l2.found)
    assert_true(cos.found)
    assert_equal(l2.feature, cos.feature)
    assert_equal(l2.bin, cos.bin)
    assert_equal(l2.default_left, cos.default_left)
    assert_equal(l2.feature, 0)
    assert_equal(l2.bin, 3)
    _auto()


def test_cosine_is_a_no_op_at_zero_lambda_wide() raises:
    """The same null on a wide pseudo-random histogram, where the winner is
    decided by the data rather than by a hand-built fixture. 24 features of
    32 bins: if the identity held only on the small fixture this would catch
    it."""
    _serial()
    var h = _wide(24, 32, 600)
    var l2 = find_best_split(h, lambda_reg=0.0, min_child_hess=1e-3)
    var cos = find_best_split(
        h, lambda_reg=0.0, min_child_hess=1e-3, score_function=SCORE_COSINE
    )
    assert_true(l2.found)
    assert_true(cos.found)
    assert_equal(l2.feature, cos.feature)
    assert_equal(l2.bin, cos.bin)
    assert_equal(l2.default_left, cos.default_left)
    _auto()


def test_cosine_admits_exactly_the_candidates_l2_admits() raises:
    """`gain > 0` is the admission test both score functions face, and at
    `lambda_l2 = 0` it admits the same set: `num > parent` if and only if
    `sqrt(num) > sqrt(parent)`. A histogram with no positive-gain candidate
    must therefore find nothing under both."""
    _serial()
    # One bin carries everything, so every threshold puts an empty child on
    # one side and no candidate can beat the unsplit node.
    var grad: List[Float64] = [0.0, 0.0, 4.0, 0.0]
    var hess: List[Float64] = [0.0, 0.0, 9.0, 0.0]
    var h = _hist(grad, hess, 4)
    var l2 = find_best_split(h, lambda_reg=0.0, min_child_hess=1e-3)
    var cos = find_best_split(
        h, lambda_reg=0.0, min_child_hess=1e-3, score_function=SCORE_COSINE
    )
    assert_false(l2.found)
    assert_false(cos.found)
    _auto()


# ------------------------------------- 4. the level reduces to the node under cosine


def test_shared_cosine_matches_single_node_cosine() raises:
    """A level of one leaf is one node. Bit-identical, not merely equal: the
    level path seeds the denominator, folds one leaf and subtracts the level
    parent, and each of those has to be the same arithmetic in the same order
    as `find_best_split`'s."""
    _serial()
    for lam in [0.0, 1.0, 3.0]:
        var h = _diverging()
        var node = find_best_split(
            h,
            lambda_reg=lam,
            min_child_hess=1e-3,
            score_function=SCORE_COSINE,
        )
        var audit = SharedSplitAudit.none()
        var hists: List[Histogram] = [_diverging()]
        var level = find_best_split_shared(
            audit,
            hists,
            lambda_reg=lam,
            min_child_hess=1e-3,
            score_function=SCORE_COSINE,
        )
        assert_true(node.found)
        _assert_same(node, level)
    _auto()


def test_shared_cosine_matches_single_node_cosine_wide() raises:
    _serial()
    var node = find_best_split(
        _wide(12, 16, 400),
        lambda_reg=3.0,
        min_child_hess=1e-3,
        score_function=SCORE_COSINE,
    )
    var audit = SharedSplitAudit.none()
    var hists: List[Histogram] = [_wide(12, 16, 400)]
    var level = find_best_split_shared(
        audit,
        hists,
        lambda_reg=3.0,
        min_child_hess=1e-3,
        score_function=SCORE_COSINE,
    )
    assert_true(node.found)
    _assert_same(node, level)
    _auto()


# --------------------------------- 5. the illegal-leaf rule under a ratio score


def _tiny_leaf() -> Histogram:
    """A leaf whose every child fails `min_child_hess = 1.0`, whatever the
    threshold, but which still carries gradient mass so its unsplit terms are
    not zero."""
    var grad: List[Float64] = [0.4, -0.3, 0.5, -0.2, 0.1]
    var hess: List[Float64] = [0.1, 0.1, 0.1, 0.1, 0.1]
    var count = List[Int](capacity=5)
    for _ in range(5):
        count.append(1)
    return Histogram.from_planes(grad^, hess^, count^, 1, 5)


def _big_leaf() -> Histogram:
    var grad: List[Float64] = [-0.7, 5.4, 0.3, -7.5, -6.6]
    var hess: List[Float64] = [8.0, 2.0, 9.0, 10.0, 1.0]
    var count: List[Int] = [8, 2, 9, 10, 1]
    return Histogram.from_planes(grad^, hess^, count^, 1, 5)


def test_illegal_leaf_does_not_veto_under_cosine() raises:
    """A leaf that can take no candidate contributes its unsplit terms and is
    split anyway. The audit counter is what proves the branch executed: the
    gate is `n_illegal == 1`, asserted, not assumed. Both arms of the rule are
    pinned -- the level is still found (no veto) and somebody was illegal (the
    fixture is not vacuous)."""
    _serial()
    var hists: List[Histogram] = [_big_leaf(), _tiny_leaf()]
    var audit = SharedSplitAudit.none()
    var level = find_best_split_shared(
        audit,
        hists,
        lambda_reg=3.0,
        min_child_hess=1.0,
        score_function=SCORE_COSINE,
    )
    assert_true(level.found)
    assert_equal(audit.n_leaves, 2)
    assert_equal(audit.n_illegal, 1)
    assert_equal(audit.n_scored, 1)
    assert_false(audit.all_illegal())

    # The L2 arm must reach the same accounting on the same fixture: the
    # Cosine branch changed what an illegal leaf contributes, not which
    # leaves are illegal.
    var l2_audit = SharedSplitAudit.none()
    var l2 = find_best_split_shared(
        l2_audit, hists, lambda_reg=3.0, min_child_hess=1.0
    )
    assert_true(l2.found)
    assert_equal(l2_audit.n_leaves, 2)
    assert_equal(l2_audit.n_illegal, 1)
    assert_equal(l2_audit.n_scored, 1)
    _auto()


def test_every_leaf_illegal_finds_nothing_under_cosine() raises:
    """A level in which no leaf can take any candidate must find nothing,
    under the ratio as under the sum. Under Cosine every candidate's
    accumulators then hold exactly the level's unsplit terms, so every
    candidate's gain is `p - p`, which is 0.0 and never beats the running
    best."""
    _serial()
    var hists: List[Histogram] = [_tiny_leaf(), _tiny_leaf()]
    var audit = SharedSplitAudit.none()
    var level = find_best_split_shared(
        audit,
        hists,
        lambda_reg=3.0,
        min_child_hess=1.0,
        score_function=SCORE_COSINE,
    )
    assert_false(level.found)
    var l2_audit = SharedSplitAudit.none()
    var l2 = find_best_split_shared(
        l2_audit, hists, lambda_reg=3.0, min_child_hess=1.0
    )
    assert_false(l2.found)
    _auto()


# ---------------------------------------------------------------- 6. determinism


def test_cosine_is_deterministic_across_workers() raises:
    """Identical bits at `MOJOTREES_NUM_WORKERS` 1, 3 and 8. The Cosine
    accumulators are folded inside one feature's own task, ascending by bin
    and then by leaf, so nothing the scheduler decides can reach a sum."""
    _serial()
    var h1 = _wide(40, 24, 900)
    var base = find_best_split(
        h1, lambda_reg=3.0, min_child_hess=1e-3, score_function=SCORE_COSINE
    )
    assert_true(base.found)
    for w in [3, 8, 1]:
        _workers(w)
        var h = _wide(40, 24, 900)
        var got = find_best_split(
            h, lambda_reg=3.0, min_child_hess=1e-3, score_function=SCORE_COSINE
        )
        assert_true(_same_split(base, got))
    _auto()


def test_cosine_shared_is_deterministic_across_workers() raises:
    """The same, through the level search over four leaves, where the
    cross-leaf fold is the sum that would move first if a leaf ever crossed a
    task boundary."""
    _serial()
    var hists0: List[Histogram] = [
        _wide(24, 16, 300, 20260816),
        _wide(24, 16, 300, 981721),
        _wide(24, 16, 300, 40507),
        _wide(24, 16, 300, 7717171),
    ]
    var a0 = SharedSplitAudit.none()
    var base = find_best_split_shared(
        a0,
        hists0,
        lambda_reg=3.0,
        min_child_hess=1e-3,
        score_function=SCORE_COSINE,
    )
    assert_true(base.found)
    for w in [3, 8, 1]:
        _workers(w)
        var hists: List[Histogram] = [
            _wide(24, 16, 300, 20260816),
            _wide(24, 16, 300, 981721),
            _wide(24, 16, 300, 40507),
            _wide(24, 16, 300, 7717171),
        ]
        var audit = SharedSplitAudit.none()
        var got = find_best_split_shared(
            audit,
            hists,
            lambda_reg=3.0,
            min_child_hess=1e-3,
            score_function=SCORE_COSINE,
        )
        assert_true(_same_split(base, got))
        assert_equal(audit.n_leaves, a0.n_leaves)
        assert_equal(audit.n_illegal, a0.n_illegal)
        assert_equal(audit.n_scored, a0.n_scored)
    _auto()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
