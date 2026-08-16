"""Whether the device split search resolves a tie the way the host does, and
if it does, what else is left to explain a CPU/GPU prediction disagreement.

The host's rule, read out of `split.find_best_split` rather than assumed:

- Within one feature, candidates are walked bins ascending, and inside a bin
  the missing-left candidate is scored before the missing-right one. Every
  acceptance is a strict `>` against a running best that starts at 0.0, so
  the *first* candidate in that order keeps a tie and a later one must
  strictly exceed to take it.
- Across features, each feature writes its own best into its own slot and
  the slots are folded serially in ascending scan order by the same strict
  `>`. So a cross-feature tie goes to the lowest scan slot, which is the
  lowest feature id whenever the caller's feature list ascends
  (`find_best_split` requires it to).
- The tolerance the host applies to "equal gain" is **zero**: the comparison
  is exact Float64 `>`. `SPLIT_TIE_RELATIVE` is not part of the host's
  decision at all; it is the width of `GpuSplitRecord.is_near_tie`, which is
  a *report* about how close a device decision was.

The device replica is `gpu_split_search.reference_search`, which calls the
kernels' own Float32 helpers and walks the candidates in the kernels' order,
so what these tests pin is the decision rule, not a second implementation of
it. `tests/test_gpu_split_search.mojo` pins the kernels to this replica, and
`tests/test_gpu_split_tie_parity.mojo` pins the kernels to these same
constructed ties on a machine that has an accelerator.

Two layers here:

1. **Constructed ties.** Histograms whose candidates have *exactly* equal
   gain, in every shape a tie can take: two bins of one feature, two
   features whose winners sit at different bins, and the two missing
   directions of one bin. Each is asserted under both gain forms, because
   `GAIN_FORM_CROSS` and `GAIN_FORM_SUBTRACTIVE` do not produce
   bit-identical gains and a tie policy that only held under one of them
   would not be a tie policy.

2. **The discriminating sweep.** Ties being resolved alike does not by
   itself explain the observed CPU/GPU disagreement, and could easily not be
   its cause. So the sweep runs a few hundred pseudo-random histograms
   through both searches and asks where they part company. If the two agree
   on every decision whose margin is comfortably outside Float32's
   resolution, then the tie *policy* is not what moves a split, and what
   moves it is the Float32 near-tie resolution the module already documents
   -- for which the remedy is `host_rescan_recommended`, not a different
   tie-break.
"""

from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_true,
    TestSuite,
)

from support import _splitmix64

from mojotrees.categorical import CategoricalParams
from mojotrees.gpu_split_search import (
    GAIN_FORM_CROSS,
    GAIN_FORM_SUBTRACTIVE,
    ChildStats,
    GpuSplitParams,
    GpuSplitRecord,
    host_rescan_recommended,
    reference_search,
)
from mojotrees.histogram import Histogram
from mojotrees.split import SplitInfo, find_best_split

comptime _TOL = 1e-4


def _forms() -> List[Int]:
    """The two arms every constructed tie is asserted under.
    `GAIN_FORM_CROSS` is the module default and `GAIN_FORM_SUBTRACTIVE` is
    what a run with `MOJOTREES_GPU_GAIN_FORM=subtractive`, or any nonzero
    `lambda_l1`, falls back to. A tie rule that held under only one of them
    would hold for only some runs."""
    return [GAIN_FORM_CROSS, GAIN_FORM_SUBTRACTIVE]


# --- Fixed-point histograms, the layout the GPU kernels consume ----------


def _zeroed(n: Int) -> List[Int32]:
    var out = List[Int32](capacity=n)
    out.resize(n, Int32(0))
    return out^


def _histogram_words(
    n_features: Int, n_bins: Int, g: List[Int], h: List[Int], c: List[Int]
) raises -> List[Int32]:
    """A `[grad | hess | count]` buffer from flat `[f * n_bins + b]` lists."""
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
    """The same numbers the host scan takes. Both fixed-point scales are 1.0
    in this file, so the conversion is exact and the two searches see
    identical values: any disagreement below is a disagreement about the
    decision, never about the input."""
    var size = n_features * n_bins
    var grad = List[Float64](capacity=size)
    var hess = List[Float64](capacity=size)
    var count = List[Int](capacity=size)
    for i in range(size):
        grad.append(Float64(words[i]))
        hess.append(Float64(words[size + i]))
        count.append(Int(words[2 * size + i]))
    return Histogram(grad^, hess^, count^, n_features, n_bins)


def _params() -> GpuSplitParams:
    """No hessian or count floor, so nothing but the gain decides. The
    admission rules have their own tests; this file is about the comparison
    that runs after them."""
    return GpuSplitParams(1.0, 0.0, 0.0, 0, CategoricalParams.default())


def _host(
    words: List[Int32],
    n_features: Int,
    n_bins: Int,
    missing_bins: List[Int] = [],
) raises -> SplitInfo:
    """`find_best_split` on the same numbers."""
    return find_best_split(
        _as_histogram(words, n_features, n_bins),
        lambda_reg=1.0,
        min_child_hess=0.0,
        min_data_in_leaf=0,
        lambda_l1=0.0,
        missing_bins=missing_bins,
    )


def _assert_device_matches_host(
    words: List[Int32],
    n_features: Int,
    n_bins: Int,
    missing_bins: List[Int] = [],
) raises:
    """Every discrete field of the decision, under both gain forms."""
    var want = _host(words, n_features, n_bins, missing_bins)
    var forms = _forms()
    for i in range(len(forms)):
        var got = reference_search(
            words,
            n_features,
            n_bins,
            1.0,
            1.0,
            _params(),
            [],
            [],
            missing_bins,
            [],
            gain_form=forms[i],
        )
        assert_equal(got.found, want.found)
        if not want.found:
            continue
        assert_equal(got.feature, want.feature)
        assert_equal(got.bin, want.bin)
        assert_equal(got.default_left, want.default_left)
        assert_equal(got.is_categorical, want.is_categorical)
        assert_almost_equal(got.gain, want.gain, atol=_TOL)


# --- The three shapes a tie can take -------------------------------------
#
# All of these use lambda_l2 = 1 and a node whose gradient sums to zero, so
# `parent_score` is 0 and each gain is a small exact fraction that Float32
# and Float64 both represent without rounding. A tie here is an *exact* tie
# in both precisions, which is the only way to test a tie policy rather than
# a rounding accident.
#
# Three four-bin patterns, laid out by `_lay_out`:
#
#   0 "early": grad -4 . . 4 / hess 1 . . 1. Bins 1 and 2 are empty, so the
#     candidates at bins 0, 1, and 2 are the *same* split and all score
#     16/2 + 16/2 = 16. The tie is reached first at bin 0.
#   1 "late": grad . -4 . 4 / hess . 1 . 1. Bin 0 is empty, so its candidate
#     scores 0 and is refused; bins 1 and 2 tie at the same 16. The identical
#     gain is reached first at bin 1.
#   2 "weak": grad -2 . . 2 / hess 1 . . 1, which scores 4/2 + 4/2 = 4 and
#     must lose to either of the others.


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
    """One four-bin pattern per feature, in the order `pick` gives."""
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


def test_a_tie_between_two_bins_of_one_feature_keeps_the_lower_bin() raises:
    # Bins 0, 1, and 2 all describe the split "(-4, 1, 10) | (4, 1, 10)" and
    # therefore all score exactly 16. Ascending scan order plus a strict `>`
    # means bin 0 keeps it, and the runner-up is the same 16.
    var words = _lay_out([0])
    var rec = reference_search(words, 1, 4, 1.0, 1.0, _params())
    assert_true(rec.found)
    assert_equal(rec.bin, 0)
    assert_almost_equal(rec.gain, 16.0, atol=_TOL)
    # The tie is real rather than an artifact of only one candidate being
    # admissible: a second candidate scored the identical gain, so the
    # margin the decision was made by is exactly zero.
    assert_almost_equal(rec.runner_gain, 16.0, atol=_TOL)
    assert_almost_equal(rec.margin(), 0.0, atol=_TOL)
    _assert_device_matches_host(words, 1, 4)


def test_a_tie_between_two_features_keeps_the_lower_slot() raises:
    # Feature 0 reaches gain 16 at bin 0, feature 1 reaches the *same* gain
    # 16 at bin 1. This is the tie a parallel reduction gets wrong when it
    # reduces gains without carrying the slot: the two winners are different
    # candidates, so picking the wrong one is a different tree, not a
    # different last bit.
    var early_first = _lay_out([0, 1])
    var rec = reference_search(early_first, 2, 4, 1.0, 1.0, _params())
    assert_true(rec.found)
    assert_equal(rec.feature, 0)
    assert_equal(rec.bin, 0)
    assert_almost_equal(rec.gain, 16.0, atol=_TOL)
    assert_almost_equal(rec.runner_gain, 16.0, atol=_TOL)
    _assert_device_matches_host(early_first, 2, 4)

    # Swapped, so that the lower slot is now the feature whose winner sits at
    # bin 1. The answer has to follow the slot and not the bin, which is the
    # half of the rule the first case cannot distinguish.
    var late_first = _lay_out([1, 0])
    var rec2 = reference_search(late_first, 2, 4, 1.0, 1.0, _params())
    assert_true(rec2.found)
    assert_equal(rec2.feature, 0)
    assert_equal(rec2.bin, 1)
    assert_almost_equal(rec2.gain, 16.0, atol=_TOL)
    _assert_device_matches_host(late_first, 2, 4)


def test_a_tie_between_the_two_missing_directions_keeps_default_left() raises:
    # Three bins, bin 2 reserved for missing rows, and the missing rows carry
    # no gradient and no hessian -- only a count. Sending them left and
    # sending them right therefore score *identically* at every threshold,
    # which is the exact tie the missing-direction rule exists for. The host
    # scores missing-left first, so the left default keeps it.
    #   grad  -4  4  0     ordinary total 0, missing 0
    #   hess   1  1  0
    #   count 10 10  5
    var words = _histogram_words(1, 3, [-4, 4, 0], [1, 1, 0], [10, 10, 5])
    var missing: List[Int] = [2]
    var rec = reference_search(
        words, 1, 3, 1.0, 1.0, _params(), [], [], missing
    )
    assert_true(rec.found)
    assert_equal(rec.bin, 0)
    assert_true(rec.default_left)
    assert_almost_equal(rec.gain, 16.0, atol=_TOL)
    # Both directions were scored and they scored alike.
    assert_almost_equal(rec.runner_gain, 16.0, atol=_TOL)
    _assert_device_matches_host(words, 1, 3, missing)


def test_a_tie_across_more_features_than_one_reduce_thread_owns() raises:
    # 96 feature slots, with the winning gain reached by slots 6, 70, and 71
    # and every other slot strictly below it. The device's threadgroup fold
    # gives thread `t` the slots `t, t + 64, ...`, so slots 6 and 70 land on
    # one thread (whose own ascending walk has to keep 6) and slot 71 lands
    # on another (which has to lose the cross-thread `block.min`). The host
    # answer this pins is simply "slot 6"; the device is held to it in
    # `tests/test_gpu_split_tie_parity.mojo`.
    var pick = List[Int](capacity=96)
    for f in range(96):
        pick.append(0 if (f == 6 or f == 70 or f == 71) else 2)
    var words = _lay_out(pick)
    var rec = reference_search(words, 96, 4, 1.0, 1.0, _params())
    assert_true(rec.found)
    assert_equal(rec.feature, 6)
    assert_equal(rec.bin, 0)
    assert_almost_equal(rec.gain, 16.0, atol=_TOL)
    # Slots 70 and 71 tied the winner, so the node's runner-up is the winning
    # gain itself and the margin is zero.
    assert_almost_equal(rec.runner_gain, 16.0, atol=_TOL)
    _assert_device_matches_host(words, 96, 4)


# --- What actually separates the two backends ----------------------------


def _sweep_words(
    seed: UInt64, n_features: Int, n_bins: Int, spread: Int, bias: Int
) raises -> List[Int32]:
    """A pseudo-random fixed-point histogram.

    `spread` sets how far the per-bin gradients vary, and `bias` shifts them
    all the same way. Together they set `parent_score / gain`, which is the
    parameter that decides how finely Float32 can separate two candidates:
    an unbiased node has gradients that cancel, a large positive `bias` is a
    nearly pure leaf whose parent score dwarfs every candidate's improvement
    on it. `gpu_split_search`'s own docstring names the second as the hard
    case, and it is the one a boosted ensemble spends its late rounds in."""
    var size = n_features * n_bins
    var g = List[Int](capacity=size)
    var h = List[Int](capacity=size)
    var c = List[Int](capacity=size)
    for i in range(size):
        var v = _splitmix64(seed * UInt64(1315423911) + UInt64(i) + 17)
        g.append(bias + Int(v % UInt64(2 * spread + 1)) - spread)
        h.append(1 + Int(_splitmix64(v + 101) % 8))
        c.append(1 + Int(_splitmix64(v + 809) % 20))
    return _histogram_words(n_features, n_bins, g, h, c)


def _run_sweep(
    n_cases: Int,
    n_features: Int,
    n_bins: Int,
    spread: Int,
    bias: Int,
    form: Int,
) raises -> List[Int]:
    """`[decided, separated, disagreements, near_tie_disagreements]` over
    `n_cases` pseudo-random histograms.

    "separated" is a decision whose runner-up is more than a thousandth of
    the winning gain behind it, which is four orders of magnitude outside
    Float32's resolution on these shapes. A disagreement there could not be
    a rounding flip and would have to be a rule difference.
    """
    var decided = 0
    var separated = 0
    var disagreements = 0
    var near_tie_disagreements = 0
    for k in range(n_cases):
        var words = _sweep_words(UInt64(k), n_features, n_bins, spread, bias)
        var got = reference_search(
            words,
            n_features,
            n_bins,
            1.0,
            1.0,
            _params(),
            [],
            [],
            [],
            [],
            gain_form=form,
        )
        var want = _host(words, n_features, n_bins)
        if not (got.found and want.found):
            # Both must agree that a split exists at all; a node with no
            # admissible candidate is not a decision to compare.
            assert_equal(got.found, want.found)
            continue
        decided += 1
        var loose = got.is_near_tie(1e-3)
        if not loose:
            separated += 1
        var agrees = (
            got.feature == want.feature
            and got.bin == want.bin
            and got.default_left == want.default_left
            and got.is_categorical == want.is_categorical
        )
        if not agrees:
            disagreements += 1
            if loose:
                near_tie_disagreements += 1
    return [decided, separated, disagreements, near_tie_disagreements]


def _assert_only_near_ties_disagree(stats: List[Int]) raises:
    # The sweep has to have decided something, or it proves nothing.
    assert_true(stats[0] > 150)
    assert_true(stats[1] > 100)
    # Every disagreement, if there is one, is a near tie.
    assert_equal(stats[2], stats[3])


def test_every_well_separated_decision_agrees_with_the_host() raises:
    """The discriminating evidence, and the reason this file does not end at
    the constructed ties.

    A tie-break rewrite would be worth nothing if the two backends also
    disagreed on decisions that are nowhere near a tie. They do not: over
    these histograms every disagreement is a near tie, and every decision
    whose margin is outside Float32's reach agrees exactly. So the tie
    *policy* is not what moves a split between the CPU and the GPU, and a
    prediction gap two orders of magnitude past the agreement limit is the
    Float32 near-tie resolution this module documents, whose remedy is
    `host_rescan_recommended` and not a different comparison.
    """
    var forms = _forms()
    for i in range(len(forms)):
        # An unbiased node: the gradients cancel, so the parent score is
        # small and Float32 separates the candidates easily.
        _assert_only_near_ties_disagree(
            _run_sweep(200, 6, 12, 40, 0, forms[i])
        )
        _assert_only_near_ties_disagree(
            _run_sweep(200, 6, 12, 4000, 0, forms[i])
        )
        # A nearly pure leaf: every gradient the same sign and far larger
        # than the variation between bins, so `parent_score / gain` runs into
        # the thousands. This is where the two backends are documented to be
        # able to part company, and where the flips actually appear.
        _assert_only_near_ties_disagree(
            _run_sweep(200, 6, 12, 60, 200000, forms[i])
        )


# --- What "the same gain" has to mean for a Float32 scan -----------------


def test_the_near_tie_width_is_absolute_and_not_only_relative() raises:
    """The host's tie tolerance is zero and the device's cannot be.

    `find_best_split` compares Float64 gains with a bare `>`, so on the host
    a tie is bitwise equality and nothing else is one. The device cannot
    borrow that definition, because two gains computed by different code
    paths over different histograms are "the same gain" long before their
    bits agree. `SPLIT_TIE_RELATIVE` is the width that exists for it, and a
    purely relative width is the wrong shape: this scan's resolution is set
    by `parent_score / gain`, not by the gain, so at a nearly pure leaf the
    absolute floor is a large multiple of `SPLIT_TIE_RELATIVE * gain` and
    the relative test alone reports "resolved" on a coin flip.

    Both arms stay reachable, and the resolution-aware one is a superset:
    it can only widen the width, never narrow it.
    """
    var rec = GpuSplitRecord()
    rec.found = True
    rec.is_categorical = False
    rec.feature = 0
    rec.bin = 0
    # A node whose parent score is ten thousand times its gain, which is the
    # regime `gpu_right_sum`'s measured table covers and where a boosted
    # ensemble spends its late rounds. `parent_score` is recovered from the
    # record as `-parent_value * total.grad`, so these two carry it.
    rec.total = ChildStats(1000.0, 100.0, 10000)
    rec.parent_value = -10.0
    rec.gain = 1.0
    # A margin of one part in a hundred thousand of the gain: far outside
    # `SPLIT_TIE_RELATIVE`, and far inside what a Float32 scan against a
    # parent score of 10000 can resolve.
    rec.runner_gain = 1.0 - 1e-5

    assert_almost_equal(rec.parent_score_bound(), 10000.0, atol=1e-6)
    # eps * max(parent, sqrt(parent * gain)) = eps * 10000, times the
    # rounding allowance, which is comfortably above the 1e-5 margin.
    assert_true(rec.resolution_floor() > rec.margin())
    assert_true(rec.is_near_tie())
    # The pre-floor policy calls the same node resolved, which is the
    # failure the floor exists to fix.
    assert_true(not rec.is_near_tie(resolution_aware=False))
    assert_true(host_rescan_recommended(rec))
    assert_true(not host_rescan_recommended(rec, resolution_aware=False))

    # A node with a real margin is not a near tie under either arm, so the
    # floor has not simply swallowed everything.
    var clear = rec.copy()
    clear.runner_gain = 0.5
    assert_true(not clear.is_near_tie())
    assert_true(not clear.is_near_tie(resolution_aware=False))

    # And with no runner-up at all there is nothing to be near, whatever the
    # parent score.
    var alone = rec.copy()
    alone.runner_gain = 0.0
    assert_true(not alone.is_near_tie())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
