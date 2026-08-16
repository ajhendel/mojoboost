"""CatBoost's `random_strength`, and the two things it has to prove.

`random_strength` is a regularizer: seeded normal noise added to every
(feature, bin) candidate's gain before the split search compares them. Two
claims carry it, and every test here serves one of them.

1. **At the default it is not there.** Not "close to not there": a fixture
   grown with the parameter absent and one grown with it explicitly 0 -- and
   with a seed set, to prove the seed is inert too -- are the same tree bit
   for bit, at every worker count. Nothing in this file compares to a
   tolerance.

2. **Above 0 the draw is keyed, not streamed.** The key is
   (seed, tree index, node id, feature, bin) and nothing else, so a
   candidate's noise cannot depend on how many candidates were scored before
   it. That is the property `MOJOTREES_NUM_WORKERS` would otherwise break,
   because the feature scan is dispatched across features and the number of
   tasks moves with the worker count. It is proved three ways: identical
   trees at 1, 3 and 8 workers; identical answers when the feature is reached
   through a subsampled feature list instead of a full scan (so its position
   in the scan changed while its id did not); and the draw asserted directly
   against its key.

**The gate assertion.** A test that passes whether or not the noise was added
is worthless, so one fixture is built to have exactly one valid candidate:
one feature, two bins, and no missing bin, which leaves bin 0 as the only
threshold (the top threshold is not a split when no missing rows can fill the
right child). The winner is therefore fixed, and the noiseless run reports
that candidate's gain `G`. The noisy run on the same fixture must report
exactly `G + random_score_noise(stdev, seed, tree, node, 0, 0)`, compared
with `to_bits()`. That equality can only hold if the draw fired, at that key,
with that value, and it reuses `G` from the noiseless run rather than
recomputing the gain, so it cannot pass by reproducing an arithmetic mistake.
A second fixture then shows the noise actually changing which split is
chosen, which is the point of a regularizer.
"""

from std.math import sqrt
from std.os import setenv
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_not_equal,
    assert_raises,
    assert_true,
)

from mojotrees.binning import BinnedMatrix, bin_equal_width
from mojotrees.boosting import SQUARED_ERROR, fill_grad_hess
from mojotrees.categorical import CategoricalSpec
from mojotrees.histogram import Histogram
from mojotrees.params import parse_params
from mojotrees.split import SplitInfo, find_best_split
from mojotrees.tree import Tree, TreeParams, grow_tree
from mojotrees.tree_parameters_extra import (
    DEFAULT_RANDOM_STRENGTH_SEED,
    ExtraTreeParams,
    derivatives_stdev_from_zero,
    model_size_decrease,
    random_score_noise,
    random_score_scale_from_gradients,
    random_score_stream,
    standard_normal,
)

from support import _uniform


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _workers(n: String):
    _ = setenv("MOJOTREES_NUM_WORKERS", n)


def _auto():
    _ = setenv("MOJOTREES_NUM_WORKERS", "")


def _bits(v: Float64) -> UInt64:
    return UInt64(v.to_bits())


def _one_candidate_histogram() raises -> Histogram:
    """One feature, two bins, and therefore exactly one threshold.

    The scan breaks at the top threshold when no missing rows exist to fill
    the right child, so bin 0 is the only candidate and the winner cannot
    move. Every value is a small integer, exactly representable, so no
    summation order can change a bit.
    """
    var h = Histogram.zeroed(1, 2)
    h.set_grad_at(0, -4.0)
    h.set_hess_at(0, 2.0)
    h.set_count_at(0, 10)
    h.set_grad_at(1, 4.0)
    h.set_hess_at(1, 2.0)
    h.set_count_at(1, 10)
    return h^


def _wide_histogram(
    n_features: Int, n_bins: Int, seed: UInt64
) raises -> Histogram:
    """A histogram with many features whose best gains are close together, so
    a modest amount of noise can reorder them.

    Gradients and hessians are drawn from the shared deterministic uniform,
    which fixes this fixture for good: whatever this file asserts about which
    split wins, it asserts about the same numbers on every machine.
    """
    var h = Histogram.zeroed(n_features, n_bins)
    for f in range(n_features):
        for b in range(n_bins):
            var i = f * n_bins + b
            h.set_grad_at(i, _uniform(seed + UInt64(i)) * 2.0 - 1.0)
            h.set_hess_at(i, 1.0 + _uniform(seed + UInt64(50_000 + i)))
            var c = _uniform(seed + UInt64(90_000 + i)) * 8.0
            h.set_count_at(i, 8 + Int(c))
    return h^


def _noisy_extra(
    strength: Float64, scale: Float64, seed: Int = 7
) raises -> ExtraTreeParams:
    var extra = ExtraTreeParams()
    extra.random_strength = strength
    extra.random_score_scale = scale
    extra.random_strength_seed = seed
    return extra^


def _make_data(
    n_rows: Int, n_features: Int, n_bins: Int, seed: UInt64
) raises -> BinnedMatrix:
    var features = List[Float64](capacity=n_rows * n_features)
    for k in range(n_rows * n_features):
        features.append(_uniform(seed + UInt64(k)))
    return bin_equal_width(features, n_rows, n_features, n_bins)


def _grad_hess(
    n_rows: Int, seed: UInt64
) raises -> Tuple[List[Float64], List[Float64]]:
    var target = List[Float64](capacity=n_rows)
    var raw = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        target.append(_uniform(seed + UInt64(r)) * 4.0 - 2.0)
        raw.append(_uniform(seed + UInt64(7_000_000 + r)) - 0.5)
    var grad = List[Float64]()
    var hess = List[Float64]()
    var weights = List[Float64]()
    fill_grad_hess(raw, target, SQUARED_ERROR, weights, 0.7, grad, hess)
    return (grad^, hess^)


def _tree_bits(tree: Tree) -> List[UInt64]:
    """Everything a grown tree decided, as integers: the structure, the
    routing, and the leaf values and split gains bit for bit."""
    var bits = List[UInt64]()
    bits.append(UInt64(len(tree.feature)))
    for i in range(len(tree.feature)):
        bits.append(UInt64(tree.feature[i] + 1))
        bits.append(UInt64(tree.threshold_bin[i] + 1))
        bits.append(UInt64(tree.left[i] + 1))
        bits.append(UInt64(tree.right[i] + 1))
        bits.append(UInt64(1) if tree.default_left[i] else UInt64(0))
        bits.append(_bits(tree.value[i]))
        bits.append(_bits(tree.split_gain[i]))
    return bits^


def _assert_same_bits(a: List[UInt64], b: List[UInt64]) raises:
    assert_equal(len(a), len(b))
    for i in range(len(a)):
        assert_equal(a[i], b[i])


def _grown(extra: ExtraTreeParams) raises -> Tree:
    var n_rows = 1531
    var data = _make_data(n_rows, 6, 21, UInt64(31_337))
    var gh = _grad_hess(n_rows, UInt64(555))
    var params = TreeParams(12, 5, 1.0, 1e-3)
    params.extra = extra.copy()
    return grow_tree(data, gh[0], gh[1], params, tree_index=3)


# ---------------------------------------------------------------------------
# 1. The default is a no-op, and provably so.
# ---------------------------------------------------------------------------


def test_default_bundle_carries_no_strength() raises:
    var extra = ExtraTreeParams()
    assert_equal(_bits(extra.random_strength), _bits(0.0))
    assert_equal(_bits(extra.random_score_scale), _bits(0.0))
    assert_equal(extra.random_strength_seed, DEFAULT_RANDOM_STRENGTH_SEED)
    assert_equal(_bits(extra.random_score_stdev()), _bits(0.0))
    # The bundle's contract: the defaults leave every grower on the path it
    # took before this parameter existed.
    assert_false(extra.is_active())
    assert_false(extra.needs_node_identity())
    assert_false(extra.needs_grower_support())


def test_explicit_zero_is_the_same_tree_as_absent() raises:
    """The first thing this lane has to prove. A tree grown with the
    parameter absent and one grown with it explicitly 0 -- with a non-default
    seed and a non-zero scale set beside it, so the only reason nothing
    happens is the strength itself -- agree on every node, every routing
    decision, every leaf value and every split gain, bit for bit."""
    _workers("1")
    var absent = _tree_bits(_grown(ExtraTreeParams()))
    var explicit = _tree_bits(_grown(_noisy_extra(0.0, 3.5, 99)))
    _assert_same_bits(absent, explicit)
    assert_true(len(absent) > 8)
    _auto()


def test_zero_is_bit_identical_at_every_worker_count() raises:
    """`MOJOTREES_NUM_WORKERS` cannot reach the arithmetic, and the default
    path of this parameter is not the exception."""
    var counts = ["1", "3", "8"]
    var reference = List[UInt64]()
    for k in range(len(counts)):
        _workers(counts[k])
        var bits = _tree_bits(_grown(_noisy_extra(0.0, 3.5, 99)))
        if k == 0:
            reference = bits^
        else:
            _assert_same_bits(reference, bits)
    _auto()


def test_zero_leaves_the_fold_and_the_tie_break_alone() raises:
    """The fold is ascending scan order with a strict `>`, so exactly equal
    gains resolve to the lowest feature id. Eleven features carrying three
    distinct slices make the winning slice a four-way exact tie; the answer
    must still be the first feature carrying it, and must not move when the
    parameter is present at 0."""
    var h = Histogram.zeroed(11, 23)
    for f in range(11):
        var kind = f % 3
        for b in range(23):
            var left = b <= (23 * (kind + 1)) // 5
            var g = -1.0 - Float64(kind) if left else 1.0
            h.set_grad_at(f * 23 + b, g)
            h.set_hess_at(f * 23 + b, 1.0 + 0.25 * Float64(b % 3))
            h.set_count_at(f * 23 + b, 4 + (b % 5))

    var counts = ["1", "3", "8"]
    for k in range(len(counts)):
        _workers(counts[k])
        var plain = find_best_split(h, lambda_reg=1.0, min_child_hess=1e-3)
        var zeroed = find_best_split(
            h,
            lambda_reg=1.0,
            min_child_hess=1e-3,
            extra=_noisy_extra(0.0, 3.5, 99),
        )
        assert_true(plain.found)
        # The tie-break: the slice is `feature % 3`, so the lowest feature
        # carrying the winning slice is below 3.
        assert_true(plain.feature < 3)
        assert_equal(plain.feature, zeroed.feature)
        assert_equal(plain.bin, zeroed.bin)
        assert_equal(_bits(plain.gain), _bits(zeroed.gain))
        assert_equal(plain.default_left, zeroed.default_left)
    _auto()


# ---------------------------------------------------------------------------
# 2. Above zero: the noise fired, and it fired at its key.
# ---------------------------------------------------------------------------


def test_the_noise_is_added_to_the_only_candidate() raises:
    """The gate assertion, on a fixture with exactly one candidate.

    One feature, two bins, no missing bin: the top threshold is not a split
    (nothing can fill the right child), so bin 0 is the only candidate and
    the winner cannot move. The noiseless run supplies the gain; the noisy
    run must report that gain plus the draw for (seed, tree, node, feature,
    bin), to the bit. Nothing here recomputes the gain, so the equality is a
    statement about the draw alone.
    """
    _workers("1")
    var h = _one_candidate_histogram()
    var base = find_best_split(h, lambda_reg=1.0, min_child_hess=1e-3)
    assert_true(base.found)
    assert_equal(base.feature, 0)
    assert_equal(base.bin, 0)

    var extra = _noisy_extra(1.0, 0.25, 7)
    var noisy = find_best_split(
        h,
        lambda_reg=1.0,
        min_child_hess=1e-3,
        extra=extra,
        node=5,
        tree_index=2,
    )
    var draw = random_score_noise(
        extra.random_score_stdev(), 7, 2, 5, 0, 0
    )
    # The draw is not zero, so the equality below is not the trivial one.
    assert_not_equal(_bits(draw), _bits(0.0))
    assert_true(noisy.found)
    assert_equal(noisy.feature, 0)
    assert_equal(noisy.bin, 0)
    assert_equal(_bits(noisy.gain), _bits(base.gain + draw))
    _auto()


def test_the_key_is_the_only_thing_the_draw_reads() raises:
    """Every coordinate of the key changes the draw, and nothing else does.

    A stream-based draw would give the same first value for every key and
    then diverge by call order; this asserts the opposite shape.
    """
    var sd = 1.0
    var a = random_score_noise(sd, 7, 2, 5, 3, 4)
    # Repeating a key repeats the draw exactly, however many other draws
    # happened in between.
    _ = random_score_noise(sd, 7, 2, 5, 9, 9)
    _ = random_score_noise(sd, 7, 2, 5, 0, 0)
    assert_equal(_bits(a), _bits(random_score_noise(sd, 7, 2, 5, 3, 4)))

    assert_not_equal(_bits(a), _bits(random_score_noise(sd, 8, 2, 5, 3, 4)))
    assert_not_equal(_bits(a), _bits(random_score_noise(sd, 7, 3, 5, 3, 4)))
    assert_not_equal(_bits(a), _bits(random_score_noise(sd, 7, 2, 6, 3, 4)))
    assert_not_equal(_bits(a), _bits(random_score_noise(sd, 7, 2, 5, 4, 4)))
    assert_not_equal(_bits(a), _bits(random_score_noise(sd, 7, 2, 5, 3, 5)))

    # Swapping the feature and the bin is a different candidate, so it must
    # be a different draw: the key is a tuple, not a sum.
    assert_not_equal(
        _bits(random_score_noise(sd, 7, 2, 5, 3, 4)),
        _bits(random_score_noise(sd, 7, 2, 5, 4, 3)),
    )
    # A negative seed is accepted rather than reinterpreted.
    assert_not_equal(
        _bits(a), _bits(random_score_noise(sd, -7, 2, 5, 3, 4))
    )

    # Scaling is linear in the standard deviation, and zero means untouched.
    assert_equal(_bits(random_score_noise(0.0, 7, 2, 5, 3, 4)), _bits(0.0))
    assert_equal(
        _bits(random_score_noise(2.0, 7, 2, 5, 3, 4)),
        _bits(2.0 * standard_normal(random_score_stream(7, 2, 5, 3, 4))),
    )


def test_noise_changes_which_split_is_chosen() raises:
    """A regularizer that never changes a decision is not one.

    The fixture has thirteen features whose best gains sit close together,
    and the noise is scaled to their size, so the winner moves. The exact
    winner is fixed by the seed, which is what makes this an assertion rather
    than a coin toss: it will report the same two splits forever.
    """
    _workers("1")
    var h = _wide_histogram(13, 17, UInt64(4_242))
    var base = find_best_split(h, lambda_reg=1.0, min_child_hess=1e-3)
    assert_true(base.found)

    var noisy = find_best_split(
        h,
        lambda_reg=1.0,
        min_child_hess=1e-3,
        extra=_noisy_extra(1.0, 0.5, 7),
        node=1,
        tree_index=0,
    )
    assert_true(noisy.found)
    var moved = (noisy.feature != base.feature) or (noisy.bin != base.bin)
    assert_true(moved)
    _auto()


def test_noise_is_bit_identical_at_every_worker_count() raises:
    """The property this lane will not trade. The feature scan is dispatched
    across features and the task count moves with the worker count, so a
    draw taken from a stream would answer differently at 1, 3 and 8. This
    grows a whole tree at each, with the noise on, and compares every node.
    """
    var counts = ["1", "3", "8"]
    var reference = List[UInt64]()
    for k in range(len(counts)):
        _workers(counts[k])
        var bits = _tree_bits(_grown(_noisy_extra(1.0, 4.0, 7)))
        if k == 0:
            reference = bits^
        else:
            _assert_same_bits(reference, bits)
    # And the noise reached the tree: the same fixture without it is a
    # different tree. Without this the loop above would pass on a parameter
    # that did nothing.
    _workers("1")
    var quiet = _tree_bits(_grown(ExtraTreeParams()))
    var loud = _tree_bits(_grown(_noisy_extra(1.0, 4.0, 7)))
    var same = len(quiet) == len(loud)
    if same:
        for i in range(len(quiet)):
            if quiet[i] != loud[i]:
                same = False
    assert_false(same)
    _auto()


def test_noise_follows_the_feature_id_not_its_scan_position() raises:
    """The other half of "not a stream". A feature reached through a
    subsampled feature list sits at a different position in the scan and on a
    different task, while its id is unchanged. Its draw must be unchanged
    too, which shows up as the same winning candidate and the same gain when
    the search is restricted to the features the full scan would have chosen
    between.
    """
    _workers("1")
    var h = _wide_histogram(13, 17, UInt64(4_242))
    var extra = _noisy_extra(1.0, 0.5, 7)
    var full = find_best_split(
        h,
        lambda_reg=1.0,
        min_child_hess=1e-3,
        extra=extra,
        node=1,
        tree_index=0,
    )
    assert_true(full.found)

    # The winner, scanned as the only feature in the list: position 0 of a
    # one-feature scan instead of position `full.feature` of a thirteen
    # feature one.
    var alone: List[Int] = [full.feature]
    var solo = find_best_split(
        h,
        lambda_reg=1.0,
        min_child_hess=1e-3,
        features=alone,
        extra=extra,
        node=1,
        tree_index=0,
    )
    assert_true(solo.found)
    assert_equal(solo.feature, full.feature)
    assert_equal(solo.bin, full.bin)
    assert_equal(_bits(solo.gain), _bits(full.gain))

    # And at a worker count that splits the full scan across tasks, the
    # thirteen-feature answer is still the one-feature answer.
    _workers("8")
    var wide = find_best_split(
        h,
        lambda_reg=1.0,
        min_child_hess=1e-3,
        extra=extra,
        node=1,
        tree_index=0,
    )
    assert_equal(wide.feature, full.feature)
    assert_equal(wide.bin, full.bin)
    assert_equal(_bits(wide.gain), _bits(full.gain))
    _auto()


# ---------------------------------------------------------------------------
# 3. CatBoost's scale, and the refusals around it.
# ---------------------------------------------------------------------------


def test_derivatives_stdev_is_the_rms_about_zero() raises:
    """CatBoost's `CalcDerivativesStDevFromZeroPlainBoosting`: no mean is
    subtracted. Chosen so the answer is exact in binary: 9 + 16 + 144 = 169,
    over 4 rows is 42.25, whose square root is 6.5."""
    var g: List[Float64] = [3.0, -4.0, 12.0, 0.0]
    assert_equal(_bits(derivatives_stdev_from_zero(g, 4)), _bits(6.5))
    # A mean-subtracting standard deviation of the same vector is a
    # different number, which is the correction this name hides.
    assert_not_equal(_bits(derivatives_stdev_from_zero(g, 4)), _bits(0.0))
    # Multiclass: the squares are summed over every output dimension and
    # divided by the rows of one. Two dimensions of two rows, same squares.
    assert_equal(_bits(derivatives_stdev_from_zero(g, 2)), _bits(sqrt(84.5)))

    with assert_raises():
        _ = derivatives_stdev_from_zero(g, 0)
    with assert_raises():
        _ = derivatives_stdev_from_zero(g, 3)


def test_model_size_decrease_fades_as_the_model_grows() raises:
    """`modelLeft / (1 + modelLeft)` with `modelLeft = exp(log(n) - L)`. It
    is in (0, 1), it starts effectively at 1, and it decreases in the model
    length. Compared by exact inequality, never to a tolerance."""
    var at_zero = model_size_decrease(1000, 0.0)
    assert_true(at_zero < 1.0)
    assert_true(at_zero > 0.99)
    var later = model_size_decrease(1000, 5.0)
    var much_later = model_size_decrease(1000, 20.0)
    assert_true(later < at_zero)
    assert_true(much_later < later)
    assert_true(much_later > 0.0)
    # More rows means a longer-lived noise term, which is the shape of
    # `log(n) - L`.
    assert_true(model_size_decrease(100_000, 5.0) > later)

    with assert_raises():
        _ = model_size_decrease(0, 1.0)


def test_scale_combines_the_two_catboost_factors() raises:
    var g: List[Float64] = [3.0, -4.0, 12.0, 0.0]
    var expect = derivatives_stdev_from_zero(g, 4) * model_size_decrease(
        4, 0.3
    )
    assert_equal(
        _bits(random_score_scale_from_gradients(g, 4, 0.3)), _bits(expect)
    )


def test_a_strength_without_a_scale_is_refused() raises:
    """The scale is the ensemble's, not the node's, and no trainer computes
    it in this build. A strength set without it would train an unregularized
    model that reported success, so it is refused where it was set and again
    where it would have been used."""
    var extra = ExtraTreeParams()
    extra.random_strength = 1.0
    with assert_raises():
        extra.check_scalars(20)

    var h = _one_candidate_histogram()
    with assert_raises():
        _ = find_best_split(
            h, lambda_reg=1.0, min_child_hess=1e-3, extra=extra
        )

    # Negative values are rejected rather than clamped, as everywhere else on
    # this bundle.
    var negative = ExtraTreeParams()
    negative.random_strength = -1.0
    with assert_raises():
        negative.check_scalars(20)

    # And with the scale supplied, the same bundle validates.
    var ok = _noisy_extra(1.0, 0.25, 7)
    ok.check_scalars(20)


def test_a_categorical_matrix_is_refused_rather_than_half_noised() raises:
    """A categorical feature's candidates are category sets searched inside
    `categorical.mojo`, and only that search's winner reaches the split
    scan. Noising the winner would noise one candidate per categorical
    feature while every numerical feature had all of its noised, which is a
    different regularizer under the same name."""
    var h = _wide_histogram(3, 9, UInt64(11))
    # Feature 1 is categorical with five categories; the other two are
    # numerical, which is the mixed matrix the refusal is about.
    var flags: List[Bool] = [False, True, False]
    var codes: List[Int] = [0, 1, 2, 3, 4]
    var offsets: List[Int] = [0, 0, 5, 5]
    var cats = CategoricalSpec(flags^, codes^, offsets^)
    assert_true(cats.any_categorical())
    with assert_raises():
        _ = find_best_split(
            h,
            lambda_reg=1.0,
            min_child_hess=1e-3,
            cats=cats,
            extra=_noisy_extra(1.0, 0.5, 7),
        )


def test_the_parameter_string_carries_the_name() raises:
    """`random_strength` is the one name on the parameter surface that is
    CatBoost's rather than LightGBM's. Zero parses and is inert. A positive
    value now PARSES on the CPU, and this test used to assert the opposite.

    The refusal was retired when the dense CPU round loops began computing
    the per-tree scale: `params.mojo` passes
    `scale_computed_per_tree=(config.device == CPU_DEVICE)`, and
    `check_random_strength` raises only when that is False. The assertion
    below was left behind asserting a refusal that no longer exists, and it
    failed for some hours before anyone ran this file.

    That is worth stating rather than quietly correcting. **A test is a
    written claim about behavior, and retiring the behavior does not retire
    the claim.** Nothing in this repository checks whether an assertion still
    describes the code, which is the same shape as a parity contract citing a
    divergence that has been fixed.

    A negative value is still refused, because that is a range error rather
    than a reachability one, and it is the assertion that proves this test is
    still discriminating rather than merely passing."""
    var zero = parse_params("objective=regression random_strength=0")
    assert_equal(_bits(zero.booster.tree.extra.random_strength), _bits(0.0))
    assert_false(zero.booster.tree.extra.is_active())

    var positive = parse_params("objective=regression random_strength=1.5")
    assert_equal(
        _bits(positive.booster.tree.extra.random_strength), _bits(1.5)
    )
    assert_true(positive.booster.tree.extra.is_active())

    with assert_raises():
        _ = parse_params("objective=regression random_strength=-1")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
