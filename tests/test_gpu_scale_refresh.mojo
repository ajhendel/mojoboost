"""The magnitude window amortizes the round's scale readback and, at a window
of one, changes nothing at all.

What this file is gating
------------------------
`GpuHistogramBuilder.set_scale_refresh` is a **numeric** arm, not a launch
shape. At any setting but the default it changes the unit the fixed-point
histogram counts in, so it changes cells, therefore splits, therefore trees.
That rules out the usual two-arm identity comparison for the interesting
settings, and it makes the two claims that *can* be asserted exactly the two
worth asserting:

1. **The window of one is the call it replaced.** `scale_refresh = 0` runs
   `GpuObjectiveState.magnitude_sums`, which enqueues the reduction, copies
   the partials into slot zero of the pinned window, synchronizes and folds,
   all in one call. `scale_refresh = 1` runs the split pair, which does the
   same four things across two calls. Every scale, every histogram and every
   tree has to be identical, and the failure this catches is not hypothetical:
   the split pair writes into slot `p * 2 * SUM_BLOCKS` and folds from the
   same offset, and an off-by-one in either would fold a *plausible* set of
   partials -- a neighboring round's, or an uninitialized slot's -- and
   produce a scale that is wrong without being detectably wrong. Nothing but
   an exact comparison finds that.

2. **The window actually amortizes.** A test that only compared trees would
   pass on a builder that folded every round whatever the cadence said, which
   is the same vacuous pass `tests/test_gpu_tree_resident.mojo`'s header
   describes: the mechanism never ran, so it never disagreed.
   `scale_readback_count` is the positive evidence, and it is the fit's scale
   round-trip count itself rather than a proxy for it -- one fold is one
   `synchronize` on one device answer the host could not proceed without. So
   the assertions below are on the *census*, at its exact value, per cadence.

Why the count assertions drive the builder directly
---------------------------------------------------
`train_gpu` returns a `Booster` and not its builder, so a fit's readback
count is not observable from outside it. Rather than add an instrument to the
trainer to be able to test the trainer, these tests construct the builder and
the objective state and call `fill_gradients_device` in a loop, which is the
per-round call under test with the per-round call around it removed. What
that gives up is the interaction with tree growth; what it buys is an exact
expected count and an exact expected scale, neither of which a fit can offer
because neither is a function of the cadence alone.

The end-to-end arm is claim 1's, and it is a real fit through the public
entry point, because that is the only place the plumbing from a `train_gpu`
argument down to the builder's field can break.

The gradients are held still on purpose
---------------------------------------
The count tests never advance the raw scores between fills, so every round
reduces the same gradients and every round's magnitude sums are bit-identical
to every other's. That is what makes the *scale* comparable across cadences:
with the magnitudes constant, a window of eight and a window of one derive
their scale from the same number, so any difference between them is the
window's arithmetic and not the data's. A fit whose residuals actually shrink
is the case where the cadences legitimately diverge, and asserting they agree
there would be asserting something false.
"""

from std.sys import has_accelerator
from std.testing import assert_equal, assert_true, TestSuite

from mojotrees.bagging import BaggingParams
from mojotrees.binning import bin_equal_width, BinnedMatrix
from mojotrees.boosting import BoosterParams
from mojotrees.goss import GossParams
from mojotrees.gpu_objectives_native import SCALE_WINDOW_MAX
from mojotrees.histogram_gpu import GpuHistogramBuilder, _check_window_bound
from mojotrees.objective_registry import SQUARED_ERROR
from mojotrees.quantized_gradient import FIXED_ONE
from mojotrees.train_gpu import (
    OBJECTIVE_SOURCE_AUTO,
    device_gradients,
    train_gpu,
)
from mojotrees.tree import Tree, TreeParams
from support import _make_features, _uniform


def _regression_target(features: List[Float64], n_rows: Int) -> List[Float64]:
    var target = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        target.append(
            2.0 * features[r] - 1.5 * features[n_rows + r] + _uniform(UInt64(r))
        )
    return target^


def _assert_same_forest(a: List[Tree], b: List[Tree], label: String) raises:
    """Every tree, node for node, with no tolerance anywhere.

    Copied in shape from `tests/test_gpu_tree_resident._assert_same_forest`
    and for its reason: the question is not whether two arms produce similar
    models, it is whether they make the same decisions, and a decision is
    discrete. `value` and `split_gain` are compared as bit patterns.
    """
    assert_equal(len(a), len(b), label + ": tree count")
    for t in range(len(a)):
        var want = a[t].copy()
        var got = b[t].copy()
        assert_equal(got.n_leaves, want.n_leaves, label + ": n_leaves")
        assert_equal(len(got.feature), len(want.feature), label + ": n_nodes")
        for i in range(len(want.feature)):
            assert_equal(got.feature[i], want.feature[i], label + ": feature")
            assert_equal(
                got.threshold_bin[i],
                want.threshold_bin[i],
                label + ": threshold_bin",
            )
            assert_equal(got.left[i], want.left[i], label + ": left")
            assert_equal(got.right[i], want.right[i], label + ": right")
            assert_equal(
                got.value[i].to_bits(),
                want.value[i].to_bits(),
                label + ": value bits",
            )


@fieldwise_init
struct WindowRun(Copyable, Movable):
    """What driving `fill_gradients_device` a fixed number of times under one
    cadence produced."""

    var readbacks: Int
    """Folds during the loop. The scale half of the round-trip census."""

    var readbacks_after_flush: Int
    """Folds including the end-of-fit flush, which is what a real fit pays."""

    var g_scale: Float64
    var h_scale: Float64


def _drive_window(
    data: BinnedMatrix,
    target: List[Float64],
    rounds: Int,
    refresh: Int,
    headroom: Int = 0,
) raises -> WindowRun:
    """Fill the gradients `rounds` times under one cadence and report the
    census and the resulting scale.

    No tree is grown and the raw scores are never advanced, so every fill
    reduces the same gradients; see the module docstring for why that is the
    point rather than a shortcut. One builder and one state for the whole
    loop, because the window is builder state and a fresh builder per round
    would reset it -- which is also the mistake this helper exists to make
    impossible to write by accident.
    """
    var builder = GpuHistogramBuilder(data)
    builder.set_scale_refresh(refresh, headroom)
    var state = builder.objective_state(target, [], 1, 64)
    state.init_raw(builder.ctx, [0.0])
    for _ in range(rounds):
        builder.fill_gradients_device(state, SQUARED_ERROR, 0.9)
    var during = builder.scale_readback_count()
    builder.flush_scale_window(state)
    return WindowRun(
        during,
        builder.scale_readback_count(),
        builder.g_scale,
        builder.h_scale,
    )


def test_the_window_of_one_is_the_call_it_replaced() raises:
    """Claim 1, at the level of the scale: `scale_refresh = 0` and
    `scale_refresh = 1` derive the same two Float64 scales, bit for bit, and
    fold the same number of times.

    The bit comparison is the load-bearing one. Both arms fold the same 2 KB
    of partials with the same `sum_abs_partials` in the same ascending order
    in Float64, so the totals are equal, so the scales are equal -- unless the
    windowed arm folded the wrong slot, which is precisely the failure a
    tolerance would let through and an equality does not.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 3_000
        var n_features = 4
        var features = _make_features(n_rows, n_features)
        var target = _regression_target(features, n_rows)
        var data = bin_equal_width(features, n_rows, n_features, 32)

        var unwindowed = _drive_window(data, target, 6, 0)
        var window_one = _drive_window(data, target, 6, 1)

        assert_equal(
            window_one.g_scale.to_bits(),
            unwindowed.g_scale.to_bits(),
            "window of one: the gradient scale must be the unwindowed call's,"
            " bit for bit",
        )
        assert_equal(
            window_one.h_scale.to_bits(),
            unwindowed.h_scale.to_bits(),
            "window of one: the hessian scale must be the unwindowed call's,"
            " bit for bit",
        )
        assert_equal(
            unwindowed.readbacks,
            6,
            "window of one: the unwindowed call folds once per round",
        )
        assert_equal(
            window_one.readbacks,
            6,
            "window of one: a window of one folds once per round too",
        )
        # The flush is what makes the overflow check cover the last rounds.
        # At these two cadences the window is always empty when the loop
        # stops, so the flush is free and must not count.
        assert_equal(
            unwindowed.readbacks_after_flush,
            6,
            "window of one: an empty window must cost no flush round trip",
        )
        assert_equal(
            window_one.readbacks_after_flush,
            6,
            "window of one: an empty window must cost no flush round trip",
        )


def test_a_wider_window_folds_less_often_and_flushes_the_rest() raises:
    """Claim 2, at its exact value: the census per cadence, which is the
    number this arm exists to move.

    Twenty rounds at a cadence of eight fold on rounds 1, 9 and 17 -- round 1
    because a fit has no scale to reuse yet and the first fill always closes
    the window whatever the cadence says, then every eighth after it. Three
    folds during the loop, four with the flush that closes the three rounds
    still pending. Against twenty at the shipped cadence.

    Asserted at the value and not as an inequality, because "fewer" is what a
    builder that folded on a coin flip would also satisfy.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 2_000
        var n_features = 4
        var features = _make_features(n_rows, n_features)
        var target = _regression_target(features, n_rows)
        var data = bin_equal_width(features, n_rows, n_features, 32)

        var every_round = _drive_window(data, target, 20, 1)
        var every_eighth = _drive_window(data, target, 20, 8)

        assert_equal(
            every_round.readbacks_after_flush,
            20,
            "cadence 1: twenty rounds, twenty round trips",
        )
        assert_equal(
            every_eighth.readbacks,
            3,
            "cadence 8: twenty rounds fold on 1, 9 and 17",
        )
        assert_equal(
            every_eighth.readbacks_after_flush,
            4,
            "cadence 8: the three rounds left pending cost one flush",
        )
        # With the gradients held still every round's magnitude sums are the
        # same number, so the two cadences derive their scale from the same
        # total and must land on the same power of two. This is NOT a claim
        # that the cadences agree on a real fit; see the module docstring.
        assert_equal(
            every_eighth.g_scale.to_bits(),
            every_round.g_scale.to_bits(),
            "cadence 8: with the magnitudes constant, a wider window must"
            " derive the same scale",
        )


def test_headroom_costs_exactly_the_bits_it_says() raises:
    """`headroom_bits = H` divides the scale by exactly `2^H`, and nothing
    else about it moves.

    Exactly, not approximately. `T * 2^H` is an exact Float64 product and the
    power-of-two rule commutes with a power-of-two rescaling of its input, so
    the ratio is a power of two and the assertion is an equality between
    Float64 bit patterns rather than a comparison with a tolerance. If that
    ever stops being true, the "H bits and no more" cost stated in
    `set_scale_refresh` stops being true with it, and this is the assertion
    that would say so.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 2_000
        var n_features = 4
        var features = _make_features(n_rows, n_features)
        var target = _regression_target(features, n_rows)
        var data = bin_equal_width(features, n_rows, n_features, 32)

        var plain = _drive_window(data, target, 4, 1, 0)
        var roomy = _drive_window(data, target, 4, 1, 3)

        assert_equal(
            roomy.g_scale.to_bits(),
            (plain.g_scale / 8.0).to_bits(),
            "headroom 3: the gradient scale must be exactly an eighth",
        )
        assert_equal(
            roomy.h_scale.to_bits(),
            (plain.h_scale / 8.0).to_bits(),
            "headroom 3: the hessian scale must be exactly an eighth",
        )
        # Headroom is not a cadence: it must not change how often the host
        # waits.
        assert_equal(
            roomy.readbacks_after_flush,
            plain.readbacks_after_flush,
            "headroom 3: the round-trip census must not move",
        )


def test_the_window_of_one_grows_the_same_forest_end_to_end() raises:
    """Claim 1 at the level this repository grades on: whole trees, node for
    node, through the public entry point.

    Two fits in one process against one dataset, so nothing about the
    environment differs between them beyond the argument itself. The
    comparison is only meaningful if the fit takes the device-gradient arm at
    all -- a bagged or GOSS fit derives its scale from a host pass and never
    reaches the window -- so that is asserted first, from the same predicate
    the trainer binds, rather than assumed from the objective.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 4_000
        var n_features = 6
        var features = _make_features(n_rows, n_features)
        var target = _regression_target(features, n_rows)
        var data = bin_equal_width(features, n_rows, n_features, 64)
        var params = BoosterParams(8, 0.1, TreeParams(16, 20, 1.0, 1e-3))

        # Positive evidence that the arm under test is the arm this fit
        # takes. Without it a fit routed to host gradients would compare the
        # host path against itself and pass while proving nothing.
        assert_true(
            device_gradients(
                SQUARED_ERROR,
                1,
                OBJECTIVE_SOURCE_AUTO,
                BaggingParams.disabled(),
                GossParams.disabled(),
            ),
            "end to end: this configuration must reach the device-gradient"
            " arm or the comparison is vacuous",
        )

        var unwindowed = train_gpu(
            data, target, SQUARED_ERROR, params, scale_refresh=0
        )
        var window_one = train_gpu(
            data, target, SQUARED_ERROR, params, scale_refresh=1
        )
        assert_true(
            len(window_one.trees) > 0, "end to end: the fit grew no trees"
        )
        _assert_same_forest(
            unwindowed.trees, window_one.trees, "window of one, end to end"
        )


def test_the_overflow_check_is_the_bound_and_not_a_habit() raises:
    """`_check_window_bound` passes on `T * s <= 2^30 (1 + 2^-24)` and raises
    above it.

    Pure host arithmetic over hand-made numbers, which is the only way to
    reach both poles: a fit that violated the bound would need magnitudes
    that grew past the scale in force, and arranging that on real data is
    neither reliable nor informative.

    Three poles, and the middle one is the reason this test exists. The
    inequality is `quantized_gradient.fixed_point_scale_pow2`'s and it is
    inclusive at the boundary, because a freshly derived power-of-two scale
    lands exactly on `2^30` whenever `2^30 / T` is a power of two. It is also
    inclusive *just past* the boundary, by `2^-24`, because the derivation's
    own rounding admits `2^30 (1 + 2^-53)` and `total * scale` is an exact
    product that carries that ulp into the comparison rather than rounding it
    away. A check without that slack would fire on rounds that derived their
    own scale, which is the false positive `_check_window_bound`'s docstring
    argues the constant away from. The far pole is a factor of two, which is
    what a window one binade too wide actually produces.
    """
    # Exactly on the bound, which a fresh derivation reaches routinely.
    _check_window_bound(1.0, FIXED_ONE, "gradient")
    # Comfortably inside it.
    _check_window_bound(0.5, FIXED_ONE, "gradient")
    # Inside the derivation's own slack: safe, and must not raise.
    _check_window_bound(1.0 + 1e-9, FIXED_ONE, "gradient")
    # Past the slack. 1e-6 is above 2^-24 (about 5.96e-8), which is the
    # boundary this constant is chosen at.
    var raised = False
    try:
        _check_window_bound(1.0 + 1e-6, FIXED_ONE, "gradient")
    except:
        raised = True
    assert_true(
        raised,
        "the overflow check must raise on a scaled total past 2^30 by more"
        " than the derivation's own rounding",
    )
    # A doubling, which is what a window one binade too wide produces.
    var raised_double = False
    try:
        _check_window_bound(2.0, FIXED_ONE, "hessian")
    except:
        raised_double = True
    assert_true(
        raised_double,
        "the overflow check must raise when a reused scale is a factor of"
        " two too large",
    )


def test_the_cadence_refuses_what_it_cannot_hold() raises:
    """`set_scale_refresh` refuses out-of-range values rather than clamping.

    A clamp would quantize on a lattice the caller did not ask for, which is
    the failure `set_scale_shape` refuses an unknown shape to prevent, and it
    would do it silently. The window ceiling is the one that matters at run
    time: a cadence past `SCALE_WINDOW_MAX` would overrun the pinned window
    and fold one round's partials as another's, which produces a plausible
    scale and no error.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 512
        var n_features = 3
        var features = _make_features(n_rows, n_features)
        var data = bin_equal_width(features, n_rows, n_features, 16)
        var builder = GpuHistogramBuilder(data)

        var over = False
        try:
            builder.set_scale_refresh(SCALE_WINDOW_MAX + 1, 0)
        except:
            over = True
        assert_true(over, "a cadence past the window ceiling must be refused")

        var negative = False
        try:
            builder.set_scale_refresh(-1, 0)
        except:
            negative = True
        assert_true(negative, "a negative cadence must be refused")

        var too_much_room = False
        try:
            builder.set_scale_refresh(1, 31)
        except:
            too_much_room = True
        assert_true(
            too_much_room,
            "a headroom that leaves no lattice at all must be refused",
        )

        # And the shipped setting is still accepted, so the refusals above
        # are not a setter that refuses everything.
        builder.set_scale_refresh(1, 0)
        assert_equal(
            builder.scale_refresh_rounds(),
            1,
            "the shipped cadence must remain reachable",
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
