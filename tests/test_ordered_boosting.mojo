"""Ordered boosting: the ladder, the permutation, and the determinism claim.

# run_tests: cpu-safe

The determinism tests are the point of this file. `ordered_boosting.mojo`
claims that a permutation is a function of `(seed, permutation index, row
count, block size)` alone -- not of a running generator, not of how the sort
was scheduled, not of the worker count -- and the only way to hold that claim
is to fix it in a test that would fail the moment somebody replaces the
counter-keyed sort with a sequential Fisher-Yates, which is what CatBoost's own
`NCB::Shuffle` uses and what we deliberately do not.

The arithmetic tests are the other point. Every ladder number here was worked
out by hand from `fold.cpp::SelectMinBatchSize` and `BuildDynamicFold`'s loop
and is written out in full, so a change to either rule shows up as a number
that no longer matches rather than as a shape that still looks plausible.
"""

from std.testing import assert_equal, assert_false, assert_true, TestSuite

from mojotrees import (
    SQUARED_ERROR,
    BaggingParams,
    BoosterParams,
    OrderedBoostingParams,
    TreeParams,
    bin_equal_width,
    catboost_auto_is_ordered,
    check_ordered_hessian_declaration,
    check_ordered_honored,
    default_permutation_block_size,
    fold_bounds,
    fold_ids,
    fold_ladder,
    min_batch_size,
    n_rungs,
    ordered_permutation,
    ordered_plane_entries,
    ordered_varies_hessian,
    permutation_choice,
    plane_offsets,
    train,
    train_more,
)
from support import _make_features


comptime MULT = 2.0


def _regression_target(features: List[Float64], n_rows: Int) -> List[Float64]:
    var y = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        var x0 = features[0 * n_rows + r]
        var x1 = features[1 * n_rows + r]
        var x2 = features[2 * n_rows + r]
        y.append(4.0 * x0 - 3.0 * x1 + 2.0 * (x2 - 0.5) * (x2 - 0.5))
    return y^


# --------------------------------------------------------------------------
# The ladder, from CatBoost's arithmetic
# --------------------------------------------------------------------------


def test_min_batch_size_matches_catboost() raises:
    """`fold.cpp::SelectMinBatchSize`: `n > 500 ? min(100, n / 50) : 1`."""
    assert_equal(min_batch_size(1), 1)
    assert_equal(min_batch_size(100), 1)
    assert_equal(min_batch_size(500), 1)
    # 501 / 50 == 10 by integer division, and 10 < 100.
    assert_equal(min_batch_size(501), 10)
    # 5000 / 50 == 100, exactly the cap.
    assert_equal(min_batch_size(5000), 100)
    # 1e6 / 50 == 20000, capped at 100.
    assert_equal(min_batch_size(1000000), 100)


def test_default_permutation_block_matches_catboost() raises:
    """`defaults_helper.h`: `min(256, docCount / 1000 + 1)`."""
    assert_equal(default_permutation_block_size(100), 1)
    assert_equal(default_permutation_block_size(999), 1)
    assert_equal(default_permutation_block_size(1000), 2)
    assert_equal(default_permutation_block_size(255000), 256)
    assert_equal(default_permutation_block_size(1000000), 256)


def test_ladder_at_1000_rows() raises:
    """Worked by hand: `b_0 = min(100, 1000 / 50) = 20`, then doubling and
    clamping at 1000 gives 20, 40, 80, 160, 320, 640, 1000."""
    var ladder = fold_ladder(1000, MULT)
    assert_equal(len(ladder), 7)
    assert_equal(n_rungs(ladder), 6)
    assert_equal(ladder[0], 20)
    assert_equal(ladder[1], 40)
    assert_equal(ladder[2], 80)
    assert_equal(ladder[3], 160)
    assert_equal(ladder[4], 320)
    assert_equal(ladder[5], 640)
    assert_equal(ladder[6], 1000)


def test_ladder_always_has_a_rung() raises:
    """CatBoost's loop is `while (BodyTailArr.empty() || ...)`, so a dataset
    smaller than the seed prefix still gets one rung."""
    var one = fold_ladder(1, MULT)
    assert_equal(n_rungs(one), 1)
    assert_equal(one[0], 1)
    assert_equal(one[1], 1)
    var tiny = fold_ladder(3, MULT)
    # b_0 = 1 at three rows, then 2, then ceil(4) clamped to 3: the ladder is
    # [1, 2, 3], which is three entries and therefore TWO rungs.
    assert_equal(len(tiny), 3)
    assert_equal(n_rungs(tiny), 2)
    assert_equal(tiny[0], 1)
    assert_equal(tiny[1], 2)
    assert_equal(tiny[2], 3)


def test_ladder_is_strictly_increasing_and_ends_at_n() raises:
    var ladder = fold_ladder(7331, MULT)
    for f in range(len(ladder) - 1):
        assert_true(ladder[f] < ladder[f + 1])
    assert_equal(ladder[len(ladder) - 1], 7331)


def test_ladder_refuses_a_multiplier_at_or_below_one() raises:
    var raised = False
    try:
        _ = fold_ladder(100, 1.0)
    except:
        raised = True
    assert_true(raised)
    raised = False
    try:
        _ = fold_ladder(100, 0.5)
    except:
        raised = True
    assert_true(raised)


def test_plane_offsets_and_entry_count() raises:
    """The planes are nested prefixes, so the total is the sum of the TAIL
    lengths -- `ladder[1..K]` -- and not `K * n`."""
    var ladder = fold_ladder(1000, MULT)
    var off = plane_offsets(ladder)
    assert_equal(len(off), n_rungs(ladder) + 1)
    assert_equal(off[0], 0)
    # 40, then 40+80, then +160, +320, +640, +1000.
    assert_equal(off[1], 40)
    assert_equal(off[2], 120)
    assert_equal(off[3], 280)
    assert_equal(off[4], 600)
    assert_equal(off[5], 1240)
    assert_equal(off[6], 2240)
    assert_equal(ordered_plane_entries(1000, MULT, 1), 2240)
    assert_equal(ordered_plane_entries(1000, MULT, 3), 6720)


def test_plane_entries_stay_under_the_derived_bound() raises:
    """The bound the module states: strictly under `n * (2m - 1) / (m - 1)`,
    which is `3n` at `m = 2`. Six row counts, none of them on the ladder."""
    var counts = [301, 1000, 4097, 7331, 20011, 65536]
    for i in range(len(counts)):
        var n = counts[i]
        var total = ordered_plane_entries(n, MULT, 1)
        assert_true(total < 3 * n)
        # And it is genuinely bigger than one plane: the top rung alone is n.
        assert_true(total >= n)


def test_plane_entries_are_logarithmic_in_rung_count() raises:
    """`K` grows like `log_m(n / b_0)`; the entry count does not grow like
    `K * n`. At 1e6 with the defaults there are 14 rungs and 2 638 200
    entries, which is 2.64n and not 14n.

    **A point value, not the rule.** The ratio sawtooths toward `3n` just
    before each rung; `test_plane_entries_stay_under_the_derived_bound` is the
    test that carries the actual bound. Do not size a buffer from the 2.64
    here: at `n = 204 801` the count is `614 201`, or `2.9990n`."""
    var ladder = fold_ladder(1000000, MULT)
    assert_equal(n_rungs(ladder), 14)
    assert_equal(ordered_plane_entries(1000000, MULT, 1), 2638200)


def test_fold_bounds_are_the_device_boundary_plane() raises:
    """`[0, b_0, ..., b_K]`: `n_folds + 1` entries for `K + 1` disjoint
    segments, ascending, starting at 0 and ending at n."""
    var ladder = fold_ladder(1000, MULT)
    var bounds = fold_bounds(ladder)
    assert_equal(len(bounds), len(ladder) + 1)
    assert_equal(bounds[0], 0)
    assert_equal(bounds[1], 20)
    assert_equal(bounds[len(bounds) - 1], 1000)
    for i in range(len(bounds) - 1):
        assert_true(bounds[i] < bounds[i + 1])


def test_fold_ids_never_read_a_model_that_saw_the_row() raises:
    """The whole claim, as an invariant over every position: outside the seed
    prefix, position `q` reads a rung whose BODY ends at or before `q`, and
    whose PLANE is long enough to hold `q`."""
    var n = 1000
    var ladder = fold_ladder(n, MULT)
    var ids = fold_ids(ladder, n)
    assert_equal(len(ids), n)
    for q in range(n):
        var f = ids[q]
        assert_true(f >= 0 and f < n_rungs(ladder))
        # The plane is in range.
        assert_true(q < ladder[f + 1])
        if q < ladder[0]:
            # The seed prefix, which CatBoost leaks on purpose.
            assert_equal(f, 0)
        else:
            # The model that produced this row's score never saw it.
            assert_true(ladder[f] <= q)
            # And it is the tightest such model: the next rung's body has
            # already passed q.
            assert_true(f == n_rungs(ladder) - 1 or ladder[f + 1] > q)


def test_fold_ids_are_nondecreasing() raises:
    var n = 4097
    var ladder = fold_ladder(n, MULT)
    var ids = fold_ids(ladder, n)
    for q in range(n - 1):
        assert_true(ids[q] <= ids[q + 1])


# --------------------------------------------------------------------------
# The permutation and its determinism
# --------------------------------------------------------------------------


def _is_permutation(perm: List[Int], n: Int) -> Bool:
    if len(perm) != n:
        return False
    var seen = List[Bool](capacity=n)
    for _ in range(n):
        seen.append(False)
    for i in range(n):
        var r = perm[i]
        if r < 0 or r >= n or seen[r]:
            return False
        seen[r] = True
    return True


def test_permutation_is_a_permutation() raises:
    var blocks = [1, 7, 64, 256]
    for i in range(len(blocks)):
        var perm = ordered_permutation(0, 0, 500, blocks[i])
        assert_true(_is_permutation(perm, 500))


def test_permutation_is_reproducible() raises:
    """Two builds of the same (seed, index, n, block) agree everywhere. This
    is the claim CatBoost's sequential Fisher-Yates cannot make about a
    permutation drawn at a different point in the run."""
    var a = ordered_permutation(7, 2, 977, 16)
    var b = ordered_permutation(7, 2, 977, 16)
    assert_equal(len(a), len(b))
    for i in range(len(a)):
        assert_equal(a[i], b[i])


def test_permutation_moves_with_seed_and_with_index() raises:
    var base = ordered_permutation(0, 0, 977, 16)
    var other_seed = ordered_permutation(1, 0, 977, 16)
    var other_index = ordered_permutation(0, 1, 977, 16)
    var seed_differs = False
    var index_differs = False
    for i in range(len(base)):
        if base[i] != other_seed[i]:
            seed_differs = True
        if base[i] != other_index[i]:
            index_differs = True
    assert_true(seed_differs)
    assert_true(index_differs)


def test_permutation_preserves_order_inside_a_block() raises:
    """`NCB::Shuffle` permutes BLOCKS and keeps rows inside a block in their
    original order. That is what makes the permuted read sequential within a
    block instead of a scatter."""
    var block = 8
    var n = 100
    var perm = ordered_permutation(3, 0, n, block)
    var i = 0
    while i < n:
        # The row at position i starts a block; every row of that block
        # follows it consecutively and ascending, until the block ends.
        var start = perm[i]
        assert_equal(start % block, 0)
        var stop = start + block
        if stop > n:
            stop = n
        for r in range(start, stop):
            assert_equal(perm[i], r)
            i += 1


def test_one_block_is_the_identity() raises:
    """A block covering the dataset is CatBoost's `has_time` state:
    `IsPermutationNeeded` false sets `FoldPermutationBlockSize =
    learnSampleCount`, one block, no shuffle."""
    var n = 321
    var perm = ordered_permutation(11, 5, n, n)
    for i in range(n):
        assert_equal(perm[i], i)
    var wider = ordered_permutation(11, 5, n, n * 4)
    for i in range(n):
        assert_equal(wider[i], i)


def test_permutation_of_an_empty_dataset_is_empty() raises:
    var perm = ordered_permutation(0, 0, 0, 4)
    assert_equal(len(perm), 0)


def test_permutation_refuses_a_nonpositive_block() raises:
    var raised = False
    try:
        _ = ordered_permutation(0, 0, 10, 0)
    except:
        raised = True
    assert_true(raised)


def test_permutation_does_not_depend_on_how_the_sort_is_cut() raises:
    """The order is a total order on `(key, block index)`, so building the
    permutation for a row count that changes only the SHORT final block
    leaves every full block in the same relative order.

    This is the property a sequential Fisher-Yates does not have: there, the
    number of elements changes every draw. Here `n = 96` and `n = 100` share
    twelve full blocks of eight, plus a thirteenth block that exists only in
    the second case, so the twelve must appear in the same relative order.
    """
    var block = 8
    var short = ordered_permutation(5, 0, 96, block)
    var long = ordered_permutation(5, 0, 100, block)
    # Read the block order out of each permutation.
    var short_order = List[Int]()
    var i = 0
    while i < 96:
        short_order.append(short[i] // block)
        i += block
    var long_order = List[Int]()
    i = 0
    while i < 100:
        var b = long[i] // block
        if b < 12:
            long_order.append(b)
        var width = block
        if b == 12:
            width = 100 - 96
        i += width
    assert_equal(len(short_order), 12)
    assert_equal(len(long_order), 12)
    for k in range(12):
        assert_equal(short_order[k], long_order[k])


def test_permutation_choice_is_pure_and_in_range() raises:
    for round in range(50):
        var c = permutation_choice(9, round, 3)
        assert_true(c >= 0 and c < 3)
        assert_equal(c, permutation_choice(9, round, 3))
    # A single permutation is always index 0 and takes no draw.
    for round in range(10):
        assert_equal(permutation_choice(9, round, 1), 0)


def test_permutation_choice_actually_varies() raises:
    """A choice that never moves would make three permutations cost three
    times the memory for one permutation's worth of decorrelation."""
    var seen_other = False
    for round in range(64):
        if permutation_choice(4, round, 3) != 0:
            seen_other = True
    assert_true(seen_other)


def test_permutation_choice_refuses_a_nonpositive_count() raises:
    var raised = False
    try:
        _ = permutation_choice(0, 0, 0)
    except:
        raised = True
    assert_true(raised)


# --------------------------------------------------------------------------
# CatBoost's auto rule, which is not what it is usually said to be
# --------------------------------------------------------------------------


def test_catboost_cpu_default_is_plain_at_every_row_count() raises:
    """`boosting_options.cpp:16` constructs Plain and
    `catboost_options.cpp:802-806` installs Ordered only under
    `TaskType == ETaskType::GPU`. The often-repeated "Ordered below 50k rows
    on CPU" is not in the source."""
    assert_false(catboost_auto_is_ordered(100, 1000, False, False))
    assert_false(catboost_auto_is_ordered(10000, 1000, False, False))
    assert_false(catboost_auto_is_ordered(49999, 1000, False, False))
    assert_false(catboost_auto_is_ordered(1000000, 1000, False, False))


def test_catboost_gpu_rule_has_two_thresholds_and_one_exception() raises:
    # Under 50k rows and at least 500 iterations: Ordered.
    assert_true(catboost_auto_is_ordered(49999, 500, True, False))
    assert_true(catboost_auto_is_ordered(1000, 1000, True, False))
    # At or above 50k rows: Plain (`defaults_helper.h:37`).
    assert_false(catboost_auto_is_ordered(50000, 1000, True, False))
    # Fewer than 500 iterations: Plain, the clause usually omitted.
    assert_false(catboost_auto_is_ordered(1000, 499, True, False))
    # Multiclass: Plain even on the GPU, at any row count.
    assert_false(catboost_auto_is_ordered(1000, 1000, True, True))


# --------------------------------------------------------------------------
# Declarations and refusals
# --------------------------------------------------------------------------


def test_ordered_does_not_vary_the_hessian() raises:
    """Ordered boosting moves the POINT the derivatives are evaluated at; it
    installs no per-row weight. So a constant-hessian declaration stays
    admissible beside it, unlike beside MVS or the Bayesian bootstrap."""
    assert_false(ordered_varies_hessian(OrderedBoostingParams.disabled()))
    assert_false(ordered_varies_hessian(OrderedBoostingParams.enable()))
    # The guard is installed and does not fire today, on either arm.
    check_ordered_hessian_declaration(OrderedBoostingParams.enable(), True)
    check_ordered_hessian_declaration(OrderedBoostingParams.enable(), False)


def test_check_ordered_honored_refuses_only_when_enabled() raises:
    check_ordered_honored(
        OrderedBoostingParams.disabled(), String("a trainer")
    )
    var raised = False
    try:
        check_ordered_honored(
            OrderedBoostingParams.enable(), String("a trainer")
        )
    except:
        raised = True
    assert_true(raised)


def test_params_validate() raises:
    OrderedBoostingParams.enable().validate()
    OrderedBoostingParams.disabled().validate()
    var raised = False
    try:
        OrderedBoostingParams.enable(permutation_count=0).validate()
    except:
        raised = True
    assert_true(raised)
    raised = False
    try:
        OrderedBoostingParams.enable(fold_len_multiplier=1.0).validate()
    except:
        raised = True
    assert_true(raised)
    raised = False
    try:
        OrderedBoostingParams.enable(permutation_block_size=-1).validate()
    except:
        raised = True
    assert_true(raised)


def test_resolve_block_size() raises:
    var auto = OrderedBoostingParams.enable()
    assert_equal(auto.resolve_block_size(1000000), 256)
    assert_equal(auto.resolve_block_size(500), 1)
    var explicit = OrderedBoostingParams.enable(permutation_block_size=32)
    assert_equal(explicit.resolve_block_size(1000000), 32)
    # An explicit block wider than the dataset collapses to one block.
    assert_equal(explicit.resolve_block_size(10), 10)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


def test_disabled_bundle_leaves_every_fit_where_it_was() raises:
    """The default is off, and off must be byte for byte the fit this loop
    grew before the bundle existed. `BoosterParams` built without the
    argument and with the disabled bundle are the same fit."""
    var n_rows = 400
    var n_features = 4
    var features = _make_features(n_rows, n_features)
    var target = _regression_target(features, n_rows)
    var data = bin_equal_width(features, n_rows, n_features, 32)
    var implicit = BoosterParams(12, 0.1, TreeParams(8, 20, 1.0, 1e-3))
    var explicit = BoosterParams(
        12,
        0.1,
        TreeParams(8, 20, 1.0, 1e-3),
        ordered=OrderedBoostingParams.disabled(),
    )
    var a = train(data, target, SQUARED_ERROR, implicit)
    var b = train(data, target, SQUARED_ERROR, explicit)
    assert_equal(len(a.trees), len(b.trees))
    for r in range(n_rows):
        assert_equal(
            a.predict_raw_row(data, r).to_bits(),
            b.predict_raw_row(data, r).to_bits(),
        )


def test_ordered_training_runs_and_is_reproducible() raises:
    var n_rows = 400
    var n_features = 4
    var features = _make_features(n_rows, n_features)
    var target = _regression_target(features, n_rows)
    var data = bin_equal_width(features, n_rows, n_features, 32)
    var params = BoosterParams(
        10,
        0.1,
        TreeParams(8, 20, 1.0, 1e-3),
        ordered=OrderedBoostingParams.enable(),
    )
    var a = train(data, target, SQUARED_ERROR, params)
    var b = train(data, target, SQUARED_ERROR, params)
    assert_true(len(a.trees) > 0)
    assert_equal(len(a.trees), len(b.trees))
    for r in range(n_rows):
        assert_equal(
            a.predict_raw_row(data, r).to_bits(),
            b.predict_raw_row(data, r).to_bits(),
        )


def test_ordered_training_differs_from_plain() raises:
    """It is a different mechanism, so it must produce a different model. A
    fit that came out identical would mean the ladder never reached the split
    search."""
    var n_rows = 400
    var n_features = 4
    var features = _make_features(n_rows, n_features)
    var target = _regression_target(features, n_rows)
    var data = bin_equal_width(features, n_rows, n_features, 32)
    var plain = BoosterParams(10, 0.1, TreeParams(8, 20, 1.0, 1e-3))
    var ordered = BoosterParams(
        10,
        0.1,
        TreeParams(8, 20, 1.0, 1e-3),
        ordered=OrderedBoostingParams.enable(),
    )
    var a = train(data, target, SQUARED_ERROR, plain)
    var b = train(data, target, SQUARED_ERROR, ordered)
    var differs = False
    for r in range(n_rows):
        if a.predict_raw_row(data, r) != b.predict_raw_row(data, r):
            differs = True
    assert_true(differs)


def test_ordered_permutation_seed_moves_the_model() raises:
    """The permutation is the mechanism, so a different permutation must be a
    different fit. If this passed with an unchanged model, the ordered
    gradients would be reaching nothing."""
    var n_rows = 400
    var n_features = 4
    var features = _make_features(n_rows, n_features)
    var target = _regression_target(features, n_rows)
    var data = bin_equal_width(features, n_rows, n_features, 32)
    var one = BoosterParams(
        10,
        0.1,
        TreeParams(8, 20, 1.0, 1e-3),
        ordered=OrderedBoostingParams.enable(seed=1),
    )
    var two = BoosterParams(
        10,
        0.1,
        TreeParams(8, 20, 1.0, 1e-3),
        ordered=OrderedBoostingParams.enable(seed=2),
    )
    var a = train(data, target, SQUARED_ERROR, one)
    var b = train(data, target, SQUARED_ERROR, two)
    var differs = False
    for r in range(n_rows):
        if a.predict_raw_row(data, r) != b.predict_raw_row(data, r):
            differs = True
    assert_true(differs)


def test_three_permutations_train() raises:
    """CatBoost's own count (`max(1, permutation_count - 1) = 3`) is
    reachable and reproducible; it is not our default only because of what it
    costs in planes."""
    var n_rows = 300
    var n_features = 3
    var features = _make_features(n_rows, n_features)
    var target = _regression_target(features, n_rows)
    var data = bin_equal_width(features, n_rows, n_features, 32)
    var params = BoosterParams(
        8,
        0.1,
        TreeParams(8, 20, 1.0, 1e-3),
        ordered=OrderedBoostingParams.enable(permutation_count=3),
    )
    var a = train(data, target, SQUARED_ERROR, params)
    var b = train(data, target, SQUARED_ERROR, params)
    assert_true(len(a.trees) > 0)
    for r in range(n_rows):
        assert_equal(
            a.predict_raw_row(data, r).to_bits(),
            b.predict_raw_row(data, r).to_bits(),
        )


def test_ordered_refuses_bagging() raises:
    var n_rows = 200
    var n_features = 3
    var features = _make_features(n_rows, n_features)
    var target = _regression_target(features, n_rows)
    var data = bin_equal_width(features, n_rows, n_features, 32)
    var params = BoosterParams(
        4,
        0.1,
        TreeParams(8, 20, 1.0, 1e-3),
        ordered=OrderedBoostingParams.enable(),
    )
    var raised = False
    try:
        _ = train(
            data,
            target,
            SQUARED_ERROR,
            params,
            [],
            0.9,
            BaggingParams(0.5, 1, 3),
        )
    except:
        raised = True
    assert_true(raised)


def test_ordered_refuses_continued_training() raises:
    """The rung planes are fit state and a model does not carry them, so
    resuming would restart every rung from the ensemble's current score --
    exactly the leak the mechanism removes."""
    var n_rows = 200
    var n_features = 3
    var features = _make_features(n_rows, n_features)
    var target = _regression_target(features, n_rows)
    var data = bin_equal_width(features, n_rows, n_features, 32)
    var params = BoosterParams(
        4,
        0.1,
        TreeParams(8, 20, 1.0, 1e-3),
        ordered=OrderedBoostingParams.enable(),
    )
    var fitted = train(data, target, SQUARED_ERROR, params)
    var raised = False
    try:
        _ = train_more(fitted, data, target, params)
    except:
        raised = True
    assert_true(raised)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
