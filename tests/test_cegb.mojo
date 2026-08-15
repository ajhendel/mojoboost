"""Cost-effective gradient boosting: the arithmetic, the ledger, and the
combinations that are refused (src/mojotrees/cegb.mojo).

Every expectation is analytical. The rule under test, in numbers small enough
to check by hand:

    delta(f, N) = tradeoff * ( penalty_split * |R(N)|
                             + coupled[f] * [f never split on]
                             + lazy[f]    * unread(f, R(N)) )

    gain_cegb(f, N) = contri[f] * gain_raw(f, N) - delta(f, N)

The two flags in that formula live for the whole ensemble, not for one tree,
which is what `test_coupled_cost_is_charged_once_per_ensemble` pins: a ledger
reset per tree would be a different, much harsher regularizer.
"""

from std.testing import assert_equal, assert_false, assert_raises, assert_true
from std.testing import TestSuite

from mojotrees.cegb import (
    CEGB_DEFAULT_TRADEOFF,
    CegbConfig,
    CegbLedger,
    CegbNodeCosts,
    cegb_adjusted_gain,
    cegb_commit_split,
    cegb_coupled_cost,
    cegb_delta_gain,
    cegb_lazy_cost,
    cegb_split_cost,
    cegb_stale_cached_gain,
    check_cegb_continued_training,
    check_cegb_device_split_search,
    check_cegb_distributed,
    check_cegb_grower_support,
    prepare_cegb_node,
)


def close(a: Float64, b: Float64) -> Bool:
    return abs(a - b) < 1e-12


def _rows(n: Int) -> List[Int]:
    """`[0, 1, ..., n)`, the row ids of a node holding the first `n` rows."""
    var out = List[Int](capacity=n)
    for r in range(n):
        out.append(r)
    return out^


# ---------------------------------------------------------------------------
# The parameters
# ---------------------------------------------------------------------------


def test_defaults_are_lightgbms_and_charge_nothing() raises:
    var c = CegbConfig()
    assert_true(close(c.tradeoff, CEGB_DEFAULT_TRADEOFF))
    assert_true(close(c.tradeoff, 1.0))
    assert_true(close(c.penalty_split, 0.0))
    assert_false(c.is_active())
    assert_false(c.needs_ledger())
    assert_equal(c.n_ledger_features(), 0)
    # The tradeoff alone, at its default, changes nothing: it multiplies
    # costs that are all zero.
    c.check(4)


def test_a_zero_tradeoff_switches_every_term_off() raises:
    var coupled: List[Float64] = [5.0, 5.0]
    var lazy: List[Float64] = [7.0, 7.0]
    var c = CegbConfig.of(0.0, 3.0, coupled, lazy)
    assert_false(c.is_active())
    assert_false(c.split_cost_active())
    assert_false(c.coupled_active())
    assert_false(c.lazy_active())
    assert_true(close(cegb_split_cost(c, 100), 0.0))


def test_an_all_zero_vector_charges_nothing_however_long() raises:
    var zeros: List[Float64] = [0.0, 0.0, 0.0]
    var c = CegbConfig.of(1.0, 0.0, zeros)
    # Length alone is not the test, which is why `coupled_active` reads the
    # values: a caller passing an explicit zero vector must not pay for a
    # ledger it will never consult.
    assert_false(c.coupled_active())
    assert_false(c.is_active())
    assert_false(c.needs_ledger())


def test_costs_must_be_finite_and_nonnegative() raises:
    # An intentional difference from LightGBM, which does not check: a
    # negative cost is a bonus for reading data and inverts the mechanism.
    with assert_raises():
        CegbConfig.of(-1.0, 0.0).check(2)
    with assert_raises():
        CegbConfig.of(1.0, -1.0).check(2)
    var negative: List[Float64] = [1.0, -2.0]
    with assert_raises():
        CegbConfig.of(1.0, 0.0, negative).check(2)
    var lazy_negative: List[Float64] = [-3.0, 1.0]
    with assert_raises():
        CegbConfig.of(1.0, 0.0, [], lazy_negative).check(2)
    # A vector must cover the dataset exactly, or not at all.
    var two: List[Float64] = [1.0, 1.0]
    with assert_raises():
        CegbConfig.of(1.0, 0.0, two).check(3)
    CegbConfig.of(1.0, 0.0, two).check(2)


# ---------------------------------------------------------------------------
# The three terms
# ---------------------------------------------------------------------------


def test_the_split_cost_is_per_row_of_the_node() raises:
    var c = CegbConfig.of(2.0, 0.5)
    # 2 * 0.5 * 4 = 4, and the charge scales with the row count because
    # every row reaching the node is compared against the threshold.
    assert_true(close(cegb_split_cost(c, 4), 4.0))
    assert_true(close(cegb_split_cost(c, 8), 8.0))
    # An empty node costs nothing.
    assert_true(close(cegb_split_cost(c, 0), 0.0))
    # It needs no ledger at all, which is why it is the one term every
    # grower charges.
    assert_false(c.needs_ledger())
    var inert = CegbLedger.none()
    assert_true(close(cegb_delta_gain(c, inert, 0, 4), 4.0))


def test_the_coupled_cost_is_charged_while_the_feature_is_unused() raises:
    var coupled: List[Float64] = [0.0, 3.0]
    var c = CegbConfig.of(2.0, 0.0, coupled)
    var ledger = CegbLedger.create(c, 2, 8)
    # Feature 1, never split on: 2 * 3 = 6. Feature 0 carries no cost.
    assert_true(close(cegb_coupled_cost(c, ledger, 1), 6.0))
    assert_true(close(cegb_coupled_cost(c, ledger, 0), 0.0))
    _ = cegb_commit_split(ledger, c, 1)
    # Once the ensemble computes it, a second split on it is free.
    assert_true(close(cegb_coupled_cost(c, ledger, 1), 0.0))
    # And it never scaled with the node's rows, unlike the split cost.
    assert_true(close(cegb_delta_gain(c, ledger, 1, 1000), 0.0))


def test_the_lazy_cost_is_per_unread_row() raises:
    var lazy: List[Float64] = [0.0, 0.5]
    var c = CegbConfig.of(2.0, 0.0, [], lazy)
    # 2 * 0.5 * 6 unread rows = 6.
    assert_true(close(cegb_lazy_cost(c, 1, 6), 6.0))
    assert_true(close(cegb_lazy_cost(c, 1, 0), 0.0))
    # Feature 0 has no lazy cost, so its unread rows are free.
    assert_true(close(cegb_lazy_cost(c, 0, 6), 0.0))
    with assert_raises():
        _ = cegb_lazy_cost(c, 1, -1)


def test_the_three_terms_sum_in_a_fixed_order() raises:
    var coupled: List[Float64] = [0.0, 3.0]
    var lazy: List[Float64] = [0.0, 0.5]
    var c = CegbConfig.of(2.0, 0.25, coupled, lazy)
    var ledger = CegbLedger.create(c, 2, 8)
    # split 2 * 0.25 * 8 = 4, coupled 2 * 3 = 6, lazy 2 * 0.5 * 8 = 8.
    assert_true(close(cegb_delta_gain(c, ledger, 1, 8, 8), 18.0))
    # And a gain is that much lower. A negative result is not an error: the
    # caller's `min_gain_to_split` floor rejects it exactly as it rejects a
    # candidate whose raw gain was too small.
    assert_true(close(cegb_adjusted_gain(20.0, c, ledger, 1, 8, 8), 2.0))
    assert_true(close(cegb_adjusted_gain(1.0, c, ledger, 1, 8, 8), -17.0))


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------


def test_an_inert_ledger_answers_already_paid() raises:
    # The answer that charges nothing, which is what a grower with no CEGB
    # configuration must get without a null check at every call site.
    var inert = CegbLedger.none()
    assert_false(inert.is_tracking())
    assert_true(inert.feature_is_used(0))
    assert_true(inert.row_has_read(0, 0))
    assert_equal(inert.count_unread(0, _rows(5)), 0)
    assert_equal(inert.bytes_allocated(), 0)


def test_the_ledger_allocates_only_what_is_charged() raises:
    var coupled: List[Float64] = [1.0, 1.0]
    var only_coupled = CegbLedger.create(
        CegbConfig.of(1.0, 0.0, coupled), 2, 640
    )
    assert_true(only_coupled.tracks_features)
    assert_false(only_coupled.tracks_rows)
    # One Bool per feature and no row bitset at all.
    assert_equal(only_coupled.bytes_allocated(), 2)

    var lazy: List[Float64] = [1.0, 1.0]
    var with_rows = CegbLedger.create(
        CegbConfig.of(1.0, 0.0, [], lazy), 2, 640
    )
    assert_true(with_rows.tracks_rows)
    # 640 rows is ten 64-bit words per feature, two features, eight bytes a
    # word, plus the feature flags.
    assert_equal(with_rows.words_per_feature, 10)
    assert_equal(with_rows.bytes_allocated(), 2 + 8 * 20)

    # And nothing at all for a configuration that needs no ledger.
    var none = CegbLedger.create(CegbConfig.of(1.0, 5.0), 2, 640)
    assert_false(none.is_tracking())


def test_coupled_cost_is_charged_once_per_ensemble() raises:
    # CEGB's premise: the model pays for a feature once, so a later tree
    # reusing it gets it free. A ledger reset per tree would charge this
    # three times.
    var coupled: List[Float64] = [4.0, 4.0]
    var c = CegbConfig.of(1.0, 0.0, coupled)
    var ledger = CegbLedger.create(c, 2, 4)
    var charged = 0.0
    for _ in range(3):
        charged += cegb_coupled_cost(c, ledger, 0)
        _ = cegb_commit_split(ledger, c, 0)
    assert_true(close(charged, 4.0))
    assert_equal(ledger.n_features_used(), 1)
    # A second feature pays its own first use, once.
    assert_true(close(cegb_coupled_cost(c, ledger, 1), 4.0))


def test_only_the_first_commit_reports_a_new_feature() raises:
    var coupled: List[Float64] = [4.0, 4.0]
    var c = CegbConfig.of(1.0, 0.0, coupled)
    var ledger = CegbLedger.create(c, 2, 4)
    var first = cegb_commit_split(ledger, c, 0)
    assert_true(first.feature_newly_used)
    assert_true(close(first.coupled_refund, 4.0))
    var second = cegb_commit_split(ledger, c, 0)
    assert_false(second.feature_newly_used)
    assert_true(close(second.coupled_refund, 0.0))


def test_the_row_ledger_counts_global_row_ids() raises:
    # Global ids are what makes the count stable across bagging rounds: a
    # row that sat out a round keeps whatever read state it had.
    var lazy: List[Float64] = [1.0]
    var c = CegbConfig.of(1.0, 0.0, [], lazy)
    var ledger = CegbLedger.create(c, 1, 10)
    var bag: List[Int] = [1, 3, 5, 7]
    assert_equal(ledger.count_unread(0, bag), 4)
    assert_equal(ledger.mark_rows_read(0, bag), 4)
    assert_equal(ledger.count_unread(0, bag), 0)
    # A later round's bag overlaps the first: only the rows this feature has
    # never been read for are still charged.
    var next_bag: List[Int] = [0, 1, 2, 3]
    assert_equal(ledger.count_unread(0, next_bag), 2)
    # Marking is idempotent, so re-committing the same node charges nothing
    # new.
    assert_equal(ledger.mark_rows_read(0, bag), 0)
    with assert_raises():
        _ = ledger.row_has_read(0, 10)
    with assert_raises():
        _ = ledger.row_has_read(1, 0)


def test_committing_a_lazy_split_needs_the_nodes_rows() raises:
    var lazy: List[Float64] = [1.0]
    var c = CegbConfig.of(1.0, 0.0, [], lazy)
    var ledger = CegbLedger.create(c, 1, 4)
    # An empty list would leave every row unmarked and recharge them forever.
    with assert_raises():
        _ = cegb_commit_split(ledger, c, 0)


def test_reset_forgets_the_ensemble_not_the_allocation() raises:
    var coupled: List[Float64] = [2.0]
    var c = CegbConfig.of(1.0, 0.0, coupled)
    var ledger = CegbLedger.create(c, 1, 4)
    _ = cegb_commit_split(ledger, c, 0)
    assert_equal(ledger.n_features_used(), 1)
    ledger.reset()
    assert_equal(ledger.n_features_used(), 0)
    assert_true(ledger.tracks_features)


# ---------------------------------------------------------------------------
# One node's prepared costs
# ---------------------------------------------------------------------------


def test_prepared_costs_reproduce_the_formula() raises:
    var coupled: List[Float64] = [0.0, 3.0]
    var c = CegbConfig.of(2.0, 0.25, coupled)
    var ledger = CegbLedger.create(c, 2, 8)
    var costs = prepare_cegb_node(c, ledger, 2, 8)
    assert_true(costs.active)
    # split 2 * 0.25 * 8 = 4; feature 1 adds its unpaid first use, 6.
    assert_true(close(costs.delta_of(0), 4.0))
    assert_true(close(costs.delta_of(1), 10.0))
    assert_true(close(costs.adjusted_gain(20.0, 1), 10.0))
    # Identical to the one-shot entry point, which is the point of having
    # both: one for a caller holding a ledger, one for a scan holding
    # prepared costs.
    assert_true(
        close(costs.delta_of(1), cegb_delta_gain(c, ledger, 1, 8))
    )


def test_the_split_term_can_be_charged_at_a_row_count_learned_later() raises:
    var c = CegbConfig.of(1.0, 0.5)
    var ledger = CegbLedger.none()
    # Costed with no row count, as `split.find_best_split` does when the
    # caller passes none and the histogram's own total is the answer.
    var costs = prepare_cegb_node(c, ledger, 3, 0)
    assert_true(close(costs.delta_of(0), 0.0))
    assert_true(close(costs.delta_at(0, 10), 5.0))
    assert_true(close(costs.adjusted_gain_at(8.0, 0, 10), 3.0))


def test_a_counted_lazy_penalty_pins_the_node_size() raises:
    var lazy: List[Float64] = [1.0, 1.0]
    var c = CegbConfig.of(1.0, 0.5, [], lazy)
    var ledger = CegbLedger.create(c, 2, 8)
    var costs = prepare_cegb_node(c, ledger, 2, 4, _rows(4))
    # The lazy term was counted over four rows, so charging the split term
    # against some other count would make one node's cost disagree about how
    # big the node is.
    assert_true(close(costs.delta_at(0, 4), 6.0))
    with assert_raises():
        _ = costs.delta_at(0, 5)


def test_a_restricted_scan_costs_only_the_features_it_scans() raises:
    var coupled: List[Float64] = [1.0, 2.0, 4.0]
    var c = CegbConfig.of(1.0, 0.0, coupled)
    var ledger = CegbLedger.create(c, 3, 4)
    var scanned: List[Int] = [0, 2]
    var costs = prepare_cegb_node(c, ledger, 3, 4, [], scanned)
    assert_true(close(costs.delta_of(0), 1.0))
    assert_true(close(costs.delta_of(2), 4.0))
    # Feature 1 was never costed, so asking for it is an error rather than a
    # silently uncharged zero.
    with assert_raises():
        _ = costs.delta_of(1)


def test_a_restricted_scan_works_with_the_split_cost_alone() raises:
    # The split cost is a property of the node, so every feature in the scan
    # set carries it even though no per-feature term was prepared. Costing
    # the node must still mark them: an unprepared feature raises, and these
    # are exactly the features the scan is about to ask about.
    var c = CegbConfig.of(1.0, 0.5)
    var ledger = CegbLedger.none()
    var scanned: List[Int] = [1, 3]
    var costs = prepare_cegb_node(c, ledger, 4, 10, [], scanned)
    assert_true(close(costs.delta_of(1), 5.0))
    assert_true(close(costs.delta_of(3), 5.0))
    with assert_raises():
        _ = costs.delta_of(0)


def test_inactive_costs_leave_a_gain_alone() raises:
    var costs = CegbNodeCosts.inactive()
    assert_false(costs.active)
    assert_true(close(costs.adjusted_gain(7.0, 0), 7.0))
    assert_true(close(costs.adjusted_gain_at(7.0, 99, 1000), 7.0))
    # An inactive bundle asks nothing of the feature id, so a scan on the
    # default path takes no new branch.
    assert_true(close(costs.delta_of(99), 0.0))


def test_costing_a_node_needs_the_ledger_the_costs_read() raises:
    var coupled: List[Float64] = [1.0]
    var coupled_only = CegbConfig.of(1.0, 0.0, coupled)
    var inert = CegbLedger.none()
    # An inert ledger answers "already paid" to everything, so charging a
    # first-use cost against one would silently charge nothing.
    with assert_raises():
        _ = prepare_cegb_node(coupled_only, inert, 1, 4)

    # A ledger built for the coupled cost does not track rows, so it cannot
    # serve a lazy penalty either.
    var lazy: List[Float64] = [1.0]
    var with_lazy = CegbConfig.of(1.0, 0.0, [], lazy)
    var feature_only = CegbLedger.create(coupled_only, 1, 4)
    with assert_raises():
        _ = prepare_cegb_node(with_lazy, feature_only, 1, 4, _rows(4))

    # And a lazy penalty needs the node's rows, exactly as many as the node
    # holds.
    var full = CegbLedger.create(with_lazy, 1, 4)
    with assert_raises():
        _ = prepare_cegb_node(with_lazy, full, 1, 4, _rows(3))


# ---------------------------------------------------------------------------
# The cached-candidate refund
# ---------------------------------------------------------------------------


def test_the_refund_restores_exactly_what_was_charged() raises:
    var coupled: List[Float64] = [0.0, 3.0]
    var c = CegbConfig.of(2.0, 0.0, coupled)
    var ledger = CegbLedger.create(c, 2, 8)
    # A leaf scores its best split on feature 1 while feature 1 still owes
    # its first use: 20 - 2 * 3 = 14.
    var cached = cegb_adjusted_gain(20.0, c, ledger, 1, 0)
    assert_true(close(cached, 14.0))
    # Another leaf then splits on feature 1 first. The model computes it now
    # whatever this leaf does, so the cached candidate is understated by
    # exactly what it was charged.
    var commit = cegb_commit_split(ledger, c, 1)
    var refund = cegb_stale_cached_gain(commit, 1, 1)
    assert_true(close(cached + refund, 20.0))


def test_only_candidates_on_the_newly_used_feature_are_refunded() raises:
    var coupled: List[Float64] = [5.0, 3.0]
    var c = CegbConfig.of(1.0, 0.0, coupled)
    var ledger = CegbLedger.create(c, 2, 8)
    var commit = cegb_commit_split(ledger, c, 1)
    # A cached candidate on feature 0 still owes feature 0's first use.
    assert_true(close(cegb_stale_cached_gain(commit, 0, 1), 0.0))
    assert_true(close(cegb_stale_cached_gain(commit, 1, 1), 3.0))
    # A leaf with no split found caches feature -1 and is refunded nothing.
    assert_true(close(cegb_stale_cached_gain(commit, -1, 1), 0.0))
    # A commit that used no new feature refunds nothing at all.
    var again = cegb_commit_split(ledger, c, 1)
    assert_true(close(cegb_stale_cached_gain(again, 1, 1), 0.0))


# ---------------------------------------------------------------------------
# The combinations that are refused rather than approximated
# ---------------------------------------------------------------------------


def test_a_grower_without_a_ledger_is_refused_not_ignored() raises:
    var split_only = CegbConfig.of(1.0, 0.5)
    # The split cost needs only the node's row count, so every grower keeps
    # it.
    check_cegb_grower_support(split_only, False, False)

    var coupled: List[Float64] = [1.0]
    var with_coupled = CegbConfig.of(1.0, 0.0, coupled)
    with assert_raises():
        check_cegb_grower_support(with_coupled, False, False)
    check_cegb_grower_support(with_coupled, True, False)

    var lazy: List[Float64] = [1.0]
    var with_lazy = CegbConfig.of(1.0, 0.0, [], lazy)
    with assert_raises():
        check_cegb_grower_support(with_lazy, True, False)
    check_cegb_grower_support(with_lazy, True, True)


def test_the_device_split_search_refuses_every_cegb_term() raises:
    # Candidates are ranked inside the kernel, so a host-side cost applied
    # after the winner is chosen ranks nothing.
    check_cegb_device_split_search(CegbConfig())
    with assert_raises():
        check_cegb_device_split_search(CegbConfig.of(1.0, 0.5))


def test_the_distributed_grower_refuses_only_the_lazy_term() raises:
    # Every rank scores from the reduced histogram and the exact global row
    # count, so the split cost agrees with no message; every rank commits the
    # same split, so the coupled flag flips at the same moment with no
    # message. The unread-row count is a sum over shards and neither.
    check_cegb_distributed(CegbConfig.of(1.0, 0.5))
    var coupled: List[Float64] = [1.0]
    check_cegb_distributed(CegbConfig.of(1.0, 0.0, coupled))
    var lazy: List[Float64] = [1.0]
    with assert_raises():
        check_cegb_distributed(CegbConfig.of(1.0, 0.0, [], lazy))


def test_resumed_training_refuses_the_terms_that_read_the_ledger() raises:
    # The ledger is training state and is not in the model file, so a run
    # resumed from disk would recharge every first use.
    var split_only = CegbConfig.of(1.0, 0.5)
    check_cegb_continued_training(split_only, True)
    var coupled: List[Float64] = [1.0]
    check_cegb_continued_training(CegbConfig.of(1.0, 0.0, coupled), False)
    with assert_raises():
        check_cegb_continued_training(CegbConfig.of(1.0, 0.0, coupled), True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
