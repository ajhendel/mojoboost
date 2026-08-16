"""CatBoost's ranking objectives and eval metrics: the formulas, the tie
rule, the grouping contract, and YetiRank's stream.

Catalog A22/A23/A24/A25, implemented in `catboost_ranking.mojo`. What this
file has to establish, in order of how badly it would hurt to be wrong:

1. **The sign is right.** CatBoost's `Der1` is the NEGATIVE gradient. A
   transcription that forgets the negation produces a model that boosts away
   from the loss and still passes every determinism test. Asserted by
   hand-derived numbers, not by a round trip.
2. **The hessian exclusion engages.** `PairLogit` and `YetiRank` have a
   per-row hessian by construction, so a constant-hessian declaration beside
   either must raise. Declaring one is a silently wrong-answer bug: the
   histogram builder rebuilds the hessian plane from the row count.
3. **YetiRank's stream is nameable and local.** A query's draws must depend
   on `(seed, iteration, query index, permutation)` and on NOTHING else --
   not on how many queries precede it, not on their sizes, not on the worker
   count. CatBoost's own stream fails this; ours must not.
4. **The tie rule is CatBoost's, not LightGBM's.** `CompareDocs` ranks the
   LOWER relevance first on a score tie. A metric that keeps input order
   instead reports a different number on any dataset that arrives sorted by
   relevance, and ranking datasets frequently do.
5. **The grouping contract is enforced loudly.** A non-contiguous group id is
   refused, never silently split.

Every expected value below is either an exact small rational or is rebuilt in
the test from the same operations in the same order, so no tolerance appears
anywhere.
"""

from std.math import log2
from std.os import setenv
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from mojotrees import BoosterParams, TreeParams, bin_equal_width
from mojotrees.catboost_ranking import (
    CatBoostNdcgParams,
    DEFAULT_PFOUND_DECAY,
    DEFAULT_YETIRANK_DECAY,
    DEFAULT_YETIRANK_PERMUTATIONS,
    METRIC_NDCG_CATBOOST,
    METRIC_PFOUND,
    NDCG_DENOMINATOR_LOG_POSITION,
    NDCG_DENOMINATOR_POSITION,
    NDCG_TYPE_BASE,
    NDCG_TYPE_EXP,
    PAIR_LOGIT,
    QUERY_RMSE,
    RankPair,
    TOP_ALL,
    YETIRANK_MAGIC_CONST,
    YETIRANK_NOISE_GUMBEL,
    YETIRANK_NOISE_NONE,
    YETI_RANK,
    YetiRankParams,
    catboost_ndcg,
    catboost_ranking_code_from_name,
    catboost_ranking_name,
    check_catboost_ranking_hessian_declaration,
    check_group_weights_constant,
    generate_pair_logit_pairs,
    group_weight,
    groups_from_group_id,
    is_catboost_ranking_objective,
    pair_logit_gradients,
    pairwise_varies_hessian,
    pfound,
    query_rmse_gradients,
    query_rmse_varies_hessian,
    train_catboost_ranker,
    yetirank_gradients,
    yetirank_pairs,
)
from mojotrees.ranking import LAMBDARANK, RankGroups, groups_from_counts


# ---------------------------------------------------------------------------
# Codes and names
# ---------------------------------------------------------------------------


def test_codes_are_distinct_and_not_lambdarank() raises:
    assert_equal(QUERY_RMSE, 13)
    assert_equal(PAIR_LOGIT, 14)
    assert_equal(YETI_RANK, 15)
    assert_equal(METRIC_PFOUND, 21)
    assert_equal(METRIC_NDCG_CATBOOST, 22)
    # None of the three may collide with LambdaRank's code, and `yetirank`
    # must never resolve to it: they are different losses (see the module
    # docstring) and an alias would make a comparison table lie.
    assert_true(QUERY_RMSE != LAMBDARANK)
    assert_true(PAIR_LOGIT != LAMBDARANK)
    assert_true(YETI_RANK != LAMBDARANK)
    assert_equal(catboost_ranking_code_from_name("yetirank"), YETI_RANK)
    with assert_raises():
        _ = catboost_ranking_code_from_name("lambdarank")
    assert_equal(catboost_ranking_name(QUERY_RMSE), String("query_rmse"))
    assert_equal(catboost_ranking_name(PAIR_LOGIT), String("pair_logit"))
    assert_equal(catboost_ranking_name(YETI_RANK), String("yetirank"))
    assert_equal(catboost_ranking_code_from_name("queryrmse"), QUERY_RMSE)
    assert_equal(catboost_ranking_code_from_name("pairlogit"), PAIR_LOGIT)
    assert_true(is_catboost_ranking_objective(YETI_RANK))
    assert_false(is_catboost_ranking_objective(LAMBDARANK))


def test_catboost_defaults_are_the_verified_ones() raises:
    # `GetYetiRankPermutations` -> 10, `GetYetiRankDecay` -> 0.85. NOT the
    # dead `Decay = 0.99` field initializer in TConfig, which the constructor
    # overwrites unconditionally.
    assert_equal(DEFAULT_YETIRANK_PERMUTATIONS, 10)
    assert_equal(DEFAULT_YETIRANK_DECAY, 0.85)
    assert_equal(YETIRANK_MAGIC_CONST, 0.15)
    assert_equal(DEFAULT_PFOUND_DECAY, 0.85)
    var d = YetiRankParams.default()
    assert_equal(d.permutations, 10)
    assert_equal(d.decay, 0.85)
    assert_equal(d.noise, YETIRANK_NOISE_GUMBEL)
    # CatBoost's NDCG default is Base + LogPosition + no truncation, which is
    # NOT LightGBM's (2^l - 1, truncated at eval_at).
    var n = CatBoostNdcgParams.default()
    assert_equal(n.dcg_type, NDCG_TYPE_BASE)
    assert_equal(n.denominator, NDCG_DENOMINATOR_LOG_POSITION)
    assert_equal(n.top, TOP_ALL)


# ---------------------------------------------------------------------------
# A25. The group_id contract
# ---------------------------------------------------------------------------


def test_group_id_must_be_contiguous_but_need_not_be_sorted() raises:
    # Unsorted ids are fine: only the runs matter (`CheckGroupIds` sorts the
    # RUN ids and looks for a duplicate).
    var g = groups_from_group_id([7, 7, 3, 3, 3])
    assert_equal(g.n_queries(), 2)
    assert_equal(g.size(0), 2)
    assert_equal(g.size(1), 3)
    assert_equal(g.n_rows, 5)
    # An id that reappears after a different id is refused, never silently
    # split into two queries. This is CatBoost's "group Ids are not
    # consecutive".
    with assert_raises():
        _ = groups_from_group_id([7, 7, 3, 3, 7])
    with assert_raises():
        _ = groups_from_group_id([1, 2, 1])


def test_group_weights_must_be_constant_inside_a_group() raises:
    var g = groups_from_counts([2, 2])
    # Empty means unit weights and passes.
    check_group_weights_constant([], g)
    check_group_weights_constant([1.0, 1.0, 2.5, 2.5], g)
    assert_equal(group_weight([1.0, 1.0, 2.5, 2.5], g, 1), 2.5)
    assert_equal(group_weight([], g, 1), 1.0)
    # Varying inside a group is refused rather than averaged: CatBoost reads
    # the group's weight off its FIRST row everywhere it uses one.
    with assert_raises():
        check_group_weights_constant([1.0, 2.0, 2.5, 2.5], g)
    with assert_raises():
        check_group_weights_constant([1.0, 1.0, 2.5], g)


# ---------------------------------------------------------------------------
# A22. QueryRMSE
# ---------------------------------------------------------------------------


def test_query_rmse_removes_the_query_mean() raises:
    # One query, scores all 0, targets 1/2/3. avg = 2, so
    # grad = a - t + avg = [1, 0, -1] and hess = [1, 1, 1].
    var groups = groups_from_counts([3])
    var scores: List[Float64] = [0.0, 0.0, 0.0]
    var targets: List[Float64] = [1.0, 2.0, 3.0]
    var grad = List[Float64]()
    var hess = List[Float64]()
    query_rmse_gradients(scores, targets, groups, grad, hess)
    assert_equal(grad[0], 1.0)
    assert_equal(grad[1], 0.0)
    assert_equal(grad[2], -1.0)
    assert_equal(hess[0], 1.0)
    assert_equal(hess[1], 1.0)
    assert_equal(hess[2], 1.0)
    # The sign matters more than anything else here: CatBoost writes
    # Der1 = t - a - avg, so a transcription that skipped the negation would
    # produce [-1, 0, 1] and boost away from the loss.
    assert_true(grad[0] > 0.0)
    # Antisymmetric within the query: nothing shifts the level.
    assert_equal(grad[0] + grad[1] + grad[2], 0.0)


def test_query_rmse_is_per_query_and_weighted() raises:
    # Two queries, each independently centered. Query 1's targets are query
    # 0's plus 10, so its gradients must be identical.
    var groups = groups_from_counts([3, 3])
    var scores: List[Float64] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    var targets: List[Float64] = [1.0, 2.0, 3.0, 11.0, 12.0, 13.0]
    var grad = List[Float64]()
    var hess = List[Float64]()
    query_rmse_gradients(scores, targets, groups, grad, hess)
    for i in range(3):
        assert_equal(grad[i], grad[3 + i])

    # Weighted: w = [1, 1, 2], t = [1, 2, 3], a = 0.
    #   querySum = 1 + 2 + 6 = 9, queryCount = 4, avg = 2.25
    #   grad = w (a - t + avg) = [1.25, 0.25, -1.5], hess = w = [1, 1, 2]
    var g1 = groups_from_counts([3])
    var s1: List[Float64] = [0.0, 0.0, 0.0]
    var t1: List[Float64] = [1.0, 2.0, 3.0]
    var w1: List[Float64] = [1.0, 1.0, 2.0]
    query_rmse_gradients(s1, t1, g1, grad, hess, w1)
    assert_equal(grad[0], 1.25)
    assert_equal(grad[1], 0.25)
    assert_equal(grad[2], -1.5)
    assert_equal(hess[0], 1.0)
    assert_equal(hess[1], 1.0)
    assert_equal(hess[2], 2.0)


def test_query_rmse_rejects_mismatched_inputs() raises:
    var groups = groups_from_counts([3])
    var grad = List[Float64]()
    var hess = List[Float64]()
    with assert_raises():
        query_rmse_gradients(
            [0.0, 0.0, 0.0], [1.0, 2.0], groups, grad, hess
        )
    with assert_raises():
        query_rmse_gradients(
            [0.0, 0.0], [1.0, 2.0], groups_from_counts([3]), grad, hess
        )
    with assert_raises():
        query_rmse_gradients(
            [0.0, 0.0, 0.0], [1.0, 2.0, 3.0], groups, grad, hess, [1.0]
        )


# ---------------------------------------------------------------------------
# A23. PairLogit
# ---------------------------------------------------------------------------


def test_pair_generation_is_catboosts_brute_force_order() raises:
    # `GenerateBruteForce`: nested loop, first < second, skip equal targets,
    # higher target wins, weight = group weight.
    var groups = groups_from_counts([3])
    var pairs = generate_pair_logit_pairs([3.0, 1.0, 2.0], groups)
    assert_equal(len(pairs), 3)
    assert_equal(pairs[0].winner, 0)
    assert_equal(pairs[0].loser, 1)
    assert_equal(pairs[1].winner, 0)
    assert_equal(pairs[1].loser, 2)
    # (1, 2) has target 1 < 2, so the SECOND index is the winner.
    assert_equal(pairs[2].winner, 2)
    assert_equal(pairs[2].loser, 1)
    for k in range(3):
        assert_equal(pairs[k].weight, 1.0)

    # Equal targets are skipped, not charged.
    var eq = generate_pair_logit_pairs([1.0, 1.0, 2.0], groups)
    assert_equal(len(eq), 2)
    assert_equal(eq[0].winner, 2)
    assert_equal(eq[0].loser, 0)
    assert_equal(eq[1].winner, 2)
    assert_equal(eq[1].loser, 1)

    # No pair ever spans two groups.
    var two = generate_pair_logit_pairs(
        [1.0, 2.0, 1.0, 2.0], groups_from_counts([2, 2])
    )
    assert_equal(len(two), 2)
    assert_equal(two[0].winner, 1)
    assert_equal(two[0].loser, 0)
    assert_equal(two[1].winner, 3)
    assert_equal(two[1].loser, 2)

    # The group weight rides on every pair of that group.
    var weighted = generate_pair_logit_pairs(
        [1.0, 2.0, 1.0, 2.0], groups_from_counts([2, 2]), [1.0, 1.0, 4.0, 4.0]
    )
    assert_equal(weighted[0].weight, 1.0)
    assert_equal(weighted[1].weight, 4.0)

    # A constant target column cannot generate pairs, and CatBoost says so
    # rather than returning an empty list.
    with assert_raises():
        _ = generate_pair_logit_pairs([2.0, 2.0, 2.0], groups)

    # Generation draws nothing, so two calls are identical.
    var again = generate_pair_logit_pairs([3.0, 1.0, 2.0], groups)
    for k in range(len(pairs)):
        assert_equal(pairs[k].winner, again[k].winner)
        assert_equal(pairs[k].loser, again[k].loser)
        assert_equal(pairs[k].weight, again[k].weight)


def test_pair_logit_gradients_at_equal_scores() raises:
    # p = sigmoid(0) = 1/2 for every pair, so each pair contributes 1/2 to the
    # gradients and 1/4 to both hessians.
    var scores: List[Float64] = [0.0, 0.0, 0.0]
    var pairs = List[RankPair]()
    pairs.append(RankPair(0, 1, 1.0))
    pairs.append(RankPair(0, 2, 1.0))
    pairs.append(RankPair(2, 1, 1.0))
    var grad = List[Float64]()
    var hess = List[Float64]()
    pair_logit_gradients(scores, pairs, grad, hess)
    # Row 0 wins twice: -1/2 - 1/2.
    assert_equal(grad[0], -1.0)
    # Row 1 loses twice: +1/2 + 1/2.
    assert_equal(grad[1], 1.0)
    # Row 2 loses once and wins once.
    assert_equal(grad[2], 0.0)
    # Every row is in exactly two pairs, so every hessian is 2 * 1/4.
    assert_equal(hess[0], 0.5)
    assert_equal(hess[1], 0.5)
    assert_equal(hess[2], 0.5)
    # The winner's gradient is NEGATIVE: a negative gradient pushes the raw
    # score UP under a Newton step. This is the sign CatBoost's Der1 hides.
    assert_true(grad[0] < 0.0)
    assert_true(grad[1] > 0.0)
    assert_equal(grad[0] + grad[1] + grad[2], 0.0)
    # Every hessian is strictly positive wherever a row is in a pair.
    for r in range(3):
        assert_true(hess[r] > 0.0)


def test_pair_logit_pair_weight_scales_both_derivatives() raises:
    var scores: List[Float64] = [0.0, 0.0]
    var one = List[RankPair]()
    one.append(RankPair(0, 1, 1.0))
    var three = List[RankPair]()
    three.append(RankPair(0, 1, 3.0))
    var g1 = List[Float64]()
    var h1 = List[Float64]()
    var g3 = List[Float64]()
    var h3 = List[Float64]()
    pair_logit_gradients(scores, one, g1, h1)
    pair_logit_gradients(scores, three, g3, h3)
    assert_equal(g3[0], 3.0 * g1[0])
    assert_equal(h3[0], 3.0 * h1[0])


def test_pair_logit_is_finite_at_extreme_scores() raises:
    # CatBoost forms e^{a_lose} / (e^{a_lose} + e^{a_win}) from stored
    # exponentials, which is inf/inf = NaN past about 709. Ours is a sigmoid
    # of the difference and stays finite and correct.
    var scores: List[Float64] = [5000.0, -5000.0]
    var pairs = List[RankPair]()
    pairs.append(RankPair(0, 1, 1.0))
    var grad = List[Float64]()
    var hess = List[Float64]()
    pair_logit_gradients(scores, pairs, grad, hess)
    # The winner already outranks the loser by 10000, so p is 0 and both
    # derivatives vanish. They must VANISH, not be NaN.
    assert_equal(grad[0], 0.0)
    assert_equal(grad[1], 0.0)
    assert_equal(hess[0], 0.0)
    assert_equal(hess[1], 0.0)

    # And the other way round: the loser far ahead gives p = 1, gradient 1,
    # hessian 0 (the logistic curvature vanishes at saturation).
    var flipped: List[Float64] = [-5000.0, 5000.0]
    pair_logit_gradients(flipped, pairs, grad, hess)
    assert_equal(grad[0], -1.0)
    assert_equal(grad[1], 1.0)
    assert_equal(hess[0], 0.0)


def test_pair_logit_rejects_out_of_range_rows() raises:
    var scores: List[Float64] = [0.0, 0.0]
    var bad = List[RankPair]()
    bad.append(RankPair(0, 7, 1.0))
    var grad = List[Float64]()
    var hess = List[Float64]()
    with assert_raises():
        pair_logit_gradients(scores, bad, grad, hess)


# ---------------------------------------------------------------------------
# Hessian declarations
# ---------------------------------------------------------------------------


def test_pairwise_objectives_forbid_a_constant_hessian() raises:
    # Unconditional: no parameter can make c*p*(1-p) constant.
    assert_true(pairwise_varies_hessian(PAIR_LOGIT))
    assert_true(pairwise_varies_hessian(YETI_RANK))
    assert_false(pairwise_varies_hessian(QUERY_RMSE))
    assert_false(pairwise_varies_hessian(LAMBDARANK))

    with assert_raises():
        check_catboost_ranking_hessian_declaration(PAIR_LOGIT, True)
    with assert_raises():
        check_catboost_ranking_hessian_declaration(YETI_RANK, True)
    # Not declaring one is always fine.
    check_catboost_ranking_hessian_declaration(PAIR_LOGIT, False)
    check_catboost_ranking_hessian_declaration(YETI_RANK, False)


def test_query_rmse_hessian_is_the_row_weight() raises:
    # Unweighted, the hessian is the literal 1.0 and the two-plane path is
    # admissible. This is the one CatBoost ranking objective for which that
    # is true.
    assert_false(query_rmse_varies_hessian([]))
    check_catboost_ranking_hessian_declaration(QUERY_RMSE, True)
    # Weighted, the hessian IS the weight, exactly as for squared error.
    assert_true(query_rmse_varies_hessian([1.0, 1.0]))
    with assert_raises():
        check_catboost_ranking_hessian_declaration(
            QUERY_RMSE, True, [1.0, 1.0]
        )
    # Keyed on emptiness, not on the values: an all-ones vector still counts.
    assert_true(query_rmse_varies_hessian([1.0, 1.0, 1.0]))


# ---------------------------------------------------------------------------
# A24. YetiRank
# ---------------------------------------------------------------------------


def _one_query_scores() -> List[Float64]:
    return [3.0, 2.0, 1.0]


def _one_query_targets() -> List[Float64]:
    return [0.0, 1.0, 2.0]


def test_yetirank_classic_weights_by_hand() raises:
    # Noise off, one permutation, decay 1/2. Sorted by score the order is
    # (0, 1, 2), so the adjacent pairs are (0,1) and (1,2):
    #   position 1: |rel0 - rel1| = 1, decay^0 = 1   -> 0.15, winner 1
    #   position 2: |rel1 - rel2| = 1, decay^1 = 1/2 -> 0.075, winner 2
    # Emission order is winner-major then loser-major, so (1,0) precedes
    # (2,1) because 1*3+0 = 3 < 2*3+1 = 7.
    var groups = groups_from_counts([3])
    var params = YetiRankParams(1, 0.5, YETIRANK_NOISE_NONE, 0)
    var pairs = yetirank_pairs(
        _one_query_scores(), _one_query_targets(), groups, params, 0
    )
    assert_equal(len(pairs), 2)
    assert_equal(pairs[0].winner, 1)
    assert_equal(pairs[0].loser, 0)
    assert_equal(pairs[0].weight, 0.15)
    assert_equal(pairs[1].winner, 2)
    assert_equal(pairs[1].loser, 1)
    assert_equal(pairs[1].weight, 0.075)


def test_yetirank_averages_over_permutations() raises:
    # With the noise off every permutation produces the same ranking, so the
    # accumulated charge is `permutations` times one draw and the division by
    # `permutations` returns it exactly. Two draws, not four, so that the sum
    # is a doubling and the division a halving: both exact in binary, which is
    # what lets this assert bits instead of a tolerance.
    var groups = groups_from_counts([3])
    var one = YetiRankParams(1, 0.5, YETIRANK_NOISE_NONE, 0)
    var two = YetiRankParams(2, 0.5, YETIRANK_NOISE_NONE, 0)
    var a = yetirank_pairs(
        _one_query_scores(), _one_query_targets(), groups, one, 0
    )
    var b = yetirank_pairs(
        _one_query_scores(), _one_query_targets(), groups, two, 0
    )
    assert_equal(len(a), len(b))
    for k in range(len(a)):
        assert_equal(a[k].winner, b[k].winner)
        assert_equal(a[k].loser, b[k].loser)
        assert_equal(a[k].weight, b[k].weight)


def test_yetirank_charges_nothing_for_equal_relevance() raises:
    # `AddWeight` has a `>` branch and a `<` branch and no `==` branch.
    var groups = groups_from_counts([3])
    var params = YetiRankParams(3, 0.85, YETIRANK_NOISE_GUMBEL, 7)
    var pairs = yetirank_pairs(
        _one_query_scores(), [2.0, 2.0, 2.0], groups, params, 0
    )
    assert_equal(len(pairs), 0)
    # A group of one document has no adjacent pair either.
    var single = yetirank_pairs(
        [1.0], [1.0], groups_from_counts([1]), params, 0
    )
    assert_equal(len(single), 0)


def test_yetirank_group_weight_scales_every_pair() raises:
    var groups = groups_from_counts([3])
    var params = YetiRankParams(1, 0.5, YETIRANK_NOISE_NONE, 0)
    var plain = yetirank_pairs(
        _one_query_scores(), _one_query_targets(), groups, params, 0
    )
    var heavy = yetirank_pairs(
        _one_query_scores(),
        _one_query_targets(),
        groups,
        params,
        0,
        [2.0, 2.0, 2.0],
    )
    assert_equal(len(plain), len(heavy))
    for k in range(len(plain)):
        assert_equal(heavy[k].weight, 2.0 * plain[k].weight)


def test_yetirank_noise_fires() raises:
    # A sampling scheme that quietly produced the deterministic ranking would
    # pass every determinism test below, so the noise is asserted to MOVE the
    # answer. Ten permutations of a three-document query with the scores in
    # relevance-inverted order cannot all agree with the noiseless ranking.
    var groups = groups_from_counts([3])
    var off = YetiRankParams(10, 0.85, YETIRANK_NOISE_NONE, 0)
    var on = YetiRankParams(10, 0.85, YETIRANK_NOISE_GUMBEL, 0)
    var a = yetirank_pairs(
        _one_query_scores(), _one_query_targets(), groups, off, 0
    )
    var b = yetirank_pairs(
        _one_query_scores(), _one_query_targets(), groups, on, 0
    )
    var moved = len(a) != len(b)
    if not moved:
        for k in range(len(a)):
            if (
                a[k].winner != b[k].winner
                or a[k].loser != b[k].loser
                or a[k].weight != b[k].weight
            ):
                moved = True
    assert_true(moved)


def test_yetirank_stream_is_keyed_and_local() raises:
    var groups = groups_from_counts([3])
    var params = YetiRankParams(6, 0.85, YETIRANK_NOISE_GUMBEL, 11)
    var base = yetirank_pairs(
        _one_query_scores(), _one_query_targets(), groups, params, 3
    )
    # Reproducible: same key, same pairs, every time.
    var again = yetirank_pairs(
        _one_query_scores(), _one_query_targets(), groups, params, 3
    )
    assert_equal(len(base), len(again))
    for k in range(len(base)):
        assert_equal(base[k].winner, again[k].winner)
        assert_equal(base[k].loser, again[k].loser)
        assert_equal(base[k].weight, again[k].weight)

    # The iteration is part of the key, so a later round redraws.
    var later = yetirank_pairs(
        _one_query_scores(), _one_query_targets(), groups, params, 4
    )
    var iteration_moved = len(later) != len(base)
    if not iteration_moved:
        for k in range(len(base)):
            if base[k].weight != later[k].weight:
                iteration_moved = True
    assert_true(iteration_moved)

    # The seed is part of the key too.
    var other_seed = YetiRankParams(6, 0.85, YETIRANK_NOISE_GUMBEL, 12)
    var reseeded = yetirank_pairs(
        _one_query_scores(), _one_query_targets(), groups, other_seed, 3
    )
    var seed_moved = len(reseeded) != len(base)
    if not seed_moved:
        for k in range(len(base)):
            if base[k].weight != reseeded[k].weight:
                seed_moved = True
    assert_true(seed_moved)


def test_yetirank_query_draws_do_not_depend_on_other_queries() raises:
    """The property CatBoost's own stream does NOT have.

    CatBoost seeds query `q` from a generator walked sequentially over the
    queries of `q`'s block, so inserting a query before it, or changing an
    earlier query's LENGTH, changes `q`'s draws. Ours keys on the query INDEX
    alone, so a query at index 1 draws the same uniforms whatever index 0
    contains and however long it is.
    """
    var params = YetiRankParams(5, 0.85, YETIRANK_NOISE_GUMBEL, 4)
    var target_query_scores = _one_query_scores()
    var target_query_targets = _one_query_targets()

    # Layout A: a two-row query, then the query under test at index 1.
    var scores_a: List[Float64] = [9.0, 8.0]
    var targets_a: List[Float64] = [1.0, 0.0]
    for i in range(3):
        scores_a.append(target_query_scores[i])
        targets_a.append(target_query_targets[i])
    var pairs_a = yetirank_pairs(
        scores_a, targets_a, groups_from_counts([2, 3]), params, 0
    )

    # Layout B: a FIVE-row query with different content, then the same query
    # under test, still at index 1.
    var scores_b: List[Float64] = [1.0, 2.0, 3.0, 4.0, 5.0]
    var targets_b: List[Float64] = [4.0, 3.0, 2.0, 1.0, 0.0]
    for i in range(3):
        scores_b.append(target_query_scores[i])
        targets_b.append(target_query_targets[i])
    var pairs_b = yetirank_pairs(
        scores_b, targets_b, groups_from_counts([5, 3]), params, 0
    )

    # Keep only the pairs of query 1 in each layout, as offsets from its
    # start, and require them to match exactly.
    var wins_a = List[Int]()
    var loses_a = List[Int]()
    var weights_a = List[Float64]()
    for k in range(len(pairs_a)):
        if pairs_a[k].winner >= 2:
            wins_a.append(pairs_a[k].winner - 2)
            loses_a.append(pairs_a[k].loser - 2)
            weights_a.append(pairs_a[k].weight)
    var wins_b = List[Int]()
    var loses_b = List[Int]()
    var weights_b = List[Float64]()
    for k in range(len(pairs_b)):
        if pairs_b[k].winner >= 5:
            wins_b.append(pairs_b[k].winner - 5)
            loses_b.append(pairs_b[k].loser - 5)
            weights_b.append(pairs_b[k].weight)

    assert_true(len(wins_a) > 0)
    assert_equal(len(wins_a), len(wins_b))
    for k in range(len(wins_a)):
        assert_equal(wins_a[k], wins_b[k])
        assert_equal(loses_a[k], loses_b[k])
        assert_equal(weights_a[k], weights_b[k])


def test_yetirank_is_worker_count_independent() raises:
    # Nothing here reads the worker count today, and this asserts the
    # CONTRACT rather than the current implementation: the day a lane
    # parallelizes the per-query loop, this is the test that has to keep
    # passing.
    var groups = groups_from_counts([4, 4])
    var scores: List[Float64] = [
        1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0
    ]
    var targets: List[Float64] = [
        0.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 0.0
    ]
    var params = YetiRankParams(8, 0.85, YETIRANK_NOISE_GUMBEL, 5)

    _ = setenv("MOJOTREES_NUM_WORKERS", "1", True)
    var one = yetirank_pairs(scores, targets, groups, params, 2)
    _ = setenv("MOJOTREES_NUM_WORKERS", "8", True)
    var eight = yetirank_pairs(scores, targets, groups, params, 2)
    _ = setenv("MOJOTREES_NUM_WORKERS", "3", True)
    var three = yetirank_pairs(scores, targets, groups, params, 2)

    assert_equal(len(one), len(eight))
    assert_equal(len(one), len(three))
    for k in range(len(one)):
        assert_equal(one[k].winner, eight[k].winner)
        assert_equal(one[k].loser, eight[k].loser)
        assert_equal(one[k].weight, eight[k].weight)
        assert_equal(one[k].weight, three[k].weight)


def test_yetirank_pairs_are_emitted_in_catboost_order() raises:
    # Winner-major then loser-major, and never spanning two groups.
    var groups = groups_from_counts([4, 4])
    var scores: List[Float64] = [
        4.0, 3.0, 2.0, 1.0, 4.0, 3.0, 2.0, 1.0
    ]
    var targets: List[Float64] = [
        0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0
    ]
    var params = YetiRankParams(3, 0.85, YETIRANK_NOISE_GUMBEL, 2)
    var pairs = yetirank_pairs(scores, targets, groups, params, 0)
    assert_true(len(pairs) > 0)
    var previous_key = -1
    var previous_group = -1
    for k in range(len(pairs)):
        var winner = pairs[k].winner
        var loser = pairs[k].loser
        var group = winner // 4
        assert_equal(group, loser // 4)
        var key = (winner % 4) * 4 + (loser % 4)
        if group == previous_group:
            assert_true(key > previous_key)
        previous_group = group
        previous_key = key
        assert_true(pairs[k].weight != 0.0)


def test_yetirank_gradients_go_through_pair_logit() raises:
    var groups = groups_from_counts([3])
    var params = YetiRankParams(1, 0.5, YETIRANK_NOISE_NONE, 0)
    var scores = _one_query_scores()
    var targets = _one_query_targets()
    var grad = List[Float64]()
    var hess = List[Float64]()
    yetirank_gradients(
        scores, targets, groups, grad, hess, params, 0
    )
    # Cross-check against the pair list applied by hand: pairs are
    # (1 beats 0, 0.15) and (2 beats 1, 0.075).
    var pairs = List[RankPair]()
    pairs.append(RankPair(1, 0, 0.15))
    pairs.append(RankPair(2, 1, 0.075))
    var g2 = List[Float64]()
    var h2 = List[Float64]()
    pair_logit_gradients(scores, pairs, g2, h2)
    for r in range(3):
        assert_equal(grad[r], g2[r])
        assert_equal(hess[r], h2[r])
    # Antisymmetric: the pairwise loss moves no query's level.
    assert_equal(grad[0] + grad[1] + grad[2], 0.0)
    for r in range(3):
        assert_true(hess[r] > 0.0)


def test_yetirank_params_refuse_unimplemented_arms() raises:
    with assert_raises():
        YetiRankParams(0, 0.85, YETIRANK_NOISE_GUMBEL, 0).validate()
    with assert_raises():
        YetiRankParams(10, 1.5, YETIRANK_NOISE_GUMBEL, 0).validate()
    with assert_raises():
        YetiRankParams(10, -0.1, YETIRANK_NOISE_GUMBEL, 0).validate()
    # The Gauss arm is refused by name rather than silently treated as Gumbel.
    with assert_raises():
        YetiRankParams(10, 0.85, 2, 0).validate()


# ---------------------------------------------------------------------------
# A25. NDCG and PFound
# ---------------------------------------------------------------------------


def test_ndcg_tie_rule_is_catboosts_not_lightgbms() raises:
    # Two documents, identical scores, relevances 1 and 0 in input order.
    # `CompareDocs` ranks the LOWER relevance first on a score tie, so the
    # ranking is (row 1, row 0) and the DCG is 0*1 + 1/log2(3).
    # A stable input-order sort -- LightGBM's rule -- would give 1*1 + 0 = 1
    # and a perfect NDCG of 1.0, which is the flattering answer this rule
    # exists to refuse.
    var groups = groups_from_counts([2])
    var scores: List[Float64] = [0.0, 0.0]
    var targets: List[Float64] = [1.0, 0.0]
    var value = catboost_ndcg(scores, targets, groups)
    assert_equal(value, 1.0 / log2(3.0))
    assert_true(value < 1.0)


def test_ndcg_base_and_exp_are_different_metrics() raises:
    # Two documents under the `Position` denominator, whose decays are 1 and
    # 1/2 and therefore exact in binary; the ranking is (row 0, row 1) by
    # score and the relevances are 1 then 3.
    #   Base: DCG = 1*1 + 3*(1/2) = 2.5,  IDCG = 3*1 + 1*(1/2) = 3.5
    #   Exp:  DCG = 1*1 + 7*(1/2) = 4.5,  IDCG = 7*1 + 1*(1/2) = 7.5
    var groups = groups_from_counts([2])
    var scores: List[Float64] = [1.0, 0.0]
    var targets: List[Float64] = [1.0, 3.0]

    var base = catboost_ndcg(
        scores,
        targets,
        groups,
        CatBoostNdcgParams(TOP_ALL, NDCG_TYPE_BASE, NDCG_DENOMINATOR_POSITION),
    )
    assert_equal(base, 2.5 / 3.5)

    var expd = catboost_ndcg(
        scores,
        targets,
        groups,
        CatBoostNdcgParams(TOP_ALL, NDCG_TYPE_EXP, NDCG_DENOMINATOR_POSITION),
    )
    assert_equal(expd, 0.6)

    # They are different numbers, which is the whole reason CatBoost's NDCG
    # cannot be scored with `ranking.ndcg` and the reverse.
    assert_true(base != expd)

    # And `LogPosition`, CatBoost's default, is a third number: its second
    # decay is 1/log2(3) rather than 1/2.
    var d1 = 1.0 / log2(3.0)
    var log_position = catboost_ndcg(
        scores,
        targets,
        groups,
        CatBoostNdcgParams(
            TOP_ALL, NDCG_TYPE_BASE, NDCG_DENOMINATOR_LOG_POSITION
        ),
    )
    assert_true(log_position != base)
    assert_true(log_position > 0.0 and log_position < 1.0)
    assert_true(d1 > 0.6 and d1 < 0.64)


def test_ndcg_truncation_and_the_zero_ideal() raises:
    var groups = groups_from_counts([3])
    var scores: List[Float64] = [3.0, 2.0, 1.0]
    var targets: List[Float64] = [0.0, 0.0, 5.0]
    # top = 1 keeps only the highest-scoring document, whose relevance is 0,
    # so the DCG is 0 while the ideal top-1 is 5.
    var top1 = catboost_ndcg(
        scores,
        targets,
        groups,
        CatBoostNdcgParams(1, NDCG_TYPE_BASE, NDCG_DENOMINATOR_LOG_POSITION),
    )
    assert_equal(top1, 0.0)
    # CatBoost's default is NO truncation, which finds the relevant document
    # at position 2.
    var untruncated = catboost_ndcg(scores, targets, groups)
    assert_equal(untruncated, (5.0 / log2(4.0)) / 5.0)

    # An all-zero-relevance query has no attainable DCG and scores 1, which is
    # `CalcNdcg`'s `idcg > 0 ? dcg / idcg : 1`.
    assert_equal(
        catboost_ndcg(scores, [0.0, 0.0, 0.0], groups), 1.0
    )


def test_ndcg_is_the_group_weighted_mean() raises:
    var groups = groups_from_counts([2, 2])
    var scores: List[Float64] = [0.0, 0.0, 1.0, 0.0]
    # Query 0 hits the tie rule and scores 1/log2(3); query 1 is perfect.
    var targets: List[Float64] = [1.0, 0.0, 1.0, 0.0]
    var d1 = 1.0 / log2(3.0)
    assert_equal(catboost_ndcg(scores, targets, groups), (d1 + 1.0) / 2.0)
    # A group weight of 3 on the perfect query pulls the mean up.
    assert_equal(
        catboost_ndcg(scores, targets, groups, CatBoostNdcgParams.default(),
                      [1.0, 1.0, 3.0, 3.0]),
        (d1 + 3.0) / 4.0,
    )


def test_pfound_recursion_and_truncation() raises:
    var groups = groups_from_counts([2])
    var scores: List[Float64] = [1.0, 0.0]
    var targets: List[Float64] = [0.5, 0.5]
    # pFound = 0.5 + 0.5 * ((1 - 0.5) * decay), rebuilt with the same
    # operations in the same order so no tolerance is needed.
    var expected = 0.5 + 0.5 * ((1.0 - 0.5) * DEFAULT_PFOUND_DECAY)
    assert_equal(pfound(scores, targets, groups), expected)
    # top = 1 stops after the first document.
    assert_equal(pfound(scores, targets, groups, 1), 0.5)
    # A fully irrelevant query contributes nothing.
    assert_equal(pfound(scores, [0.0, 0.0], groups), 0.0)
    # A certainly-relevant document at the top saturates it.
    assert_equal(pfound(scores, [1.0, 1.0], groups), 1.0)


def test_pfound_uses_the_same_tie_rule() raises:
    # Identical scores, relevances 1 then 0: the lower relevance is ranked
    # first, so the relevant document is found at position 1 with pLook
    # already decayed.
    var groups = groups_from_counts([2])
    var scores: List[Float64] = [0.0, 0.0]
    var targets: List[Float64] = [1.0, 0.0]
    assert_equal(
        pfound(scores, targets, groups), 1.0 * ((1.0 - 0.0) * DEFAULT_PFOUND_DECAY)
    )


def test_pfound_is_the_group_weighted_mean_and_validates() raises:
    var groups = groups_from_counts([2, 2])
    var scores: List[Float64] = [1.0, 0.0, 1.0, 0.0]
    var targets: List[Float64] = [1.0, 0.0, 0.0, 0.0]
    assert_equal(pfound(scores, targets, groups), 0.5)
    assert_equal(
        pfound(scores, targets, groups, TOP_ALL, DEFAULT_PFOUND_DECAY,
               [3.0, 3.0, 1.0, 1.0]),
        0.75,
    )
    with assert_raises():
        _ = pfound(scores, targets, groups, 0)
    with assert_raises():
        _ = pfound(scores, targets, groups, TOP_ALL, 1.5)
    with assert_raises():
        _ = pfound(scores, [1.0, 0.0], groups)


# ---------------------------------------------------------------------------
# The fit path
# ---------------------------------------------------------------------------


def test_fit_runs_and_carries_the_objective() raises:
    # Two queries of four documents, one feature that is exactly the
    # relevance order. A handful of shallow trees, which is a unit-test fit
    # and not a training run.
    var features: List[Float64] = [
        0.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 0.0
    ]
    var targets: List[Float64] = [
        0.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 0.0
    ]
    var groups = groups_from_counts([4, 4])
    var data = bin_equal_width(features, 8, 1, 4)
    var params = BoosterParams(4, 0.3, TreeParams(4, 1, 1.0, 1e-3))

    var codes: List[Int] = [QUERY_RMSE, PAIR_LOGIT, YETI_RANK]
    for c in range(len(codes)):
        var objective = codes[c]
        var booster = train_catboost_ranker(
            data, targets, groups, objective, params
        )
        assert_equal(booster.objective, objective)
        # All three boost from zero: the level is unidentifiable.
        assert_equal(booster.base_score, 0.0)
        assert_true(len(booster.trees) > 0)
        # No leaf may be a NaN. `0/0` in a leaf denominator is the failure
        # this asserts against.
        var scores = List[Float64](capacity=8)
        for r in range(8):
            var s = booster.predict_row(data, r)
            assert_true(s == s)
            scores.append(s)
        # The fit is reproducible, YetiRank included.
        var again = train_catboost_ranker(
            data, targets, groups, objective, params
        )
        for r in range(8):
            assert_equal(again.predict_row(data, r), scores[r])


def test_fit_rejects_bad_inputs() raises:
    var features: List[Float64] = [0.0, 1.0, 2.0, 3.0]
    var targets: List[Float64] = [0.0, 1.0, 2.0, 3.0]
    var data = bin_equal_width(features, 4, 1, 4)
    var params = BoosterParams(2, 0.3, TreeParams(4, 1, 1.0, 1e-3))
    # Not one of the three codes.
    with assert_raises():
        _ = train_catboost_ranker(
            data, targets, groups_from_counts([4]), LAMBDARANK, params
        )
    # Groups covering the wrong number of rows.
    with assert_raises():
        _ = train_catboost_ranker(
            data, targets, groups_from_counts([3]), QUERY_RMSE, params
        )
    # A weight that varies inside a group.
    with assert_raises():
        _ = train_catboost_ranker(
            data,
            targets,
            groups_from_counts([4]),
            PAIR_LOGIT,
            params,
            YetiRankParams.default(),
            [1.0, 2.0, 1.0, 1.0],
        )
    # A constant target column: no pair can be generated from it.
    with assert_raises():
        _ = train_catboost_ranker(
            data,
            [1.0, 1.0, 1.0, 1.0],
            groups_from_counts([4]),
            PAIR_LOGIT,
            params,
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
