"""Tests for LambdaRank learning to rank.

Every expected lambda, hessian, and NDCG below is derived by hand from the
formulas in ranking.mojo and written twice: once as the algebra evaluated
from the same primitives the implementation uses (exact to machine
precision), and once as a decimal constant computed independently, checked
to 1e-8 because Mojo's `log2` carries about 1e-10 of relative error.
"""

from std.math import log2
from std.os import remove
from std.testing import assert_equal, assert_true, assert_raises, TestSuite

from mojoboost.bagging import BaggingParams
from mojoboost.binning import bin_equal_width
from mojoboost.boosting import BoosterParams
from mojoboost.ranking import (
    LAMBDARANK,
    RankerParams,
    _refresh_query_bag,
    check_groups,
    fit_ranker,
    groups_from_counts,
    groups_from_query_ids,
    label_gain,
    lambdarank_gradients,
    max_dcg,
    ndcg,
    ndcg_at_cutoffs,
    train_ranker,
    train_ranker_with_valid,
)
from mojoboost.serialize import load_model, save_model
from mojoboost.tree import TreeParams

comptime _TMP_PATH = "./.test_ranker_roundtrip.tmp"

# Position discounts 1 / log2(rank + 2) for ranks 0..3.
comptime _D0 = 1.0
comptime _D1 = 0.6309297535714575
comptime _D2 = 0.5
comptime _D3 = 0.43067655807339306


def _close(a: Float64, b: Float64, tol: Float64 = 1e-8) -> Bool:
    return abs(a - b) <= tol * (1.0 + abs(b))


def _params(n_rounds: Int) -> BoosterParams:
    return BoosterParams(n_rounds, 0.3, TreeParams(8, 1, 1.0, 1e-3))


# ---------------------------------------------------------------- groups


def test_groups_from_counts() raises:
    var g = groups_from_counts([2, 3, 1])
    assert_equal(g.n_queries(), 3)
    assert_equal(g.n_rows, 6)
    assert_equal(g.start(0), 0)
    assert_equal(g.start(1), 2)
    assert_equal(g.start(2), 5)
    assert_equal(g.size(0), 2)
    assert_equal(g.size(1), 3)
    assert_equal(g.size(2), 1)
    assert_equal(g.max_size(), 3)


def test_groups_from_counts_rejects_malformed() raises:
    with assert_raises():
        _ = groups_from_counts([])
    with assert_raises():
        _ = groups_from_counts([2, 0, 1])
    with assert_raises():
        _ = groups_from_counts([2, -1])


def test_check_groups_rejects_row_count_mismatch() raises:
    var g = groups_from_counts([2, 3])
    check_groups(g, 5)
    with assert_raises():
        check_groups(g, 6)


def test_groups_from_query_ids_contiguous() raises:
    # Ids need not be sorted or dense, only contiguous.
    var g = groups_from_query_ids([7, 7, 3, 3, 3, 9])
    assert_equal(g.n_queries(), 3)
    assert_equal(g.size(0), 2)
    assert_equal(g.size(1), 3)
    assert_equal(g.size(2), 1)
    assert_equal(g.n_rows, 6)


def test_groups_from_query_ids_rejects_noncontiguous() raises:
    # Query 0 comes back after query 1 started: two runs of one id.
    with assert_raises():
        _ = groups_from_query_ids([0, 0, 1, 1, 0])
    with assert_raises():
        _ = groups_from_query_ids([5, 6, 5])
    with assert_raises():
        _ = groups_from_query_ids([])


# ------------------------------------------------------------------ NDCG


def test_label_gain_is_two_to_the_label_minus_one() raises:
    assert_equal(label_gain(0), 0.0)
    assert_equal(label_gain(1), 1.0)
    assert_equal(label_gain(3), 7.0)
    assert_equal(label_gain(30), 1073741823.0)


def test_ndcg_hand_calculated() raises:
    # One query, scores already descending, so the ranking is the input
    # order. Labels [3, 2, 3, 0] give gains [7, 3, 7, 0].
    #   DCG@4    = 7*1 + 3/log2(3) + 7*0.5 + 0
    #   maxDCG@4 = 7*1 + 7/log2(3) + 3*0.5 + 0   (labels sorted 3, 3, 2, 0)
    var scores: List[Float64] = [0.5, 0.4, 0.3, 0.2]
    var labels: List[Int] = [3, 2, 3, 0]
    var g = groups_from_counts([4])

    var dcg4 = 7.0 * _D0 + 3.0 / log2(3.0) + 7.0 * _D2
    var max4 = 7.0 * _D0 + 7.0 / log2(3.0) + 3.0 * _D2
    # 1e-14 rather than equality: the implementation accumulates its sums in
    # rank order, which associates the same terms differently.
    assert_true(_close(ndcg(scores, labels, g, 4), dcg4 / max4, 1e-14))
    assert_true(_close(ndcg(scores, labels, g, 4), 0.9594535145926796))

    var dcg2 = 7.0 * _D0 + 3.0 / log2(3.0)
    var max2 = 7.0 * _D0 + 7.0 / log2(3.0)
    assert_true(_close(ndcg(scores, labels, g, 2), dcg2 / max2, 1e-14))
    assert_true(_close(ndcg(scores, labels, g, 2), 0.7789412530088334))

    # The top document already has the best label, so NDCG@1 is perfect.
    assert_equal(ndcg(scores, labels, g, 1), 1.0)

    # A cutoff past the end of the query is the whole query.
    assert_equal(ndcg(scores, labels, g, 99), ndcg(scores, labels, g, 4))

    assert_true(_close(max_dcg(labels, 0, 4, 4), max4, 1e-14))
    assert_true(_close(max_dcg(labels, 0, 4, 2), max2, 1e-14))


def test_ndcg_perfect_and_reversed() raises:
    var labels: List[Int] = [0, 1, 2, 3]
    var g = groups_from_counts([4])
    var perfect: List[Float64] = [0.0, 1.0, 2.0, 3.0]
    assert_equal(ndcg(perfect, labels, g, 4), 1.0)

    # Exactly reversed: DCG is the gains in ascending order.
    var reversed: List[Float64] = [3.0, 2.0, 1.0, 0.0]
    var dcg = 0.0 * _D0 + 1.0 * _D1 + 3.0 * _D2 + 7.0 * _D3
    var best = 7.0 * _D0 + 3.0 * _D1 + 1.0 * _D2 + 0.0 * _D3
    assert_true(_close(ndcg(reversed, labels, g, 4), dcg / best))
    assert_true(_close(ndcg(reversed, labels, g, 4), 0.547831481922746))
    assert_true(
        ndcg(reversed, labels, g, 4) < ndcg(perfect, labels, g, 4)
    )
    # NDCG@1 is the harshest cutoff: the worst document is on top.
    assert_equal(ndcg(reversed, labels, g, 1), 0.0)


def test_ndcg_ties_keep_input_order() raises:
    # Equal scores must rank in the order the rows arrived, LightGBM's
    # stable sort, so a query handed in best-first scores 1.0.
    var labels: List[Int] = [3, 2, 1, 0]
    var g = groups_from_counts([4])
    var flat: List[Float64] = [0.0, 0.0, 0.0, 0.0]
    assert_equal(ndcg(flat, labels, g, 4), 1.0)

    var worst: List[Int] = [0, 1, 2, 3]
    assert_true(ndcg(flat, worst, g, 4) < 1.0)


def test_ndcg_all_zero_label_query_counts_as_one() raises:
    # A query with nothing relevant has no attainable DCG; LightGBM counts
    # it as 1.0 rather than dropping it.
    var scores: List[Float64] = [0.3, 0.9, 0.1, 0.2]
    var labels: List[Int] = [0, 0, 1, 0]
    var g = groups_from_counts([2, 2])
    # Query 0 contributes 1.0; query 1 ranks its relevant document second.
    var q1 = _D1 / _D0
    assert_true(_close(ndcg(scores, labels, g, 2), (1.0 + q1) / 2.0))


def test_ndcg_averages_over_queries_never_across_them() raises:
    # Two queries whose rows would interleave badly if pooled: every
    # document of query 1 outscores every document of query 0, yet each
    # query is ranked perfectly inside itself.
    var scores: List[Float64] = [1.0, 0.0, 11.0, 10.0]
    var labels: List[Int] = [1, 0, 1, 0]
    var g = groups_from_counts([2, 2])
    assert_equal(ndcg(scores, labels, g, 2), 1.0)


def test_ndcg_at_cutoffs_matches_single_cutoff() raises:
    var scores: List[Float64] = [0.5, 0.4, 0.3, 0.2, 0.9, 0.1]
    var labels: List[Int] = [3, 2, 3, 0, 0, 2]
    var g = groups_from_counts([4, 2])
    var many = ndcg_at_cutoffs(scores, labels, g, [1, 2, 4])
    assert_equal(len(many), 3)
    assert_equal(many[0], ndcg(scores, labels, g, 1))
    assert_equal(many[1], ndcg(scores, labels, g, 2))
    assert_equal(many[2], ndcg(scores, labels, g, 4))


def test_ndcg_validates_inputs() raises:
    var scores: List[Float64] = [0.5, 0.4]
    var labels: List[Int] = [1, 0]
    var g = groups_from_counts([2])
    with assert_raises():
        _ = ndcg(scores, labels, g, 0)
    with assert_raises():
        _ = ndcg_at_cutoffs(scores, labels, g, [])
    with assert_raises():
        _ = ndcg(scores, [1, 0, 1], g, 2)
    with assert_raises():
        _ = ndcg(scores, [1, -1], g, 2)
    with assert_raises():
        _ = ndcg(scores, [1, 31], g, 2)


# ------------------------------------------------------- pairwise lambdas


def test_single_pair_lambdas_hand_calculated() raises:
    # One query, two documents, labels [1, 0], both scores 0.
    #   maxDCG@2      = 1 * 1 = 1, so inverse_max_dcg = 1
    #   paired discount = |1 - 1/log2(3)| = 0.3690702464285425
    #   |dNDCG|       = (1 - 0) * 0.3690702464285425 * 1
    #   rho           = 1 / (1 + exp(0)) = 0.5
    #   lambda        = -1 * 0.5 * |dNDCG| = -0.18453512321427125
    #   hess          = 1 * 0.5 * 0.5 * |dNDCG| = 0.09226756160713562
    # Document 0 is the relevant one, so it takes the negative gradient
    # (a positive Newton step) and document 1 the positive one.
    var scores: List[Float64] = [0.0, 0.0]
    var labels: List[Int] = [1, 0]
    var g = groups_from_counts([2])
    var rp = RankerParams(30, 1.0, False, 5)

    var grad = List[Float64]()
    var hess = List[Float64]()
    lambdarank_gradients(scores, labels, g, grad, hess, rp)

    var paired = 1.0 - 1.0 / log2(3.0)
    var delta_ndcg = paired
    assert_equal(grad[0], -0.5 * delta_ndcg)
    assert_equal(grad[1], 0.5 * delta_ndcg)
    assert_equal(hess[0], 0.25 * delta_ndcg)
    assert_equal(hess[1], 0.25 * delta_ndcg)
    assert_true(_close(grad[0], -0.18453512321427123))
    assert_true(_close(hess[0], 0.09226756160713562))


def test_norm_rescales_the_query_hand_calculated() raises:
    # Same query with lambdarank_norm on. Every score is equal, so the
    # per-pair 0.01 + |delta| division is skipped (best == worst) and only
    # the per-query factor log2(1 + S) / S applies, with S the summed
    # lambda magnitude 2 * 0.18453512321427125.
    var scores: List[Float64] = [0.0, 0.0]
    var labels: List[Int] = [1, 0]
    var g = groups_from_counts([2])

    var grad = List[Float64]()
    var hess = List[Float64]()
    lambdarank_gradients(
        scores, labels, g, grad, hess, RankerParams(30, 1.0, True, 5)
    )

    var delta_ndcg = 1.0 - 1.0 / log2(3.0)
    var s = 2.0 * 0.5 * delta_ndcg
    var factor = log2(1.0 + s) / s
    assert_equal(grad[0], -0.5 * delta_ndcg * factor)
    assert_equal(hess[0], 0.25 * delta_ndcg * factor)
    assert_true(_close(grad[0], -0.22659823629063455))
    assert_true(_close(hess[0], 0.11329911814531728))


def test_equal_labels_produce_no_pair() raises:
    # LightGBM skips pairs whose labels match: nothing to reorder.
    var scores: List[Float64] = [0.0, 5.0, 0.0]
    var labels: List[Int] = [2, 2, 2]
    var g = groups_from_counts([3])
    var grad = List[Float64]()
    var hess = List[Float64]()
    lambdarank_gradients(scores, labels, g, grad, hess)
    for r in range(3):
        assert_equal(grad[r], 0.0)
        assert_equal(hess[r], 0.0)


def test_all_zero_label_query_has_zero_lambdas() raises:
    var scores: List[Float64] = [0.1, 0.2, 0.3]
    var labels: List[Int] = [0, 0, 0]
    var g = groups_from_counts([3])
    var grad = List[Float64]()
    var hess = List[Float64]()
    lambdarank_gradients(scores, labels, g, grad, hess)
    for r in range(3):
        assert_equal(grad[r], 0.0)


def test_single_document_query_has_zero_lambdas() raises:
    var scores: List[Float64] = [0.7]
    var labels: List[Int] = [3]
    var g = groups_from_counts([1])
    var grad = List[Float64]()
    var hess = List[Float64]()
    lambdarank_gradients(scores, labels, g, grad, hess)
    assert_equal(grad[0], 0.0)
    assert_equal(hess[0], 0.0)


def test_truncation_level_drops_deep_pairs() raises:
    # Three documents, labels [2, 1, 0], all scores equal so the ranking is
    # the input order. With truncation 30 all three pairs count; with
    # truncation 1 only the pairs whose better-ranked member is at rank 0
    # count, which drops the (rank 1, rank 2) pair entirely.
    #
    # As in LightGBM, the truncation level is also the cutoff of the maxDCG
    # the lambdas are divided by, so lowering it to 1 replaces maxDCG@3
    # (3 + 1/log2(3)) with maxDCG@1 (3) and rescales every surviving pair.
    # By hand at truncation 1, with inverse maxDCG 1/3 and rho = 1/2:
    #   pair (rank 0, rank 1): |dNDCG| = (3 - 1) * (1 - 1/log2(3)) / 3
    #                          lambda  = -0.12302341547618082
    #   pair (rank 0, rank 2): |dNDCG| = (3 - 0) * (1 - 1/2) / 3 = 1/2
    #                          lambda  = -0.25
    var scores: List[Float64] = [0.0, 0.0, 0.0]
    var labels: List[Int] = [2, 1, 0]
    var g = groups_from_counts([3])

    var full_grad = List[Float64]()
    var full_hess = List[Float64]()
    lambdarank_gradients(
        scores, labels, g, full_grad, full_hess, RankerParams(30, 1.0, False, 5)
    )
    assert_true(_close(full_grad[0], -0.3082048737868866))
    assert_true(_close(full_grad[1], 0.0836164261630733))
    assert_true(_close(full_grad[2], 0.22458844762381333))
    assert_true(_close(full_hess[0], 0.1541024368934433))

    var trunc_grad = List[Float64]()
    var trunc_hess = List[Float64]()
    lambdarank_gradients(
        scores,
        labels,
        g,
        trunc_grad,
        trunc_hess,
        RankerParams(1, 1.0, False, 5),
    )
    var pair01 = (1.0 - 1.0 / log2(3.0)) * (1.0 / 3.0)
    assert_equal(trunc_grad[0], -(pair01 + 0.25))
    assert_equal(trunc_grad[1], pair01)
    assert_equal(trunc_grad[2], 0.25)
    assert_true(_close(trunc_grad[0], -0.3730234154761808))
    assert_true(_close(trunc_grad[1], 0.12302341547618082))
    # The dropped pair is the one that separated documents 1 and 2, so
    # document 2's whole gradient is now the single pair against the top.
    assert_true(trunc_grad[1] != full_grad[1])
    assert_true(trunc_hess[2] != full_hess[2])


def test_lambdas_are_antisymmetric_within_a_query() raises:
    # Each pair adds and subtracts the same lambda, so a query's gradients
    # sum to zero: a ranker learns order, never level.
    var scores: List[Float64] = [0.3, -0.2, 1.4, 0.0, 0.9, 0.5, -1.0]
    var labels: List[Int] = [1, 3, 0, 2, 0, 4, 1]
    var g = groups_from_counts([4, 3])
    var grad = List[Float64]()
    var hess = List[Float64]()
    lambdarank_gradients(scores, labels, g, grad, hess)
    for q in range(g.n_queries()):
        var total = 0.0
        for r in range(g.start(q), g.start(q) + g.size(q)):
            total += grad[r]
            assert_true(hess[r] >= 0.0)
        assert_true(abs(total) < 1e-12)


def test_relevant_documents_get_negative_gradients() raises:
    # The most relevant document is pushed up (negative gradient, positive
    # Newton step) and the least relevant down.
    var scores: List[Float64] = [0.0, 0.0, 0.0]
    var labels: List[Int] = [0, 1, 2]
    var g = groups_from_counts([3])
    var grad = List[Float64]()
    var hess = List[Float64]()
    lambdarank_gradients(scores, labels, g, grad, hess)
    assert_true(grad[2] < 0.0)
    assert_true(grad[0] > 0.0)


def test_queries_do_not_mix() raises:
    # A row's lambda depends only on its own query. Two identical queries
    # produce bit-identical blocks, and rewriting the second query's scores
    # and labels leaves the first query's gradients untouched.
    var one_scores: List[Float64] = [0.4, 0.1, 0.9]
    var one_labels: List[Int] = [2, 0, 1]
    var one = groups_from_counts([3])
    var base_grad = List[Float64]()
    var base_hess = List[Float64]()
    lambdarank_gradients(one_scores, one_labels, one, base_grad, base_hess)

    var two_scores: List[Float64] = [0.4, 0.1, 0.9, 0.4, 0.1, 0.9]
    var two_labels: List[Int] = [2, 0, 1, 2, 0, 1]
    var two = groups_from_counts([3, 3])
    var grad = List[Float64]()
    var hess = List[Float64]()
    lambdarank_gradients(two_scores, two_labels, two, grad, hess)
    for i in range(3):
        assert_equal(grad[i], base_grad[i])
        assert_equal(hess[i], base_hess[i])
        assert_equal(grad[3 + i], base_grad[i])
        assert_equal(hess[3 + i], base_hess[i])

    var other_scores: List[Float64] = [0.4, 0.1, 0.9, 7.0, -7.0, 2.5]
    var other_labels: List[Int] = [2, 0, 1, 4, 4, 0]
    var other_grad = List[Float64]()
    var other_hess = List[Float64]()
    lambdarank_gradients(
        other_scores, other_labels, two, other_grad, other_hess
    )
    for i in range(3):
        assert_equal(other_grad[i], base_grad[i])
        assert_equal(other_hess[i], base_hess[i])


def test_sample_weight_scales_lambdas() raises:
    var scores: List[Float64] = [0.4, 0.1, 0.9, 0.2]
    var labels: List[Int] = [2, 0, 1, 3]
    var g = groups_from_counts([2, 2])
    var plain_grad = List[Float64]()
    var plain_hess = List[Float64]()
    lambdarank_gradients(scores, labels, g, plain_grad, plain_hess)

    var ones: List[Float64] = [1.0, 1.0, 1.0, 1.0]
    var ones_grad = List[Float64]()
    var ones_hess = List[Float64]()
    lambdarank_gradients(
        scores,
        labels,
        g,
        ones_grad,
        ones_hess,
        RankerParams.default(),
        ones,
    )
    for r in range(4):
        assert_equal(ones_grad[r], plain_grad[r])

    # Weights multiply after the per-query normalization, so they scale
    # each row exactly.
    var w: List[Float64] = [2.0, 0.0, 3.0, 0.5]
    var w_grad = List[Float64]()
    var w_hess = List[Float64]()
    lambdarank_gradients(
        scores, labels, g, w_grad, w_hess, RankerParams.default(), w
    )
    for r in range(4):
        assert_equal(w_grad[r], plain_grad[r] * w[r])
        assert_equal(w_hess[r], plain_hess[r] * w[r])


def test_gradients_validate_inputs() raises:
    var scores: List[Float64] = [0.4, 0.1]
    var labels: List[Int] = [1, 0]
    var g = groups_from_counts([2])
    var grad = List[Float64]()
    var hess = List[Float64]()
    with assert_raises():
        lambdarank_gradients(scores, [1, 0, 1], g, grad, hess)
    with assert_raises():
        lambdarank_gradients(scores, [1, -2], g, grad, hess)
    with assert_raises():
        lambdarank_gradients(
            scores, labels, g, grad, hess, RankerParams(0, 1.0, True, 5)
        )
    with assert_raises():
        lambdarank_gradients(
            scores, labels, g, grad, hess, RankerParams(30, 0.0, True, 5)
        )
    with assert_raises():
        lambdarank_gradients(
            scores, labels, g, grad, hess, RankerParams(30, 1.0, True, 0)
        )
    with assert_raises():
        lambdarank_gradients(
            scores,
            labels,
            g,
            grad,
            hess,
            RankerParams.default(),
            [1.0, 1.0, 1.0],
        )


# -------------------------------------------------------------- training


def _ranking_dataset(
    n_queries: Int, mut features: List[Float64], mut labels: List[Int]
) -> List[Int]:
    """`n_queries` queries of four documents each. The single feature is the
    relevance, but the rows arrive in an order that is not sorted by it, so
    an untrained model (all scores 0, stable order) does not already rank
    them perfectly."""
    var counts = List[Int](capacity=n_queries)
    var order: List[Int] = [0, 3, 1, 2]
    for q in range(n_queries):
        for i in range(4):
            var rel = order[i]
            features.append(Float64(rel) + 0.25 * Float64(q % 2))
            labels.append(rel)
        counts.append(4)
    return counts^


def test_training_learns_a_perfect_ranking() raises:
    var features = List[Float64]()
    var labels = List[Int]()
    var counts = _ranking_dataset(6, features, labels)
    var groups = groups_from_counts(counts)
    var data = bin_equal_width(features, 24, 1, 8)

    var untrained = List[Float64]()
    untrained.resize(24, 0.0)
    assert_true(ndcg(untrained, labels, groups, 4) < 0.95)

    var booster = train_ranker(data, labels, groups, _params(60))
    assert_equal(booster.objective, LAMBDARANK)
    assert_equal(booster.base_score, 0.0)
    assert_true(len(booster.trees) > 0)

    var scores = List[Float64](capacity=24)
    for r in range(24):
        scores.append(booster.predict_row(data, r))
    assert_equal(ndcg(scores, labels, groups, 4), 1.0)
    assert_equal(ndcg(scores, labels, groups, 1), 1.0)


def test_training_is_deterministic() raises:
    var features = List[Float64]()
    var labels = List[Int]()
    var counts = _ranking_dataset(5, features, labels)
    var groups = groups_from_counts(counts)
    var data = bin_equal_width(features, 20, 1, 8)

    var a = train_ranker(data, labels, groups, _params(25))
    var b = train_ranker(data, labels, groups, _params(25))
    assert_equal(len(a.trees), len(b.trees))
    for r in range(20):
        assert_equal(a.predict_row(data, r), b.predict_row(data, r))


def test_training_validates_groups_and_labels() raises:
    var features = List[Float64]()
    var labels = List[Int]()
    var counts = _ranking_dataset(2, features, labels)
    var groups = groups_from_counts(counts)
    var data = bin_equal_width(features, 8, 1, 4)

    # Groups covering the wrong number of rows.
    with assert_raises():
        _ = train_ranker(data, labels, groups_from_counts([4, 3]), _params(3))
    # A label out of range.
    var bad = labels.copy()
    bad[0] = 31
    with assert_raises():
        _ = train_ranker(data, bad, groups, _params(3))
    # A sample_weight of the wrong length.
    with assert_raises():
        _ = train_ranker(
            data, labels, groups, _params(3), RankerParams.default(), [1.0]
        )


def test_unit_weights_match_unweighted_training() raises:
    var features = List[Float64]()
    var labels = List[Int]()
    var counts = _ranking_dataset(4, features, labels)
    var groups = groups_from_counts(counts)
    var data = bin_equal_width(features, 16, 1, 8)

    var ones = List[Float64]()
    ones.resize(16, 1.0)
    var plain = train_ranker(data, labels, groups, _params(15))
    var weighted = train_ranker(
        data, labels, groups, _params(15), RankerParams.default(), ones
    )
    for r in range(16):
        assert_equal(plain.predict_row(data, r), weighted.predict_row(data, r))


def test_early_stopping_truncates_on_group_aware_validation() raises:
    var features = List[Float64]()
    var labels = List[Int]()
    var counts = _ranking_dataset(8, features, labels)
    var groups = groups_from_counts(counts)
    var data = bin_equal_width(features, 32, 1, 8)

    # A validation set whose labels are unrelated to the feature: NDCG
    # cannot keep improving, so training must stop well short of 200 rounds.
    var valid_features = List[Float64]()
    var valid_labels = List[Int]()
    var valid_counts = List[Int]()
    var pattern: List[Int] = [2, 0, 3, 1]
    for q in range(6):
        for i in range(4):
            valid_features.append(Float64((i * 3 + q) % 4))
            valid_labels.append(pattern[(i + q) % 4])
        valid_counts.append(4)
    var valid_groups = groups_from_counts(valid_counts)
    var valid_data = bin_equal_width(valid_features, 24, 1, 8)

    var rp = RankerParams(30, 1.0, True, 4)
    var booster = train_ranker_with_valid(
        data,
        labels,
        groups,
        valid_data,
        valid_labels,
        valid_groups,
        _params(200),
        3,
        rp,
    )
    assert_true(len(booster.trees) > 0)
    assert_true(len(booster.trees) < 200)

    # The truncated ensemble is the best round: no later prefix scored
    # higher on the validation set.
    var valid_raw = List[Float64]()
    valid_raw.resize(24, 0.0)
    var best = -1.0
    for t in range(len(booster.trees)):
        for r in range(24):
            valid_raw[r] += (
                booster.learning_rate * booster.trees[t].predict_row(
                    valid_data, r
                )
            )
        var score = ndcg(valid_raw, valid_labels, valid_groups, 4)
        if score > best:
            best = score
    assert_equal(
        ndcg(valid_raw, valid_labels, valid_groups, 4),
        best,
    )


def test_early_stopping_validates_its_inputs() raises:
    var features = List[Float64]()
    var labels = List[Int]()
    var counts = _ranking_dataset(2, features, labels)
    var groups = groups_from_counts(counts)
    var data = bin_equal_width(features, 8, 1, 4)

    with assert_raises():
        _ = train_ranker_with_valid(
            data,
            labels,
            groups,
            data,
            labels,
            groups_from_counts([8]),  # right rows, but check the count
            _params(5),
            0,  # early_stopping_rounds must be positive
        )
    with assert_raises():
        _ = train_ranker_with_valid(
            data,
            labels,
            groups,
            data,
            labels,
            groups_from_counts([3, 3]),  # covers 6 of 8 valid rows
            _params(5),
            2,
        )


def test_query_bagging_never_splits_a_query() raises:
    # The bag is drawn over queries and expanded to rows, so a sampled
    # query contributes all of its rows and an unsampled one none.
    var groups = groups_from_counts([3, 1, 4, 2, 5])
    var bagging = BaggingParams(0.5, 1, 7)
    var query_bag = List[Int]()
    var row_bag = List[Int]()
    for iteration in range(4):
        _refresh_query_bag(query_bag, row_bag, groups, bagging, iteration)
        assert_true(len(query_bag) > 0)
        var expected = 0
        for i in range(len(query_bag)):
            expected += groups.size(query_bag[i])
        assert_equal(len(row_bag), expected)
        var pos = 0
        for i in range(len(query_bag)):
            var q = query_bag[i]
            for r in range(groups.start(q), groups.start(q) + groups.size(q)):
                assert_equal(row_bag[pos], r)
                pos += 1


def test_training_with_query_bagging_runs_and_is_deterministic() raises:
    var features = List[Float64]()
    var labels = List[Int]()
    var counts = _ranking_dataset(10, features, labels)
    var groups = groups_from_counts(counts)
    var data = bin_equal_width(features, 40, 1, 8)
    var bagging = BaggingParams(0.6, 1, 11)

    var a = train_ranker(
        data, labels, groups, _params(30), RankerParams.default(), [], bagging
    )
    var b = train_ranker(
        data, labels, groups, _params(30), RankerParams.default(), [], bagging
    )
    assert_equal(len(a.trees), len(b.trees))
    for r in range(40):
        assert_equal(a.predict_row(data, r), b.predict_row(data, r))

    var scores = List[Float64](capacity=40)
    for r in range(40):
        scores.append(a.predict_row(data, r))
    assert_true(ndcg(scores, labels, groups, 4) > 0.99)


def test_fit_and_serialization_roundtrip_exact() raises:
    var features = List[Float64]()
    var labels = List[Int]()
    var counts = _ranking_dataset(6, features, labels)

    var model = fit_ranker(features, 24, 1, labels, counts, _params(30))
    assert_equal(model.booster.objective, LAMBDARANK)

    var rows = List[List[Float64]]()
    var before = List[Float64]()
    for r in range(24):
        var row: List[Float64] = [features[r]]
        before.append(model.predict(row))
        rows.append(row^)

    save_model(model, _TMP_PATH)
    var loaded = load_model(_TMP_PATH)
    remove(_TMP_PATH)
    assert_equal(loaded.booster.objective, LAMBDARANK)
    for r in range(24):
        assert_equal(loaded.predict(rows[r]), before[r])

    # Predictions are raw ranking scores, so predict and predict_raw agree.
    for r in range(24):
        assert_equal(model.predict_raw(rows[r]), before[r])

    var groups = groups_from_counts(counts)
    assert_equal(ndcg(before, labels, groups, 4), 1.0)


def test_fit_ranker_rejects_bad_groups() raises:
    var features = List[Float64]()
    var labels = List[Int]()
    _ = _ranking_dataset(2, features, labels)
    with assert_raises():
        _ = fit_ranker(features, 8, 1, labels, [4, 3], _params(3))
    with assert_raises():
        _ = fit_ranker(features, 8, 1, labels, [4, 0, 4], _params(3))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
