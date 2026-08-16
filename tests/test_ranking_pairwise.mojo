"""CatBoost's group-and-pair ranking derivatives: the host reference.

`src/mojotrees/ranking_pairwise.mojo` is the definition the device kernels in
gpu_objectives_native.mojo are held to, and it deliberately imports nothing
from `max.gpu.*` so that a CPU-only build can run it. This file is the reason
that matters: the reference is tested here, on every runner, and
`tests/test_gpu_ranking_device.mojo` compares the device against it on the
runners that have an accelerator.

What is covered:

  conventions   the group convention is `ranking.RankGroups`'s -- contiguous
                runs, a boundary array -- and the pair convention is global
                row indices inside one group. Both are asserted rather than
                described, including the refusals that make them enforceable.
  query rmse    the group-mean-shifted residual against a hand-computed
                group, weighted and unweighted
  pairwise      the pair contribution against a hand-computed logistic, and
                the antisymmetry that makes a group's gradients sum to zero
  singleton     a group of one, at both ends of the fixture and in the middle
                of a set of unequal sizes: the case where a pairwise
                derivative is degenerate and where an implementation divides
                by zero or produces a NaN if it is going to
  adjacency     the CSR expansion: symmetry, fixed order, dropped zero-weight
                pairs, and the sign encoding
  refusals      every configuration the module declines, by name

The fixture used throughout has groups of sizes **1, 4, 2, 3, 1, 5** over 16
rows: not all the same size, a singleton first, a singleton in the middle, and
a group larger than any of its neighbours.
"""

from std.math import isfinite
from std.testing import assert_equal, assert_raises, assert_true, TestSuite

from mojotrees.ranking import (
    RankGroups,
    groups_from_counts,
    groups_from_query_ids,
)
from mojotrees.ranking_pairwise import (
    RANK_PAIR_LOGIT,
    RANK_QUERY_RMSE,
    RANK_YETI_RANK,
    PairAdjacency,
    RankPairs,
    check_pairs,
    check_rank_kind,
    check_rank_sample_weight,
    check_yeti_rank_pairs,
    describe_rank_kind,
    pair_adjacency,
    pairwise_grad_hess,
    query_rmse_grad_hess,
    rank_grad_hess,
    rank_kind_is_pairwise,
    rank_kind_regenerates_pairs,
)

from support import _uniform


comptime N_ROWS = 16


def _group_sizes() -> List[Int]:
    """Unequal group sizes with a singleton at the front, a singleton in the
    middle, and sixteen rows in total. Every test in this file uses it."""
    return [1, 4, 2, 3, 1, 5]


def _fixture_groups() raises -> RankGroups:
    return groups_from_counts(_group_sizes())


def _raw() -> List[Float64]:
    var out = List[Float64](capacity=N_ROWS)
    for r in range(N_ROWS):
        out.append(-1.5 + 3.0 * _uniform(UInt64(r) * 13 + 5))
    return out^


def _target() -> List[Float64]:
    var out = List[Float64](capacity=N_ROWS)
    for r in range(N_ROWS):
        out.append(-2.0 + 4.0 * _uniform(UInt64(r) * 7 + 11))
    return out^


def _weights() -> List[Float64]:
    var out = List[Float64](capacity=N_ROWS)
    for r in range(N_ROWS):
        out.append(0.25 + 1.5 * _uniform(UInt64(r) * 17 + 3))
    return out^


def _assert_close(
    got: Float64, want: Float64, tol: Float64, what: String
) raises:
    assert_true(
        abs(got - want) <= tol,
        String(what, ": got ", got, ", want ", want),
    )


def _all_pairs_in_groups(groups: RankGroups) raises -> RankPairs:
    """Every ordered pair inside every group, lower row as the winner, with a
    weight that varies per pair so a bug that ignores the weight shows up."""
    var winner = List[Int]()
    var loser = List[Int]()
    var weight = List[Float64]()
    for q in range(groups.n_queries()):
        var start = groups.start(q)
        var cnt = groups.size(q)
        for a in range(cnt):
            for b in range(a + 1, cnt):
                winner.append(start + a)
                loser.append(start + b)
                weight.append(
                    0.5 + 2.0 * _uniform(UInt64(start + a) * 31 + UInt64(b))
                )
    return RankPairs(winner^, loser^, weight^)


# ---------------------------------------------------------------------------
# The group convention
# ---------------------------------------------------------------------------


def test_group_convention_is_contiguous_runs() raises:
    """The convention this lane matched, asserted rather than described.

    `RankGroups` is a boundary array over contiguous runs. That is what makes
    the device-side grouping plane an `n_groups + 1` array instead of a per-row
    group id, and what makes a global pair index unambiguous.
    """
    var groups = _fixture_groups()
    var sizes = _group_sizes()
    assert_equal(groups.n_queries(), len(sizes))
    assert_equal(groups.n_rows, N_ROWS)
    assert_equal(groups.start(0), 0)
    assert_equal(groups.starts[groups.n_queries()], N_ROWS)
    var expect = 0
    for q in range(groups.n_queries()):
        assert_equal(groups.start(q), expect)
        assert_equal(groups.size(q), sizes[q])
        expect += sizes[q]
    # Sizes really are unequal, with singletons at two different places.
    assert_equal(groups.size(0), 1)
    assert_equal(groups.size(4), 1)
    assert_equal(groups.max_size(), 5)


def test_noncontiguous_query_ids_are_refused() raises:
    """The CPU refuses interleaved query ids rather than splitting a query in
    two, which is precisely what licenses the window shape on the device."""
    # 0 0 1 0 -- query 0's rows are not one run.
    with assert_raises():
        _ = groups_from_query_ids([0, 0, 1, 0])
    # The same ids in one run apiece are fine.
    var ok = groups_from_query_ids([0, 0, 1, 1, 1])
    assert_equal(ok.n_queries(), 2)
    assert_equal(ok.size(0), 2)
    assert_equal(ok.size(1), 3)


# ---------------------------------------------------------------------------
# QueryRMSE
# ---------------------------------------------------------------------------


def test_query_rmse_matches_hand_computed_group() raises:
    """`grad_i = w_i (avg - r_i)`, `hess_i = w_i`, against the same arithmetic
    written out group by group."""
    var groups = _fixture_groups()
    var raw = _raw()
    var target = _target()
    var grad = List[Float64]()
    var hess = List[Float64]()
    query_rmse_grad_hess(raw, target, groups, [], grad, hess)

    assert_equal(len(grad), N_ROWS)
    for q in range(groups.n_queries()):
        var start = groups.start(q)
        var cnt = groups.size(q)
        var total = 0.0
        for i in range(start, start + cnt):
            total += target[i] - raw[i]
        var avg = total / Float64(cnt)
        for i in range(start, start + cnt):
            _assert_close(
                grad[i],
                avg - (target[i] - raw[i]),
                1e-12,
                String("query rmse grad row ", i),
            )
            _assert_close(hess[i], 1.0, 0.0, String("query rmse hess row ", i))


def test_query_rmse_weighted_uses_a_weighted_average() raises:
    """CatBoost's `CalcQueryAvrg` is weighted, and the hessian is the weight.
    A per-row weight therefore changes the *average*, not only the scale."""
    var groups = _fixture_groups()
    var raw = _raw()
    var target = _target()
    var w = _weights()
    var grad = List[Float64]()
    var hess = List[Float64]()
    query_rmse_grad_hess(raw, target, groups, w, grad, hess)

    for q in range(groups.n_queries()):
        var start = groups.start(q)
        var cnt = groups.size(q)
        var sum_w = 0.0
        var sum_wr = 0.0
        for i in range(start, start + cnt):
            sum_w += w[i]
            sum_wr += w[i] * (target[i] - raw[i])
        var avg = sum_wr / sum_w
        for i in range(start, start + cnt):
            _assert_close(
                grad[i],
                w[i] * (avg - (target[i] - raw[i])),
                1e-12,
                String("weighted query rmse grad row ", i),
            )
            _assert_close(
                hess[i], w[i], 0.0, String("weighted query rmse hess row ", i)
            )


def test_query_rmse_gradients_sum_to_zero_within_a_group() raises:
    """The property the fixed-point argument rests on: a ranking round can
    shift no group's score level, so `sum|g|` carries no cancellation."""
    var groups = _fixture_groups()
    var raw = _raw()
    var target = _target()
    var w = _weights()
    var grad = List[Float64]()
    var hess = List[Float64]()
    query_rmse_grad_hess(raw, target, groups, w, grad, hess)
    for q in range(groups.n_queries()):
        var total = 0.0
        for i in range(groups.start(q), groups.start(q) + groups.size(q)):
            total += grad[i]
        _assert_close(total, 0.0, 1e-12, String("group ", q, " gradient sum"))


def test_query_rmse_singleton_group_is_exactly_zero() raises:
    """A group of one has no ordering to learn. `avg` is that row's own
    residual, so the gradient is an exact zero rather than a small number, and
    it is reached without a special case and without a division by a count
    that could have been zero."""
    var groups = _fixture_groups()
    var raw = _raw()
    var target = _target()
    var grad = List[Float64]()
    var hess = List[Float64]()
    query_rmse_grad_hess(raw, target, groups, [], grad, hess)
    # Rows 0 and 10 are the two singleton groups (sizes 1, 4, 2, 3, 1, 5).
    assert_equal(grad[0], 0.0)
    assert_equal(grad[10], 0.0)
    assert_equal(hess[0], 1.0)
    assert_equal(hess[10], 1.0)


def test_query_rmse_zero_weight_group_is_zero_not_nan() raises:
    """Every weight in a group zero means `sum_i w_i == 0`. The average is set
    to zero rather than computed, so nothing divides by zero and the whole
    group's derivatives are exactly zero."""
    var groups = groups_from_counts([2, 2])
    var raw: List[Float64] = [0.5, -0.25, 1.0, 2.0]
    var target: List[Float64] = [1.0, 1.0, 1.0, 1.0]
    var w: List[Float64] = [0.0, 0.0, 1.0, 1.0]
    var grad = List[Float64]()
    var hess = List[Float64]()
    query_rmse_grad_hess(raw, target, groups, w, grad, hess)
    assert_equal(grad[0], 0.0)
    assert_equal(grad[1], 0.0)
    assert_equal(hess[0], 0.0)
    assert_equal(hess[1], 0.0)
    assert_true(isfinite(grad[0]) and isfinite(grad[1]))
    # The live group is unaffected by its neighbour's degeneracy.
    _assert_close(grad[2] + grad[3], 0.0, 1e-12, "live group gradient sum")
    assert_true(abs(grad[2]) > 0.0)


# ---------------------------------------------------------------------------
# Pairwise (PairLogit and YetiRank share these derivatives)
# ---------------------------------------------------------------------------


def _rho(d: Float64) -> Float64:
    """`1 / (1 + exp(d))`, written out here rather than imported, so the test
    is an independent statement of the formula."""
    from std.math import exp

    return 1.0 / (1.0 + exp(d))


def test_pairwise_matches_hand_computed_pairs() raises:
    var groups = _fixture_groups()
    var pairs = _all_pairs_in_groups(groups)
    check_pairs(pairs, groups)
    var adj = pair_adjacency(pairs, N_ROWS)
    var raw = _raw()
    var grad = List[Float64]()
    var hess = List[Float64]()
    pairwise_grad_hess(raw, adj, grad, hess)

    var want_g = List[Float64]()
    want_g.resize(N_ROWS, 0.0)
    var want_h = List[Float64]()
    want_h.resize(N_ROWS, 0.0)
    for p in range(pairs.n_pairs()):
        var i = pairs.winner[p]
        var j = pairs.loser[p]
        var w = pairs.weight[p]
        var rho = _rho(raw[i] - raw[j])
        want_g[i] -= w * rho
        want_g[j] += w * rho
        want_h[i] += w * rho * (1.0 - rho)
        want_h[j] += w * rho * (1.0 - rho)
    for r in range(N_ROWS):
        _assert_close(grad[r], want_g[r], 1e-12, String("pair grad row ", r))
        _assert_close(hess[r], want_h[r], 1e-12, String("pair hess row ", r))


def test_pairwise_gradients_sum_to_zero_within_a_group() raises:
    var groups = _fixture_groups()
    var pairs = _all_pairs_in_groups(groups)
    var adj = pair_adjacency(pairs, N_ROWS)
    var raw = _raw()
    var grad = List[Float64]()
    var hess = List[Float64]()
    pairwise_grad_hess(raw, adj, grad, hess)
    for q in range(groups.n_queries()):
        var total = 0.0
        for i in range(groups.start(q), groups.start(q) + groups.size(q)):
            total += grad[i]
        _assert_close(total, 0.0, 1e-12, String("group ", q, " gradient sum"))


def test_pairwise_singleton_group_is_exactly_zero() raises:
    """A group of one admits no pair -- `check_pairs` refuses a self-pair and
    refuses a pair that crosses a group boundary -- so its row has an empty CSR
    slice. The derivative is an exact zero and the hessian is an exact zero,
    and there is no division anywhere in the function for a degenerate group to
    reach."""
    var groups = _fixture_groups()
    var pairs = _all_pairs_in_groups(groups)
    var adj = pair_adjacency(pairs, N_ROWS)
    # Rows 0 and 10 are the singleton groups; both slices are empty.
    assert_equal(adj.offsets[1] - adj.offsets[0], 0)
    assert_equal(adj.offsets[11] - adj.offsets[10], 0)
    var raw = _raw()
    var grad = List[Float64]()
    var hess = List[Float64]()
    pairwise_grad_hess(raw, adj, grad, hess)
    assert_equal(grad[0], 0.0)
    assert_equal(hess[0], 0.0)
    assert_equal(grad[10], 0.0)
    assert_equal(hess[10], 0.0)
    assert_true(isfinite(grad[0]) and isfinite(hess[10]))


def test_pairwise_hessian_is_never_constant() raises:
    """The fact the constant-hessian refusal rests on: two rows of one round
    have different hessians, so no declaration could be made whatever the
    weights are."""
    var groups = _fixture_groups()
    var pairs = _all_pairs_in_groups(groups)
    var adj = pair_adjacency(pairs, N_ROWS)
    var raw = _raw()
    var grad = List[Float64]()
    var hess = List[Float64]()
    pairwise_grad_hess(raw, adj, grad, hess)
    var distinct = False
    for r in range(1, N_ROWS):
        if hess[r] != hess[0]:
            distinct = True
    assert_true(distinct, "pairwise hessians should vary per row")


def test_yeti_rank_shares_the_pairwise_derivatives() raises:
    """CatBoost's `TYetiRankError` applies `TPairLogitError`'s expression to
    its generated competitor list. The two kinds therefore differ only in where
    the pairs come from, and `rank_grad_hess` produces the same numbers for
    both over the same pair set."""
    var groups = _fixture_groups()
    var pairs = _all_pairs_in_groups(groups)
    var adj = pair_adjacency(pairs, N_ROWS)
    var raw = _raw()
    var target = _target()
    var g1 = List[Float64]()
    var h1 = List[Float64]()
    var g2 = List[Float64]()
    var h2 = List[Float64]()
    rank_grad_hess(
        RANK_PAIR_LOGIT, raw, target, groups, adj, [], g1, h1
    )
    rank_grad_hess(RANK_YETI_RANK, raw, target, groups, adj, [], g2, h2)
    for r in range(N_ROWS):
        assert_equal(g1[r], g2[r])
        assert_equal(h1[r], h2[r])


# ---------------------------------------------------------------------------
# The adjacency
# ---------------------------------------------------------------------------


def test_adjacency_is_symmetric_and_signed() raises:
    """Two entries per surviving pair, one on each endpoint, with the sign of
    the weight carrying which side won."""
    var groups = groups_from_counts([3])
    var pairs = RankPairs([0, 1], [1, 2], [2.0, 0.5])
    check_pairs(pairs, groups)
    var adj = pair_adjacency(pairs, 3)
    assert_equal(adj.n_entries(), 4)
    assert_equal(adj.offsets[0], 0)
    # Row 0 wins one pair; row 1 loses one and wins one; row 2 loses one.
    assert_equal(adj.offsets[1] - adj.offsets[0], 1)
    assert_equal(adj.offsets[2] - adj.offsets[1], 2)
    assert_equal(adj.offsets[3] - adj.offsets[2], 1)
    assert_equal(adj.other[adj.offsets[0]], 1)
    assert_equal(adj.signed_weight[adj.offsets[0]], 2.0)
    # Row 1's first entry is the pair it lost (input order), so negative.
    assert_equal(adj.other[adj.offsets[1]], 0)
    assert_equal(adj.signed_weight[adj.offsets[1]], -2.0)
    assert_equal(adj.other[adj.offsets[1] + 1], 2)
    assert_equal(adj.signed_weight[adj.offsets[1] + 1], 0.5)
    assert_equal(adj.signed_weight[adj.offsets[2]], -0.5)


def test_adjacency_drops_zero_weight_pairs() raises:
    """A zero-weight pair contributes exactly nothing, and zero has no sign to
    carry a direction in, so it is dropped rather than stored."""
    var groups = groups_from_counts([3])
    var pairs = RankPairs([0, 1], [1, 2], [0.0, 1.5])
    check_pairs(pairs, groups)
    var adj = pair_adjacency(pairs, 3)
    assert_equal(adj.n_entries(), 2)
    assert_equal(adj.offsets[1] - adj.offsets[0], 0)
    for e in range(adj.n_entries()):
        assert_true(adj.signed_weight[e] != 0.0)


def test_adjacency_order_is_the_input_order() raises:
    """Within a row's slice, entries appear in the order the pairs were given.
    That is a fixed order rather than merely a deterministic one, which is what
    lets the host reference and the device kernel accumulate the same terms in
    the same sequence."""
    var groups = groups_from_counts([4])
    var a = RankPairs([0, 0, 0], [1, 2, 3], [1.0, 2.0, 3.0])
    var b = RankPairs([0, 0, 0], [3, 2, 1], [3.0, 2.0, 1.0])
    var adj_a = pair_adjacency(a, 4)
    var adj_b = pair_adjacency(b, 4)
    assert_equal(adj_a.other[0], 1)
    assert_equal(adj_a.other[1], 2)
    assert_equal(adj_a.other[2], 3)
    assert_equal(adj_b.other[0], 3)
    assert_equal(adj_b.other[1], 2)
    assert_equal(adj_b.other[2], 1)
    # A rerun of the same input reproduces the same expansion exactly.
    var again = pair_adjacency(a, 4)
    for e in range(adj_a.n_entries()):
        assert_equal(again.other[e], adj_a.other[e])
        assert_equal(again.signed_weight[e], adj_a.signed_weight[e])


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_pairs_crossing_a_group_are_refused() raises:
    """The refusal that makes the global-index convention safe: a pair must
    compare two documents of one query."""
    var groups = _fixture_groups()
    # Row 0 is group 0's only row; row 1 belongs to group 1.
    with assert_raises():
        check_pairs(RankPairs([0], [1], [1.0]), groups)
    # Two rows of group 1 (rows 1..4) are fine.
    check_pairs(RankPairs([1], [2], [1.0]), groups)


def test_self_pairs_and_bad_weights_are_refused() raises:
    var groups = _fixture_groups()
    with assert_raises():
        check_pairs(RankPairs([1], [1], [1.0]), groups)
    with assert_raises():
        check_pairs(RankPairs([1], [2], [-1.0]), groups)
    with assert_raises():
        check_pairs(RankPairs([1], [99], [1.0]), groups)
    with assert_raises():
        check_pairs(RankPairs([1, 2], [2], [1.0]), groups)


def test_per_row_weights_are_refused_on_pairwise_kinds() raises:
    """PairLogit and YetiRank carry their weight on the pair. A document
    weight on top would be applied once per pair the row appears in, which
    weights nothing anybody asked for, so it is refused by name rather than
    dropped."""
    check_rank_sample_weight(RANK_QUERY_RMSE, True)
    check_rank_sample_weight(RANK_PAIR_LOGIT, False)
    with assert_raises():
        check_rank_sample_weight(RANK_PAIR_LOGIT, True)
    with assert_raises():
        check_rank_sample_weight(RANK_YETI_RANK, True)
    # And through the dispatcher, which runs the same refusals first.
    var groups = _fixture_groups()
    var adj = pair_adjacency(_all_pairs_in_groups(groups), N_ROWS)
    var grad = List[Float64]()
    var hess = List[Float64]()
    with assert_raises():
        rank_grad_hess(
            RANK_PAIR_LOGIT,
            _raw(),
            _target(),
            groups,
            adj,
            _weights(),
            grad,
            hess,
        )


def test_yeti_rank_without_generated_pairs_is_refused() raises:
    """YetiRank redraws its pairs every round. A round with none is an error,
    and the message points at `device='cpu'` and says the generator is not
    implemented rather than guessing at it."""
    check_yeti_rank_pairs(RANK_YETI_RANK, True)
    check_yeti_rank_pairs(RANK_PAIR_LOGIT, False)
    with assert_raises():
        check_yeti_rank_pairs(RANK_YETI_RANK, False)


def test_unknown_kinds_are_refused() raises:
    check_rank_kind(RANK_QUERY_RMSE)
    check_rank_kind(RANK_PAIR_LOGIT)
    check_rank_kind(RANK_YETI_RANK)
    with assert_raises():
        check_rank_kind(99)
    with assert_raises():
        _ = describe_rank_kind(-1)
    assert_equal(describe_rank_kind(RANK_QUERY_RMSE), String("QueryRMSE"))
    assert_equal(describe_rank_kind(RANK_PAIR_LOGIT), String("PairLogit"))
    assert_equal(describe_rank_kind(RANK_YETI_RANK), String("YetiRank"))
    assert_true(not rank_kind_is_pairwise(RANK_QUERY_RMSE))
    assert_true(rank_kind_is_pairwise(RANK_PAIR_LOGIT))
    assert_true(rank_kind_is_pairwise(RANK_YETI_RANK))
    assert_true(not rank_kind_regenerates_pairs(RANK_PAIR_LOGIT))
    assert_true(rank_kind_regenerates_pairs(RANK_YETI_RANK))


def test_mismatched_shapes_are_refused() raises:
    var groups = _fixture_groups()
    var raw = _raw()
    var grad = List[Float64]()
    var hess = List[Float64]()
    # Target length must equal the row count.
    with assert_raises():
        query_rmse_grad_hess(raw, [0.0, 1.0], groups, [], grad, hess)
    # Weight length must equal the row count.
    with assert_raises():
        query_rmse_grad_hess(raw, _target(), groups, [1.0], grad, hess)
    # Groups must cover the raw scores.
    with assert_raises():
        query_rmse_grad_hess(
            [0.0, 1.0], [0.0, 1.0], groups, [], grad, hess
        )
    # An adjacency built for a different row count is refused.
    var adj = pair_adjacency(_all_pairs_in_groups(groups), N_ROWS)
    with assert_raises():
        pairwise_grad_hess([0.0, 1.0], adj, grad, hess)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
