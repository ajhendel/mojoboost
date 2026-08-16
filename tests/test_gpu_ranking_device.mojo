"""CatBoost's ranking derivatives on the device: parity, degeneracy, refusals.

`tests/test_ranking_pairwise.mojo` tests the host reference and runs on every
machine. This file runs `GpuRankingState` against that reference on a machine
with an accelerator, and covers the things only the device shape can get
wrong: the block-per-group reduction, the CSR sweep, the sign encoding that
carries a pair's direction through a `bitcast`, the per-round pair refresh, and
every refusal.

Inputs are rounded through Float32 (`_f32`) before the host reference runs, so
the two backends see bit-identical inputs and a difference is the kernel's
rather than the upload's. The device carries Float64 nowhere -- Apple GPUs have
none -- so agreement is to Float32 with a relative tolerance and an absolute
floor, the same bar `tests/test_gpu_objectives_native.mojo` holds the built-in
objectives to. Where a value is an *exact* zero on both sides (a singleton
group, an empty pair slice, a zero-weight group), it is asserted as an exact
zero rather than a close one, because that is what the kernels produce and a
tolerance would hide a NaN.

The fixture has groups of sizes **1, 4, 2, 3, 1, 5** over 16 rows: unequal
sizes, a singleton first, and a singleton in the middle. A group of one is
where a pairwise derivative is degenerate and where an implementation divides
by zero or produces a NaN if it is going to.

Skips (passing) when no accelerator is present so the suite stays green on
CPU-only machines.
"""

from std.math import isfinite
from std.sys import has_accelerator
from std.testing import assert_equal, assert_raises, assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojotrees.binning import BinnedMatrix
from mojotrees.boosting import SQUARED_ERROR
from mojotrees.gpu_objectives_native import (
    GpuObjectiveState,
    GpuRankingState,
)
from mojotrees.histogram_gpu import GpuHistogramBuilder
from mojotrees.ranking import RankGroups, groups_from_counts
from mojotrees.ranking_pairwise import (
    RANK_PAIR_LOGIT,
    RANK_QUERY_RMSE,
    RANK_YETI_RANK,
    PairAdjacency,
    RankPairs,
    check_pairs,
    pair_adjacency,
    pairwise_grad_hess,
    query_rmse_grad_hess,
)

from support import _uniform


comptime N_ROWS = 16


def _f32(x: Float64) -> Float64:
    """`x` rounded to what Float32 can hold. Inputs built through this are
    identical on both backends."""
    return Float64(Float32(x))


def _assert_close(
    got: Float64, want: Float64, rel: Float64, floor: Float64, what: String
) raises:
    var diff = abs(got - want)
    var bound = floor + rel * abs(want)
    assert_true(
        diff <= bound,
        String(what, ": got ", got, ", want ", want, ", diff ", diff),
    )


def _group_sizes() -> List[Int]:
    return [1, 4, 2, 3, 1, 5]


def _fixture_groups() raises -> RankGroups:
    return groups_from_counts(_group_sizes())


def _raw() -> List[Float64]:
    var out = List[Float64](capacity=N_ROWS)
    for r in range(N_ROWS):
        out.append(_f32(-1.5 + 3.0 * _uniform(UInt64(r) * 13 + 5)))
    return out^


def _target() -> List[Float64]:
    var out = List[Float64](capacity=N_ROWS)
    for r in range(N_ROWS):
        out.append(_f32(-2.0 + 4.0 * _uniform(UInt64(r) * 7 + 11)))
    return out^


def _weights() -> List[Float64]:
    var out = List[Float64](capacity=N_ROWS)
    for r in range(N_ROWS):
        out.append(_f32(0.25 + 1.5 * _uniform(UInt64(r) * 17 + 3)))
    return out^


def _all_pairs_in_groups(groups: RankGroups) raises -> RankPairs:
    """Every ordered pair inside every group, lower row as the winner, with a
    per-pair weight so a bug that ignores the weight or its sign shows."""
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
                    _f32(
                        0.5
                        + 2.0 * _uniform(UInt64(start + a) * 31 + UInt64(b))
                    )
                )
    return RankPairs(winner^, loser^, weight^)


def _binned(n_rows: Int, n_features: Int, n_bins: Int) raises -> BinnedMatrix:
    """A binned matrix, needed only to construct a `GpuHistogramBuilder` for
    the half of the constant-hessian contract that lives on it. Its values do
    not reach any assertion here."""
    var bins = List[UInt8](capacity=n_rows * n_features)
    for f in range(n_features):
        for r in range(n_rows):
            var v = _uniform(UInt64(f * n_rows + r) + 0x0DF1A5E3)
            bins.append(UInt8(Int(v * Float64(n_bins)) % n_bins))
    return BinnedMatrix(bins^, n_rows, n_features, n_bins)


def _device_rank_grad_hess(
    ctx: DeviceContext,
    groups: RankGroups,
    target: List[Float64],
    raw: List[Float64],
    sample_weight: List[Float64],
    adjacency: PairAdjacency,
    has_pairs: Bool,
    kind: Int,
) raises -> List[Float64]:
    """One ranking round on the device, returned as `[grad | hess]`."""
    var n = len(raw)
    var state = GpuObjectiveState(ctx, target, sample_weight)
    state.set_raw(ctx, raw)
    var ranking = GpuRankingState(ctx, groups)
    if has_pairs:
        ranking.refresh_pairs(ctx, adjacency, 0)
    var grad_dev = ctx.enqueue_create_buffer[DType.float32](n)
    var hess_dev = ctx.enqueue_create_buffer[DType.float32](n)
    ranking.fill_grad_hess(ctx, state, kind, grad_dev, hess_dev, 0)
    return state.download_grad_hess(ctx, grad_dev, hess_dev)


# ---------------------------------------------------------------------------
# QueryRMSE
# ---------------------------------------------------------------------------


def test_device_query_rmse_matches_host() raises:
    """The block-per-group reduction against the host's sequential Float64
    fold, over groups of unequal size."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var ctx = DeviceContext()
        var groups = _fixture_groups()
        var raw = _raw()
        var target = _target()
        var empty = List[Float64]()
        var got = _device_rank_grad_hess(
            ctx,
            groups,
            target,
            raw,
            empty,
            pair_adjacency(RankPairs.empty(), N_ROWS),
            False,
            RANK_QUERY_RMSE,
        )
        var g = List[Float64]()
        var h = List[Float64]()
        query_rmse_grad_hess(raw, target, groups, empty, g, h)
        for r in range(N_ROWS):
            _assert_close(
                got[r], g[r], 1e-5, 1e-6, String("query rmse grad row ", r)
            )
            assert_equal(got[N_ROWS + r], 1.0)


def test_device_query_rmse_weighted_matches_host() raises:
    """The weighted average is weighted on the device too, and the hessian is
    the weight -- which is why a weighted QueryRMSE round cannot declare a
    constant hessian."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var ctx = DeviceContext()
        var groups = _fixture_groups()
        var raw = _raw()
        var target = _target()
        var w = _weights()
        var got = _device_rank_grad_hess(
            ctx,
            groups,
            target,
            raw,
            w,
            pair_adjacency(RankPairs.empty(), N_ROWS),
            False,
            RANK_QUERY_RMSE,
        )
        var g = List[Float64]()
        var h = List[Float64]()
        query_rmse_grad_hess(raw, target, groups, w, g, h)
        for r in range(N_ROWS):
            _assert_close(
                got[r],
                g[r],
                1e-5,
                1e-6,
                String("weighted query rmse grad row ", r),
            )
            _assert_close(
                got[N_ROWS + r],
                w[r],
                1e-6,
                1e-7,
                String("weighted query rmse hess row ", r),
            )


def test_device_query_rmse_singleton_group_is_exactly_zero() raises:
    """The degenerate case, on the device. A block whose group holds one row
    computes `avg = r_0` and then writes `w_0 * (r_0 - r_0)`, which is an exact
    Float32 zero -- not a small number, not a NaN -- and it gets there without
    a branch. Rows 0 and 10 are the fixture's two singleton groups."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var ctx = DeviceContext()
        var groups = _fixture_groups()
        var empty = List[Float64]()
        var got = _device_rank_grad_hess(
            ctx,
            groups,
            _target(),
            _raw(),
            empty,
            pair_adjacency(RankPairs.empty(), N_ROWS),
            False,
            RANK_QUERY_RMSE,
        )
        assert_equal(got[0], 0.0)
        assert_equal(got[10], 0.0)
        assert_true(isfinite(got[0]) and isfinite(got[10]))
        # And the non-singleton groups did not collapse with them.
        var live = False
        for r in range(1, 5):
            if abs(got[r]) > 0.0:
                live = True
        assert_true(live, "the four-row group should have a live gradient")


def test_device_query_rmse_zero_weight_group_is_zero_not_nan() raises:
    """`sum_i w_i == 0` takes the guarded branch to `avg = 0`, so the division
    never happens and the group's derivatives are exactly zero. Without the
    guard every value here would be a NaN, and a NaN reaching the magnitude
    reduction fails a whole fit."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var ctx = DeviceContext()
        var groups = groups_from_counts([2, 2])
        var raw: List[Float64] = [0.5, -0.25, 1.0, 2.0]
        var target: List[Float64] = [1.0, 1.0, 1.0, 1.0]
        var w: List[Float64] = [0.0, 0.0, 1.0, 1.0]
        var got = _device_rank_grad_hess(
            ctx,
            groups,
            target,
            raw,
            w,
            pair_adjacency(RankPairs.empty(), 4),
            False,
            RANK_QUERY_RMSE,
        )
        assert_equal(got[0], 0.0)
        assert_equal(got[1], 0.0)
        assert_equal(got[4], 0.0)
        assert_equal(got[5], 0.0)
        for i in range(8):
            assert_true(isfinite(got[i]), String("finite slot ", i))


# ---------------------------------------------------------------------------
# PairLogit and YetiRank
# ---------------------------------------------------------------------------


def test_device_pair_logit_matches_host() raises:
    """The CSR sweep against the host reference, including the sign encoding:
    a wrong sign flips a winner into a loser and the gradient changes sign, so
    this comparison is what establishes the `bitcast` round trip."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var ctx = DeviceContext()
        var groups = _fixture_groups()
        var pairs = _all_pairs_in_groups(groups)
        check_pairs(pairs, groups)
        var adj = pair_adjacency(pairs, N_ROWS)
        var raw = _raw()
        var empty = List[Float64]()
        var got = _device_rank_grad_hess(
            ctx, groups, _target(), raw, empty, adj, True, RANK_PAIR_LOGIT
        )
        var g = List[Float64]()
        var h = List[Float64]()
        pairwise_grad_hess(raw, adj, g, h)
        for r in range(N_ROWS):
            _assert_close(
                got[r], g[r], 1e-5, 1e-6, String("pair grad row ", r)
            )
            _assert_close(
                got[N_ROWS + r], h[r], 1e-5, 1e-6, String("pair hess row ", r)
            )


def test_device_yeti_rank_shares_the_pair_kernel() raises:
    """CatBoost's YetiRank walks its generated competitor list and applies
    PairLogit's expression to it. Over one pair set the two kinds must produce
    bit-identical numbers on the device, because they run the same kernel with
    the same arguments; what separates them is only where the pairs came
    from."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var ctx = DeviceContext()
        var groups = _fixture_groups()
        var adj = pair_adjacency(_all_pairs_in_groups(groups), N_ROWS)
        var raw = _raw()
        var target = _target()
        var empty = List[Float64]()
        var a = _device_rank_grad_hess(
            ctx, groups, target, raw, empty, adj, True, RANK_PAIR_LOGIT
        )
        var b = _device_rank_grad_hess(
            ctx, groups, target, raw, empty, adj, True, RANK_YETI_RANK
        )
        for i in range(2 * N_ROWS):
            assert_equal(a[i], b[i])


def test_device_pairwise_singleton_group_is_exactly_zero() raises:
    """A singleton group admits no pair, so its row's CSR slice is empty, so
    its thread stores an exact zero into both planes without executing the
    loop body once. There is no division in the pairwise kernel at all, which
    is the property this shape was chosen for rather than guarded into."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var ctx = DeviceContext()
        var groups = _fixture_groups()
        var adj = pair_adjacency(_all_pairs_in_groups(groups), N_ROWS)
        assert_equal(adj.offsets[1] - adj.offsets[0], 0)
        assert_equal(adj.offsets[11] - adj.offsets[10], 0)
        var empty = List[Float64]()
        var got = _device_rank_grad_hess(
            ctx, groups, _target(), _raw(), empty, adj, True, RANK_PAIR_LOGIT
        )
        assert_equal(got[0], 0.0)
        assert_equal(got[N_ROWS + 0], 0.0)
        assert_equal(got[10], 0.0)
        assert_equal(got[N_ROWS + 10], 0.0)
        for i in range(2 * N_ROWS):
            assert_true(isfinite(got[i]), String("finite slot ", i))


def test_device_pairwise_all_singleton_groups_is_an_empty_round() raises:
    """Every group a singleton: no pair exists anywhere, so every gradient and
    every hessian is an exact zero. That is the terminal state the module
    docstring's fixed-point paragraph names -- `sum|g| = 0` floors to
    `MAGNITUDE_FLOOR`, the histogram quantizes to all zeros, and the split
    search finds no gain -- and it is correct rather than degenerate: a query
    with one document has nothing to order."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var ctx = DeviceContext()
        var counts = List[Int]()
        for _ in range(6):
            counts.append(1)
        var groups = groups_from_counts(counts)
        var n = 6
        var raw = List[Float64]()
        var target = List[Float64]()
        for r in range(n):
            raw.append(_f32(0.25 * Float64(r) - 0.5))
            target.append(_f32(Float64(r % 3)))
        var adj = pair_adjacency(RankPairs.empty(), n)
        var empty = List[Float64]()
        var got = _device_rank_grad_hess(
            ctx, groups, target, raw, empty, adj, True, RANK_PAIR_LOGIT
        )
        for i in range(2 * n):
            assert_equal(got[i], 0.0)
        # QueryRMSE reaches the same terminal state by the other route.
        var q = _device_rank_grad_hess(
            ctx, groups, target, raw, empty, adj, False, RANK_QUERY_RMSE
        )
        for r in range(n):
            assert_equal(q[r], 0.0)
            assert_equal(q[n + r], 1.0)


def test_device_refresh_pairs_replaces_the_plane() raises:
    """The per-round upload YetiRank needs. A second refresh with a different
    pair set must produce different gradients from the same raw scores, and a
    refresh that needs more entries than the last must grow the buffers rather
    than truncate."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var ctx = DeviceContext()
        var groups = _fixture_groups()
        var raw = _raw()
        var target = _target()
        var empty = List[Float64]()
        var state = GpuObjectiveState(ctx, target, empty)
        state.set_raw(ctx, raw)
        var ranking = GpuRankingState(ctx, groups)
        var grad_dev = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
        var hess_dev = ctx.enqueue_create_buffer[DType.float32](N_ROWS)

        # Round 0: one pair inside the four-row group.
        var small = pair_adjacency(RankPairs([1], [2], [1.0]), N_ROWS)
        ranking.refresh_pairs(ctx, small, 0)
        assert_equal(ranking.n_entries, 2)
        ranking.fill_grad_hess(
            ctx, state, RANK_YETI_RANK, grad_dev, hess_dev, 0
        )
        var first = state.download_grad_hess(ctx, grad_dev, hess_dev)
        var gs = List[Float64]()
        var hs = List[Float64]()
        pairwise_grad_hess(raw, small, gs, hs)
        for r in range(N_ROWS):
            _assert_close(
                first[r], gs[r], 1e-5, 1e-6, String("round 0 grad row ", r)
            )

        # Round 1: the full pair set, which needs many more entries.
        var big = pair_adjacency(_all_pairs_in_groups(groups), N_ROWS)
        ranking.refresh_pairs(ctx, big, 1)
        assert_true(ranking.n_entries > 2)
        assert_true(ranking.pair_capacity >= ranking.n_entries)
        ranking.fill_grad_hess(
            ctx, state, RANK_YETI_RANK, grad_dev, hess_dev, 1
        )
        var second = state.download_grad_hess(ctx, grad_dev, hess_dev)
        var gb = List[Float64]()
        var hb = List[Float64]()
        pairwise_grad_hess(raw, big, gb, hb)
        var moved = False
        for r in range(N_ROWS):
            _assert_close(
                second[r], gb[r], 1e-5, 1e-6, String("round 1 grad row ", r)
            )
            if second[r] != first[r]:
                moved = True
        assert_true(moved, "a new pair plane must change the gradients")


# ---------------------------------------------------------------------------
# Non-ranking fits are untouched
# ---------------------------------------------------------------------------


def test_non_ranking_derivatives_are_bit_identical() raises:
    """How this lane knows non-ranking fits did not move.

    The structural half of the argument is that nothing was edited inside
    `_grad_hess_kernel`, `_softmax_prob_kernel`, `_softmax_class_kernel`, the
    four raw-score update kernels, `_abs_sum_kernel`, or any method of
    `GpuObjectiveState`; the ranking path is new symbols beside them. A
    structural argument can be wrong about a shared buffer or a stale plane,
    so this is the executable half: the same squared-error round is run
    before a ranking state exists, then after one has been constructed,
    uploaded a group plane, uploaded a pair plane and run both ranking
    kernels over the same context, and the two are compared **bit for bit**
    rather than to a tolerance.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var ctx = DeviceContext()
        var groups = _fixture_groups()
        var raw = _raw()
        var target = _target()
        var empty = List[Float64]()
        var state = GpuObjectiveState(ctx, target, empty)
        state.set_raw(ctx, raw)
        var grad_dev = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
        var hess_dev = ctx.enqueue_create_buffer[DType.float32](N_ROWS)

        state.fill_grad_hess(ctx, SQUARED_ERROR, 0.9, grad_dev, hess_dev)
        var before = state.download_grad_hess(ctx, grad_dev, hess_dev)
        var mags_before = state.magnitude_sums(ctx, grad_dev, hess_dev)

        var ranking = GpuRankingState(ctx, groups)
        var adj = pair_adjacency(_all_pairs_in_groups(groups), N_ROWS)
        ranking.refresh_pairs(ctx, adj, 0)
        ranking.fill_grad_hess(
            ctx, state, RANK_PAIR_LOGIT, grad_dev, hess_dev, 0
        )
        ranking.fill_grad_hess(
            ctx, state, RANK_QUERY_RMSE, grad_dev, hess_dev, 0
        )

        state.fill_grad_hess(ctx, SQUARED_ERROR, 0.9, grad_dev, hess_dev)
        var after = state.download_grad_hess(ctx, grad_dev, hess_dev)
        var mags_after = state.magnitude_sums(ctx, grad_dev, hess_dev)
        for i in range(2 * N_ROWS):
            assert_equal(before[i], after[i])
        assert_equal(mags_before.grad, mags_after.grad)
        assert_equal(mags_before.hess, mags_after.hess)

        # The raw scores the ranking kernels read were not written by them
        # either: every kernel here writes only `grad_dev` and `hess_dev`.
        var raw_back = state.download_raw(ctx)
        for r in range(N_ROWS):
            assert_equal(raw_back[r], raw[r])


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_device_ranking_refusals() raises:
    """Every configuration `GpuRankingState` declines, and each one is a
    refusal rather than a silent fallback."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var ctx = DeviceContext()
        var groups = _fixture_groups()
        var raw = _raw()
        var target = _target()
        var empty = List[Float64]()
        var grad_dev = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
        var hess_dev = ctx.enqueue_create_buffer[DType.float32](N_ROWS)

        var state = GpuObjectiveState(ctx, target, empty)
        var ranking = GpuRankingState(ctx, groups)

        # Raw scores must exist before a round can be differentiated.
        with assert_raises():
            ranking.fill_grad_hess(
                ctx, state, RANK_QUERY_RMSE, grad_dev, hess_dev, 0
            )
        state.set_raw(ctx, raw)

        # An unknown kind is not quietly treated as QueryRMSE.
        with assert_raises():
            ranking.fill_grad_hess(ctx, state, 99, grad_dev, hess_dev, 0)

        # A pairwise kind with no pair plane.
        with assert_raises():
            ranking.fill_grad_hess(
                ctx, state, RANK_PAIR_LOGIT, grad_dev, hess_dev, 0
            )
        with assert_raises():
            ranking.fill_grad_hess(
                ctx, state, RANK_YETI_RANK, grad_dev, hess_dev, 0
            )

        # A YetiRank round whose pairs were drawn for a different round.
        var adj = pair_adjacency(_all_pairs_in_groups(groups), N_ROWS)
        ranking.refresh_pairs(ctx, adj, 3)
        with assert_raises():
            ranking.fill_grad_hess(
                ctx, state, RANK_YETI_RANK, grad_dev, hess_dev, 4
            )
        # PairLogit's pairs are the fit's, so the round index does not bind it.
        ranking.fill_grad_hess(
            ctx, state, RANK_PAIR_LOGIT, grad_dev, hess_dev, 4
        )
        ranking.fill_grad_hess(
            ctx, state, RANK_YETI_RANK, grad_dev, hess_dev, 3
        )

        # A per-row sample_weight on a pairwise kind: the weight belongs on
        # the pair. QueryRMSE takes one.
        var weighted_state = GpuObjectiveState(ctx, target, _weights())
        weighted_state.set_raw(ctx, raw)
        with assert_raises():
            ranking.fill_grad_hess(
                ctx, weighted_state, RANK_PAIR_LOGIT, grad_dev, hess_dev, 0
            )
        ranking.fill_grad_hess(
            ctx, weighted_state, RANK_QUERY_RMSE, grad_dev, hess_dev, 0
        )

        # A multiclass objective state cannot serve a ranking round.
        var labels = List[Float64]()
        for r in range(N_ROWS):
            labels.append(Float64(r % 3))
        var multi = GpuObjectiveState(ctx, labels, empty, 3)
        multi.init_raw(ctx, [0.0, 0.0, 0.0])
        with assert_raises():
            ranking.fill_grad_hess(
                ctx, multi, RANK_QUERY_RMSE, grad_dev, hess_dev, 0
            )

        # A state built for a different row count.
        var short_target = List[Float64]()
        for r in range(4):
            short_target.append(target[r])
        var short_state = GpuObjectiveState(ctx, short_target, empty)
        short_state.init_raw(ctx, [0.0])
        with assert_raises():
            ranking.fill_grad_hess(
                ctx, short_state, RANK_QUERY_RMSE, grad_dev, hess_dev, 0
            )


def test_device_refresh_pairs_refuses_a_malformed_adjacency() raises:
    """A malformed adjacency does not crash a kernel, it produces a plausible
    wrong gradient, so every field is checked before anything is uploaded."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var ctx = DeviceContext()
        var groups = _fixture_groups()
        var ranking = GpuRankingState(ctx, groups)
        var good = pair_adjacency(_all_pairs_in_groups(groups), N_ROWS)

        # Wrong row count.
        with assert_raises():
            ranking.refresh_pairs(
                ctx, PairAdjacency([0, 0], [], [], 1), 0
            )
        # Offsets that do not end at the entry count.
        var bad_off = good.offsets.copy()
        bad_off[N_ROWS] = 0
        with assert_raises():
            ranking.refresh_pairs(
                ctx,
                PairAdjacency(
                    bad_off^, good.other.copy(), good.signed_weight.copy(),
                    N_ROWS
                ),
                0,
            )
        # An endpoint out of range would read another row's raw score.
        var bad_other = good.other.copy()
        bad_other[0] = N_ROWS
        with assert_raises():
            ranking.refresh_pairs(
                ctx,
                PairAdjacency(
                    good.offsets.copy(), bad_other^,
                    good.signed_weight.copy(), N_ROWS
                ),
                0,
            )
        # A zero weight has no sign to carry a direction in.
        var bad_w = good.signed_weight.copy()
        bad_w[0] = 0.0
        with assert_raises():
            ranking.refresh_pairs(
                ctx,
                PairAdjacency(
                    good.offsets.copy(), good.other.copy(), bad_w^, N_ROWS
                ),
                0,
            )
        # And the well-formed one still uploads afterwards.
        ranking.refresh_pairs(ctx, good, 0)
        assert_true(ranking.has_pairs)


def test_builder_refuses_ranking_under_constant_hessian() raises:
    """The builder's half of the contract.

    No ranking objective may be accumulated against a constant-hessian
    declaration. PairLogit's and YetiRank's hessians vary per row on every
    round; QueryRMSE's is the row weight, and even the unweighted case -- whose
    hessian really is exactly 1.0 -- is refused, because
    `histogram.objective_has_constant_hessian` is the one statement of which
    objectives qualify and this lane does not extend it. So every ranking round
    stages both derivative planes: 9 bytes per (row, feature) visit where an
    unweighted squared-error round is on 7.

    The refusal is asserted only when the declaration was actually adopted, as
    in the weight test beside it: `MOJOTREES_CONST_HESSIAN=0` refuses it
    outright and `constant_hessian` reports what was adopted.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var groups = _fixture_groups()
        var raw = _raw()
        var target = _target()
        var data = _binned(N_ROWS, 4, 16)
        var builder = GpuHistogramBuilder(data)
        var state = builder.objective_state(target)
        state.set_raw(builder.ctx, raw)
        var ranking = builder.ranking_state(groups)
        ranking.refresh_pairs(
            builder.ctx,
            pair_adjacency(_all_pairs_in_groups(groups), N_ROWS),
            0,
        )

        builder.set_constant_hessian(True)
        if builder.constant_hessian():
            with assert_raises():
                builder.fill_rank_gradients_device(
                    ranking, state, RANK_QUERY_RMSE, 0
                )
            with assert_raises():
                builder.fill_rank_gradients_device(
                    ranking, state, RANK_PAIR_LOGIT, 0
                )

        builder.set_constant_hessian(False)
        builder.fill_rank_gradients_device(ranking, state, RANK_PAIR_LOGIT, 0)
        var got = state.download_grad_hess(
            builder.ctx, builder.grad_dev, builder.hess_dev
        )
        var g = List[Float64]()
        var h = List[Float64]()
        pairwise_grad_hess(
            raw, pair_adjacency(_all_pairs_in_groups(groups), N_ROWS), g, h
        )
        for r in range(N_ROWS):
            _assert_close(
                got[r], g[r], 1e-5, 1e-6, String("builder rank grad row ", r)
            )
            _assert_close(
                got[N_ROWS + r],
                h[r],
                1e-5,
                1e-6,
                String("builder rank hess row ", r),
            )
        # The round derived a fixed-point scale from these magnitudes, by the
        # same rule and at the same cadence a regression round uses.
        assert_true(builder.g_scale > 0.0)
        assert_true(builder.h_scale > 0.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
