"""CatBoost's group-and-pair ranking derivatives, and the host reference the
device kernels are held to.

Three objectives live here: **QueryRMSE**, **PairLogit**, and **YetiRank**.
All three differ from every objective in boosting.mojo in the same way: a
row's gradient is not a function of that row's own raw score and label. It is
a function of the *other rows in the row's query group*. That single fact
decides every shape in this file and in the device kernels that read it.

WHAT IS HERE AND WHAT IS NOT
----------------------------
This module imports nothing from `max.gpu.*`, so a CPU-only build can call
every function in it. That is deliberate: it is the reference the device
kernels in gpu_objectives_native.mojo are compared against, and a reference
that only exists on a machine with an accelerator is not a reference.

**The host implementations here are provisional and are labelled as such.**
The CPU campaign owns the ranking objectives proper (`lane/ranking-objectives`
and whatever lands in boosting.mojo / a ranking module beside it). When those
land, the three `*_grad_hess` functions below become one-line forwards to them
and the cross-backend test becomes one line. Until then this is the definition
the device is held to, and it was written from CatBoost's derivative
conventions rather than derived independently -- see each function for the
correspondence, stated in CatBoost's own `Der1`/`Der2` sign convention and
then converted once.

THE SIGN CONVENTION, CONVERTED ONCE
-----------------------------------
CatBoost's error functions report `Der1` and `Der2` of the quantity they
*maximize*, so `Der1 = -dL/dApprox` and `Der2 = -d2L/dApprox2`. mojotrees
carries `grad = dL/draw` and `hess = d2L/draw2` throughout (squared error is
`grad = raw - y`, `hess = 1`). So every formula below is CatBoost's with the
sign flipped exactly once, at the point of translation, and never again:

    grad = -Der1        hess = -Der2

CatBoost's `Der2` for these three is negative (`-1` for QueryRMSE,
`-w p (1-p)` for the pairwise pair), so `hess` comes out nonnegative, which is
what the Newton step and the histogram both require.

THE GROUP CONVENTION: CONTIGUOUS RUNS, AND WHY THAT IS NOT A CHOICE MADE HERE
----------------------------------------------------------------------------
**Rows of one group must be contiguous.** This is not a convention this lane
invented and it is not a third convention beside the CPU's: it is
`ranking.RankGroups`, imported below rather than restated, whose docstring
says "Query q owns rows `[starts[q], starts[q + 1])`" and whose builder
`groups_from_query_ids` *rejects* a query id whose rows are split into more
than one run ("query ids must be contiguous: rows of query N are not
consecutive"). `ranking.check_groups` is the validator, and this module calls
it rather than writing a second one.

The consequence for the device is the whole reason the question was worth
asking. A group is a **window**, so the device-side grouping plane is the
`n_groups + 1` boundary array and nothing else: no per-row `group_id` column,
no gather, no sort, no permutation. A kernel block owns group `q` and reads
`starts[q]` and `starts[q + 1]`; a row's group is implied by where the row
sits. Per-row group ids would have cost `4 * n_rows` bytes uploaded and a
gather per row in the derivative kernel, and would have bought exactly nothing
that the boundary array does not already give, because the rows really are
contiguous.

THE PAIR CONVENTION: GLOBAL ROW INDICES
---------------------------------------
A pair is `(winner, loser, weight)` with **global row indices**, not
group-local ones. That follows from the group convention rather than being a
second decision: groups are contiguous windows over the global row space, so a
global index names a row unambiguously and `check_pairs` can verify in one
comparison that both endpoints fall in the same group. Group-local indices
would have needed the group id carried alongside every pair to be
interpretable at all, which is three numbers to say what two say.

`winner` is the document that *should* rank above `loser`. That is CatBoost's
`TCompetitor` orientation: its walker iterates documents and each document's
competitor list holds the documents it beats.

WHAT THE DEVICE ACTUALLY READS
------------------------------
Not the pair list. `PairAdjacency` below expands the pair list into a
symmetric CSR -- for every row, the list of pairs it takes part in, each
tagged with whether this row was the winner. The reason is arithmetic, not
taste: a row appears in many pairs, so a one-thread-per-pair kernel would
accumulate into `grad[i]` and `grad[j]` from several threads at once, and
**Metal has no floating-point atomic add** (quantized_gradient.mojo's module
docstring is where that is established for this repository). One thread per
*row*, sweeping that row's own CSR slice in a fixed host-built order, needs no
atomic, is deterministic run to run, and reads each pair twice instead of
once. Reading a pair twice is the price and it is paid in bandwidth, which is
the cheap currency here.

WHERE THE WEIGHT GOES
---------------------
QueryRMSE takes a per-row `sample_weight`: CatBoost's `TQueryRmseError`
multiplies both `Der1` and `Der2` by `weights[docId]` and computes its query
average as a *weighted* average, and both are reproduced exactly.

PairLogit and YetiRank **refuse** a per-row `sample_weight`. Their weight is a
property of the pair, not of the document, and folding a document weight in on
top of a pair weight would apply a weight twice for a row that appears in two
pairs and once for a row that appears in one -- which is not a weighting
scheme, it is a bug with a plausible shape. The refusal names the pair weight
as the place to put it. This is a refusal and not a silent drop, on the
standing rule this repository adopted after `leaf_estimation_iterations` was
ignored without comment by every GPU entry point for months.

WHAT A RANKING ROUND COSTS THE STAGING ARM
------------------------------------------
`histogram.objective_has_constant_hessian` is a statement about an objective's
second derivative, and it admits exactly four codes (squared error, L1, huber,
quantile) and only when the fit is unweighted, because a per-row weight *is*
the hessian for those four. Ranking sits on both sides of that line and it is
worth being exact about which:

- **QueryRMSE unweighted** has `hess = 1.0` for every row at every raw score,
  which is `histogram.CONSTANT_HESSIAN` exactly. It would qualify. It is not
  declared, because `objective_has_constant_hessian` is histogram.mojo's and
  this lane does not edit it; the device path therefore stages both planes.
- **QueryRMSE weighted** has `hess = w`, so the declaration is refused for the
  same reason it is refused for squared error under weights.
- **PairLogit and YetiRank** have `hess = sum over the row's pairs of
  w p (1 - p)`, which varies per row on every round and at every raw score.
  They never qualify, weighted or not.

Priced on the Int16 gradient-staging arm exactly as the
gpu_objectives_native.mojo module docstring prices it: a constant-hessian
round stages the gradient alone, 2 bytes per row; a round that stages both
planes pays 4. At the default feature group of one, each (row, feature) visit
fetches 4 bytes of row index plus the staged derivative plus 1 bin byte, so
**every ranking round on this path is on 9 bytes per visit where an unweighted
squared-error round is on 7.** That is the arithmetic of the declaration. It
is by construction and it is not a regression in anything measured here.

WHAT THE FIXED-POINT LATTICE DOES WITH A RANKING GRADIENT
---------------------------------------------------------
Ranking gradients have a magnitude profile regression gradients do not, and
the scale rule is derived per round from that profile, so the interaction is
worth stating with arithmetic rather than discovering later.

*Every* gradient vector these three produce sums to zero **within each group**.
QueryRMSE: `sum_i w_i (avg - r_i) = avg * sum_i w_i - sum_i w_i r_i = 0` by the
definition of `avg`. Pairwise: each pair adds `-w rho` to the winner and
`+w rho` to the loser. So the total is zero and, since the rule sums
*magnitudes*, there is no cancellation for `sum|g|` to lose.

The rule is `quantized_gradient.fixed_point_scale_pow2`: with `T = sum|g|`,
`s` is the largest power of two at or below `fl(2^30 / T)`, so
`2^29 / T < s <= 2^30 / T`. Two consequences, both derived:

1. **Overflow cannot happen and is checked anyway.** The bound is on the
   total, `T * s <= 2^30`, and it holds whatever the distribution of the
   magnitudes across rows is -- concentration cannot violate a bound on a sum.
   A round reusing an older round's scale is the only way past it, and
   `histogram_gpu._check_window_bound` raises there rather than corrupting.
2. **Concentration costs resolution, and the threshold is a ratio.** A row is
   lost from the histogram when it rounds to zero, that is when
   `|g_r| * s < 1/2`. Since `s > 2^29 / T`, a row is *guaranteed* to survive
   whenever `|g_r| >= T / 2^30`: **a row is at risk only if it carries less
   than one part in 2^30 of the round's whole gradient magnitude.**

   Worked at a shape where the concentration is severe -- a million rows in a
   hundred thousand groups of ten, one group badly mis-ordered with `|g| = 1`
   per row and every other row at `|g| = 1e-8`. Then
   `T = 10 + (10^6 - 10) * 1e-8 = 10.0099999`, `2^30 / T = 1.0727e8`, so
   `s = 2^26 = 6.7109e7`, and the small rows quantize to
   `round(1e-8 * 6.7109e7) = round(0.671) = 1`. They survive, and the honest
   reading is that they **only just** survive: the guaranteed threshold here is
   `T / 2^30 = 9.32e-9` against rows at `1e-8`, a margin of 1.07 on the bound
   and 1.34 on the realized scale. One more order of magnitude of concentration
   -- the same round with the quiet rows at `1e-9` -- and every one of them
   rounds to zero and contributes nothing to any histogram.

   That is not a defect of this lane and it is not new: it is the "single
   enormous outlier" case `fixed_point_scale_pow2` already names, it pays that
   rule's full one-bit cost and nothing beyond it, and the same arithmetic
   applies to a regression round whose gradients are equally concentrated. What
   is specific to ranking is that the profile is *reached* far more often,
   because a ranker's gradient is exactly zero on every group it has already
   ordered correctly, so the mass concentrates on the shrinking set of groups
   it has not.

The case ranking reaches that regression does not is the *other* end: a
converged ranker has an exactly zero gradient on every well-ordered group, and
a dataset of singleton groups has an exactly zero gradient everywhere (see
`query_rmse_grad_hess` and `pairwise_grad_hess`). Then `T = 0`, floored to
`MAGNITUDE_FLOOR = 1e-12`, `s ~ 1.15e21`, every value quantizes to zero, the
histogram is empty and the split search finds no gain. That is the correct
answer -- there is nothing left to learn -- and it is the same terminal state
`fixed_point_scale_pow2` documents for all-zero gradients. It is reached far
more often here than under a regression objective, which is the fact worth
recording.
"""

from std.math import isfinite

from .ranking import RankGroups, _pair_sigmoid, check_groups


# ---------------------------------------------------------------------------
# Kernel selectors
# ---------------------------------------------------------------------------
#
# These are **not** `objective_registry` objective codes and must not be used
# as one. An objective code is a number in a serialized model and crosses the
# Python boundary as an integer, so claiming three of them here -- while the
# CPU campaign is concurrently landing the same three objectives and may
# number them differently -- would create two definitions of a model-facing
# number, which is the one kind of collision that cannot be fixed after a
# model has been written.
#
# What these select is which derivative kernel a ranking round runs. When the
# objective codes land in objective_registry.mojo, the bridge is one function
# (`rank_kind_for_objective`) and nothing here is renumbered.

comptime RANK_QUERY_RMSE = 0
"""CatBoost `QueryRMSE`: squared error against the group-mean-shifted
residual. Group-wise, not pairwise; needs the group boundaries and no pairs."""

comptime RANK_PAIR_LOGIT = 1
"""CatBoost `PairLogit`: logistic loss on a caller-supplied set of weighted
ordered pairs, fixed for the whole fit."""

comptime RANK_YETI_RANK = 2
"""CatBoost `YetiRank`: the same pairwise-logit derivatives as
`RANK_PAIR_LOGIT`, over pairs that are *regenerated every round* by a
stochastic sampler. The derivative arithmetic is shared, deliberately and not
by coincidence: CatBoost's `TYetiRankError` walks a competitor list and
applies exactly `TPairLogitError`'s expression to it. What is distinct about
YetiRank is entirely in where the pairs and their weights come from.

**The generator is not implemented in this build.** See
`check_yeti_rank_pairs`."""


def describe_rank_kind(kind: Int) raises -> String:
    if kind == RANK_QUERY_RMSE:
        return String("QueryRMSE")
    if kind == RANK_PAIR_LOGIT:
        return String("PairLogit")
    if kind == RANK_YETI_RANK:
        return String("YetiRank")
    raise Error("unknown ranking kind code ", kind)


@always_inline
def rank_kind_is_pairwise(kind: Int) -> Bool:
    """Whether this kind's derivatives come from a pair plane rather than
    from the group boundaries alone."""
    return kind == RANK_PAIR_LOGIT or kind == RANK_YETI_RANK


@always_inline
def rank_kind_regenerates_pairs(kind: Int) -> Bool:
    """Whether the pair plane is a property of the round rather than of the
    fit, and so must be refreshed before every round."""
    return kind == RANK_YETI_RANK


def check_rank_kind(kind: Int) raises:
    if (
        kind != RANK_QUERY_RMSE
        and kind != RANK_PAIR_LOGIT
        and kind != RANK_YETI_RANK
    ):
        raise Error(
            "unknown ranking kind code ",
            kind,
            "; expected RANK_QUERY_RMSE, RANK_PAIR_LOGIT, or RANK_YETI_RANK",
        )


def check_rank_sample_weight(kind: Int, weighted: Bool) raises:
    """Refuse a per-row `sample_weight` on a pairwise kind, by name.

    QueryRMSE takes one and applies it exactly as CatBoost does. PairLogit and
    YetiRank do not: their weight belongs to the pair. A row in two pairs and
    a row in one pair would receive the document weight twice and once
    respectively, which weights nothing anybody asked for. The module
    docstring argues it; this is where it is refused rather than dropped.
    """
    if weighted and rank_kind_is_pairwise(kind):
        raise Error(
            "objective '",
            describe_rank_kind(kind),
            "' carries its weight on the pair, not on the document; a per-row"
            " sample_weight would be applied once per pair the row appears"
            " in. Put the weight in the pair list, or train with"
            " device='cpu' under an objective that weights documents",
        )


# ---------------------------------------------------------------------------
# Pairs
# ---------------------------------------------------------------------------


@fieldwise_init
struct RankPairs(Copyable, Movable):
    """A set of weighted ordered pairs over one row matrix.

    Pair `p` says row `winner[p]` should rank above row `loser[p]`, with
    strength `weight[p]`. **Indices are global row indices**, on the argument
    the module docstring gives; both endpoints must lie in the same group,
    which `check_pairs` enforces against the group boundaries.

    `weight` is CatBoost's `TCompetitor::Weight`. It is nonnegative; a
    zero-weight pair contributes exactly nothing to any derivative and is
    dropped when the adjacency is built, which is the same answer as carrying
    it and multiplying by zero, reached without a device-side load.
    """

    var winner: List[Int]
    var loser: List[Int]
    var weight: List[Float64]

    @staticmethod
    def empty() -> RankPairs:
        return RankPairs(List[Int](), List[Int](), List[Float64]())

    def n_pairs(self) -> Int:
        return len(self.winner)


def _group_of_row(groups: RankGroups, row: Int) raises -> Int:
    """The group a row belongs to, by binary search over the boundaries.

    Host-side only, and only for validation. The device never does this: a
    block owns a group and a row's group is implied by the window it sits in
    (module docstring). Contiguity is exactly what makes the search valid.
    """
    var lo = 0
    var hi = groups.n_queries() - 1
    while lo < hi:
        var mid = (lo + hi + 1) // 2
        if groups.starts[mid] <= row:
            lo = mid
        else:
            hi = mid - 1
    return lo


def check_pairs(pairs: RankPairs, groups: RankGroups) raises:
    """Validate a pair set against the group boundaries.

    Four things are checked and each is a way a silently-accepted pair set
    would produce a plausible wrong model rather than an error: parallel array
    lengths, indices in range, a finite nonnegative weight, and **both
    endpoints in the same group**. The last is the one that matters: a pair
    that straddles two groups compares documents retrieved for two different
    queries, which is exactly the comparison every ranking objective exists to
    avoid making.

    A self-pair (`winner == loser`) is refused rather than treated as a
    no-op. Its derivative is `rho(0) = 1/2` applied in both directions, which
    cancels in the gradient and does *not* cancel in the hessian, so accepting
    it would inflate curvature for a comparison that says nothing.
    """
    var n = pairs.n_pairs()
    if len(pairs.loser) != n or len(pairs.weight) != n:
        raise Error("pair winner/loser/weight lengths must match")
    for p in range(n):
        var i = pairs.winner[p]
        var j = pairs.loser[p]
        if i < 0 or i >= groups.n_rows or j < 0 or j >= groups.n_rows:
            raise Error("pair row index out of range")
        if i == j:
            raise Error(
                "a pair must compare two different rows; row ",
                i,
                " is paired with itself"
            )
        if not isfinite(pairs.weight[p]) or pairs.weight[p] < 0.0:
            raise Error("pair weights must be finite and nonnegative")
        if _group_of_row(groups, i) != _group_of_row(groups, j):
            raise Error(
                "pair (",
                i,
                ", ",
                j,
                ") crosses a group boundary; a ranking pair compares two"
                " documents of one query",
            )


@fieldwise_init
struct PairAdjacency(Copyable, Movable):
    """The pair set, expanded per row: what the device kernels actually read.

    Row `r` owns entries `[offsets[r], offsets[r + 1])`. Entry `e` names the
    other endpoint in `other[e]` and carries the pair's weight in
    `signed_weight[e]`, **signed**: positive when `r` was the pair's winner
    and negative when it was the loser.

    Why the sign rather than a fourth plane. The direction is one bit per
    entry and the weight is strictly positive once zero-weight pairs are
    dropped, so the sign bit of the weight is free storage that is already
    being fetched. A separate `is_winner` plane would be a second device
    buffer, a second copy per round on the YetiRank path, and a second load in
    the innermost loop, to carry a bit that has nowhere else to go. Zero is
    unrepresentable in this encoding, which is why zero-weight pairs are
    dropped rather than stored -- and a dropped zero-weight pair contributes
    the same nothing it would have contributed.

    The order within a row's slice is the order the pairs were given in.
    That is a *fixed* order rather than merely a deterministic one: the
    counting-sort fill below visits pairs in input order, so the host
    reference and the device kernel accumulate the same terms in the same
    sequence, and a rerun reproduces it whatever the thread count.
    """

    var offsets: List[Int]
    var other: List[Int]
    var signed_weight: List[Float64]
    var n_rows: Int

    def n_entries(self) -> Int:
        return len(self.other)


def pair_adjacency(pairs: RankPairs, n_rows: Int) raises -> PairAdjacency:
    """Expand a pair list into the per-row symmetric CSR the kernels read.

    Two entries per surviving pair, one on each endpoint. Zero-weight pairs
    are dropped (see `PairAdjacency`). Counting sort in input order, so the
    result is a function of the input alone.
    """
    if n_rows < 1:
        raise Error("pair adjacency needs at least one row")
    var counts = List[Int](capacity=n_rows)
    counts.resize(n_rows, 0)
    var n = pairs.n_pairs()
    for p in range(n):
        if pairs.weight[p] <= 0.0:
            continue
        var i = pairs.winner[p]
        var j = pairs.loser[p]
        if i < 0 or i >= n_rows or j < 0 or j >= n_rows:
            raise Error("pair row index out of range")
        counts[i] += 1
        counts[j] += 1

    var offsets = List[Int](capacity=n_rows + 1)
    offsets.append(0)
    var total = 0
    for r in range(n_rows):
        total += counts[r]
        offsets.append(total)

    var cursor = List[Int](capacity=n_rows)
    for r in range(n_rows):
        cursor.append(offsets[r])
    var other = List[Int](capacity=total)
    other.resize(total, 0)
    var signed = List[Float64](capacity=total)
    signed.resize(total, 0.0)
    for p in range(n):
        var w = pairs.weight[p]
        if w <= 0.0:
            continue
        var i = pairs.winner[p]
        var j = pairs.loser[p]
        other[cursor[i]] = j
        signed[cursor[i]] = w
        cursor[i] += 1
        other[cursor[j]] = i
        signed[cursor[j]] = -w
        cursor[j] += 1
    return PairAdjacency(offsets^, other^, signed^, n_rows)


def check_yeti_rank_pairs(kind: Int, generated: Bool) raises:
    """Refuse a YetiRank round whose pairs were not generated for it.

    YetiRank's pairs are a property of the *round*: CatBoost redraws them
    every iteration by adding noise to the current approxes, sorting, and
    weighting the resulting inversions, so a pair set from an earlier round
    describes an ordering the model has already moved away from. Reusing one
    silently would train a model that is not YetiRank and would report nothing.

    **The generator itself is not implemented here, on purpose.** CatBoost's
    sampler (a Gumbel-style perturbation of the approxes, repeated
    `PermutationCount` times, with a decay-weighted pair count) has specifics
    -- the noise transform, the permutation count, the decay, and the seeding
    -- that this lane will not guess at, because guessing them would produce a
    plausible objective that is not the one the name promises. It belongs with
    the host objective. What this module and the device kernels provide is
    everything downstream of it: the plane the pairs and their weights travel
    in, the per-round refresh, and the derivative kernel, which is
    `RANK_PAIR_LOGIT`'s unchanged and is CatBoost's too.
    """
    if kind != RANK_YETI_RANK:
        return
    if not generated:
        raise Error(
            "YetiRank redraws its pairs every round; this round has none."
            " Supply them through refresh_pairs, or train with device='cpu'."
            " The generator (CatBoost's noise-permutation sampler) is not"
            " implemented in this build and is deliberately not guessed at"
        )


# ---------------------------------------------------------------------------
# Host reference derivatives
# ---------------------------------------------------------------------------


def query_rmse_grad_hess(
    raw: List[Float64],
    target: List[Float64],
    groups: RankGroups,
    sample_weight: List[Float64],
    mut grad: List[Float64],
    mut hess: List[Float64],
) raises:
    """CatBoost `QueryRMSE`, host reference.

    CatBoost's `TQueryRmseError` computes, per query, the weighted average
    residual and then differentiates the residual against it:

        avg    = sum_i w_i (y_i - a_i) / sum_i w_i
        Der1_i = w_i * (y_i - a_i - avg)
        Der2_i = -w_i

    Converted once through `grad = -Der1`, `hess = -Der2` (module docstring):

        r_i    = y_i - raw_i
        avg    = sum_i w_i r_i / sum_i w_i
        grad_i = w_i * (avg - r_i)
        hess_i = w_i

    with `w_i = 1` when the fit is unweighted, in which case `hess_i` is
    exactly `1.0`.

    **A singleton group produces an exactly zero gradient.** With one row,
    `avg = r_0` identically, so `grad_0 = w_0 * (r_0 - r_0) = 0` -- not
    approximately zero, and reached without any special case. That is correct
    rather than degenerate: a query with one document has no ordering to
    learn, so it contributes no signal, exactly as a pairwise objective's
    empty pair set does.

    **A group whose weights sum to zero produces zero, not NaN.** Every row of
    such a group has `w_i = 0`, so every gradient and hessian is zero whatever
    `avg` is; `avg` is set to zero rather than computed, so no division by
    zero happens. The device kernel takes the same branch, in the same place.
    """
    var n = groups.n_rows
    check_groups(groups, len(raw))
    if len(target) != n:
        raise Error("target length must equal n_rows")
    var weighted = len(sample_weight) > 0
    if weighted and len(sample_weight) != n:
        raise Error("sample_weight length must equal n_rows")

    grad.clear()
    grad.resize(n, 0.0)
    hess.clear()
    hess.resize(n, 0.0)

    for q in range(groups.n_queries()):
        var start = groups.start(q)
        var cnt = groups.size(q)
        var sum_w = 0.0
        var sum_wr = 0.0
        for i in range(start, start + cnt):
            var w = sample_weight[i] if weighted else 1.0
            sum_w += w
            sum_wr += w * (target[i] - raw[i])
        var avg = 0.0
        if sum_w > 0.0:
            avg = sum_wr / sum_w
        for i in range(start, start + cnt):
            var w = sample_weight[i] if weighted else 1.0
            grad[i] = w * (avg - (target[i] - raw[i]))
            hess[i] = w


def pairwise_grad_hess(
    raw: List[Float64],
    adjacency: PairAdjacency,
    mut grad: List[Float64],
    mut hess: List[Float64],
) raises:
    """CatBoost `PairLogit` / `YetiRank`, host reference.

    For one pair with winner `i`, loser `j` and weight `w`, CatBoost computes
    `p = Sigmoid(a_j - a_i)` -- the probability mass on the *wrong* order --
    and accumulates

        Der1_i += w * p        Der1_j -= w * p
        Der2_i += -w p (1 - p) Der2_j += -w p (1 - p)

    Converted once through `grad = -Der1`, `hess = -Der2`, and writing
    `d = raw_i - raw_j` so that `p = 1 / (1 + exp(d))`:

        grad_i -= w * rho      grad_j += w * rho
        hess_i += w rho (1-rho)  hess_j += w rho (1-rho)

    where `rho = ranking._pair_sigmoid(d, 1.0)`, **the same function
    LambdaRank's lambdas already use in this repository**, imported rather
    than restated so the branch-on-sign overflow guard has one definition.
    Both directions push the winner up and the loser down, and the pair's two
    contributions to the gradient are exact negatives, which is what makes a
    group's gradients sum to zero.

    That the two objectives share this function is CatBoost's arrangement and
    not a shortcut: `TYetiRankError` walks its generated competitor list and
    applies `TPairLogitError`'s expression to it unchanged. The two differ in
    where the pairs come from, which is `check_yeti_rank_pairs`'s subject.

    **A row in no pair gets an exactly zero gradient and an exactly zero
    hessian**, because its CSR slice is empty and the accumulators start at
    zero. Every row of a singleton group is such a row: `check_pairs` refuses
    a pair that crosses a group boundary and refuses a self-pair, so a group
    of one admits no pair at all. There is no division anywhere in this
    function and therefore nothing for a degenerate group to divide by --
    which is the failure this shape was chosen to make unreachable rather than
    to guard against.
    """
    var n = adjacency.n_rows
    if len(raw) != n:
        raise Error("raw length must equal the adjacency row count")
    grad.clear()
    grad.resize(n, 0.0)
    hess.clear()
    hess.resize(n, 0.0)

    for r in range(n):
        var g = 0.0
        var h = 0.0
        for e in range(adjacency.offsets[r], adjacency.offsets[r + 1]):
            var o = adjacency.other[e]
            var sw = adjacency.signed_weight[e]
            var w = abs(sw)
            var d = (raw[r] - raw[o]) if sw > 0.0 else (raw[o] - raw[r])
            var rho = _pair_sigmoid(d, 1.0)
            var contrib = w * rho
            if sw > 0.0:
                g -= contrib
            else:
                g += contrib
            h += w * rho * (1.0 - rho)
        grad[r] = g
        hess[r] = h


def rank_grad_hess(
    kind: Int,
    raw: List[Float64],
    target: List[Float64],
    groups: RankGroups,
    adjacency: PairAdjacency,
    sample_weight: List[Float64],
    mut grad: List[Float64],
    mut hess: List[Float64],
) raises:
    """The host reference for one ranking round, dispatched by kind.

    The single entry point a cross-backend test compares the device against,
    and the one that will forward to the CPU campaign's objectives when they
    land. It runs the same refusals the device path runs first, so a
    configuration the device declines is declined identically here rather than
    quietly producing a number the device would not have produced.
    """
    check_rank_kind(kind)
    check_rank_sample_weight(kind, len(sample_weight) > 0)
    if kind == RANK_QUERY_RMSE:
        query_rmse_grad_hess(
            raw, target, groups, sample_weight, grad, hess
        )
        return
    check_groups(groups, len(raw))
    if adjacency.n_rows != groups.n_rows:
        raise Error(
            "pair adjacency and group boundaries disagree on the row count"
        )
    pairwise_grad_hess(raw, adjacency, grad, hess)
