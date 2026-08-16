"""CatBoost's ranking objectives and eval metrics: QueryRMSE, PairLogit,
YetiRank, NDCG and PFound.

Catalog A22/A23/A24/A25. Everything here is off by default, reaches no
existing trainer, and changes no existing default. `ranking.mojo` (LambdaRank)
and `ranking_advanced.mojo` are untouched; this module reuses their
`RankGroups` and their group-id validation and adds nothing to them.

Why this module exists
----------------------
mojotrees could not be compared against CatBoost on a ranking task at all.
`lambdarank` is LightGBM's loss, and CatBoost trains none of the three losses
below; the two libraries also disagree about what `NDCG` means (see
`catboost_ndcg`). This is a bucket-C gap -- comparison validity -- as much as
a bucket-B one.

The three objectives, and their one shared fact
-----------------------------------------------
CatBoost's `TDers` holds derivatives of the LOG-LIKELIHOOD, so its `Der1` is
the negative gradient and its `Der2` is the negative hessian. Everything below
is transcribed from CatBoost and then negated exactly once, into LightGBM's
`grad = dL/df`, `hess = d2L/df2`. The negation is stated at each site. A lane
that copies `Der1` into `grad` gets a model that boosts away from the loss.

`QueryRMSE` (`error_functions.h::TQueryRmseError`)
    Squared error with the query's weighted mean residual removed:
    `grad[i] = w_i (a_i - t_i + avg)`, `hess[i] = w_i`. Its hessian is the row
    weight, so under unit weights it is the literal 1.0 and the constant-
    hessian path is admissible. It is the only one of the three for which
    that is true.

`PairLogit` (`error_functions.h::TPairLogitError`)
    Pairwise logistic over (winner, loser, weight) triples inside a group:
    `p = sigmoid(a_lose - a_win)`, `grad[win] -= c p`, `grad[lose] += c p`,
    `hess[both] += c p (1 - p)`. With `max_pairs` unset CatBoost generates
    every label-distinct pair in each group, in nested-loop order, drawing
    nothing (`pairs/util.cpp::GenerateBruteForce`).

`YetiRank` (`algo/yetirank_helpers.cpp`)
    A SAMPLED pairwise scheme. Per round, per query, draw `permutations`
    (default 10) noisy rankings of the current scores, and in each ranking
    charge every ADJACENT pair `0.15 * decay^k * |rel_i - rel_j|` to the more
    relevant member. Average the charges over the draws and hand the result to
    the `PairLogit` gradient. Nothing about the objective is a closed form.

Hessians, and the declaration that must go with them
----------------------------------------------------
`PairLogit` and `YetiRank` have a per-row hessian BY CONSTRUCTION: `c p (1-p)`
depends on the current scores of both members of every pair a row appears in,
and no setting makes it constant. `pairwise_varies_hessian` is therefore
unconditional rather than a predicate over parameters, and
`check_catboost_ranking_hessian_declaration` refuses a constant-hessian
declaration beside either of them with no escape hatch. Same machinery as
`sampling.check_mvs_hessian_declaration`; the difference is that MVS's
exclusion follows from a knob and this one follows from the loss.

Determinism
-----------
`QueryRMSE` and `PairLogit` draw nothing.

`YetiRank` draws, and CatBoost's stream cannot be copied. `UpdatePairsForYetiRank`
splits the queries into `CB_THREAD_LIMIT` blocks, seeds one generator per
block, and then walks the queries of a block SEQUENTIALLY taking one
`GenRand()` each; inside a query the per-document uniforms are one sequential
run shared by all ten permutations. Query `q`'s draws therefore depend on how
many queries precede it inside its block. That is deterministic in CatBoost
only because `CB_THREAD_LIMIT` is a compile-time constant, and it is a
sequential stream, which this repository forbids (see the `_MVS_DOMAIN` and
`_LANGEVIN_ROW_DOMAIN` notes for the same argument twice already).

`_yetirank_stream` keys on `(seed, iteration, query, permutation)` and
document `d` reads `stream + d`. It is a START, not a running state: nothing
advances, no permutation shares a draw with another, and no draw depends on
how many queries or documents were handled first. The pair weights are
therefore identical at every `MOJOTREES_NUM_WORKERS` and on every machine.
The numbers differ from CatBoost's; the distribution does not, and
bit-identity was never on offer because their stream has no name.

Deliberate divergences, all in the accuracy direction
-----------------------------------------------------
- The pair probability is `sigmoid(a_lose - a_win)` through a sign branch,
  not `e^{a_lose} / (e^{a_lose} + e^{a_win})` from stored exponentials.
  CatBoost's form is `inf/inf = NaN` once an approx passes about 709. Same
  value wherever theirs is finite, defined where theirs is not, and one `exp`
  instead of two plus a division.
- The YetiRank noise is applied on the SCORE scale as
  `a + log(u) - log(1.000001 - u)` rather than as a multiplication of
  `exp(a)`. Identical ordering, no overflow. CatBoost also computes its `u`
  and its `1.000001f` in single precision; this is Float64 throughout.
- The pair weights are accumulated in a flat sorted list, not in a dense
  `querySize x querySize` matrix that is then scanned in full. The sort key
  `(winner, loser)` reproduces CatBoost's emission order exactly, so the pair
  SEQUENCE handed to the gradient is theirs; only the per-cell summation is
  reassociated, which moves the last ulp.

Not implemented, and refused rather than ignored
------------------------------------------------
- `YetiRank` `mode` in {DCG, NDCG, MRR, ERR, MAP}, `noise=Gauss`,
  `num_neighbors != 1`. `Classic` + `Gumbel` + 1 is the constructed default
  and is the whole of what a default `loss_function=YetiRank` run does.
- `YetiRankPairwise`, which shares the pair generator but trains through
  CatBoost's pairwise SCORING path rather than through leafwise derivatives.
- `PairLogit` `max_pairs`. It shuffles with an unnameable generator and its
  branch condition is arithmetically wrong in CatBoost (see the catalog).
- `PFound` `subgroup_id` deduplication. There is no subgroup column here.
"""

from std.math import exp, log, log2

from .binning import BinnedMatrix
from .boosting import Booster, BoosterParams
from .ranking import RankGroups, check_groups, groups_from_query_ids
from .rng import GOLDEN, splitmix64, uniform
from .tree import Tree, grow_tree


# ---------------------------------------------------------------------------
# Objective and metric codes
# ---------------------------------------------------------------------------

# The three objective codes, and the two metric codes, continuing
# `objective_registry`'s numbering (built-ins 0..12, `MULTICLASS` -1; metrics
# 0..20). They are spelled HERE rather than there because
# `objective_registry.mojo` is shared glue this lane does not own: the
# registry additions are delivered as a diff. This is the same two-places
# arrangement `ranking.LAMBDARANK` lived under, and it resolves the same way
# -- when the registry diff lands, these five lines become
# `from .objective_registry import ... as _...` bindings and the numbers have
# one definition again.
#
# A code is a number in a serialized model and crosses the Python boundary as
# an integer, so once assigned none of these may be renumbered.
comptime QUERY_RMSE = 13
comptime PAIR_LOGIT = 14
comptime YETI_RANK = 15

comptime METRIC_PFOUND = 21
comptime METRIC_NDCG_CATBOOST = 22


# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

# Separates YetiRank's per-(query, permutation) stream from every other
# counter stream in the package. Bagging, feature sampling, GOSS, MVS, the
# Bayesian bootstrap and Langevin are each keyed by a small seed and a small
# index and each default their seed to 0, so a caller who sets one seed and
# turns two of them on would draw the same uniforms twice without a domain
# separator. `sampling._MVS_DOMAIN` and `langevin._LANGEVIN_ROW_DOMAIN` are
# the same device for the same reason.
comptime _YETIRANK_DOMAIN = UInt64(0x59657469_52616E6B)

# The width of one query's document substream, so query `q`'s draws for
# permutation `p` can never reach query `q`'s draws for permutation `p + 1`.
# A query longer than this raises rather than silently colliding: a stream
# that overlaps its neighbour is a correlation bug no single-query test can
# see. 2^24 documents in one query is four orders of magnitude past anything
# a ranking dataset carries.
comptime YETIRANK_MAX_QUERY_SIZE = 1 << 24


# ---------------------------------------------------------------------------
# Defaults, verified from CatBoost source
# ---------------------------------------------------------------------------

# `loss_description.cpp::GetYetiRankPermutations`:
# `GetParamOrDefault(lossFunctionConfig, "permutations", 10)`.
comptime DEFAULT_YETIRANK_PERMUTATIONS = 10

# `loss_description.cpp::GetYetiRankDecay`:
# `GetParamOrDefault(lossFunctionConfig, "decay", 0.85)`.
#
# NOT 0.99. `TYetiRankPairWeightsCalcer::TConfig` carries `double Decay = 0.99`
# as a field initializer and its constructor overwrites it unconditionally on
# the next line with this value. Reading 0.99 out of the header and calling it
# CatBoost's default is wrong by a factor that compounds down the ranking.
comptime DEFAULT_YETIRANK_DECAY = 0.85

# `TYetiRankPairWeightsCalcer::CalcWeightsClassic`: `const double magicConst =
# 0.15; // Like in GPU`. It scales every pair weight inside a query uniformly,
# so it is absorbed by the learning rate up to its interaction with
# `reg_lambda`; carried across because that interaction is real.
comptime YETIRANK_MAGIC_CONST = 0.15

# `TYetiRankPairWeightsCalcer::AddNoise`, Gumbel arm: `expApproxes[docId] *=
# uniformValue / (1.000001f - uniformValue)`. The guard keeps the denominator
# positive at `u` arbitrarily close to 1.
comptime YETIRANK_NOISE_GUARD = 1.000001

# No CatBoost equivalent: their seed is one draw from the unnamed global
# learn-progress generator. Ours is a parameter, because a stream that cannot
# be named cannot be reproduced.
comptime DEFAULT_YETIRANK_SEED = 0

# `metric.cpp::TPFoundMetric`: `DefaultDecay = 0.85`, `DefaultTopSize = -1`.
# The -1 is handed to a `ui32` parameter, so it becomes `Max<ui32>`: no
# truncation. `TOP_ALL` carries that meaning without the sign trick.
comptime DEFAULT_PFOUND_DECAY = 0.85
comptime TOP_ALL = -1

# `metric.cpp::TDcgMetric`: `DefaultMetricType = ENdcgMetricType::Base`,
# `DefaultDenominatorType = ENdcgDenominatorType::LogPosition`,
# `DefaultTopSize = -1`.
#
# `Base` is the numerator `rel` itself. LightGBM's NDCG -- and therefore
# `ranking.ndcg` -- uses `2^rel - 1`, which is CatBoost's `Exp`. Scoring a
# graded relevance set under the wrong one is not a rescaling; it reorders
# which of two models is better.
comptime NDCG_TYPE_BASE = 0
comptime NDCG_TYPE_EXP = 1

# `log(2)`, for the non-integral fallback in `_dcg_numerator`.
comptime _LN2 = 0.6931471805599453
comptime NDCG_DENOMINATOR_LOG_POSITION = 0
comptime NDCG_DENOMINATOR_POSITION = 1

# `yetirank_helpers.cpp` noise arms. `Gauss` is not implemented; `No` is, so a
# caller can measure what the sampling is worth on their own data.
comptime YETIRANK_NOISE_NONE = 0
comptime YETIRANK_NOISE_GUMBEL = 1


# ---------------------------------------------------------------------------
# Group id: the contract, verified from source
# ---------------------------------------------------------------------------


def groups_from_group_id(group_id: List[Int]) raises -> RankGroups:
    """Query boundaries from CatBoost's `group_id` column.

    `catboost/libs/data/objects.cpp::CheckGroupIds` is the whole contract and
    it says exactly three things:

    - Rows of one group must be **contiguous**. CatBoost walks the column
      once collecting the id of each maximal run, sorts the run ids and calls
      `std::adjacent_find`; a repeat means one group's rows were interleaved
      with another's, and it raises `"group Ids are not consecutive"`.
    - Groups need **not** be sorted, and the ids need not be dense, ordered
      or numeric -- CatBoost hashes string ids through `CalcGroupIdFor`. Only
      the runs matter.
    - The failure is **loud**. CatBoost does not regroup, does not sort and
      does not drop, because a ranking objective reading a mis-grouped column
      produces a plausible model and a meaningless one, and nothing
      downstream can tell.

    `ranking.groups_from_query_ids` was built against LightGBM and implements
    that rule line for line, including the sort-and-adjacent-find. It is
    reused unchanged; this is the name a CatBoost user reaches for, and it
    exists so that no second implementation of the check appears.
    """
    return groups_from_query_ids(group_id)


def check_group_weights_constant(
    weights: List[Float64], groups: RankGroups
) raises:
    """Refuse a row-weight vector that varies inside a group.

    `catboost/libs/data/target.cpp::CheckGroupWeights` requires the weight to
    be `FuzzyEquals`-constant across a group's objects: a group weight is a
    property of the GROUP, and CatBoost reads it off the group's first row
    (`(**weights)[group.Begin]`) everywhere it is used -- pair generation,
    YetiRank's `queryWeight`, and both eval metrics. A vector that varies
    inside a group would therefore be silently read as its first row's value
    by some code and averaged by other code.

    An empty vector means unit weights and passes. Comparison is exact rather
    than fuzzy: a caller who built the vector by broadcasting a per-group
    value has bit-identical entries, and a caller who did not should be told.
    """
    if len(weights) == 0:
        return
    if len(weights) != groups.n_rows:
        raise Error(
            "sample_weight length must equal n_rows for a grouped objective"
        )
    for q in range(groups.n_queries()):
        var start = groups.start(q)
        var w0 = weights[start]
        for r in range(start + 1, groups.starts[q + 1]):
            if weights[r] != w0:
                raise Error(
                    "sample_weight must be constant inside a group: group ",
                    q,
                    " has ",
                    w0,
                    " at row ",
                    start,
                    " and ",
                    weights[r],
                    " at row ",
                    r,
                )


def group_weight(weights: List[Float64], groups: RankGroups, q: Int) -> Float64:
    """The weight of query `q`, read off its first row as CatBoost does.

    Unit when the vector is empty. `check_group_weights_constant` is what
    makes "its first row" the same as "the group's", and a caller that skips
    that check gets CatBoost's own behavior rather than an average.
    """
    if len(weights) == 0:
        return 1.0
    return weights[groups.start(q)]


# ---------------------------------------------------------------------------
# Hessian declarations
# ---------------------------------------------------------------------------


def pairwise_varies_hessian(objective: Int) -> Bool:
    """Whether this objective's hessian is per row, so that
    `histogram.CONSTANT_HESSIAN` must not be declared for a fit of it.

    True for `PAIR_LOGIT` and `YETI_RANK`, **unconditionally and with no
    parameter consulted**. Their hessian is `c p (1 - p)` summed over the
    pairs a row appears in, where `p` is a sigmoid of the difference of two
    current scores; it is not constant at any setting, on any data, in any
    round. This is a stronger statement than `sampling.mvs_varies_hessian`
    can make, which is why there is no knob in the signature to be wrong
    about.

    False for `QUERY_RMSE`, whose hessian is the row weight -- the literal
    1.0 under unit weights, exactly as for squared error. Use
    `query_rmse_varies_hessian` for the weighted case; the weight rule is the
    one squared error already lives under and is not this objective's
    business.

    Any other code returns False, because this function answers only for the
    three codes it owns.
    """
    return objective == PAIR_LOGIT or objective == YETI_RANK


def query_rmse_varies_hessian(sample_weight: List[Float64]) -> Bool:
    """Whether a `QUERY_RMSE` fit has a per-row hessian.

    `TQueryRmseError` sets `Der2 = -1` and then multiplies by `weights[docId]`
    when the weight vector is non-empty, so the hessian is exactly the row
    weight. Non-empty vector, per-row hessian; empty vector, the constant
    1.0. Deliberately keyed on emptiness rather than on whether the weights
    happen to be all 1: the declaration is held for a whole fit and a safety
    predicate that inspects `Float64` values is one edit from being wrong
    (the argument `sampling.mvs_varies_hessian` makes at length).
    """
    return len(sample_weight) > 0


def check_catboost_ranking_hessian_declaration(
    objective: Int, const_hessian: Bool, sample_weight: List[Float64] = []
) raises:
    """Refuse a constant-hessian declaration beside a CatBoost ranking
    objective that does not have one.

    The twin of `sampling.check_mvs_hessian_declaration` and
    `langevin.check_langevin_hessian_declaration`, and it exists for the same
    structural reason: `boosting.round_has_constant_hessian` makes the
    declaration from three inputs -- the objective, the sample weights and the
    GOSS parameters -- and its signature is a GPU-visible contract no CPU lane
    may widen. So the guard lives beside the objective, costs one branch per
    fit, and converts a silently wrong hessian plane into an exception at fit
    setup.

    Declaring a constant hessian beside a pairwise gradient is not a slow
    path or an approximation. The histogram builder rebuilds the hessian
    plane from the row COUNT, so every leaf denominator becomes the number of
    rows instead of the summed curvature, and the fit returns a plausible
    model that optimized nothing. It is the single worst failure mode this
    module can have.
    """
    if not const_hessian:
        return
    if pairwise_varies_hessian(objective):
        raise Error(
            "a fit under the '",
            catboost_ranking_name(objective),
            "' objective must not declare a constant hessian: its hessian is"
            " c*p*(1-p) summed over the pairs each row appears in, which is"
            " per row and per round by construction",
        )
    if objective == QUERY_RMSE and query_rmse_varies_hessian(sample_weight):
        raise Error(
            "a weighted query_rmse fit must not declare a constant hessian:"
            " CatBoost's TQueryRmseError multiplies Der2 by the row weight, so"
            " the hessian is the weight"
        )


def catboost_ranking_name(objective: Int) raises -> String:
    """CatBoost's own spelling of one of the three codes, lowercased as this
    package spells objective names.

    `yetirank` is deliberately NOT an alias of `lambdarank` and never becomes
    one. LambdaRank weights a pair by the NDCG change swapping it would cause
    on the deterministic current ranking; YetiRank weights a pair by how often
    it lands adjacent in a NOISY ranking, with a positional decay and a
    relevance-difference factor, and computes no NDCG anywhere. They agree
    only in being pairwise, and an alias would make a comparison table claim
    two libraries ran the same loss when they did not.
    """
    if objective == QUERY_RMSE:
        return String("query_rmse")
    if objective == PAIR_LOGIT:
        return String("pair_logit")
    if objective == YETI_RANK:
        return String("yetirank")
    raise Error("not a CatBoost ranking objective code: ", objective)


def catboost_ranking_code_from_name(name: String) raises -> Int:
    """One of the three codes from CatBoost's name or an accepted alias.

    CatBoost spells them `QueryRMSE`, `PairLogit` and `YetiRank`; the names
    here are those lowercased, plus the undivided spellings a user is likely
    to type. `queryrmse` and `query_rmse` both resolve; `lambdarank` does not
    resolve to `yetirank` and must not be made to.
    """
    if name == "query_rmse" or name == "queryrmse":
        return QUERY_RMSE
    if name == "pair_logit" or name == "pairlogit":
        return PAIR_LOGIT
    if name == "yetirank" or name == "yeti_rank":
        return YETI_RANK
    raise Error(
        "unknown CatBoost ranking objective '",
        name,
        "'; expected query_rmse, pair_logit, or yetirank",
    )


def is_catboost_ranking_objective(objective: Int) -> Bool:
    """Whether a code names one of the three. The predicate a trainer or a
    device provider asks before assuming a row-wise closed form exists."""
    return (
        objective == QUERY_RMSE
        or objective == PAIR_LOGIT
        or objective == YETI_RANK
    )


# ---------------------------------------------------------------------------
# Stable sorts
# ---------------------------------------------------------------------------
#
# All three are bottom-up merge sorts on an index permutation, the shape
# `ranking._argsort_desc_range` already uses. They are stable, which is not a
# convenience here: CatBoost's `StableSort` is what decides the ranking of
# documents whose keys tie, and the tie rule is part of every metric below.


def _stable_argsort_desc(values: List[Float64], cnt: Int) -> List[Int]:
    """Local indices `0..cnt-1` ordered by `values[i]` DESCENDING, stable.

    `GenerateYetiRankPairsForQuery`'s
    `StableSort(indices, [](i, j){ return approx[i] > approx[j]; })`.
    """
    var idx = List[Int](capacity=cnt)
    for i in range(cnt):
        idx.append(i)
    var buf = List[Int](capacity=cnt)
    for _ in range(cnt):
        buf.append(0)
    var width = 1
    while width < cnt:
        var lo = 0
        while lo < cnt:
            var mid = lo + width
            if mid > cnt:
                mid = cnt
            var hi = lo + 2 * width
            if hi > cnt:
                hi = cnt
            var i = lo
            var j = mid
            var k = lo
            while i < mid and j < hi:
                if values[idx[i]] >= values[idx[j]]:
                    buf[k] = idx[i]
                    i += 1
                else:
                    buf[k] = idx[j]
                    j += 1
                k += 1
            while i < mid:
                buf[k] = idx[i]
                i += 1
                k += 1
            while j < hi:
                buf[k] = idx[j]
                j += 1
                k += 1
            lo += 2 * width
        for t in range(cnt):
            idx[t] = buf[t]
        width *= 2
    return idx^


def _compare_docs(
    approx_l: Float64, target_l: Float64, approx_r: Float64, target_r: Float64
) -> Bool:
    """`catboost/libs/metrics/doc_comparator.h::CompareDocs`, verbatim:

        approxLeft != approxRight ? approxLeft > approxRight
                                  : targetLeft < targetRight

    On an exact score tie the LOWER relevance ranks FIRST. That is
    adversarial on purpose: it is what makes an untrained model score near
    its floor rather than at whatever the input order happened to give.
    LightGBM stable-sorts and keeps input order, which flatters a model on any
    dataset that arrives sorted by relevance -- and ranking datasets
    frequently do. Both mojotrees ranking metric families therefore need their
    own sort, and `ranking.ndcg` must not be reused for a CatBoost comparison.
    """
    if approx_l != approx_r:
        return approx_l > approx_r
    return target_l < target_r


def _argsort_docs(
    scores: List[Float64], targets: List[Float64], start: Int, cnt: Int
) -> List[Int]:
    """Local indices `0..cnt-1` of one query ordered by `_compare_docs`,
    stable.

    CatBoost's `GetTopSortedTargets` uses `PartialSort` with an explicit
    `lhs < rhs` index tiebreak when `top < size` and `StableSort` otherwise.
    Both are the SAME total order -- `CompareDocs`, then input index -- so one
    stable sort followed by a truncation is exact, and is one code path
    instead of two.
    """
    var idx = List[Int](capacity=cnt)
    for i in range(cnt):
        idx.append(i)
    var buf = List[Int](capacity=cnt)
    for _ in range(cnt):
        buf.append(0)
    var width = 1
    while width < cnt:
        var lo = 0
        while lo < cnt:
            var mid = lo + width
            if mid > cnt:
                mid = cnt
            var hi = lo + 2 * width
            if hi > cnt:
                hi = cnt
            var i = lo
            var j = mid
            var k = lo
            while i < mid and j < hi:
                # `not compare(right, left)` keeps the left run first on a
                # tie, which is what makes the merge stable.
                var a = idx[i]
                var b = idx[j]
                if not _compare_docs(
                    scores[start + b],
                    targets[start + b],
                    scores[start + a],
                    targets[start + a],
                ):
                    buf[k] = a
                    i += 1
                else:
                    buf[k] = b
                    j += 1
                k += 1
            while i < mid:
                buf[k] = idx[i]
                i += 1
                k += 1
            while j < hi:
                buf[k] = idx[j]
                j += 1
                k += 1
            lo += 2 * width
        for t in range(cnt):
            idx[t] = buf[t]
        width *= 2
    return idx^


def _stable_argsort_keys(keys: List[Int]) -> List[Int]:
    """Indices of `keys` in ascending key order, stable.

    Used to put YetiRank's charged pairs into CatBoost's emission order.
    Stability is what makes the per-cell summation order the permutation
    order, which is what makes the sum reproducible.
    """
    var cnt = len(keys)
    var idx = List[Int](capacity=cnt)
    for i in range(cnt):
        idx.append(i)
    var buf = List[Int](capacity=cnt)
    for _ in range(cnt):
        buf.append(0)
    var width = 1
    while width < cnt:
        var lo = 0
        while lo < cnt:
            var mid = lo + width
            if mid > cnt:
                mid = cnt
            var hi = lo + 2 * width
            if hi > cnt:
                hi = cnt
            var i = lo
            var j = mid
            var k = lo
            while i < mid and j < hi:
                if keys[idx[i]] <= keys[idx[j]]:
                    buf[k] = idx[i]
                    i += 1
                else:
                    buf[k] = idx[j]
                    j += 1
                k += 1
            while i < mid:
                buf[k] = idx[i]
                i += 1
                k += 1
            while j < hi:
                buf[k] = idx[j]
                j += 1
                k += 1
            lo += 2 * width
        for t in range(cnt):
            idx[t] = buf[t]
        width *= 2
    return idx^


# ---------------------------------------------------------------------------
# A22. QueryRMSE
# ---------------------------------------------------------------------------


def query_rmse_gradients(
    scores: List[Float64],
    targets: List[Float64],
    groups: RankGroups,
    mut grad: List[Float64],
    mut hess: List[Float64],
    sample_weight: List[Float64] = [],
) raises:
    """`TQueryRmseError::CalcDersForQueries`, negated into this package's
    convention.

    Per query, with `w = 1` when the weight vector is empty:

        avg     = sum_j w_j (t_j - a_j) / sum_j w_j    (0 if the sum is <= 0)
        grad[i] = w_i (a_i - t_i + avg)
        hess[i] = w_i

    CatBoost writes `Der1 = w (t - a - avg)` and `Der2 = -w`; both are
    negated here exactly once. `CalcQueryAvrg` accumulates `querySum` and
    `queryCount` in document order and guards the division on
    `queryCount > 0`, which is transcribed rather than replaced by a
    tolerance.

    `grad` and `hess` are overwritten with one entry per row. A row's
    gradient depends only on its own query, so the result is independent of
    any partition of the queries across workers.
    """
    check_groups(groups, len(scores))
    if len(targets) != len(scores):
        raise Error("targets length must equal scores length")
    if len(sample_weight) != 0 and len(sample_weight) != len(scores):
        raise Error("sample_weight length must equal n_rows")
    var n = len(scores)
    grad.clear()
    hess.clear()
    grad.resize(n, 0.0)
    hess.resize(n, 0.0)
    var weighted = len(sample_weight) > 0
    for q in range(groups.n_queries()):
        var start = groups.start(q)
        var end = groups.starts[q + 1]
        var query_sum = 0.0
        var query_count = 0.0
        for r in range(start, end):
            var w = sample_weight[r] if weighted else 1.0
            query_sum += (targets[r] - scores[r]) * w
            query_count += w
        var avg = 0.0
        if query_count > 0.0:
            avg = query_sum / query_count
        for r in range(start, end):
            var w = sample_weight[r] if weighted else 1.0
            grad[r] = w * (scores[r] - targets[r] + avg)
            hess[r] = w


# ---------------------------------------------------------------------------
# A23. PairLogit, and its pairs
# ---------------------------------------------------------------------------


@fieldwise_init
struct RankPair(Copyable, Movable):
    """One (winner, loser, weight) triple, CatBoost's `TCompetitor` flattened.

    `winner` and `loser` are ABSOLUTE row indices, not indices within a
    query, because the gradient writes into a full-length buffer and a pair
    can never span two queries anyway. `weight` is the group weight for
    generated `PairLogit` pairs and the averaged YetiRank charge for
    `YetiRank` pairs.
    """

    var winner: Int
    var loser: Int
    var weight: Float64


def generate_pair_logit_pairs(
    targets: List[Float64],
    groups: RankGroups,
    sample_weight: List[Float64] = [],
) raises -> List[RankPair]:
    """CatBoost's automatic pair generation with `max_pairs` unset:
    `pairs/util.cpp::GeneratePairLogitPairs` taking the `GenerateBruteForce`
    branch with truncation disabled.

    Every ordered index pair `(first, second)` with `first < second` inside a
    group and `targets[first] != targets[second]`, the higher target as
    winner, weight = the group weight, emitted in nested-loop order. **No RNG
    is consulted**, so the pair list is a pure function of the labels and the
    grouping.

    `data_providers.cpp::GeneratePairs` refuses a constant target column
    ("Target data is constant. Cannot generate pairs.") because it would
    produce no pairs at all; that check is kept, and phrased against the whole
    column as CatBoost phrases it. A single group that happens to be constant
    is legal there and here -- it simply contributes nothing.

    `max_pairs` is not implemented. It shuffles with `TRestorableFastRng64`,
    drawn from CatBoost's unnamed global learn-progress generator, and its
    branch condition (`pairCount / 2 < maxPairCount`, on a `pairCount` that
    was halved on the previous line) is arithmetically wrong. Reproducing a
    bug we cannot seed is worth nothing.
    """
    check_groups(groups, len(targets))
    check_group_weights_constant(sample_weight, groups)
    var lo = targets[0]
    var hi = targets[0]
    for r in range(1, len(targets)):
        if targets[r] < lo:
            lo = targets[r]
        if targets[r] > hi:
            hi = targets[r]
    if lo == hi:
        raise Error(
            "target data is constant; pair_logit cannot generate pairs from it"
        )
    var pairs = List[RankPair]()
    for q in range(groups.n_queries()):
        var start = groups.start(q)
        var end = groups.starts[q + 1]
        var w = group_weight(sample_weight, groups, q)
        for first in range(start, end):
            for second in range(first + 1, end):
                if targets[first] == targets[second]:
                    continue
                if targets[first] > targets[second]:
                    pairs.append(RankPair(first, second, w))
                else:
                    pairs.append(RankPair(second, first, w))
    return pairs^


def _logistic(x: Float64) -> Float64:
    """`1 / (1 + exp(-x))`, finite for every finite `x`.

    The branch is what makes it finite: `exp(-x)` overflows below about -709
    and `exp(x)` above about 709, and exactly one of the two arms is
    evaluated on the safe side. CatBoost instead forms
    `e^{a_lose} / (e^{a_lose} + e^{a_win})` from stored exponentials, which is
    `inf / inf = NaN` once either approx passes 709. Same value wherever
    theirs is finite; one `exp` and one divide instead of two `exp`, one add
    and one divide.
    """
    if x >= 0.0:
        return 1.0 / (1.0 + exp(-x))
    var e = exp(x)
    return e / (1.0 + e)


def pair_logit_gradients(
    scores: List[Float64],
    pairs: List[RankPair],
    mut grad: List[Float64],
    mut hess: List[Float64],
) raises:
    """`TPairLogitError::CalcDersForQueries`, negated into this package's
    convention.

    For each `(winner i, loser j, weight c)`:

        p        = sigmoid(a_j - a_i)
        grad[i] -= c p          grad[j] += c p
        hess[i] += c p (1 - p)  hess[j] += c p (1 - p)

    CatBoost writes `Der1[i] += c p`, `Der1[j] -= c p` and
    `Der2[both] += c p (p - 1)` with `p = expApprox[j] / (expApprox[j] +
    expApprox[i])`; `IsStoreExpApprox` lists `PairLogit`, so `expApprox` is
    `exp(a)` and the ratio is `sigmoid(a_j - a_i)`. Both derivatives are
    negated here exactly once.

    `grad` and `hess` are overwritten with one entry per row. Rows in no pair
    keep 0 in both, which is correct -- they contribute nothing to the loss --
    and is why a fit under this objective needs a positive `min_child_weight`
    or a positive `reg_lambda` to be sure no leaf divides zero by zero.
    `train_catboost_ranker` checks the total.

    Accumulation is in pair order, once, with no partial sums, so the result
    does not move with the worker count. Pair order is the caller's; both
    producers in this module hand over CatBoost's own order.
    """
    var n = len(scores)
    grad.clear()
    hess.clear()
    grad.resize(n, 0.0)
    hess.resize(n, 0.0)
    for k in range(len(pairs)):
        var winner = pairs[k].winner
        var loser = pairs[k].loser
        if winner < 0 or winner >= n or loser < 0 or loser >= n:
            raise Error("pair row index out of range")
        var c = pairs[k].weight
        var p = _logistic(scores[loser] - scores[winner])
        var g = c * p
        var h = g * (1.0 - p)
        grad[winner] -= g
        grad[loser] += g
        hess[winner] += h
        hess[loser] += h


# ---------------------------------------------------------------------------
# A24. YetiRank
# ---------------------------------------------------------------------------


@fieldwise_init
struct YetiRankParams(Copyable, Movable):
    """CatBoost's `YetiRank` loss parameters, at CatBoost's defaults.

    `permutations` and `decay` are `GetYetiRankPermutations` and
    `GetYetiRankDecay`. `noise` selects `TYetiRankPairWeightsCalcer`'s arm.
    `seed` has no CatBoost counterpart because theirs is an unnamed draw from
    the global learn-progress generator; ours is a parameter, because a
    stream that cannot be named cannot be reproduced.

    `mode`, `top`, `num_neighbors`, `dcg_type`, `dcg_denominator` and
    `noise_power` are absent rather than present-and-ignored. `Classic` +
    `Gumbel` + `num_neighbors=1` is the constructed default and is the whole
    of what a default `loss_function=YetiRank` run does; `top` is not read by
    `CalcWeightsClassic` at all.
    """

    var permutations: Int
    var decay: Float64
    var noise: Int
    var seed: Int

    @staticmethod
    def default() -> YetiRankParams:
        return YetiRankParams(
            DEFAULT_YETIRANK_PERMUTATIONS,
            DEFAULT_YETIRANK_DECAY,
            YETIRANK_NOISE_GUMBEL,
            DEFAULT_YETIRANK_SEED,
        )

    def validate(self) raises:
        if self.permutations < 1:
            raise Error("yetirank permutations must be at least 1")
        if not (self.decay >= 0.0 and self.decay <= 1.0):
            raise Error("yetirank decay must lie in [0, 1]")
        if (
            self.noise != YETIRANK_NOISE_NONE
            and self.noise != YETIRANK_NOISE_GUMBEL
        ):
            raise Error(
                "yetirank noise must be YETIRANK_NOISE_GUMBEL (CatBoost's"
                " default) or YETIRANK_NOISE_NONE; the Gauss arm is not"
                " implemented"
            )


def _yetirank_stream(
    seed: Int, iteration: Int, query: Int, permutation: Int
) -> UInt64:
    """Start of the counter stream for one (query, permutation) draw.

    Same shape as `sampling._mvs_stream` and `langevin._langevin_row_stream`:
    mix the seed against this module's domain constant, then fold each index
    in after spreading it by the golden-ratio increment. Sign bits are masked
    off so a negative seed is accepted without relying on signed-to-unsigned
    conversion.

    The result is a **start**, not a running state. Document `d` reads
    `stream + d` and nothing advances, so no document's noise depends on how
    many documents were drawn before it, no permutation shares a draw with
    another, and no query's stream depends on which block, thread or worker
    handled it. That last clause is the whole point: CatBoost's YetiRank seed
    for query `q` is `GenRand()` off a per-BLOCK generator walked
    sequentially over the queries of that block, which is reproducible there
    only because their block count is a compile-time constant.
    """
    var h = splitmix64(UInt64(seed & 0x7FFFFFFFFFFFFFFF) ^ _YETIRANK_DOMAIN)
    h = splitmix64(h ^ (UInt64(iteration & 0x7FFFFFFFFFFFFFFF) * GOLDEN))
    h = splitmix64(h ^ (UInt64(query & 0x7FFFFFFFFFFFFFFF) * GOLDEN))
    return splitmix64(h ^ (UInt64(permutation & 0x7FFFFFFFFFFFFFFF) * GOLDEN))


def yetirank_pairs(
    scores: List[Float64],
    targets: List[Float64],
    groups: RankGroups,
    params: YetiRankParams,
    iteration: Int,
    sample_weight: List[Float64] = [],
) raises -> List[RankPair]:
    """One round's YetiRank pairs: `UpdatePairsForYetiRank` +
    `GenerateYetiRankPairsForQuery` + `CalcWeightsClassic`.

    Per query, for each of `params.permutations` draws:

    1. Perturb the scores. In the `Gumbel` arm CatBoost does
       `expApprox[d] *= u / (1.000001f - u)`; since `IsStoreExpApprox` lists
       `YetiRank`, that is `a_d + log(u) - log(1.000001 - u)` on the score
       scale -- **logistic** noise, not Gumbel. The name in CatBoost's enum is
       wrong; the arithmetic is what it is. The additive form is used here
       because the multiplicative one overflows `exp` at large approxes, and
       the ordering is identical. `u = 0` gives `-inf`, which sinks the
       document to the bottom exactly as CatBoost's multiply-by-zero does.
    2. Stable-sort descending by the perturbed score.
    3. Walk the sorted list once and charge each ADJACENT pair
       `0.15 * decay^k * |rel_first - rel_second|` to whichever member has the
       larger relevance, with `k` the position. An equal-relevance pair is
       charged nothing: `AddWeight` has a `>` branch and a `<` branch and no
       `==` branch.

    Then every charged (winner, loser) cell becomes a pair of weight
    `queryWeight * total / permutations`.

    **Storage.** CatBoost allocates a dense `querySize x querySize` matrix per
    query and scans all of it, which is quadratic in the group size for a mode
    that can only ever touch `permutations * (querySize - 1)` cells. Here the
    charges go into a flat list, are stably sorted by
    `winner * querySize + loser`, and equal keys are merged. That key is
    exactly CatBoost's winner-major/loser-major scan order, so the emitted
    pair SEQUENCE is theirs; the per-cell sum is a reassociation of the same
    at-most-`permutations` addends, in permutation order, and moves the last
    ulp.

    Deterministic across `MOJOTREES_NUM_WORKERS` and machines: every draw is
    `uniform(_yetirank_stream(seed, iteration, q, p) + d)` and nothing is
    sequential.
    """
    params.validate()
    check_groups(groups, len(scores))
    if len(targets) != len(scores):
        raise Error("targets length must equal scores length")
    check_group_weights_constant(sample_weight, groups)
    if iteration < 0:
        raise Error("iteration must be nonnegative")

    var pairs = List[RankPair]()
    var noisy = List[Float64]()
    var keys = List[Int]()
    var charges = List[Float64]()
    for q in range(groups.n_queries()):
        var start = groups.start(q)
        var size = groups.starts[q + 1] - start
        if size > YETIRANK_MAX_QUERY_SIZE:
            raise Error(
                "yetirank supports at most ",
                YETIRANK_MAX_QUERY_SIZE,
                " documents in one query; group ",
                q,
                " has ",
                size,
            )
        if size < 2:
            continue
        var qw = group_weight(sample_weight, groups, q)
        keys.clear()
        charges.clear()
        for p in range(params.permutations):
            noisy.clear()
            if params.noise == YETIRANK_NOISE_GUMBEL:
                var stream = _yetirank_stream(params.seed, iteration, q, p)
                for d in range(size):
                    var u = uniform(stream + UInt64(d))
                    # `log(0)` is -inf, which is the intended sink; the guard
                    # keeps the second logarithm's argument positive for every
                    # `u` in [0, 1).
                    noisy.append(
                        scores[start + d]
                        + log(u)
                        - log(YETIRANK_NOISE_GUARD - u)
                    )
            else:
                for d in range(size):
                    noisy.append(scores[start + d])
            var order = _stable_argsort_desc(noisy, size)
            var decay_coefficient = 1.0
            for pos in range(1, size):
                var first = order[pos - 1]
                var second = order[pos]
                var rel_first = targets[start + first]
                var rel_second = targets[start + second]
                if rel_first != rel_second:
                    var charge = (
                        YETIRANK_MAGIC_CONST
                        * decay_coefficient
                        * abs(rel_first - rel_second)
                    )
                    var winner = first
                    var loser = second
                    if rel_second > rel_first:
                        winner = second
                        loser = first
                    keys.append(winner * size + loser)
                    charges.append(charge)
                decay_coefficient *= params.decay

        if len(keys) == 0:
            continue
        var order2 = _stable_argsort_keys(keys)
        var i = 0
        while i < len(order2):
            var key = keys[order2[i]]
            var total = 0.0
            var j = i
            while j < len(order2) and keys[order2[j]] == key:
                total += charges[order2[j]]
                j += 1
            var weight = qw * total / Float64(params.permutations)
            if weight != 0.0:
                pairs.append(
                    RankPair(
                        start + key // size, start + key % size, weight
                    )
                )
            i = j
    return pairs^


def yetirank_gradients(
    scores: List[Float64],
    targets: List[Float64],
    groups: RankGroups,
    mut grad: List[Float64],
    mut hess: List[Float64],
    params: YetiRankParams = YetiRankParams.default(),
    iteration: Int = 0,
    sample_weight: List[Float64] = [],
) raises:
    """One round of YetiRank derivatives: sample the pairs, then apply the
    `PairLogit` gradient to them.

    `YetiRankRecalculation` does exactly this -- it rebuilds the query pair
    lists from the current approxes and then lets `TPairLogitError` do the
    differentiation -- which is why there is no separate YetiRank derivative
    anywhere in CatBoost.

    `iteration` is part of the RNG key, so two rounds at the same scores draw
    different rankings, as CatBoost's per-round reseeding does.
    """
    var pairs = yetirank_pairs(
        scores, targets, groups, params, iteration, sample_weight
    )
    pair_logit_gradients(scores, pairs, grad, hess)


# ---------------------------------------------------------------------------
# A25. The eval metrics
# ---------------------------------------------------------------------------


@fieldwise_init
struct CatBoostNdcgParams(Copyable, Movable):
    """`TDcgMetric`'s parameters at CatBoost's defaults.

    `dcg_type` defaults to `NDCG_TYPE_BASE`, the raw relevance. **This is the
    difference that matters**: LightGBM's NDCG, and therefore `ranking.ndcg`,
    uses `2^rel - 1`, which is CatBoost's `Exp`. `top` defaults to `TOP_ALL`
    because `DefaultTopSize = -1` is handed to a `ui32` parameter and becomes
    `Max<ui32>`; LightGBM's `eval_at` defaults to 5.
    """

    var top: Int
    var dcg_type: Int
    var denominator: Int

    @staticmethod
    def default() -> CatBoostNdcgParams:
        return CatBoostNdcgParams(
            TOP_ALL, NDCG_TYPE_BASE, NDCG_DENOMINATOR_LOG_POSITION
        )

    def validate(self) raises:
        if self.top != TOP_ALL and self.top < 1:
            raise Error("ndcg top must be TOP_ALL or at least 1")
        if self.dcg_type != NDCG_TYPE_BASE and self.dcg_type != NDCG_TYPE_EXP:
            raise Error(
                "ndcg dcg_type must be NDCG_TYPE_BASE or NDCG_TYPE_EXP"
            )
        if (
            self.denominator != NDCG_DENOMINATOR_LOG_POSITION
            and self.denominator != NDCG_DENOMINATOR_POSITION
        ):
            raise Error(
                "ndcg denominator must be NDCG_DENOMINATOR_LOG_POSITION or"
                " NDCG_DENOMINATOR_POSITION"
            )


def _dcg_decay(denominator: Int, position: Int) -> Float64:
    """`dcg.cpp::FillDcgDecay`, one entry: `decay[0] = 1`, then
    `1 / log2(i + 2)` (LogPosition) or `1 / (i + 1)` (Position)."""
    if position == 0:
        return 1.0
    if denominator == NDCG_DENOMINATOR_POSITION:
        return 1.0 / Float64(position + 1)
    return 1.0 / log2(Float64(position) + 2.0)


def _dcg_numerator(relevance: Float64, dcg_type: Int) -> Float64:
    """`dcg.cpp::CalcDcgSorted`: the relevance itself under `Base`,
    `pow(2, rel) - 1` under `Exp`.

    An integral relevance in `[0, 52]` -- which is every graded relevance a
    ranking dataset carries, and more than `ranking.MAX_RELEVANCE_LABEL`
    allows -- takes the exact integer shift `ranking.label_gain` uses, so
    `2^3 - 1` is the literal 7 rather than 7 plus a rounding error from
    `exp(3 ln 2)`. Anything else falls back to the transcendental, which is
    what CatBoost's `Exp2` is in every case.
    """
    if dcg_type != NDCG_TYPE_EXP:
        return relevance
    if relevance >= 0.0 and relevance <= 52.0:
        var k = Int(relevance)
        if Float64(k) == relevance:
            return Float64((Int(1) << k) - 1)
    return exp(relevance * _LN2) - 1.0


def _effective_top(top: Int, size: Int) -> Int:
    """`Min<ui32>(topSizeRequested, samples.size())` with `TOP_ALL` standing
    for CatBoost's `Max<ui32>`."""
    if top == TOP_ALL or top > size:
        return size
    return top


def query_ndcg(
    scores: List[Float64],
    targets: List[Float64],
    groups: RankGroups,
    q: Int,
    params: CatBoostNdcgParams,
) raises -> Float64:
    """One query's CatBoost NDCG: `dcg.cpp::CalcNdcg` on that query's
    samples.

        DCG  = sum_{i < top} numerator(target[order[i]]) * decay[i]
        IDCG = the same, over targets sorted DESCENDING by target alone
        NDCG = IDCG > 0 ? DCG / IDCG : 1

    `order` is `_argsort_docs`, so an exact score tie ranks the LOWER
    relevance first. The ideal ordering is a stable sort by target alone
    (`CalcIDcg`'s comparator is `left.Target > right.Target`, with no score
    term), which matters only when the ranking is truncated.
    """
    params.validate()
    var start = groups.start(q)
    var size = groups.starts[q + 1] - start
    var top = _effective_top(params.top, size)
    var order = _argsort_docs(scores, targets, start, size)
    var dcg = 0.0
    for i in range(top):
        dcg += _dcg_numerator(
            targets[start + order[i]], params.dcg_type
        ) * _dcg_decay(params.denominator, i)

    var ideal = List[Float64](capacity=size)
    for i in range(size):
        ideal.append(targets[start + i])
    var ideal_order = _stable_argsort_desc(ideal, size)
    var idcg = 0.0
    for i in range(top):
        idcg += _dcg_numerator(
            ideal[ideal_order[i]], params.dcg_type
        ) * _dcg_decay(params.denominator, i)

    if idcg > 0.0:
        return dcg / idcg
    return 1.0


def catboost_ndcg(
    scores: List[Float64],
    targets: List[Float64],
    groups: RankGroups,
    params: CatBoostNdcgParams = CatBoostNdcgParams.default(),
    sample_weight: List[Float64] = [],
) raises -> Float64:
    """CatBoost's `NDCG` over a whole dataset: `TDcgMetric::EvalSingleThread`
    plus `GetFinalError`.

        reported = sum_q qw_q NDCG_q / sum_q qw_q      (0 when the sum is 0)

    `UseWeights` defaults to true in CatBoost, so the group weights are on by
    default here too; an empty vector means unit weights.

    **This is not `ranking.ndcg`.** That one is LightGBM's: gain `2^l - 1`,
    truncated at `eval_at` (5), ties left in input order. This one defaults to
    the raw relevance, no truncation, and CatBoost's adversarial tie rule.
    Scoring a CatBoost run with the LightGBM metric, or the reverse, produces
    a number that is not comparable to anything either library reports.

    Higher is better. The queries are summed in query order, once, so the
    value does not move with the worker count.
    """
    params.validate()
    check_groups(groups, len(scores))
    if len(targets) != len(scores):
        raise Error("targets length must equal scores length")
    check_group_weights_constant(sample_weight, groups)
    var total = 0.0
    var weight_sum = 0.0
    for q in range(groups.n_queries()):
        var qw = group_weight(sample_weight, groups, q)
        total += qw * query_ndcg(scores, targets, groups, q, params)
        weight_sum += qw
    if weight_sum == 0.0:
        return 0.0
    return total / weight_sum


def query_pfound(
    scores: List[Float64],
    targets: List[Float64],
    groups: RankGroups,
    q: Int,
    top: Int = TOP_ALL,
    decay: Float64 = DEFAULT_PFOUND_DECAY,
) raises -> Float64:
    """One query's PFound: `pfound.h::TPFoundCalcer::AddQuery`, verbatim.

        order   = stable sort by CompareDocs   (the SAME tie rule as NDCG)
        pLook = 1, pFound = 0
        for position in [0, min(size, top)):
            pRel    = relevance[order[position]]
            pFound += pRel * pLook
            pLook  *= (1 - pRel) * decay

    **Relevances are probabilities here, not grades.** The recursion is only a
    probability if `pRel` lies in [0, 1]. CatBoost does not check and neither
    does this, because refusing would break the CatBoost-compatible reading of
    a graded column -- but the value is meaningless outside that range and a
    negative relevance can drive `pLook` above 1.

    `subgroup_id` deduplication is not implemented: CatBoost skips a document
    whose subgroup was already seen at a higher position, and there is no
    subgroup column in mojotrees. A PFound computed here on data that HAS
    subgroups in CatBoost will read high.
    """
    if not (decay >= 0.0 and decay <= 1.0):
        raise Error("pfound decay must lie in [0, 1]")
    if top != TOP_ALL and top < 1:
        raise Error("pfound top must be TOP_ALL or at least 1")
    var start = groups.start(q)
    var size = groups.starts[q + 1] - start
    var depth = _effective_top(top, size)
    var order = _argsort_docs(scores, targets, start, size)
    var p_look = 1.0
    var p_found = 0.0
    for position in range(depth):
        var p_rel = targets[start + order[position]]
        p_found += p_rel * p_look
        p_look *= (1.0 - p_rel) * decay
    return p_found


def pfound(
    scores: List[Float64],
    targets: List[Float64],
    groups: RankGroups,
    top: Int = TOP_ALL,
    decay: Float64 = DEFAULT_PFOUND_DECAY,
    sample_weight: List[Float64] = [],
) raises -> Float64:
    """CatBoost's `PFound` over a whole dataset: `TPFoundMetric` plus
    `TPFoundCalcer::Score`.

        reported = sum_q qw_q pFound_q / sum_q qw_q     (0 when the sum is 0)

    `UseWeights` defaults to true in CatBoost, so group weights are on by
    default here too. Higher is better.

    CatBoost's `GetBestValue` reports `Max` with `bestValue = 0`, which is a
    quirk -- the attainable maximum is 1 -- and is not carried across. Only
    "higher is better" is.
    """
    check_groups(groups, len(scores))
    if len(targets) != len(scores):
        raise Error("targets length must equal scores length")
    check_group_weights_constant(sample_weight, groups)
    var total = 0.0
    var weight_sum = 0.0
    for q in range(groups.n_queries()):
        var qw = group_weight(sample_weight, groups, q)
        total += qw * query_pfound(scores, targets, groups, q, top, decay)
        weight_sum += qw
    if weight_sum == 0.0:
        return 0.0
    return total / weight_sum


# ---------------------------------------------------------------------------
# The fit path
# ---------------------------------------------------------------------------


def catboost_ranking_gradients(
    objective: Int,
    scores: List[Float64],
    targets: List[Float64],
    groups: RankGroups,
    mut grad: List[Float64],
    mut hess: List[Float64],
    pairs: List[RankPair] = [],
    yeti: YetiRankParams = YetiRankParams.default(),
    iteration: Int = 0,
    sample_weight: List[Float64] = [],
) raises:
    """One round of derivatives for whichever of the three the code names.

    `pairs` is read only under `PAIR_LOGIT`, where it is the pair list the
    fit generated once at setup; under `YETI_RANK` the pairs are redrawn from
    the current scores every round, which is the whole mechanism, so passing
    them in would be meaningless.
    """
    if objective == QUERY_RMSE:
        query_rmse_gradients(
            scores, targets, groups, grad, hess, sample_weight
        )
        return
    if objective == PAIR_LOGIT:
        pair_logit_gradients(scores, pairs, grad, hess)
        return
    if objective == YETI_RANK:
        yetirank_gradients(
            scores,
            targets,
            groups,
            grad,
            hess,
            yeti,
            iteration,
            sample_weight,
        )
        return
    raise Error("not a CatBoost ranking objective code: ", objective)


def train_catboost_ranker(
    data: BinnedMatrix,
    targets: List[Float64],
    groups: RankGroups,
    objective: Int,
    params: BoosterParams,
    yeti: YetiRankParams = YetiRankParams.default(),
    sample_weight: List[Float64] = [],
) raises -> Booster:
    """Train one of the three CatBoost ranking objectives on a pre-binned
    matrix.

    Boosting starts from a raw score of 0 for all three. `QueryRMSE` removes
    the query mean from every residual, and the two pairwise losses see only
    score DIFFERENCES, so in all three cases the level is unidentifiable and
    there is nothing to average a base score from. That is
    `objective_init_kind`'s `INIT_ZERO`, the same answer `LAMBDARANK` gets,
    and it is why the returned `Booster` carries a base score of 0.0.

    `PAIR_LOGIT` generates its pairs ONCE, at setup, because
    `GeneratePairLogitPairs` reads only the labels and the grouping.
    `YETI_RANK` redraws its pairs every round from the current scores, keyed
    by the round index, because that is the objective.

    The total hessian is checked to be positive before the first tree.
    Under the pairwise losses a row that appears in no pair has both `grad`
    and `hess` exactly 0, so a fit whose pair list is empty would hand the
    grower an all-zero hessian plane and every leaf would compute
    `-0/(0 + reg_lambda)`. With `reg_lambda = 0`, LightGBM's stock default,
    that is `0/0`. Refusing here is the difference between an error and an
    ensemble of NaN leaves.

    No bagging, no early stopping and no validation set: `ranking.train_ranker`
    and `ranking_advanced.train_ranker_advanced` own those for LambdaRank and
    this lane did not widen them. They are the obvious next step.
    """
    if not is_catboost_ranking_objective(objective):
        raise Error("not a CatBoost ranking objective code: ", objective)
    if len(targets) != data.n_rows:
        raise Error("targets length must equal n_rows")
    check_groups(groups, data.n_rows)
    check_group_weights_constant(sample_weight, groups)
    yeti.validate()

    var n = data.n_rows
    var raw = List[Float64](capacity=n)
    raw.resize(n, 0.0)
    var pairs = List[RankPair]()
    if objective == PAIR_LOGIT:
        pairs = generate_pair_logit_pairs(targets, groups, sample_weight)

    var trees = List[Tree]()
    var grad = List[Float64]()
    var hess = List[Float64]()
    for i in range(params.n_estimators):
        catboost_ranking_gradients(
            objective,
            raw,
            targets,
            groups,
            grad,
            hess,
            pairs,
            yeti,
            i,
            sample_weight,
        )
        if i == 0:
            var hess_total = 0.0
            for r in range(n):
                hess_total += hess[r]
            if not (hess_total > 0.0):
                raise Error(
                    "the '",
                    catboost_ranking_name(objective),
                    "' objective produced a zero total hessian on the first"
                    " round: there is no pair to separate, so every leaf"
                    " would divide zero by zero",
                )
        var tree = grow_tree(data, grad, hess, params.tree, [], i)

        # A single-leaf tree with a near-zero value means no pair is left to
        # separate. `ranking.train_ranker` stops on the same condition.
        if tree.n_leaves == 1 and abs(tree.value[0]) < 1e-12:
            break

        for r in range(n):
            raw[r] += params.learning_rate * tree.predict_row(data, r)
        trees.append(tree^)

    return Booster(
        trees^,
        0.0,
        params.learning_rate,
        objective,
        params.tree.monotone.copy(),
    )
