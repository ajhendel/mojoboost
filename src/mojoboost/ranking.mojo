"""Learning to rank: LambdaRank with an NDCG-weighted pairwise objective.

Ranking data arrives as a flat row matrix plus *query boundaries*: rows
[start, end) of one query are the candidate documents retrieved for one
search, and `labels[r]` is that document's graded relevance (a nonnegative
integer, 0 = irrelevant). Every quantity in this module is computed inside
one query and never across two: gradients, the NDCG metric, the validation
signal, and bagging all treat a query as the indivisible unit.

Rows of a query must be contiguous. `groups_from_counts` takes LightGBM's
`group` array (one row count per query, in row order); `groups_from_query_ids`
takes a per-row query id column and rejects any id whose rows are split into
more than one run, because a silently truncated query would renormalize
every NDCG it appears in.

Objective (LightGBM's `LambdarankNDCG::GetGradientsForOneQuery`)
---------------------------------------------------------------
Within a query, documents are ranked by their current model score. For each
pair (i, j) of documents with *different* labels, where `high` is the more
relevant and `low` the less relevant one,

    rho    = 1 / (1 + exp(sigmoid * (score[high] - score[low])))
    |dNDCG| = (gain(label[high]) - gain(label[low]))
              * |discount(rank[high]) - discount(rank[low])| / maxDCG
    lambda = -sigmoid * rho * |dNDCG|
    hess   = sigmoid^2 * rho * (1 - rho) * |dNDCG|

with `gain(l) = 2^l - 1`, `discount(r) = 1 / log2(r + 2)`, and maxDCG the
best DCG this query's labels can reach, both truncated at
`truncation_level`. `lambda` is added to the high document's gradient and
subtracted from the low one's, so the high document is pushed up. Pairs are
enumerated as LightGBM does: the better-ranked member must sit within
`truncation_level` of the top, the worse-ranked member may sit anywhere.

With `norm` on (LightGBM's `lambdarank_norm`, on by default) each pair's
|dNDCG| is divided by `0.01 + |score difference|` unless every score in the
query is equal, and the query's gradients and hessians are then scaled by
`log2(1 + sum_lambdas) / sum_lambdas`, which keeps queries with many
mis-ordered pairs from dominating a round.

Gradients are exactly antisymmetric within a query, so each query's
gradients sum to zero and no query can shift the global score level; the
model only learns relative order. Boosting therefore starts from a base
score of 0 (LightGBM does not boost lambdarank from an average either).

Metric
------
`ndcg` and `ndcg_at_cutoffs` compute the mean NDCG over queries at one or
more cutoff positions. A query whose labels are all 0 has no attainable DCG
and counts as 1.0, matching LightGBM.

INTENTIONAL DIFFERENCES FROM LightGBM
-------------------------------------
- The pairwise sigmoid is evaluated exactly. LightGBM reads it from a
  2^20-entry lookup table over [-50, 50] and clamps outside that range, so
  its lambdas carry a small table-quantization error that this does not
  reproduce.
- `label_gain` is fixed at `2^l - 1` and labels are capped at 30, the range
  of LightGBM's default table. LightGBM allows a custom gain vector.
- Bagging samples whole queries, never rows. This is LightGBM's
  `bagging_by_query=true` behavior; its default samples rows, which splits
  queries and silently changes the maxDCG the surviving rows are normalized
  against.
- Early stopping compares against the best round seen so far and never
  against a "zero trees" baseline, because the NDCG of an all-zero score
  vector reflects nothing but the order the rows were handed in.
- Positional/unbiased-lambdarank extensions and LightGBM's `kMinScore`
  filtering of dropped documents are not implemented.
"""

from std.math import exp, log2

from .bagging import BaggingParams, bagging_enabled, check_bagging, sample_rows
from .binning import BinMapper, BinnedMatrix, fit_bins
from .boosting import Booster, BoosterParams, _check_sample_weight
from .model import Model
from .tree import Tree, TreeParams, grow_tree

# Objective code for a LambdaRank booster. Codes 0..5 are boosting.mojo's
# built-in objectives and 6 is CUSTOM; this continues that one registry, and
# a serialized ranker is an ordinary single-output model that carries it.
# Ranker predictions are raw scores: `Booster.predict_row` has no link
# function for this code, and only the order of the scores is meaningful.
comptime LAMBDARANK = 7

# The largest relevance label, the range of LightGBM's default label_gain.
comptime MAX_RELEVANCE_LABEL = 30

# LightGBM's lambdarank_truncation_level, sigmoid, and the first entry of
# its default eval_at list for ranking.
comptime DEFAULT_TRUNCATION_LEVEL = 30
comptime DEFAULT_SIGMOID = 1.0
comptime DEFAULT_NDCG_EVAL_AT = 5


@fieldwise_init
struct RankGroups(Copyable, Movable):
    """Query boundaries over a row matrix.

    Query q owns rows [starts[q], starts[q + 1]). `starts` has n_queries + 1
    entries, starts[0] is 0, and the last entry is `n_rows`. Build one with
    `groups_from_counts` or `groups_from_query_ids` rather than by hand:
    those validate the invariants the rest of this module relies on.
    """

    var starts: List[Int]
    var n_rows: Int

    def n_queries(self) -> Int:
        return len(self.starts) - 1

    def start(self, q: Int) -> Int:
        return self.starts[q]

    def size(self, q: Int) -> Int:
        return self.starts[q + 1] - self.starts[q]

    def max_size(self) -> Int:
        var m = 0
        for q in range(self.n_queries()):
            var s = self.starts[q + 1] - self.starts[q]
            if s > m:
                m = s
        return m


def groups_from_counts(counts: List[Int]) raises -> RankGroups:
    """Query boundaries from LightGBM's `group` array: one row count per
    query, in row order, so query q owns the next `counts[q]` rows.

    Raises on an empty array or a nonpositive count. The caller must still
    check `RankGroups.n_rows` against the dataset; the trainers do."""
    if len(counts) == 0:
        raise Error("group must contain at least one query")
    var starts = List[Int](capacity=len(counts) + 1)
    starts.append(0)
    var total = 0
    for q in range(len(counts)):
        if counts[q] <= 0:
            raise Error("group counts must be positive")
        total += counts[q]
        starts.append(total)
    return RankGroups(starts^, total)


def groups_from_query_ids(query_ids: List[Int]) raises -> RankGroups:
    """Query boundaries from a per-row query id column.

    Ids may be any integers in any order, but each query's rows must form
    one unbroken run: an id that reappears after a different id is rejected
    as noncontiguous rather than silently split into two queries."""
    var n = len(query_ids)
    if n == 0:
        raise Error("query ids must contain at least one row")
    var starts = List[Int]()
    starts.append(0)
    var run_ids = List[Int]()
    run_ids.append(query_ids[0])
    for r in range(1, n):
        if query_ids[r] != query_ids[r - 1]:
            starts.append(r)
            run_ids.append(query_ids[r])
    starts.append(n)

    # One run per id: a duplicate among the run ids means some query's rows
    # were interleaved with another's.
    var sorted_ids = run_ids.copy()
    sort(sorted_ids)
    for i in range(1, len(sorted_ids)):
        if sorted_ids[i] == sorted_ids[i - 1]:
            raise Error(
                "query ids must be contiguous: rows of query "
                + String(sorted_ids[i])
                + " are not consecutive"
            )
    return RankGroups(starts^, n)


def check_groups(groups: RankGroups, n_rows: Int) raises:
    """Validate query boundaries against a dataset of `n_rows` rows."""
    if len(groups.starts) < 2:
        raise Error("group must contain at least one query")
    if groups.starts[0] != 0:
        raise Error("group boundaries must start at row 0")
    for q in range(groups.n_queries()):
        if groups.starts[q + 1] <= groups.starts[q]:
            raise Error("group boundaries must be strictly increasing")
    if groups.starts[groups.n_queries()] != groups.n_rows:
        raise Error("group boundaries must end at n_rows")
    if groups.n_rows != n_rows:
        raise Error("group counts must sum to n_rows")


def check_labels(labels: List[Int]) raises:
    """Relevance labels must be in [0, MAX_RELEVANCE_LABEL]."""
    for r in range(len(labels)):
        if labels[r] < 0:
            raise Error("relevance labels must be nonnegative")
        if labels[r] > MAX_RELEVANCE_LABEL:
            raise Error(
                "relevance labels must be at most "
                + String(MAX_RELEVANCE_LABEL)
            )


def label_gain(label: Int) -> Float64:
    """LightGBM's default label_gain: 2^label - 1, exact for label <= 30."""
    return Float64((1 << label) - 1)


def _discounts(n: Int) -> List[Float64]:
    """Position discounts 1 / log2(rank + 2) for ranks 0..n-1."""
    var d = List[Float64](capacity=n)
    for i in range(n):
        d.append(1.0 / log2(Float64(i) + 2.0))
    return d^


def _argsort_desc_range(
    values: List[Float64], start: Int, cnt: Int
) -> List[Int]:
    """Local indices 0..cnt-1 of `values[start : start + cnt]` ordered by
    value descending, stable (ties keep their input order), matching
    LightGBM's `std::stable_sort` with a `>` comparator."""
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
                if values[start + idx[i]] >= values[start + idx[j]]:
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


def _sorted_gains(labels: List[Int], start: Int, cnt: Int) -> List[Float64]:
    """The query's label gains in descending order, by counting sort over
    the fixed label range."""
    var counts = List[Int](capacity=MAX_RELEVANCE_LABEL + 1)
    for _ in range(MAX_RELEVANCE_LABEL + 1):
        counts.append(0)
    for i in range(cnt):
        counts[labels[start + i]] += 1
    var out = List[Float64](capacity=cnt)
    for label in range(MAX_RELEVANCE_LABEL, -1, -1):
        for _ in range(counts[label]):
            out.append(label_gain(label))
    return out^


def max_dcg(labels: List[Int], start: Int, cnt: Int, k: Int) -> Float64:
    """The best DCG at cutoff k attainable by this query's labels."""
    var gains = _sorted_gains(labels, start, cnt)
    var m = k if k < cnt else cnt
    var discounts = _discounts(m)
    var total = 0.0
    for j in range(m):
        total += gains[j] * discounts[j]
    return total


def _inverse_max_dcgs(
    labels: List[Int], groups: RankGroups, k: Int
) -> List[Float64]:
    """1 / maxDCG@k per query, or 0.0 for a query with no attainable DCG
    (every label 0), which zeroes that query's lambdas as in LightGBM."""
    var out = List[Float64](capacity=groups.n_queries())
    for q in range(groups.n_queries()):
        var m = max_dcg(labels, groups.start(q), groups.size(q), k)
        out.append(1.0 / m if m > 0.0 else 0.0)
    return out^


def ndcg_at_cutoffs(
    scores: List[Float64],
    labels: List[Int],
    groups: RankGroups,
    cutoffs: List[Int],
) raises -> List[Float64]:
    """Mean NDCG over queries at each cutoff in `cutoffs`.

    Documents are ranked by `scores` within their own query, never across
    queries. A query whose labels are all 0 counts as 1.0 at every cutoff,
    matching LightGBM's handling of queries with no attainable DCG."""
    check_groups(groups, len(scores))
    if len(labels) != len(scores):
        raise Error("labels length must equal scores length")
    check_labels(labels)
    if len(cutoffs) == 0:
        raise Error("ndcg needs at least one cutoff")
    for i in range(len(cutoffs)):
        if cutoffs[i] < 1:
            raise Error("ndcg cutoffs must be positive")

    var totals = List[Float64](capacity=len(cutoffs))
    for _ in range(len(cutoffs)):
        totals.append(0.0)
    var discounts = _discounts(groups.max_size())

    for q in range(groups.n_queries()):
        var start = groups.start(q)
        var cnt = groups.size(q)
        var order = _argsort_desc_range(scores, start, cnt)
        var gains = _sorted_gains(labels, start, cnt)
        for c in range(len(cutoffs)):
            var m = cutoffs[c] if cutoffs[c] < cnt else cnt
            var dcg = 0.0
            var best = 0.0
            for j in range(m):
                dcg += label_gain(labels[start + order[j]]) * discounts[j]
                best += gains[j] * discounts[j]
            totals[c] += dcg / best if best > 0.0 else 1.0

    var n_queries = Float64(groups.n_queries())
    for c in range(len(cutoffs)):
        totals[c] /= n_queries
    return totals^


def ndcg(
    scores: List[Float64],
    labels: List[Int],
    groups: RankGroups,
    k: Int = DEFAULT_NDCG_EVAL_AT,
) raises -> Float64:
    """Mean NDCG@k over queries."""
    var cutoffs = List[Int](capacity=1)
    cutoffs.append(k)
    return ndcg_at_cutoffs(scores, labels, groups, cutoffs)[0]


@fieldwise_init
struct RankerParams(Copyable, Movable):
    """LambdaRank hyperparameters, LightGBM names in parentheses.

    `truncation_level` (lambdarank_truncation_level) bounds how far down the
    ranking the better member of a pair may sit, and is also the cutoff of
    the maxDCG the lambdas are normalized by. `sigmoid` (sigmoid) scales the
    pairwise logistic. `norm` (lambdarank_norm) enables LightGBM's per-pair
    and per-query lambda normalization. `ndcg_eval_at` is the cutoff of the
    validation metric that drives early stopping.
    """

    var truncation_level: Int
    var sigmoid: Float64
    var norm: Bool
    var ndcg_eval_at: Int

    @staticmethod
    def default() -> RankerParams:
        return RankerParams(
            DEFAULT_TRUNCATION_LEVEL,
            DEFAULT_SIGMOID,
            True,
            DEFAULT_NDCG_EVAL_AT,
        )


def check_ranker_params(params: RankerParams) raises:
    if params.truncation_level < 1:
        raise Error("lambdarank_truncation_level must be positive")
    if params.sigmoid <= 0.0:
        raise Error("sigmoid must be positive")
    if params.ndcg_eval_at < 1:
        raise Error("ndcg_eval_at must be positive")


def _pair_sigmoid(delta: Float64, sigma: Float64) -> Float64:
    """1 / (1 + exp(sigma * delta)), evaluated without overflow."""
    var z = sigma * delta
    if z >= 0.0:
        var e = exp(-z)
        return e / (1.0 + e)
    var e = exp(z)
    return 1.0 / (1.0 + e)


def _fill_query_lambdas(
    scores: List[Float64],
    labels: List[Int],
    start: Int,
    cnt: Int,
    inverse_max_dcg: Float64,
    discounts: List[Float64],
    params: RankerParams,
    mut grad: List[Float64],
    mut hess: List[Float64],
):
    """LightGBM's GetGradientsForOneQuery, writing grad/hess[start + i]."""
    for i in range(cnt):
        grad[start + i] = 0.0
        hess[start + i] = 0.0
    if cnt < 2:
        return

    var order = _argsort_desc_range(scores, start, cnt)
    var best_score = scores[start + order[0]]
    var worst_score = scores[start + order[cnt - 1]]
    var sum_lambdas = 0.0

    var top = cnt - 1
    if params.truncation_level < top:
        top = params.truncation_level
    for i in range(top):
        for j in range(i + 1, cnt):
            if labels[start + order[i]] == labels[start + order[j]]:
                continue
            var high_rank: Int
            var low_rank: Int
            if labels[start + order[i]] > labels[start + order[j]]:
                high_rank = i
                low_rank = j
            else:
                high_rank = j
                low_rank = i
            var high = start + order[high_rank]
            var low = start + order[low_rank]

            var delta_score = scores[high] - scores[low]
            var dcg_gap = label_gain(labels[high]) - label_gain(labels[low])
            var paired_discount = abs(
                discounts[high_rank] - discounts[low_rank]
            )
            var delta_ndcg = dcg_gap * paired_discount * inverse_max_dcg
            if params.norm and best_score != worst_score:
                delta_ndcg /= 0.01 + abs(delta_score)

            var p_lambda = _pair_sigmoid(delta_score, params.sigmoid)
            var p_hess = p_lambda * (1.0 - p_lambda)
            p_lambda *= -params.sigmoid * delta_ndcg
            p_hess *= params.sigmoid * params.sigmoid * delta_ndcg

            grad[low] -= p_lambda
            hess[low] += p_hess
            grad[high] += p_lambda
            hess[high] += p_hess
            # p_lambda is negative, so subtract to accumulate its magnitude.
            sum_lambdas -= 2.0 * p_lambda

    if params.norm and sum_lambdas > 0.0:
        var factor = log2(1.0 + sum_lambdas) / sum_lambdas
        for i in range(cnt):
            grad[start + i] *= factor
            hess[start + i] *= factor


def _fill_lambdas(
    scores: List[Float64],
    labels: List[Int],
    groups: RankGroups,
    inverse_max_dcg: List[Float64],
    discounts: List[Float64],
    params: RankerParams,
    weights: List[Float64],
    mut grad: List[Float64],
    mut hess: List[Float64],
):
    """Lambdas and hessians for every query, then per-row sample weights
    (applied after per-query normalization, as LightGBM does)."""
    var n = groups.n_rows
    grad.clear()
    grad.resize(n, 0.0)
    hess.clear()
    hess.resize(n, 0.0)
    for q in range(groups.n_queries()):
        _fill_query_lambdas(
            scores,
            labels,
            groups.start(q),
            groups.size(q),
            inverse_max_dcg[q],
            discounts,
            params,
            grad,
            hess,
        )
    if len(weights) > 0:
        for r in range(n):
            grad[r] *= weights[r]
            hess[r] *= weights[r]


def lambdarank_gradients(
    scores: List[Float64],
    labels: List[Int],
    groups: RankGroups,
    mut grad: List[Float64],
    mut hess: List[Float64],
    params: RankerParams = RankerParams.default(),
    sample_weight: List[Float64] = [],
) raises:
    """LambdaRank gradients and hessians for one set of model scores.

    `grad` and `hess` are overwritten with one entry per row. Every pair
    compared lies inside a single query, so a row's gradient depends only on
    its own query's scores and labels."""
    check_groups(groups, len(scores))
    if len(labels) != len(scores):
        raise Error("labels length must equal scores length")
    check_labels(labels)
    check_ranker_params(params)
    _check_sample_weight(sample_weight, len(scores))
    var inverse_max_dcg = _inverse_max_dcgs(
        labels, groups, params.truncation_level
    )
    var discounts = _discounts(groups.max_size())
    _fill_lambdas(
        scores,
        labels,
        groups,
        inverse_max_dcg,
        discounts,
        params,
        sample_weight,
        grad,
        hess,
    )


def _expand_queries(
    groups: RankGroups, query_bag: List[Int], mut rows: List[Int]
):
    """Rows of the sampled queries, ascending. Queries are contiguous and
    the bag is ascending, so the row list is too."""
    rows.clear()
    for i in range(len(query_bag)):
        var q = query_bag[i]
        for r in range(groups.start(q), groups.starts[q + 1]):
            rows.append(r)


def _refresh_query_bag(
    mut query_bag: List[Int],
    mut row_bag: List[Int],
    groups: RankGroups,
    bagging: BaggingParams,
    iteration: Int,
) raises:
    """Redraw the bag on the rounds LightGBM would, sampling whole queries.

    Bagging ranking data by row would leave a query holding some of its
    documents, whose maxDCG is then that of the survivors rather than of the
    query that was actually served, so mojoboost always bags by query."""
    if not bagging_enabled(bagging):
        return
    if iteration % bagging.freq != 0:
        return
    sample_rows(
        bagging, groups.n_queries(), iteration // bagging.freq, query_bag
    )
    _expand_queries(groups, query_bag, row_bag)


def train_ranker(
    data: BinnedMatrix,
    labels: List[Int],
    groups: RankGroups,
    params: BoosterParams,
    rank_params: RankerParams = RankerParams.default(),
    sample_weight: List[Float64] = [],
    bagging: BaggingParams = BaggingParams.disabled(),
) raises -> Booster:
    """Train a LambdaRank ensemble on a pre-binned matrix.

    `labels` are graded relevances in [0, 30] and `groups` marks the query
    each row belongs to; rows of one query must be contiguous. Boosting
    starts from a base score of 0 because lambdas are antisymmetric within a
    query and carry no information about the score level. A non-empty
    `sample_weight` scales each row's lambda and hessian. `bagging` samples
    whole queries per round (see `_refresh_query_bag`)."""
    if len(labels) != data.n_rows:
        raise Error("labels length must equal n_rows")
    check_groups(groups, data.n_rows)
    check_labels(labels)
    check_ranker_params(rank_params)
    _check_sample_weight(sample_weight, data.n_rows)
    check_bagging(bagging)

    var n = data.n_rows
    var raw = List[Float64](capacity=n)
    raw.resize(n, 0.0)
    var inverse_max_dcg = _inverse_max_dcgs(
        labels, groups, rank_params.truncation_level
    )
    var discounts = _discounts(groups.max_size())

    var trees = List[Tree]()
    var grad = List[Float64](capacity=n)
    var hess = List[Float64](capacity=n)
    var query_bag = List[Int]()
    var row_bag = List[Int]()
    for i in range(params.n_estimators):
        _refresh_query_bag(query_bag, row_bag, groups, bagging, i)
        _fill_lambdas(
            raw,
            labels,
            groups,
            inverse_max_dcg,
            discounts,
            rank_params,
            sample_weight,
            grad,
            hess,
        )
        var tree = grow_tree(data, grad, hess, params.tree, row_bag, i)

        # A single-leaf tree with a near-zero value means no pair is left to
        # separate; under bagging it only means that of this bag, so the
        # round is skipped and the next bag gets its turn.
        if tree.n_leaves == 1 and abs(tree.value[0]) < 1e-12:
            if bagging_enabled(bagging):
                continue
            break

        for r in range(n):
            raw[r] += params.learning_rate * tree.predict_row(data, r)
        trees.append(tree^)

    return Booster(trees^, 0.0, params.learning_rate, LAMBDARANK)


def train_ranker_with_valid(
    data: BinnedMatrix,
    labels: List[Int],
    groups: RankGroups,
    valid_data: BinnedMatrix,
    valid_labels: List[Int],
    valid_groups: RankGroups,
    params: BoosterParams,
    early_stopping_rounds: Int,
    rank_params: RankerParams = RankerParams.default(),
    min_delta: Float64 = 0.0,
    sample_weight: List[Float64] = [],
    bagging: BaggingParams = BaggingParams.disabled(),
) raises -> Booster:
    """Train with query-aware validation and early stopping.

    The validation set carries its own query boundaries and its NDCG is the
    mean over its queries, each ranked within itself, so no validation row
    is ever compared against a row of another query. Stops when mean
    NDCG@`rank_params.ndcg_eval_at` has not improved by more than min_delta
    for `early_stopping_rounds` consecutive rounds, and truncates the
    ensemble to its best round. sample_weight and bagging apply to training
    rows only; validation rows are never bagged."""
    if len(labels) != data.n_rows:
        raise Error("labels length must equal n_rows")
    if len(valid_labels) != valid_data.n_rows:
        raise Error("valid_labels length must equal valid n_rows")
    if valid_data.n_features != data.n_features:
        raise Error("valid_data must have the same features")
    if early_stopping_rounds < 1:
        raise Error("early_stopping_rounds must be positive")
    check_groups(groups, data.n_rows)
    check_groups(valid_groups, valid_data.n_rows)
    check_labels(labels)
    check_labels(valid_labels)
    check_ranker_params(rank_params)
    _check_sample_weight(sample_weight, data.n_rows)
    check_bagging(bagging)

    var n = data.n_rows
    var n_valid = valid_data.n_rows
    var raw = List[Float64](capacity=n)
    raw.resize(n, 0.0)
    var valid_raw = List[Float64](capacity=n_valid)
    valid_raw.resize(n_valid, 0.0)
    var inverse_max_dcg = _inverse_max_dcgs(
        labels, groups, rank_params.truncation_level
    )
    var discounts = _discounts(groups.max_size())

    var trees = List[Tree]()
    var grad = List[Float64](capacity=n)
    var hess = List[Float64](capacity=n)
    var query_bag = List[Int]()
    var row_bag = List[Int]()
    # NDCG is nonnegative, so the first completed round always sets the
    # best; there is no "zero trees" baseline to beat (see the module note).
    var best_ndcg = -1.0
    var best_n_trees = 0
    for i in range(params.n_estimators):
        _refresh_query_bag(query_bag, row_bag, groups, bagging, i)
        _fill_lambdas(
            raw,
            labels,
            groups,
            inverse_max_dcg,
            discounts,
            rank_params,
            sample_weight,
            grad,
            hess,
        )
        var tree = grow_tree(data, grad, hess, params.tree, row_bag, i)
        if tree.n_leaves == 1 and abs(tree.value[0]) < 1e-12:
            if bagging_enabled(bagging):
                continue
            break

        for r in range(n):
            raw[r] += params.learning_rate * tree.predict_row(data, r)
        for r in range(n_valid):
            valid_raw[r] += (
                params.learning_rate * tree.predict_row(valid_data, r)
            )
        trees.append(tree^)

        var score = ndcg(
            valid_raw, valid_labels, valid_groups, rank_params.ndcg_eval_at
        )
        if score > best_ndcg + min_delta:
            best_ndcg = score
            best_n_trees = len(trees)
        elif len(trees) - best_n_trees >= early_stopping_rounds:
            break

    while len(trees) > best_n_trees:
        _ = trees.pop()
    return Booster(trees^, 0.0, params.learning_rate, LAMBDARANK)


def fit_ranker(
    features: List[Float64],
    n_rows: Int,
    n_features: Int,
    labels: List[Int],
    group_counts: List[Int],
    params: BoosterParams,
    rank_params: RankerParams = RankerParams.default(),
    max_bins: Int = 255,
    sample_weight: List[Float64] = [],
    bagging: BaggingParams = BaggingParams.disabled(),
    use_missing: Bool = True,
) raises -> Model:
    """Fit a ranker on a column-major raw feature matrix
    (`features[f * n_rows + r]`) with LightGBM's `group` array of per-query
    row counts. The returned `Model` predicts raw ranking scores and
    serializes through `save_model` / `load_model` like any single-output
    model; query boundaries are a property of the training data, not of the
    fitted model."""
    var groups = groups_from_counts(group_counts)
    var mapper = fit_bins(
        features, n_rows, n_features, max_bins, use_missing=use_missing
    )
    var data = mapper.transform(features, n_rows)
    var booster = train_ranker(
        data, labels, groups, params, rank_params, sample_weight, bagging
    )
    return Model(mapper^, booster^)
