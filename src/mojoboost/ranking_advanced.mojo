"""Advanced learning to rank: the pieces `ranking.mojo` deliberately left out.

`ranking.mojo` is the authoritative LambdaRank implementation and stays that
way. This module is the layer above it, and it exists for the ranking
features that cannot be expressed as a call into the existing one:

- **unbiased LambdaRank** (LightGBM's `Dataset.position` +
  `lambdarank_position_bias_regularization`): a per-position additive score
  offset, learned by a Newton step on the same lambdas the trees are fitted
  to, so that a document is not rewarded merely for having been shown high
- **a caller-supplied `label_gain` vector**, replacing the fixed `2^l - 1`
- **several evaluation positions at once** (`eval_at`), carried as metadata
  rather than as a bare mean, with the per-query values, the query weights,
  and the counts of the queries that were scored by convention rather than
  measured
- **query weighting** in the metric, LightGBM's `Metadata::query_weights_`
- **deterministic pair sampling**, an extension LightGBM does not have, for
  queries whose document count makes the O(truncation * cnt) pair loop the
  round's cost
- explicit contracts for the three places a query, not a row, is the unit:
  **bagging**, **cross-validation folds**, and **distributed partitions**

WHAT IS AND IS NOT VALIDATED
----------------------------
Nothing here has been run. No test, benchmark, or differential comparison
against LightGBM has been executed for any function in this file, and the
module is not exported from `src/mojoboost/__init__.mojo`, is not reachable
from `bindings/_mojoboost.mojo`, and is named by no parity row. **This
module does not entitle anyone to claim that mojoboost implements unbiased
LambdaRank.** `docs/LIGHTGBM_PARITY.md` keeps
`lambdarank_position_bias_regularization` and `Dataset.position` at
`deferred` until the differential evidence listed in
`handoffs/remaining_05_ranking.md` exists. See docs/RANKING_ADVANCED.md.

HOW DUPLICATION IS AVOIDED
--------------------------
The pair loop is the thing worth not having twice, so this module routes
around writing a second one wherever it can:

- **Position bias is a score offset.** `state.adjust_scores` produces
  `score[r] + bias[position[r]]` and hands it to `ranking._fill_lambdas`
  unchanged. The unbiased objective therefore runs LightGBM's own pair loop,
  through mojoboost's one implementation of it, and adds a per-round Newton
  update afterwards. This is exactly how LightGBM factors it
  (`RankingObjective::GetGradients` builds `score_adjusted` and calls the
  ordinary `GetGradientsForOneQuery`).
- **`AdvancedRankParams.uses_baseline_lambdas`** is the routing predicate.
  When it holds - the default gain vector, no pair sampling, and the maxDCG
  cutoff still tied to the truncation level - every gradient in this module
  comes from `ranking._fill_lambdas` and this file contributes no arithmetic
  at all.
- Only a **custom gain vector** or **pair sampling** reaches
  `_fill_query_lambdas_general`, because neither can be expressed as a
  transform of the scores. That function is a generalization of
  `ranking._fill_query_lambdas`, not a fork of it: with the default gain
  table, a keep rate of 1, and `dcg_cutoff = truncation_level` it must
  reproduce it bit for bit, and the differential check that says so is
  listed UNRUN in the handoff.
- The metrics compose `ranking`'s primitives (`_argsort_desc_range`,
  `_sorted_gains`, `_discounts`, `label_gain`, `max_dcg`) rather than
  recomputing NDCG from scratch. `ranking.ndcg_at_cutoffs` remains the
  authority for the unweighted, default-gain mean; `ndcg_eval` here exists
  because a mean cannot carry per-query values, query weights, a custom gain
  vector, or the degenerate-query counts, and it must agree with it on the
  case they share.

UNBIASED LAMBDARANK, PRECISELY
------------------------------
Each row carries a *position id* in `[0, n_positions)` - which slot of the
result page the document was shown in when the click log was collected. The
model learns one additive bias per position, `b[p]`, in score space. Within
a round:

    adjusted[r] = raw[r] + b[position[r]]
    (grad, hess) = LambdaRank(adjusted)          # unchanged pair loop
    grad, hess  *= sample_weight                 # unchanged
    b[p] += learning_rate * D1[p] / (|D2[p]| + 1e-3)

with, over the rows at position p,

    D1[p] = -sum(grad[r]) - b[p] * reg * count[p]
    D2[p] = -sum(hess[r]) - reg * count[p]

This is LightGBM's `UpdatePositionBiasFactors` term for term, including the
`1e-3` floor on the denominator, the L2 regularization scaled by the row
count at that position, and the use of the *weighted* gradients. Two
consequences that are easy to get wrong and are load bearing here:

- the bias is a training-time nuisance parameter. `predict` must never add
  it, and the metrics in this module score unadjusted scores, because the
  ranking a user is served has no position column yet.
- the biases are state that lives *across* rounds and is not part of the
  fitted model. A serialized ranker does not carry them (see
  docs/RANKING_ADVANCED.md, "What a model file does not hold"), which is why
  continued training on a positioned dataset must be refused rather than
  resumed from zeros.

INTENTIONAL DIFFERENCES FROM LightGBM
-------------------------------------
- A custom `label_gain` must be nondecreasing and start at zero. LightGBM
  only requires that the vector be long enough for the largest label. A
  decreasing entry makes "more relevant label" and "larger gain" disagree,
  and the pair loop picks its `high` document by *label*, so `dcg_gap` would
  go negative and the lambda would push the less relevant document up. That
  is a silent inversion of the objective, so it is rejected.
- `max_dcg_cutoff` can be set independently of `truncation_level`.
  LightGBM ties them. The default (0) keeps them tied, so this is off unless
  asked for.
- Deterministic pair sampling has no LightGBM counterpart at all. It is off
  by default (`pair_sampling_rate = 1.0`) and, when on, rescales every kept
  pair by `1 / rate` so the expected lambda is unchanged. It does *not*
  leave the round bit-identical, and it interacts with `lambdarank_norm`:
  `sum_lambdas` accumulates the rescaled magnitudes, so the per-query
  `log2(1 + sum) / sum` factor sees an estimate rather than the exact sum.
- Queries are the unit of bagging, folding, and partitioning, everywhere,
  as in `ranking.mojo`. `distributed.partition_rows` splits on `r * n // W`
  and would cut a query in half, so ranking uses `partition_queries` here
  instead (see the handoff for the `partition_rows_at` request that makes
  the two meet).
"""

from std.math import isfinite, log2

from .bagging import BaggingParams, bagging_enabled, check_bagging
from .binning import BinnedMatrix, fit_bins
from .boosting import Booster, BoosterParams, _check_sample_weight
from .metrics import METRIC_MAP, METRIC_NDCG
from .model import Model
from .ranking import (
    DEFAULT_TRUNCATION_LEVEL,
    LAMBDARANK,
    MAX_RELEVANCE_LABEL,
    RankGroups,
    RankerParams,
    _argsort_desc_range,
    _discounts,
    _expand_queries,
    _fill_lambdas,
    _inverse_max_dcgs,
    _pair_sigmoid,
    _refresh_query_bag,
    _sorted_gains,
    check_groups,
    check_ranker_params,
    groups_from_counts,
    label_gain,
    max_dcg,
)
from .sampling import _splitmix64, _uniform
from .tree import Tree, grow_tree

# LightGBM's `lambdarank_position_bias_regularization` default: no
# regularization, which is also the value at which the Newton step is the
# plain one.
comptime DEFAULT_POSITION_BIAS_REGULARIZATION = 0.0

# The floor LightGBM puts under the Newton denominator so a position seen by
# no row, or by rows whose hessians all vanished, cannot divide by zero.
comptime POSITION_BIAS_EPSILON = 1e-3

# Pair sampling is off by default: every pair the truncation level admits is
# visited, which is LightGBM's behavior and the only one with parity
# evidence behind it.
comptime DEFAULT_PAIR_SAMPLING_RATE = 1.0

# LightGBM's `objective_seed`. The pair draw is the objective's own
# randomness, distinct from `bagging_seed` (3), `feature_fraction_seed` (2),
# and `extra_seed` (6), so it takes the seed LightGBM reserves for it.
comptime DEFAULT_PAIR_SAMPLING_SEED = 5

# Domain separator for the pair-sampling stream, so a (seed, iteration,
# query) triple can never collide with the (seed, tree, tag) streams
# sampling.mojo derives or the (seed, bag) streams bagging.mojo derives.
comptime _PAIR_DOMAIN = UInt64(0xD1B54A32D192ED03)

# Domain separator for the fold shuffle, for the same reason.
comptime _FOLD_DOMAIN = UInt64(0x27BB2EE687B0B0FD)

comptime _GOLDEN = UInt64(0x9E3779B97F4A7C15)

# What to do with a query the `group` array says holds no rows. LightGBM's
# reader rejects one; `ranking.groups_from_counts` rejects one too. DROP is
# offered because an empty query owns no rows, so removing it renumbers
# queries and leaves every row index untouched (see `sanitize_group_counts`).
comptime EMPTY_QUERY_RAISE = 0
comptime EMPTY_QUERY_DROP = 1


def default_eval_at() -> List[Int]:
    """LightGBM's default `eval_at` for the ranking metrics: positions 1
    through 5. A function rather than a constant because a `List` is a
    runtime value; every caller gets its own copy to own."""
    return [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Label gains
# ---------------------------------------------------------------------------


struct LabelGain(Copyable, Movable):
    """The gain a relevance label is worth, LightGBM's `label_gain`.

    `gains[l]` is the gain of label `l`, so the vector's length is the
    largest label the data may carry plus one. `LabelGain.default()` is
    LightGBM's default and mojoboost's only shipped one: `2^l - 1` for
    `l` in `[0, 30]`, the same table `ranking.label_gain` computes, built
    from that function so the two cannot drift.

    The `is_lightgbm_default` flag is not cosmetic. It is the routing bit
    that sends a round to `ranking._fill_lambdas` instead of to this
    module's generalized kernel, so it is set by `default()` alone and never
    inferred from the contents: a caller who passes an identical table by
    hand gets the general path, which computes the same numbers more slowly.
    """

    var gains: List[Float64]
    var is_lightgbm_default: Bool

    def __init__(out self, var gains: List[Float64]):
        """A caller-supplied gain vector. Validate it with
        `check_label_gain` before training on it."""
        self.gains = gains^
        self.is_lightgbm_default = False

    @staticmethod
    def default() -> LabelGain:
        """LightGBM's default gain vector, `2^l - 1` for labels 0..30."""
        var g = List[Float64](capacity=MAX_RELEVANCE_LABEL + 1)
        for l in range(MAX_RELEVANCE_LABEL + 1):
            g.append(label_gain(l))
        var out = LabelGain(g^)
        out.is_lightgbm_default = True
        return out^

    def max_label(self) -> Int:
        """The largest label this vector can score."""
        return len(self.gains) - 1

    def of(self, label: Int) -> Float64:
        """The gain of `label`. The caller has already validated the label
        against `max_label`, which `check_relevance_labels` does."""
        return self.gains[label]


def check_label_gain(gain: LabelGain) raises:
    """Validate a gain vector: nonempty, finite, nonnegative, nondecreasing,
    and starting at zero.

    The nondecreasing rule is stricter than LightGBM and is argued for in
    the module docstring: the pair loop chooses which document to push up by
    *label*, so a gain that falls as the label rises inverts the objective
    without any error surfacing. `gains[0] == 0` is the definition of an
    irrelevant document contributing nothing to DCG; LightGBM's default
    satisfies it, and a table that does not would add a constant to every
    query's DCG that does not cancel in the NDCG ratio."""
    if len(gain.gains) == 0:
        raise Error("label_gain must have at least one entry")
    if gain.gains[0] != 0.0:
        raise Error("label_gain[0] must be 0: label 0 is the irrelevant one")
    for l in range(len(gain.gains)):
        var g = gain.gains[l]
        if not isfinite(g):
            raise Error("label_gain entries must be finite")
        if g < 0.0:
            raise Error("label_gain entries must be nonnegative")
        if l > 0 and g < gain.gains[l - 1]:
            raise Error(
                "label_gain must be nondecreasing: entry ",
                l,
                " is smaller than entry ",
                l - 1,
                ", which would make a more relevant label worth less and"
                " invert the pairwise objective",
            )


def check_relevance_labels(labels: List[Int], gain: LabelGain) raises:
    """Relevance labels must index the gain vector.

    `ranking.check_labels` fixes the range at `[0, 30]` because the default
    table is the only one it knows. This is the same check against whatever
    table the caller supplied, and it agrees with `ranking.check_labels`
    exactly when that table is `LabelGain.default()`."""
    var top = gain.max_label()
    for r in range(len(labels)):
        if labels[r] < 0:
            raise Error("relevance labels must be nonnegative")
        if labels[r] > top:
            raise Error(
                "relevance label ",
                labels[r],
                " has no gain: label_gain covers 0..",
                top,
            )


def _sorted_gains_table(
    labels: List[Int], start: Int, cnt: Int, gain: LabelGain
) -> List[Float64]:
    """The query's gains in descending order, by counting sort over the gain
    vector's label range.

    The generalization of `ranking._sorted_gains` to a caller-supplied
    table. Descending label order is descending gain order because
    `check_label_gain` requires the table to be nondecreasing, which is what
    makes one counting sort enough."""
    var top = gain.max_label()
    var counts = List[Int](capacity=top + 1)
    for _ in range(top + 1):
        counts.append(0)
    for i in range(cnt):
        counts[labels[start + i]] += 1
    var out = List[Float64](capacity=cnt)
    for label in range(top, -1, -1):
        for _ in range(counts[label]):
            out.append(gain.of(label))
    return out^


def _query_sorted_gains(
    labels: List[Int], start: Int, cnt: Int, gain: LabelGain
) -> List[Float64]:
    """Descending gains for one query, from `ranking._sorted_gains` when the
    table is the default one and from `_sorted_gains_table` otherwise."""
    if gain.is_lightgbm_default:
        return _sorted_gains(labels, start, cnt)
    return _sorted_gains_table(labels, start, cnt, gain)


def ideal_dcg(
    labels: List[Int], start: Int, cnt: Int, k: Int, gain: LabelGain
) raises -> Float64:
    """The best DCG at cutoff k this query's labels can reach under `gain`.

    `ranking.max_dcg` for the default table, so the default path keeps one
    implementation; the counting sort here for any other."""
    if gain.is_lightgbm_default:
        return max_dcg(labels, start, cnt, k)
    var gains = _sorted_gains_table(labels, start, cnt, gain)
    var m = k if k < cnt else cnt
    var discounts = _discounts(m)
    var total = 0.0
    for j in range(m):
        total += gains[j] * discounts[j]
    return total


def _inverse_max_dcgs_table(
    labels: List[Int], groups: RankGroups, k: Int, gain: LabelGain
) raises -> List[Float64]:
    """`1 / maxDCG@k` per query, or 0.0 for a query with nothing to gain,
    which zeroes that query's lambdas as in LightGBM.

    `ranking._inverse_max_dcgs` when the table is the default one."""
    if gain.is_lightgbm_default:
        return _inverse_max_dcgs(labels, groups, k)
    var out = List[Float64](capacity=groups.n_queries())
    for q in range(groups.n_queries()):
        var m = ideal_dcg(labels, groups.start(q), groups.size(q), k, gain)
        out.append(1.0 / m if m > 0.0 else 0.0)
    return out^


# ---------------------------------------------------------------------------
# Positions and the position-bias state
# ---------------------------------------------------------------------------


@fieldwise_init
struct PositionMap(Copyable, Movable):
    """Which slot of the result page each row was shown in.

    LightGBM's `Dataset.position` field: `ids[r]` is a dense position id in
    `[0, n_positions)`, and `n_positions` is `Metadata::num_position_ids`.
    An empty map (`PositionMap.absent()`) means the dataset carries no
    position column, which is the ordinary biased LambdaRank case and the
    only case with parity evidence behind it.

    Build one with `positions_from_codes`, which validates by construction,
    or validate a hand-built one with `check_positions`. This follows
    `RankGroups`: the struct holds the arrays, a checker holds the
    invariants, so a caller cannot be surprised by a constructor that
    raises.

    Position ids are a property of the *training log*, not of the model, and
    they are not needed to predict: a served ranking has no positions yet.
    """

    var ids: List[Int]
    var n_positions: Int

    @staticmethod
    def absent() -> PositionMap:
        """No position column: ordinary LambdaRank."""
        return PositionMap(List[Int](), 0)

    def is_absent(self) -> Bool:
        return self.n_positions == 0 or len(self.ids) == 0

    def n_rows(self) -> Int:
        return len(self.ids)


struct PositionEncoding(Copyable, Movable):
    """A dense `PositionMap` plus the original codes its ids stand for.

    `codes[p]` is the caller's code for dense id `p`, so a caller can report
    a learned bias against the position it actually named."""

    var positions: PositionMap
    var codes: List[Int]

    def __init__(out self, var positions: PositionMap, var codes: List[Int]):
        self.positions = positions^
        self.codes = codes^


def positions_from_codes(codes: List[Int]) raises -> PositionEncoding:
    """Dense position ids from an arbitrary per-row integer code column.

    Ids are assigned in order of first appearance, which is deterministic in
    the row order and needs no sort, so two runs over the same data agree and
    a learned bias vector means the same thing in both. Unlike query ids,
    positions may repeat and interleave freely: a position is a label on a
    row, not a run of rows, so nothing here rejects a code that reappears."""
    if len(codes) == 0:
        raise Error("position codes must contain at least one row")
    var table = List[Int]()
    var ids = List[Int](capacity=len(codes))
    for r in range(len(codes)):
        var found = -1
        for p in range(len(table)):
            if table[p] == codes[r]:
                found = p
                break
        if found < 0:
            found = len(table)
            table.append(codes[r])
        ids.append(found)
    var n = len(table)
    return PositionEncoding(PositionMap(ids^, n), table^)


def check_positions(positions: PositionMap, n_rows: Int) raises:
    """Validate a position map against a dataset of `n_rows` rows. An absent
    map passes: it is the ordinary no-position case."""
    if positions.n_positions < 0:
        raise Error("n_positions must be nonnegative")
    if positions.n_positions == 0:
        if len(positions.ids) != 0:
            raise Error("a position map with no positions must have no rows")
        return
    if len(positions.ids) != n_rows:
        raise Error("position ids must have one entry per row")
    for r in range(len(positions.ids)):
        if positions.ids[r] < 0 or positions.ids[r] >= positions.n_positions:
            raise Error(
                "position id ",
                positions.ids[r],
                " is outside [0, ",
                positions.n_positions,
                ")",
            )


struct PositionBiasState(Copyable, Movable, Writable):
    """The learned per-position score offsets, and the evidence behind them.

    `biases[p]` is added to the raw score of every row shown at position p
    before the pair loop runs, and is updated once per round by a Newton
    step on the same lambdas the trees are fitted to. `counts[p]` and the
    two derivative vectors record the last update, so a caller can report
    why a bias moved (or did not) without instrumenting the trainer.

    This is **training state, not model state.** A fitted `Booster` does not
    carry it and `serialize.save_model` does not write it; the biases exist
    to make the trees fitted alongside them unbiased, and the trees are what
    is served. That is also why continued training on a positioned dataset
    must be refused rather than resumed: resuming from zero biases would fit
    the next round's trees against a correction the earlier rounds already
    made.
    """

    var biases: List[Float64]
    var counts: List[Int]
    var last_first_derivative: List[Float64]
    var last_second_derivative: List[Float64]
    var updates: Int

    def __init__(out self, n_positions: Int) raises:
        if n_positions < 0:
            raise Error("n_positions must be nonnegative")
        self.biases = List[Float64](capacity=n_positions)
        self.counts = List[Int](capacity=n_positions)
        self.last_first_derivative = List[Float64](capacity=n_positions)
        self.last_second_derivative = List[Float64](capacity=n_positions)
        for _ in range(n_positions):
            self.biases.append(0.0)
            self.counts.append(0)
            self.last_first_derivative.append(0.0)
            self.last_second_derivative.append(0.0)
        self.updates = 0

    @staticmethod
    def for_positions(positions: PositionMap) raises -> PositionBiasState:
        """A zeroed state sized for `positions`, or an empty one when the
        dataset carries no position column."""
        return PositionBiasState(positions.n_positions)

    def n_positions(self) -> Int:
        return len(self.biases)

    def is_absent(self) -> Bool:
        return len(self.biases) == 0

    def adjust_scores(
        self, scores: List[Float64], positions: PositionMap
    ) raises -> List[Float64]:
        """`scores[r] + biases[positions.ids[r]]`, the vector the pair loop
        actually sees. Returns a copy of `scores` unchanged when there is no
        position column, so the caller has one code path."""
        if self.is_absent() or positions.is_absent():
            return scores.copy()
        if len(positions.ids) != len(scores):
            raise Error("position ids must have one entry per row")
        var out = List[Float64](capacity=len(scores))
        for r in range(len(scores)):
            out.append(scores[r] + self.biases[positions.ids[r]])
        return out^

    def write_to(self, mut writer: Some[Writer]):
        writer.write("PositionBiasState(updates=", self.updates, ", biases=[")
        for p in range(len(self.biases)):
            if p > 0:
                writer.write(", ")
            writer.write(self.biases[p])
        writer.write("])")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def update_position_bias(
    mut state: PositionBiasState,
    positions: PositionMap,
    grad: List[Float64],
    hess: List[Float64],
    regularization: Float64,
    learning_rate: Float64,
) raises:
    """One Newton step on the position biases, LightGBM's
    `UpdatePositionBiasFactors`.

    `grad` and `hess` are the round's lambdas and hessians *after* sample
    weights have been applied, which is the order LightGBM uses: a row that
    counts for less in the trees counts for less in the bias too. The
    derivatives are of the utility, so they are the negated gradients; the
    L2 term is scaled by the number of rows at the position, so a position
    seen a thousand times is not shrunk as hard as one seen twice; and the
    denominator carries the 1e-3 floor so a position with no rows leaves its
    bias where it was.

    Does nothing when there is no position column, so a caller may call it
    unconditionally."""
    if state.is_absent() or positions.is_absent():
        return
    if regularization < 0.0:
        raise Error(
            "lambdarank_position_bias_regularization must be nonnegative"
        )
    if len(positions.ids) != len(grad):
        raise Error("position ids must have one entry per row")
    if len(hess) != len(grad):
        raise Error("gradients and hessians must have the same length")

    var n_pos = state.n_positions()
    var d1 = List[Float64](capacity=n_pos)
    var d2 = List[Float64](capacity=n_pos)
    var counts = List[Int](capacity=n_pos)
    for _ in range(n_pos):
        d1.append(0.0)
        d2.append(0.0)
        counts.append(0)

    for r in range(len(grad)):
        var p = positions.ids[r]
        d1[p] -= grad[r]
        d2[p] -= hess[r]
        counts[p] += 1

    for p in range(n_pos):
        var count = Float64(counts[p])
        var first = d1[p] - state.biases[p] * regularization * count
        var second = d2[p] - regularization * count
        state.biases[p] += (
            learning_rate * first / (abs(second) + POSITION_BIAS_EPSILON)
        )
        state.counts[p] = counts[p]
        state.last_first_derivative[p] = first
        state.last_second_derivative[p] = second
    state.updates += 1


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@fieldwise_init
struct AdvancedRankParams(Copyable, Movable):
    """Everything `RankerParams` does not carry.

    `base` holds LightGBM's `lambdarank_truncation_level`, `sigmoid`,
    `lambdarank_norm`, and `ndcg_eval_at`, unchanged and still validated by
    `ranking.check_ranker_params`. The rest:

    - `gain` is LightGBM's `label_gain`
    - `eval_at` is LightGBM's list of evaluation positions, which the
      metrics in this module report all of at once. `base.ndcg_eval_at`
      stays the single cutoff early stopping watches, and must appear in
      `eval_at` so that the number a run stops on is one of the numbers it
      reports
    - `max_dcg_cutoff` decouples the maxDCG the lambdas are normalized
      against from the truncation level; 0 keeps LightGBM's tie
    - `position_bias_regularization` is LightGBM's
      `lambdarank_position_bias_regularization`
    - `pair_sampling_rate` and `pair_sampling_seed` drive the deterministic
      pair draw, which is off at rate 1.0
    - `weight_queries` makes the metrics average per-query values under
      LightGBM's query weights (the mean of a query's row weights) instead
      of unweighted
    - `empty_query_policy` is what `sanitize_group_counts` does with a query
      the `group` array says holds no rows
    """

    var base: RankerParams
    var gain: LabelGain
    var eval_at: List[Int]
    var max_dcg_cutoff: Int
    var position_bias_regularization: Float64
    var pair_sampling_rate: Float64
    var pair_sampling_seed: Int
    var weight_queries: Bool
    var empty_query_policy: Int

    @staticmethod
    def default() -> AdvancedRankParams:
        """LightGBM's defaults, and every extension off: the configuration
        under which this module's gradients are `ranking._fill_lambdas`'s
        gradients and nothing else."""
        return AdvancedRankParams(
            RankerParams.default(),
            LabelGain.default(),
            default_eval_at(),
            0,
            DEFAULT_POSITION_BIAS_REGULARIZATION,
            DEFAULT_PAIR_SAMPLING_RATE,
            DEFAULT_PAIR_SAMPLING_SEED,
            False,
            EMPTY_QUERY_RAISE,
        )

    def dcg_cutoff(self) -> Int:
        """The cutoff the lambdas' maxDCG is taken at: the truncation level
        unless `max_dcg_cutoff` overrides it."""
        if self.max_dcg_cutoff > 0:
            return self.max_dcg_cutoff
        return self.base.truncation_level

    def samples_pairs(self) -> Bool:
        return self.pair_sampling_rate < 1.0

    def uses_baseline_lambdas(self) -> Bool:
        """Whether a round can be computed by `ranking._fill_lambdas`.

        Position bias is deliberately not part of this test: it is a score
        offset, so the baseline kernel computes the unbiased round too. Only
        a custom gain vector, a decoupled maxDCG cutoff, or pair sampling
        force the generalized kernel."""
        return (
            self.gain.is_lightgbm_default
            and self.max_dcg_cutoff == 0
            and not self.samples_pairs()
        )


def check_advanced_rank_params(params: AdvancedRankParams) raises:
    """Validate everything that does not depend on the data."""
    check_ranker_params(params.base)
    check_label_gain(params.gain)
    if len(params.eval_at) == 0:
        raise Error("eval_at must name at least one position")
    for i in range(len(params.eval_at)):
        if params.eval_at[i] < 1:
            raise Error("eval_at positions must be positive")
        for j in range(i):
            if params.eval_at[j] == params.eval_at[i]:
                raise Error(
                    "eval_at position ", params.eval_at[i], " is listed twice"
                )
    var watched = False
    for i in range(len(params.eval_at)):
        if params.eval_at[i] == params.base.ndcg_eval_at:
            watched = True
    if not watched:
        raise Error(
            "ndcg_eval_at (",
            params.base.ndcg_eval_at,
            ") must appear in eval_at, so the cutoff early stopping watches"
            " is one of the cutoffs the run reports",
        )
    if params.max_dcg_cutoff < 0:
        raise Error(
            "max_dcg_cutoff must be nonnegative (0 follows"
            " lambdarank_truncation_level)"
        )
    if params.position_bias_regularization < 0.0:
        raise Error(
            "lambdarank_position_bias_regularization must be nonnegative"
        )
    if not (
        params.pair_sampling_rate > 0.0 and params.pair_sampling_rate <= 1.0
    ):
        raise Error("pair_sampling_rate must be in (0, 1]")
    if (
        params.empty_query_policy != EMPTY_QUERY_RAISE
        and params.empty_query_policy != EMPTY_QUERY_DROP
    ):
        raise Error("empty_query_policy must be EMPTY_QUERY_RAISE or DROP")


# ---------------------------------------------------------------------------
# Deterministic pair sampling
# ---------------------------------------------------------------------------


def _pair_stream(seed: Int, iteration: Int, query: Int) -> UInt64:
    """Start of the counter stream for one query's pair draw in one round.

    Counter based, like every other draw in mojoboost: the decision for pair
    (i, j) depends only on (seed, iteration, query, i, j), so it survives
    reordering, threading, bagging, and being replayed from the middle of a
    run. Sign bits are masked off so negative seeds are accepted."""
    var h = _splitmix64(UInt64(seed & 0x7FFFFFFFFFFFFFFF) ^ _PAIR_DOMAIN)
    h = _splitmix64(h ^ UInt64(iteration & 0x7FFFFFFFFFFFFFFF))
    return _splitmix64(h ^ UInt64(query & 0x7FFFFFFFFFFFFFFF))


def _pair_kept(stream: UInt64, i: Int, j: Int, rate: Float64) -> Bool:
    """Whether the pair at ranks (i, j) survives the draw. The two ranks are
    mixed rather than added so that (i, j) and (j, i) - and every other pair
    with the same sum - draw independently."""
    var counter = _splitmix64(stream ^ (UInt64(i) * _GOLDEN)) + UInt64(j)
    return _uniform(counter) < rate


def pair_budget(groups: RankGroups, truncation_level: Int) -> Int:
    """How many ordered pairs the truncated loop would visit over every
    query, ignoring labels.

    The cost model behind `pair_sampling_rate`: a query of `cnt` documents
    contributes the sum over `i < min(cnt - 1, truncation)` of
    `cnt - 1 - i`. Label ties only remove work from this, never add it, so
    this is an upper bound on the round's pair count and
    `GroupAudit.n_pairs` is the exact one."""
    var total = 0
    for q in range(groups.n_queries()):
        var cnt = groups.size(q)
        var top = cnt - 1
        if truncation_level < top:
            top = truncation_level
        for i in range(top):
            total += cnt - 1 - i
    return total


# ---------------------------------------------------------------------------
# The generalized pair loop
# ---------------------------------------------------------------------------


def _fill_query_lambdas_general(
    scores: List[Float64],
    labels: List[Int],
    start: Int,
    cnt: Int,
    inverse_max_dcg: Float64,
    discounts: List[Float64],
    params: AdvancedRankParams,
    pair_stream: UInt64,
    mut grad: List[Float64],
    mut hess: List[Float64],
):
    """`ranking._fill_query_lambdas` generalized over the gain vector and the
    pair draw.

    Every line that is not about those two is the same line, and with
    `LabelGain.default()` and `pair_sampling_rate = 1.0` this must produce
    bit-identical gradients to `ranking._fill_query_lambdas`; the routing in
    `uses_baseline_lambdas` means that configuration never actually gets
    here, and the differential check that pins the equality is listed UNRUN
    in the handoff.

    Zero- and one-document queries return zeroed gradients without visiting
    a pair, which is not a special case but what `cnt < 2` already means: a
    query with nothing to compare cannot express a preference, so it must
    not move any leaf.
    """
    for i in range(cnt):
        grad[start + i] = 0.0
        hess[start + i] = 0.0
    if cnt < 2:
        return

    var order = _argsort_desc_range(scores, start, cnt)
    var best_score = scores[start + order[0]]
    var worst_score = scores[start + order[cnt - 1]]
    var sum_lambdas = 0.0
    var sampling = params.pair_sampling_rate < 1.0
    var pair_scale = 1.0 / params.pair_sampling_rate

    var top = cnt - 1
    if params.base.truncation_level < top:
        top = params.base.truncation_level
    for i in range(top):
        for j in range(i + 1, cnt):
            # Tied labels express no preference, so the pair carries no
            # information in either direction and is skipped before the draw.
            # Skipping it after would spend randomness on nothing and make
            # which pairs survive depend on the labels.
            if labels[start + order[i]] == labels[start + order[j]]:
                continue
            if sampling and not _pair_kept(
                pair_stream, i, j, params.pair_sampling_rate
            ):
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
            var dcg_gap = params.gain.of(labels[high]) - params.gain.of(
                labels[low]
            )
            var paired_discount = abs(
                discounts[high_rank] - discounts[low_rank]
            )
            var delta_ndcg = dcg_gap * paired_discount * inverse_max_dcg
            if params.base.norm and best_score != worst_score:
                delta_ndcg /= 0.01 + abs(delta_score)
            if sampling:
                # Inverse probability weighting: the kept pairs stand in for
                # the dropped ones, so the expected lambda is unchanged.
                delta_ndcg *= pair_scale

            var p_lambda = _pair_sigmoid(delta_score, params.base.sigmoid)
            var p_hess = p_lambda * (1.0 - p_lambda)
            p_lambda *= -params.base.sigmoid * delta_ndcg
            p_hess *= params.base.sigmoid * params.base.sigmoid * delta_ndcg

            grad[low] -= p_lambda
            hess[low] += p_hess
            grad[high] += p_lambda
            hess[high] += p_hess
            # p_lambda is negative, so subtract to accumulate its magnitude.
            sum_lambdas -= 2.0 * p_lambda

    if params.base.norm and sum_lambdas > 0.0:
        var factor = log2(1.0 + sum_lambdas) / sum_lambdas
        for i in range(cnt):
            grad[start + i] *= factor
            hess[start + i] *= factor


def _fill_lambdas_general(
    scores: List[Float64],
    labels: List[Int],
    groups: RankGroups,
    inverse_max_dcg: List[Float64],
    discounts: List[Float64],
    params: AdvancedRankParams,
    weights: List[Float64],
    iteration: Int,
    mut grad: List[Float64],
    mut hess: List[Float64],
):
    """The generalized kernel over every query, then per-row sample weights,
    applied after per-query normalization exactly as `ranking._fill_lambdas`
    applies them."""
    var n = groups.n_rows
    grad.clear()
    grad.resize(n, 0.0)
    hess.clear()
    hess.resize(n, 0.0)
    for q in range(groups.n_queries()):
        _fill_query_lambdas_general(
            scores,
            labels,
            groups.start(q),
            groups.size(q),
            inverse_max_dcg[q],
            discounts,
            params,
            _pair_stream(params.pair_sampling_seed, iteration, q),
            grad,
            hess,
        )
    if len(weights) > 0:
        for r in range(n):
            grad[r] *= weights[r]
            hess[r] *= weights[r]


def advanced_lambdarank_gradients(
    scores: List[Float64],
    labels: List[Int],
    groups: RankGroups,
    positions: PositionMap,
    mut bias: PositionBiasState,
    mut grad: List[Float64],
    mut hess: List[Float64],
    params: AdvancedRankParams = AdvancedRankParams.default(),
    sample_weight: List[Float64] = [],
    iteration: Int = 0,
    learning_rate: Float64 = 0.1,
) raises:
    """One round of gradients, with every advanced feature that applies.

    The whole round, in order, and each step is the LightGBM step:

    1. the raw scores are offset by the current position biases
    2. the lambdas are computed on the offset scores - by
       `ranking._fill_lambdas` whenever `params.uses_baseline_lambdas()`,
       which is every configuration except a custom gain vector, a decoupled
       maxDCG cutoff, or pair sampling
    3. sample weights scale the per-row lambdas and hessians
    4. the position biases take one Newton step on those weighted lambdas

    `iteration` and `learning_rate` are the boosting round index and the
    boosting learning rate. The first keys the pair draw, so a round is
    reproducible from its index alone; the second is the step size of the
    bias update, as it is in LightGBM, where the objective reads
    `config.learning_rate` for exactly this."""
    check_groups(groups, len(scores))
    if len(labels) != len(scores):
        raise Error("labels length must equal scores length")
    check_relevance_labels(labels, params.gain)
    check_advanced_rank_params(params)
    check_positions(positions, len(scores))
    _check_sample_weight(sample_weight, len(scores))
    if not positions.is_absent():
        if bias.n_positions() != positions.n_positions:
            raise Error(
                "the position bias state was built for a different number of"
                " positions than the position map names"
            )
    if iteration < 0:
        raise Error("iteration must be nonnegative")

    var adjusted = bias.adjust_scores(scores, positions)
    var inv = _inverse_max_dcgs_table(
        labels, groups, params.dcg_cutoff(), params.gain
    )
    var discounts = _discounts(groups.max_size())

    if params.uses_baseline_lambdas():
        _fill_lambdas(
            adjusted,
            labels,
            groups,
            inv,
            discounts,
            params.base,
            sample_weight,
            grad,
            hess,
        )
    else:
        _fill_lambdas_general(
            adjusted,
            labels,
            groups,
            inv,
            discounts,
            params,
            sample_weight,
            iteration,
            grad,
            hess,
        )

    update_position_bias(
        bias,
        positions,
        grad,
        hess,
        params.position_bias_regularization,
        learning_rate,
    )


# ---------------------------------------------------------------------------
# Group audit, sanitation, and pruning
# ---------------------------------------------------------------------------


@fieldwise_init
struct GroupAudit(Copyable, Movable, Writable):
    """What a query set is actually made of.

    Ranking data goes wrong quietly. A query with one document, a query
    whose labels are all equal, and a query whose labels are all zero each
    contribute *nothing* to the objective while still counting as a query in
    the metric's denominator, so a set made mostly of them trains on far
    less than its row count suggests and reports an NDCG far higher than the
    ranking deserves. This counts them.
    """

    var n_queries: Int
    var n_rows: Int
    var min_size: Int
    var max_size: Int
    var n_single_document: Int
    var n_all_zero_label: Int
    var n_tied_label: Int
    var n_pairable: Int
    var n_pairs: Int

    def trains_on_nothing(self) -> Bool:
        """Whether no query in the set can produce a single lambda."""
        return self.n_pairs == 0

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "GroupAudit(queries=",
            self.n_queries,
            ", rows=",
            self.n_rows,
            ", sizes=[",
            self.min_size,
            ", ",
            self.max_size,
            "], single_document=",
            self.n_single_document,
            ", all_zero_label=",
            self.n_all_zero_label,
            ", tied_label=",
            self.n_tied_label,
            ", pairable=",
            self.n_pairable,
            ", pairs=",
            self.n_pairs,
            ")",
        )

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def audit_groups(
    labels: List[Int],
    groups: RankGroups,
    truncation_level: Int = DEFAULT_TRUNCATION_LEVEL,
) raises -> GroupAudit:
    """Count the degenerate shapes in a query set.

    `n_tied_label` counts queries whose labels are all equal, which includes
    the all-zero ones and the single-document ones, because all three are
    the same fact: no pair of documents in the query disagrees.
    `n_pairable` counts the queries that can produce at least one lambda,
    and `n_pairs` the pairs the truncated loop would actually visit - the
    label-differing subset of `pair_budget`, and the honest measure of a
    round's cost.

    Ranks are not known before the scores are, so `n_pairs` counts pairs of
    *positions in the input order* under the truncation. It is exact when
    the truncation level is at least the largest query, which is the default
    for query sets of 30 documents or fewer, and an estimate above that."""
    check_groups(groups, groups.n_rows)
    if len(labels) != groups.n_rows:
        raise Error("labels length must equal the group row count")
    if truncation_level < 1:
        raise Error("truncation_level must be positive")

    var min_size = groups.size(0)
    var max_size = 0
    var n_single = 0
    var n_zero = 0
    var n_tied = 0
    var n_pairable = 0
    var n_pairs = 0

    for q in range(groups.n_queries()):
        var start = groups.start(q)
        var cnt = groups.size(q)
        if cnt < min_size:
            min_size = cnt
        if cnt > max_size:
            max_size = cnt
        if cnt == 1:
            n_single += 1

        var all_zero = True
        var all_equal = True
        for i in range(cnt):
            if labels[start + i] != 0:
                all_zero = False
            if labels[start + i] != labels[start]:
                all_equal = False
        if all_zero:
            n_zero += 1
        if all_equal:
            n_tied += 1

        var top = cnt - 1
        if truncation_level < top:
            top = truncation_level
        var pairs = 0
        for i in range(top):
            for j in range(i + 1, cnt):
                if labels[start + i] != labels[start + j]:
                    pairs += 1
        # A query with no attainable DCG has every lambda zeroed by its
        # inverse maxDCG, whatever its pairs say, so it is not pairable.
        if pairs > 0 and not all_zero:
            n_pairable += 1
            n_pairs += pairs

    return GroupAudit(
        groups.n_queries(),
        groups.n_rows,
        min_size,
        max_size,
        n_single,
        n_zero,
        n_tied,
        n_pairable,
        n_pairs,
    )


struct SanitizedGroups(Copyable, Movable):
    """A repaired `group` array and what was removed to get it.

    `kept[i]` is the original index of the query that became query `i`, so a
    caller can map a per-query result back to the array it was given."""

    var counts: List[Int]
    var kept: List[Int]
    var n_dropped: Int

    def __init__(
        out self, var counts: List[Int], var kept: List[Int], n_dropped: Int
    ):
        self.counts = counts^
        self.kept = kept^
        self.n_dropped = n_dropped


def sanitize_group_counts(
    counts: List[Int], policy: Int = EMPTY_QUERY_RAISE
) raises -> SanitizedGroups:
    """Repair a `group` array that names queries holding no rows.

    Under `EMPTY_QUERY_RAISE` this is `ranking.groups_from_counts`'s rule and
    an empty query is an error. Under `EMPTY_QUERY_DROP` empty queries are
    removed, and the operation is safe in the one way that matters: **an
    empty query owns no rows, so dropping it renumbers queries and leaves
    every row index, label, weight, and position exactly where it was.** No
    other degenerate query has that property, which is why no other one is
    dropped here (see `prunable_queries`).

    A negative count is always an error: it is not an empty query, it is a
    corrupt array, and the rows it claims are unaccounted for."""
    if policy != EMPTY_QUERY_RAISE and policy != EMPTY_QUERY_DROP:
        raise Error("empty_query_policy must be EMPTY_QUERY_RAISE or DROP")
    if len(counts) == 0:
        raise Error("group must contain at least one query")

    var out = List[Int]()
    var kept = List[Int]()
    var dropped = 0
    for q in range(len(counts)):
        if counts[q] < 0:
            raise Error("group counts must not be negative")
        if counts[q] == 0:
            if policy == EMPTY_QUERY_RAISE:
                raise Error("group counts must be positive")
            dropped += 1
            continue
        out.append(counts[q])
        kept.append(q)
    if len(out) == 0:
        raise Error("every query in group is empty; nothing to train on")
    return SanitizedGroups(out^, kept^, dropped)


def groups_from_counts_sanitized(
    counts: List[Int], policy: Int = EMPTY_QUERY_RAISE
) raises -> RankGroups:
    """`ranking.groups_from_counts` after `sanitize_group_counts`, so a
    caller that tolerates empty queries has one call rather than two."""
    var clean = sanitize_group_counts(counts, policy)
    return groups_from_counts(clean.counts)


def prunable_queries(
    labels: List[Int],
    groups: RankGroups,
    truncation_level: Int = DEFAULT_TRUNCATION_LEVEL,
) raises -> List[Int]:
    """The queries that contribute exactly zero to the objective: no pair of
    differing labels within the truncation, or no attainable DCG at all.

    Dropping these from **training** changes no gradient, because each of
    their rows already receives a lambda of exactly 0.0. It is not free,
    though: `min_data_in_leaf` and every other count-based rule *do* see
    those rows, so pruning them changes which splits are legal. Treat this
    as a cost decision, not an identity.

    Dropping them from **evaluation** is not safe under any reading. They
    are counted as 1.0 by both NDCG and MAP, so removing them changes the
    reported number; `RankEval.n_degenerate` reports how many there were
    instead."""
    if len(labels) != groups.n_rows:
        raise Error("labels length must equal the group row count")
    if truncation_level < 1:
        raise Error("truncation_level must be positive")
    var out = List[Int]()
    for q in range(groups.n_queries()):
        var start = groups.start(q)
        var cnt = groups.size(q)
        var all_zero = True
        for i in range(cnt):
            if labels[start + i] != 0:
                all_zero = False
                break
        if all_zero:
            out.append(q)
            continue
        var top = cnt - 1
        if truncation_level < top:
            top = truncation_level
        var pairs = 0
        for i in range(top):
            for j in range(i + 1, cnt):
                if labels[start + i] != labels[start + j]:
                    pairs += 1
                    break
            if pairs > 0:
                break
        if pairs == 0:
            out.append(q)
    return out^


def rows_of_queries(
    groups: RankGroups, queries: List[Int]
) raises -> List[Int]:
    """Every row of the named queries, ascending, provided the query list is
    ascending - which is what `ranking._expand_queries` assumes and what
    every producer in this module hands it."""
    for i in range(len(queries)):
        if queries[i] < 0 or queries[i] >= groups.n_queries():
            raise Error("query index out of range")
        if i > 0 and queries[i] <= queries[i - 1]:
            raise Error("query indices must be strictly ascending")
    var rows = List[Int]()
    _expand_queries(groups, queries, rows)
    return rows^


def group_counts_of_queries(
    groups: RankGroups, queries: List[Int]
) raises -> List[Int]:
    """The `group` array of a subset of queries, in the order given: the
    companion of `rows_of_queries`, since a row list without its query
    boundaries is not a ranking dataset."""
    var out = List[Int](capacity=len(queries))
    for i in range(len(queries)):
        if queries[i] < 0 or queries[i] >= groups.n_queries():
            raise Error("query index out of range")
        out.append(groups.size(queries[i]))
    return out^


# ---------------------------------------------------------------------------
# Query weighting and metric metadata
# ---------------------------------------------------------------------------


def query_weights(
    groups: RankGroups, sample_weight: List[Float64]
) raises -> List[Float64]:
    """One weight per query, LightGBM's `Metadata::CalculateQueryWeights`:
    the **mean** of the query's row weights, not their sum.

    The mean is the right rule and the choice is not arbitrary. A query's
    NDCG is already normalized by its own maxDCG, so it is a number in
    [0, 1] that says nothing about how many documents produced it; weighting
    by the sum would make a 500-document query count a hundred times a
    5-document one purely for being longer, which is the thing per-query
    normalization exists to prevent.

    An empty `sample_weight` returns an empty list, meaning unweighted."""
    if len(sample_weight) == 0:
        return List[Float64]()
    if len(sample_weight) != groups.n_rows:
        raise Error("sample_weight length must equal n_rows")
    var out = List[Float64](capacity=groups.n_queries())
    for q in range(groups.n_queries()):
        var start = groups.start(q)
        var cnt = groups.size(q)
        var total = 0.0
        for i in range(cnt):
            total += sample_weight[start + i]
        out.append(total / Float64(cnt))
    return out^


struct RankEval(Copyable, Movable, Writable):
    """A ranking metric with the metadata a bare mean throws away.

    `values[c]` is the aggregate at `cutoffs[c]`. `per_query`, when it was
    asked for, is cutoff major: `per_query[c * n_queries + q]`.

    The counts are the point of this type. `n_degenerate` is the number of
    queries that were **scored by convention rather than measured** - all
    labels zero for NDCG, no relevant document for MAP - each of which
    contributes exactly 1.0. `n_single_document` is the number that could
    not have contributed anything else, since a query of one document is
    always perfectly ranked at every cutoff. A reported NDCG of 0.9 over a
    set that is a third degenerate is a different number from a reported
    NDCG of 0.9 over a set that is not, and nothing downstream can tell them
    apart without these.
    """

    var metric: Int
    var cutoffs: List[Int]
    var values: List[Float64]
    var per_query: List[Float64]
    var n_queries: Int
    var n_degenerate: Int
    var n_single_document: Int
    var total_weight: Float64
    var weighted: Bool

    def __init__(
        out self,
        metric: Int,
        var cutoffs: List[Int],
        var values: List[Float64],
        var per_query: List[Float64],
        n_queries: Int,
        n_degenerate: Int,
        n_single_document: Int,
        total_weight: Float64,
        weighted: Bool,
    ):
        self.metric = metric
        self.cutoffs = cutoffs^
        self.values = values^
        self.per_query = per_query^
        self.n_queries = n_queries
        self.n_degenerate = n_degenerate
        self.n_single_document = n_single_document
        self.total_weight = total_weight
        self.weighted = weighted

    def name(self) raises -> String:
        """The metric's LightGBM name, for reporting."""
        if self.metric == METRIC_NDCG:
            return String("ndcg")
        if self.metric == METRIC_MAP:
            return String("map")
        raise Error("unknown ranking metric code ", self.metric)

    def at(self, cutoff: Int) raises -> Float64:
        """The aggregate at one cutoff, by cutoff rather than by index, so a
        caller does not have to know where in `eval_at` it landed."""
        for c in range(len(self.cutoffs)):
            if self.cutoffs[c] == cutoff:
                return self.values[c]
        raise Error("cutoff ", cutoff, " was not evaluated")

    def query_value(self, cutoff: Int, query: Int) raises -> Float64:
        """One query's own value, when per-query values were requested."""
        if len(self.per_query) == 0:
            raise Error(
                "per-query values were not requested; pass per_query=True"
            )
        if query < 0 or query >= self.n_queries:
            raise Error("query index out of range")
        for c in range(len(self.cutoffs)):
            if self.cutoffs[c] == cutoff:
                return self.per_query[c * self.n_queries + query]
        raise Error("cutoff ", cutoff, " was not evaluated")

    def degenerate_fraction(self) -> Float64:
        """How much of the reported number is convention rather than
        measurement."""
        if self.n_queries == 0:
            return 0.0
        return Float64(self.n_degenerate) / Float64(self.n_queries)

    def write_to(self, mut writer: Some[Writer]):
        if self.metric == METRIC_NDCG:
            writer.write("ndcg(")
        elif self.metric == METRIC_MAP:
            writer.write("map(")
        else:
            writer.write("rank_metric(")
        for c in range(len(self.cutoffs)):
            if c > 0:
                writer.write(", ")
            writer.write("@", self.cutoffs[c], "=", self.values[c])
        writer.write(
            "; queries=",
            self.n_queries,
            ", degenerate=",
            self.n_degenerate,
            ", single_document=",
            self.n_single_document,
        )
        if self.weighted:
            writer.write(", query weighted")
        writer.write(")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def _check_eval_inputs(
    scores: List[Float64],
    labels: List[Int],
    groups: RankGroups,
    params: AdvancedRankParams,
    sample_weight: List[Float64],
) raises:
    check_groups(groups, len(scores))
    if len(labels) != len(scores):
        raise Error("labels length must equal scores length")
    check_relevance_labels(labels, params.gain)
    check_advanced_rank_params(params)
    if len(sample_weight) != 0 and len(sample_weight) != len(scores):
        raise Error("sample_weight length must equal n_rows")


def _eval_query_weights(
    groups: RankGroups,
    params: AdvancedRankParams,
    sample_weight: List[Float64],
) raises -> List[Float64]:
    """The per-query weights an evaluation averages under, or an empty list
    when the run is unweighted."""
    if not params.weight_queries:
        return List[Float64]()
    return query_weights(groups, sample_weight)


def ndcg_eval(
    scores: List[Float64],
    labels: List[Int],
    groups: RankGroups,
    params: AdvancedRankParams = AdvancedRankParams.default(),
    sample_weight: List[Float64] = [],
    per_query: Bool = False,
) raises -> RankEval:
    """NDCG at every position in `params.eval_at`, with the aggregation
    metadata.

    Composed from `ranking`'s primitives: `_argsort_desc_range` orders the
    documents, `_sorted_gains` (or the gain-table form) builds the ideal
    ranking, `_discounts` supplies `1 / log2(rank + 2)`. What is here and
    not in `ranking.ndcg_at_cutoffs` is the per-query vector, the query
    weights, the custom gain vector, and the counts.

    `ranking.ndcg_at_cutoffs` remains the authority for the case the two
    share - unweighted, default gains - and this must agree with it there.
    Ties in the scores keep their input order, as they do there, because the
    same stable ordering does the sorting.

    Scores are the model's raw scores. A position bias, if one was learned,
    is *not* added: the ranking a user is served has no position column, so
    scoring the adjusted scores would report a number no deployment can
    reach."""
    _check_eval_inputs(scores, labels, groups, params, sample_weight)

    var cutoffs = params.eval_at.copy()
    var n_c = len(cutoffs)
    var n_q = groups.n_queries()
    var qw = _eval_query_weights(groups, params, sample_weight)
    var weighted = len(qw) > 0

    var totals = List[Float64](capacity=n_c)
    for _ in range(n_c):
        totals.append(0.0)
    var values = List[Float64](capacity=n_c)
    var per = List[Float64]()
    if per_query:
        per.resize(n_c * n_q, 0.0)
    var discounts = _discounts(groups.max_size())
    var total_weight = 0.0
    var n_degenerate = 0
    var n_single = 0

    for q in range(n_q):
        var start = groups.start(q)
        var cnt = groups.size(q)
        var w = qw[q] if weighted else 1.0
        total_weight += w
        if cnt == 1:
            n_single += 1
        var order = _argsort_desc_range(scores, start, cnt)
        var gains = _query_sorted_gains(labels, start, cnt, params.gain)
        # The ideal DCG is zero at every cutoff exactly when the largest
        # gain in the query is zero, so degeneracy is a property of the
        # query and not of a cutoff.
        if gains[0] <= 0.0:
            n_degenerate += 1
        for c in range(n_c):
            var m = cutoffs[c] if cutoffs[c] < cnt else cnt
            var dcg = 0.0
            var best = 0.0
            for j in range(m):
                dcg += params.gain.of(labels[start + order[j]]) * discounts[j]
                best += gains[j] * discounts[j]
            var v = dcg / best if best > 0.0 else 1.0
            totals[c] += w * v
            if per_query:
                per[c * n_q + q] = v

    for c in range(n_c):
        values.append(totals[c] / total_weight if total_weight > 0.0 else 1.0)
    return RankEval(
        METRIC_NDCG,
        cutoffs^,
        values^,
        per^,
        n_q,
        n_degenerate,
        n_single,
        total_weight,
        weighted,
    )


def map_eval(
    scores: List[Float64],
    labels: List[Int],
    groups: RankGroups,
    params: AdvancedRankParams = AdvancedRankParams.default(),
    sample_weight: List[Float64] = [],
    per_query: Bool = False,
) raises -> RankEval:
    """Mean average precision at every position in `params.eval_at`, with
    the same metadata.

    Relevance is binary (any label above 0), AP@k is divided by
    `min(k, relevant in query)`, and a query with nothing relevant counts as
    1.0 - the conventions `ranking.map_at_cutoffs` documents, and this must
    agree with it on the unweighted case. The gain vector plays no part: MAP
    does not read graded relevance, which is exactly why a run should report
    both metrics rather than either alone."""
    _check_eval_inputs(scores, labels, groups, params, sample_weight)

    var cutoffs = params.eval_at.copy()
    var n_c = len(cutoffs)
    var n_q = groups.n_queries()
    var qw = _eval_query_weights(groups, params, sample_weight)
    var weighted = len(qw) > 0

    var totals = List[Float64](capacity=n_c)
    for _ in range(n_c):
        totals.append(0.0)
    var values = List[Float64](capacity=n_c)
    var per = List[Float64]()
    if per_query:
        per.resize(n_c * n_q, 0.0)
    var total_weight = 0.0
    var n_degenerate = 0
    var n_single = 0

    for q in range(n_q):
        var start = groups.start(q)
        var cnt = groups.size(q)
        var w = qw[q] if weighted else 1.0
        total_weight += w
        if cnt == 1:
            n_single += 1
        var order = _argsort_desc_range(scores, start, cnt)
        var n_relevant = 0
        for i in range(cnt):
            if labels[start + i] > 0:
                n_relevant += 1
        if n_relevant == 0:
            n_degenerate += 1
        for c in range(n_c):
            var v: Float64
            if n_relevant == 0:
                v = 1.0
            else:
                var m = cutoffs[c] if cutoffs[c] < cnt else cnt
                var hits = 0.0
                var precision_sum = 0.0
                for j in range(m):
                    if labels[start + order[j]] > 0:
                        hits += 1.0
                        precision_sum += hits / Float64(j + 1)
                var denom = (
                    cutoffs[c] if cutoffs[c] < n_relevant else n_relevant
                )
                v = precision_sum / Float64(denom)
            totals[c] += w * v
            if per_query:
                per[c * n_q + q] = v

    for c in range(n_c):
        values.append(totals[c] / total_weight if total_weight > 0.0 else 1.0)
    return RankEval(
        METRIC_MAP,
        cutoffs^,
        values^,
        per^,
        n_q,
        n_degenerate,
        n_single,
        total_weight,
        weighted,
    )


# ---------------------------------------------------------------------------
# Group-safe bagging
# ---------------------------------------------------------------------------


def check_query_bagging(bagging: BaggingParams, groups: RankGroups) raises:
    """Validate query bagging against the query count, not the row count.

    `bagging.check_bagging` validates the fraction and the frequency; this
    adds the check only ranking can make. `bagging.sample_rows` guarantees a
    nonempty bag by falling back to the single smallest draw, so a fraction
    of 0.001 over 20 queries does not fail - it silently trains every round
    on one query. That is a configuration mistake, not a random outcome, and
    it is refused here rather than discovered from a flat NDCG curve."""
    check_bagging(bagging)
    if not bagging_enabled(bagging):
        return
    if groups.n_queries() < 2:
        raise Error(
            "query bagging needs at least two queries; this dataset has ",
            groups.n_queries(),
        )
    var expected = Float64(groups.n_queries()) * bagging.fraction
    if expected < 1.0:
        raise Error(
            "bagging_fraction ",
            bagging.fraction,
            " over ",
            groups.n_queries(),
            " queries draws fewer than one query per round; a ranking bag is"
            " whole queries, so raise the fraction or use more queries",
        )


def refresh_query_bag(
    mut query_bag: List[Int],
    mut row_bag: List[Int],
    groups: RankGroups,
    bagging: BaggingParams,
    iteration: Int,
) raises:
    """Redraw the query bag on the rounds LightGBM would.

    A thin re-export of `ranking._refresh_query_bag`, which is the one
    implementation: whole queries, on the `iteration % freq == 0` schedule,
    expanded to rows in ascending order because queries are contiguous. It
    is named here so that a caller working in this module never reaches for
    `bagging.refresh_bag`, which would sample **rows** and split queries."""
    _refresh_query_bag(query_bag, row_bag, groups, bagging, iteration)


# ---------------------------------------------------------------------------
# Group-safe cross-validation
# ---------------------------------------------------------------------------


struct QueryFold(Copyable, Movable):
    """One cross-validation fold of a ranking dataset.

    Both sides carry their rows *and* their `group` arrays, because a row
    list without query boundaries is not a ranking dataset: the maxDCG a
    fold's rows are normalized against is decided by which of them share a
    query."""

    var train_rows: List[Int]
    var test_rows: List[Int]
    var train_counts: List[Int]
    var test_counts: List[Int]
    var train_queries: List[Int]
    var test_queries: List[Int]

    def __init__(
        out self,
        var train_rows: List[Int],
        var test_rows: List[Int],
        var train_counts: List[Int],
        var test_counts: List[Int],
        var train_queries: List[Int],
        var test_queries: List[Int],
    ):
        self.train_rows = train_rows^
        self.test_rows = test_rows^
        self.train_counts = train_counts^
        self.test_counts = test_counts^
        self.train_queries = train_queries^
        self.test_queries = test_queries^


def _fold_shuffle(n: Int, seed: Int) -> List[Int]:
    """A deterministic permutation of `0..n-1`, Fisher-Yates over a
    counter-based stream, so the order depends on (seed, n) alone and not on
    how many draws came before it."""
    var order = List[Int](capacity=n)
    for i in range(n):
        order.append(i)
    var stream = _splitmix64(UInt64(seed & 0x7FFFFFFFFFFFFFFF) ^ _FOLD_DOMAIN)
    for i in range(n - 1, 0, -1):
        var j = Int(_uniform(stream + UInt64(i)) * Float64(i + 1))
        if j > i:
            j = i
        var tmp = order[i]
        order[i] = order[j]
        order[j] = tmp
    return order^


def query_folds(
    groups: RankGroups,
    n_folds: Int,
    shuffle: Bool = False,
    seed: Int = 0,
) raises -> List[QueryFold]:
    """Split a ranking dataset into `n_folds` folds of **whole queries**.

    A query split across the two sides of a fold is the ranking form of
    training on the test set: the held-out half is normalized against a
    maxDCG computed from documents the model was trained on. So the query is
    the unit here, exactly as it is for bagging.

    Fold f holds queries `[f * Q // K, (f + 1) * Q // K)` of the query
    order - the same contiguous chunking
    `python/mojoboost/cv.py::_chunk_folds` does, so the Mojo and Python
    paths agree fold for fold when they are given the same order. With
    `shuffle=True` that order is a counter-based permutation of the query
    indices, never of the rows.

    Every query lands in exactly one test fold and in every other fold's
    training side, so the folds partition the query set, and both row lists
    come back ascending because the query lists are sorted before they are
    expanded."""
    if n_folds < 2:
        raise Error("n_folds must be at least 2")
    var n_q = groups.n_queries()
    if n_q < n_folds:
        raise Error(
            "a ranking cv needs at least one query per fold: ",
            n_folds,
            " folds over ",
            n_q,
            " queries",
        )

    var order: List[Int]
    if shuffle:
        order = _fold_shuffle(n_q, seed)
    else:
        order = List[Int](capacity=n_q)
        for q in range(n_q):
            order.append(q)

    var folds = List[QueryFold]()
    for f in range(n_folds):
        var lo = f * n_q // n_folds
        var hi = (f + 1) * n_q // n_folds
        var test_q = List[Int]()
        var train_q = List[Int]()
        for i in range(n_q):
            if i >= lo and i < hi:
                test_q.append(order[i])
            else:
                train_q.append(order[i])
        # `rows_of_queries` needs ascending queries, and a shuffled chunk is
        # not; sorting here keeps every downstream row list ascending, which
        # is what tree growth and `_expand_queries` assume.
        sort(test_q)
        sort(train_q)
        var train_rows = rows_of_queries(groups, train_q)
        var test_rows = rows_of_queries(groups, test_q)
        var train_counts = group_counts_of_queries(groups, train_q)
        var test_counts = group_counts_of_queries(groups, test_q)
        folds.append(
            QueryFold(
                train_rows^,
                test_rows^,
                train_counts^,
                test_counts^,
                train_q^,
                test_q^,
            )
        )
    return folds^


# ---------------------------------------------------------------------------
# Group-safe distributed partitioning
# ---------------------------------------------------------------------------


@fieldwise_init
struct QueryPartition(Copyable, Movable):
    """One rank's slice of a ranking dataset, cut on query boundaries.

    Half open in both coordinates: the rank owns queries
    `[query_start, query_end)` and rows `[row_start, row_end)`, and the two
    agree by construction. A rank may own no queries, in which case it owns
    no rows and takes part in every collective as a zero contribution, which
    is what `distributed.mojo` already expects of an empty shard."""

    var query_start: Int
    var query_end: Int
    var row_start: Int
    var row_end: Int

    def n_queries(self) -> Int:
        return self.query_end - self.query_start

    def n_rows(self) -> Int:
        return self.row_end - self.row_start

    def is_empty(self) -> Bool:
        return self.query_end <= self.query_start


def partition_queries(
    groups: RankGroups, world_size: Int
) raises -> List[QueryPartition]:
    """Partition a ranking dataset across `world_size` ranks on query
    boundaries.

    Why this exists rather than `distributed.partition_rows`: that function
    cuts at `r * n_rows // W`, which lands inside a query whenever the row
    count does not divide evenly, and a rank holding half a query computes
    lambdas normalized against the maxDCG of the half it can see. The
    gradients would be wrong on every rank that held a fragment, and no
    all-reduce would catch it, because the ranking objective needs no
    cross-rank reduction at all: **every lambda is computed inside one
    query, so a query-aligned partition makes the distributed gradient
    exactly the single-node gradient, rank by rank.** That is the whole
    contract, and it holds only if no query is split.

    The cut is the contiguous, order-preserving one closest to an even row
    split: boundary `r` is placed at the query boundary nearest to
    `r * n_rows // W`, and boundaries are forced nondecreasing so a run of
    large queries yields empty ranks rather than a crossed partition.
    Concatenating the partitions in ascending rank order reproduces the
    dataset row for row, which is the property
    `distributed.partition_rows` documents and the floating point
    equivalence argument rests on."""
    if world_size < 1:
        raise Error("world_size must be positive")
    var n_q = groups.n_queries()
    var n_rows = groups.n_rows

    var cuts = List[Int](capacity=world_size + 1)
    cuts.append(0)
    for r in range(1, world_size):
        var target = r * n_rows // world_size
        # `groups.starts` is ascending, so the nearest boundary is found by
        # a scan that starts where the previous rank ended and stops at the
        # first boundary that has reached the target.
        var best_q = cuts[r - 1]
        var best_gap = -1
        for q in range(cuts[r - 1], n_q + 1):
            var gap = groups.starts[q] - target
            if gap < 0:
                gap = -gap
            if best_gap < 0 or gap < best_gap:
                best_gap = gap
                best_q = q
            if groups.starts[q] >= target:
                break
        cuts.append(best_q)
    cuts.append(n_q)

    var out = List[QueryPartition]()
    for r in range(world_size):
        var lo = cuts[r]
        var hi = cuts[r + 1]
        if hi < lo:
            hi = lo
        out.append(
            QueryPartition(lo, hi, groups.starts[lo], groups.starts[hi])
        )
    return out^


def partition_group_counts(
    groups: RankGroups, part: QueryPartition
) raises -> List[Int]:
    """The `group` array one rank needs: the row count of each query it
    owns, in row order, which is what `ranking.groups_from_counts` takes."""
    if part.query_start < 0 or part.query_end > groups.n_queries():
        raise Error("partition names a query outside the dataset")
    var out = List[Int](capacity=part.n_queries())
    for q in range(part.query_start, part.query_end):
        out.append(groups.size(q))
    return out^


def check_query_partition(
    parts: List[QueryPartition], groups: RankGroups
) raises:
    """Validate a partition: contiguous, order preserving, query aligned,
    and covering every row exactly once.

    This is the check a distributed ranking run makes on every rank before
    the first round, so a partition that lost or duplicated a query fails
    identically everywhere rather than producing a quietly different model
    on one rank."""
    if len(parts) == 0:
        raise Error("a partition must name at least one rank")
    var expect_q = 0
    var expect_row = 0
    for r in range(len(parts)):
        var p = parts[r]
        if p.query_start != expect_q:
            raise Error(
                "partition for rank ",
                r,
                " starts at query ",
                p.query_start,
                ", expected ",
                expect_q,
            )
        if p.query_end < p.query_start:
            raise Error("partition for rank ", r, " ends before it starts")
        if p.query_end > groups.n_queries():
            raise Error("partition for rank ", r, " names a missing query")
        if p.row_start != groups.starts[p.query_start]:
            raise Error(
                "partition for rank ",
                r,
                " does not start on a query boundary",
            )
        if p.row_end != groups.starts[p.query_end]:
            raise Error(
                "partition for rank ", r, " does not end on a query boundary"
            )
        if p.row_start != expect_row:
            raise Error("partition for rank ", r, " skips or repeats rows")
        expect_q = p.query_end
        expect_row = p.row_end
    if expect_q != groups.n_queries():
        raise Error("the partition does not cover every query")
    if expect_row != groups.n_rows:
        raise Error("the partition does not cover every row")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


struct TrainedAdvancedRanker(Copyable, Movable):
    """A trained ranking ensemble and the position biases learned beside it.

    Two values rather than one because they have different lifetimes: the
    `Booster` is the model and serializes, the `PositionBiasState` is
    training state that no model file holds. Returning them together is what
    keeps the biases recoverable at all - a caller that throws this half
    away has thrown away the only record of the correction the trees were
    fitted under."""

    var booster: Booster
    var bias: PositionBiasState

    def __init__(out self, var booster: Booster, var bias: PositionBiasState):
        self.booster = booster^
        self.bias = bias^


struct FittedAdvancedRanker(Copyable, Movable):
    """`TrainedAdvancedRanker` with the bin mapper folded in, the ranking
    counterpart of `ranking.fit_ranker`'s `Model` return."""

    var model: Model
    var bias: PositionBiasState

    def __init__(out self, var model: Model, var bias: PositionBiasState):
        self.model = model^
        self.bias = bias^


def train_ranker_advanced(
    data: BinnedMatrix,
    labels: List[Int],
    groups: RankGroups,
    params: BoosterParams,
    rank_params: AdvancedRankParams = AdvancedRankParams.default(),
    positions: PositionMap = PositionMap.absent(),
    sample_weight: List[Float64] = [],
    bagging: BaggingParams = BaggingParams.disabled(),
) raises -> TrainedAdvancedRanker:
    """Train a LambdaRank ensemble with the advanced features applied.

    A second outer loop, and it is honest about why: the position biases are
    state that changes every round and that `ranking.train_ranker` has
    nowhere to put. Everything inside the loop is shared -
    `advanced_lambdarank_gradients` for the round's gradients (which is
    `ranking._fill_lambdas` itself in every default configuration),
    `ranking._refresh_query_bag` for the bag, `tree.grow_tree` for the tree,
    and an ordinary `boosting.Booster` of ordinary `tree.Tree` values for
    the result. The handoff carries the patch that folds this back into
    `ranking.train_ranker` as an optional argument, at which point this
    function goes away.

    The returned booster boosts from 0.0 as `ranking.train_ranker`'s does,
    carries the `LAMBDARANK` objective code, and serializes and predicts
    through exactly the same paths. A model file therefore does not record
    that position bias was used, which is the serialization consequence
    recorded in docs/RANKING_ADVANCED.md, and the returned
    `PositionBiasState` is the only copy of what was learned.
    """
    if len(labels) != data.n_rows:
        raise Error("labels length must equal n_rows")
    check_groups(groups, data.n_rows)
    check_relevance_labels(labels, rank_params.gain)
    check_advanced_rank_params(rank_params)
    check_positions(positions, data.n_rows)
    _check_sample_weight(sample_weight, data.n_rows)
    check_query_bagging(bagging, groups)

    var n = data.n_rows
    var raw = List[Float64](capacity=n)
    raw.resize(n, 0.0)
    var bias = PositionBiasState.for_positions(positions)

    var trees = List[Tree]()
    var grad = List[Float64](capacity=n)
    var hess = List[Float64](capacity=n)
    var query_bag = List[Int]()
    var row_bag = List[Int]()
    for i in range(params.n_estimators):
        _refresh_query_bag(query_bag, row_bag, groups, bagging, i)
        advanced_lambdarank_gradients(
            raw,
            labels,
            groups,
            positions,
            bias,
            grad,
            hess,
            rank_params,
            sample_weight,
            i,
            params.learning_rate,
        )
        var tree = grow_tree(data, grad, hess, params.tree, row_bag, i)

        # A single-leaf tree with a near-zero value means no pair is left to
        # separate; under bagging it only means that of this bag, so the
        # round is skipped and the next bag gets its turn. This is
        # `ranking.train_ranker`'s rule, kept identical on purpose.
        if tree.n_leaves == 1 and abs(tree.value[0]) < 1e-12:
            if bagging_enabled(bagging):
                continue
            break

        for r in range(n):
            raw[r] += params.learning_rate * tree.predict_row(data, r)
        trees.append(tree^)

    var booster = Booster(
        trees^,
        0.0,
        params.learning_rate,
        LAMBDARANK,
        params.tree.monotone.copy(),
    )
    return TrainedAdvancedRanker(booster^, bias^)


def fit_ranker_advanced(
    features: List[Float64],
    n_rows: Int,
    n_features: Int,
    labels: List[Int],
    group_counts: List[Int],
    params: BoosterParams,
    rank_params: AdvancedRankParams = AdvancedRankParams.default(),
    positions: PositionMap = PositionMap.absent(),
    max_bins: Int = 255,
    sample_weight: List[Float64] = [],
    bagging: BaggingParams = BaggingParams.disabled(),
    use_missing: Bool = True,
    categorical_features: List[Int] = [],
) raises -> FittedAdvancedRanker:
    """`ranking.fit_ranker` with the advanced parameters.

    Binning, the raw column-major layout (`features[f * n_rows + r]`),
    `use_missing`, and `categorical_features` mean exactly what they do
    there; the `group` array goes through `sanitize_group_counts` first, so
    `rank_params.empty_query_policy` decides whether a query holding no rows
    is an error or is dropped."""
    var groups = groups_from_counts_sanitized(
        group_counts, rank_params.empty_query_policy
    )
    if groups.n_rows != n_rows:
        raise Error("group counts must sum to n_rows")
    var mapper = fit_bins(
        features,
        n_rows,
        n_features,
        max_bins,
        use_missing=use_missing,
        categorical_features=categorical_features,
    )
    var data = mapper.transform(features, n_rows)
    var trained = train_ranker_advanced(
        data,
        labels,
        groups,
        params,
        rank_params,
        positions,
        sample_weight,
        bagging,
    )
    var model = Model(mapper^, trained.booster.copy())
    return FittedAdvancedRanker(model^, trained.bias^)
