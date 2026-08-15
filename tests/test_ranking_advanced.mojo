"""Tests for the advanced ranking layer (ranking_advanced.mojo) and its route
into the entry points.

What each test pins down: the default configuration is not "advanced" and
routes to `ranking.train_ranker` (and, run through the advanced loop, gives
the same ensemble); a custom `label_gain` is validated and changes the
trained ensemble; a position column trains a bias per position and leaves
the served scores unbiased; pair sampling and a decoupled maxDCG cutoff are
requested and refused when malformed.
"""

from std.testing import assert_equal, assert_true, assert_raises, TestSuite

from mojotrees.binning import bin_equal_width
from mojotrees.boosting import BoosterParams
from mojotrees.custom_metric import (
    CustomMetric,
    MetricSuite,
    ValidSet,
    train_ranker_with_metrics,
)
from mojotrees.ranking import (
    LAMBDARANK,
    RankerParams,
    groups_from_counts,
    ndcg,
    train_ranker,
)
from mojotrees.ranking_advanced import (
    AdvancedRankParams,
    LabelGain,
    PositionMap,
    advanced_ranking_requested,
    check_advanced_rank_params,
    positions_from_codes,
    train_ranker_advanced,
)
from mojotrees.tree import TreeParams


def _params(n_rounds: Int) -> BoosterParams:
    return BoosterParams(n_rounds, 0.3, TreeParams(8, 1, 1.0, 1e-3))


def _ranking_dataset(
    n_queries: Int, mut features: List[Float64], mut labels: List[Int]
) -> List[Int]:
    var counts = List[Int](capacity=n_queries)
    var order: List[Int] = [0, 3, 1, 2]
    for q in range(n_queries):
        for i in range(4):
            var rel = order[i]
            features.append(Float64(rel) + 0.25 * Float64(q % 2))
            labels.append(rel)
        counts.append(4)
    return counts^


def _custom_gain() -> AdvancedRankParams:
    var p = AdvancedRankParams.default()
    # Linear gains instead of 2^l - 1: nondecreasing, starts at zero.
    var g: List[Float64] = [0.0, 1.0, 2.0, 3.0]
    p.gain = LabelGain(g^)
    return p^


def test_default_is_not_advanced_and_matches_train_ranker() raises:
    var p = AdvancedRankParams.default()
    assert_true(not advanced_ranking_requested(p, PositionMap.absent()))
    assert_true(advanced_ranking_requested(_custom_gain(), PositionMap.absent()))

    var features = List[Float64]()
    var labels = List[Int]()
    var counts = _ranking_dataset(6, features, labels)
    var groups = groups_from_counts(counts)
    var data = bin_equal_width(features, 24, 1, 8)

    var base = train_ranker(data, labels, groups, _params(20))
    var adv = train_ranker_advanced(data, labels, groups, _params(20), p)
    assert_equal(len(adv.booster.trees), len(base.trees))
    for r in range(24):
        assert_equal(adv.booster.predict_row(data, r), base.predict_row(data, r))
    assert_equal(adv.booster.objective, LAMBDARANK)


def test_custom_label_gain_changes_the_ensemble_and_still_ranks() raises:
    var features = List[Float64]()
    var labels = List[Int]()
    var counts = _ranking_dataset(6, features, labels)
    var groups = groups_from_counts(counts)
    var data = bin_equal_width(features, 24, 1, 8)

    var base = train_ranker(data, labels, groups, _params(30))
    var adv = train_ranker_advanced(
        data, labels, groups, _params(30), _custom_gain()
    )
    var differs = False
    var scores = List[Float64](capacity=24)
    for r in range(24):
        var s = adv.booster.predict_row(data, r)
        scores.append(s)
        if s != base.predict_row(data, r):
            differs = True
    assert_true(differs)
    assert_equal(ndcg(scores, labels, groups, 4), 1.0)

    # A decreasing gain vector inverts the objective and is refused.
    var bad = AdvancedRankParams.default()
    var g: List[Float64] = [0.0, 2.0, 1.0]
    bad.gain = LabelGain(g^)
    with assert_raises():
        check_advanced_rank_params(bad)
    # A gain vector shorter than the labels is refused at training.
    var short = AdvancedRankParams.default()
    var g2: List[Float64] = [0.0, 1.0]
    short.gain = LabelGain(g2^)
    with assert_raises():
        _ = train_ranker_advanced(data, labels, groups, _params(3), short)


def test_metric_trainer_honors_the_advanced_parameters() raises:
    """`custom_metric.train_ranker_with_metrics` (the eval_set path) uses the
    advanced lambdas when they are requested, so a custom label_gain there
    trains the ensemble `train_ranker_advanced` trains, and the default
    call is still the baseline loop."""
    var features = List[Float64]()
    var labels = List[Int]()
    var counts = _ranking_dataset(6, features, labels)
    var groups = groups_from_counts(counts)
    var data = bin_equal_width(features, 24, 1, 8)
    var targets = List[Float64](capacity=24)
    for r in range(24):
        targets.append(Float64(labels[r]))

    def constant_metric(
        metric: Int, valid: Int, pred: List[Float64], y: List[Float64]
    ) raises -> Float64:
        return 0.0

    def run(
        advanced: AdvancedRankParams,
    ) raises {imm data, imm labels, imm groups, imm targets} -> List[Float64]:
        var valid_sets: List[ValidSet] = [
            ValidSet("valid", data.copy(), targets.copy())
        ]
        var metrics: List[CustomMetric] = [CustomMetric("zero")]
        var result = train_ranker_with_metrics(
            data,
            labels,
            groups,
            valid_sets^,
            _params(30),
            MetricSuite(metrics^, constant_metric, 0),
            advanced=advanced,
        )
        var out = List[Float64](capacity=24)
        for r in range(24):
            out.append(result.booster.predict_raw_row(data, r))
        return out^

    var base = train_ranker(data, labels, groups, _params(30))
    var plain = run(AdvancedRankParams.default())
    for r in range(24):
        assert_equal(plain[r], base.predict_row(data, r))

    var adv = train_ranker_advanced(
        data, labels, groups, _params(30), _custom_gain()
    )
    var gained = run(_custom_gain())
    var differs = False
    for r in range(24):
        assert_equal(gained[r], adv.booster.predict_row(data, r))
        if gained[r] != plain[r]:
            differs = True
    assert_true(differs)


def test_position_column_learns_a_bias_and_trains() raises:
    var features = List[Float64]()
    var labels = List[Int]()
    var counts = _ranking_dataset(6, features, labels)
    var groups = groups_from_counts(counts)
    var data = bin_equal_width(features, 24, 1, 8)

    # Each query's rows were shown at slots 0..3.
    var codes = List[Int](capacity=24)
    for _ in range(6):
        for i in range(4):
            codes.append(10 + i)
    var enc = positions_from_codes(codes)
    assert_equal(enc.positions.n_positions, 4)
    var p = AdvancedRankParams.default()
    p.position_bias_regularization = 0.1
    assert_true(advanced_ranking_requested(p, enc.positions))

    var adv = train_ranker_advanced(
        data, labels, groups, _params(20), p, enc.positions
    )
    assert_true(len(adv.booster.trees) > 0)
    assert_equal(len(adv.bias.biases), 4)
    var moved = False
    for i in range(4):
        if adv.bias.biases[i] != 0.0:
            moved = True
    assert_true(moved)
    var scores = List[Float64](capacity=24)
    for r in range(24):
        scores.append(adv.booster.predict_row(data, r))
    assert_true(ndcg(scores, labels, groups, 4) > 0.9)

    # A position column of the wrong length is refused.
    var wrong = PositionMap([0, 1, 2], 3)
    with assert_raises():
        _ = train_ranker_advanced(data, labels, groups, _params(3), p, wrong)


def test_pair_sampling_and_cutoff_are_validated() raises:
    var p = AdvancedRankParams.default()
    p.pair_sampling_rate = 0.5
    assert_true(advanced_ranking_requested(p, PositionMap.absent()))
    check_advanced_rank_params(p)
    p.pair_sampling_rate = 0.0
    with assert_raises():
        check_advanced_rank_params(p)
    var q = AdvancedRankParams.default()
    q.max_dcg_cutoff = 2
    assert_true(advanced_ranking_requested(q, PositionMap.absent()))
    check_advanced_rank_params(q)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
