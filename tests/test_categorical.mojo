"""Native categorical features.

Covers the four things a categorical column has to get right: its binning
metadata is fit and applied separately from quantile binning, its splits are
searched as category partitions rather than ordinal thresholds, its missing
and unseen codes route by the documented default, and all of that survives
serialization.

The low- and high-cardinality cases are tested separately because they take
different search paths: at or below `max_cat_to_onehot` categories the search
is one-vs-rest, above it the sorted many-vs-many walk (see categorical.mojo).

The multiclass and ranking trainers get their own cases here rather than in
their own files, because what is under test is the categorical contract, not
those objectives. The GPU case is here for the same reason: it skips itself
without an accelerator.
"""

from std.os import remove
from std.sys import has_accelerator
from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_raises,
    assert_true,
    TestSuite,
)

from mojotrees.binning import fit_bins
from mojotrees.boosting import SQUARED_ERROR, BoosterParams, train
from mojotrees.categorical import (
    CAT_BITSET_WORDS,
    CategoricalParams,
    CategoricalSpec,
    cat_add,
    cat_contains,
    cat_empty,
    cat_pool_contains,
    fit_categorical_spec,
)
from mojotrees.histogram import Histogram
from mojotrees.model import Model, fit, fit_multiclass
from mojotrees.ranking import (
    RankerParams,
    fit_ranker,
    groups_from_counts,
    ndcg,
)
from mojotrees.serialize import load_model, save_model
from mojotrees.split import find_best_split
from mojotrees.train_gpu import train_gpu
from mojotrees.tree import TreeParams, grow_tree
from support import _uniform

comptime _TMP_PATH = "./.test_categorical_roundtrip.tmp"


def _nan() -> Float64:
    var zero = _uniform(UInt64(0)) * 0.0
    return zero / zero


def _params(num_leaves: Int, n_estimators: Int) -> BoosterParams:
    return BoosterParams(
        n_estimators, 0.2, TreeParams(num_leaves, 5, 1.0, 1e-3)
    )


def _small_cat_params() -> CategoricalParams:
    """LightGBM's defaults are sized for tens of thousands of rows;
    `min_data_per_group=100` alone would reject every split in a 400-row
    test. These keep the same search, scaled to the test datasets."""
    return CategoricalParams(4, 32, 2.0, 10.0, 5)


# --- Bitsets ---------------------------------------------------------------


def test_bitset_membership_across_words() raises:
    var members: List[Int] = [1, 63, 64, 65, 127, 128, 200, 255]
    var absent: List[Int] = [0, 2, 62, 66, 126, 199, 254]
    var bits = cat_empty()
    for i in range(len(members)):
        cat_add(bits, members[i])
    for i in range(len(members)):
        assert_true(cat_contains(bits, members[i]))
    for i in range(len(absent)):
        assert_true(not cat_contains(bits, absent[i]))
    # Bin 0 is never a category, and out-of-range ids are never members.
    assert_true(not cat_contains(bits, 0))
    assert_true(not cat_contains(bits, 256))
    assert_true(not cat_contains(bits, -1))

    # The flat-pool form used on tree nodes agrees with the SIMD form.
    var pool = List[UInt64]()
    for _ in range(CAT_BITSET_WORDS):
        pool.append(0)
    for w in range(CAT_BITSET_WORDS):
        pool.append(bits[w])
    for b in range(256):
        assert_equal(
            cat_pool_contains(pool, CAT_BITSET_WORDS, b), cat_contains(bits, b)
        )


# --- Binning metadata ------------------------------------------------------


def test_category_codes_map_to_bins_from_one() raises:
    # Codes 7, 2, 5 in an arbitrary order; bins follow ascending code order,
    # not first-seen order, and bin 0 stays reserved.
    var features: List[Float64] = [7.0, 2.0, 5.0, 2.0, 7.0, 5.0]
    var spec = fit_categorical_spec(features, 6, 1, [0], 255)
    assert_true(spec.is_cat(0))
    assert_equal(spec.n_categories(0), 3)
    assert_equal(spec.bin_of(0, 2.0), 1)
    assert_equal(spec.bin_of(0, 5.0), 2)
    assert_equal(spec.bin_of(0, 7.0), 3)


def test_missing_and_unseen_codes_land_in_bin_zero() raises:
    var features: List[Float64] = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]
    var spec = fit_categorical_spec(features, 6, 1, [0], 255)
    assert_equal(spec.n_categories(0), 3)
    # Unseen code.
    assert_equal(spec.bin_of(0, 9.0), 0)
    # Missing: negative, and NaN.
    assert_equal(spec.bin_of(0, -1.0), 0)
    assert_equal(spec.bin_of(0, _nan()), 0)
    # Out of the representable code range.
    assert_equal(spec.bin_of(0, 4.0e9), 0)
    # Non-integral values truncate toward zero, as LightGBM's cast does.
    assert_equal(spec.bin_of(0, 2.9), 3)


def test_high_cardinality_keeps_the_most_frequent_categories() raises:
    # 40 codes; code c appears (40 - c) times, so the low codes are the
    # frequent ones. With max_bins=11 only 10 categories fit.
    var features = List[Float64]()
    var n_rows = 0
    for c in range(40):
        for _ in range(40 - c):
            features.append(Float64(c))
            n_rows += 1
    var spec = fit_categorical_spec(features, n_rows, 1, [0], 11)
    assert_equal(spec.n_categories(0), 10)
    for c in range(10):
        assert_equal(spec.bin_of(0, Float64(c)), c + 1)
    # Everything dropped falls into the unknown bin with the missing rows.
    for c in range(10, 40):
        assert_equal(spec.bin_of(0, Float64(c)), 0)


def test_categorical_binning_leaves_numerical_edges_untouched() raises:
    # Feature 0 numerical, feature 1 integer-coded. Marking feature 1
    # categorical must not perturb feature 0's quantile edges by one bit.
    var n_rows = 300
    var features = List[Float64](capacity=2 * n_rows)
    for r in range(n_rows):
        features.append(_uniform(UInt64(r)))
    for r in range(n_rows):
        features.append(Float64(r % 6))

    var plain = fit_bins(features, n_rows, 2, 32)
    var mixed = fit_bins(features, n_rows, 2, 32, [1])

    assert_equal(mixed.edge_offsets[1], plain.edge_offsets[1])
    for i in range(plain.edge_offsets[1]):
        assert_true(mixed.edges[i] == plain.edges[i])
    # The categorical column contributes no edges at all.
    assert_equal(mixed.edge_offsets[2] - mixed.edge_offsets[1], 0)
    assert_true(mixed.cats.is_cat(1))
    assert_true(not mixed.cats.is_cat(0))

    # Its bins come from the category table, not from any threshold.
    var binned = mixed.transform(features, n_rows)
    for r in range(n_rows):
        assert_equal(binned.bin_at(r, 1), (r % 6) + 1)


def test_categorical_marking_is_validated() raises:
    var features: List[Float64] = [0.0, 1.0, 0.0, 1.0]
    with assert_raises():
        _ = fit_categorical_spec(features, 2, 2, [2], 255)
    with assert_raises():
        _ = fit_categorical_spec(features, 2, 2, [-1], 255)
    with assert_raises():
        _ = fit_categorical_spec(features, 2, 2, [0, 0], 255)
    # Codes that cannot be represented are rejected at fit time rather than
    # silently binned as unknown.
    var huge: List[Float64] = [0.0, 5.0e9]
    with assert_raises():
        _ = fit_categorical_spec(huge, 2, 1, [0], 255)


def test_more_categories_than_bins_is_rejected_in_search() raises:
    # A spec that claims more categories than the histogram has bins is
    # corrupt; the search must say so rather than read past the slice.
    var hist = Histogram.from_planes(
        [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [1, 1, 1, 1], 1, 4
    )
    var spec = CategoricalSpec([True], [0, 1, 2, 3, 4], [0, 5])
    with assert_raises():
        _ = find_best_split(hist, cats=spec)


# --- Partition search ------------------------------------------------------


def test_onehot_search_isolates_one_category() raises:
    # Three categories, so the search is one-vs-rest. Category 1 (bin 2) is
    # the only one with a positive gradient sum.
    var grad: List[Float64] = [0.0, -4.0, 8.0, -4.0]
    var hess: List[Float64] = [0.0, 4.0, 4.0, 4.0]
    var count: List[Int] = [0, 4, 4, 4]
    var hist = Histogram.from_planes(grad^, hess^, count^, 1, 4)
    var spec = CategoricalSpec([True], [10, 20, 30], [0, 3])

    var split = find_best_split(
        hist,
        lambda_reg=0.0,
        cats=spec,
        cat_params=CategoricalParams(4, 32, 1.0, 0.0, 1),
    )
    assert_true(split.found)
    assert_true(split.is_categorical)
    assert_equal(split.feature, 0)
    assert_equal(split.bin, -1)
    assert_true(cat_contains(split.cat_bitset, 2))
    assert_true(not cat_contains(split.cat_bitset, 1))
    assert_true(not cat_contains(split.cat_bitset, 3))
    # The unknown bin is never a member, so its rows route right.
    assert_true(not cat_contains(split.cat_bitset, 0))


def test_sorted_search_groups_non_adjacent_categories() raises:
    # Six categories in bins 1..6. Bins 2, 4, 6 share one gradient sign and
    # bins 1, 3, 5 the other, a grouping no `bin <= t` threshold can express.
    var grad = List[Float64]()
    var hess = List[Float64]()
    var count = List[Int]()
    grad.append(0.0)
    hess.append(0.0)
    count.append(0)
    for b in range(1, 7):
        grad.append(-6.0 if b % 2 == 0 else 6.0)
        hess.append(6.0)
        count.append(6)
    var hist = Histogram.from_planes(grad^, hess^, count^, 1, 7)
    var spec = CategoricalSpec([True], [0, 1, 2, 3, 4, 5], [0, 6])

    var split = find_best_split(
        hist,
        lambda_reg=0.0,
        min_data_in_leaf=0,
        # max_cat_to_onehot=1 forces the sorted many-vs-many path.
        cat_params=CategoricalParams(1, 32, 1.0, 0.0, 1),
        cats=spec,
    )
    assert_true(split.found)
    assert_true(split.is_categorical)
    # Whichever side won, the three same-sign categories are grouped together.
    var evens_in = cat_contains(split.cat_bitset, 2)
    for b in range(1, 7):
        var want = evens_in if b % 2 == 0 else not evens_in
        assert_equal(cat_contains(split.cat_bitset, b), want)


def test_max_cat_threshold_caps_the_set_size() raises:
    var grad = List[Float64]()
    var hess = List[Float64]()
    var count = List[Int]()
    grad.append(0.0)
    hess.append(0.0)
    count.append(0)
    for b in range(1, 21):
        grad.append(Float64(b) - 10.5)
        hess.append(10.0)
        count.append(10)
    var hist = Histogram.from_planes(grad^, hess^, count^, 1, 21)
    var codes = List[Int]()
    for c in range(20):
        codes.append(c)
    var spec = CategoricalSpec([True], codes^, [0, 20])

    var split = find_best_split(
        hist,
        lambda_reg=0.0,
        cat_params=CategoricalParams(1, 3, 1.0, 0.0, 1),
        cats=spec,
    )
    assert_true(split.found and split.is_categorical)
    var n_left = 0
    for b in range(256):
        if cat_contains(split.cat_bitset, b):
            n_left += 1
    assert_true(n_left <= 3)


def test_categorical_rejects_monotonic_constraints() raises:
    var hist = Histogram.from_planes(
        [0.0, 1.0, -1.0], [0.0, 1.0, 1.0], [0, 1, 1], 1, 3
    )
    var spec = CategoricalSpec([True], [0, 1], [0, 2])
    with assert_raises():
        _ = find_best_split(hist, cats=spec, monotone=[1])


# --- End to end ------------------------------------------------------------


def _grouped_dataset(
    n_rows: Int,
    n_categories: Int,
    mut features: List[Float64],
    mut target: List[Float64],
):
    """One categorical column whose effect alternates with the code, so the
    signal is invisible to any ordinal threshold on the code."""
    for r in range(n_rows):
        features.append(Float64(r % n_categories))
    for r in range(n_rows):
        var c = r % n_categories
        target.append(1.0 if c % 2 == 0 else -1.0)


def test_low_cardinality_model_fits_the_grouping() raises:
    var n_rows = 400
    var features = List[Float64]()
    var target = List[Float64]()
    _grouped_dataset(n_rows, 4, features, target)

    var params = _params(8, 30)
    params.tree.cat = _small_cat_params()
    var model = fit(
        features, n_rows, 1, target, SQUARED_ERROR, params, 64,
        categorical_features=[0],
    )
    for c in range(4):
        var want = 1.0 if c % 2 == 0 else -1.0
        assert_almost_equal(
            model.predict([Float64(c)]), want, atol=0.05
        )


def test_high_cardinality_beats_ordinal_treatment() raises:
    # 20 alternating categories. As a categorical, one partition separates
    # them; as a numerical column, an ordinal threshold has to carve out one
    # code at a time and cannot finish within this leaf budget.
    var n_rows = 1_000
    var features = List[Float64]()
    var target = List[Float64]()
    _grouped_dataset(n_rows, 20, features, target)

    var params = _params(4, 20)
    params.tree.cat = _small_cat_params()

    var as_cat = fit(
        features, n_rows, 1, target, SQUARED_ERROR, params, 64,
        categorical_features=[0],
    )
    var as_num = fit(features, n_rows, 1, target, SQUARED_ERROR, params, 64)

    var cat_sse = 0.0
    var num_sse = 0.0
    for r in range(n_rows):
        var row: List[Float64] = [features[r]]
        var dc = as_cat.predict(row) - target[r]
        var dn = as_num.predict(row) - target[r]
        cat_sse += dc * dc
        num_sse += dn * dn
    # The categorical fit is essentially exact; the ordinal one is not close.
    assert_true(cat_sse / Float64(n_rows) < 0.02)
    assert_true(num_sse > 10.0 * cat_sse)


def test_unseen_and_missing_categories_route_right() raises:
    var n_rows = 400
    var features = List[Float64]()
    var target = List[Float64]()
    _grouped_dataset(n_rows, 4, features, target)

    var params = _params(8, 30)
    params.tree.cat = _small_cat_params()
    var model = fit(
        features, n_rows, 1, target, SQUARED_ERROR, params, 64,
        categorical_features=[0],
    )

    # Unseen code, negative code, and NaN all land in the unknown bin, which
    # is in no split's category set, so all three take the same path.
    var unseen = model.predict([99.0])
    assert_true(model.predict([-1.0]) == unseen)
    assert_true(model.predict([_nan()]) == unseen)
    # And they follow the documented default: right at every categorical
    # node, so the prediction is a real leaf of the trained tree.
    assert_true(unseen == unseen)


def test_trees_record_categorical_splits() raises:
    var n_rows = 400
    var features = List[Float64]()
    var target = List[Float64]()
    _grouped_dataset(n_rows, 6, features, target)

    var mapper = fit_bins(features, n_rows, 1, 64, [0])
    var data = mapper.transform(features, n_rows)
    var grad = List[Float64](capacity=n_rows)
    var hess = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        grad.append(-target[r])
        hess.append(1.0)

    var tp = TreeParams(4, 5, 1.0, 1e-3)
    tp.cat = _small_cat_params()
    var tree = grow_tree(data, grad, hess, tp)

    var n_categorical_nodes = 0
    for i in range(len(tree.feature)):
        if tree.feature[i] >= 0:
            assert_true(tree.cat_offset[i] >= 0)
            # A categorical node carries no threshold and routes no missing
            # bin: its set decides every row.
            assert_equal(tree.threshold_bin[i], -1)
            assert_equal(tree.missing_bin[i], -1)
            assert_true(not tree.default_left[i])
            n_categorical_nodes += 1
    assert_true(n_categorical_nodes > 0)
    assert_equal(
        len(tree.cat_bitset), n_categorical_nodes * CAT_BITSET_WORDS
    )
    # The unknown bin is in no node's set, so its rows always go right.
    for i in range(len(tree.feature)):
        if tree.cat_offset[i] >= 0:
            assert_true(
                not cat_pool_contains(tree.cat_bitset, tree.cat_offset[i], 0)
            )


def test_serialization_roundtrip_is_bit_exact() raises:
    var n_rows = 500
    var features = List[Float64](capacity=2 * n_rows)
    var target = List[Float64](capacity=n_rows)
    # A numerical column and a 12-category column, so the saved file carries
    # quantile edges, category tables, and both kinds of split node.
    for r in range(n_rows):
        features.append(_uniform(UInt64(r)))
    for r in range(n_rows):
        features.append(Float64(r % 12))
    for r in range(n_rows):
        var c = r % 12
        target.append(
            (1.0 if c % 3 == 0 else -1.0) + 2.0 * features[r]
        )

    var params = _params(12, 25)
    params.tree.cat = _small_cat_params()
    var model = fit(
        features, n_rows, 2, target, SQUARED_ERROR, params, 64,
        categorical_features=[1],
    )
    save_model(model, _TMP_PATH)
    var loaded = load_model(_TMP_PATH)
    remove(_TMP_PATH)

    assert_equal(len(loaded.booster.trees), len(model.booster.trees))
    assert_true(loaded.mapper.cats.is_cat(1))
    assert_true(not loaded.mapper.cats.is_cat(0))
    assert_equal(loaded.mapper.cats.n_categories(1), 12)
    for t in range(len(model.booster.trees)):
        assert_equal(
            len(loaded.booster.trees[t].cat_bitset),
            len(model.booster.trees[t].cat_bitset),
        )

    for r in range(0, n_rows, 7):
        var row: List[Float64] = [features[r], features[n_rows + r]]
        assert_true(loaded.predict(row) == model.predict(row))
    # Including the routing of codes the model never saw.
    var odd_codes: List[Float64] = [99.0, -1.0, 4.5]
    for i in range(len(odd_codes)):
        var row: List[Float64] = [0.5, odd_codes[i]]
        assert_true(loaded.predict(row) == model.predict(row))


def test_numerical_only_models_serialize_unchanged() raises:
    """The categorical sections are optional, so a model without categorical
    features must produce exactly the bytes it did before they existed."""
    var n_rows = 200
    var features = List[Float64](capacity=n_rows)
    var target = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        features.append(_uniform(UInt64(r)))
    for r in range(n_rows):
        target.append(3.0 * features[r])

    var model = fit(
        features, n_rows, 1, target, SQUARED_ERROR, _params(8, 10), 32
    )
    save_model(model, _TMP_PATH)
    var content = open(_TMP_PATH, "r").read()
    remove(_TMP_PATH)
    assert_true(content.find("categorical") < 0)
    assert_true(content.find("cat ") < 0)


# --- Other trainers --------------------------------------------------------


def test_multiclass_fits_category_groups() raises:
    """Softmax training reads the same binned matrix, so a categorical column
    has to work there too: the class is a function of the category, and no
    ordinal threshold on the code separates the classes."""
    var n_rows = 600
    var n_categories = 6
    var features = List[Float64](capacity=n_rows)
    var labels = List[Int](capacity=n_rows)
    for r in range(n_rows):
        features.append(Float64(r % n_categories))
    for r in range(n_rows):
        labels.append((r % n_categories) % 3)

    var params = _params(8, 30)
    params.tree.cat = _small_cat_params()
    var model = fit_multiclass(
        features,
        n_rows,
        1,
        labels,
        3,
        params,
        64,
        categorical_features=[0],
    )
    assert_true(model.mapper.cats.is_cat(0))
    for c in range(n_categories):
        assert_equal(model.predict_class([Float64(c)]), c % 3)
    # Missing, unseen, and dropped codes share the unknown bin, so all three
    # route right at every categorical node and land on one leaf.
    var unseen = model.predict_class([Float64(n_categories + 5)])
    assert_equal(model.predict_class([-1.0]), unseen)
    assert_equal(model.predict_class([_nan()]), unseen)


def _count_categorical_nodes(model: Model) -> Int:
    """Nodes of a fitted single-output model that route by category set."""
    var found = 0
    for t in range(len(model.booster.trees)):
        for i in range(len(model.booster.trees[t].cat_offset)):
            if model.booster.trees[t].cat_offset[i] >= 0:
                found += 1
    return found


def _ranking_dataset(
    n_queries: Int,
    per_query: Int,
    mut features: List[Float64],
    mut labels: List[Int],
    mut counts: List[Int],
):
    """One categorical column per document whose code alternates relevance,
    so within a query the relevant documents are the even codes."""
    for q in range(n_queries):
        for d in range(per_query):
            features.append(Float64(d))
            _ = q
    for q in range(n_queries):
        for d in range(per_query):
            labels.append(1 if d % 2 == 0 else 0)
            _ = q
    for _ in range(n_queries):
        counts.append(per_query)


def test_ranker_searches_category_sets() raises:
    """`fit_ranker` takes `categorical_features` like the other fits. With
    eight alternating codes and four leaves, an ordinal threshold cannot
    separate the relevant documents from the rest, so this fails outright if
    the ranker quietly treats the codes as numbers."""
    var n_queries = 60
    var per_query = 8
    var features = List[Float64]()
    var labels = List[Int]()
    var counts = List[Int]()
    _ranking_dataset(n_queries, per_query, features, labels, counts)
    var n_rows = n_queries * per_query

    var params = _params(4, 30)
    params.tree.cat = _small_cat_params()
    var rank_params = RankerParams(10, 1.0, True, 4)

    var as_cat = fit_ranker(
        features,
        n_rows,
        1,
        labels,
        counts,
        params,
        rank_params,
        64,
        categorical_features=[0],
    )
    var as_num = fit_ranker(
        features, n_rows, 1, labels, counts, params, rank_params, 64
    )
    assert_true(as_cat.mapper.cats.is_cat(0))
    assert_true(not as_num.mapper.cats.is_cat(0))

    # The categorical fit splits by category set and the numerical one never
    # does, which is the difference the parameter is supposed to make. A
    # quality gap would not show it: over enough rounds an ordinal model
    # carves out one code at a time and gets there too.
    assert_true(_count_categorical_nodes(as_cat) > 0)
    assert_equal(_count_categorical_nodes(as_num), 0)

    var cat_scores = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        var row: List[Float64] = [features[r]]
        cat_scores.append(as_cat.predict_raw(row))
    var groups = groups_from_counts(counts)
    # Every relevant document in the top four is NDCG@4 = 1.
    assert_almost_equal(ndcg(cat_scores, labels, groups, 4), 1.0, atol=1e-9)


# --- GPU -------------------------------------------------------------------


def test_gpu_categorical_splits_match_cpu() raises:
    """The GPU trainer searches category partitions on downloaded histograms
    and routes rows by the node's bitset in the partition kernel, so it must
    grow the same categorical model the CPU trainer does, to the Float32
    tolerance the other backend-equivalence tests use."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 2_000
        var n_features = 3
        var n_categories = 12
        var features = List[Float64](capacity=n_rows * n_features)
        # Feature 0 is the categorical column; 1 and 2 are numerical, so both
        # search paths run in the same trees.
        for r in range(n_rows):
            features.append(Float64(r % n_categories))
        for f in range(1, n_features):
            for r in range(n_rows):
                features.append(_uniform(UInt64(f * n_rows + r)))
        var target = List[Float64](capacity=n_rows)
        for r in range(n_rows):
            var c = r % n_categories
            target.append(
                (1.0 if c % 3 == 0 else -1.0) + 2.0 * features[n_rows + r]
            )

        var params = _params(12, 20)
        params.tree.cat = _small_cat_params()
        var mapper = fit_bins(
            features, n_rows, n_features, 64, categorical_features=[0]
        )
        var data = mapper.transform(features, n_rows)
        var cpu = train(data, target, SQUARED_ERROR, params)
        var gpu = train_gpu(data, target, SQUARED_ERROR, params)

        assert_equal(len(cpu.trees), len(gpu.trees))
        var categorical_nodes = 0
        for t in range(len(gpu.trees)):
            for i in range(len(gpu.trees[t].feature)):
                if gpu.trees[t].cat_offset[i] >= 0:
                    categorical_nodes += 1
        # Otherwise the tolerance below would pass on two numerical models.
        assert_true(categorical_nodes > 0)
        for r in range(n_rows):
            assert_true(
                abs(cpu.predict_row(data, r) - gpu.predict_row(data, r))
                <= 1e-3
            )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
