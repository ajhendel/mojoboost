"""Row and feature sampling primitives.

Covers the pieces sampling.mojo adds on top of the per-tree and per-node
feature draws that tests/test_feature_sampling.mojo already pins down: the
per-level draw and its composition with the other two, the parameter alias
table, class-conditional row bagging, and the dense range/mask/scale forms a
device-side active-row pass consumes. The GOSS case checks the amplification
those dense forms have to preserve, against goss.mojo's own selection.
"""

from std.testing import assert_almost_equal, assert_equal, assert_true, TestSuite

from mojotrees.bagging import BaggingParams, DEFAULT_BAGGING_SEED, sample_rows
from mojotrees.goss import GossParams, goss_select
from mojotrees.sampling import (
    ClassBaggingParams,
    DEFAULT_FEATURE_FRACTION_BYLEVEL,
    canonical_data_sample_strategy,
    canonical_sampling_param,
    check_row_set,
    contiguous_ranges,
    expand_row_scale,
    has_positive_rows,
    is_sampling_param,
    ranges_row_count,
    row_mask,
    sample_rows_by_class,
    sampling_param_names,
    select_level_features,
    select_node_features,
    select_split_features,
    select_tree_features,
    selection_count,
    refresh_class_bag,
)


def _same(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def _is_subset(sub: List[Int], of: List[Int]) -> Bool:
    """Both lists ascending, so one walk decides it."""
    var i = 0
    for j in range(len(sub)):
        while i < len(of) and of[i] != sub[j]:
            i += 1
        if i >= len(of):
            return False
        i += 1
    return True


def _alternating_labels(n_pos: Int, n_neg: Int) -> List[Float64]:
    """Positives at even indices until they run out, so neither class sits in
    one contiguous block."""
    var labels = List[Float64](capacity=n_pos + n_neg)
    var pos_left = n_pos
    var neg_left = n_neg
    for r in range(n_pos + n_neg):
        var take_pos = (r % 2 == 0 and pos_left > 0) or neg_left == 0
        if take_pos:
            labels.append(1.0)
            pos_left -= 1
        else:
            labels.append(0.0)
            neg_left -= 1
    return labels^


def test_bylevel_selection_is_ordered_and_a_subset_of_the_tree_set() raises:
    var tree_features = select_tree_features(60, 0.5, 5, 2)
    for depth in range(6):
        var level = select_level_features(tree_features, 0.4, 5, 2, depth)
        assert_equal(len(level), selection_count(len(tree_features), 0.4))
        for i in range(1, len(level)):
            assert_true(level[i] > level[i - 1])
        assert_true(_is_subset(level, tree_features))


def test_bylevel_selection_is_deterministic_and_depth_dependent() raises:
    var tree_features = select_tree_features(60, 0.5, 5, 2)
    var a = select_level_features(tree_features, 0.4, 5, 2, 3)
    var b = select_level_features(tree_features, 0.4, 5, 2, 3)
    assert_true(_same(a, b))
    # A different depth, tree, or seed each moves the draw. 12 of 30 features:
    # agreement by chance is remote.
    assert_true(not _same(a, select_level_features(tree_features, 0.4, 5, 2, 4)))
    assert_true(not _same(a, select_level_features(tree_features, 0.4, 5, 3, 3)))
    assert_true(not _same(a, select_level_features(tree_features, 0.4, 6, 2, 3)))


def test_bylevel_one_leaves_the_node_draw_untouched() raises:
    """The composed call has to be a drop-in for the per-node call, or turning
    the parameter on by default would silently change every model."""
    var tree_features = select_tree_features(40, 0.6, 11, 1)
    assert_true(DEFAULT_FEATURE_FRACTION_BYLEVEL == 1.0)
    for node in range(8):
        var depth = node // 2
        var composed = select_split_features(
            tree_features,
            DEFAULT_FEATURE_FRACTION_BYLEVEL,
            0.5,
            11,
            1,
            depth,
            node,
        )
        var plain = select_node_features(tree_features, 0.5, 11, 1, node)
        assert_true(_same(composed, plain))
    # And with both fractions at 1.0 the node keeps the whole tree set.
    var everything = select_split_features(
        tree_features, 1.0, 1.0, 11, 1, 3, 7
    )
    assert_true(_same(everything, tree_features))


def test_fraction_accounting_composes_across_the_three_draws() raises:
    """Each stage rounds by selection_count in turn, so the node count is the
    three formulas applied in order and never the product of the fractions
    rounded once."""
    var n_features = 100
    var tree_features = select_tree_features(n_features, 0.5, 4, 0)
    assert_equal(len(tree_features), selection_count(n_features, 0.5))
    var level = select_level_features(tree_features, 0.6, 4, 0, 2)
    assert_equal(len(level), selection_count(len(tree_features), 0.6))
    var node = select_split_features(tree_features, 0.6, 0.5, 4, 0, 2, 9)
    assert_equal(len(node), selection_count(len(level), 0.5))
    # 100 -> 50 -> 30 -> 15, and the node's features come from the level's.
    assert_equal(len(node), 15)
    assert_true(_is_subset(node, level))
    assert_true(_is_subset(level, tree_features))


def test_bylevel_rejects_bad_fractions_and_depths() raises:
    var tree_features = select_tree_features(20, 1.0, 1, 0)
    var bad: List[Float64] = [0.0, -0.5, 1.5]
    for i in range(len(bad)):
        var raised = False
        try:
            _ = select_level_features(tree_features, bad[i], 1, 0, 0)
        except:
            raised = True
        assert_true(raised)
    var raised_depth = False
    try:
        _ = select_level_features(tree_features, 0.5, 1, 0, -1)
    except:
        raised_depth = True
    assert_true(raised_depth)


def test_alias_table_accepts_the_spellings_the_audit_flags() raises:
    # The two the parity audit records as missing.
    assert_equal(canonical_sampling_param("colsample_bytree"), "feature_fraction")
    assert_equal(
        canonical_sampling_param("colsample_bynode"), "feature_fraction_bynode"
    )
    # XGBoost's per-level name, which mojotrees adds.
    assert_equal(
        canonical_sampling_param("colsample_bylevel"),
        "feature_fraction_bylevel",
    )
    # LightGBM's own row-sampling aliases.
    assert_equal(canonical_sampling_param("subsample"), "bagging_fraction")
    assert_equal(canonical_sampling_param("sub_row"), "bagging_fraction")
    assert_equal(canonical_sampling_param("subsample_freq"), "bagging_freq")
    assert_equal(
        canonical_sampling_param("bagging_fraction_seed"), "bagging_seed"
    )
    assert_equal(canonical_sampling_param("sub_feature"), "feature_fraction")
    assert_equal(
        canonical_sampling_param("pos_subsample"), "pos_bagging_fraction"
    )
    assert_equal(
        canonical_sampling_param("neg_sub_row"), "neg_bagging_fraction"
    )
    assert_equal(
        canonical_sampling_param("pos_bagging"), "pos_bagging_fraction"
    )
    assert_equal(
        canonical_sampling_param("neg_bagging"), "neg_bagging_fraction"
    )
    # `bagging` alone is the row fraction, not a class fraction, so the three
    # spellings must not collapse into each other.
    assert_equal(canonical_sampling_param("bagging"), "bagging_fraction")


def test_every_canonical_name_resolves_to_itself() raises:
    var names = sampling_param_names()
    assert_true(len(names) > 0)
    for i in range(len(names)):
        assert_equal(canonical_sampling_param(names[i]), names[i])
        assert_true(is_sampling_param(names[i]))


def test_unknown_parameter_names_are_rejected() raises:
    var unknown: List[String] = [
        String("colsample"),
        String("bagging_fractions"),
        String("learning_rate"),
        String(""),
    ]
    for i in range(len(unknown)):
        assert_true(not is_sampling_param(unknown[i]))
        var raised = False
        try:
            _ = canonical_sampling_param(unknown[i])
        except:
            raised = True
        assert_true(raised)


def test_data_sample_strategy_values() raises:
    assert_equal(canonical_data_sample_strategy("bagging"), "bagging")
    assert_equal(canonical_data_sample_strategy("goss"), "goss")
    var raised = False
    try:
        _ = canonical_data_sample_strategy("gbdt")
    except:
        raised = True
    assert_true(raised)


def test_equal_class_fractions_reproduce_a_uniform_bag() raises:
    """Class bagging draws from the same stream uniform bagging does, so
    equal fractions must give the same rows. 400 rows at 0.5 leaves the
    empty-class guard unreachable, which is the only way the two can part."""
    var labels = _alternating_labels(200, 200)
    var uniform = List[Int]()
    var by_class = List[Int]()
    for bag_index in range(4):
        sample_rows(BaggingParams(0.5, 1, 7), 400, bag_index, uniform)
        sample_rows_by_class(
            ClassBaggingParams(0.5, 0.5, 1, 7), labels, bag_index, by_class
        )
        assert_true(len(uniform) > 0)
        assert_true(_same(uniform, by_class))


def test_each_class_is_kept_at_its_own_rate() raises:
    var n_pos = 1000
    var n_neg = 3000
    var labels = _alternating_labels(n_pos, n_neg)
    var rows = List[Int]()
    sample_rows_by_class(
        ClassBaggingParams(0.8, 0.2, 1, 13), labels, 0, rows
    )
    var kept_pos = 0
    var kept_neg = 0
    for i in range(len(rows)):
        if labels[rows[i]] > 0.0:
            kept_pos += 1
        else:
            kept_neg += 1
    # Binomial(1000, 0.8) and Binomial(3000, 0.2): four standard deviations
    # are well inside these bounds, and the draw is fixed by the seed anyway.
    assert_true(kept_pos > 750 and kept_pos < 850)
    assert_true(kept_neg > 540 and kept_neg < 660)
    # The unbagged rates would be 1000 and 3000, so both classes really moved.
    assert_true(kept_pos < n_pos)
    assert_true(kept_neg < n_neg)


def test_class_bag_is_deterministic_ascending_and_in_range() raises:
    var labels = _alternating_labels(60, 140)
    var params = ClassBaggingParams(0.4, 0.6, 1, 21)
    var a = List[Int]()
    var b = List[Int]()
    sample_rows_by_class(params, labels, 2, a)
    sample_rows_by_class(params, labels, 2, b)
    assert_true(_same(a, b))
    check_row_set(a, len(labels))
    # A different bag index moves the draw.
    var other = List[Int]()
    sample_rows_by_class(params, labels, 3, other)
    assert_true(not _same(a, other))


def test_a_present_class_never_vanishes_from_the_bag() raises:
    """A fraction small enough to drop a whole class must still leave one row
    of it, or the tree would train on a single-class bag."""
    var labels = _alternating_labels(3, 200)
    var rows = List[Int]()
    sample_rows_by_class(
        ClassBaggingParams(1e-9, 1e-9, 1, 5), labels, 0, rows
    )
    var kept_pos = 0
    var kept_neg = 0
    for i in range(len(rows)):
        if labels[rows[i]] > 0.0:
            kept_pos += 1
        else:
            kept_neg += 1
    assert_equal(kept_pos, 1)
    assert_equal(kept_neg, 1)
    # An absent class adds nothing: all-positive labels give a positive-only
    # bag rather than a fabricated negative row.
    var only_pos: List[Float64] = [1.0, 1.0, 1.0, 1.0]
    var pos_rows = List[Int]()
    sample_rows_by_class(
        ClassBaggingParams(1e-9, 1e-9, 1, 5), only_pos, 0, pos_rows
    )
    assert_equal(len(pos_rows), 1)


def test_class_bagging_schedule_and_disabled_state() raises:
    var labels = _alternating_labels(100, 100)
    var bag = List[Int]()
    # Disabled: the bag stays empty, which means "all rows" downstream.
    for i in range(4):
        refresh_class_bag(bag, ClassBaggingParams.disabled(), labels, i)
        assert_equal(len(bag), 0)
    assert_true(not ClassBaggingParams.disabled().enabled())
    assert_true(not ClassBaggingParams(0.5, 1.0, 0, 3).enabled())
    assert_true(ClassBaggingParams(0.5, 1.0, 1, 3).enabled())
    assert_true(ClassBaggingParams(1.0, 0.5, 1, 3).enabled())
    assert_equal(ClassBaggingParams.disabled().seed, Int(DEFAULT_BAGGING_SEED))

    # freq 3: a new bag on rounds 0, 3, 6 and the same bag in between.
    var params = ClassBaggingParams(0.5, 0.5, 3, 9)
    var expected = List[Int]()
    refresh_class_bag(bag, params, labels, 0)
    sample_rows_by_class(params, labels, 0, expected)
    assert_true(_same(bag, expected))
    refresh_class_bag(bag, params, labels, 1)
    refresh_class_bag(bag, params, labels, 2)
    assert_true(_same(bag, expected))
    refresh_class_bag(bag, params, labels, 3)
    sample_rows_by_class(params, labels, 1, expected)
    assert_true(_same(bag, expected))


def test_positive_row_test_gates_balanced_bagging() raises:
    """LightGBM turns balanced bagging off outright when the dataset holds no
    positive row, and falls back to plain bagging_fraction. The parameters
    cannot see that on their own, so the labels decide it."""
    var mixed = _alternating_labels(2, 8)
    assert_true(has_positive_rows(mixed))
    var all_negative: List[Float64] = [0.0, 0.0, 0.0]
    assert_true(not has_positive_rows(all_negative))
    # -1/+1 labels count as positive on the same > 0 test the sampler uses.
    var signed: List[Float64] = [-1.0, -1.0, 1.0]
    assert_true(has_positive_rows(signed))
    var no_rows = List[Float64]()
    assert_true(not has_positive_rows(no_rows))
    # The gate a trainer applies is both halves together.
    var params = ClassBaggingParams(0.5, 0.5, 1, 3)
    assert_true(params.enabled() and has_positive_rows(mixed))
    assert_true(not (params.enabled() and has_positive_rows(all_negative)))


def test_invalid_class_bagging_settings_raise() raises:
    var labels = _alternating_labels(5, 5)
    var rows = List[Int]()
    var bad: List[ClassBaggingParams] = [
        ClassBaggingParams(0.0, 0.5, 1, 3),
        ClassBaggingParams(0.5, 1.5, 1, 3),
        ClassBaggingParams(-0.1, 0.5, 1, 3),
        ClassBaggingParams(0.5, 0.5, -1, 3),
    ]
    for i in range(len(bad)):
        var raised = False
        try:
            sample_rows_by_class(bad[i], labels, 0, rows)
        except:
            raised = True
        assert_true(raised)
    # An empty label list and a negative bag index are rejected too.
    var no_labels = List[Float64]()
    var raised_empty = False
    try:
        sample_rows_by_class(ClassBaggingParams(0.5, 0.5, 1, 3), no_labels, 0, rows)
    except:
        raised_empty = True
    assert_true(raised_empty)
    var raised_index = False
    try:
        sample_rows_by_class(
            ClassBaggingParams(0.5, 0.5, 1, 3), labels, -1, rows
        )
    except:
        raised_index = True
    assert_true(raised_index)


def test_row_set_validation_catches_unordered_and_out_of_range_sets() raises:
    var good: List[Int] = [0, 3, 7]
    check_row_set(good, 8)
    var cases: List[List[Int]] = [
        [0, 3, 3],
        [3, 1],
        [0, 8],
        [-1, 2],
    ]
    for i in range(len(cases)):
        var raised = False
        try:
            check_row_set(cases[i], 8)
        except:
            raised = True
        assert_true(raised)


def test_contiguous_ranges_cover_exactly_the_sampled_rows() raises:
    var rows: List[Int] = [1, 2, 3, 7, 9, 10]
    var ranges = contiguous_ranges(rows, 12)
    var expected: List[Int] = [1, 4, 7, 8, 9, 11]
    assert_true(_same(ranges, expected))
    assert_equal(ranges_row_count(ranges), len(rows))

    # A contiguous set collapses to one range, which is the case a device-side
    # pass can walk without an index indirection.
    var block: List[Int] = [4, 5, 6, 7]
    var one = contiguous_ranges(block, 12)
    assert_equal(len(one), 2)
    assert_equal(one[0], 4)
    assert_equal(one[1], 8)

    # The "every row" convention yields the whole dataset as one range.
    var everything = contiguous_ranges(List[Int](), 12)
    assert_equal(len(everything), 2)
    assert_equal(everything[0], 0)
    assert_equal(everything[1], 12)
    assert_equal(ranges_row_count(everything), 12)

    # Singletons stay singletons.
    var sparse: List[Int] = [0, 2, 4]
    var singles = contiguous_ranges(sparse, 5)
    assert_equal(len(singles), 6)
    assert_equal(ranges_row_count(singles), 3)


def test_row_mask_matches_the_ranges() raises:
    var rows: List[Int] = [1, 2, 3, 7, 9, 10]
    var mask = row_mask(rows, 12)
    assert_equal(len(mask), 12)
    var ranges = contiguous_ranges(rows, 12)
    var from_ranges = List[Bool](capacity=12)
    from_ranges.resize(12, False)
    for i in range(0, len(ranges), 2):
        for r in range(ranges[i], ranges[i + 1]):
            from_ranges[r] = True
    var covered = 0
    for r in range(12):
        assert_true(mask[r] == from_ranges[r])
        if mask[r]:
            covered += 1
    assert_equal(covered, len(rows))
    # Empty means every row.
    var all_rows = row_mask(List[Int](), 5)
    for r in range(5):
        assert_true(all_rows[r])


def test_expand_row_scale_carries_goss_amplification() raises:
    """GOSS keeps the large-gradient rows at weight 1 and scales the sampled
    small-gradient rows by (n - top_k) / other_k, so the sampled weight adds
    back up to the full row count. The dense form has to preserve that, and
    has to zero the rows outside the sample so a kernel that skips the mask
    still builds the sampled histogram."""
    var n = 100
    # Distinct importances, so no tie can inflate the kept set past top_k.
    var importance = List[Float64](capacity=n)
    for r in range(n):
        importance.append(Float64(r + 1) / Float64(n))
    var params = GossParams.enable(0.2, 0.1, 7, 0)
    var selection = goss_select(importance, params, 0)

    var top_k = 20
    var other_k = 10
    assert_equal(selection.n_top, top_k)
    assert_equal(selection.n_other, other_k)
    assert_almost_equal(selection.multiplier, 8.0, atol=1e-12)

    var dense = expand_row_scale(selection.rows, selection.scale, n)
    assert_equal(len(dense), n)

    var mask = row_mask(selection.rows, n)
    var amplified = 0
    var kept = 0
    var total_weight = 0.0
    for r in range(n):
        total_weight += dense[r]
        if not mask[r]:
            # Outside the sample the multiplier is zero, not one.
            assert_equal(dense[r], 0.0)
            continue
        if dense[r] == 1.0:
            kept += 1
        else:
            assert_almost_equal(dense[r], selection.multiplier, atol=1e-12)
            amplified += 1
    assert_equal(kept, top_k)
    assert_equal(amplified, other_k)
    # 20 rows at weight 1 plus 10 at weight 8 is the full 100 rows of weight.
    assert_almost_equal(total_weight, Float64(n), atol=1e-9)

    # The dense form reproduces the sampled gradient sum over every row.
    var grad = List[Float64](capacity=n)
    for r in range(n):
        grad.append(Float64(r) - 40.0)
    var dense_sum = 0.0
    for r in range(n):
        dense_sum += dense[r] * grad[r]
    var sampled_sum = 0.0
    for i in range(len(selection.rows)):
        sampled_sum += selection.scale[i] * grad[selection.rows[i]]
    assert_almost_equal(dense_sum, sampled_sum, atol=1e-9)


def test_expand_row_scale_edges() raises:
    # An unweighted sample (a bag) needs no scale list.
    var rows: List[Int] = [0, 2]
    var ones = expand_row_scale(rows, List[Float64](), 4)
    assert_equal(ones[0], 1.0)
    assert_equal(ones[1], 0.0)
    assert_equal(ones[2], 1.0)
    assert_equal(ones[3], 0.0)
    # "Every row" gives all ones, so unsampled training needs no branch.
    var everything = expand_row_scale(List[Int](), List[Float64](), 3)
    for r in range(3):
        assert_equal(everything[r], 1.0)
    # A mismatched scale list is rejected.
    var bad_scale: List[Float64] = [1.0]
    var raised = False
    try:
        _ = expand_row_scale(rows, bad_scale, 4)
    except:
        raised = True
    assert_true(raised)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
