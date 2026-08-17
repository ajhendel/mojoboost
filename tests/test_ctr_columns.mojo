"""CTR columns in a design matrix (src/mojotrees/ctr_columns.mojo), catalog A19.

**UNRUN. Nothing in this file has been executed, compiled, or type-checked.**
The lane that wrote it was under a standing order not to run anything -- no
suite, no single file, no `mojo build`, no `pixi run` of any kind -- so every
expectation below is analytical, worked out by hand from CatBoost's source, and
none of it is evidence that the code compiles. Treat a green run of this file as
the first evidence, not a confirmation.

What each test pins, and why it is here rather than in `test_ctr.mojo` (which
covers the arithmetic and the four loops, and which does run):

- **The column count.** Four numeric columns per categorical column at
  CatBoost's CPU defaults, not one: `Borders x 3 priors x 1 target border`, plus
  `Counter x 1 prior`. This is A19's headline number and the reason CTRs cost
  what they cost, and it is the single thing most likely to be got wrong by
  someone reading only the paper.
- **The one-hot cutoff.** A column earns CTRs only when it is too wide to
  one-hot (`uniqueValues > one_hot_max_size`), which is the same quantity that
  turns CatBoost's permutation on at all (`IsPermutationNeeded`'s `hasCtrs`).
  A two-level column must produce nothing.
- **Train is not predict.** The ordered columns and the static tables must be
  built from the same category array and must NOT agree in general -- one is a
  prefix in a permutation, the other is a full-set count. A test that asserted
  they matched would be asserting the bug.
- **Counter is not permutation-dependent.** Its column must be byte-identical
  under two different permutations, because `CalcOnlineCTRCounter` reads a table
  filled once before any row is emitted.
- **Determinism.** The same config and the same data must produce the same
  bytes, and the permutation must not depend on how it was consumed.
- **The refusals hold.** An enabled complexity above 1, a `Full`
  counter_calc_method, an unresolved target border, and a fitted set of tables
  reaching a writer must each raise by name.
"""

from std.testing import (
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
    TestSuite,
)

from mojotrees.ctr import (
    CTR_BORDERS,
    CTR_COUNTER,
    CtrParams,
    check_ctr_model_support,
    check_ctr_trainer_support,
    ctr_predict_bucket,
    ctr_predict_border,
)
from mojotrees.ctr_columns import (
    CTR_ONE_HOT_MAX_SIZE,
    CTR_TARGET_BORDER_MIN_ENTROPY,
    CTR_TARGET_BORDER_MULTICLASS,
    CtrTables,
    SimpleCtrConfig,
    build_ctr_train_columns,
    catboost_simple_ctr_defaults,
    check_ctr_serializable,
    ctr_columns_per_categorical,
    ctr_predict_columns,
    ctr_predict_row,
    ctr_source_features,
    ctr_train_permutation,
    default_target_borders,
    fit_ctr_tables,
    plan_ctr_columns,
    select_target_borders,
    target_classes,
)


# ---------------------------------------------------------------------------
# Fixtures, small enough to check on paper
# ---------------------------------------------------------------------------


def _binary_config() raises -> SimpleCtrConfig:
    """CatBoost's CPU `simple_ctr` defaults with the borders already resolved
    for a 0/1 target, so `validate` is satisfied."""
    var borders = List[Float64]()
    borders.append(0.5)
    return SimpleCtrConfig.catboost_defaults(borders^)


def _one_wide_column() raises -> CtrTables:
    """Feature 0 numerical, feature 1 categorical with four kept categories."""
    var flags = List[Bool]()
    flags.append(False)
    flags.append(True)
    var counts = List[Int]()
    counts.append(0)
    counts.append(4)
    var tables = plan_ctr_columns(_binary_config(), flags, counts, 255)
    # The bucket table. Codes are 1..4 so that a raw value equals its bucket,
    # which keeps every assertion below readable; the mapping itself is
    # `CtrTables.bucket_of` and is exercised by the unseen-code test.
    tables.slot_codes = [1, 2, 3, 4]
    tables.slot_code_offsets = [0, 4]
    return tables^


# ---------------------------------------------------------------------------
# The column count and the layout
# ---------------------------------------------------------------------------


def test_four_columns_per_categorical_column_at_the_defaults() raises:
    # A19's headline number, and the reason CTRs are expensive:
    #   Borders x 3 priors x (2 classes - 1 border) = 3
    #   Counter x 1 prior                           = 1
    var config = _binary_config()
    assert_equal(ctr_columns_per_categorical(config), 4)

    var tables = _one_wide_column()
    assert_true(tables.is_active())
    assert_equal(tables.n_columns(), 4)
    assert_equal(tables.n_slots(), 1)
    assert_equal(tables.n_base_features, 2)
    assert_equal(tables.total_features(), 6)
    # `n_buckets = n_categories + 1`: bin 0 is `categorical.UNKNOWN_BIN` and is
    # a real bucket with real statistics, not CatBoost's absent-category case.
    assert_equal(tables.slot_buckets[0], 5)


def test_the_column_order_is_allocate_ctr_data_order() raises:
    # `AllocateCtrData(ctrIdx, targetBorderCount, priors.size())`
    # (`online_ctr.cpp:741`): description, then target border, then prior.
    var tables = _one_wide_column()
    assert_equal(tables.columns[0].ctr_type, CTR_BORDERS)
    assert_equal(tables.columns[0].prior, 0.0)
    assert_equal(tables.columns[1].ctr_type, CTR_BORDERS)
    assert_equal(tables.columns[1].prior, 0.5)
    assert_equal(tables.columns[2].ctr_type, CTR_BORDERS)
    assert_equal(tables.columns[2].prior, 1.0)
    assert_equal(tables.columns[3].ctr_type, CTR_COUNTER)
    assert_equal(tables.columns[3].prior, 0.0)
    for c in range(4):
        assert_equal(tables.columns[c].source_feature, 1)
        assert_equal(tables.columns[c].slot, 0)
        # `CalcNormalization` is the identity at every default prior.
        assert_equal(tables.columns[c].shift, 0.0)
        assert_equal(tables.columns[c].norm, 1.0)
        assert_equal(tables.columns[c].scale, 15.0)
        assert_equal(tables.columns[c].n_buckets(), 16)


def test_a_narrow_categorical_column_earns_no_ctrs() raises:
    # `greedy_tensor_search.cpp:469` and `IsPermutationNeeded`'s `hasCtrs`:
    # a column is given CTRs only when `uniqueValues > one_hot_max_size`.
    assert_equal(CTR_ONE_HOT_MAX_SIZE, 2)
    var config = _binary_config()
    var flags = List[Bool]()
    flags.append(True)
    flags.append(True)
    var counts = List[Int]()
    counts.append(2)  # one-hot, no CTRs
    counts.append(3)  # one over the cutoff, CTRs
    var sources = ctr_source_features(config, flags, counts, 255)
    assert_equal(len(sources), 1)
    assert_equal(sources[0], 1)

    # And a dataset with nothing but narrow categorical columns gets an
    # INACTIVE table rather than an active one with no columns, so no later
    # call pays a branch for an empty mechanism.
    var narrow_counts = List[Int]()
    narrow_counts.append(2)
    narrow_counts.append(1)
    var none_tables = plan_ctr_columns(config, flags, narrow_counts, 255)
    assert_false(none_tables.is_active())
    assert_equal(none_tables.n_columns(), 0)


def test_a_disabled_config_plans_nothing() raises:
    var flags = List[Bool]()
    flags.append(True)
    var counts = List[Int]()
    counts.append(10)
    var tables = plan_ctr_columns(
        SimpleCtrConfig.disabled(), flags, counts, 255
    )
    assert_false(tables.is_active())
    assert_equal(tables.n_columns(), 0)


# ---------------------------------------------------------------------------
# The target classifier
# ---------------------------------------------------------------------------


def test_the_default_target_border_is_the_midpoint_of_a_two_valued_label(
) raises:
    # A30's appendix: at `maxBordersCount = 1` the MinEntropy DP does not run
    # its main loop at all and collapses to `(values[t] + values[t+1]) / 2`
    # over the distinct target values. At two values that is the midpoint and
    # no entropy is evaluated.
    var label = List[Float64]()
    label.append(0.0)
    label.append(1.0)
    label.append(1.0)
    label.append(0.0)
    var borders = default_target_borders(label)
    assert_equal(len(borders), 1)
    assert_equal(borders[0], 0.5)

    # Strict `>` in `GetTargetClass`, so 0.0 is class 0 and 1.0 is class 1.
    var classes = target_classes(label, borders)
    assert_equal(classes[0], 0)
    assert_equal(classes[1], 1)
    assert_equal(classes[2], 1)
    assert_equal(classes[3], 0)


def test_a_continuous_target_is_binarized_rather_than_refused() raises:
    # This test asserted a REFUSAL until 2026-08-17, and the refusal was the
    # defect: `BuildTargetClassifier` (`target_classifier.cpp:39-118`, v1.2.10)
    # sends every non-multiclass loss to `SelectBorders`, which runs the border
    # selector over the raw target. There is no two-valued precondition
    # anywhere on that path.
    #
    # Three distinct values, counts 4 / 1 / 1, so the argmin is decided rather
    # than tied. W = [4, 5, 6], total 6, P(w) = w*log(w + 1e-8):
    #
    #   i = 0:  P(4) + P(2) = 4*log4 + 2*log2 = 5.545 + 1.386 = 6.931
    #   i = 1:  P(5) + P(1) = 5*log5 + 1*log1 = 8.047 + 0.000 = 8.047
    #
    # so t = 0 and the border is (0 + 10) / 2 = 5. A mid-scan two-value guess
    # would have put it at 5 as well, which is why the counts are skewed: 5 is
    # only reachable through the penalty when it beats i = 1.
    var label = List[Float64]()
    label.append(0.0)
    label.append(0.0)
    label.append(0.0)
    label.append(0.0)
    label.append(10.0)
    label.append(20.0)
    var borders = default_target_borders(label)
    assert_equal(len(borders), 1)
    assert_equal(borders[0], 5.0)

    # The other side of the same argmin, on the mirrored label: counts
    # 1 / 1 / 4 give W = [1, 2, 6], and
    #   i = 0:  P(1) + P(5) = 0.000 + 8.047 = 8.047
    #   i = 1:  P(2) + P(4) = 1.386 + 5.545 = 6.931
    # so t = 1 and the border is (10 + 20) / 2 = 15. The two fixtures differ
    # only in where the mass sits, so a rule that ignored the counts would
    # answer the same number twice.
    var mirrored = List[Float64]()
    mirrored.append(0.0)
    mirrored.append(10.0)
    mirrored.append(20.0)
    mirrored.append(20.0)
    mirrored.append(20.0)
    mirrored.append(20.0)
    var mirrored_borders = default_target_borders(mirrored)
    assert_equal(len(mirrored_borders), 1)
    assert_equal(mirrored_borders[0], 15.0)


def test_a_tie_keeps_the_smallest_index() raises:
    # `binarization.cpp:649` compares with a strict `<`, so an equal score does
    # not displace the incumbent and the SMALLEST index wins. Balanced class
    # codes tie exactly, which makes this reachable rather than theoretical:
    # for 0/1/2 at two rows each, W = [2, 4, 6] and
    #   i = 0:  P(2) + P(4)
    #   i = 1:  P(4) + P(2)
    # are the same sum. t = 0, border = 0.5 -- which is also
    # `GetMultiClassBorders(1)`, so the two arms agree on a balanced label.
    var label = List[Float64]()
    label.append(0.0)
    label.append(0.0)
    label.append(1.0)
    label.append(1.0)
    label.append(2.0)
    label.append(2.0)
    var borders = default_target_borders(label)
    assert_equal(len(borders), 1)
    assert_equal(borders[0], 0.5)

    var multi = select_target_borders(label, 1, CTR_TARGET_BORDER_MULTICLASS)
    assert_equal(len(multi), 1)
    assert_equal(multi[0], 0.5)


def test_a_constant_target_still_refuses() raises:
    # `SelectBorders`' `CB_ENSURE(borders.ysize() > 0 || allowConstLabel,
    # "0 target borders")` (`:29`) and the `targetBounds.Min != targetBounds.Max`
    # check above it (`:55-57`). `allowConstLabel` is false on every path that
    # reaches here.
    var constant = List[Float64]()
    constant.append(1.0)
    constant.append(1.0)
    with assert_raises():
        _ = default_target_borders(constant)

    var empty = List[Float64]()
    with assert_raises():
        _ = default_target_borders(empty)


def test_more_borders_than_the_early_return_covers_are_refused_by_name(
) raises:
    # `wsize <= bins` (`binarization.cpp:208-216`) is exact at any count: with
    # at most `count + 1` distinct values there is one border between every
    # adjacent pair and no penalty is evaluated. Four distinct values at
    # `count = 3` is that case.
    var four = List[Float64]()
    four.append(0.0)
    four.append(1.0)
    four.append(2.0)
    four.append(4.0)
    var borders = select_target_borders(four, 3)
    assert_equal(len(borders), 3)
    assert_equal(borders[0], 0.5)
    assert_equal(borders[1], 1.5)
    assert_equal(borders[2], 3.0)

    # Above it the real banded DP is needed and is not ported, so this refuses
    # rather than approximating. The refusal names the DP; it says nothing
    # about the shape of the target, which is the difference from what it
    # replaced.
    var five = List[Float64]()
    five.append(0.0)
    five.append(1.0)
    five.append(2.0)
    five.append(4.0)
    five.append(8.0)
    with assert_raises():
        _ = select_target_borders(five, 3)


def test_the_multiclass_arm_reads_no_target_value() raises:
    # `GetMultiClassBorders(cnt)` (`target_classifier.cpp:10-16`) is
    # `borders[i] = 0.5 + i` and never touches the target, so a label whose
    # MinEntropy border would be somewhere else entirely gets the class-code
    # borders and nothing about the label changes them.
    var label = List[Float64]()
    label.append(0.0)
    label.append(100.0)
    label.append(200.0)
    var borders = select_target_borders(
        label, 2, CTR_TARGET_BORDER_MULTICLASS
    )
    assert_equal(len(borders), 2)
    assert_equal(borders[0], 0.5)
    assert_equal(borders[1], 1.5)


def test_an_unresolved_target_border_is_refused_not_defaulted() raises:
    # An empty border list means one class, and one class makes
    # `GetTargetBorderCount` return 0 for Borders -- every column would silently
    # disappear. That is the exact defect this sequence exists to remove.
    var config = SimpleCtrConfig.catboost_defaults()
    with assert_raises():
        config.validate()


# ---------------------------------------------------------------------------
# Train versus predict
# ---------------------------------------------------------------------------


def _four_row_fixture() raises -> List[Int]:
    """One slot, four rows, categories 1, 1, 2, 2 (slot-major)."""
    var cat = List[Int]()
    cat.append(1)
    cat.append(1)
    cat.append(2)
    cat.append(2)
    return cat^


def _identity_permutation(n: Int) -> List[Int]:
    var p = List[Int]()
    for i in range(n):
        p.append(i)
    return p^


def test_the_ordered_column_is_a_prefix_and_the_first_row_gets_the_prior(
) raises:
    # `CalcOnlineCTRSimple` reads before it writes, in permutation order, so the
    # first row of a category sees `0 / 0` and gets the pure prior. At prior 0
    # that is `floor(0 / 1 * 15) = 0`.
    var tables = _one_wide_column()
    var cat = _four_row_fixture()
    var classes = List[Int]()
    classes.append(1)
    classes.append(1)
    classes.append(0)
    classes.append(1)
    var perm = _identity_permutation(4)
    var out = build_ctr_train_columns(tables, cat, 4, classes, perm)
    assert_equal(len(out), 4 * 4)

    # Column 0 is Borders at prior 0. Category 1 occupies rows 0 and 1:
    #   row 0: c = 0, t = 0 -> floor(0/1 * 15) = 0
    #   row 1: c = 1, t = 1 -> floor((1/2) * 15) = 7
    # Category 2 occupies rows 2 and 3:
    #   row 2: c = 0, t = 0 -> 0
    #   row 3: c = 0, t = 1 -> floor((0/2) * 15) = 0
    assert_equal(Int(out[0]), 0)
    assert_equal(Int(out[1]), 7)
    assert_equal(Int(out[2]), 0)
    assert_equal(Int(out[3]), 0)


def test_a_row_never_sees_its_own_target_through_the_column_builder() raises:
    # The leakage argument, restated one layer up from `test_ctr.mojo`'s
    # version: two label columns differing only in the LAST row must produce
    # identical ordered CTR columns, the changed row included.
    var tables = _one_wide_column()
    var cat = _four_row_fixture()
    var perm = _identity_permutation(4)

    var a = List[Int]()
    a.append(1)
    a.append(0)
    a.append(1)
    a.append(0)
    var b = List[Int]()
    b.append(1)
    b.append(0)
    b.append(1)
    b.append(1)

    var out_a = build_ctr_train_columns(tables, cat, 4, a, perm)
    var out_b = build_ctr_train_columns(tables, cat, 4, b, perm)
    assert_equal(len(out_a), len(out_b))
    for i in range(len(out_a)):
        assert_equal(out_a[i], out_b[i])


def test_the_counter_column_ignores_the_permutation() raises:
    # `IsPermutationDependentCtrType` returns false for Counter:
    # `CountOnlineCTRTotal` fills the table once over the whole array before any
    # row is emitted, so every row of a category gets the same value and the
    # column cannot move with the permutation.
    var tables = _one_wide_column()
    var cat = _four_row_fixture()
    var classes = List[Int]()
    classes.append(1)
    classes.append(0)
    classes.append(1)
    classes.append(0)

    var forward = _identity_permutation(4)
    var reversed = List[Int]()
    reversed.append(3)
    reversed.append(2)
    reversed.append(1)
    reversed.append(0)

    var out_f = build_ctr_train_columns(tables, cat, 4, classes, forward)
    var out_r = build_ctr_train_columns(tables, cat, 4, classes, reversed)

    # Column 3 is the Counter column.
    for r in range(4):
        assert_equal(out_f[3 * 4 + r], out_r[3 * 4 + r])
    # Both categories have count 2 and the denominator is the largest count, 2,
    # so every row gets `floor((2 + 0) / (2 + 1) * 15) = floor(10.0) = 10`.
    # This is the same 2/3 that `test_ctr.mojo` pins in Float32.
    for r in range(4):
        assert_equal(Int(out_f[3 * 4 + r]), 10)

    # And the Borders columns DO move with the permutation, which is what makes
    # the assertion above meaningful rather than vacuous.
    var moved = False
    for r in range(4):
        if out_f[r] != out_r[r]:
            moved = True
    assert_true(moved)


def test_the_static_table_is_a_full_count_not_a_prefix() raises:
    # `CalcFinalCtrsImpl` is a plain loop over every row, no permutation and no
    # prefix. So the inference bin for a category is the SAME for every row of
    # it, and in general differs from any of that category's training bins --
    # asserting that they agreed would be asserting the bug.
    var tables = _one_wide_column()
    var cat = _four_row_fixture()
    var classes = List[Int]()
    classes.append(1)
    classes.append(1)
    classes.append(0)
    classes.append(0)
    fit_ctr_tables(tables, cat, 4, classes)

    var predicted = ctr_predict_columns(tables, cat, 4)
    assert_equal(len(predicted), 4 * 4)
    # Column 0, category 1: full counts are `good = 2, total = 2`, so
    # `(2 + 0) / (2 + 1) * 15 = 10.0` and `ctr_predict_bucket(10.0, 15) = 10`.
    assert_equal(Int(predicted[0]), 10)
    assert_equal(Int(predicted[1]), 10)
    # Category 2: `good = 0, total = 2` -> `0 * 15 = 0`.
    assert_equal(Int(predicted[2]), 0)
    assert_equal(Int(predicted[3]), 0)

    # The training column for the same data is a prefix, so row 0 of category 1
    # is the pure prior (0) where inference says 10. Different by construction.
    var perm = _identity_permutation(4)
    var trained = build_ctr_train_columns(tables, cat, 4, classes, perm)
    assert_equal(Int(trained[0]), 0)
    assert_true(trained[0] != predicted[0])


def test_a_row_and_a_batch_score_the_same_bin() raises:
    # `ctr_predict_row` and `ctr_predict_columns` read the same lookup, so a row
    # scored singly and the same row scored in a batch must agree. A model whose
    # `predict` and `predict_batch` disagreed would be the silent failure this
    # lane's brief names.
    var tables = _one_wide_column()
    var cat = _four_row_fixture()
    var classes = List[Int]()
    classes.append(1)
    classes.append(0)
    classes.append(1)
    classes.append(1)
    fit_ctr_tables(tables, cat, 4, classes)
    var batch = ctr_predict_columns(tables, cat, 4)

    for r in range(4):
        # Base bins for a two-feature row: feature 0 numerical, feature 1 the
        # categorical column whose bucket this row holds.
        # A RAW row now: feature 0 numerical, feature 1 the categorical
        # column's raw code. `ctr_predict_row` maps it through the same
        # `bucket_of` the batch path uses, which is what makes the two agree.
        var raw = List[Float64]()
        raw.append(0.0)
        raw.append(Float64(cat[r]))
        var single = ctr_predict_row(tables, raw)
        assert_equal(len(single), tables.n_columns())
        for c in range(tables.n_columns()):
            assert_equal(single[c], Int(batch[c * 4 + r]))


def test_an_unseen_category_lands_in_bucket_zero_and_gets_the_prior() raises:
    # CatBoost's `bucket < 0` arm is unreachable here: every value maps to a
    # bucket, and an unknown one maps to `categorical.UNKNOWN_BIN = 0`. When no
    # training row landed in bucket 0 its counts are zero and the reader returns
    # the pure prior -- the same number CatBoost's absent-category arm returns,
    # by a different route.
    var tables = _one_wide_column()
    var cat = _four_row_fixture()  # only buckets 1 and 2 occur
    var classes = List[Int]()
    classes.append(1)
    classes.append(1)
    classes.append(1)
    classes.append(1)
    fit_ctr_tables(tables, cat, 4, classes)

    var raw = List[Float64]()
    raw.append(0.0)
    raw.append(99.0)  # a code absent from slot_codes, so bucket 0
    var bins = ctr_predict_row(tables, raw)
    # Column 0 is prior 0: `(0 + 0) / (0 + 1) * 15 = 0`.
    assert_equal(bins[0], 0)
    # Column 2 is prior 1: `(0 + 1) / (0 + 1) * 15 = 15`, the top bucket.
    assert_equal(bins[2], 15)


# ---------------------------------------------------------------------------
# The bucket rule that bridges the two formulas
# ---------------------------------------------------------------------------


def test_the_predict_bucket_agrees_with_emulate_ui8_rounding() raises:
    # `ctr_predict_bucket` is `EmulateUi8Rounding` solved for the bucket:
    # `bucket > border` must hold exactly when `x > border + 0.999999f`.
    for border in range(15):
        var threshold = ctr_predict_border(border)
        # Just above the threshold: the row goes right.
        assert_true(ctr_predict_bucket(threshold + 0.001, 15) > border)
        # Just below it: the row goes left.
        assert_false(ctr_predict_bucket(threshold - 0.001, 15) > border)
    # Clamped at both ends.
    assert_equal(ctr_predict_bucket(-5.0, 15), 0)
    assert_equal(ctr_predict_bucket(1000.0, 15), 15)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_the_permutation_does_not_depend_on_the_worker_count() raises:
    # `ctr_permutation` is a keyed block sort and nothing here consumes it in
    # an order-dependent way, so two calls with the same config and row count
    # must be the same list. This is the property the whole determinism claim
    # rests on; it cannot be tested against a real worker count from inside a
    # single-threaded test, so what is asserted is the weaker, real prerequisite
    # -- that the answer is a pure function of (seed, index, n_rows).
    var config = _binary_config()
    var a = ctr_train_permutation(config, 5000)
    var b = ctr_train_permutation(config, 5000)
    assert_equal(len(a), 5000)
    for i in range(len(a)):
        assert_equal(a[i], b[i])


def test_has_time_makes_the_permutation_the_identity() raises:
    # `IsPermutationNeeded`'s first line: `if (hasTime) { return false; }`. The
    # CTR degenerates to a prefix statistic in dataset order, which is the
    # correct behavior for a genuinely time-ordered pool.
    var config = _binary_config()
    config.has_time = True
    var p = ctr_train_permutation(config, 100)
    for i in range(100):
        assert_equal(p[i], i)


def test_the_same_config_and_data_produce_the_same_columns() raises:
    var tables = _one_wide_column()
    var cat = _four_row_fixture()
    var classes = List[Int]()
    classes.append(1)
    classes.append(0)
    classes.append(1)
    classes.append(0)
    var perm = _identity_permutation(4)
    var a = build_ctr_train_columns(tables, cat, 4, classes, perm)
    var b = build_ctr_train_columns(tables, cat, 4, classes, perm)
    for i in range(len(a)):
        assert_equal(a[i], b[i])


# ---------------------------------------------------------------------------
# The refusals, each by name
# ---------------------------------------------------------------------------


def test_combinations_are_still_refused() raises:
    # A30's guard survives this lane: `max_ctr_complexity > 1` needs the
    # candidate enumeration driven by a grow loop, which is `tree.mojo`'s and
    # not this one's.
    var borders = List[Float64]()
    borders.append(0.5)
    var config = SimpleCtrConfig.catboost_defaults(
        borders^, max_ctr_complexity=2
    )
    with assert_raises():
        config.validate()


def test_counter_calc_full_is_refused_by_name() raises:
    # `Full` counts the learn rows plus every test set. A `Dataset` holds learn
    # rows only, so the switch would have nothing to include and would be
    # silently equivalent to `SkipTest` -- accept-and-ignore, refused.
    var config = _binary_config()
    config.counter_calc_method = 1  # COUNTER_CALC_FULL
    with assert_raises():
        config.validate()


def test_fitted_tables_refuse_to_reach_a_writer() raises:
    # The tables are read off the target, so a model that lost them scores wrong
    # rather than failing. `serialize.mojo` has no ctr section and belongs to the
    # model-export lane (catalog A29); until it lands, this refuses.
    check_ctr_serializable(CtrTables.none())
    var tables = _one_wide_column()
    with assert_raises():
        check_ctr_serializable(tables)


def test_a_ctr_dataset_refuses_to_become_a_model() raises:
    # The refusal that actually holds in this branch. It is at the trainer
    # boundary rather than at the writer because this lane cannot install a call
    # in `serialize.mojo`; a model that is never produced is never saved wrong.
    check_ctr_model_support(False)
    with assert_raises():
        check_ctr_model_support(True)
    # A19's original bundle-shaped guard still refuses too.
    check_ctr_trainer_support(CtrParams.disabled())
    with assert_raises():
        check_ctr_trainer_support(CtrParams.enable())


def test_the_default_descriptions_are_borders_and_counter() raises:
    var descs = catboost_simple_ctr_defaults()
    assert_equal(len(descs), 2)
    assert_equal(descs[0].ctr_type, CTR_BORDERS)
    assert_equal(descs[0].n_priors(), 3)
    assert_equal(descs[1].ctr_type, CTR_COUNTER)
    assert_equal(descs[1].n_priors(), 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
