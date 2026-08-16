"""Ordered target statistics (src/mojotrees/ctr.mojo), catalog A17.

Every expectation here is analytical and was worked out by hand from CatBoost's
`CalcCTR` before it was run. The arithmetic under test, in numbers small enough
to check on paper at the default `ctr_border_count = 15` and prior 0, where
`shift = 0` and `norm = 1`:

    bin(c, t) = floor( (c / (t + 1)) * 15 )

    bin(0, 0) = 0      bin(1, 1) = 7      bin(2, 2) = 10     bin(2, 3) = 7
    bin(1, 2) = 5      bin(1, 3) = 3      bin(3, 3) = 11

The two thirds -- `bin(2, 2)` and `bin(1, 2)` -- are the only cases here whose
answer is not exact in binary, and they are in the suite on purpose: `2/3` in
Float32 is 0.66666668653, times 15 is 10.000000298, and the nearest Float32 to
that is exactly 10.0, so the truncation lands on 10 and not on 9. If someone
moves the arithmetic to Float64 or reorders the multiply those two assertions
are what notices.

Three properties get their own tests because they are the mechanism rather than
the arithmetic.

`test_a_row_never_sees_its_own_target` is the leakage argument written as an
assertion. Two datasets that differ only in the LAST row's class must produce
identical CTR columns, including for the row that changed. The single edit that
breaks it -- swapping the read and the write in the online loop -- is the edit
this file exists to catch.

`test_block_order_does_not_depend_on_how_many_blocks_there_are` is the
determinism claim. A block's key is `splitmix64(stream + b)` and nothing
advances, so blocks 0 to 9 must sort the same way among themselves whether the
permutation has ten blocks or twenty. A running RNG would fail it, which is the
point: the claim is that the permutation reproduces across
`MOJOTREES_NUM_WORKERS` and across machines, and a keyed sort is how that is
bought.

`test_training_bin_and_inference_value_agree` pins the train-versus-predict
asymmetry. The two formulas are different, over different data, with different
result types, and the only thing that ties them together is
`Scale = borderCount / norm` plus `EmulateUi8Rounding`. Given the same counts
they must land in the same bucket.
"""

from std.math import floor
from std.testing import (
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
    TestSuite,
)

from mojotrees.ctr import (
    COUNTER_CALC_FULL,
    COUNTER_CALC_SKIP_TEST,
    CPU_PRIOR_DENOM,
    CTR_BINARIZED_TARGET_MEAN,
    CTR_BORDERS,
    CTR_BUCKETS,
    CTR_COUNTER,
    CtrParams,
    DEFAULT_CTR_BORDER_COUNT,
    DEFAULT_MAX_CTR_COMPLEXITY,
    DEFAULT_PERMUTATION_COUNT,
    PERMUTATION_BLOCK_SIZE_NOT_SET,
    calc_normalization,
    check_ctr_border_type,
    check_ctr_complexity,
    check_ctr_prior_denom,
    check_ctr_trainer_support,
    counter_ctr,
    counter_denominator,
    ctr_bucket_count,
    ctr_combination_hash,
    ctr_feature_count,
    ctr_fold_index,
    ctr_norm,
    ctr_permutation,
    ctr_predict_border,
    ctr_predict_scale,
    ctr_predict_value,
    ctr_projection_hash,
    ctr_shift,
    ctr_target_border_count,
    ctr_train_bin,
    default_permutation_block_size,
    default_priors,
    final_ctr_class_table,
    final_ctr_counter_table,
    final_ctr_mean_table,
    learning_fold_count,
    ordered_ctr_borders_binary,
    ordered_ctr_classes,
    ordered_ctr_mean,
    permutation_is_needed,
    predict_ctr_class,
    predict_ctr_counter,
    predict_ctr_mean,
    resolve_permutation_block_size,
    target_class,
    target_classes_count,
)


def _close(a: Float64, b: Float64) -> Bool:
    return abs(a - b) < 1e-5


def _identity(n: Int) -> List[Int]:
    var p = List[Int]()
    for i in range(n):
        p.append(i)
    return p^


def _reversed(n: Int) -> List[Int]:
    var p = List[Int]()
    for i in range(n):
        p.append(n - 1 - i)
    return p^


def _assert_ints_equal(got: List[Int], want: List[Int]) raises:
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        assert_equal(got[i], want[i])


# --- The arithmetic --------------------------------------------------------


def test_train_bin_is_calc_ctr() raises:
    # `CalcCTR`: (c + prior) / (t + 1), normalized, times borderCount,
    # truncated. At prior 0 the normalization is the identity.
    assert_equal(ctr_train_bin(0.0, 0, 0.0, 0.0, 1.0, 15), 0)
    assert_equal(ctr_train_bin(1.0, 1, 0.0, 0.0, 1.0, 15), 7)
    assert_equal(ctr_train_bin(1.0, 3, 0.0, 0.0, 1.0, 15), 3)
    assert_equal(ctr_train_bin(3.0, 3, 0.0, 0.0, 1.0, 15), 11)
    # The two thirds. See the module docstring for why these are here.
    assert_equal(ctr_train_bin(2.0, 2, 0.0, 0.0, 1.0, 15), 10)
    assert_equal(ctr_train_bin(1.0, 2, 0.0, 0.0, 1.0, 15), 5)


def test_the_denominator_is_total_plus_one_not_total() raises:
    # The single most-copied error in a reimplementation. With c = t = 1 the
    # "+ 1" is the difference between 0.5 and 1.0, which is seven buckets.
    assert_equal(ctr_train_bin(1.0, 1, 0.0, 0.0, 1.0, 15), 7)
    assert_true(ctr_train_bin(1.0, 1, 0.0, 0.0, 1.0, 15) != 15)


def test_the_top_bucket_is_reachable_so_there_are_border_count_plus_one() raises:
    # prior 1 with c == t gives (t + 1) / (t + 1) = 1 exactly, which scales to
    # borderCount and truncates to borderCount. That is why `GetBucketCount`
    # returns BorderCount + 1.
    assert_equal(ctr_train_bin(1.0, 1, 1.0, 0.0, 1.0, 15), 15)
    assert_equal(ctr_bucket_count(15), 16)


def test_normalization_is_the_identity_at_every_default_prior() raises:
    var ps: List[Float64] = [0.0, 0.5, 1.0]
    for i in range(len(ps)):
        assert_equal(ctr_shift(ps[i]), 0.0)
        assert_equal(ctr_norm(ps[i]), 1.0)


def test_normalization_maps_an_out_of_range_prior_back_in() raises:
    # shift = -min(0, prior); norm = max(1, prior) - min(0, prior).
    assert_equal(ctr_shift(-1.0), 1.0)
    assert_equal(ctr_norm(-1.0), 2.0)
    assert_equal(ctr_shift(2.0), 0.0)
    assert_equal(ctr_norm(2.0), 2.0)
    var priors: List[Float64] = [-1.0, 0.5, 2.0]
    var shifts = List[Float64]()
    var norms = List[Float64]()
    calc_normalization(priors, shifts, norms)
    assert_equal(len(shifts), 3)
    assert_equal(shifts[0], 1.0)
    assert_equal(norms[0], 2.0)
    assert_equal(shifts[1], 0.0)
    assert_equal(norms[1], 1.0)
    assert_equal(shifts[2], 0.0)
    assert_equal(norms[2], 2.0)
    # And the normalized value still lands inside the bucket range.
    var b = ctr_train_bin(0.0, 0, -1.0, 1.0, 2.0, 15)
    assert_true(b >= 0 and b <= 15)


def test_train_bin_refuses_nonsense() raises:
    with assert_raises():
        _ = ctr_train_bin(0.0, 0, 0.0, 0.0, 1.0, 0)
    with assert_raises():
        _ = ctr_train_bin(0.0, -1, 0.0, 0.0, 1.0, 15)
    with assert_raises():
        _ = ctr_train_bin(0.0, 0, 0.0, 0.0, 0.0, 15)


# --- Train versus predict --------------------------------------------------


def test_training_bin_and_inference_value_agree() raises:
    # Scale = borderCount / norm is the whole reconciliation. Given the same
    # counts the unquantized inference value must truncate to the training
    # bucket.
    var scale = ctr_predict_scale(15, 1.0)
    assert_equal(scale, 15.0)
    var cs: List[Int] = [0, 1, 2, 1, 3, 2]
    var ts: List[Int] = [0, 1, 2, 2, 3, 3]
    for i in range(len(cs)):
        var want_bin = ctr_train_bin(
            Float64(cs[i]), ts[i], 0.0, 0.0, 1.0, 15
        )
        var v = ctr_predict_value(
            Float64(cs[i]), Float64(ts[i]), 0.0, CPU_PRIOR_DENOM, 0.0, scale
        )
        assert_equal(Int(floor(v)), want_bin)


def test_inference_border_is_the_ui8_rounding_emulation() raises:
    # split.h: EmulateUi8Rounding(value) { return value + 0.999999f; }
    assert_true(_close(ctr_predict_border(7), 7.999999))
    assert_true(_close(ctr_predict_border(0), 0.999999))
    # A value that trained into bucket 8 must clear the bucket-7 threshold, and
    # a value that trained into bucket 7 must not.
    var scale = ctr_predict_scale(15, 1.0)
    var v8 = ctr_predict_value(3.0, 4.0, 0.0, 1.0, 0.0, scale)
    assert_equal(ctr_train_bin(3.0, 4, 0.0, 0.0, 1.0, 15), 9)
    assert_true(v8 > ctr_predict_border(7))
    var v7 = ctr_predict_value(1.0, 1.0, 0.0, 1.0, 0.0, scale)
    assert_equal(ctr_train_bin(1.0, 1, 0.0, 0.0, 1.0, 15), 7)
    assert_false(v7 > ctr_predict_border(7))


def test_an_unseen_category_gets_the_pure_prior() raises:
    # ctr_calcer.py: `if bucket is None: result = ctr.calc(0, 0)`. Training
    # cannot reach this case; inference must.
    var scale = ctr_predict_scale(15, 1.0)
    var v = ctr_predict_value(0.0, 0.0, 0.5, CPU_PRIOR_DENOM, 0.0, scale)
    assert_true(_close(v, 7.5))
    var table = List[Int]()
    table.resize(4, 0)
    var unseen = predict_ctr_class(
        table, 2, -1, CTR_BORDERS, 0, 0.5, CPU_PRIOR_DENOM, 0.0, scale
    )
    assert_true(_close(unseen, 7.5))


# --- Permutations ----------------------------------------------------------


def test_default_permutation_block_size() raises:
    # min(256, docCount / 1000 + 1)
    assert_equal(default_permutation_block_size(0), 1)
    assert_equal(default_permutation_block_size(999), 1)
    assert_equal(default_permutation_block_size(1000), 2)
    assert_equal(default_permutation_block_size(254999), 255)
    assert_equal(default_permutation_block_size(255000), 256)
    assert_equal(default_permutation_block_size(1000000), 256)


def test_resolve_permutation_block_size() raises:
    assert_equal(
        resolve_permutation_block_size(
            PERMUTATION_BLOCK_SIZE_NOT_SET, 10000, True
        ),
        11,
    )
    # An unpermuted fold gets the whole row count, CatBoost's way of saying the
    # permutation is the identity.
    assert_equal(
        resolve_permutation_block_size(
            PERMUTATION_BLOCK_SIZE_NOT_SET, 10000, False
        ),
        10000,
    )
    assert_equal(resolve_permutation_block_size(64, 10000, True), 64)


def test_permutation_zero_is_the_identity() raises:
    var p = ctr_permutation(10, 4, 7, 0)
    _assert_ints_equal(p, _identity(10))


def test_permutation_is_a_bijection() raises:
    var n = 37
    for idx in range(1, 5):
        var p = ctr_permutation(n, 5, 11, idx)
        assert_equal(len(p), n)
        var seen = List[Bool]()
        seen.resize(n, False)
        for i in range(n):
            assert_true(p[i] >= 0 and p[i] < n)
            assert_false(seen[p[i]])
            seen[p[i]] = True


def test_permutation_keeps_rows_within_a_block_in_order() raises:
    # NCB::Shuffle reorders blocks and emits each block's rows in their original
    # relative order. So consecutive positions inside one block ascend by one,
    # and a row's ordered prefix is NOT a uniform random subset.
    var block = 4
    var n = 20
    var p = ctr_permutation(n, block, 3, 1)
    var pos = 0
    while pos < n:
        var start = p[pos]
        assert_equal(start % block, 0)
        var k = 1
        while k < block and pos + k < n:
            assert_equal(p[pos + k], start + k)
            k += 1
        pos += k


def test_permutation_is_a_pure_function_of_its_arguments() raises:
    var a = ctr_permutation(64, 8, 5, 2)
    var b = ctr_permutation(64, 8, 5, 2)
    _assert_ints_equal(a, b)


def test_a_different_seed_or_index_gives_a_different_permutation() raises:
    var a = ctr_permutation(64, 8, 5, 1)
    var b = ctr_permutation(64, 8, 6, 1)
    var c = ctr_permutation(64, 8, 5, 2)
    var differs_by_seed = False
    var differs_by_index = False
    for i in range(64):
        if a[i] != b[i]:
            differs_by_seed = True
        if a[i] != c[i]:
            differs_by_index = True
    assert_true(differs_by_seed)
    assert_true(differs_by_index)


def test_block_order_does_not_depend_on_how_many_blocks_there_are() raises:
    # THE determinism test. Block b's key is splitmix64(stream + b) and nothing
    # advances, so blocks 0..9 must sort the same way among themselves whether
    # the permutation covers ten blocks or twenty. A sequential Fisher-Yates
    # would fail this, which is exactly the property being bought.
    var block = 4
    var small = ctr_permutation(10 * block, block, 21, 1)
    var large = ctr_permutation(20 * block, block, 21, 1)

    var small_order = List[Int]()
    var pos = 0
    while pos < len(small):
        small_order.append(small[pos] // block)
        pos += block

    var large_order = List[Int]()
    pos = 0
    while pos < len(large):
        var b = large[pos] // block
        if b < 10:
            large_order.append(b)
        pos += block

    _assert_ints_equal(large_order, small_order)


def test_fold_selection_is_keyed_by_tree_not_by_a_running_stream() raises:
    for t in range(20):
        var f = ctr_fold_index(9, t, 3)
        assert_true(f >= 0 and f < 3)
        assert_equal(ctr_fold_index(9, t, 3), f)
    with assert_raises():
        _ = ctr_fold_index(9, 0, 0)


def test_learning_fold_count_is_permutation_count_minus_one() raises:
    # CountLearningFolds. At the default 4 there are THREE learning folds; the
    # fourth permutation is spent on the separate AveragingFold.
    assert_equal(learning_fold_count(4, True), 3)
    assert_equal(learning_fold_count(1, True), 1)
    assert_equal(learning_fold_count(2, True), 1)
    assert_equal(learning_fold_count(4, False), 1)
    with assert_raises():
        _ = learning_fold_count(0, True)


def test_has_time_disables_the_permutation_entirely() raises:
    # IsPermutationNeeded's first line. A time-ordered pool gets a prefix
    # statistic in dataset order, whatever else is set.
    assert_false(permutation_is_needed(True, True, True))
    # A wide categorical column turns the permutation on even for plain
    # boosting: CatBoost's `hasCtrs` is a property of the DATA, not a request.
    assert_true(permutation_is_needed(False, True, False))
    assert_false(permutation_is_needed(False, False, False))
    assert_true(permutation_is_needed(False, False, True))


# --- The target classifier -------------------------------------------------


def test_target_class_is_a_strict_greater_than_walk() raises:
    var borders: List[Float64] = [0.5]
    assert_equal(target_class(0.0, borders), 0)
    assert_equal(target_class(0.5, borders), 0)
    assert_equal(target_class(0.51, borders), 1)
    assert_equal(target_class(1.0, borders), 1)
    assert_equal(target_classes_count(borders), 2)
    var three: List[Float64] = [0.25, 0.75]
    assert_equal(target_class(0.1, three), 0)
    assert_equal(target_class(0.5, three), 1)
    assert_equal(target_class(0.9, three), 2)
    assert_equal(target_classes_count(three), 3)


def test_feature_counts_and_the_four_columns_per_categorical() raises:
    assert_equal(ctr_target_border_count(CTR_BORDERS, 2), 1)
    assert_equal(ctr_target_border_count(CTR_BORDERS, 3), 2)
    assert_equal(ctr_target_border_count(CTR_BUCKETS, 3), 3)
    assert_equal(ctr_target_border_count(CTR_BINARIZED_TARGET_MEAN, 5), 1)
    assert_equal(ctr_target_border_count(CTR_COUNTER, 5), 1)
    # The stock default: Borders x 3 priors plus Counter x 1 prior, so four
    # numeric columns per categorical column, each in 16 buckets.
    var borders_cols = ctr_feature_count(CTR_BORDERS, 2, 3)
    var counter_cols = ctr_feature_count(CTR_COUNTER, 2, 1)
    assert_equal(borders_cols, 3)
    assert_equal(counter_cols, 1)
    assert_equal(borders_cols + counter_cols, 4)


def test_default_priors() raises:
    var target = default_priors(CTR_BORDERS)
    assert_equal(len(target), 3)
    assert_equal(target[0], 0.0)
    assert_equal(target[1], 0.5)
    assert_equal(target[2], 1.0)
    var counter = default_priors(CTR_COUNTER)
    assert_equal(len(counter), 1)
    assert_equal(counter[0], 0.0)


# --- The ordered loops -----------------------------------------------------


def test_ordered_borders_binary_on_hand_arithmetic() raises:
    # bucket 0 sees classes 1, 1, 0, 1 in order; bucket 1 sees 0, 1.
    var categories: List[Int] = [0, 0, 0, 0, 1, 1]
    var classes: List[Int] = [1, 1, 0, 1, 0, 1]
    var out = List[Int]()
    ordered_ctr_borders_binary(
        categories, classes, _identity(6), 2, 0.0, 15, out
    )
    #  r0: c=0 t=0 -> 0        r3: c=2 t=3 -> 7
    #  r1: c=1 t=1 -> 7        r4: c=0 t=0 -> 0
    #  r2: c=2 t=2 -> 10       r5: c=0 t=1 -> 0
    var want: List[Int] = [0, 7, 10, 7, 0, 0]
    _assert_ints_equal(out, want)


def test_the_first_row_of_a_category_gets_the_pure_prior() raises:
    var categories: List[Int] = [0]
    var classes: List[Int] = [1]
    var out = List[Int]()
    # prior 0 -> 0/1 -> bucket 0. prior 1 -> 1/1 -> bucket 15.
    ordered_ctr_borders_binary(
        categories, classes, _identity(1), 1, 0.0, 15, out
    )
    assert_equal(out[0], 0)
    ordered_ctr_borders_binary(
        categories, classes, _identity(1), 1, 1.0, 15, out
    )
    assert_equal(out[0], 15)


def test_a_row_never_sees_its_own_target() raises:
    # The leakage argument as an assertion. Two datasets that differ only in the
    # LAST row's class must produce identical columns -- including for the row
    # that changed, whose value depends on its prefix alone. Swapping the read
    # and the write in the online loop is the edit this catches.
    var categories: List[Int] = [0, 0, 0, 0, 0]
    var a: List[Int] = [1, 0, 1, 1, 0]
    var b: List[Int] = [1, 0, 1, 1, 1]
    var out_a = List[Int]()
    var out_b = List[Int]()
    ordered_ctr_borders_binary(categories, a, _identity(5), 1, 0.0, 15, out_a)
    ordered_ctr_borders_binary(categories, b, _identity(5), 1, 0.0, 15, out_b)
    _assert_ints_equal(out_a, out_b)


def test_the_permutation_changes_the_answer() raises:
    var categories: List[Int] = [0, 0, 0, 0]
    var classes: List[Int] = [1, 1, 0, 0]
    var forward = List[Int]()
    var backward = List[Int]()
    ordered_ctr_borders_binary(
        categories, classes, _identity(4), 1, 0.0, 15, forward
    )
    ordered_ctr_borders_binary(
        categories, classes, _reversed(4), 1, 0.0, 15, backward
    )
    var differs = False
    for i in range(4):
        if forward[i] != backward[i]:
            differs = True
    assert_true(differs)


def test_ordered_loops_refuse_a_bad_permutation() raises:
    var categories: List[Int] = [0, 0]
    var classes: List[Int] = [0, 1]
    var out = List[Int]()
    var not_a_bijection: List[Int] = [0, 0]
    with assert_raises():
        ordered_ctr_borders_binary(
            categories, classes, not_a_bijection, 1, 0.0, 15, out
        )
    var wrong_length: List[Int] = [0]
    with assert_raises():
        ordered_ctr_borders_binary(
            categories, classes, wrong_length, 1, 0.0, 15, out
        )
    var out_of_range: List[Int] = [0, 5]
    with assert_raises():
        ordered_ctr_borders_binary(
            out_of_range, classes, _identity(2), 1, 0.0, 15, out
        )


def test_borders_above_two_classes_counts_the_classes_above_the_border() raises:
    # classes 0, 1, 2, 2 in one bucket. Borders at index k counts classes > k.
    var categories: List[Int] = [0, 0, 0, 0]
    var classes: List[Int] = [0, 1, 2, 2]
    var out = List[Int]()
    ordered_ctr_classes(
        categories, classes, _identity(4), 1, 3, CTR_BORDERS, 0, 0.0, 15, out
    )
    #  r0: good 0 t 0 -> 0     r2: good 1 t 2 -> 5
    #  r1: good 0 t 1 -> 0     r3: good 2 t 3 -> 7
    var want0: List[Int] = [0, 0, 5, 7]
    _assert_ints_equal(out, want0)
    ordered_ctr_classes(
        categories, classes, _identity(4), 1, 3, CTR_BORDERS, 1, 0.0, 15, out
    )
    #  only class 2 is above border 1, and it first appears at r2
    var want1: List[Int] = [0, 0, 0, 3]
    _assert_ints_equal(out, want1)


def test_buckets_counts_only_its_own_class() raises:
    var categories: List[Int] = [0, 0, 0, 0]
    var classes: List[Int] = [0, 1, 2, 2]
    var out = List[Int]()
    ordered_ctr_classes(
        categories, classes, _identity(4), 1, 3, CTR_BUCKETS, 0, 0.0, 15, out
    )
    #  class 0 appears once, at r0
    #  r0: good 0 t 0 -> 0     r2: good 1 t 2 -> 5
    #  r1: good 1 t 1 -> 7     r3: good 1 t 3 -> 3
    var want0: List[Int] = [0, 7, 5, 3]
    _assert_ints_equal(out, want0)
    ordered_ctr_classes(
        categories, classes, _identity(4), 1, 3, CTR_BUCKETS, 2, 0.0, 15, out
    )
    var want2: List[Int] = [0, 0, 0, 3]
    _assert_ints_equal(out, want2)


def test_buckets_has_one_more_border_than_borders() raises:
    var categories: List[Int] = [0, 0]
    var classes: List[Int] = [0, 1]
    var out = List[Int]()
    # Borders over 2 classes has one border, so index 1 is out of range.
    with assert_raises():
        ordered_ctr_classes(
            categories,
            classes,
            _identity(2),
            1,
            2,
            CTR_BORDERS,
            1,
            0.0,
            15,
            out,
        )
    # Buckets over 2 classes has two, so index 1 is fine.
    ordered_ctr_classes(
        categories, classes, _identity(2), 1, 2, CTR_BUCKETS, 1, 0.0, 15, out
    )
    assert_equal(len(out), 2)


def test_binary_borders_agrees_with_the_general_class_loop() raises:
    # CalcOnlineCTRSimple is CalcOnlineCTRClasses specialized to two classes.
    # They must not disagree.
    var categories: List[Int] = [0, 1, 0, 1, 0, 0]
    var classes: List[Int] = [1, 0, 0, 1, 1, 1]
    var simple = List[Int]()
    var general = List[Int]()
    ordered_ctr_borders_binary(
        categories, classes, _identity(6), 2, 0.5, 15, simple
    )
    ordered_ctr_classes(
        categories, classes, _identity(6), 2, 2, CTR_BORDERS, 0, 0.5, 15, general
    )
    _assert_ints_equal(simple, general)


def test_binarized_target_mean_runs_a_running_mean() raises:
    # Two classes, so targetBorderCount is 1 and the increment is the class
    # index itself.
    var categories: List[Int] = [0, 0, 0]
    var classes: List[Int] = [1, 1, 0]
    var out = List[Int]()
    ordered_ctr_mean(categories, classes, _identity(3), 1, 2, 0.0, 15, out)
    #  r0: sum 0 count 0 -> 0
    #  r1: sum 1 count 1 -> 7
    #  r2: sum 2 count 2 -> 10
    var want: List[Int] = [0, 7, 10]
    _assert_ints_equal(out, want)


def test_binarized_target_mean_normalizes_the_class_index() raises:
    # Three classes, so the increment is classIndex / 2 and a class-2 row adds
    # 1.0 where a class-1 row adds 0.5.
    var categories: List[Int] = [0, 0, 0]
    var classes: List[Int] = [2, 1, 0]
    var out = List[Int]()
    ordered_ctr_mean(categories, classes, _identity(3), 1, 3, 0.0, 15, out)
    #  r0: sum 0.0 count 0 -> 0
    #  r1: sum 1.0 count 1 -> 1/2 -> 7
    #  r2: sum 1.5 count 2 -> 1.5/3 = 0.5 -> 7
    var want: List[Int] = [0, 7, 7]
    _assert_ints_equal(out, want)


# --- Counter, which is not ordered ----------------------------------------


def test_counter_is_a_full_count_over_a_max_denominator() raises:
    # Every row of a category gets the same value, and the denominator is the
    # LARGEST category's count, not the row count.
    var categories: List[Int] = [0, 0, 0, 1, 1]
    var out = List[Int]()
    counter_ctr(categories, 2, 5, 0.0, 15, out)
    #  bucket 0: 3 / (3 + 1) = 0.75 -> 11
    #  bucket 1: 2 / (3 + 1) = 0.50 -> 7
    var want: List[Int] = [11, 11, 11, 7, 7]
    _assert_ints_equal(out, want)


def test_counter_calc_method_is_a_transduction_switch() raises:
    # SkipTest counts the learn rows only; Full counts learn plus test. The
    # test rows still receive a value either way.
    var categories: List[Int] = [0, 0, 0, 1, 1]
    var skip = List[Int]()
    var full = List[Int]()
    counter_ctr(categories, 2, 3, 0.0, 15, skip)
    counter_ctr(categories, 2, 5, 0.0, 15, full)
    var want_skip: List[Int] = [11, 11, 11, 0, 0]
    _assert_ints_equal(skip, want_skip)
    var want_full: List[Int] = [11, 11, 11, 7, 7]
    _assert_ints_equal(full, want_full)
    assert_true(COUNTER_CALC_SKIP_TEST != COUNTER_CALC_FULL)


def test_counter_denominator_is_the_max() raises:
    var counts: List[Int] = [3, 7, 2]
    assert_equal(counter_denominator(counts), 7)
    var empty = List[Int]()
    assert_equal(counter_denominator(empty), 0)


# --- The inference half ----------------------------------------------------


def test_the_final_table_is_not_a_prefix() raises:
    # CalcFinalCtrsImpl is a plain loop over every row with no permutation.
    # This is the whole difference between what training sees and what
    # inference sees.
    var categories: List[Int] = [0, 0, 0, 1]
    var classes: List[Int] = [1, 0, 1, 1]
    var table = List[Int]()
    final_ctr_class_table(categories, classes, 2, 2, 4, table)
    #  bucket 0: one class-0 row, two class-1 rows
    #  bucket 1: zero class-0 rows, one class-1 row
    assert_equal(table[0], 1)
    assert_equal(table[1], 2)
    assert_equal(table[2], 0)
    assert_equal(table[3], 1)


def test_predict_class_reads_borders_and_buckets_the_way_train_wrote_them() raises:
    var categories: List[Int] = [0, 0, 0, 0]
    var classes: List[Int] = [0, 1, 2, 2]
    var table = List[Int]()
    final_ctr_class_table(categories, classes, 1, 3, 4, table)
    var scale = ctr_predict_scale(15, 1.0)
    # Borders at index 0: classes above 0 are 1 and 2, so good = 3, total = 4.
    var b0 = predict_ctr_class(
        table, 3, 0, CTR_BORDERS, 0, 0.0, CPU_PRIOR_DENOM, 0.0, scale
    )
    assert_equal(Int(floor(b0)), ctr_train_bin(3.0, 4, 0.0, 0.0, 1.0, 15))
    # Buckets at index 2: good = 2, total = 4.
    var k2 = predict_ctr_class(
        table, 3, 0, CTR_BUCKETS, 2, 0.0, CPU_PRIOR_DENOM, 0.0, scale
    )
    assert_equal(Int(floor(k2)), ctr_train_bin(2.0, 4, 0.0, 0.0, 1.0, 15))


def test_predict_mean_and_counter_read_their_own_tables() raises:
    var categories: List[Int] = [0, 0, 1]
    var classes: List[Int] = [1, 0, 1]
    var sums = List[Float64]()
    var counts = List[Int]()
    final_ctr_mean_table(categories, classes, 2, 2, 3, sums, counts)
    assert_equal(sums[0], 1.0)
    assert_equal(counts[0], 2)
    assert_equal(sums[1], 1.0)
    assert_equal(counts[1], 1)
    var scale = ctr_predict_scale(15, 1.0)
    var m = predict_ctr_mean(
        sums, counts, 0, 0.0, CPU_PRIOR_DENOM, 0.0, scale
    )
    assert_equal(Int(floor(m)), ctr_train_bin(1.0, 2, 0.0, 0.0, 1.0, 15))

    var ccounts = List[Int]()
    final_ctr_counter_table(categories, 2, 3, ccounts)
    assert_equal(ccounts[0], 2)
    assert_equal(ccounts[1], 1)
    var denom = counter_denominator(ccounts)
    assert_equal(denom, 2)
    var c = predict_ctr_counter(
        ccounts, denom, 0, 0.0, CPU_PRIOR_DENOM, 0.0, scale
    )
    assert_equal(Int(floor(c)), ctr_train_bin(2.0, 2, 0.0, 0.0, 1.0, 15))


# --- Projection hashing, for the combinations lane -------------------------


def test_combination_hash_is_calc_hash() raises:
    # MAGIC_MULT * (a + MAGIC_MULT * b), with unsigned wraparound.
    var magic = UInt64(0x4906BA494954CB65)
    assert_equal(ctr_combination_hash(UInt64(0), UInt64(0)), UInt64(0))
    assert_equal(ctr_combination_hash(UInt64(0), UInt64(1)), magic * magic)
    assert_equal(ctr_combination_hash(UInt64(1), UInt64(0)), magic)


def test_projection_hash_folds_in_order_from_zero() raises:
    # A one-element projection is calc_hash(0, v), NOT v -- worth knowing before
    # someone shortcuts the single-column case.
    var one: List[UInt64] = [UInt64(5)]
    assert_equal(
        ctr_projection_hash(one), ctr_combination_hash(UInt64(0), UInt64(5))
    )
    assert_true(ctr_projection_hash(one) != UInt64(5))
    var ab: List[UInt64] = [UInt64(3), UInt64(9)]
    var ba: List[UInt64] = [UInt64(9), UInt64(3)]
    assert_true(ctr_projection_hash(ab) != ctr_projection_hash(ba))
    var empty = List[UInt64]()
    assert_equal(ctr_projection_hash(empty), UInt64(0))


# --- Parameters and guards -------------------------------------------------


def test_the_default_bundle_is_off_and_carries_catboost_numbers() raises:
    var p = CtrParams.disabled()
    assert_false(p.enabled)
    assert_false(p.is_active())
    assert_equal(p.ctr_type, CTR_BORDERS)
    assert_equal(p.ctr_border_count, DEFAULT_CTR_BORDER_COUNT)
    assert_equal(p.ctr_border_count, 15)
    assert_equal(p.permutation_count, DEFAULT_PERMUTATION_COUNT)
    assert_equal(p.permutation_count, 4)
    assert_equal(p.counter_calc_method, COUNTER_CALC_SKIP_TEST)
    assert_equal(p.max_ctr_complexity, DEFAULT_MAX_CTR_COMPLEXITY)
    assert_equal(p.max_ctr_complexity, 4)
    assert_equal(p.prior_denom, CPU_PRIOR_DENOM)
    # A disabled bundle validates even though its max_ctr_complexity of 4 would
    # be refused if it were on. Nothing is checked that nothing will read.
    p.validate()


def test_an_enabled_bundle_takes_the_default_priors() raises:
    var p = CtrParams.enable()
    assert_true(p.enabled)
    assert_equal(p.n_priors(), 3)
    assert_true(p.is_permutation_dependent())
    assert_true(p.needs_target_classifier())
    var c = CtrParams.enable(CTR_COUNTER)
    assert_equal(c.n_priors(), 1)
    # Counter is not permutation dependent and needs no target classifier.
    assert_false(c.is_permutation_dependent())
    assert_false(c.needs_target_classifier())
    # It is enabled with the default complexity of 4, which validate refuses.
    with assert_raises():
        p.validate()


def test_prior_denom_other_than_one_is_refused_by_name() raises:
    check_ctr_prior_denom(1.0)
    with assert_raises():
        check_ctr_prior_denom(2.0)
    with assert_raises():
        check_ctr_prior_denom(0.0)


def test_non_uniform_ctr_binarization_is_refused_by_name() raises:
    check_ctr_border_type("Uniform")
    with assert_raises():
        check_ctr_border_type("MinEntropy")
    with assert_raises():
        check_ctr_border_type("Median")


def test_combinations_are_refused_by_name() raises:
    check_ctr_complexity(1)
    with assert_raises():
        check_ctr_complexity(4)
    with assert_raises():
        check_ctr_complexity(0)


def test_an_enabled_bundle_is_refused_at_a_trainer_boundary() raises:
    # Nothing appends CTR columns to a design matrix yet. Deleting this guard is
    # the first step of the wiring lane, not a side effect of it.
    check_ctr_trainer_support(CtrParams.disabled())
    with assert_raises():
        check_ctr_trainer_support(CtrParams.enable())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
