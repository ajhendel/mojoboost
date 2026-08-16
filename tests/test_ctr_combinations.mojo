"""CTR feature combinations (src/mojotrees/ctr_combinations.mojo), catalog A22.

Every expectation here was worked out from CatBoost source before it was run,
and the hash constants were produced by running `calc_hash` from
`catboost/libs/model/model_export/resources/ctr_calcer.py` -- CatBoost's own
reference implementation, in Python, against the same inputs:

    MAGIC_MULT = 0x4906ba494954cb65
    calc_hash(a, b) = (MAGIC_MULT * ((a + MAGIC_MULT * b) & M64)) & M64

    calc_hash(0, 0) = 0
    calc_hash(0, 1) = 4731097034409465305
    calc_hash(0, 2) = 9462194068818930610
    calc_hash(calc_hash(0, 1), 0) = 8714454114147103133
    calc_hash(calc_hash(0, 1), 1) = 13445551148556568438

`calc_hash(0, 0) = 0` is worth pausing on, because it is why the training fold
adds one to the categorical bin: without the `+ 1`, bin 0 would be the fold's
identity element and a projection could not tell "category 0" from "no category
folded yet".

Four tests are the mechanism rather than the arithmetic and are named for it.

`test_a_binarized_member_folds_a_bit_and_not_a_value` is the piece A19 flagged
as most likely to be missed. Two rows whose float bins are 13 and 200 must get
the SAME projection hash against a split at bin 12, because both are on the
same side of it. An implementation that folds the bin instead of the test
result passes every other test in this file and fails this one.

`test_the_bin_and_one_hot_blob_counts_as_one_toward_complexity` pins
`GetFullProjectionLength`. Getting it wrong in either direction changes which
combinations are reachable at `max_ctr_complexity = 4`.

`test_the_kept_set_does_not_depend_on_row_order` is the determinism claim for
the top-K reindex. CatBoost's `std::nth_element` reads only the frequency and
is not stable, so its answer at a frequency tie depends on the standard
library; ours orders by `(count descending, hash ascending)`, which is a strict
total order because the hashes are distinct.

`test_the_overflow_bucket_is_the_last_kept_bucket` reproduces the discrepancy
between `index_hash_calcer.h:42` (which says the overflow goes to a fresh
bucket) and `index_hash_calcer.cpp:222` (which sends it to `Size() - 1`, the
last kept one). The code wins. This test exists so nobody "fixes" it.
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
    check_ctr_complexity,
    ctr_combination_hash,
    default_priors,
)
from mojotrees.ctr_combinations import (
    BinSplit,
    CTR_LEAF_COUNT_UNLIMITED,
    CtrRouting,
    MAX_CTR_COMPLEXITY_LIMIT,
    OneHotSplit,
    Projection,
    SMALL_ITERATION_COUNT,
    cat_value_final,
    cat_value_train,
    check_ctr_combination_trainer_support,
    check_max_ctr_complexity,
    compute_reindex_hash,
    ctr_candidates_per_projection,
    ctr_info_for_projection,
    ctr_projection_count_bound,
    ctr_routing_warning,
    grow_tree_ctr_projections,
    projection_bucket_space_bound,
    projection_hashes,
    projection_row_hash,
    resolve_ctr_leaf_count_limit,
    resolve_max_ctr_complexity,
    simple_ctr_projections,
    tree_base_projections,
    update_reindex_hash,
)


comptime _H01 = UInt64(4731097034409465305)
comptime _H02 = UInt64(9462194068818930610)
comptime _H01_0 = UInt64(8714454114147103133)
comptime _H01_1 = UInt64(13445551148556568438)


# ---------------------------------------------------------------------------
# The two categorical folds
# ---------------------------------------------------------------------------


def test_the_training_fold_adds_one_to_the_bin() raises:
    # `index_hash_calcer.cpp:104`, `CalcHash(hashArr[i], (ui64)block[i] + 1)`.
    assert_equal(cat_value_train(0), UInt64(1))
    assert_equal(cat_value_train(7), UInt64(8))


def test_the_final_table_fold_sign_extends() raises:
    # `index_hash_calcer.cpp:117`, `CalcHash(hashArr[i], (int)origValsView[...])`.
    # The `(int)` narrows before the implicit widening to ui64, so anything with
    # the top bit set is sign-extended. No `+ 1` on this side.
    assert_equal(cat_value_final(UInt32(0)), UInt64(0))
    assert_equal(cat_value_final(UInt32(7)), UInt64(7))
    assert_equal(cat_value_final(UInt32(0x7FFFFFFF)), UInt64(0x7FFFFFFF))
    assert_equal(
        cat_value_final(UInt32(0x80000000)), UInt64(0xFFFFFFFF80000000)
    )
    assert_equal(
        cat_value_final(UInt32(0xFFFFFFFF)), UInt64(0xFFFFFFFFFFFFFFFF)
    )


def test_the_hash_seed_is_zero_and_zero_is_the_folds_fixed_point() raises:
    # `calc_hash(0, 0) == 0`, which is exactly why the training fold adds one.
    assert_equal(ctr_combination_hash(UInt64(0), UInt64(0)), UInt64(0))
    assert_equal(ctr_combination_hash(UInt64(0), UInt64(1)), _H01)
    assert_equal(ctr_combination_hash(UInt64(0), UInt64(2)), _H02)


# ---------------------------------------------------------------------------
# `calc_hashes`, both halves
# ---------------------------------------------------------------------------


def test_a_single_cat_projection_hashes_to_calc_hash_of_zero_and_bin_plus_one() raises:
    var proj = Projection.single_cat(0)
    var cats = List[List[Int]]()
    cats.append([0, 1])
    var floats = List[List[Int]]()
    var out = List[UInt64]()
    projection_hashes(proj, cats, floats, 2, out)
    assert_equal(out[0], _H01)
    assert_equal(out[1], _H02)


def test_a_binarized_member_folds_a_bit_and_not_a_value() raises:
    # THE test for the half A19 said was most likely to be missed. Float bins
    # 13 and 200 are both above the split at 12, so `IsTrueHistogram` is 1 for
    # each and the two rows must land in the SAME bucket. Bin 12 is not above
    # 12 -- the test is strict `>` -- so it folds a 0, and `calc_hash(0, 0)` is
    # 0.
    var proj = Projection()
    proj.add_bin_split(BinSplit(0, 12))
    var cats = List[List[Int]]()
    var floats = List[List[Int]]()
    floats.append([12, 13, 200])
    var out = List[UInt64]()
    projection_hashes(proj, cats, floats, 3, out)
    assert_equal(out[0], UInt64(0))
    assert_equal(out[1], _H01)
    assert_equal(out[2], _H01)
    assert_true(out[1] == out[2])
    assert_false(out[0] == out[1])


def test_a_one_hot_member_tests_equality_not_a_threshold() raises:
    # `IsTrueOneHotFeature(featureValue, splitValue) { return featureValue == splitValue; }`
    # so bin 4 is NOT true against value 3 even though it is larger.
    var proj = Projection()
    proj.add_one_hot_split(OneHotSplit(0, 3))
    var cats = List[List[Int]]()
    cats.append([3, 4, 2])
    var floats = List[List[Int]]()
    var out = List[UInt64]()
    projection_hashes(proj, cats, floats, 3, out)
    assert_equal(out[0], _H01)
    assert_equal(out[1], UInt64(0))
    assert_equal(out[2], UInt64(0))


def test_a_combination_folds_categoricals_first_then_binarized() raises:
    # `CalcHashes` runs CatFeatures, then BinFeatures, then OneHotFeatures.
    # cat bin 0 -> calc_hash(0, 1); then the float split's bit.
    var proj = Projection.single_cat(0)
    proj.add_bin_split(BinSplit(0, 12))
    var cats = List[List[Int]]()
    cats.append([0, 0])
    var floats = List[List[Int]]()
    floats.append([5, 13])
    var out = List[UInt64]()
    projection_hashes(proj, cats, floats, 2, out)
    assert_equal(out[0], _H01_0)
    assert_equal(out[1], _H01_1)


def test_the_row_level_and_bulk_hashes_agree() raises:
    var proj = Projection.single_cat(0)
    proj.add_bin_split(BinSplit(0, 12))
    var cat_values = List[UInt64]()
    cat_values.append(cat_value_train(0))
    var float_bins = List[Int]()
    float_bins.append(13)
    var one_hot_bins = List[Int]()
    assert_equal(
        projection_row_hash(proj, cat_values, float_bins, one_hot_bins),
        _H01_1,
    )


def test_construction_order_does_not_change_a_projection() raises:
    # Every insert sorts, matching `AddCatFeature`/`AddBinFeature`/`AddOneHotFeature`,
    # so `TProjection::operator==`'s elementwise compare is order-independent.
    var a = Projection()
    a.add_cat_feature(3)
    a.add_cat_feature(1)
    var b = Projection()
    b.add_cat_feature(1)
    b.add_cat_feature(3)
    assert_true(a.equals(b))
    assert_equal(a.cat_features[0], 1)
    assert_equal(a.cat_features[1], 3)


# ---------------------------------------------------------------------------
# Projection predicates
# ---------------------------------------------------------------------------


def test_the_bin_and_one_hot_blob_counts_as_one_toward_complexity() raises:
    # `GetFullProjectionLength` (`projection.h:138`):
    #   CatFeatures.size() + (BinFeatures.size() + OneHotFeatures.size() > 0)
    # Five float splits and three one-hot splits together contribute ONE.
    var p = Projection()
    p.add_cat_feature(1)
    p.add_cat_feature(2)
    for i in range(5):
        p.add_bin_split(BinSplit(i, 3))
    for i in range(3):
        p.add_one_hot_split(OneHotSplit(i, 7))
    assert_equal(p.full_projection_length(), 3)
    assert_equal(p.n_members(), 10)

    var q = Projection()
    q.add_cat_feature(1)
    q.add_cat_feature(2)
    assert_equal(q.full_projection_length(), 2)

    var blob = Projection()
    blob.add_bin_split(BinSplit(0, 1))
    assert_equal(blob.full_projection_length(), 1)


def test_one_cat_column_plus_a_float_split_is_already_a_combination() raises:
    # `IsSingleCatFeature` needs BinFeatures AND OneHotFeatures empty, so this
    # is the predicate `GetCtrInfo` routes on and it is not "one categorical
    # column".
    var single = Projection.single_cat(0)
    assert_true(single.is_single_cat_feature())

    var withbin = Projection.single_cat(0)
    withbin.add_bin_split(BinSplit(1, 4))
    assert_false(withbin.is_single_cat_feature())

    var withonehot = Projection.single_cat(0)
    withonehot.add_one_hot_split(OneHotSplit(1, 4))
    assert_false(withonehot.is_single_cat_feature())


def test_a_repeated_member_is_redundant() raises:
    var p = Projection()
    p.add_cat_feature(2)
    assert_false(p.is_redundant())
    p.add_cat_feature(2)
    assert_true(p.is_redundant())

    var q = Projection()
    q.add_bin_split(BinSplit(0, 3))
    q.add_bin_split(BinSplit(0, 4))
    assert_false(q.is_redundant())
    q.add_bin_split(BinSplit(0, 3))
    assert_true(q.is_redundant())


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------


def test_only_columns_too_wide_to_one_hot_get_a_simple_ctr() raises:
    # `greedy_tensor_search.cpp:469`: `uniq <= oneHotMaxSize` returns early.
    # Column 2 has exactly `one_hot_max_size` levels, so it is one-hot and gets
    # no CTR -- the boundary is `<=`, not `<`.
    var projs = simple_ctr_projections([0, 1, 2], [10, 10, 2], 2)
    assert_equal(len(projs), 2)
    assert_equal(projs[0].cat_features[0], 0)
    assert_equal(projs[1].cat_features[0], 1)


def test_the_tree_bases_are_one_blob_plus_the_used_ctrs() raises:
    # `seenProj` is ALL the tree's bin and one-hot splits as ONE projection,
    # plus one projection per used CTR. Not the subsets of the tree's splits,
    # which is what keeps the enumeration polynomial.
    var bins = List[BinSplit]()
    bins.append(BinSplit(5, 3))
    bins.append(BinSplit(2, 9))
    var ones = List[OneHotSplit]()
    var used = List[Projection]()
    used.append(Projection.single_cat(0))
    var bases = tree_base_projections(bins, ones, used)
    assert_equal(len(bases), 2)
    # The blob sorts first: its cat vector is empty and empty is less.
    assert_equal(len(bases[0].cat_features), 0)
    assert_equal(len(bases[0].bin_splits), 2)
    assert_equal(len(bases[1].cat_features), 1)


def test_an_empty_tree_contributes_no_bases() raises:
    # `if (baseProj.IsEmpty()) continue;` -- at depth 0 only the simple CTRs
    # are candidates.
    var bins = List[BinSplit]()
    var ones = List[OneHotSplit]()
    var used = List[Projection]()
    assert_equal(len(tree_base_projections(bins, ones, used)), 0)


def test_growth_adds_one_categorical_column_and_rejects_the_rest() raises:
    var bins = List[BinSplit]()
    bins.append(BinSplit(5, 3))
    var ones = List[OneHotSplit]()
    var used = List[Projection]()
    used.append(Projection.single_cat(0))
    var bases = tree_base_projections(bins, ones, used)

    # Column 2 is one-hot sized and never joins a projection; base {c0} + c0 is
    # redundant. What survives at complexity 2 is blob+c0, blob+c1, {c0,c1}.
    var grown = grow_tree_ctr_projections(bases, [0, 1, 2], [10, 10, 2], 2, 2)
    assert_equal(len(grown), 3)
    # `Projection.less` order: cats [0] < cats [0,1] < cats [1].
    assert_equal(len(grown[0].cat_features), 1)
    assert_equal(grown[0].cat_features[0], 0)
    assert_equal(len(grown[0].bin_splits), 1)
    assert_equal(len(grown[1].cat_features), 2)
    assert_equal(len(grown[1].bin_splits), 0)
    assert_equal(len(grown[2].cat_features), 1)
    assert_equal(grown[2].cat_features[0], 1)
    assert_equal(len(grown[2].bin_splits), 1)


def test_complexity_one_reaches_no_combination_at_all() raises:
    # Every candidate here has length 2, so complexity 1 rejects all of them.
    # This is the setting CatBoost silently picks below 200 iterations.
    var bins = List[BinSplit]()
    bins.append(BinSplit(5, 3))
    var ones = List[OneHotSplit]()
    var used = List[Projection]()
    used.append(Projection.single_cat(0))
    var bases = tree_base_projections(bins, ones, used)
    assert_equal(
        len(grow_tree_ctr_projections(bases, [0, 1, 2], [10, 10, 2], 2, 1)), 0
    )


def test_the_complexity_cap_counts_the_blob_as_one() raises:
    # Base = three cat columns plus a float split, so length 4. Adding a fourth
    # categorical column makes it 5 and is rejected at complexity 4; the same
    # base without the float split has length 3 and the add succeeds.
    var withblob = Projection()
    withblob.add_cat_feature(0)
    withblob.add_cat_feature(1)
    withblob.add_cat_feature(2)
    withblob.add_bin_split(BinSplit(9, 1))
    assert_equal(withblob.full_projection_length(), 4)

    var noblob = Projection()
    noblob.add_cat_feature(0)
    noblob.add_cat_feature(1)
    noblob.add_cat_feature(2)
    assert_equal(noblob.full_projection_length(), 3)

    var b1 = List[Projection]()
    b1.append(withblob^)
    assert_equal(len(grow_tree_ctr_projections(b1, [3], [10], 2, 4)), 0)

    var b2 = List[Projection]()
    b2.append(noblob^)
    assert_equal(len(grow_tree_ctr_projections(b2, [3], [10], 2, 4)), 1)


def test_enumeration_does_not_depend_on_the_order_of_its_inputs() raises:
    # CatBoost walks a THashSet, so its order is a hash table's bucket layout.
    # Ours is `TProjection::operator<` and must be identical whichever order
    # the caller hands the features and bases in.
    var bins = List[BinSplit]()
    bins.append(BinSplit(5, 3))
    var ones = List[OneHotSplit]()
    var used = List[Projection]()
    used.append(Projection.single_cat(0))

    var forward = grow_tree_ctr_projections(
        tree_base_projections(bins, ones, used), [0, 1], [10, 10], 2, 2
    )
    var reversed = grow_tree_ctr_projections(
        tree_base_projections(bins, ones, used), [1, 0], [10, 10], 2, 2
    )
    assert_equal(len(forward), len(reversed))
    for i in range(len(forward)):
        assert_true(forward[i].equals(reversed[i]))


def test_four_candidates_per_projection_at_the_cpu_defaults() raises:
    # `AddCtrsToCandList`: sum over descriptions of targetBorderCount * priors.
    # Borders at two classes is 1 border x 3 priors; Counter is 1 x 1.
    var routing = CtrRouting.catboost_cpu_defaults()
    assert_equal(ctr_candidates_per_projection(routing.simple, 2), 4)
    assert_equal(ctr_candidates_per_projection(routing.combination, 2), 4)
    # Three target classes: Borders emits classes - 1 = 2 borders x 3 priors,
    # Counter still 1.
    assert_equal(ctr_candidates_per_projection(routing.simple, 3), 7)


def test_the_projection_count_bound() raises:
    # C * D * (D + 3) / 2. Ten wide columns at depth 6 is 270 projections per
    # tree; this is a derived bound, not a measurement.
    assert_equal(ctr_projection_count_bound(10, 6), 270)
    assert_equal(ctr_projection_count_bound(10, 8), 440)
    assert_equal(ctr_projection_count_bound(1, 1), 2)
    assert_equal(ctr_projection_count_bound(10, 0), 0)


# ---------------------------------------------------------------------------
# `max_ctr_complexity`
# ---------------------------------------------------------------------------


def test_the_complexity_bound_is_catboosts_own() raises:
    check_max_ctr_complexity(1)
    check_max_ctr_complexity(4)
    check_max_ctr_complexity(MAX_CTR_COMPLEXITY_LIMIT - 1)
    with assert_raises():
        check_max_ctr_complexity(0)
    with assert_raises():
        check_max_ctr_complexity(MAX_CTR_COMPLEXITY_LIMIT)


def test_ctr_mojos_guard_no_longer_refuses_combinations() raises:
    # A19 left `check_ctr_complexity` refusing anything above 1 BY NAME so this
    # lane would delete a refusal rather than discover an assumption. It now
    # enforces CatBoost's own bound and nothing else.
    check_ctr_complexity(1)
    check_ctr_complexity(4)
    check_ctr_complexity(15)
    with assert_raises():
        check_ctr_complexity(16)
    with assert_raises():
        check_ctr_complexity(0)


def test_a_short_fit_silently_drops_to_complexity_one() raises:
    # `catboost_options.cpp:1046` with `IsSmallIterationCount(n) = n < 200`.
    # A harness that fits 100 trees at the defaults is measuring complexity 1.
    assert_equal(resolve_max_ctr_complexity(4, False, 100), 1)
    assert_equal(
        resolve_max_ctr_complexity(4, False, SMALL_ITERATION_COUNT), 4
    )
    assert_equal(resolve_max_ctr_complexity(4, False, 1000), 4)
    # Set by the user, so the override does not fire.
    assert_equal(resolve_max_ctr_complexity(4, True, 100), 4)


# ---------------------------------------------------------------------------
# `ctr_leaf_count_limit`
# ---------------------------------------------------------------------------


def test_the_default_limit_gives_first_seen_bucket_ids() raises:
    var hashes = List[UInt64]()
    hashes.append(UInt64(5))
    hashes.append(UInt64(3))
    hashes.append(UInt64(5))
    hashes.append(UInt64(9))
    var ids = List[Int]()
    var bucket_hash = List[UInt64]()
    var bucket_count = List[Int]()
    var n = compute_reindex_hash(
        CTR_LEAF_COUNT_UNLIMITED, hashes, ids, bucket_hash, bucket_count
    )
    assert_equal(n, 3)
    assert_equal(ids[0], 0)
    assert_equal(ids[1], 1)
    assert_equal(ids[2], 0)
    assert_equal(ids[3], 2)
    assert_equal(bucket_hash[0], UInt64(5))
    assert_equal(bucket_hash[1], UInt64(3))
    assert_equal(bucket_hash[2], UInt64(9))
    assert_equal(bucket_count[0], 2)
    assert_equal(bucket_count[1], 1)
    assert_equal(bucket_count[2], 1)


def test_an_unreached_limit_changes_nothing() raises:
    # CatBoost's branch 2 numbers the buckets in hash-table order; ours keeps
    # first-seen order so branches 1 and 2 agree with each other. The partition
    # is identical either way and that is what the CTR loops consume.
    var hashes = List[UInt64]()
    hashes.append(UInt64(5))
    hashes.append(UInt64(3))
    hashes.append(UInt64(5))
    hashes.append(UInt64(9))
    var a_ids = List[Int]()
    var a_bh = List[UInt64]()
    var a_bc = List[Int]()
    var b_ids = List[Int]()
    var b_bh = List[UInt64]()
    var b_bc = List[Int]()
    var na = compute_reindex_hash(
        CTR_LEAF_COUNT_UNLIMITED, hashes, a_ids, a_bh, a_bc
    )
    var nb = compute_reindex_hash(3, hashes, b_ids, b_bh, b_bc)
    assert_equal(na, nb)
    for i in range(len(a_ids)):
        assert_equal(a_ids[i], b_ids[i])


def test_the_overflow_bucket_is_the_last_kept_bucket() raises:
    # hashes 7,7,7 / 5,5 / 3 / 9 -- counts 3, 2, 1, 1 over four distinct values.
    # top_size 2 keeps 7 (bucket 0) and 5 (bucket 1). `index_hash_calcer.cpp:222`
    # sends everything else to `reindexHash.Size() - 1`, i.e. bucket 1, the
    # LAST KEPT bucket and not a fresh third one. The header comment at
    # `index_hash_calcer.h:42` says otherwise; the code wins.
    var hashes = List[UInt64]()
    hashes.append(UInt64(7))
    hashes.append(UInt64(7))
    hashes.append(UInt64(7))
    hashes.append(UInt64(5))
    hashes.append(UInt64(5))
    hashes.append(UInt64(3))
    hashes.append(UInt64(9))
    var ids = List[Int]()
    var bucket_hash = List[UInt64]()
    var bucket_count = List[Int]()
    var n = compute_reindex_hash(2, hashes, ids, bucket_hash, bucket_count)
    assert_equal(n, 2)
    assert_equal(bucket_hash[0], UInt64(7))
    assert_equal(bucket_hash[1], UInt64(5))
    assert_equal(ids[0], 0)
    assert_equal(ids[1], 0)
    assert_equal(ids[2], 0)
    assert_equal(ids[3], 1)
    assert_equal(ids[4], 1)
    assert_equal(ids[5], 1)
    assert_equal(ids[6], 1)
    # 3 kept rows in bucket 0; bucket 1 is its own 2 plus the 2 dropped rows.
    assert_equal(bucket_count[0], 3)
    assert_equal(bucket_count[1], 4)


def test_the_kept_set_does_not_depend_on_row_order() raises:
    # Same multiset of hashes, different row order. `std::nth_element` would be
    # free to answer differently; `(count desc, hash asc)` is a strict total
    # order over distinct hashes and cannot.
    var a = List[UInt64]()
    a.append(UInt64(7))
    a.append(UInt64(7))
    a.append(UInt64(7))
    a.append(UInt64(5))
    a.append(UInt64(5))
    a.append(UInt64(3))
    a.append(UInt64(9))
    var b = List[UInt64]()
    b.append(UInt64(9))
    b.append(UInt64(3))
    b.append(UInt64(5))
    b.append(UInt64(7))
    b.append(UInt64(5))
    b.append(UInt64(7))
    b.append(UInt64(7))
    var a_ids = List[Int]()
    var a_bh = List[UInt64]()
    var a_bc = List[Int]()
    var b_ids = List[Int]()
    var b_bh = List[UInt64]()
    var b_bc = List[Int]()
    assert_equal(compute_reindex_hash(2, a, a_ids, a_bh, a_bc), 2)
    assert_equal(compute_reindex_hash(2, b, b_ids, b_bh, b_bc), 2)
    for i in range(2):
        assert_equal(a_bh[i], b_bh[i])
        assert_equal(a_bc[i], b_bc[i])


def test_a_frequency_tie_keeps_the_smaller_hash() raises:
    # 9 appears first and 3 second, both twice. The tie-break is the hash, not
    # the order of appearance, so 3 is kept and everything lands in bucket 0.
    var hashes = List[UInt64]()
    hashes.append(UInt64(9))
    hashes.append(UInt64(9))
    hashes.append(UInt64(3))
    hashes.append(UInt64(3))
    hashes.append(UInt64(5))
    var ids = List[Int]()
    var bucket_hash = List[UInt64]()
    var bucket_count = List[Int]()
    assert_equal(compute_reindex_hash(1, hashes, ids, bucket_hash, bucket_count), 1)
    assert_equal(bucket_hash[0], UInt64(3))
    assert_equal(bucket_count[0], 5)
    for i in range(5):
        assert_equal(ids[i], 0)


def test_test_rows_append_new_buckets_in_scan_order() raises:
    # `UpdateReindexHash` (`index_hash_calcer.cpp:231`), the pass over each test
    # set after the learn pass.
    var learn = List[UInt64]()
    learn.append(UInt64(5))
    learn.append(UInt64(3))
    learn.append(UInt64(5))
    learn.append(UInt64(9))
    var ids = List[Int]()
    var bucket_hash = List[UInt64]()
    var bucket_count = List[Int]()
    _ = compute_reindex_hash(
        CTR_LEAF_COUNT_UNLIMITED, learn, ids, bucket_hash, bucket_count
    )
    var test = List[UInt64]()
    test.append(UInt64(9))
    test.append(UInt64(11))
    test.append(UInt64(3))
    var test_ids = List[Int]()
    assert_equal(update_reindex_hash(test, test_ids, bucket_hash), 4)
    assert_equal(test_ids[0], 2)
    assert_equal(test_ids[1], 3)
    assert_equal(test_ids[2], 1)
    assert_equal(bucket_hash[3], UInt64(11))


def test_store_all_simple_ctr_exempts_a_column_and_never_a_combination() raises:
    # `online_ctr.cpp:690`: the exemption is gated on `IsSingleCatFeature()`.
    var single = Projection.single_cat(0)
    var combo = Projection.single_cat(0)
    combo.add_cat_feature(1)
    assert_equal(resolve_ctr_leaf_count_limit(100, single, False), 100)
    assert_equal(
        resolve_ctr_leaf_count_limit(100, single, True),
        CTR_LEAF_COUNT_UNLIMITED,
    )
    assert_equal(resolve_ctr_leaf_count_limit(100, combo, True), 100)
    with assert_raises():
        _ = resolve_ctr_leaf_count_limit(0, single, False)


def test_the_bucket_space_bound() raises:
    # `approxBucketsCount`, plus the 2^b CatBoost's version drops.
    assert_equal(projection_bucket_space_bound([10, 10, 10], 0, 100000), 1000)
    assert_equal(projection_bucket_space_bound([10, 10, 10], 2, 100000), 4000)
    # The early break caps at the row count rather than overflowing.
    assert_equal(projection_bucket_space_bound([10, 10, 10], 0, 500), 500)
    assert_equal(
        projection_bucket_space_bound([1000, 1000, 1000, 1000], 0, 1000000),
        1000000,
    )
    var none = List[Int]()
    assert_equal(projection_bucket_space_bound(none, 0, 500), 1)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def test_a_combination_is_routed_to_the_combination_list() raises:
    # `GetCtrInfo`: `IsSingleCatFeature()` -> SimpleCtrs, everything else ->
    # TreeCtrs. The two lists are given different lengths here so the routing
    # is observable.
    var routing = CtrRouting()
    routing.simple.append(CtrParams.enable(CTR_BORDERS, default_priors(CTR_BORDERS)))
    routing.combination.append(
        CtrParams.enable(CTR_BORDERS, default_priors(CTR_BORDERS))
    )
    routing.combination.append(
        CtrParams.enable(CTR_COUNTER, default_priors(CTR_COUNTER))
    )

    var single = Projection.single_cat(0)
    assert_equal(len(ctr_info_for_projection(routing, single)), 1)

    var two_cats = Projection.single_cat(0)
    two_cats.add_cat_feature(1)
    assert_equal(len(ctr_info_for_projection(routing, two_cats)), 2)

    # One categorical column plus a float split is ALREADY a combination.
    var one_cat_one_split = Projection.single_cat(0)
    one_cat_one_split.add_bin_split(BinSplit(4, 2))
    assert_equal(len(ctr_info_for_projection(routing, one_cat_one_split)), 2)

    with assert_raises():
        _ = ctr_info_for_projection(routing, Projection())


def test_the_two_default_lists_are_built_independently() raises:
    # `SetCtrDefaults` builds `defaultSimpleCtrs` and `defaultTreeCtrs` from
    # separate constructor calls. On the CPU their content matches; that is a
    # coincidence of `CreateDefaultCounter` ignoring its projection type there,
    # not inheritance.
    var routing = CtrRouting.catboost_cpu_defaults()
    routing.validate()
    assert_equal(len(routing.simple), 2)
    assert_equal(len(routing.combination), 2)
    assert_equal(routing.simple[0].ctr_type, CTR_BORDERS)
    assert_equal(routing.simple[0].n_priors(), 3)
    assert_equal(routing.simple[1].ctr_type, CTR_COUNTER)
    assert_equal(routing.simple[1].n_priors(), 1)
    assert_equal(routing.combination[0].ctr_type, CTR_BORDERS)
    assert_equal(routing.combination[0].n_priors(), 3)
    assert_equal(routing.combination[1].ctr_type, CTR_COUNTER)
    assert_equal(routing.combination[1].n_priors(), 1)

    # Mutating one list cannot reach the other.
    routing.simple.append(CtrParams.enable(CTR_BORDERS, default_priors(CTR_BORDERS)))
    assert_equal(len(routing.simple), 3)
    assert_equal(len(routing.combination), 2)


def test_an_empty_combination_list_is_refused() raises:
    var routing = CtrRouting()
    routing.simple.append(CtrParams.enable(CTR_BORDERS, default_priors(CTR_BORDERS)))
    with assert_raises():
        routing.validate()


def test_the_routing_warning_is_catboosts_own_text() raises:
    assert_equal(
        ctr_routing_warning(True, False),
        String("Change of simpleCtr will not affect combinations ctrs."),
    )
    assert_equal(
        ctr_routing_warning(False, True),
        String("Change of combinations ctrs will not affect simple ctrs"),
    )
    assert_equal(ctr_routing_warning(True, True), String(""))
    assert_equal(ctr_routing_warning(False, False), String(""))


# ---------------------------------------------------------------------------
# The unreached refusal
# ---------------------------------------------------------------------------


def test_an_enabled_combination_is_refused_at_a_trainer_boundary() raises:
    # Nothing drives `grow_tree_ctr_projections` from a grow loop, so an
    # enabled complexity above 1 would compute nothing. Deleting this guard is
    # the wiring lane's second step, not a side effect of its first.
    check_ctr_combination_trainer_support(1, True)
    check_ctr_combination_trainer_support(4, False)
    with assert_raises():
        check_ctr_combination_trainer_support(4, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
