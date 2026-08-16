"""CatBoost text processing and text features
(src/mojotrees/text_processing.mojo, src/mojotrees/text_features.mojo).

The centre of this file is the leakage pair: `test_the_ordered_pass_reads_
before_it_writes` and `test_train_and_predict_are_different_numbers`. Every
other test is either a transcription check against a hand-computed CatBoost
formula or a determinism check on the dictionary ordering.

The two NaiveBayes and three BM25 constants below were computed by hand from
the formulas in `docs/design/CATBOOST_CATALOG.md`'s A18 note, on the corpus
that `_toy_bags` builds, and are not outputs of this implementation echoed
back at it.
"""

from std.testing import assert_almost_equal, assert_equal, assert_false
from std.testing import assert_raises, assert_true
from std.testing import TestSuite

from mojotrees.text_features import (
    BM25_DEFAULT_B,
    BM25_DEFAULT_K,
    BM25_DEFAULT_TRUNCATE_BORDER,
    Bm25Calcer,
    DEFAULT_TOP_TOKENS_COUNT,
    NAIVE_BAYES_DEFAULT_PRIOR,
    NaiveBayesCalcer,
    bm25_apply,
    bm25_fit,
    bm25_online_features,
    bow_feature_count,
    bow_features,
    check_permutation,
    identity_permutation,
    naive_bayes_apply,
    naive_bayes_fit,
    naive_bayes_online_features,
)
from mojotrees.text_processing import (
    DEFAULT_MAX_DICTIONARY_SIZE,
    DEFAULT_OCCURRENCE_LOWER_BOUND,
    DictionaryParams,
    MAX_DICTIONARY_SIZE_UNBOUNDED,
    TOKEN_NOT_IN_DICTIONARY,
    TextBag,
    TokenizerParams,
    build_dictionary,
    default_dictionaries,
    digitize,
    ngram_keys,
    tokenize,
)


def _bag(var ids: List[Int], var counts: List[Int]) -> TextBag:
    """A bag built directly, bypassing the dictionary, so the arithmetic
    tests state their inputs instead of deriving them."""
    return TextBag(ids^, counts^)


def _docs1() -> List[String]:
    var out: List[String] = [String("a")]
    return out^


def _toy_bags() -> List[TextBag]:
    """Three documents over a two-token dictionary.

    row 0: token 0 once      class 0
    row 1: token 1 once      class 1
    row 2: tokens 0 and 1    class 0
    """
    var out = List[TextBag]()
    out.append(_bag([0], [1]))
    out.append(_bag([1], [1]))
    out.append(_bag([0, 1], [1, 1]))
    return out^


def _toy_classes() -> List[Int]:
    var out: List[Int] = [0, 1, 0]
    return out^


# --------------------------------------------------------------------------
# Tokenizer.
# --------------------------------------------------------------------------


def test_the_default_tokenizer_does_almost_nothing() raises:
    # CatBoost's default is StringSplitter(s).SplitByString(" ").SkipEmpty()
    # and nothing else: no lowercasing, no punctuation stripping, no number
    # handling. "Cat," and "cat" are two different tokens at the defaults,
    # and the run of two spaces and the trailing space contribute nothing.
    var params = TokenizerParams()
    assert_equal(params.delimiter, String(" "))
    assert_false(params.lowercasing)
    assert_true(params.skip_empty)
    assert_false(params.split_by_set)

    var tokens = tokenize(String("Cat, cat  CAT "), params)
    assert_equal(len(tokens), 3)
    assert_equal(tokens[0], String("Cat,"))
    assert_equal(tokens[1], String("cat"))
    assert_equal(tokens[2], String("CAT"))


def test_skip_empty_off_keeps_the_empty_pieces() raises:
    var params = TokenizerParams(String(" "), False, False, False)
    var tokens = tokenize(String("a  b"), params)
    assert_equal(len(tokens), 3)
    assert_equal(tokens[1], String(""))


def test_lowercasing_is_opt_in() raises:
    var params = TokenizerParams(String(" "), False, True, True)
    var tokens = tokenize(String("Cat CAT"), params)
    assert_equal(tokens[0], String("cat"))
    assert_equal(tokens[1], String("cat"))


def test_split_by_set_treats_each_byte_as_a_delimiter() raises:
    var params = TokenizerParams(String(" ,"), True, True, False)
    var tokens = tokenize(String("a,b c"), params)
    assert_equal(len(tokens), 3)
    assert_equal(tokens[0], String("a"))
    assert_equal(tokens[1], String("b"))
    assert_equal(tokens[2], String("c"))


# --------------------------------------------------------------------------
# Dictionary: the determinism the lane was most likely to get wrong.
# --------------------------------------------------------------------------


def test_dictionary_ids_are_frequency_then_lexicographic() raises:
    # Counts: a=2, b=2, c=1. The tie between a and b is broken by the key
    # ascending, which is `FinishBuilding`'s comparator, so a=0, b=1, c=2.
    # Ids being frequency-ordered is what makes BoW's "top n tokens" the ids
    # 0..n-1 with no lookup.
    var docs: List[String] = [String("b a"), String("a b"), String("c")]
    var params = DictionaryParams(1, 1, MAX_DICTIONARY_SIZE_UNBOUNDED)
    var d = build_dictionary(docs, TokenizerParams(), params)
    assert_equal(d.size(), 3)
    assert_equal(d.keys[0], String("a"))
    assert_equal(d.keys[1], String("b"))
    assert_equal(d.keys[2], String("c"))
    assert_equal(d.counts[0], 2)
    assert_equal(d.counts[1], 2)
    assert_equal(d.counts[2], 1)


def test_dictionary_order_does_not_depend_on_document_order() raises:
    # The counts live in a hash map, and a hash map iterated in insertion
    # order is not a deterministic ordering. The total-order sort is what
    # erases that, so feeding the same corpus in a different order -- which
    # changes both the insertion order and the map's internal layout -- must
    # give the identical id assignment.
    var params = DictionaryParams(1, 1, MAX_DICTIONARY_SIZE_UNBOUNDED)
    var forward: List[String] = [
        String("b a"), String("a b"), String("c"), String("d d d")
    ]
    var backward: List[String] = [
        String("d d d"), String("c"), String("a b"), String("b a")
    ]
    var df = build_dictionary(forward, TokenizerParams(), params)
    var db = build_dictionary(backward, TokenizerParams(), params)
    assert_equal(df.size(), db.size())
    for i in range(df.size()):
        assert_equal(df.keys[i], db.keys[i])
        assert_equal(df.counts[i], db.counts[i])


def test_the_two_bounds_compose_in_one_order_only() raises:
    # occurrence_lower_bound DROPS and max_dictionary_size TRUNCATES, in that
    # order. Raising the size bound cannot recover a key the occurrence bound
    # removed: c has count 1 and is gone at bound 2 whatever the size is.
    var docs: List[String] = [String("b a"), String("a b"), String("c")]
    var lenient = DictionaryParams(1, 1, MAX_DICTIONARY_SIZE_UNBOUNDED)
    assert_equal(
        build_dictionary(docs, TokenizerParams(), lenient).size(), 3
    )

    var filtered = DictionaryParams(1, 2, MAX_DICTIONARY_SIZE_UNBOUNDED)
    var d2 = build_dictionary(docs, TokenizerParams(), filtered)
    assert_equal(d2.size(), 2)
    assert_equal(d2.lookup(String("c")), TOKEN_NOT_IN_DICTIONARY)

    var filtered_wide = DictionaryParams(1, 2, 100)
    assert_equal(
        build_dictionary(docs, TokenizerParams(), filtered_wide).size(), 2
    )

    var truncated = DictionaryParams(1, 1, 1)
    var d3 = build_dictionary(docs, TokenizerParams(), truncated)
    assert_equal(d3.size(), 1)
    assert_equal(d3.keys[0], String("a"))


def test_catboost_bounds_are_three_and_fifty_thousand() raises:
    # CatBoost's DEFAULT_DICTIONARY_BUILDER_OPTIONS overrides the dictionary
    # library's own {50, -1}. Quoting the library's numbers for CatBoost
    # would be wrong.
    assert_equal(DEFAULT_OCCURRENCE_LOWER_BOUND, 3)
    assert_equal(DEFAULT_MAX_DICTIONARY_SIZE, 50000)
    var defaults = DictionaryParams()
    assert_equal(defaults.gram_order, 1)
    assert_equal(defaults.occurrence_lower_bound, 3)
    assert_equal(defaults.max_dictionary_size, 50000)


def test_the_untouched_default_is_two_dictionaries() raises:
    # SetDefault installs BiGram and Word, in that order, and BoW consumes
    # both. The single-"Word" fallback is a DIFFERENT default, reached only
    # when the user supplied a partial text_processing block.
    var d = default_dictionaries()
    assert_equal(len(d), 2)
    assert_equal(d[0].gram_order, 2)
    assert_equal(d[1].gram_order, 1)


def test_bigram_keys_and_the_short_document_rule() raises:
    var bigram = DictionaryParams(2, 1, MAX_DICTIONARY_SIZE_UNBOUNDED)
    var tokens = tokenize(String("a b c"), TokenizerParams())
    var keys = ngram_keys(tokens, bigram)
    assert_equal(len(keys), 2)

    # A document shorter than the gram order contributes NOTHING, not a
    # padded gram: `if (tokenCount < GramOrder) return;`.
    var short = tokenize(String("a"), TokenizerParams())
    assert_equal(len(ngram_keys(short, bigram)), 0)

    var docs: List[String] = [String("a b c"), String("a b c")]
    var d = build_dictionary(docs, TokenizerParams(), bigram)
    assert_equal(d.size(), 2)
    assert_equal(d.counts[0], 2)


def test_unknown_tokens_are_dropped_not_mapped() raises:
    # EUnknownTokenPolicy::Skip is the default at every Apply overload, so no
    # sentinel id ever enters a bag.
    var docs: List[String] = [String("a a a")]
    var params = DictionaryParams(1, 1, MAX_DICTIONARY_SIZE_UNBOUNDED)
    var d = build_dictionary(docs, TokenizerParams(), params)
    var probe: List[String] = [String("a zzz a")]
    var bags = digitize(probe, TokenizerParams(), d)
    assert_equal(len(bags[0].token_ids), 1)
    assert_equal(bags[0].token_ids[0], 0)
    assert_equal(bags[0].counts[0], 2)


def test_a_bag_is_ascending_and_run_length_encoded() raises:
    # TText(TVector<ui32>&&) sorts then RLEs, and every accumulation in
    # text_features walks a bag in that order, which is what fixes the
    # summation order without anyone arranging for it.
    var docs: List[String] = [String("b a b a b")]
    var params = DictionaryParams(1, 1, MAX_DICTIONARY_SIZE_UNBOUNDED)
    var d = build_dictionary(docs, TokenizerParams(), params)
    # b occurs 3 times, a twice, so b is id 0.
    assert_equal(d.keys[0], String("b"))
    var probe: List[String] = [String("a b a")]
    var bags = digitize(probe, TokenizerParams(), d)
    assert_equal(len(bags[0].token_ids), 2)
    assert_equal(bags[0].token_ids[0], 0)
    assert_equal(bags[0].token_ids[1], 1)
    assert_equal(bags[0].counts[0], 1)
    assert_equal(bags[0].counts[1], 2)
    assert_equal(bags[0].total_tokens(), 3)


def test_the_refused_dictionary_options_are_refused_by_name() raises:
    with assert_raises():
        _ = build_dictionary(_docs1(), TokenizerParams(), DictionaryParams(0))
    with assert_raises():
        _ = build_dictionary(_docs1(), TokenizerParams(), DictionaryParams(6))
    with assert_raises():
        _ = build_dictionary(
            _docs1(), TokenizerParams(), DictionaryParams(1, 0)
        )
    with assert_raises():
        _ = build_dictionary(
            _docs1(), TokenizerParams(), DictionaryParams(1, 1, 0)
        )


# --------------------------------------------------------------------------
# BoW.
# --------------------------------------------------------------------------


def test_bow_is_a_binary_indicator_not_a_count() raises:
    # TBagOfWordsEstimator sets one bit per (token, document) and never reads
    # Count(). A token occurring five times still gives 1.0.
    var bags: List[TextBag] = [_bag([0], [5]), _bag([1], [1])]
    var out = bow_features(bags, 2, 2)
    assert_equal(len(out), 4)
    # Column-major: out[f * n_rows + r].
    assert_equal(out[0 * 2 + 0], 1.0)
    assert_equal(out[0 * 2 + 1], 0.0)
    assert_equal(out[1 * 2 + 0], 0.0)
    assert_equal(out[1 * 2 + 1], 1.0)


def test_bow_top_tokens_clamps_and_refuses() raises:
    assert_equal(DEFAULT_TOP_TOKENS_COUNT, 2000)
    assert_equal(bow_feature_count(5, 2), 2)
    assert_equal(bow_feature_count(2, 5), 2)
    with assert_raises():
        _ = bow_feature_count(5, 0)
    with assert_raises():
        _ = bow_feature_count(0, 5)


def test_bow_selects_the_top_ids_because_ids_are_frequency_ordered() raises:
    # Row 2 holds token id 2, which is outside the top 2, so it contributes
    # nothing. This is the whole content of GetTopTokens being an xrange.
    var bags: List[TextBag] = [_bag([0], [1]), _bag([2], [1])]
    var out = bow_features(bags, 3, 2)
    assert_equal(len(out), 4)
    assert_equal(out[0 * 2 + 0], 1.0)
    assert_equal(out[0 * 2 + 1], 0.0)
    assert_equal(out[1 * 2 + 0], 0.0)
    assert_equal(out[1 * 2 + 1], 0.0)


# --------------------------------------------------------------------------
# NaiveBayes.
# --------------------------------------------------------------------------


def test_naive_bayes_binary_emits_one_feature() raises:
    # BaseFeatureCount = numClasses > 2 ? numClasses : 1. Binary emits the
    # softmax probability of class 0 and nothing else.
    assert_equal(NaiveBayesCalcer(2).feature_count(), 1)
    assert_equal(NaiveBayesCalcer(3).feature_count(), 3)
    assert_equal(NAIVE_BAYES_DEFAULT_PRIOR, 0.5)


def test_naive_bayes_matches_the_hand_computed_formula() raises:
    # Fitted on the toy corpus in natural order the state is
    #   class 0: docs 2, freq {0: 2, 1: 1}, total 3
    #   class 1: docs 1, freq {1: 1},       total 1
    #   distinct tokens seen: 2
    # so for the bag {token 0: once}
    #   logProb(0) = log(2.5) + log(2.5) - log(4.5)
    #   logProb(1) = log(1.5) + log(0.5) - log(3.0)
    # and the max-shifted softmax of those two is 0.847457627118644.
    var bags = _toy_bags()
    var classes = _toy_classes()
    var calcer = naive_bayes_fit(bags, classes, 2)
    assert_equal(calcer.class_docs[0], 2)
    assert_equal(calcer.class_docs[1], 1)
    assert_equal(calcer.class_total_tokens[0], 3)
    assert_equal(calcer.class_total_tokens[1], 1)
    assert_equal(calcer.num_seen_tokens, 2)

    var value = calcer.compute(_bag([0], [1]))
    assert_equal(len(value), 1)
    assert_almost_equal(value[0], 0.847457627118644, atol=1e-12)


def test_naive_bayes_apply_is_column_major_and_target_free() raises:
    # This is the predict-time shape: no class labels are passed at all.
    var bags = _toy_bags()
    var calcer = naive_bayes_fit(bags, _toy_classes(), 2)
    var out = naive_bayes_apply(calcer, bags)
    assert_equal(len(out), 3)
    assert_almost_equal(out[0], 0.847457627118644, atol=1e-12)


def test_naive_bayes_refuses_a_non_classification_target() raises:
    var bags = _toy_bags()
    with assert_raises():
        _ = naive_bayes_fit(bags, _toy_classes(), 1)
    with assert_raises():
        var bad: List[Int] = [0, 1, 7]
        _ = naive_bayes_fit(bags, bad, 2)


# --------------------------------------------------------------------------
# THE LEAKAGE PAIR. This is what the lane exists for.
# --------------------------------------------------------------------------


def test_the_ordered_pass_reads_before_it_writes() raises:
    # The strongest statement of no-leak available without a second dataset:
    # flip the class label of the row that comes LAST in the permutation.
    # No row reads that label -- it is written into the calcer after the last
    # read -- so every online feature value must be bit-identical.
    var bags = _toy_bags()
    var perm: List[Int] = [2, 0, 1]

    var classes_a: List[Int] = [0, 1, 0]
    var classes_b: List[Int] = [0, 0, 0]  # row 1 is last; flip it.
    var a = naive_bayes_online_features(bags, classes_a, 2, perm)
    var b = naive_bayes_online_features(bags, classes_b, 2, perm)
    assert_equal(len(a), 3)
    for i in range(len(a)):
        assert_equal(a[i], b[i])

    # And the converse, so the test above cannot pass vacuously: flipping the
    # label of a row that IS read by a later row does move the numbers.
    var classes_c: List[Int] = [1, 1, 0]  # row 0 is second.
    var c = naive_bayes_online_features(bags, classes_c, 2, perm)
    var moved = False
    for i in range(len(a)):
        if a[i] != c[i]:
            moved = True
    assert_true(moved)


def test_the_first_row_of_a_permutation_sees_an_empty_calcer() raises:
    # Row perm[0] is scored against a calcer with no rows in it at all, which
    # is correct and is why the leading rows of a single permutation carry
    # almost no information. With two classes and an empty calcer both class
    # log probabilities are equal by symmetry -- docs 0, totals 0, no token
    # in either table -- so the softmax is exactly 0.5.
    var bags = _toy_bags()
    var perm: List[Int] = [2, 0, 1]
    var out = naive_bayes_online_features(bags, _toy_classes(), 2, perm)
    assert_equal(out[2], 0.5)


def test_train_and_predict_are_different_numbers() raises:
    # The asymmetry is not that the feature is unavailable at predict time.
    # It is that the training-time value is a PREFIX statistic and the
    # predict-time value is a FULL-CORPUS statistic, and for the same
    # document they are systematically different numbers. An implementation
    # in which these agree has leaked.
    var bags = _toy_bags()
    var classes = _toy_classes()
    var perm: List[Int] = [2, 0, 1]

    var online = naive_bayes_online_features(bags, classes, 2, perm)
    var frozen = naive_bayes_apply(
        naive_bayes_fit(bags, classes, 2), bags
    )
    assert_equal(len(online), len(frozen))
    var differ = False
    for i in range(len(online)):
        if online[i] != frozen[i]:
            differ = True
    assert_true(differ)


def test_the_ordered_pass_writes_every_row_in_row_order() raises:
    # The permutation controls the READING order; the output is indexed by
    # the original row, so a caller can concatenate the column onto its
    # matrix without permuting anything back. Two different permutations must
    # therefore both fill all n slots.
    var bags = _toy_bags()
    var classes = _toy_classes()
    var forward: List[Int] = [0, 1, 2]
    var backward: List[Int] = [2, 1, 0]
    var a = naive_bayes_online_features(bags, classes, 2, forward)
    var b = naive_bayes_online_features(bags, classes, 2, backward)
    assert_equal(len(a), 3)
    assert_equal(len(b), 3)
    # Row 0 leads permutation a and trails permutation b, so it is scored
    # against an empty calcer in one and a two-row prefix in the other.
    assert_equal(a[0], 0.5)
    assert_true(b[0] != 0.5)


def test_a_permutation_is_validated_not_assumed() raises:
    var ok: List[Int] = [2, 0, 1]
    check_permutation(ok, 3)
    check_permutation(identity_permutation(4), 4)
    var too_short: List[Int] = [0, 1]
    with assert_raises():
        check_permutation(too_short, 3)
    var repeated: List[Int] = [0, 0, 1]
    with assert_raises():
        check_permutation(repeated, 3)
    var out_of_range: List[Int] = [0, 1, 3]
    with assert_raises():
        check_permutation(out_of_range, 3)
    var bags = _toy_bags()
    with assert_raises():
        _ = naive_bayes_online_features(bags, _toy_classes(), 2, repeated)


# --------------------------------------------------------------------------
# BM25, including its three divergences from textbook BM25.
# --------------------------------------------------------------------------


def test_bm25_defaults_and_feature_count() raises:
    assert_equal(BM25_DEFAULT_K, 1.5)
    assert_equal(BM25_DEFAULT_B, 0.75)
    assert_equal(BM25_DEFAULT_TRUNCATE_BORDER, 1e-3)
    # Unlike NaiveBayes, binary emits TWO features.
    assert_equal(Bm25Calcer(2).feature_count(), 2)
    assert_equal(Bm25Calcer(3).feature_count(), 3)
    # TotalTokens starts at 1, not 0.
    assert_equal(Bm25Calcer(2).total_tokens, 1)


def test_bm25_matches_the_hand_computed_formula() raises:
    # Fitted on the toy corpus: class totals 3 and 1, TotalTokens
    # 1 + 3 + 1 = 5, so meanClassLength = 2.5. For two classes the
    # class-level IDF is log(1.0) = 0 for a term in one class and log(0.2)
    # for a term in both, and BOTH are floored at truncateBorder = 1e-3, so
    # every score here carries the floor rather than a real IDF. That is
    # CatBoost's behavior at two classes and it is worth seeing.
    var bags = _toy_bags()
    var calcer = bm25_fit(bags, _toy_classes(), 2)
    assert_equal(calcer.class_total_tokens[0], 3)
    assert_equal(calcer.class_total_tokens[1], 1)
    assert_equal(calcer.total_tokens, 5)

    var one = calcer.compute(_bag([0], [1]))
    assert_equal(len(one), 2)
    assert_almost_equal(one[0], 0.0015094339622641511, atol=1e-15)
    # Token 0 is absent from class 1, so tf is 0 and the early return fires
    # before any division. Exactly zero, not a small number.
    assert_equal(one[1], 0.0)

    var both = calcer.compute(_bag([0, 1], [1, 1]))
    assert_almost_equal(both[0], 0.0025905150433452322, atol=1e-15)
    assert_almost_equal(both[1], 0.00059701492537313433, atol=1e-15)


def test_bm25_discards_the_in_document_term_frequency() raises:
    # The loop reads Token() and never Count(), so a token occurring five
    # times in a document contributes exactly as much as one occurring once.
    # This is a real CatBoost behavior, not a transcription slip.
    var calcer = bm25_fit(_toy_bags(), _toy_classes(), 2)
    var once = calcer.compute(_bag([0], [1]))
    var five_times = calcer.compute(_bag([0], [5]))
    assert_equal(once[0], five_times[0])
    assert_equal(once[1], five_times[1])


def test_bm25_survives_a_class_with_no_tokens_yet() raises:
    # The first rows of every ordered pass have classes with zero total
    # tokens, and the length normalization divides by that. The tf == 0 early
    # return is the only guard, and it is exactly sufficient: a class with a
    # nonzero term frequency necessarily has a nonzero length.
    var calcer = Bm25Calcer(2)
    calcer.update(0, _bag([0], [1]))
    assert_equal(calcer.class_total_tokens[1], 0)
    var value = calcer.compute(_bag([0], [1]))
    assert_equal(value[1], 0.0)
    assert_true(value[0] > 0.0)


def test_bm25_online_and_frozen_disagree_too() raises:
    var bags = _toy_bags()
    var classes = _toy_classes()
    var perm: List[Int] = [2, 0, 1]
    var online = bm25_online_features(bags, classes, 2, perm)
    var frozen = bm25_apply(bm25_fit(bags, classes, 2), bags)
    assert_equal(len(online), 6)
    assert_equal(len(frozen), 6)
    var differ = False
    for i in range(len(online)):
        if online[i] != frozen[i]:
            differ = True
    assert_true(differ)

    # And the same read-before-write statement as NaiveBayes: flipping the
    # class of the permutation's last row moves nothing.
    var flat: List[Int] = [0, 0, 0]
    var flipped = bm25_online_features(bags, flat, 2, perm)
    for i in range(len(online)):
        assert_equal(online[i], flipped[i])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
