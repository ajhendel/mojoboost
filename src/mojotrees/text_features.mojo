"""CatBoost's `text_features`: the `BoW`, `NaiveBayes` and `BM25` estimators.

Host-side feature GENERATION. Every function here returns a column-major
`List[Float64]` laid out `out[f * n_rows + r]`, which is exactly what
`binning.fit_bins` and `raw_data.RawData.dense` already read. A caller
concatenates these columns onto its numeric matrix and the existing pipeline
takes it from there. There is no new trainer, no new binner and no new split
rule in this file.

Nothing here is on by default: no existing entry point calls this module and
no existing module imports it.

Imports
-------
Only `.text_processing`, which itself imports nothing from the package. The
pair is a leaf of the import graph and cannot participate in the
`efb -> binning -> tree_parameters_extra` cycle.

THE LEAKAGE POINT, read this before writing any caller
------------------------------------------------------
`BoW` never sees the target. **`NaiveBayes` and `BM25` do**, and CatBoost
prevents the leak the same way it prevents the CTR leak: row `i`'s feature is
computed from a calcer holding the statistics of exactly the rows that
PRECEDE `i` in the fold's learn permutation, and row `i`'s own target enters
the calcer only afterwards. `base_text_feature_estimator.h`:

    for (ui64 line : learnPermutation) {
        Compute(featureCalcer, ds.GetText(line), ...);          // read
        calcerVisitor.Update(target.Classes[line], text, ...);  // then write
    }

The permutation is the FOLD's -- `TFold::InitOnlineEstimatedFeatures` passes
`GetLearnPermutationArray()`, the same array the CTRs use, and it is called
from both fold builders immediately before `InitOnlineCtrs`. **This module
therefore CONSUMES a permutation and never builds one.**
`naive_bayes_online_features` and `bm25_online_features` take it as an
argument and validate it. Building it belongs to the ordered-target-statistic
work; two independent permutation implementations in one package would be a
worse outcome than either failing.

Three calcer states exist and confusing any two is where this goes wrong:

1. Learn rows during training -- prefix state, per fold
   (`*_online_features` here).
2. Test rows during training -- the calcer AFTER the whole permutation, not
   a prefix (`*_fit` then `*_apply` here).
3. Predict time from a saved model -- `MakeFinalFeatureCalcer` walks the
   learn set in NATURAL row order with no permutation and freezes that
   (`*_fit` then `*_apply` here, same functions as 2).

So at predict time no target is read and none is needed: it was consumed at
fit time into the frozen per-class tables. The asymmetry is not that the
feature is unavailable -- it is that the training-time value is a prefix
statistic and the predict-time value is a full-corpus statistic, and they are
systematically different numbers for the same document.

Verified against CatBoost `master` (August 2026); see
`docs/design/CATBOOST_CATALOG.md`, the A18 note, for the citation list, for
the exact formulas, and for the three places CatBoost's BM25 diverges from
textbook BM25 (inverted length normalization, discarded in-document term
frequency, class-level IDF floored at `truncateBorder`).

Determinism
-----------
No random draw, so no seeded stream and no domain constant. Every
accumulation walks a `TextBag`, whose token ids are ascending by
construction, so the Float64 summation order for a document is fixed by the
data structure. The ordered pass is sequential over the caller's permutation
by definition -- it cannot be parallelized without changing what it computes,
which is the point of it. `bow_features` and `*_apply` are per-row pure maps
and are result-invariant under any parallelization, but are sequential here
because nothing has measured a reason to change that.
"""

from std.math import exp, log

from .text_processing import TextBag


# `TBagOfWordsEstimator`: `TopTokensCount("top_tokens_count", 2000)`.
comptime DEFAULT_TOP_TOKENS_COUNT = 2000

# `TMultinomialNaiveBayes::DEFAULT_PRIOR` and `SEEN_TOKENS_PRIOR`.
comptime NAIVE_BAYES_DEFAULT_PRIOR = 0.5
comptime NAIVE_BAYES_SEEN_TOKENS_PRIOR = 1

# `TBM25`'s constructor defaults, which `TBM25Estimator` takes wholesale:
# `truncateBorder = 1e-3`, `k = 1.5`, `b = 0.75`.
comptime BM25_DEFAULT_K = 1.5
comptime BM25_DEFAULT_B = 0.75
comptime BM25_DEFAULT_TRUNCATE_BORDER = 1e-3


def check_classification_target(
    classes: List[Int], num_classes: Int
) raises:
    """`NaiveBayes` and `BM25` are classification-only.

    `CreateEstimators` says so in a comment ("There're no online text
    estimators for regression for now") and
    `TTextProcessingOptions::Validate` refuses a classification-only calcer
    on a regression objective by name. Refusing here rather than producing a
    number from a real-valued target is the same decision.
    """
    if num_classes < 2:
        raise Error(
            "text_features: the target-aware estimators need at least two"
            " classes; they are classification-only in CatBoost and are"
            " refused for regression rather than approximated"
        )
    for i in range(len(classes)):
        if classes[i] < 0 or classes[i] >= num_classes:
            raise Error(
                "text_features: class label out of range [0, num_classes)"
            )


def check_permutation(permutation: List[Int], n_rows: Int) raises:
    """A permutation of `0 .. n_rows - 1`, and nothing else.

    This is the contract with whichever lane owns the ordered permutation.
    An arbitrary index list would still produce numbers, and they would
    silently be neither a prefix statistic nor a full one.
    """
    if len(permutation) != n_rows:
        raise Error(
            "text_features: permutation length does not match the row count"
        )
    var seen = List[Bool](length=n_rows, fill=False)
    for i in range(n_rows):
        var p = permutation[i]
        if p < 0 or p >= n_rows:
            raise Error("text_features: permutation entry out of range")
        if seen[p]:
            raise Error("text_features: permutation entry repeated")
        seen[p] = True


def identity_permutation(n_rows: Int) -> List[Int]:
    """Natural row order.

    This is what `EstimateFeatureCalcer` uses (the predict-time fit), and it
    is what a test uses to exercise the ordered pass before a real
    permutation exists. It is NOT a substitute for one: an ordered pass on
    the identity permutation still refuses to leak, but it gives every row a
    prefix determined by row index, which is a systematic bias CatBoost's
    random permutations exist to remove.
    """
    var out = List[Int](capacity=n_rows)
    for i in range(n_rows):
        out.append(i)
    return out^


# ---------------------------------------------------------------------------
# BoW: target-free, offline.
# ---------------------------------------------------------------------------


def bow_feature_count(
    dictionary_size: Int, top_tokens_count: Int = DEFAULT_TOP_TOKENS_COUNT
) raises -> Int:
    """`TBagOfWordsEstimator`'s constructor: refuse zero, clamp to the
    dictionary size."""
    if top_tokens_count <= 0:
        raise Error(
            "text_features: top_tokens_count must be greater than zero"
        )
    if dictionary_size <= 0:
        raise Error(
            "text_features: dictionary size is 0; check the data or lower"
            " occurrence_lower_bound"
        )
    if top_tokens_count > dictionary_size:
        return dictionary_size
    return top_tokens_count


def bow_features(
    bags: List[TextBag],
    dictionary_size: Int,
    top_tokens_count: Int = DEFAULT_TOP_TOKENS_COUNT,
) raises -> List[Float64]:
    """Binary presence of each of the top tokens, column-major.

    Feature `t` of row `r` is 1 if token id `t` occurs in row `r`, else 0. A
    binary INDICATOR, not a count and not a TF-IDF -- `TBagOfWordsEstimator`
    sets one bit per (token, document) and never reads `Count()`. CatBoost
    hints `UniqueValuesUpperBoundHint = 2` per feature for exactly this
    reason, so these columns cost two bins each after binning.

    "The top `n` tokens" is the ids `0 .. n-1` with no lookup, because
    dictionary ids are assigned in descending count order
    (`text_processing`, A17); `TDictionaryProxy::GetTopTokens` is a bare
    `xrange` for the same reason.

    **The memory here is the reason this is the one estimator whose default
    should not be shipped as-is.** At the default 2000 features the returned
    Float64 matrix is `2000 * n_rows * 8` bytes, which is 16 KB per row: a
    derived bound of 16 GB at a million rows. CatBoost packs these 32 to a
    `ui32` and hands the binner a packed binary feature; this returns plain
    Float64 columns because that is what the existing binning path reads. A
    packed representation is the obvious next step and is deliberately not
    built here.
    """
    var n_features = bow_feature_count(dictionary_size, top_tokens_count)
    var n_rows = len(bags)
    var out = List[Float64](length=n_features * n_rows, fill=0.0)
    for r in range(n_rows):
        for i in range(len(bags[r].token_ids)):
            var t = bags[r].token_ids[i]
            if t < n_features:
                out[t * n_rows + r] = 1.0
    return out^


# ---------------------------------------------------------------------------
# NaiveBayes.
# ---------------------------------------------------------------------------


struct NaiveBayesCalcer(Copyable, Movable):
    """`TMultinomialNaiveBayes` plus its `TNaiveBayesVisitor`.

    The calcer and its visitor are one struct here because CatBoost's split
    between them exists to let the visitor be handed to a template, not
    because the state is separable: `SeenTokens` lives on the visitor and
    `NumSeenTokens` on the calcer, and they are the same number.
    """

    var num_classes: Int
    var class_prior: Float64
    var token_prior: Float64
    var num_seen_tokens: Int
    var class_docs: List[Int]
    var class_total_tokens: List[Int]
    var frequencies: List[Dict[Int, Int]]
    var seen_tokens: Dict[Int, Bool]

    def __init__(
        out self,
        num_classes: Int,
        class_prior: Float64 = NAIVE_BAYES_DEFAULT_PRIOR,
        token_prior: Float64 = NAIVE_BAYES_DEFAULT_PRIOR,
    ):
        self.num_classes = num_classes
        self.class_prior = class_prior
        self.token_prior = token_prior
        self.num_seen_tokens = 0
        self.class_docs = List[Int](length=num_classes, fill=0)
        self.class_total_tokens = List[Int](length=num_classes, fill=0)
        self.frequencies = List[Dict[Int, Int]]()
        for _ in range(num_classes):
            self.frequencies.append(Dict[Int, Int]())
        self.seen_tokens = Dict[Int, Bool]()

    def feature_count(self) -> Int:
        """`BaseFeatureCount(numClasses) = numClasses > 2 ? numClasses : 1`.

        **Binary classification emits ONE feature**, the softmax probability
        of class 0, not two. The default `ActiveFeatureIndices` is
        `[0, BaseFeatureCount)`, so index 1 is simply never emitted for two
        classes.
        """
        if self.num_classes > 2:
            return self.num_classes
        return 1

    def update(mut self, class_id: Int, bag: TextBag) raises:
        """`TNaiveBayesVisitor::Update`."""
        for i in range(len(bag.token_ids)):
            var t = bag.token_ids[i]
            var c = bag.counts[i]
            self.seen_tokens[t] = True
            if t in self.frequencies[class_id]:
                self.frequencies[class_id][t] = (
                    self.frequencies[class_id][t] + c
                )
            else:
                self.frequencies[class_id][t] = c
            self.class_total_tokens[class_id] += c
        self.class_docs[class_id] += 1
        self.num_seen_tokens = len(self.seen_tokens)

    def _log_prob(self, class_id: Int, bag: TextBag) raises -> Float64:
        """`TMultinomialNaiveBayes::LogProb`, including its two quirks.

        `classTokensCount` is a BY-VALUE parameter in CatBoost, so the
        `+= TokenPrior` an unseen token adds is local to the class currently
        being scored: the denominator grows by the number of the document's
        tokens THIS class has never seen, and therefore differs per class for
        the same document. And `NumSeenTokens` is the count of distinct token
        ids seen across ALL classes so far, so in the ordered pass it grows
        with the prefix. Both are reproduced rather than tidied.
        """
        var value = log(Float64(self.class_docs[class_id]) + self.class_prior)
        var denom = Float64(self.class_total_tokens[class_id]) + (
            self.token_prior
            * Float64(self.num_seen_tokens + NAIVE_BAYES_SEEN_TOKENS_PRIOR)
        )
        var text_len: Float64 = 0.0
        for i in range(len(bag.token_ids)):
            var t = bag.token_ids[i]
            var count = Float64(bag.counts[i])
            text_len += count
            var num = self.token_prior
            if t in self.frequencies[class_id]:
                num += Float64(self.frequencies[class_id][t])
            else:
                denom += self.token_prior
            value += count * log(num)
        value -= text_len * log(denom)
        return value

    def compute(self, bag: TextBag) raises -> List[Float64]:
        """Softmax over the per-class log probabilities, then the active
        features. `Softmax` in `text_features/helpers.h` is max-shifted, and
        the shift is reproduced because it changes the bits."""
        var log_probs = List[Float64](capacity=self.num_classes)
        for c in range(self.num_classes):
            log_probs.append(self._log_prob(c, bag))

        var max_value = log_probs[0]
        for c in range(self.num_classes):
            if log_probs[c] > max_value:
                max_value = log_probs[c]
        var total: Float64 = 0.0
        for c in range(self.num_classes):
            log_probs[c] = exp(log_probs[c] - max_value)
            total += log_probs[c]

        var out = List[Float64](capacity=self.feature_count())
        for f in range(self.feature_count()):
            out.append(log_probs[f] / total)
        return out^


def naive_bayes_fit(
    bags: List[TextBag], classes: List[Int], num_classes: Int
) raises -> NaiveBayesCalcer:
    """`EstimateFeatureCalcer`: the whole learn set in NATURAL row order.

    This is calcer state 3 -- what a saved model freezes and what predict
    time uses -- and also state 2, what a test row sees during training. It
    is NOT what a learn row sees during training; for that, see
    `naive_bayes_online_features`.
    """
    check_classification_target(classes, num_classes)
    if len(bags) != len(classes):
        raise Error("text_features: bag count and class count differ")
    var calcer = NaiveBayesCalcer(num_classes)
    for r in range(len(bags)):
        calcer.update(classes[r], bags[r])
    return calcer^


def naive_bayes_apply(
    calcer: NaiveBayesCalcer, bags: List[TextBag]
) raises -> List[Float64]:
    """Apply a fitted calcer to any rows, column-major. No target is read
    and none is needed; this is the predict-time shape."""
    var n_features = calcer.feature_count()
    var n_rows = len(bags)
    var out = List[Float64](length=n_features * n_rows, fill=0.0)
    for r in range(n_rows):
        var values = calcer.compute(bags[r])
        for f in range(n_features):
            out[f * n_rows + r] = values[f]
    return out^


def naive_bayes_online_features(
    bags: List[TextBag],
    classes: List[Int],
    num_classes: Int,
    permutation: List[Int],
) raises -> List[Float64]:
    """THE ORDERED PASS. Read strictly before write, over the caller's
    permutation.

    Row `permutation[0]` is scored against an EMPTY calcer, which is correct
    and is the reason the first rows of a permutation carry almost no
    information: there is nothing behind them. CatBoost accepts that and
    averages it away across several permutations; a single permutation here
    does not, and a caller running one permutation should know it.

    The permutation is not built here. See the module docstring.
    """
    check_classification_target(classes, num_classes)
    if len(bags) != len(classes):
        raise Error("text_features: bag count and class count differ")
    check_permutation(permutation, len(bags))

    var calcer = NaiveBayesCalcer(num_classes)
    var n_features = calcer.feature_count()
    var n_rows = len(bags)
    var out = List[Float64](length=n_features * n_rows, fill=0.0)

    for i in range(n_rows):
        var row = permutation[i]
        var values = calcer.compute(bags[row])
        for f in range(n_features):
            out[f * n_rows + row] = values[f]
        calcer.update(classes[row], bags[row])
    return out^


# ---------------------------------------------------------------------------
# BM25.
# ---------------------------------------------------------------------------


struct Bm25Calcer(Copyable, Movable):
    """`TBM25` plus its `TBM25Visitor`.

    The class, not the document, plays the role of the BM25 document: the
    frequency tables are per class, and a "document length" is a class's
    total token count. CatBoost's own comment says so ("Convert to classical:
    class = document (BoW), query = text").
    """

    var num_classes: Int
    var k: Float64
    var b: Float64
    var truncate_border: Float64
    var total_tokens: Int
    var class_total_tokens: List[Int]
    var frequencies: List[Dict[Int, Int]]
    var inv_class_freq: List[Float64]

    def __init__(
        out self,
        num_classes: Int,
        k: Float64 = BM25_DEFAULT_K,
        b: Float64 = BM25_DEFAULT_B,
        truncate_border: Float64 = BM25_DEFAULT_TRUNCATE_BORDER,
    ):
        self.num_classes = num_classes
        self.k = k
        self.b = b
        self.truncate_border = truncate_border
        # `TotalTokens(1)` in the constructor's initializer list, not 0.
        # It is the numerator of `meanClassLength` and starting it at 1 keeps
        # the very first rows of an ordered pass off a zero mean.
        self.total_tokens = 1
        self.class_total_tokens = List[Int](length=num_classes, fill=0)
        self.frequencies = List[Dict[Int, Int]]()
        for _ in range(num_classes):
            self.frequencies.append(Dict[Int, Int]())
        # `InitTruncatedInvClassFreq`: indexed by the number of classes whose
        # table contains the term, so it has `num_classes + 1` entries.
        self.inv_class_freq = List[Float64](
            length=num_classes + 1, fill=0.0
        )
        for j in range(num_classes + 1):
            var raw = log(
                (Float64(num_classes) - Float64(j) + 0.5)
                / (Float64(j) + 0.5)
            )
            if raw < truncate_border:
                self.inv_class_freq[j] = truncate_border
            else:
                self.inv_class_freq[j] = raw

    def feature_count(self) -> Int:
        """`BaseFeatureCount(numClasses) = numClasses`. Unlike NaiveBayes,
        binary classification emits TWO features here."""
        return self.num_classes

    def update(mut self, class_id: Int, bag: TextBag) raises:
        """`TBM25Visitor::Update`."""
        for i in range(len(bag.token_ids)):
            var t = bag.token_ids[i]
            var c = bag.counts[i]
            if t in self.frequencies[class_id]:
                self.frequencies[class_id][t] = (
                    self.frequencies[class_id][t] + c
                )
            else:
                self.frequencies[class_id][t] = c
            self.class_total_tokens[class_id] += c
            self.total_tokens += c

    def _score(self, term_freq: Float64, class_length: Float64) -> Float64:
        """CatBoost's `Score`, with the length normalization INVERTED
        relative to textbook BM25.

        Textbook BM25 has `b * docLen / avgDocLen`, which penalizes long
        documents. This has `b * meanClassLength / classLength`, which
        REWARDS long classes. That is CatBoost's formula and is reproduced,
        not corrected; correcting it would make the comparison meaningless.

        The `term_freq == 0` early return also has to stay AHEAD of the
        division: a class with no tokens yet has `class_length == 0`, which
        happens on the first rows of every ordered pass. A class with a
        nonzero term frequency necessarily has a nonzero length, so the guard
        is exactly sufficient.
        """
        if term_freq == 0.0:
            return 0.0
        var mean_class_length = (
            Float64(self.total_tokens) / Float64(self.num_classes)
        )
        return (
            term_freq
            * (self.k + 1.0)
            / (
                term_freq
                + self.k
                * (1.0 - self.b + self.b * mean_class_length / class_length)
            )
        )

    def compute(self, bag: TextBag) raises -> List[Float64]:
        """`TBM25::Compute`.

        Note what is NOT read: `bag.counts`. The loop takes each distinct
        token in the document once, so a token occurring five times
        contributes exactly as much as one occurring once. That is CatBoost's
        behavior (`tokenToCount.Token()` is read, `Count()` is not) and is
        the second of the three divergences from textbook BM25.
        """
        var scores = List[Float64](length=self.num_classes, fill=0.0)
        for i in range(len(bag.token_ids)):
            var t = bag.token_ids[i]
            var non_zero = 0
            var term_freq = List[Float64](
                length=self.num_classes, fill=0.0
            )
            for c in range(self.num_classes):
                if t in self.frequencies[c]:
                    term_freq[c] = Float64(self.frequencies[c][t])
                    non_zero += 1
            for c in range(self.num_classes):
                scores[c] += self.inv_class_freq[non_zero] * self._score(
                    term_freq[c], Float64(self.class_total_tokens[c])
                )
        return scores^


def bm25_fit(
    bags: List[TextBag], classes: List[Int], num_classes: Int
) raises -> Bm25Calcer:
    """`EstimateFeatureCalcer` for BM25: the whole learn set in natural row
    order. Calcer states 2 and 3; see the module docstring."""
    check_classification_target(classes, num_classes)
    if len(bags) != len(classes):
        raise Error("text_features: bag count and class count differ")
    var calcer = Bm25Calcer(num_classes)
    for r in range(len(bags)):
        calcer.update(classes[r], bags[r])
    return calcer^


def bm25_apply(
    calcer: Bm25Calcer, bags: List[TextBag]
) raises -> List[Float64]:
    """Apply a fitted BM25 calcer to any rows, column-major. No target is
    read; this is the predict-time shape."""
    var n_features = calcer.feature_count()
    var n_rows = len(bags)
    var out = List[Float64](length=n_features * n_rows, fill=0.0)
    for r in range(n_rows):
        var values = calcer.compute(bags[r])
        for f in range(n_features):
            out[f * n_rows + r] = values[f]
    return out^


def bm25_online_features(
    bags: List[TextBag],
    classes: List[Int],
    num_classes: Int,
    permutation: List[Int],
) raises -> List[Float64]:
    """THE ORDERED PASS for BM25. Read strictly before write.

    Same contract as `naive_bayes_online_features`: the permutation is the
    caller's and is not built here.

    Note that `BM25` is NOT a CatBoost default -- `SetDefault` installs `BoW`
    always and `NaiveBayes` for classification, and nothing turns BM25 on
    unless the user names it. A comparison that enables BM25 is not a
    comparison against CatBoost's defaults.
    """
    check_classification_target(classes, num_classes)
    if len(bags) != len(classes):
        raise Error("text_features: bag count and class count differ")
    check_permutation(permutation, len(bags))

    var calcer = Bm25Calcer(num_classes)
    var n_features = calcer.feature_count()
    var n_rows = len(bags)
    var out = List[Float64](length=n_features * n_rows, fill=0.0)

    for i in range(n_rows):
        var row = permutation[i]
        var values = calcer.compute(bags[row])
        for f in range(n_features):
            out[f * n_rows + row] = values[f]
        calcer.update(classes[row], bags[row])
    return out^


# ---------------------------------------------------------------------------
# The pipeline, for a caller who wants the whole thing.
# ---------------------------------------------------------------------------


def text_column_memory_bound_bytes(
    n_rows: Int, n_features: Int
) -> Int:
    """A DERIVED BOUND on the Float64 matrix a text column produces, not a
    measurement.

    `n_features * n_rows * 8`. Worth calling before enabling `BoW`: at its
    default 2000 features that is 16 KB per row, so a million rows is 16 GB
    and the answer is a packed representation rather than more memory.
    """
    return n_features * n_rows * 8
