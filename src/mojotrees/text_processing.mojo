"""CatBoost's `text_processing`: the tokenizer, the dictionary, and the
digitizer that turns a raw text column into a bag of token ids.

This module is the front half of CatBoost's text pipeline. It produces token
bags; `text_features.mojo` turns bags into numeric columns; and those columns
go to `binning.fit_bins` like any other numeric feature. No trainer, no
binner and no split rule changes anywhere for either half.

Nothing here is on by default: no existing entry point calls this module and
no existing module imports it.

Imports
-------
**This module imports nothing from the package**, deliberately, for the same
reason `cegb.mojo` does not: the package already carries one real import
cycle (`efb -> binning -> tree_parameters_extra`) and a new leaf module has
no business risking a second. `text_features.mojo` imports this file and
nothing else from the package, so the pair is a leaf of the import graph.

Verified against CatBoost `master` (August 2026); see
`docs/design/CATBOOST_CATALOG.md`, the A17 note, for the citation list and
for every place this file diverges. The short form:

- `library/cpp/text_processing/tokenizer/options.h` -- `TTokenizerOptions`
  and its defaults: `Lowercasing=false`, `Delimiter=" "`, `SplitBySet=false`,
  `SkipEmpty=true`, `SeparatorType=ByDelimiter`.
- `library/cpp/text_processing/tokenizer/tokenizer.cpp::SplitByDelimiter` --
  the default tokenizer, which at those options is
  `StringSplitter(s).SplitByString(" ").SkipEmpty()` and nothing else.
- `library/cpp/text_processing/dictionary/dictionary_builder.cpp` --
  `FinishBuilding` and `Filter`: drop below `OccurrenceLowerBound`, sort by
  (count descending, token ascending), truncate to `MaxDictionarySize`,
  assign ids in that order.
- `catboost/private/libs/options/text_processing_options.h` --
  `DEFAULT_DICTIONARY_BUILDER_OPTIONS = {3, 50000}`. Note these are
  CatBoost's overrides; the dictionary library's own defaults are `{50, -1}`
  and quoting those for CatBoost would be wrong.
- `catboost/private/libs/data_types/text.h::TText` -- a document is a
  token-id-ascending run-length list, which is what fixes the summation
  order in `text_features.mojo` without anyone arranging for it.
- `catboost/private/libs/text_processing/dictionary.cpp::TDictionaryProxy` --
  unknown tokens are skipped (`EUnknownTokenPolicy::Skip`), and
  `GetTopTokens(n)` is `xrange(n)` precisely because ids are frequency-ordered.

Determinism
-----------
There is no random draw in this module, so there is no seeded stream and no
domain constant. The one real hazard is dictionary construction, because
counts have to accumulate in a hash map and **a hash map iterated in
insertion order is not a deterministic ordering**. The map is therefore never
iterated into the result: `_dictionary_order` sorts the surviving keys by
(count descending, key ascending), which is a strict TOTAL order on distinct
keys, so the id assignment does not depend on hash order, on insertion order,
on allocation addresses, or on `MOJOTREES_NUM_WORKERS`. That is also exactly
what CatBoost's `FinishBuilding` does, so the agreement is structural rather
than coincidental.

Everything else is a sequential pass in ascending row index. Nothing in this
module reads a worker count or spawns a task. Tokenizing and digitizing are
per-row pure functions and are trivially parallelizable later without moving
a single bit; dictionary counting is not, and must not be.
"""


# CatBoost's `DEFAULT_DICTIONARY_BUILDER_OPTIONS`
# (`catboost/private/libs/options/text_processing_options.h`), NOT the
# dictionary library's own `{50, -1}` in
# `library/cpp/text_processing/dictionary/options.h`.
comptime DEFAULT_OCCURRENCE_LOWER_BOUND = 3
comptime DEFAULT_MAX_DICTIONARY_SIZE = 50000

# `GetMaxDictionarySize(-1)` is `Max<ui32>()`; -1 is the only non-positive
# value the library accepts and it means unbounded.
comptime MAX_DICTIONARY_SIZE_UNBOUNDED = -1

# The n-gram key separator. CatBoost keys a multigram by a tuple of internal
# token ids and tie-breaks by comparing the grams' TOKEN STRINGS one by one
# (`CompareNGram`). Joining the grams with a byte that is strictly less than
# every byte a token can contain makes plain lexicographic comparison of the
# joined keys identical to that gram-by-gram comparison, which is why this is
# a NUL and not a space: a space cannot appear in a token under the default
# tokenizer, but a caller who changed the delimiter could produce one, and a
# NUL byte cannot arrive from a text column at all.
comptime _NGRAM_SEPARATOR = "\x00"

# `TTokenId::ILLEGAL_TOKEN_ID` has no counterpart here: unknown tokens are
# dropped rather than mapped, so no sentinel id ever enters a bag.
comptime TOKEN_NOT_IN_DICTIONARY = -1


struct TokenizerParams(Copyable, ImplicitlyCopyable, Movable):
    """CatBoost's `TTokenizerOptions`, restricted to the reachable subset.

    Every default here is CatBoost's default, and CatBoost's default
    tokenizer does almost nothing: it splits the raw bytes on the literal
    one-character string `" "`, drops empty pieces, and stops. It does not
    lowercase, strip punctuation, drop numbers, lemmatize or normalize
    Unicode, so `"Cat,"` and `"cat"` are two different tokens at the
    defaults. Anyone surprised by a comparison should read that sentence
    first.

    `lowercasing` is CatBoost's `Lowercasing`, off by default, and is
    ASCII-only here (see `_lowercase_ascii`); CatBoost's `ToLower` is
    Unicode-aware, so a fit with `lowercasing=True` on non-ASCII text is a
    known divergence and is stated rather than hidden.
    """

    var delimiter: String
    var split_by_set: Bool
    var skip_empty: Bool
    var lowercasing: Bool

    def __init__(out self):
        self.delimiter = String(" ")
        self.split_by_set = False
        self.skip_empty = True
        self.lowercasing = False

    def __init__(
        out self,
        var delimiter: String,
        split_by_set: Bool = False,
        skip_empty: Bool = True,
        lowercasing: Bool = False,
    ):
        self.delimiter = delimiter^
        self.split_by_set = split_by_set
        self.skip_empty = skip_empty
        self.lowercasing = lowercasing


struct DictionaryParams(Copyable, ImplicitlyCopyable, Movable):
    """CatBoost's `TDictionaryOptions` + `TDictionaryBuilderOptions`,
    restricted to the reachable subset.

    `gram_order` 1 is the `"Word"` dictionary and 2 is the `"BiGram"`
    dictionary; CatBoost's untouched defaults build **both** and give `BoW`
    both, which is why `gram_order` is a parameter of a dictionary rather
    than of the pipeline.

    `occurrence_lower_bound` and `max_dictionary_size` compose in that order
    and are not interchangeable: the bound DROPS every key below it, and the
    size then TRUNCATES what survived. Raising the size bound cannot recover
    a key the occurrence bound removed.
    """

    var gram_order: Int
    var occurrence_lower_bound: Int
    var max_dictionary_size: Int

    def __init__(out self):
        self.gram_order = 1
        self.occurrence_lower_bound = DEFAULT_OCCURRENCE_LOWER_BOUND
        self.max_dictionary_size = DEFAULT_MAX_DICTIONARY_SIZE

    def __init__(
        out self,
        gram_order: Int,
        occurrence_lower_bound: Int = DEFAULT_OCCURRENCE_LOWER_BOUND,
        max_dictionary_size: Int = DEFAULT_MAX_DICTIONARY_SIZE,
    ):
        self.gram_order = gram_order
        self.occurrence_lower_bound = occurrence_lower_bound
        self.max_dictionary_size = max_dictionary_size


def check_dictionary_params(params: DictionaryParams) raises:
    """Refuse by name rather than silently ignore.

    `gram_order` above 5 is refused because CatBoost's own dispatch stops at
    5 (`TDictionaryBuilder`'s switch ends in `Y_ENSURE(false, "Unsupported
    gram order")`), so there is nothing above it to be compatible with.
    """
    if params.gram_order < 1:
        raise Error("text_processing: gram_order must be at least 1")
    if params.gram_order > 5:
        raise Error(
            "text_processing: gram_order above 5 is refused; CatBoost's own"
            " TDictionaryBuilder dispatch stops at 5"
        )
    if params.occurrence_lower_bound < 1:
        raise Error(
            "text_processing: occurrence_lower_bound must be at least 1"
        )
    if (
        params.max_dictionary_size <= 0
        and params.max_dictionary_size != MAX_DICTIONARY_SIZE_UNBOUNDED
    ):
        raise Error(
            "text_processing: max_dictionary_size must be positive or -1"
            " (unbounded), matching GetMaxDictionarySize"
        )


def _lowercase_ascii(var s: String) -> String:
    """ASCII-only lowercasing.

    CatBoost's `ProcessWordToken` calls `ToLower` on a `TUtf16String`, which
    is Unicode-aware. This is not. `lowercasing` is off in every CatBoost
    default, so the divergence is unreachable at the defaults, but it is a
    real divergence the moment a caller turns it on over non-ASCII text and
    it is recorded here rather than in a footnote.
    """
    var out = String()
    var b = s.as_bytes()
    for i in range(len(b)):
        var c = Int(b[i])
        if c >= 65 and c <= 90:
            out += chr(c + 32)
        else:
            out += chr(c)
    return out^


def tokenize(text: String, params: TokenizerParams) raises -> List[String]:
    """CatBoost's `SplitByDelimiter` at the reachable options.

    `split_by_set` treats each byte of `delimiter` as its own delimiter,
    which is CatBoost's `SplitBySet`; the default `False` splits on the whole
    delimiter string.
    """
    if params.delimiter.byte_length() == 0:
        raise Error("text_processing: delimiter must not be empty")

    var raw = List[String]()
    if params.split_by_set:
        var set_bytes = params.delimiter.as_bytes()
        var current = String()
        var b = text.as_bytes()
        for i in range(len(b)):
            var c = Int(b[i])
            var is_delim = False
            for j in range(len(set_bytes)):
                if Int(set_bytes[j]) == c:
                    is_delim = True
                    break
            if is_delim:
                raw.append(current)
                current = String()
            else:
                current += chr(c)
        raw.append(current)
    else:
        for piece in text.split(params.delimiter):
            raw.append(String(piece))

    var out = List[String]()
    for i in range(len(raw)):
        if params.skip_empty and raw[i].byte_length() == 0:
            continue
        if params.lowercasing:
            out.append(_lowercase_ascii(raw[i]))
        else:
            out.append(raw[i])
    return out^


def ngram_keys(
    tokens: List[String], params: DictionaryParams
) raises -> List[String]:
    """The dictionary keys a single document contributes.

    Unigram (`gram_order == 1`) is CatBoost's
    `TUnigramDictionaryBuilderImpl::AddImpl` at `ETokenLevelType::Word`: one
    key per token occurrence, no end-of-sentence token (the Word-level
    `EEndOfSentenceTokenPolicy` default is `Skip`).

    Multigram is `TMultigramDictionaryBuilderImpl::AddImpl` at `SkipStep=0`:
    a document shorter than `gram_order` contributes nothing at all, and
    otherwise every contiguous window contributes one key. Skip-grams
    (`SkipStep > 0`) are refused by `check_dictionary_params`' caller rather
    than approximated.
    """
    var out = List[String]()
    var n = len(tokens)
    if params.gram_order == 1:
        for i in range(n):
            out.append(tokens[i])
        return out^

    if n < params.gram_order:
        return out^
    for start in range(n - params.gram_order + 1):
        var key = String(tokens[start])
        for g in range(1, params.gram_order):
            key += _NGRAM_SEPARATOR
            key += tokens[start + g]
        out.append(key^)
    return out^


def _dictionary_order(
    keys: List[String], counts: List[Int]
) -> List[Int]:
    """Indices ordered by count DESCENDING, key ASCENDING.

    This is `FinishBuilding`'s comparator verbatim, and it is the single
    reason this module is deterministic. Keys are distinct by construction,
    so the comparator is a strict TOTAL order and the result cannot depend on
    the order the hash map handed its entries over, on insertion order, on
    allocation addresses, or on the worker count. The sort itself is the
    repository's bottom-up merge sort over indices (`metrics._argsort`'s
    shape, re-spelled here rather than imported, because this module imports
    nothing from the package); its stability is belt and braces on top of a
    comparator that already has no ties.
    """
    var n = len(keys)
    var idx = List[Int](capacity=n)
    for i in range(n):
        idx.append(i)
    var buf = List[Int](capacity=n)
    for _ in range(n):
        buf.append(0)

    var width = 1
    while width < n:
        var lo = 0
        while lo < n:
            var mid = lo + width
            if mid > n:
                mid = n
            var hi = lo + 2 * width
            if hi > n:
                hi = n
            var i = lo
            var j = mid
            var k = lo
            while i < mid and j < hi:
                var a = idx[i]
                var b = idx[j]
                var take_left: Bool
                if counts[a] != counts[b]:
                    take_left = counts[a] > counts[b]
                else:
                    take_left = keys[a] <= keys[b]
                if take_left:
                    buf[k] = a
                    i += 1
                else:
                    buf[k] = b
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
        for t in range(n):
            idx[t] = buf[t]
        width *= 2
    return idx^


struct TextBag(Copyable, Movable):
    """One document, as CatBoost's `TText`: token ids ASCENDING and unique,
    with the count of each in the document.

    The ascending order is not decoration. Every accumulation in
    `text_features.mojo` walks a bag in this order, so the Float64 summation
    order for a given document is fixed by the data structure and is the same
    on every machine and at every worker count.
    """

    var token_ids: List[Int]
    var counts: List[Int]

    def __init__(out self):
        self.token_ids = List[Int]()
        self.counts = List[Int]()

    def __init__(out self, var token_ids: List[Int], var counts: List[Int]):
        self.token_ids = token_ids^
        self.counts = counts^

    def distinct_tokens(self) -> Int:
        return len(self.token_ids)

    def total_tokens(self) -> Int:
        """The document length in tokens, counting repeats. This is
        `textLen` in `TMultinomialNaiveBayes::LogProb`."""
        var total = 0
        for i in range(len(self.counts)):
            total += self.counts[i]
        return total


def _bag_from_ids(var ids: List[Int]) -> TextBag:
    """`TText(TVector<ui32>&&)`: sort ascending, then run-length encode."""
    sort(ids)
    var token_ids = List[Int]()
    var counts = List[Int]()
    for i in range(len(ids)):
        if len(token_ids) == 0 or token_ids[len(token_ids) - 1] != ids[i]:
            token_ids.append(ids[i])
            counts.append(1)
        else:
            counts[len(counts) - 1] += 1
    return TextBag(token_ids^, counts^)


struct TextDictionary(Copyable, Movable):
    """A frequency-ordered dictionary, fitted on the learn corpus only.

    Token id order IS frequency order, exactly as in CatBoost, and that is
    what makes "the top `n` tokens" the ids `0 .. n-1` with no separate
    lookup (`TDictionaryProxy::GetTopTokens` is a bare `xrange`). Anything
    that changes the id assignment silently changes what `BoW` selects.
    """

    var keys: List[String]
    var counts: List[Int]
    var index: Dict[String, Int]
    var params: DictionaryParams

    def __init__(
        out self,
        var keys: List[String],
        var counts: List[Int],
        var index: Dict[String, Int],
        params: DictionaryParams,
    ):
        self.keys = keys^
        self.counts = counts^
        self.index = index^
        self.params = params

    def size(self) -> Int:
        return len(self.keys)

    def lookup(self, key: String) raises -> Int:
        """The id of `key`, or `TOKEN_NOT_IN_DICTIONARY`.

        `EUnknownTokenPolicy::Skip` is CatBoost's default at every `Apply`
        overload, so callers drop this rather than mapping it to a sentinel
        id, and no unknown token ever enters a bag.
        """
        if key in self.index:
            return self.index[key]
        return TOKEN_NOT_IN_DICTIONARY

    def apply(self, tokens: List[String]) raises -> TextBag:
        """`TDictionaryProxy::Apply`: n-grams, drop the unknown, sort, RLE."""
        var keys = ngram_keys(tokens, self.params.copy())
        var ids = List[Int]()
        for i in range(len(keys)):
            var id = self.lookup(keys[i])
            if id != TOKEN_NOT_IN_DICTIONARY:
                ids.append(id)
        return _bag_from_ids(ids^)

    def memory_bound_bytes(self) -> Int:
        """A DERIVED BOUND on this dictionary's resident bytes, not a
        measurement.

        Each key is stored twice, once in `keys` and once as the `index`
        map's key, at its byte length plus a per-`String` overhead taken here
        as 16 bytes; the counts and the map's values are 8 bytes each. The
        number is worth stating because `max_dictionary_size = 50000` is what
        keeps it bounded at all: the PRE-filter distinct-key count is
        unbounded in the corpus, and `occurrence_lower_bound = 3` is what
        cuts that first.
        """
        var key_bytes = 0
        for i in range(len(self.keys)):
            key_bytes += self.keys[i].byte_length() + 16
        return 2 * key_bytes + 16 * len(self.keys)


def build_dictionary(
    docs: List[String],
    tokenizer: TokenizerParams,
    params: DictionaryParams,
) raises -> TextDictionary:
    """`CreateDictionary` + `TDictionaryBuilder::FinishBuilding`.

    Fitted on the LEARN corpus only, which is what CatBoost does
    (`CreateDictionaries` runs inside learn quantization; a test pool reuses
    the fitted digitizer). Handing this function the test documents would be
    a leak of a different kind from the one A18 is about, and a caller who
    does it gets a dictionary that could not have been built at fit time.

    The count accumulated per key is TOTAL OCCURRENCES over the corpus, not
    document frequency: `TokenToCount[token] += weight` runs per occurrence
    with `weight = 1` per document.

    The pass over `docs` is strictly sequential in ascending row index. It
    must stay that way: a parallel count would have to merge partial maps,
    and while the final total-order sort would still fix the ids, the
    intermediate counts would be summed in a worker-dependent Float-free
    order that is only accidentally safe. There is no reason to take that
    risk for a build that runs once.
    """
    check_dictionary_params(params)

    var counter = Dict[String, Int]()
    for d in range(len(docs)):
        var tokens = tokenize(docs[d], tokenizer)
        var keys = ngram_keys(tokens, params)
        for i in range(len(keys)):
            var key = keys[i]
            if key in counter:
                counter[key] = counter[key] + 1
            else:
                counter[key] = 1

    # Drop below the occurrence bound FIRST. The map's iteration order is
    # not deterministic and is not relied on: `_dictionary_order` below
    # imposes a total order that erases it.
    var surviving_keys = List[String]()
    var surviving_counts = List[Int]()
    for key in counter:
        var c = counter[key]
        if c < params.occurrence_lower_bound:
            continue
        surviving_keys.append(key)
        surviving_counts.append(c)

    var order = _dictionary_order(surviving_keys, surviving_counts)

    var limit = len(order)
    if params.max_dictionary_size != MAX_DICTIONARY_SIZE_UNBOUNDED:
        if params.max_dictionary_size < limit:
            limit = params.max_dictionary_size

    var keys_out = List[String]()
    var counts_out = List[Int]()
    var index = Dict[String, Int]()
    for i in range(limit):
        var src = order[i]
        index[surviving_keys[src]] = i
        keys_out.append(surviving_keys[src])
        counts_out.append(surviving_counts[src])

    return TextDictionary(keys_out^, counts_out^, index^, params)


def digitize(
    docs: List[String],
    tokenizer: TokenizerParams,
    dictionary: TextDictionary,
) raises -> List[TextBag]:
    """`TTextColumnBuilder`: tokenize each document, then apply the fitted
    dictionary. A pure per-row map; the result for row `r` depends on nothing
    but `docs[r]` and the dictionary."""
    var out = List[TextBag]()
    for d in range(len(docs)):
        var tokens = tokenize(docs[d], tokenizer)
        out.append(dictionary.apply(tokens))
    return out^


def default_dictionaries() -> List[DictionaryParams]:
    """CatBoost's untouched default: a BIGRAM dictionary and a UNIGRAM
    dictionary, in that order, both consumed by `BoW`
    (`TTextProcessingOptions::SetDefault`).

    Two, not one. The single-`"Word"` fallback in
    `SetNotSpecifiedOptionsToDefaults` is a DIFFERENT default, taken only
    when the user supplied a partial `text_processing` block, and confusing
    the two is the easiest way to compare against the wrong CatBoost.
    """
    return [DictionaryParams(2), DictionaryParams(1)]
