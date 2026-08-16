"""CTR feature combinations: CatBoost's `TProjection` above complexity 1.

Catalog A22, and it extends A19 rather than replacing it. Off by default,
reached by nothing, and it changes no existing default. Verified against
CatBoost `master`, 2026-08-16; every formula below names the file and function
it came from and `docs/design/CATBOOST_CATALOG.md` carries the long-form
reading.

What A19 left, and what this file is
------------------------------------
A19 built the ordered target statistic over ONE categorical column. It left
`ctr_combination_hash` (`CalcHash` verbatim) and `ctr_projection_hash` (the
categorical half of `calc_hashes`) in place, and named four things it did not
build. This file is those four things.

1. The BINARIZED half of `calc_hashes`. A projection is not a tuple of
   categorical columns; it may also carry float splits and one-hot splits,
   each folded in as the 0 or 1 of a threshold test.
2. Candidate ENUMERATION, `greedy_tensor_search.cpp`, which decides which
   combinations are tried at all.
3. `ctr_leaf_count_limit`'s top-K REINDEXING, which is optional for a single
   column and is not optional for a wide combination.
4. The ROUTING rule: a multi-feature projection goes to `TreeCtrs`, a separate
   description list with its own priors and its own binarization, and it
   inherits nothing from `simple_ctr`.

Every CTR entry point in `ctr.mojo` takes an already-hashed integer bucket per
row. `compute_reindex_hash` below produces exactly that, so a combination feeds
`ordered_ctr_borders_binary` and its three siblings with no signature change.
That was A19's design intent and it holds.

A projection has three members, not one
---------------------------------------
`TProjection` (`catboost/private/libs/algo/projection.h:61`):

    TVector<int> CatFeatures;
    TVector<TBinFeature> BinFeatures;      // {FloatFeature, SplitIdx}
    TVector<TOneHotSplit> OneHotFeatures;  // {CatFeatureIdx, Value}

and `CalcHashes` (`index_hash_calcer.cpp:70`) folds all three into one `ui64`
per row with `CalcHash`, in that order. What is folded differs per kind:

    categorical, TRAINING     (ui64)quantizedBin + 1          (:104)
    categorical, FINAL TABLE  (int)originalHashedValue        (:117)
    float split               IsTrueHistogram(bin, splitIdx)  (:138) = bin > splitIdx
    one-hot split             IsTrueOneHotFeature(bin, val)   (:159) = bin == val

The float and one-hot members contribute a BIT, not a value. A projection over
one categorical column and two float splits has a bucket space of
`cardinality * 4`, not `cardinality * bins * bins`. This is the piece A19
flagged as most likely to be missed and it is why `full_projection_length`
counts the whole bin/one-hot set as ONE.

The training fold is `+ 1` and the final-table fold is not, so the two hash
spaces are different. It does not matter, because `compute_reindex_hash` renames
every hash to a dense bucket id on each side separately; it matters a great deal
to anyone comparing a training bucket id against a model bucket id.

`ComputeOnlineCTRs` (`online_ctr.cpp:626`) does not call `CalcHashes` at all for
a single categorical column -- it copies the quantized column straight into the
hash array. So a complexity-1 projection's TRAINING hash is the raw bin and not
`calc_hash(0, bin + 1)`. The induced partition is the same either way once the
reindex has run; only the labels differ.

Enumeration is grown from the tree, not exhaustive
--------------------------------------------------
This is the question that changes the cost by orders of magnitude and the answer
is unambiguous. `AddTreeCtrs` (`greedy_tensor_search.cpp:491`) takes the splits
ALREADY IN THE CURRENT TREE as its bases and adds ONE categorical feature to
each:

    base set = { all the tree's bin and one-hot splits, as ONE projection }
             u { the projection of every CTR already used in the tree }

    for each non-empty base, for each wide categorical feature f:
        proj = base + f
        reject if proj.IsRedundant() or proj.GetFullProjectionLength() > max_ctr_complexity
        reject if already added this level

So a 4-way combination is reachable only if a 3-way combination was itself
chosen as a split earlier in the same tree. It is NOT exhaustive to depth 4 and
the folklore reading is wrong. `SelectFeaturesForScoring` runs this at EVERY
depth (`:1189`, `:997-1008`), not once per tree.

`GetFullProjectionLength` (`projection.h:138`) is
`CatFeatures.size() + (BinFeatures.size() + OneHotFeatures.size() > 0 ? 1 : 0)`,
so `max_ctr_complexity = 4` permits four categorical columns, or three
categorical columns plus arbitrarily many float and one-hot splits.

There is a default override nobody documents. `catboost_options.cpp:1046` sets
`max_ctr_complexity = 1` whenever the user did not set it and the iteration
count is below 200 (`IsSmallIterationCount`, `catboost_options.h:88`). A
comparison harness that fits 100 trees and believes it is measuring complexity 4
is measuring 1. `resolve_max_ctr_complexity` is that rule.

Cost, as a derived bound
------------------------
No number here is measured. This lane has no clock.

With `C` categorical columns wide enough to escape one-hot, depth `D`, `n` rows,
and `P` split candidates per projection (4 at the CPU defaults with a binary
target, `Borders x 3 priors` plus `Counter x 1 prior`):

    projections considered per tree  <=  C * D * (D + 3) / 2

which is `27C` at `D = 6`. Each NEW projection costs at least `n` hash folds,
`n` reindex probes and `P` passes of `n` writes, so at least `6n` element
operations, and produces `4n` bytes of CTR column. At `n = 10^6`, `C = 10`,
`D = 6` that is `1.6 * 10^9` element operations and `1.1 GB` of columns PER
TREE, against roughly `3 * 10^8` for the histogram build of a 50-feature tree.

Stated plainly, because it is the answer to the question that was asked: at 1M
rows the default `max_ctr_complexity = 4` is not affordable here and should not
be this repo's default. Complexity 1 costs `C` projections per tree and fits
comfortably under the histogram budget. Complexity 2 costs `C * (D + 1)` and is
the first setting that has to be paid for rather than absorbed.

The bucket-space bound is the other half. A projection of `k` categorical
columns with cardinalities `m_i` and `b` binarized members spans
`2^b * prod(m_i)` buckets -- `10^12` for four columns of a thousand levels. The
REALIZED count is bounded by `min(n, distinct observed)`, and that bound is the
problem and not the reassurance: as it approaches `n`, every bucket's ordered
prefix holds zero or one row, `ctr_train_bin` returns the pure prior almost
everywhere, and the feature is noise plus a constant. `ctr_leaf_count_limit` is
the only mechanism that bounds it and it is OFF BY DEFAULT (`Max<ui64>()`,
`cat_feature_options.cpp:236`).

Determinism
-----------
Two orderings in CatBoost are not reproducible and both are replaced here.

`AddTreeCtrs` iterates `seenProj`, a `THashSet<TProjection>`, so the order bases
are expanded in is a hash table's bucket layout. Ours sorts by
`TProjection::operator<`'s own key (cat features, then bin splits, then one-hot
splits, lexicographically) and emits in that order. The SET of candidates is
identical; only the order is, and the order is what a tie-breaking candidate
selector would see.

`ComputeReindexHash`'s top-K branch uses `std::nth_element` with a comparator
that only reads the frequency, so two equally frequent hashes at the boundary
are kept or dropped according to the standard library's partition -- which
differs between libstdc++ and libc++ and between versions of each. Ours sorts by
`(count descending, hash ascending)`, a strict total order because the hashes are
distinct, with a stable merge sort. This is the one place where matching CatBoost
would cost the determinism claim.

Nothing here iterates a `Dict`. The one `Dict` is probed and never walked; every
id is assigned in row-scan order or in the explicit sorted order above. That is
`ctr_permutation`'s keyed-sort discipline applied to a different object: the
answer does not depend on how many workers touched it or in what order.

No floating-point value is produced or compared anywhere in this file. Bucket
ids, hashes and counts are integers, so "identical across machines" is a
statement about integer arithmetic rather than a hope about rounding modes.

Not built, deliberately
-----------------------
The Rsm coin flip in `AddSimpleCtrs`/`AddTreeCtrs` (`greedy_tensor_search.cpp:476`,
`:525`), which draws off the run's single advancing RNG and would make a tree's
candidate set depend on every draw any earlier tree made; `SelectCtrsToDropAfterCalc`
(`:580`), which reads `NMemInfo::GetMemInfo().RSS` and so makes CatBoost's
answer depend on what else is running on the machine; `PerFeatureCtrs`, an
option surface rather than a mechanism; the tree-level CTR cache and its
`MAX_ONLINE_CTR_FEATURES = 50` trim (`:42`, `:65`), which is a caching policy
for a trainer that does not exist yet; and the call into A19's four loops, which
needs no new code here, only a caller.

Imports
-------
`.ctr` and `std`, and nothing else. `.ctr` imports only `.rng`, so this module
stays outside the `efb -> binning -> tree_parameters_extra` cycle for the same
reason A19's does.
"""

from .ctr import (
    CTR_BORDERS,
    CTR_COUNTER,
    MAX_CTR_COMPLEXITY_LIMIT,
    CtrParams,
    ctr_combination_hash,
    ctr_feature_count,
    default_priors,
)


# ---------------------------------------------------------------------------
# Defaults and limits, every one read out of CatBoost source
# ---------------------------------------------------------------------------

# `MAX_CTR_COMPLEXITY_LIMIT` is `GetMaxTreeDepth()` (`restrictions.h:14`) and
# lives in `ctr.mojo` because `check_ctr_complexity` there enforces the same
# bound and cannot import from here without closing a cycle.

# `IsSmallIterationCount(n) { return n < 200; }` (`catboost_options.h:88`).
# Below this, an unset `max_ctr_complexity` becomes 1 (`catboost_options.cpp:1046`).
comptime SMALL_ITERATION_COUNT = 200

# `MAX_ONLINE_CTR_FEATURES` (`greedy_tensor_search.cpp:42`). Recorded because it
# is the reason the enumeration above is survivable; the cache it bounds is not
# built here.
comptime MAX_ONLINE_CTR_PROJECTIONS = 50

# `CtrLeafCountLimit("ctr_leaf_count_limit", Max<ui64>())`
# (`cat_feature_options.cpp:236`) -- no limit. Spelled as the largest positive
# `Int` so `top_size > n_rows` is true for any pool.
comptime CTR_LEAF_COUNT_UNLIMITED = 0x7FFFFFFFFFFFFFFF

# `StoreAllSimpleCtrs("store_all_simple_ctr", false)` (`cat_feature_options.cpp:235`).
# It can exempt a SINGLE column from the leaf-count limit and can never exempt a
# combination (`online_ctr.cpp:690`).
comptime DEFAULT_STORE_ALL_SIMPLE_CTRS = False


# ---------------------------------------------------------------------------
# The two binarized members of a projection
# ---------------------------------------------------------------------------


@fieldwise_init
struct BinSplit(Copyable, Movable):
    """`TBinFeature` (`projection.h:25`): a float feature and a split index.

    The projection folds in the RESULT of the threshold test, a 0 or a 1, not
    the feature's value. That is the whole of what A19 flagged as the piece most
    likely to be missed.
    """

    var feature: Int
    var split_idx: Int

    def is_true(self, bin: Int) -> Bool:
        """`IsTrueHistogram(bucket, splitIdx) { return bucket > splitIdx; }`
        (`catboost/libs/model/split.h:12`). Strict `>`, matching every other
        threshold test in CatBoost."""
        return bin > self.split_idx

    def less(self, other: BinSplit) -> Bool:
        """`TBinFeature::operator<` (`projection.h:42`), a tie on
        `(FloatFeature, SplitIdx)`."""
        if self.feature != other.feature:
            return self.feature < other.feature
        return self.split_idx < other.split_idx

    def equals(self, other: BinSplit) -> Bool:
        return (
            self.feature == other.feature
            and self.split_idx == other.split_idx
        )


@fieldwise_init
struct OneHotSplit(Copyable, Movable):
    """`TOneHotSplit` (`catboost/libs/model/split.h`): a categorical feature and
    one of its values, tested for equality."""

    var cat_feature: Int
    var value: Int

    def is_true(self, bin: Int) -> Bool:
        """`IsTrueOneHotFeature(featureValue, splitValue) { return featureValue == splitValue; }`
        (`catboost/libs/model/split.h:16`)."""
        return bin == self.value

    def less(self, other: OneHotSplit) -> Bool:
        if self.cat_feature != other.cat_feature:
            return self.cat_feature < other.cat_feature
        return self.value < other.value

    def equals(self, other: OneHotSplit) -> Bool:
        return (
            self.cat_feature == other.cat_feature
            and self.value == other.value
        )


# ---------------------------------------------------------------------------
# The projection
# ---------------------------------------------------------------------------


struct Projection(Copyable, Movable):
    """`TProjection` (`projection.h:61`), the three vectors and their rules.

    Every insert keeps its vector sorted, exactly as `AddCatFeature`,
    `AddBinFeature` and `AddOneHotFeature` do (`:110-123`), because
    `TProjection::operator==` compares the vectors elementwise and two
    projections built in different orders must compare equal.
    """

    var cat_features: List[Int]
    var bin_splits: List[BinSplit]
    var one_hot_splits: List[OneHotSplit]

    def __init__(out self):
        self.cat_features = List[Int]()
        self.bin_splits = List[BinSplit]()
        self.one_hot_splits = List[OneHotSplit]()

    @staticmethod
    def single_cat(feature: Int) raises -> Projection:
        """The complexity-1 projection, which is what A19 built the loops for.
        """
        var p = Projection()
        p.add_cat_feature(feature)
        return p^

    def add_cat_feature(mut self, feature: Int) raises:
        """`AddCatFeature` (`projection.h:110`): push, then sort."""
        if feature < 0:
            raise Error("categorical feature index must be nonnegative")
        var i = len(self.cat_features)
        self.cat_features.append(feature)
        while i > 0 and self.cat_features[i - 1] > self.cat_features[i]:
            var t = self.cat_features[i - 1]
            self.cat_features[i - 1] = self.cat_features[i]
            self.cat_features[i] = t
            i -= 1

    def add_bin_split(mut self, split: BinSplit) raises:
        """`AddBinFeature` (`projection.h:115`)."""
        if split.feature < 0:
            raise Error("float feature index must be nonnegative")
        if split.split_idx < 0:
            raise Error("split index must be nonnegative")
        var i = len(self.bin_splits)
        self.bin_splits.append(split.copy())
        while i > 0 and split.less(self.bin_splits[i - 1]):
            var t = self.bin_splits[i - 1].copy()
            self.bin_splits[i - 1] = self.bin_splits[i].copy()
            self.bin_splits[i] = t^
            i -= 1

    def add_one_hot_split(mut self, split: OneHotSplit) raises:
        """`AddOneHotFeature` (`projection.h:120`)."""
        if split.cat_feature < 0:
            raise Error("categorical feature index must be nonnegative")
        var i = len(self.one_hot_splits)
        self.one_hot_splits.append(split.copy())
        while i > 0 and split.less(self.one_hot_splits[i - 1]):
            var t = self.one_hot_splits[i - 1].copy()
            self.one_hot_splits[i - 1] = self.one_hot_splits[i].copy()
            self.one_hot_splits[i] = t^
            i -= 1

    def is_empty(self) -> Bool:
        """`IsEmpty` (`projection.h:98`)."""
        return (
            len(self.cat_features) == 0
            and len(self.bin_splits) == 0
            and len(self.one_hot_splits) == 0
        )

    def is_redundant(self) -> Bool:
        """`IsRedundant` (`projection.h:94`): `HasDuplicates` on any of the
        three vectors. Each is sorted here, so the quadratic scan CatBoost
        writes becomes an adjacent-pair scan with the same answer."""
        for i in range(1, len(self.cat_features)):
            if self.cat_features[i] == self.cat_features[i - 1]:
                return True
        for i in range(1, len(self.bin_splits)):
            if self.bin_splits[i].equals(self.bin_splits[i - 1]):
                return True
        for i in range(1, len(self.one_hot_splits)):
            if self.one_hot_splits[i].equals(self.one_hot_splits[i - 1]):
                return True
        return False

    def is_single_cat_feature(self) -> Bool:
        """`IsSingleCatFeature` (`projection.h:102`).

            BinFeatures.empty() && OneHotFeatures.empty() && CatFeatures.ysize() == 1

        This is the predicate `TCtrHelper::GetCtrInfo` branches on, so ONE
        categorical column plus one float split is ALREADY a combination and is
        routed to `TreeCtrs`. See `ctr_info_for_projection`.
        """
        return (
            len(self.bin_splits) == 0
            and len(self.one_hot_splits) == 0
            and len(self.cat_features) == 1
        )

    def has_single_feature(self) -> Bool:
        """`HasSingleFeature` (`projection.h:106`):
        `BinFeatures.ysize() + CatFeatures.ysize() == 1`. Note that it does NOT
        count one-hot members, which is CatBoost's asymmetry and not a typo
        here."""
        return len(self.bin_splits) + len(self.cat_features) == 1

    def full_projection_length(self) -> Int:
        """`GetFullProjectionLength` (`projection.h:138`), verbatim.

            CatFeatures.size() + (BinFeatures.size() + OneHotFeatures.size() > 0 ? 1 : 0)

        The WHOLE bin and one-hot set counts as ONE. So `max_ctr_complexity = 4`
        permits four categorical columns, or three categorical columns plus
        arbitrarily many float and one-hot splits. Getting this wrong in either
        direction changes which combinations are reachable.
        """
        var addition = 0
        if len(self.bin_splits) + len(self.one_hot_splits) > 0:
            addition = 1
        return len(self.cat_features) + addition

    def n_members(self) -> Int:
        """How many values are folded into one row hash, which is the per-row
        cost of `projection_row_hash`."""
        return (
            len(self.cat_features)
            + len(self.bin_splits)
            + len(self.one_hot_splits)
        )

    def equals(self, other: Projection) -> Bool:
        """`TProjection::operator==` (`projection.h:67`), elementwise on all
        three vectors. Sorted inserts are what make it order-independent."""
        if len(self.cat_features) != len(other.cat_features):
            return False
        if len(self.bin_splits) != len(other.bin_splits):
            return False
        if len(self.one_hot_splits) != len(other.one_hot_splits):
            return False
        for i in range(len(self.cat_features)):
            if self.cat_features[i] != other.cat_features[i]:
                return False
        for i in range(len(self.bin_splits)):
            if not self.bin_splits[i].equals(other.bin_splits[i]):
                return False
        for i in range(len(self.one_hot_splits)):
            if not self.one_hot_splits[i].equals(other.one_hot_splits[i]):
                return False
        return True

    def less(self, other: Projection) -> Bool:
        """`TProjection::operator<` (`projection.h:77`), the lexicographic tie
        on `(CatFeatures, BinFeatures, OneHotFeatures)`.

        This is the canonical order the enumeration emits in, and it exists
        because CatBoost's own order is a `THashSet`'s bucket layout, which is
        not reproducible across builds and is exactly the kind of thing this
        repo refuses to depend on.
        """
        var na = len(self.cat_features)
        var nb = len(other.cat_features)
        var n = na if na < nb else nb
        for i in range(n):
            if self.cat_features[i] != other.cat_features[i]:
                return self.cat_features[i] < other.cat_features[i]
        if na != nb:
            return na < nb

        na = len(self.bin_splits)
        nb = len(other.bin_splits)
        n = na if na < nb else nb
        for i in range(n):
            if not self.bin_splits[i].equals(other.bin_splits[i]):
                return self.bin_splits[i].less(other.bin_splits[i])
        if na != nb:
            return na < nb

        na = len(self.one_hot_splits)
        nb = len(other.one_hot_splits)
        n = na if na < nb else nb
        for i in range(n):
            if not self.one_hot_splits[i].equals(other.one_hot_splits[i]):
                return self.one_hot_splits[i].less(other.one_hot_splits[i])
        return na < nb


# ---------------------------------------------------------------------------
# Hashing: the whole of `calc_hashes`, both halves
# ---------------------------------------------------------------------------


def cat_value_train(quantized_bin: Int) -> UInt64:
    """The categorical fold's TRAINING value, from `index_hash_calcer.cpp:104`.


        hashArr[i] = CalcHash(hashArr[i], (ui64)block[i] + 1);

    The perfect-hash bin index PLUS ONE. The `+ 1` is CatBoost's and its effect
    is that bin 0 does not act as the fold's identity element.
    """
    return UInt64(quantized_bin) + 1


def cat_value_final(original_hash: UInt32) -> UInt64:
    """The categorical fold's FINAL-TABLE value, from `index_hash_calcer.cpp:117`.


        hashArr[i] = CalcHash(hashArr[i], (int)origValsView[block[i]]);

    The ORIGINAL hashed categorical value, with no `+ 1`, so the training hash
    space and the model's hash space are different. That is fine because
    `compute_reindex_hash` renames both to dense bucket ids independently, and
    it is a trap for anyone comparing a training bucket id to a model one.

    The `(int)` narrows a `ui32` before the implicit widening to `CalcHash`'s
    `ui64` parameter, so any original hash with the top bit set is
    SIGN-EXTENDED to `0xFFFFFFFF........`. That is CatBoost's arithmetic as
    written; it is stable rather than wrong, and it is not what a reimplementer
    would produce by accident, so it is reproduced explicitly here. The sign
    extension is written as an explicit or-mask rather than a cast chain,
    because a cast chain is exactly the thing a reader would have to trust.
    """
    var v = UInt64(original_hash)
    if v >= UInt64(0x80000000):
        return v | UInt64(0xFFFFFFFF00000000)
    return v


def projection_row_hash(
    proj: Projection,
    cat_values: List[UInt64],
    float_bins: List[Int],
    one_hot_bins: List[Int],
) raises -> UInt64:
    """`CalcHashes` (`index_hash_calcer.cpp:70`) for ONE row.

    The three loops run in CatBoost's order -- categorical, then float splits,
    then one-hot splits -- folding with `CalcHash`, which is A19's
    `ctr_combination_hash`. The seed is 0, matching `ParallelFill(0, ...)` at
    `online_ctr.cpp:656` and `result = 0` at `ctr_calcer.py:10`.

    `cat_values[k]` is the already-folded value for `proj.cat_features[k]` --
    `cat_value_train` on the training side, `cat_value_final` on the model side.
    Taking it already transformed is deliberate and is the same discipline A19
    used for its `categories` argument: the caller owns which hash space it is
    in, and this function owns the fold.

    `float_bins[k]` is the quantized bin of `proj.bin_splits[k].feature` and
    `one_hot_bins[k]` the quantized bin of `proj.one_hot_splits[k].cat_feature`;
    each contributes the 0 or 1 of its threshold test and NOT its value.
    """
    if len(cat_values) != len(proj.cat_features):
        raise Error("cat_values length must match the projection's cat features")
    if len(float_bins) != len(proj.bin_splits):
        raise Error("float_bins length must match the projection's bin splits")
    if len(one_hot_bins) != len(proj.one_hot_splits):
        raise Error(
            "one_hot_bins length must match the projection's one-hot splits"
        )
    var result = UInt64(0)
    for k in range(len(proj.cat_features)):
        result = ctr_combination_hash(result, cat_values[k])
    for k in range(len(proj.bin_splits)):
        var bit = UInt64(1) if proj.bin_splits[k].is_true(
            float_bins[k]
        ) else UInt64(0)
        result = ctr_combination_hash(result, bit)
    for k in range(len(proj.one_hot_splits)):
        var bit = UInt64(1) if proj.one_hot_splits[k].is_true(
            one_hot_bins[k]
        ) else UInt64(0)
        result = ctr_combination_hash(result, bit)
    return result


def projection_hashes(
    proj: Projection,
    cat_columns: List[List[Int]],
    float_columns: List[List[Int]],
    n_rows: Int,
    mut out: List[UInt64],
) raises:
    """`CalcHashes` over a whole learn set, the training hash space.

    `cat_columns[f]` and `float_columns[f]` are indexed by GLOBAL feature id and
    hold quantized bins. One-hot splits read `cat_columns`, matching
    `GetCatFeature(feature.CatFeatureIdx)` at `index_hash_calcer.cpp:148`.

    Cost is `n_rows * proj.n_members()` folds, which is the first term of the
    per-projection bound in the module docstring.
    """
    if n_rows < 0:
        raise Error("n_rows must be nonnegative")
    for k in range(len(proj.cat_features)):
        var f = proj.cat_features[k]
        if f >= len(cat_columns) or len(cat_columns[f]) != n_rows:
            raise Error("cat column missing or wrong length for projection")
    for k in range(len(proj.bin_splits)):
        var f = proj.bin_splits[k].feature
        if f >= len(float_columns) or len(float_columns[f]) != n_rows:
            raise Error("float column missing or wrong length for projection")
    for k in range(len(proj.one_hot_splits)):
        var f = proj.one_hot_splits[k].cat_feature
        if f >= len(cat_columns) or len(cat_columns[f]) != n_rows:
            raise Error("cat column missing or wrong length for one-hot split")

    out.clear()
    out.resize(n_rows, UInt64(0))
    for k in range(len(proj.cat_features)):
        ref col = cat_columns[proj.cat_features[k]]
        for r in range(n_rows):
            out[r] = ctr_combination_hash(out[r], cat_value_train(col[r]))
    for k in range(len(proj.bin_splits)):
        var split = proj.bin_splits[k].copy()
        ref col = float_columns[split.feature]
        for r in range(n_rows):
            var bit = UInt64(1) if split.is_true(col[r]) else UInt64(0)
            out[r] = ctr_combination_hash(out[r], bit)
    for k in range(len(proj.one_hot_splits)):
        var split = proj.one_hot_splits[k].copy()
        ref col = cat_columns[split.cat_feature]
        for r in range(n_rows):
            var bit = UInt64(1) if split.is_true(col[r]) else UInt64(0)
            out[r] = ctr_combination_hash(out[r], bit)


# ---------------------------------------------------------------------------
# `ctr_leaf_count_limit`: the top-K reindexing
# ---------------------------------------------------------------------------


def _sort_by_count_desc_then_hash(
    counts: List[Int], keys: List[UInt64]
) raises -> List[Int]:
    """Distinct-hash ordinals ordered by `(count descending, hash ascending)`.

    A bottom-up stable merge sort, taking the left run on a tie. The hashes are
    distinct by construction, so `(count, hash)` is a STRICT TOTAL ORDER and the
    result cannot depend on the sort's stability, on the machine, or on
    `MOJOTREES_NUM_WORKERS`.

    CatBoost uses `std::nth_element` with a comparator that reads only the
    count, which is not stable and which partitions ties differently between
    libstdc++ and libc++ and between versions of each. That is the one place in
    this file where reproducing CatBoost exactly would forfeit the determinism
    claim, so it is not reproduced.
    """
    var n = len(counts)
    var order = List[Int]()
    for i in range(n):
        order.append(i)
    if n < 2:
        return order^
    var buf = List[Int]()
    buf.resize(n, 0)
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
                var a = order[i]
                var b = order[j]
                var take_right: Bool
                if counts[b] != counts[a]:
                    take_right = counts[b] > counts[a]
                else:
                    take_right = keys[b] < keys[a]
                if take_right:
                    buf[k] = b
                    j += 1
                else:
                    buf[k] = a
                    i += 1
                k += 1
            while i < mid:
                buf[k] = order[i]
                i += 1
                k += 1
            while j < hi:
                buf[k] = order[j]
                j += 1
                k += 1
            lo += 2 * width
        for t in range(n):
            order[t] = buf[t]
        width *= 2
    return order^


def compute_reindex_hash(
    top_size: Int,
    hashes: List[UInt64],
    mut ids: List[Int],
    mut bucket_hash: List[UInt64],
    mut bucket_count: List[Int],
) raises -> Int:
    """`ComputeReindexHash` (`index_hash_calcer.cpp:171`): dense bucket ids,
    bounded by `ctr_leaf_count_limit`.

    Returns the bucket count. `ids[r]` is row `r`'s bucket, which is exactly the
    `categories` argument A19's four ordered loops take -- so a combination
    reaches `ordered_ctr_borders_binary` with no signature change.

    Three branches in the source, and the third is the one that matters.

    1. `topSize > learnSize`: ids in first-seen scan order. This is the DEFAULT
       path, because `ctr_leaf_count_limit` defaults to `Max<ui64>()`.
    2. distinct count `<= topSize`: same partition, ids assigned by iterating
       the hash table. Ours assigns first-seen ids here too, so branches 1 and 2
       agree with each other; CatBoost's do not. The partition is identical and
       only the labels differ.
    3. distinct count `> topSize`: keep the `topSize` most frequent hashes,
       number them `0 .. topSize-1`, and map EVERY OTHER HASH TO `topSize - 1`.

    Read branch 3 twice. The overflow target is `reindexHash.Size() - 1`
    (`:222`), which is the LAST KEPT bucket, not a fresh extra one. The header
    two files up says "map other hash values to value `reindexHash.Size()`"
    (`index_hash_calcer.h:42`), i.e. a fresh bucket. **The comment and the code
    disagree and the code wins.** Every rare combination is therefore merged
    into the least frequent of the kept buckets. It is reproduced here, rather
    than "fixed", because it decides which rows share a statistic and is
    behavior; this docstring exists so nobody corrects it by accident.

    `bucket_hash[b]` is bucket `b`'s representative hash and `bucket_count[b]`
    its learn-row count, both needed by the model side. Under branch 3 the
    overflow bucket's representative is the least frequent KEPT hash, because
    that is the hash the bucket was named after.
    """
    if top_size < 1:
        raise Error("ctr_leaf_count_limit must be positive")
    var n = len(hashes)
    ids.clear()
    ids.resize(n, 0)
    bucket_hash.clear()
    bucket_count.clear()
    if n == 0:
        return 0

    # One pass, first-seen ids. The Dict is PROBED and never iterated, so no
    # answer here depends on a hash table's layout.
    var seen = Dict[UInt64, Int]()
    var keys = List[UInt64]()
    var counts = List[Int]()
    for r in range(n):
        var h = hashes[r]
        var slot: Int
        if h in seen:
            slot = seen[h]
        else:
            slot = len(keys)
            seen[h] = slot
            keys.append(h)
            counts.append(0)
        ids[r] = slot
        counts[slot] += 1

    var distinct = len(keys)
    if distinct <= top_size:
        bucket_hash = keys.copy()
        bucket_count = counts.copy()
        return distinct

    # Branch 3. A total order, not `nth_element`.
    var order = _sort_by_count_desc_then_hash(counts, keys)
    var overflow = top_size - 1
    var remap = List[Int]()
    remap.resize(distinct, overflow)
    bucket_hash.resize(top_size, UInt64(0))
    bucket_count.resize(top_size, 0)
    for b in range(top_size):
        var old = order[b]
        remap[old] = b
        bucket_hash[b] = keys[old]
        bucket_count[b] = counts[old]
    # Everything dropped lands on the last kept bucket, so its count absorbs
    # them. `bucket_count[overflow]` already holds the kept hash's own count.
    for b in range(top_size, distinct):
        bucket_count[overflow] += counts[order[b]]
    for r in range(n):
        ids[r] = remap[ids[r]]
    return top_size


def update_reindex_hash(
    hashes: List[UInt64],
    mut ids: List[Int],
    mut bucket_hash: List[UInt64],
) raises -> Int:
    """`UpdateReindexHash` (`index_hash_calcer.cpp:231`), the test-set pass.

    An unseen hash gets a NEW bucket appended in scan order; a seen one gets its
    learn bucket. `ComputeOnlineCTRs` calls this once per test set after
    `ComputeReindexHash` has run over the learn rows (`online_ctr.cpp:707`).

    One consequence worth stating because it is easy to miss. After a branch-3
    truncation the dropped hashes are GONE from the table (`reindexHash.MakeEmpty()`
    at `:214` keeps only the `topSize` survivors), so a test row whose hash
    matches a DROPPED learn hash does not land in the overflow bucket -- it gets
    a brand new bucket of its own with a learn count of zero, and therefore the
    pure prior. That is CatBoost's behavior as written and it is reproduced.
    """
    var n = len(hashes)
    ids.clear()
    ids.resize(n, 0)
    var seen = Dict[UInt64, Int]()
    for b in range(len(bucket_hash)):
        seen[bucket_hash[b]] = b
    for r in range(n):
        var h = hashes[r]
        var slot: Int
        if h in seen:
            slot = seen[h]
        else:
            slot = len(bucket_hash)
            seen[h] = slot
            bucket_hash.append(h)
        ids[r] = slot
    return len(bucket_hash)


def resolve_ctr_leaf_count_limit(
    ctr_leaf_count_limit: Int,
    proj: Projection,
    store_all_simple_ctrs: Bool,
) raises -> Int:
    """The `topSize` resolution, `online_ctr.cpp:689-692`, verbatim.


        ui64 topSize = catFeatureParams.CtrLeafCountLimit;
        if (proj.IsSingleCatFeature() && catFeatureParams.StoreAllSimpleCtrs) {
            topSize = Max<ui64>();
        }

    So `store_all_simple_ctr` can exempt a SINGLE categorical column from the
    limit and can NEVER exempt a combination. This is the source-level statement
    of A19's claim that the top-K reindex is optional for a single column and
    not optional for a wide one.
    """
    if ctr_leaf_count_limit < 1:
        raise Error("ctr_leaf_count_limit must be positive")
    if proj.is_single_cat_feature() and store_all_simple_ctrs:
        return CTR_LEAF_COUNT_UNLIMITED
    return ctr_leaf_count_limit


def projection_bucket_space_bound(
    cardinalities: List[Int], n_binarized_members: Int, cap: Int
) raises -> Int:
    """`approxBucketsCount` (`online_ctr.cpp:680-687`), plus the factor CatBoost
    drops.

        size_t approxBucketsCount = 1;
        for (auto cf : proj.CatFeatures) {
            approxBucketsCount *= uniqueValuesCount(cf);
            if (approxBucketsCount > learnSampleCount) break;
        }
        rehashHashVal->MakeEmpty(Min(learnSampleCount, approxBucketsCount));

    CatBoost's loop multiplies only the CATEGORICAL cardinalities and ignores
    the bin and one-hot members, each of which contributes a factor of 2. That
    understates the space by `2^b`; it costs CatBoost only a hash-table
    reservation, so it is a sizing hint rather than a bug, but the honest bound
    includes it and this function does.

    `cap` is the learn row count. The early break is reproduced, so this cannot
    overflow: the product stops the moment it exceeds the cap.

    This is the DERIVED BOUND the catalog quotes. Four categorical columns of a
    thousand levels each span `10^12` buckets; the realized count is bounded by
    the row count, and that bound is the problem rather than the reassurance --
    as it approaches `n`, every bucket's ordered prefix holds zero or one row
    and `ctr_train_bin` returns the pure prior almost everywhere.
    """
    if cap < 0:
        raise Error("cap must be nonnegative")
    if n_binarized_members < 0:
        raise Error("n_binarized_members must be nonnegative")
    var product = 1
    for i in range(len(cardinalities)):
        if cardinalities[i] < 1:
            raise Error("a cardinality must be positive")
        product *= cardinalities[i]
        if product > cap:
            return cap
    for _ in range(n_binarized_members):
        product *= 2
        if product > cap:
            return cap
    return product


# ---------------------------------------------------------------------------
# `max_ctr_complexity`
# ---------------------------------------------------------------------------


def check_max_ctr_complexity(max_ctr_complexity: Int) raises:
    """`TCatFeatureParams::Validate` (`cat_feature_options.cpp:266-271`).


        const ui32 ctrComplexityLimit = GetMaxTreeDepth();      // 16
        CB_ENSURE(MaxTensorComplexity.Get() < ctrComplexityLimit, ...);

    This is the bound that replaced A19's `check_ctr_complexity`'s refusal of
    anything above 1. That refusal was the guard A19 left for this lane to
    delete, and deleting it is the whole of the change to `ctr.mojo`.
    """
    if max_ctr_complexity < 1:
        raise Error("max_ctr_complexity must be positive")
    if max_ctr_complexity >= MAX_CTR_COMPLEXITY_LIMIT:
        raise Error(
            "max_ctr_complexity must be below 16: CatBoost bounds it by"
            " GetMaxTreeDepth() at cat_feature_options.cpp:269"
        )


def resolve_max_ctr_complexity(
    max_ctr_complexity: Int, was_set_by_user: Bool, iteration_count: Int
) raises -> Int:
    """The override nobody documents, `catboost_options.cpp:1046`.


        if (MaxTensorComplexity.NotSet() && IsSmallIterationCount(IterationCount)) {
            MaxTensorComplexity = 1;
        }

    with `IsSmallIterationCount(n) { return n < 200; }` (`catboost_options.h:88`).

    So **a default CatBoost fit of fewer than 200 iterations builds no
    combinations at all.** Any comparison harness that fits 100 trees and
    believes it is measuring `max_ctr_complexity = 4` is measuring 1, and any
    port that hard-codes 4 will do several times the work CatBoost did and then
    report a speed ratio against it.
    """
    if iteration_count < 0:
        raise Error("iteration_count must be nonnegative")
    check_max_ctr_complexity(max_ctr_complexity)
    if not was_set_by_user and iteration_count < SMALL_ITERATION_COUNT:
        return 1
    return max_ctr_complexity


# ---------------------------------------------------------------------------
# Candidate enumeration
# ---------------------------------------------------------------------------


def _sort_and_dedupe(projections: List[Projection]) raises -> List[Projection]:
    """Canonical order, duplicates removed.

    The order is `Projection.less`, which is `TProjection::operator<`. Selection
    sort over an ordinal list, which is quadratic and is the right shape here:
    the input is one level's candidate set, bounded by `(depth + 1) *
    n_categorical` and therefore tens to low hundreds, and a quadratic pass over
    that is cheaper than the allocations a merge sort of a non-trivial element
    type would cost. If a caller ever hands this thousands of projections the
    complexity cap has already failed and the sort is not the problem.
    """
    var n = len(projections)
    var order = List[Int]()
    for i in range(n):
        order.append(i)
    for i in range(n):
        var best = i
        for j in range(i + 1, n):
            if projections[order[j]].less(projections[order[best]]):
                best = j
        var t = order[i]
        order[i] = order[best]
        order[best] = t
    var out = List[Projection]()
    for t in range(n):
        var p = projections[order[t]].copy()
        if len(out) > 0 and out[len(out) - 1].equals(p):
            continue
        out.append(p^)
    return out^


def simple_ctr_projections(
    cat_feature_ids: List[Int],
    unique_value_counts: List[Int],
    one_hot_max_size: Int,
) raises -> List[Projection]:
    """`AddSimpleCtrs` (`greedy_tensor_search.cpp:457`), minus the Rsm draw.

    One complexity-1 projection per categorical column WIDE ENOUGH to escape
    one-hot:

        if (GetUniqueValuesCounts(catFeatureIdx).OnLearnOnly <= oneHotMaxSize) { return; }

    (`:469`). That test is the coupling between catalog A16 and A19 -- a narrow
    column gets a one-hot split and no CTR, a wide one gets a CTR and no one-hot
    split -- and it is also what `permutation_is_needed` keys on.

    The Rsm coin flip at `:476` is NOT here. It draws off the run's single
    advancing RNG, so CatBoost's candidate set for a given tree depends on every
    draw any earlier tree made; reproducing that would forfeit the determinism
    this repo requires. Feature sampling here already has its own keyed-stream
    mechanism and a sampler is the caller's business.
    """
    if one_hot_max_size < 0:
        raise Error("one_hot_max_size must be nonnegative")
    if len(cat_feature_ids) != len(unique_value_counts):
        raise Error("cat_feature_ids and unique_value_counts must be the same length")
    var out = List[Projection]()
    for i in range(len(cat_feature_ids)):
        if unique_value_counts[i] < 1:
            raise Error("a unique value count must be positive")
        if unique_value_counts[i] <= one_hot_max_size:
            continue
        out.append(Projection.single_cat(cat_feature_ids[i]))
    return _sort_and_dedupe(out)


def tree_base_projections(
    tree_bin_splits: List[BinSplit],
    tree_one_hot_splits: List[OneHotSplit],
    tree_ctr_projections: List[Projection],
) raises -> List[Projection]:
    """`AddTreeCtrs`' `seenProj` (`greedy_tensor_search.cpp:503-514`).

        TProjection binAndOneHotFeaturesTree;
        binAndOneHotFeaturesTree.BinFeatures    = currentTree.GetBinFeatures();
        binAndOneHotFeaturesTree.OneHotFeatures = currentTree.GetOneHotFeatures();
        seenProj.insert(binAndOneHotFeaturesTree);
        for (const auto& ctr : currentTree.GetUsedCtrs()) { seenProj.insert(ctr.Projection); }

    Note what this is NOT. **All** of the tree's float and one-hot splits go into
    ONE projection. There is no enumeration over subsets of the tree's splits,
    so the base count at depth `d` is at most `d + 1` rather than `2^d`, and
    that single fact is what keeps the enumeration polynomial.

    Empty bases are dropped, matching `if (baseProj.IsEmpty()) continue;` at
    `:518` -- so at depth 0, where the tree has no splits and no CTRs, this
    returns nothing and only `simple_ctr_projections` contributes.
    """
    var bases = List[Projection]()
    var blob = Projection()
    for i in range(len(tree_bin_splits)):
        blob.add_bin_split(tree_bin_splits[i])
    for i in range(len(tree_one_hot_splits)):
        blob.add_one_hot_split(tree_one_hot_splits[i])
    if not blob.is_empty():
        bases.append(blob^)
    for i in range(len(tree_ctr_projections)):
        if tree_ctr_projections[i].is_empty():
            continue
        bases.append(tree_ctr_projections[i].copy())
    return _sort_and_dedupe(bases)


def grow_tree_ctr_projections(
    bases: List[Projection],
    cat_feature_ids: List[Int],
    unique_value_counts: List[Int],
    one_hot_max_size: Int,
    max_ctr_complexity: Int,
) raises -> List[Projection]:
    """`AddTreeCtrs`' body (`greedy_tensor_search.cpp:517-551`), minus the Rsm
    draw.

        TProjection proj = baseProj;
        proj.AddCatFeature(f);
        if (proj.IsRedundant() || proj.GetFullProjectionLength() > MaxTensorComplexity) return;
        if (addedProjHash.contains(proj)) return;

    ONE categorical feature is added per step. A four-way combination is
    therefore reachable only if a three-way combination was itself chosen as a
    split earlier in the same tree; the enumeration is GREEDY and grown from the
    tree, not exhaustive to depth 4. That distinction is the difference between
    `C * D * (D + 3) / 2` projections per tree and `C choose 4` of them.

    One-hot-sized columns are skipped here too (`:523-527`), on the same
    `unique <= one_hot_max_size` test `simple_ctr_projections` uses.

    Deduplication is by projection VALUE, matching `addedProjHash`, and the
    result is in `Projection.less` order rather than a hash table's bucket
    order. The set is CatBoost's; the order is ours and is reproducible.
    """
    check_max_ctr_complexity(max_ctr_complexity)
    if one_hot_max_size < 0:
        raise Error("one_hot_max_size must be nonnegative")
    if len(cat_feature_ids) != len(unique_value_counts):
        raise Error("cat_feature_ids and unique_value_counts must be the same length")
    var out = List[Projection]()
    for b in range(len(bases)):
        if bases[b].is_empty():
            continue
        for i in range(len(cat_feature_ids)):
            if unique_value_counts[i] < 1:
                raise Error("a unique value count must be positive")
            if unique_value_counts[i] <= one_hot_max_size:
                continue
            var proj = bases[b].copy()
            proj.add_cat_feature(cat_feature_ids[i])
            if proj.is_redundant():
                continue
            if proj.full_projection_length() > max_ctr_complexity:
                continue
            out.append(proj^)
    return _sort_and_dedupe(out)


def ctr_candidates_per_projection(
    descriptions: List[CtrParams], n_target_classes: Int
) raises -> Int:
    """`AddCtrsToCandList` (`greedy_tensor_search.cpp:400-429`), counted.

        for ctrIdx: for border in [0, GetTargetBorderCount): for prior in priors:
            emit one TSplitCandidate

    so the count is the sum over descriptions of `target_border_count * priors`,
    which is A19's `ctr_feature_count` summed over a list. At the CPU defaults
    with a binary target it is `Borders: 1 x 3` plus `Counter: 1 x 1` = **4**.

    That 4 is the `P` in the module docstring's cost bound and it is the reason
    CTRs are expensive: the arithmetic is eight lines, the multiplicity is not.
    """
    var total = 0
    for i in range(len(descriptions)):
        total += ctr_feature_count(
            descriptions[i].ctr_type,
            n_target_classes,
            descriptions[i].n_priors(),
        )
    return total


def ctr_projection_count_bound(
    n_wide_cat_features: Int, depth: Int
) raises -> Int:
    """The DERIVED BOUND on projections considered per tree, computed rather
    than asserted.

    At depth `d` there are at most `d + 1` non-empty bases
    (`tree_base_projections`: one bin/one-hot blob plus at most `d` used CTRs),
    so `grow_tree_ctr_projections` considers at most `(d + 1) * C` and
    `simple_ctr_projections` a further `C`. Summing `d = 0 .. depth-1`:

        sum (d + 2) * C  =  C * depth * (depth + 3) / 2

    `27C` at depth 6, `44C` at depth 8. Redundancy rejection and the complexity
    cap only reduce it. Multiply by `ctr_candidates_per_projection` for the
    split-candidate count and by at least `6 * n_rows` element operations for
    the work.

    This is a bound and not a measurement. Nothing in this lane was timed.
    """
    if n_wide_cat_features < 0:
        raise Error("n_wide_cat_features must be nonnegative")
    if depth < 0:
        raise Error("depth must be nonnegative")
    return n_wide_cat_features * depth * (depth + 3) // 2


# ---------------------------------------------------------------------------
# Routing: `SimpleCtrs` versus `TreeCtrs`
# ---------------------------------------------------------------------------


struct CtrRouting(Copyable, Movable):
    """`TCtrHelper`'s two description lists (`ctr_helper.h:75-77`).

    `SimpleCtrs` comes from the `simple_ctr` option and `TreeCtrs` from
    `combinations_ctr` (`ctr_helper.cpp:68-69`), and they are two SEPARATE
    lists, each with its own priors, its own `ctr_binarization` and its own
    `target_binarization`, each passed through `MakeCtrInfo` independently.

    CatBoost warns about exactly this, twice (`catboost_options.cpp:455-460`):

        "Change of simpleCtr will not affect combinations ctrs."
        "Change of combinations ctrs will not affect simple ctrs"

    `ctr_routing_warning` carries those strings verbatim.

    `PerFeatureCtrs`, which `GetCtrInfo` consults ahead of `SimpleCtrs` for a
    single column, is an option surface rather than a mechanism and is not built.
    """

    var simple: List[CtrParams]
    var combination: List[CtrParams]

    def __init__(out self):
        self.simple = List[CtrParams]()
        self.combination = List[CtrParams]()

    @staticmethod
    def catboost_cpu_defaults() raises -> CtrRouting:
        """`SetCtrDefaults`' `default:` arm (`catboost_options.cpp:449-452`).

            defaultSimpleCtrs = {TCtrDescription(Borders, GetDefaultPriors(Borders)),
                                 CreateDefaultCounter(SimpleCtr)};
            defaultTreeCtrs   = {TCtrDescription(Borders, GetDefaultPriors(Borders)),
                                 CreateDefaultCounter(TreeCtr)};

        The two lists are built INDEPENDENTLY here, exactly as they are there,
        so that a caller mutating one cannot reach the other. On the CPU they
        come out with identical content, because `CreateDefaultCounter` ignores
        its `EProjectionType` argument on the CPU (`:394-395`). On the GPU they
        genuinely differ -- `FeatureFreq` with `MinEntropy` binarization for
        simple, `FeatureFreq` with `Median` for combinations (`:398-414`) -- so
        an implementation that shares one list is correct at the CPU defaults
        and silently wrong the moment a user sets `simple_ctr`.

        This is the PairLogit arm's opposite: for `PairLogit` and
        `PairLogitPairwise` both lists are `Counter` only (`:443-447`). That arm
        belongs to whichever lane owns the pairwise objectives.
        """
        var r = CtrRouting()
        r.simple.append(CtrParams.enable(CTR_BORDERS, default_priors(CTR_BORDERS)))
        r.simple.append(CtrParams.enable(CTR_COUNTER, default_priors(CTR_COUNTER)))
        r.combination.append(
            CtrParams.enable(CTR_BORDERS, default_priors(CTR_BORDERS))
        )
        r.combination.append(
            CtrParams.enable(CTR_COUNTER, default_priors(CTR_COUNTER))
        )
        return r^

    def validate(self) raises:
        if len(self.simple) == 0:
            raise Error("ctr routing needs at least one simple description")
        if len(self.combination) == 0:
            raise Error(
                "ctr routing needs at least one combination description:"
                " combinations do not inherit simple_ctr, so an empty"
                " combinations list means no combination CTR is ever built"
            )
        for i in range(len(self.simple)):
            self.simple[i].validate()
        for i in range(len(self.combination)):
            self.combination[i].validate()


def ctr_info_for_projection(
    routing: CtrRouting, proj: Projection
) raises -> List[CtrParams]:
    """`TCtrHelper::GetCtrInfo` (`ctr_helper.h:54`), verbatim minus `PerFeatureCtrs`.


        if (projection.IsSingleCatFeature()) {
            if (PerFeatureCtrs.contains(featureId)) return PerFeatureCtrs.at(featureId);
            else return SimpleCtrs;
        }
        return TreeCtrs;

    and `IsSingleCatFeature()` needs `BinFeatures.empty() && OneHotFeatures.empty()
    && CatFeatures.size() == 1`. So **one categorical column plus one float split
    is already a `TreeCtrs` projection**, not a simple one. That is the trap
    this function exists to close: "is it a combination" is not "does it have
    more than one categorical column".
    """
    if proj.is_empty():
        raise Error("an empty projection has no ctr description")
    routing.validate()
    if proj.is_single_cat_feature():
        return routing.simple.copy()
    return routing.combination.copy()


def ctr_routing_warning(
    simple_was_set: Bool, combination_was_set: Bool
) -> String:
    """CatBoost's two warnings (`catboost_options.cpp:455-460`), verbatim, and
    the empty string when neither fires.

    This WARNS rather than raising, because CatBoost warns and because nothing
    is being silently ignored -- the setting applies, in full, to exactly one of
    the two lists. The defect this repo refuses is a setting that does nothing;
    a setting that does less than a user expected is a documentation problem and
    the documentation is this string.
    """
    if simple_was_set and not combination_was_set:
        return String("Change of simpleCtr will not affect combinations ctrs.")
    if combination_was_set and not simple_was_set:
        return String("Change of combinations ctrs will not affect simple ctrs")
    return String("")


# ---------------------------------------------------------------------------
# The honest "unreached" refusal
# ---------------------------------------------------------------------------


def check_ctr_combination_trainer_support(
    max_ctr_complexity: Int, enabled: Bool
) raises:
    """Refuses an enabled combination bundle at a trainer boundary.

    A19's `check_ctr_trainer_support` says the same thing about CTRs at all: no
    trainer appends CTR columns to a design matrix yet. This one is narrower and
    outlives it -- when the wiring lane deletes A19's guard it will wire the
    complexity-1 path first, and an enabled `max_ctr_complexity > 1` must keep
    refusing until the enumeration above is actually driven by `tree.mojo`'s
    grow loop, which is a different lane's file.

    Deleting this is the second step of the wiring lane, not a side effect of
    the first.
    """
    check_max_ctr_complexity(max_ctr_complexity)
    if enabled and max_ctr_complexity > 1:
        raise Error(
            "ctr feature combinations are built but unreached: no grow loop"
            " calls grow_tree_ctr_projections, so max_ctr_complexity above 1"
            " would compute nothing (catalog A22)"
        )
