"""Native categorical features.

A categorical feature's integer codes carry no order, so splitting them with
an ordinal bin threshold (`bin <= t`) is wrong: it depends on how the codes
happen to be numbered. This module implements LightGBM-style *set* splits
instead. A categorical node holds a set of categories that route left;
everything else routes right. No one-hot expansion of the feature matrix is
involved, so a k-category feature still costs one column and k + 1 bins.

Layout
------
Categorical features get their own binning metadata, entirely separate from
the quantile edges used for numerical features (`binning.fit_bins` computes
no edges for a categorical column). `CategoricalSpec` holds, per feature, the
ascending list of category codes kept at fit time. The i-th kept code of
feature f maps to bin i + 1.

Bin 0 is reserved and never a category. It absorbs, at both fit and predict
time:

- missing values: any negative value, or NaN
- unseen categories: codes not present in the training data
- dropped categories: codes present in training but not kept, because the
  column had more distinct categories than `max_bins - 1`. See "Capacity"
  below: this is a hard ceiling here and is not one in LightGBM.

Default routing
---------------
Bin 0 is never placed in a split's category set, so rows that are missing,
unseen, or dropped always route **right** at a categorical node. This
matches LightGBM's `CategoricalDecision`, which sends any value not found in
the node's bitset (including negatives and NaN) to the right child.

Partition search
----------------
`find_best_categorical_split` follows LightGBM's
`FindBestThresholdCategoricalInner`:

- With at most `max_cat_to_onehot` categories, every category is tried
  one-vs-rest, with the L2 term `lambda_l2`.
- Otherwise categories are ordered by the gradient/Hessian ratio
  `sum_grad / (sum_hess + cat_smooth)` and prefixes of that order are
  accumulated from both ends, up to `max_cat_threshold` categories per side,
  with the L2 term `lambda_l2 + cat_l2`. This is the "sort by gradient
  statistics" heuristic of Fisher (1958): for the second-order objective the
  optimal many-vs-many partition is a prefix of that order.
- Either way the parent's gain shift uses `lambda_l2`, exactly as LightGBM
  does, so categorical and numerical gains are compared on the same footing
  as they are there.
- L1 regularization soft-thresholds every gradient sum here exactly as it
  does in the numerical scan; both go through `gain.mojo`.
- The gain floor (`min_gain_to_split`), the per-feature gain multipliers
  (`feature_contri`), and the CEGB split cost are *not* applied here. They
  are charged once per feature by `split.find_best_split`, on the winning
  partition's gain exactly as on a numerical candidate's, so the two kinds of
  candidate still compete on equal footing.

Intentional differences from LightGBM
-------------------------------------
- Row counts are exact. LightGBM estimates a bin's row count from its
  Hessian sum (`round(hess * num_data / sum_hess)`) because its histograms do
  not carry counts; mojotrees histograms do, so `min_data_in_leaf`,
  `min_data_per_group`, and the `cat_smooth` count filter use exact counts.
- One-vs-rest search is selected on the number of categories, matching the
  documented meaning of `max_cat_to_onehot`, rather than on an internal bin
  count that may or may not include the unknown bin. LightGBM tests
  `num_bin <= max_cat_to_onehot` (`FindBestThresholdCategoricalInner`,
  `src/treelearner/feature_histogram.cpp`), and its `num_bin` counts the
  bin 0 dummy, so at the default of 4 LightGBM one-hots up to 3 categories
  while this module one-hots up to 4. A feature holding exactly
  `max_cat_to_onehot` categories is searched by different algorithms on the
  two sides. That is a real divergence, not a naming nicety.
- **Capacity.** LightGBM's `max_bin` is a *floor* on a categorical
  feature's bin count, not a ceiling. `BinMapper::FindBin`
  (`src/io/bin.cpp`) keeps admitting categories while
  `used_cnt < cut_cnt || num_bin_ < max_bin`, and that disjunction stops
  the loop only once it has both covered 99% of the rows and spent
  `max_bin` bins, so a 500-category column at `max_bin=255` gets 496 bins
  there. This module cannot do that. A bin id is one byte
  (`binning.MAX_BINS` is 256) and a node's category set is a fixed 256-bit
  `CatBitset`, so a feature keeps at most `max_bins - 1` categories and
  every other code falls into bin 0 alongside the missing rows. Past
  roughly 254 distinct categories that is lost resolution rather than a
  different tie-break: `bench/real_data`'s `categorical_missing` scenario
  carries a 500-category column, and this module lumps 246 of its
  categories, near half of the rows, into one bin that can never be
  isolated because bin 0 never joins a split set. Raising the ceiling means
  widening the binned matrix element and the node bitset, which is not a
  change this module can make on its own.
- LightGBM additionally drops categories holding fewer than
  `min_data_in_bin` rows, and stops at 99% coverage even when bins are
  still available. mojotrees drops nothing but the overflow above; rare
  categories are instead handled by the `cat_smooth` filter during split
  search.
- LightGBM's `kEpsilon` (1e-15) Hessian nudges are omitted.
"""

from .gain import leaf_score
from .metrics import _argsort

# The arithmetic below used to be written inline here. It now lives in one
# place so the category partition search and the ordinal scan score, filter,
# and cap through the same formulas; `handoffs/task12_tree_parameters.md`
# asks for exactly this substitution, and no parameter moves with it -- this
# module still owns every categorical hyperparameter.
from .tree_parameters_extra import (
    cat_enters_search,
    cat_partition_gain,
    cat_side_cap,
    cat_sort_key,
)

# A 256-bit set covers every bin id representable in the UInt8 binned
# matrix, so a categorical node's set is a fixed 4-word bitset.
comptime CAT_BITSET_WORDS = 4
comptime CAT_MAX_BINS = 64 * CAT_BITSET_WORDS
comptime CatBitset = SIMD[DType.uint64, CAT_BITSET_WORDS]

# Bin 0: missing, unseen, or dropped. Never a member of a split set.
comptime UNKNOWN_BIN = 0

# Category codes must be representable as Int32 the way LightGBM's
# `static_cast<int>` requires.
comptime _MAX_CATEGORY = 1 << 31


def cat_empty() -> CatBitset:
    return CatBitset(0)


def cat_contains(bitset: CatBitset, bin: Int) -> Bool:
    """Whether `bin` is in the category set."""
    if bin <= 0 or bin >= CAT_MAX_BINS:
        return False
    return ((bitset[bin >> 6] >> UInt64(bin & 63)) & 1) != 0


def cat_add(mut bitset: CatBitset, bin: Int):
    """Add `bin` to the category set."""
    bitset[bin >> 6] |= UInt64(1) << UInt64(bin & 63)


def cat_pool_contains(pool: List[UInt64], offset: Int, bin: Int) -> Bool:
    """Membership test against a flat pool of `CAT_BITSET_WORDS`-word sets,
    as stored on tree nodes."""
    if bin <= 0 or bin >= CAT_MAX_BINS:
        return False
    return (
        (pool[offset + (bin >> 6)] >> UInt64(bin & 63)) & 1
    ) != 0


@fieldwise_init
struct CategoricalParams(Copyable, Movable):
    """LightGBM's categorical hyperparameters, same names and defaults.

    - `max_cat_to_onehot`: use one-vs-rest search at or below this many
      categories.
    - `max_cat_threshold`: cap on the number of categories on one side of a
      many-vs-many split.
    - `cat_smooth`: added to a category's Hessian in the sort key, and the
      minimum row count for a category to enter the sorted search at all.
    - `cat_l2`: extra L2 added to the child gain terms of a many-vs-many
      split.
    - `min_data_per_group`: minimum rows added per accepted step of the
      sorted search, and a floor on the right child's row count.
    """

    var max_cat_to_onehot: Int
    var max_cat_threshold: Int
    var cat_smooth: Float64
    var cat_l2: Float64
    var min_data_per_group: Int

    @staticmethod
    def default() -> CategoricalParams:
        return CategoricalParams(4, 32, 10.0, 10.0, 100)


@fieldwise_init
struct CategoricalSpec(Copyable, Movable):
    """Which features are categorical, and their fitted category tables.

    Feature f's kept category codes are `codes[offsets[f] : offsets[f + 1]]`,
    strictly ascending; the i-th maps to bin i + 1. Numerical features have
    an empty slice. An entirely empty spec means every feature is numerical.
    """

    var is_categorical: List[Bool]
    var codes: List[Int]
    var offsets: List[Int]

    @staticmethod
    def none() -> CategoricalSpec:
        """A spec with no feature information: everything is numerical."""
        return CategoricalSpec(List[Bool](), List[Int](), List[Int]())

    @staticmethod
    def all_numerical(n_features: Int) -> CategoricalSpec:
        """A normalized spec of the right width, all features numerical."""
        var flags = List[Bool](capacity=n_features)
        for _ in range(n_features):
            flags.append(False)
        var offsets = List[Int](capacity=n_features + 1)
        for _ in range(n_features + 1):
            offsets.append(0)
        return CategoricalSpec(flags^, List[Int](), offsets^)

    def is_cat(self, feature: Int) -> Bool:
        if feature < 0 or feature >= len(self.is_categorical):
            return False
        return self.is_categorical[feature]

    def n_categories(self, feature: Int) -> Int:
        """Number of categories kept for `feature` (0 when numerical)."""
        if feature < 0 or feature + 1 >= len(self.offsets):
            return 0
        return self.offsets[feature + 1] - self.offsets[feature]

    def any_categorical(self) -> Bool:
        for f in range(len(self.is_categorical)):
            if self.is_categorical[f]:
                return True
        return False

    def bin_of(self, feature: Int, v: Float64) -> Int:
        """Bin id of a raw value of a categorical feature.

        Returns `UNKNOWN_BIN` for missing values (negative or NaN), for
        categories not kept by the fitted table, and for values outside the
        representable code range. Non-integral values truncate toward zero,
        matching LightGBM's `static_cast<int>`.
        """
        # `not (v >= 0.0)` also rejects NaN.
        if not (v >= 0.0) or v >= Float64(_MAX_CATEGORY):
            return UNKNOWN_BIN
        var code = Int(v)
        var begin = self.offsets[feature]
        var end = self.offsets[feature + 1]
        var lo = begin
        var hi = end
        while lo < hi:
            var mid = (lo + hi) // 2
            if self.codes[mid] < code:
                lo = mid + 1
            else:
                hi = mid
        if lo < end and self.codes[lo] == code:
            return lo - begin + 1
        return UNKNOWN_BIN


def _distinct_codes_and_counts[
    features_origin: ImmOrigin, //
](
    features: Span[Float64, features_origin],
    n_rows: Int,
    feature: Int,
    mut codes: List[Int],
    mut counts: List[Int],
) raises:
    """Ascending distinct non-missing category codes of one column, with
    their row counts. Negative and NaN values are missing and excluded."""
    var col = feature * n_rows
    var present = List[Int]()
    for r in range(n_rows):
        var v = features[col + r]
        if not (v >= 0.0):
            continue
        if v >= Float64(_MAX_CATEGORY):
            raise Error(
                "categorical feature values must be below 2^31; use smaller"
                " integer codes"
            )
        present.append(Int(v))
    sort(present)
    var i = 0
    while i < len(present):
        var j = i
        while j + 1 < len(present) and present[j + 1] == present[i]:
            j += 1
        codes.append(present[i])
        counts.append(j - i + 1)
        i = j + 1


def _keep_most_frequent(
    codes: List[Int], counts: List[Int], keep: Int
) raises -> List[Int]:
    """The `keep` most frequent codes, ties broken toward the smaller code,
    returned in ascending code order."""
    if len(codes) <= keep:
        return codes.copy()
    # `_argsort` is stable and `codes` is ascending, so ordering by negated
    # count breaks ties toward the smaller code deterministically.
    var neg_counts = List[Float64](capacity=len(counts))
    for i in range(len(counts)):
        neg_counts.append(-Float64(counts[i]))
    var order = _argsort(neg_counts)
    var kept = List[Int](capacity=keep)
    for i in range(keep):
        kept.append(codes[order[i]])
    sort(kept)
    return kept^


def fit_categorical_spec[
    features_origin: ImmOrigin, //
](
    features: Span[Float64, features_origin],
    n_rows: Int,
    n_features: Int,
    categorical_features: List[Int],
    max_bins: Int,
) raises -> CategoricalSpec:
    """Fit category tables for the features named in `categorical_features`.

    Indices must be in `[0, n_features)` and distinct. Columns not named stay
    numerical and get no category table. A column with more distinct
    categories than `max_bins - 1` keeps only the most frequent ones; the
    rest fall into the unknown bin.
    """
    if max_bins < 2:
        raise Error("max_bins must be at least 2")
    var flags = List[Bool](capacity=n_features)
    for _ in range(n_features):
        flags.append(False)
    for i in range(len(categorical_features)):
        var f = categorical_features[i]
        if f < 0 or f >= n_features:
            raise Error("categorical feature index out of range")
        if flags[f]:
            raise Error("duplicate categorical feature index")
        flags[f] = True

    var codes = List[Int]()
    var offsets = List[Int](capacity=n_features + 1)
    offsets.append(0)
    for f in range(n_features):
        if flags[f]:
            var raw_codes = List[Int]()
            var raw_counts = List[Int]()
            _distinct_codes_and_counts(
                features, n_rows, f, raw_codes, raw_counts
            )
            var kept = _keep_most_frequent(raw_codes, raw_counts, max_bins - 1)
            for i in range(len(kept)):
                codes.append(kept[i])
        offsets.append(len(codes))
    return CategoricalSpec(flags^, codes^, offsets^)


@fieldwise_init
struct CatSplit(Copyable, Movable):
    """Best category partition found for one feature at one node."""

    var gain: Float64
    var bitset: CatBitset
    var found: Bool

    @staticmethod
    def none() -> CatSplit:
        return CatSplit(0.0, cat_empty(), False)


def _onehot_search(
    grad: List[Float64],
    hess: List[Float64],
    count: List[Int],
    base: Int,
    n_categories: Int,
    total_g: Float64,
    total_h: Float64,
    total_c: Int,
    parent_score: Float64,
    lambda_reg: Float64,
    lambda_l1: Float64,
    min_child_hess: Float64,
    min_data_in_leaf: Int,
) raises -> CatSplit:
    """One-vs-rest over every category, LightGBM's `use_onehot` branch. The
    single category goes left."""
    var best = CatSplit.none()
    for t in range(1, n_categories + 1):
        var left_g = grad[base + t]
        var left_h = hess[base + t]
        var left_c = count[base + t]
        if left_c < min_data_in_leaf or left_h < min_child_hess:
            continue
        var right_c = total_c - left_c
        if right_c < min_data_in_leaf:
            continue
        var right_h = total_h - left_h
        if right_h < min_child_hess:
            continue
        var right_g = total_g - left_g
        # One-vs-rest scores its children with plain lambda_l2, so the shared
        # partition formula is used with no cat_l2.
        var gain = cat_partition_gain(
            left_g,
            left_h,
            right_g,
            right_h,
            lambda_l1,
            lambda_reg,
            0.0,
            parent_score,
        )
        if gain > best.gain:
            var bitset = cat_empty()
            cat_add(bitset, t)
            best = CatSplit(gain, bitset, True)
    return best^


def _sorted_search(
    grad: List[Float64],
    hess: List[Float64],
    count: List[Int],
    base: Int,
    n_categories: Int,
    total_g: Float64,
    total_h: Float64,
    total_c: Int,
    parent_score: Float64,
    lambda_reg: Float64,
    lambda_l1: Float64,
    min_child_hess: Float64,
    min_data_in_leaf: Int,
    cat: CategoricalParams,
) raises -> CatSplit:
    """Many-vs-many over prefixes of the gradient/Hessian ordering, walked
    from both ends, LightGBM's sorted branch."""
    var best = CatSplit.none()

    # Categories with too few rows never enter the search; their rows stay
    # with the right child by default.
    var candidates = List[Int]()
    var keys = List[Float64]()
    for t in range(1, n_categories + 1):
        if not cat_enters_search(count[base + t], cat.cat_smooth):
            continue
        candidates.append(t)
        keys.append(
            cat_sort_key(grad[base + t], hess[base + t], cat.cat_smooth)
        )
    var used = len(candidates)
    if used < 2:
        return best^

    var order = _argsort(keys)
    var sorted_bins = List[Int](capacity=used)
    for i in range(used):
        sorted_bins.append(candidates[order[i]])

    # The children's L2 term is `cat_effective_l2(lambda_reg, cat.cat_l2)`,
    # applied inside `cat_partition_gain` below.
    var max_num_cat = cat_side_cap(used, cat.max_cat_threshold)

    for d in range(2):
        var dir = 1 if d == 0 else -1
        var start_pos = 0 if d == 0 else used - 1
        var pos = start_pos
        var cnt_cur_group = 0
        var left_g = 0.0
        var left_h = 0.0
        var left_c = 0
        var steps = used if used < max_num_cat else max_num_cat
        for i in range(steps):
            var t = sorted_bins[pos]
            pos += dir
            left_g += grad[base + t]
            left_h += hess[base + t]
            left_c += count[base + t]
            cnt_cur_group += count[base + t]

            if left_c < min_data_in_leaf or left_h < min_child_hess:
                continue
            var right_c = total_c - left_c
            if right_c < min_data_in_leaf or right_c < cat.min_data_per_group:
                break
            var right_h = total_h - left_h
            if right_h < min_child_hess:
                break
            if cnt_cur_group < cat.min_data_per_group:
                continue
            cnt_cur_group = 0

            var right_g = total_g - left_g
            var gain = cat_partition_gain(
                left_g,
                left_h,
                right_g,
                right_h,
                lambda_l1,
                lambda_reg,
                cat.cat_l2,
                parent_score,
            )
            if gain > best.gain:
                var bitset = cat_empty()
                var p = start_pos
                for _ in range(i + 1):
                    cat_add(bitset, sorted_bins[p])
                    p += dir
                best = CatSplit(gain, bitset, True)
    return best^


def find_best_categorical_split(
    grad: List[Float64],
    hess: List[Float64],
    count: List[Int],
    base: Int,
    n_categories: Int,
    total_g: Float64,
    total_h: Float64,
    total_c: Int,
    lambda_reg: Float64,
    lambda_l1: Float64,
    min_child_hess: Float64,
    min_data_in_leaf: Int,
    cat: CategoricalParams,
) raises -> CatSplit:
    """Best category partition for one categorical feature at one node.

    `grad`/`hess`/`count` are a node's histogram arrays and `base` is the
    feature's offset into them; bins `1 ..= n_categories` are the categories
    and bin 0 (missing, unseen, dropped) is excluded from every candidate
    set. Totals are over all of the node's bins for this feature. Returns a
    partition only when its gain is positive.
    """
    if n_categories < 2:
        return CatSplit.none()
    # LightGBM computes the parent's gain shift with lambda_l2 even when the
    # children use the larger cat_l2, so this score is the same one the
    # numerical scan subtracts.
    var parent_score = leaf_score(total_g, total_h, lambda_l1, lambda_reg)
    if n_categories <= cat.max_cat_to_onehot:
        return _onehot_search(
            grad,
            hess,
            count,
            base,
            n_categories,
            total_g,
            total_h,
            total_c,
            parent_score,
            lambda_reg,
            lambda_l1,
            min_child_hess,
            min_data_in_leaf,
        )
    return _sorted_search(
        grad,
        hess,
        count,
        base,
        n_categories,
        total_g,
        total_h,
        total_c,
        parent_score,
        lambda_reg,
        lambda_l1,
        min_child_hess,
        min_data_in_leaf,
        cat,
    )
