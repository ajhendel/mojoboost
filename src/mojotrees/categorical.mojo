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
- One-vs-rest selection now matches LightGBM exactly and used to not.
  LightGBM tests `num_bin <= max_cat_to_onehot`
  (`FindBestThresholdCategoricalInner`,
  `src/treelearner/feature_histogram.cpp:183`) and its `num_bin` counts the
  bin 0 dummy (`src/io/bin.cpp:456-460`), so at the default of 4 it one-hots
  up to 3 real categories. This module compared `n_categories` and so
  one-hotted up to 4, and a feature holding exactly `max_cat_to_onehot`
  categories was searched by a different algorithm on each side. Corrected
  2026-08-16 to `n_categories + 1 <= max_cat_to_onehot`; the default value
  did not move, the quantity compared did.
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
# and cap through the same formulas; `handoffs/task12_tree_parameters.md (deleted, recover with git log --all --diff-filter=D -- handoffs/task12_tree_parameters.md)`
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


comptime ONE_HOT_MAX_SIZE_OFF = -1
"""`CategoricalParams.one_hot_max_size` at `-1` is off: the one-hot decision
is LightGBM's `max_cat_to_onehot`, which is what this file has always used."""

comptime CATBOOST_DEFAULT_ONE_HOT_MAX_SIZE = 2
"""CatBoost's `one_hot_max_size` default, verified from
`catboost/private/libs/options/cat_feature_options.cpp:232`
(`OneHotMaxSize("one_hot_max_size", 2)`). Recorded, and NOT this repo's
default: the default here is off."""


struct CategoricalParams(Copyable, Movable):
    """LightGBM's categorical hyperparameters, same names and defaults, plus
    one CatBoost knob that is off unless asked for.

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
    - `one_hot_max_size`: CatBoost's one-hot threshold, `-1` for off. See
      `find_best_categorical_split` and `docs/design/CATBOOST_CATALOG.md` A16.

    `one_hot_max_size` is a trailing argument with a default, not a sixth
    positional field, so every existing five-argument construction still
    means exactly what it meant.
    """

    var max_cat_to_onehot: Int
    var max_cat_threshold: Int
    var cat_smooth: Float64
    var cat_l2: Float64
    var min_data_per_group: Int
    var one_hot_max_size: Int

    def __init__(
        out self,
        max_cat_to_onehot: Int,
        max_cat_threshold: Int,
        cat_smooth: Float64,
        cat_l2: Float64,
        min_data_per_group: Int,
        one_hot_max_size: Int = ONE_HOT_MAX_SIZE_OFF,
    ):
        self.max_cat_to_onehot = max_cat_to_onehot
        self.max_cat_threshold = max_cat_threshold
        self.cat_smooth = cat_smooth
        self.cat_l2 = cat_l2
        self.min_data_per_group = min_data_per_group
        self.one_hot_max_size = one_hot_max_size

    @staticmethod
    def default() -> CategoricalParams:
        return CategoricalParams(4, 32, 10.0, 10.0, 100)

    def uses_catboost_one_hot(self) -> Bool:
        """Whether the CatBoost threshold is in force rather than
        LightGBM's."""
        return self.one_hot_max_size >= 0

    def check_one_hot(self) raises:
        """Refuse an unusable `one_hot_max_size`.

        CatBoost's own ceiling is `OneHotMaxSizeLimit`, `GetMaxBinCount()` on
        CPU and 256 on GPU
        (`cat_feature_options.cpp:233`, enforced with `<=` at :267). Ours is
        `CAT_MAX_BINS`, because a node's category set is a fixed 256-bit
        bitset and a threshold above that could never be reached anyway.
        """
        if self.one_hot_max_size < ONE_HOT_MAX_SIZE_OFF:
            raise Error(
                "one_hot_max_size must be -1 (off) or non-negative, got ",
                self.one_hot_max_size,
            )
        if self.one_hot_max_size > CAT_MAX_BINS:
            raise Error(
                "one_hot_max_size must be at most ",
                CAT_MAX_BINS,
                ", got ",
                self.one_hot_max_size,
            )


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
        """Whether any feature is DECLARED categorical.

        Declaration and not searchability. A column can carry this flag and be
        outside the pool a split search is offered -- that is exactly what
        `binning.append_ctr_columns` leaves a CatBoost-mode source column in
        once its CTR columns have replaced it. A guard that refuses a shape
        because a *category-partition search* would be reached wants
        `any_searchable_categorical` or `BinnedMatrix.any_usable_categorical`
        instead; this one is right for questions about the DATA, such as
        whether a serialized model needs a category table.
        """
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


def distinct_category_codes[
    features_origin: ImmOrigin, //
](
    features: Span[Float64, features_origin],
    n_rows: Int,
    feature: Int,
) raises -> List[Int]:
    """EVERY distinct non-missing category code of one column, ascending, with
    nothing evicted.

    `fit_categorical_spec` keeps `max_bins - 1` of these and drops the rest
    into `UNKNOWN_BIN`, which is the ceiling this module documents under
    "Capacity". This function is the un-truncated view of the same scan, and
    it exists for the one consumer that must not inherit that ceiling:
    an ordered target statistic (`ctr_columns.mojo`, catalog A19).

    **Why a CTR needs this and cannot use the bins.** A target statistic is
    the mechanism for a column too wide to bin, so computing it from the bins
    would be circular: `binning.ctr_slot_columns` used to do exactly that, and
    because every evicted level shares bin 0 it shares one statistic, the CTR
    carried no information the truncated column did not, and it measured as
    two nulls on 2026-08-16. Sourced from here instead, a 200,000-level column
    yields 200,000 distinguishable statistics, which then bin as one ORDINARY
    NUMERIC column. That is the whole point: the cardinality moves out of the
    bin id, where a byte caps it, and into a real value, where it is not
    capped at all.

    The returned list is the bucket table: code `codes[i]` is bucket `i + 1`,
    and bucket 0 is reserved for missing and for a code unseen at fit time,
    exactly as `bin_of` reserves bin 0. `ctr_columns.CtrTables.bucket_of` is
    the lookup, and is deliberately the only one: a second copy of that binary
    search is a second chance for train and predict to disagree about which
    bucket a raw value belongs to.
    """
    var codes = List[Int]()
    var counts = List[Int]()
    _distinct_codes_and_counts(features, n_rows, feature, codes, counts)
    return codes^


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
    gh: List[Float64],
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
        var left_g = gh[2 * (base + t)]
        var left_h = gh[2 * (base + t) + 1]
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
    gh: List[Float64],
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
            cat_sort_key(
                gh[2 * (base + t)], gh[2 * (base + t) + 1], cat.cat_smooth
            )
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
            # `t` walks the categories in SORTED order, so this is a random
            # bin per step, not a scan: the interleaved pair costs one cache
            # line where two planes cost two.
            left_g += gh[2 * (base + t)]
            left_h += gh[2 * (base + t) + 1]
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


def any_searchable_categorical(
    cats: CategoricalSpec,
    n_features: Int,
    features: List[Int],
    allowed: List[Bool],
) -> Bool:
    """Whether a scan over `features` under `allowed` would reach a
    categorical feature, and so reach `find_best_categorical_split`.

    The admission test is `split.find_best_split`'s own, copied rather than
    approximated: an empty `features` means every feature of the histogram, an
    id outside `[0, n_features)` is skipped, and a non-empty `allowed` admits
    only `allowed[f]`. If the two ever diverge this predicate is the one that
    is wrong, because the scan is the definition.

    **Why a guard wants this and not `CategoricalSpec.any_categorical`.** A
    refusal whose reason is "a category partition search would be reached"
    must fire when that search will be reached. `any_categorical` fires when a
    column was DECLARED categorical, which is a strictly larger set: in
    CatBoost mode a column above `one_hot_max_size` is replaced by its CTR
    columns and dropped from `BinnedMatrix.usable`, so it is never in
    `features` and its partition search never runs, and refusing on it refuses
    a fit for a search that does not happen.

    **The one consequence, stated rather than discovered.** `features` is the
    per-NODE draw, so under `feature_fraction < 1` a categorical column can be
    absent from the root's draw and present at a later node's. A refusal that
    used to arrive before any tree was grown can therefore now arrive part way
    into a fit. That is a change in WHEN the message arrives and not in
    whether it is right: each of these refusals is a statement about the scan
    it guards, and a scan that is not offered a categorical feature computes a
    correct answer. The alternative -- refusing up front on `usable`, which is
    per tree -- would re-introduce the over-refusal for any fit whose draws
    happen never to include the column, which is the defect being removed.

    Callers wanting the early message have the per-tree question available as
    `BinnedMatrix.any_usable_categorical`, which is what
    `tree._check_oblivious_supported` asks; it is a strictly larger set than
    this one and is right there because the oblivious grower cannot search a
    categorical column at ANY node.
    """
    if len(features) == 0:
        for f in range(n_features):
            if len(allowed) > 0 and (f >= len(allowed) or not allowed[f]):
                continue
            if cats.is_cat(f):
                return True
        return False
    for i in range(len(features)):
        var f = features[i]
        if f < 0 or f >= n_features:
            continue
        if len(allowed) > 0 and (f >= len(allowed) or not allowed[f]):
            continue
        if cats.is_cat(f):
            return True
    return False


def find_best_categorical_split(
    gh: List[Float64],
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

    `gh` is a node's interleaved `(gradient, hessian)` histogram plane (cell
    `i` at `[2 * i]` and `[2 * i + 1]`, as in `Histogram`), `count` is its
    count plane, and `base` is the feature's CELL offset into them; bins
    `1 ..= n_categories` are the categories and bin 0 (missing, unseen,
    dropped) is excluded from every candidate
    set. Totals are over all of the node's bins for this feature. Returns a
    partition only when its gain is positive.
    """
    if n_categories < 2:
        return CatSplit.none()
    # LightGBM computes the parent's gain shift with lambda_l2 even when the
    # children use the larger cat_l2, so this score is the same one the
    # numerical scan subtracts.
    var parent_score = leaf_score(total_g, total_h, lambda_l1, lambda_reg)
    # Which threshold decides one-hot. `one_hot_max_size` at its default of
    # -1 is off and this is the LightGBM-named test this function has always
    # run; at >= 0 it is CatBoost's, verified from
    # `catboost/private/libs/algo/greedy_tensor_search.cpp:182`
    # (`if ((onLearnOnlyCount > oneHotMaxSize) || (onLearnOnlyCount <= 1)) return;`).
    # CatBoost's comparison is `<=` on the count of REAL categories seen on
    # the learn set, with no dummy bin in the count, and its `<= 1` guard is
    # the `n_categories < 2` return above.
    #
    # The two thresholds are NOT the same boundary and must not be collapsed
    # onto one comparison. LightGBM's test is `num_bin <= max_cat_to_onehot`
    # (`src/treelearner/feature_histogram.cpp:183`, verified verbatim on
    # 4.7.0.99: `bool use_onehot = meta_->num_bin <=
    # meta_->config->max_cat_to_onehot;`) where `num_bin` is
    # `kept_categories + 1` because `src/io/bin.cpp:456-460` pushes a dummy
    # bin at index 0 first and sets `num_bin_ = 1` before admitting any
    # category. So LightGBM one-hots up to `max_cat_to_onehot - 1` REAL
    # categories: three, at the default of 4.
    #
    # FIXED here on 2026-08-16 by the `wide-categorical-bins` lane, which
    # this comment used to name as the owner of the move. The test below is
    # now `n_categories + 1`, LightGBM's `num_bin`, so a column of exactly
    # `max_cat_to_onehot` categories takes the sorted many-vs-many branch on
    # both sides instead of one-vs-rest here and sorted there. The default
    # value 4 did not move and neither did any other categorical default;
    # what moved is the quantity being compared. Giving the CatBoost boundary
    # its own parameter is what kept the two from being fixed as if they were
    # one. See `docs/design/CATBOOST_CATALOG.md` A16.
    #
    # DIVERGENCE above the threshold: CatBoost sends a wider feature to CTRs
    # (`AddSimpleCtrs` returns early on `<= oneHotMaxSize`, :469), which
    # mojotrees does not have. Here a wider feature falls to the many-vs-many
    # sorted search, so this parameter selects CatBoost's boundary and not
    # CatBoost's other side.
    var one_hot: Bool
    if cat.uses_catboost_one_hot():
        one_hot = n_categories <= cat.one_hot_max_size
    else:
        one_hot = n_categories + 1 <= cat.max_cat_to_onehot
    if one_hot:
        return _onehot_search(
            gh,
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
        gh,
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
