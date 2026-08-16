"""Ordered target statistics: CatBoost's CTRs, the simple projection.

Catalog A19. Off by default, reached by nothing, and it changes no existing
default. Verified against CatBoost `master`, 2026-08-16; every formula below
names the file and function it came from, and `docs/design/CATBOOST_CATALOG.md`
carries the long-form reading.

What the mechanism is
---------------------
A categorical column cannot be compared, so a tree cannot threshold it. CatBoost
replaces it with a *numeric* column whose value at row `i` is a statistic of the
target over the rows that come before `i` in a random permutation. Row `i` never
sees its own target, which is the whole leakage argument, and the first row of a
category sees nothing at all and gets the prior.

    value(i) = (count_in_class(before i) + prior) / (count(before i) + 1)

That ratio is then shifted, scaled and truncated to a bucket index in
`[0, ctr_border_count]`. `CalcCTR` in
`catboost/private/libs/algo/online_ctr.h:128` is the entire arithmetic and it is
eight lines:

    float ctr = (countInClass + prior) / (totalCount + 1);
    return (ctr + shift) / norm * borderCount;      // returns ui8

Three consequences, all of which this module reproduces.

**The training denominator is `totalCount + 1`, not `totalCount + prior_denom`.**
There is no `prior_denom` in the CPU training path. The option exists as the
second element of a prior pair and `ctr_helper.cpp:50` refuses anything but 1
(`"Error: CPU could use only 1 as denom for ctrs currently"`), keeping only
`prior[0]` in `TCtrInfo::Priors`. It survives into the *model* as a real field
because the GPU path can set it. `check_ctr_prior_denom` refuses it here by
name.

**The training value is already a bucket index.** `CalcCTR` returns `ui8`. There
is no separate binarization pass over CTR values, which is why the CPU refuses
anything but uniform CTR binarization (`ctr_helper.cpp:29`). The bucket count is
`ctr_border_count + 1`, which is what `split.cpp`'s `GetBucketCount` returns for
a CTR candidate.

**`shift` and `norm` are the identity at every default prior.**
`CalcNormalization` (`online_ctr.cpp:102`) sets `shift = -min(0, prior)` and
`norm = max(1, prior) - min(0, prior)`; at the default priors 0, 0.5 and 1 that
is `shift = 0, norm = 1`. They exist only to map a user-supplied prior outside
`[0, 1]` back into it.

Train versus predict, which is the asymmetry
--------------------------------------------
These are different formulas, with different result types, over different data,
and getting one of the three wrong is the classic way to break this mechanism.

    training    ((c + prior) / (t + 1) + shift) / norm * borderCount  -> ui8
                c and t are the ONLINE PREFIX in one permutation

    inference   ((c + PriorNum) / (t + PriorDenom) + Shift) * Scale   -> float
                c and t are a STATIC TABLE over the whole learn set

`TModelCtr::Calc` is `catboost/libs/model/online_ctr.h:289`; the static table is
`CalcFinalCtrsImpl` (`online_ctr.cpp:875`), a plain loop with no permutation and
no prefix. The two are reconciled in `split.cpp:78-82` by folding the training
multiply into the inference scale, `Scale = borderCount / norm`, so both produce
the same real number -- and then putting the truncation back at the *threshold*
rather than the value:

    EmulateUi8Rounding(value) { return value + 0.999999f; }   // split.h:512

so a training comparison `bin > BinBorder` on integers becomes an inference
comparison `x > BinBorder + 0.999999f` on the unquantized float. The name says
exactly what it is. `ctr_train_bin`, `ctr_predict_value`, `ctr_predict_scale`
and `ctr_predict_border` are those four pieces, one function each.

A category never seen during training has no bucket, and inference answers
`calc(0, 0)` -- the pure prior (`ctr_calcer.py:35`). Training cannot reach that
case, because the first row of a category *is* the prefix's zero state.

The four CTR types
------------------
Dispatch is `ComputeOnlineCTRs`' final `ExecRange` (`online_ctr.cpp:732`), and
it branches on the type AND on the target class count.

- `Borders` at exactly two classes -> `ordered_ctr_borders_binary`, the default
  path for a binary target and the one that matters.
- `Buckets`, and `Borders` above two classes -> `ordered_ctr_classes`. The only
  difference between the two is `UpdateGoodCount` (`online_ctr.cpp:115`):
  `Buckets` takes the count of *this* class, `Borders` takes the count of every
  class *above* the border. One emits `n_classes` features, the other
  `n_classes - 1`.
- `BinarizedTargetMeanValue` -> `ordered_ctr_mean`, a running mean of the class
  index normalized to `[0, 1]`, one feature.
- `Counter` -> `counter_ctr`, and **it is not ordered at all**. The count is a
  full count over the learn rows, taken once before any row is emitted, and the
  denominator is the *largest category's* count, not the row count
  (`CalcOnlineCTRCounter`, `CountOnlineCTRTotal`). Every row of a category gets
  the same value. `IsPermutationDependentCtrType` says so directly.

At the stock defaults a single categorical column therefore produces
`Borders x 3 priors = 3` features plus `Counter x 1 prior = 1`, so **four**
numeric features, each in 16 buckets. `ctr_feature_count` computes it.

Determinism
-----------
The permutation is the hard part, because it is the one random object here.

`ctr_permutation` gives block `b` the key `splitmix64(stream + b)` and sorts
blocks ascending by `(key, b)`. It is a *keyed sort*, not a shuffle. CatBoost's
`CreateShuffledIndices` is a sequential Fisher-Yates whose answer depends on the
order the draws are consumed in; ours does not depend on consumption order at
all, so it is identical across `MOJOTREES_NUM_WORKERS`, identical across
machines, and could be computed in parallel without changing its answer. It
touches no floating-point value at any point, so "across machines" is a
statement about integer arithmetic rather than a hope about rounding modes.

This differs from `sampling._mvs_stream` and `langevin._langevin_row_stream` in
what it has to produce, not in how it is seeded. Those need one *independent*
draw per row, and `uniform(stream + row)` gives that directly. A permutation
needs a *bijection*, which no per-index draw gives you on its own; sorting by a
per-index key is how you get one back while keeping the per-index property. The
seed derivation is the same three-step house recipe -- mix the seed against a
domain constant, spread the index by `GOLDEN`, mix again -- and there are two
domain constants for the reason `langevin.mojo` has two: a permutation ordinal
and a tree ordinal occupy the same small integers, and sharing one domain would
make permutation 7 and tree 7 draw the same stream.

The online accumulation itself runs in row order, sequentially. CatBoost
parallelizes it (`CalcStatsForEachBlock` + `SumCtrsFromBlocks` +
`CalcQuantizedCtrs`, each block seeded from the exclusive prefix of the blocks
before it) and gets away with it *only because those accumulators are integers*
-- an exact count of earlier rows is the same under any blocking. That escape
does not extend to `BinarizedTargetMeanValue`, whose accumulator is a float sum,
and CatBoost accordingly leaves `CalcOnlineCTRMean` serial. Ours is serial for
all four, which is deterministic for the same reason and stricter than it needs
to be for three of them.

Intentional differences from CatBoost
-------------------------------------
- The permutation is a keyed sort, so for the same seed it is a *different*
  permutation from CatBoost's. Nothing here claims bit-identity with CatBoost.
- Accumulators are `Int` and `Float64`; only `ctr_train_bin` narrows to
  `Float32`, to reproduce the `float` truncation that decides the bucket.
  CatBoost accumulates the mean type's sum in `float` (`TCtrMeanHistory::Sum`).
  Narrowing once at the end is strictly more accurate and moves a bucket index
  only where the exact value sits within one ULP of a bucket edge.
- Output is indexed by original row id, not by position in the permutation.
  CatBoost writes `featureData[docId]` into an array that is itself permuted;
  ours writes back through the permutation so a caller holds one row order.

Not built
---------
`FloatTargetMeanValue`; the GPU-only `FeatureFreq`; `PriorEstimation`;
per-feature CTR descriptions; `ctr_history_unit = Group`; and the `Dynamic`
fold shape, which is ordered boosting (catalog A7) and a different mechanism.

Combinations (`max_ctr_complexity > 1`) are no longer on that list. They live
in `ctr_combinations.mojo` (catalog A30), which holds the binarized half of
`calc_hashes`, the candidate enumeration from `greedy_tensor_search.cpp`,
`ctr_leaf_count_limit`'s top-K reindexing, and the `SimpleCtrs`-versus-
`TreeCtrs` routing rule. It imports this module and produces exactly the
`List[Int]` bucket per row that the four ordered loops below already take, so
a combination reaches them with no signature change -- which is why every entry
point here was written to take an already-hashed bucket rather than a raw
column.

Imports
-------
`.rng` and `std.math`, and nothing else from the package. `binning` imports
`categorical` and `categorical` imports `tree_parameters_extra`; a module that
wants to be importable from either of those must not reach back into them, which
is the rule `cegb.mojo` states and follows. Nothing in this file needs a
`CategoricalSpec`: a CTR entry point takes an already-hashed integer bucket per
row, which is also the input shape a *combination* has, so the combinations lane
does not have to change these signatures.
"""

from std.math import floor

from .rng import GOLDEN, splitmix64


# ---------------------------------------------------------------------------
# CTR types
# ---------------------------------------------------------------------------

# `ECtrType` (`catboost/private/libs/ctr_description/ctr_type.h`), the four the
# CPU trainer reaches. `FloatTargetMeanValue` and the GPU-only `FeatureFreq`
# are named there and not built here.
comptime CTR_BORDERS = 0
comptime CTR_BUCKETS = 1
comptime CTR_BINARIZED_TARGET_MEAN = 2
comptime CTR_COUNTER = 3


# ---------------------------------------------------------------------------
# Defaults, every one of them read out of CatBoost source
# ---------------------------------------------------------------------------

# `TCtrDescription(type, priors)` delegates to
# `TBinarizationOptions(EBorderSelectionType::Uniform, 15)`
# (`cat_feature_options.cpp:169`), so a CTR feature has 15 borders and
# `15 + 1 = 16` buckets.
comptime DEFAULT_CTR_BORDER_COUNT = 15

# `TargetBinarization("target_binarization",
# TBinarizationOptions(EBorderSelectionType::MinEntropy, 1))`
# (`cat_feature_options.cpp:230`). One border, so two target classes, and the
# border is chosen by MinEntropy -- which is border *selection*, catalog A15's
# mechanism, so this module takes borders as an input and does not derive them.
comptime DEFAULT_TARGET_BORDER_COUNT = 1

# `boosting_options.cpp`, `PermutationCount("permutation_count", 4)`. Note that
# only `permutation_count - 1` of them are learning folds; see
# `learning_fold_count`.
comptime DEFAULT_PERMUTATION_COUNT = 4

# `cat_feature_options.cpp:231`, `MaxTensorComplexity("max_ctr_complexity", 4)`.
# The projection itself lives in `ctr_combinations.mojo` (catalog A30); this
# module builds the complexity-1 loops that consume its bucket ids.
comptime DEFAULT_MAX_CTR_COMPLEXITY = 4

# `GetMaxTreeDepth()` (`restrictions.h:14`), the bound
# `TCatFeatureParams::Validate` checks `max_ctr_complexity` against
# (`cat_feature_options.cpp:269-271`). `ctr_combinations.mojo` re-exports it
# rather than redefining it.
comptime MAX_CTR_COMPLEXITY_LIMIT = 16

# `cat_feature_options.cpp:234`,
# `CounterCalcMethod("counter_calc_method", ECounterCalc::SkipTest)`.
comptime COUNTER_CALC_SKIP_TEST = 0
comptime COUNTER_CALC_FULL = 1
comptime DEFAULT_COUNTER_CALC_METHOD = COUNTER_CALC_SKIP_TEST

# The CPU's only legal `prior_denom`, checked at option-load time by
# `ctr_helper.cpp:50` and written unconditionally into the model by
# `split.cpp:79`.
comptime CPU_PRIOR_DENOM = 1.0

# `EmulateUi8Rounding` (`split.h:512`), the constant that turns an integer
# bucket comparison into a float threshold comparison.
comptime _UI8_ROUNDING_EPS = 0.999999

# `CalcHash` (`catboost/libs/model/hash.h:11`). Left here for the combinations
# lane; used by nothing in this file.
comptime _CTR_HASH_MAGIC = UInt64(0x4906BA494954CB65)

# `SIMPLE_CLASSES_COUNT` (`online_ctr.h:49`): the count at which `Borders` takes
# the two-class fast path instead of the general per-class loop.
comptime SIMPLE_CLASSES_COUNT = 2

# CatBoost draws every permutation from the fit's global `random_seed`, whose
# default is 0. mojotrees gives the CTR permutation its own seed so it composes
# with the bagging, feature and bootstrap seeds instead of sharing one counter,
# and 0 is kept as the value so a port of a default CatBoost configuration reads
# the same.
comptime DEFAULT_CTR_SEED = 0

# `FoldPermutationBlockSizeNotSet`: `boosting_options.cpp` constructs
# `PermutationBlockSize("fold_permutation_block", 0)` and `TFoldsCreationParams`
# replaces a 0 with `DefaultFoldPermutationBlockSize(learnSampleCount)`.
comptime PERMUTATION_BLOCK_SIZE_NOT_SET = 0


# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

# Separates the permutation stream from every other seeded stream in the
# package. Bagging, feature sampling, GOSS, MVS, the Bayesian bootstrap and
# Langevin are all keyed by a small integer and all default their seed to a
# small integer, so a caller who sets one seed and turns two of them on would
# otherwise draw the same values twice. `sampling._MVS_DOMAIN` and
# `langevin._LANGEVIN_ROW_DOMAIN` are the same device for the same reason.
comptime _CTR_PERMUTATION_DOMAIN = UInt64(0xC7B0_5EED_9E1D_0C71)

# The per-tree fold draw gets its OWN domain rather than sharing the
# permutation domain with a different index, because both are keyed by
# (seed, small ordinal) and a permutation ordinal and a tree ordinal occupy the
# same small integers. Sharing would make tree 7 pick its fold from the very
# stream that built permutation 7. This is `langevin.mojo`'s two-domain pattern,
# and it is here for the identical reason.
comptime _CTR_FOLD_DOMAIN = UInt64(0xC7B0_F01D_9E1D_0C71)


def _ctr_permutation_stream(seed: Int, permutation_index: Int) -> UInt64:
    """Start of the counter stream for one permutation's block keys.

    Same shape as `sampling._mvs_stream` and `langevin._langevin_row_stream`:
    mix the seed against this module's domain constant, spread the ordinal by
    the golden-ratio increment, mix again. Sign bits are masked off so a
    negative seed is accepted without relying on signed-to-unsigned conversion.

    The result is a *start*, not a running state. Nothing advances it, so block
    `b` reads `stream + b` and its key does not depend on how many blocks were
    keyed before it or on which worker keyed them.
    """
    var h = splitmix64(
        UInt64(seed & 0x7FFFFFFFFFFFFFFF) ^ _CTR_PERMUTATION_DOMAIN
    )
    return splitmix64(
        h ^ (UInt64(permutation_index & 0x7FFFFFFFFFFFFFFF) * GOLDEN)
    )


def _ctr_fold_stream(seed: Int, tree_index: Int) -> UInt64:
    """Start of the counter stream for one tree's fold draw. A second domain,
    so permutation `j` and tree `j` never share a stream."""
    var h = splitmix64(UInt64(seed & 0x7FFFFFFFFFFFFFFF) ^ _CTR_FOLD_DOMAIN)
    return splitmix64(h ^ (UInt64(tree_index & 0x7FFFFFFFFFFFFFFF) * GOLDEN))


# ---------------------------------------------------------------------------
# Permutations
# ---------------------------------------------------------------------------


def default_permutation_block_size(n_rows: Int) -> Int:
    """`DefaultFoldPermutationBlockSize` (`defaults_helper.h:9`), verbatim:
    `Min(256, docCount / 1000 + 1)`.

    So a million-row pool is permuted in blocks of 256 and a ten-thousand-row
    pool in blocks of 11; only below a thousand rows is it a true row shuffle.
    Rows inside one block keep their original relative order, which means a
    row's ordered prefix is NOT a uniform random subset -- rows sharing a block
    always see each other the same way round. That is a cache concession
    CatBoost makes deliberately and it weakens the ordering guarantee the paper
    argues for. It is reproduced rather than fixed because the block size is a
    published, user-settable option.
    """
    var b = n_rows // 1000 + 1
    if b > 256:
        b = 256
    if b < 1:
        b = 1
    return b


def resolve_permutation_block_size(
    permutation_block_size: Int, n_rows: Int, permuted: Bool
) raises -> Int:
    """`TFoldsCreationParams`' three-line resolution (`learn_context.cpp:89`).

    An unset block size becomes `default_permutation_block_size`, and an
    unpermuted fold gets a block size of the whole row count -- which is
    CatBoost's way of saying the permutation is the identity.
    """
    if n_rows < 0:
        raise Error("n_rows must be nonnegative")
    if permutation_block_size < 0:
        raise Error("permutation_block_size must be nonnegative")
    if not permuted:
        if n_rows == 0:
            return 1
        return n_rows
    if permutation_block_size == PERMUTATION_BLOCK_SIZE_NOT_SET:
        return default_permutation_block_size(n_rows)
    return permutation_block_size


def permutation_is_needed(
    has_time: Bool, has_wide_categorical: Bool, ordered_boosting: Bool
) -> Bool:
    """`IsPermutationNeeded` (`learn_context.cpp:38`), for a learning fold.

    Three lines in CatBoost and the first one is the one that matters.

        if (hasTime) { return false; }
        if (hasCtrs)  { return true;  }
        return isOrderedBoosting && !isAveragingFold;

    Two things fall out that surprise people. `has_time` disables the
    permutation ENTIRELY, which is the correct behavior for a genuinely
    time-ordered pool and makes the CTR a prefix statistic in dataset order.
    And CatBoost's `hasCtrs` is not "the user asked for CTRs" -- it is
    `CalcMaxCategoricalFeaturesUniqueValuesCountOnLearn() > OneHotMaxSize`, so
    the mere presence of a categorical column too wide to one-hot turns the
    permutation on, for plain boosting as much as for ordered. That is the
    coupling between catalog A16 and A19 and it runs in the direction people do
    not expect, which is why the argument here is named
    `has_wide_categorical` rather than `has_ctrs`.
    """
    if has_time:
        return False
    if has_wide_categorical:
        return True
    return ordered_boosting


def learning_fold_count(permutation_count: Int, permuted: Bool) raises -> Int:
    """`CountLearningFolds` (`learn_context.cpp:48`), verbatim:
    `isPermutationNeededForLearning ? Max<ui32>(1, permutationCount - 1) : 1`.

    At the default `permutation_count = 4` there are **three** learning folds,
    not four. The fourth is spent on the separate `AveragingFold`
    (`learn_context.cpp:575`), which supplies the leaf values while a learning
    fold supplies the structure. A tree is therefore grown against one
    permutation and valued against another.
    """
    if permutation_count < 1:
        raise Error("permutation_count must be positive")
    if not permuted:
        return 1
    var n = permutation_count - 1
    if n < 1:
        return 1
    return n


def _sort_blocks_by_key(keys: List[UInt64]) raises -> List[Int]:
    """Block ordinals sorted ascending by `(key, ordinal)`.

    A bottom-up merge sort, stable, taking the left run on a tie. The initial
    order is ascending ordinal, so stability is exactly the `(key, ordinal)`
    tiebreak and two blocks that collide on a 64-bit key still order the same
    way on every machine.
    """
    var n = len(keys)
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
                if keys[order[j]] < keys[order[i]]:
                    buf[k] = order[j]
                    j += 1
                else:
                    buf[k] = order[i]
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


def ctr_permutation(
    n_rows: Int, block_size: Int, seed: Int, permutation_index: Int
) raises -> List[Int]:
    """One learning fold's permutation. `perm[position] = row`.

    Permutation 0 is the IDENTITY, matching `shuffle = (foldIdx != 0)` at
    `learn_context.cpp:502` and `:529`. `InitPermutationData`'s unshuffled
    branch builds `std::iota(...)` and says in a comment that it exists only
    because "implementation requires permutation vectors to exist even if they
    are not shuffled". One of the three learning folds is therefore dataset
    order, and callers get that here by asking for index 0.

    Every other index is a BLOCK permutation, matching `NCB::Shuffle`
    (`objects_grouping.cpp:205`): the blocks are reordered and each block's rows
    are emitted in their original relative order.

    The reordering is a keyed sort rather than a shuffle -- block `b` gets the
    key `splitmix64(stream + b)` and blocks sort ascending by `(key, b)`. See
    the module docstring for why, but the short version is that the answer does
    not depend on the order the keys are computed in, so it is identical across
    `MOJOTREES_NUM_WORKERS` and across machines by construction, and no
    floating-point value is involved at any point.
    """
    if n_rows < 0:
        raise Error("n_rows must be nonnegative")
    if block_size < 1:
        raise Error("block_size must be positive")
    if permutation_index < 0:
        raise Error("permutation_index must be nonnegative")
    var perm = List[Int]()
    if n_rows == 0:
        return perm^
    if permutation_index == 0:
        for r in range(n_rows):
            perm.append(r)
        return perm^
    var n_blocks = (n_rows + block_size - 1) // block_size
    var stream = _ctr_permutation_stream(seed, permutation_index)
    var keys = List[UInt64]()
    for b in range(n_blocks):
        keys.append(splitmix64(stream + UInt64(b)))
    var order = _sort_blocks_by_key(keys)
    for t in range(n_blocks):
        var b = order[t]
        var start = b * block_size
        var end = start + block_size
        if end > n_rows:
            end = n_rows
        for r in range(start, end):
            perm.append(r)
    return perm^


def ctr_fold_index(seed: Int, tree_index: Int, n_folds: Int) raises -> Int:
    """Which learning fold a tree's structure is searched against.

    `train.cpp:208` is
    `&ctx->LearnProgress->Folds[ctx->LearnProgress->Rand.GenRand() % foldCount]`
    -- a draw off the run's single advancing RNG, so CatBoost's choice for tree
    `t` depends on every draw any earlier tree made. Ours is keyed by
    `(seed, tree_index)` and depends on nothing else, so a resumed fit, a fit
    with a different sampler enabled, and a fit on a different worker count all
    pick the same fold for the same tree.
    """
    if n_folds < 1:
        raise Error("n_folds must be positive")
    return Int(_ctr_fold_stream(seed, tree_index) % UInt64(n_folds))


# ---------------------------------------------------------------------------
# The target classifier
# ---------------------------------------------------------------------------


def target_class(value: Float64, borders: List[Float64]) -> Int:
    """`TTargetClassifier::GetTargetClass`
    (`catboost/libs/model/target_classifier.h:24`), verbatim.

        int resClass = 0;
        while (resClass < Borders.ysize() && target > Borders[resClass]) {
            ++resClass;
        }
        return resClass;

    Strict `>`, ascending borders. Border *selection* is `MinEntropy` at
    `ctr_target_border_count` borders and belongs to catalog A15, so the borders
    arrive here already chosen.
    """
    var res = 0
    while res < len(borders) and value > borders[res]:
        res += 1
    return res


def target_classes_count(borders: List[Float64]) -> Int:
    """`GetClassesCount`: `Borders.ysize() + 1`."""
    return len(borders) + 1


def ctr_target_border_count(
    ctr_type: Int, n_target_classes: Int
) raises -> Int:
    """`GetTargetBorderCount` (`ctr_helper.h:35`), verbatim.

    `BinarizedTargetMeanValue` and `Counter` emit one feature each whatever the
    class count; `Buckets` emits one per class; `Borders` emits one per class
    boundary.
    """
    if ctr_type == CTR_BINARIZED_TARGET_MEAN or ctr_type == CTR_COUNTER:
        return 1
    if n_target_classes < 1:
        raise Error("n_target_classes must be positive")
    if ctr_type == CTR_BUCKETS:
        return n_target_classes
    if ctr_type == CTR_BORDERS:
        return n_target_classes - 1
    raise Error("unknown ctr type")


def ctr_feature_count(
    ctr_type: Int, n_target_classes: Int, n_priors: Int
) raises -> Int:
    """How many numeric columns one CTR description produces for one
    categorical column.

    `AllocateCtrData(ctrIdx, targetBorderCount, priors.size())`
    (`online_ctr.cpp:741`) allocates exactly this product, and it is the number
    that makes CTRs expensive rather than the arithmetic. At the stock defaults
    a single categorical column produces `Borders x 3 priors x 1 border = 3`
    plus `Counter x 1 prior = 1`, so four columns each in 16 buckets.
    """
    if n_priors < 1:
        raise Error("n_priors must be positive")
    return ctr_target_border_count(ctr_type, n_target_classes) * n_priors


def default_priors(ctr_type: Int) raises -> List[Float64]:
    """`GetDefaultPriors` (`cat_feature_options.cpp:118`).

    The source returns pairs -- `{{0, 1}, {0.5, 1}, {1, 1}}` for the target
    types and `{{0.0, 1}}` for `Counter` -- and `MakeCtrInfo` keeps only the
    numerators after checking that every denominator is 1. These are those
    numerators.
    """
    if (
        ctr_type == CTR_BORDERS
        or ctr_type == CTR_BUCKETS
        or ctr_type == CTR_BINARIZED_TARGET_MEAN
    ):
        var p = List[Float64]()
        p.append(0.0)
        p.append(0.5)
        p.append(1.0)
        return p^
    if ctr_type == CTR_COUNTER:
        var c = List[Float64]()
        c.append(0.0)
        return c^
    raise Error("unknown ctr type")


# ---------------------------------------------------------------------------
# The arithmetic: four functions, one per piece
# ---------------------------------------------------------------------------


def ctr_shift(prior: Float64) -> Float64:
    """`CalcNormalization`'s shift (`online_ctr.cpp:109`): `-Min(0.0f, prior)`.
    """
    if prior < 0.0:
        return -prior
    return 0.0


def ctr_norm(prior: Float64) -> Float64:
    """`CalcNormalization`'s norm (`online_ctr.cpp:110`):
    `Max(1.0f, prior) - Min(0.0f, prior)`."""
    var right = 1.0
    if prior > 1.0:
        right = prior
    var left = 0.0
    if prior < 0.0:
        left = prior
    return right - left


def calc_normalization(
    priors: List[Float64],
    mut shifts: List[Float64],
    mut norms: List[Float64],
):
    """`CalcNormalization` (`online_ctr.cpp:102`), the whole vector at once.

    At every default prior (0, 0.5, 1) this is `shift = 0, norm = 1`, so the
    normalization is the identity for a stock fit and bites only for a
    user-supplied prior outside `[0, 1]`.
    """
    shifts.clear()
    norms.clear()
    for i in range(len(priors)):
        shifts.append(ctr_shift(priors[i]))
        norms.append(ctr_norm(priors[i]))


def ctr_train_bin(
    count_in_class: Float64,
    total_count: Int,
    prior: Float64,
    shift: Float64,
    norm: Float64,
    ctr_border_count: Int,
) raises -> Int:
    """`CalcCTR` (`online_ctr.h:128`): the training-time value, which is
    already a bucket index.

        float ctr = (countInClass + prior) / (totalCount + 1);
        return (ctr + shift) / norm * borderCount;      // ui8

    The denominator is `totalCount + 1` and NOT `totalCount + prior_denom`;
    there is no `prior_denom` in the CPU training path at all. See
    `check_ctr_prior_denom`.

    The arithmetic is done in `Float32` on purpose, because CatBoost's `float`
    truncation is what picks the bucket and reproducing the bucket means
    reproducing the width. The accumulators that feed it are `Int` and
    `Float64`, which is the one place this module is deliberately more accurate
    than its reference.

    The result is clamped to `[0, ctr_border_count]`. That is a guard rather
    than a behavior: `(ctr + shift) / norm` provably lands in `[0, 1]` for any
    prior, since `ctr` lies between `min(0, prior)` and `max(1, prior)`, so the
    scaled value lies in `[0, ctr_border_count]` before truncation and the
    clamp can only catch a rounding artifact at an endpoint.
    """
    if ctr_border_count < 1:
        raise Error("ctr_border_count must be positive")
    if total_count < 0:
        raise Error("total_count must be nonnegative")
    if norm <= 0.0:
        raise Error("norm must be positive")
    var ctr = (Float32(count_in_class) + Float32(prior)) / Float32(
        total_count + 1
    )
    var scaled = (ctr + Float32(shift)) / Float32(norm) * Float32(
        ctr_border_count
    )
    var b = Int(floor(Float64(scaled)))
    if b < 0:
        b = 0
    if b > ctr_border_count:
        b = ctr_border_count
    return b


def ctr_bucket_count(ctr_border_count: Int) raises -> Int:
    """`GetBucketCount`'s CTR arm (`split.cpp`):
    `return splitCandidate.Ctr.BorderCount + 1;`. Sixteen at the default."""
    if ctr_border_count < 1:
        raise Error("ctr_border_count must be positive")
    return ctr_border_count + 1


def ctr_predict_scale(ctr_border_count: Int, norm: Float64) raises -> Float64:
    """`split.cpp:81`: `Scale = ctrInfo.BorderCount / norm[Ctr.PriorIdx]`.

    This is where the training-time multiply by `borderCount` and divide by
    `norm` are folded into a single inference-time scale, so the two formulas
    produce the same real number and only the training one truncates.
    """
    if ctr_border_count < 1:
        raise Error("ctr_border_count must be positive")
    if norm <= 0.0:
        raise Error("norm must be positive")
    return Float64(ctr_border_count) / norm


def ctr_predict_value(
    count_in_class: Float64,
    total_count: Float64,
    prior_num: Float64,
    prior_denom: Float64,
    shift: Float64,
    scale: Float64,
) raises -> Float64:
    """`TModelCtr::Calc` (`catboost/libs/model/online_ctr.h:289`), verbatim.

        float ctr = (countInClass + PriorNum) / (totalCount + PriorDenom);
        return (ctr + Shift) * Scale;

    Unquantized, unlike its training counterpart, and `prior_denom` is a real
    serialized field here even though the CPU trainer always writes 1 into it.
    `calc(0, 0)` is a legal and reachable call: it is what a category unseen
    during training gets (`ctr_calcer.py:35`), and it evaluates to the pure
    prior.
    """
    var denom = total_count + prior_denom
    if denom == 0.0:
        raise Error(
            "ctr denominator is zero: total_count + prior_denom must be"
            " nonzero, and prior_denom is 1 on CatBoost's CPU path"
        )
    var ctr = (Float32(count_in_class) + Float32(prior_num)) / Float32(denom)
    return Float64((ctr + Float32(shift)) * Float32(scale))


def ctr_predict_border(bin_border: Int) -> Float64:
    """`EmulateUi8Rounding` (`split.h:512`): `return value + 0.999999f;`.

    The bridge between the two formulas. A training comparison `bin > border`
    on truncated integers becomes an inference comparison
    `x > border + 0.999999f` on the unquantized float, so the model file can
    carry one float threshold and still mean the integer test the search made.
    """
    return Float64(Float32(bin_border) + Float32(_UI8_ROUNDING_EPS))


def ctr_predict_bucket(value: Float64, ctr_border_count: Int) raises -> Int:
    """`ctr_predict_border`'s rule solved for the bucket index instead.

    CatBoost carries a float threshold in the model and tests
    `x > border + 0.999999f`. mojotrees carries an **integer bin id** in the
    tree and tests `bin > border`, so the epsilon that CatBoost puts on the
    threshold has to be put back on the value. Solving for the bucket that makes
    the two tests agree for every integer `border`:

        x > border + 0.999999          (CatBoost)
        <=>  (x - 0.999999) > border
        <=>  floor(x - 0.999999) + 1 > border

    so `bucket = floor(x - 0.999999) + 1`, clamped to `[0, ctr_border_count]`.
    They differ only at the single point `x == border + 0.999999f` exactly,
    where CatBoost's strict `>` sends the row left and this sends it right; the
    band between `floor(x)` and this answer is `1e-6` wide in units of a bucket,
    which is the whole of the divergence and it is stated rather than hidden.

    The arithmetic is `Float32` for the reason `ctr_train_bin`'s is: CatBoost's
    `float` is what decides the bucket, and reproducing the bucket means
    reproducing the width.
    """
    if ctr_border_count < 1:
        raise Error("ctr_border_count must be positive")
    var shifted = Float32(value) - Float32(_UI8_ROUNDING_EPS)
    var b = Int(floor(Float64(shifted))) + 1
    if b < 0:
        b = 0
    if b > ctr_border_count:
        b = ctr_border_count
    return b


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


struct CtrParams(Copyable, Movable):
    """One CTR description, plus the permutation settings it needs.

    `enabled` is False by default and nothing in the package reads this struct,
    so constructing one changes nothing. Every other field carries CatBoost's
    verified default, so a port of a stock CatBoost configuration reads the
    same here even though this module refuses to run.
    """

    var enabled: Bool
    var ctr_type: Int
    var ctr_border_count: Int
    var target_border_count: Int
    var priors: List[Float64]
    var prior_denom: Float64
    var counter_calc_method: Int
    var permutation_count: Int
    var permutation_block_size: Int
    var has_time: Bool
    var max_ctr_complexity: Int
    var seed: Int

    def __init__(out self):
        """CatBoost's defaults, disabled."""
        self.enabled = False
        self.ctr_type = CTR_BORDERS
        self.ctr_border_count = DEFAULT_CTR_BORDER_COUNT
        self.target_border_count = DEFAULT_TARGET_BORDER_COUNT
        self.priors = List[Float64]()
        self.prior_denom = CPU_PRIOR_DENOM
        self.counter_calc_method = DEFAULT_COUNTER_CALC_METHOD
        self.permutation_count = DEFAULT_PERMUTATION_COUNT
        self.permutation_block_size = PERMUTATION_BLOCK_SIZE_NOT_SET
        self.has_time = False
        self.max_ctr_complexity = DEFAULT_MAX_CTR_COMPLEXITY
        self.seed = DEFAULT_CTR_SEED

    @staticmethod
    def disabled() -> CtrParams:
        """The inactive bundle, spelled out at a call site."""
        return CtrParams()

    @staticmethod
    def enable(
        ctr_type: Int = CTR_BORDERS,
        priors: List[Float64] = [],
        ctr_border_count: Int = DEFAULT_CTR_BORDER_COUNT,
        target_border_count: Int = DEFAULT_TARGET_BORDER_COUNT,
        permutation_count: Int = DEFAULT_PERMUTATION_COUNT,
        counter_calc_method: Int = DEFAULT_COUNTER_CALC_METHOD,
        seed: Int = DEFAULT_CTR_SEED,
    ) raises -> CtrParams:
        """An active bundle. An empty `priors` takes `default_priors`, which is
        what `SetDefaultPriorsIfNeeded` (`catboost_options.cpp:1209`) does."""
        var out = CtrParams()
        out.enabled = True
        out.ctr_type = ctr_type
        out.ctr_border_count = ctr_border_count
        out.target_border_count = target_border_count
        out.permutation_count = permutation_count
        out.counter_calc_method = counter_calc_method
        out.seed = seed
        if len(priors) == 0:
            out.priors = default_priors(ctr_type)
        else:
            out.priors = priors.copy()
        return out^

    def n_priors(self) -> Int:
        return len(self.priors)

    def is_active(self) -> Bool:
        return self.enabled

    def is_permutation_dependent(self) -> Bool:
        """`IsPermutationDependentCtrType` (`ctr_type.cpp`): `Counter` and
        `FeatureFreq` return false, every target-reading type returns true.

        A `Counter` fit therefore needs no permutation at all, which is worth
        knowing before building one.
        """
        return self.ctr_type != CTR_COUNTER

    def needs_target_classifier(self) -> Bool:
        """`NeedTargetClassifier` (`ctr_type.cpp`). `Counter` does not, and
        CatBoost still keeps a fake classifier at index 0 for it, with a comment
        calling it a dirty hack (`ctr_helper.h:22`)."""
        return self.ctr_type != CTR_COUNTER

    def validate(self) raises:
        if not self.enabled:
            return
        if (
            self.ctr_type != CTR_BORDERS
            and self.ctr_type != CTR_BUCKETS
            and self.ctr_type != CTR_BINARIZED_TARGET_MEAN
            and self.ctr_type != CTR_COUNTER
        ):
            raise Error(
                "ctr_type must be one of CTR_BORDERS, CTR_BUCKETS,"
                " CTR_BINARIZED_TARGET_MEAN, CTR_COUNTER"
            )
        if self.ctr_border_count < 1:
            raise Error("ctr_border_count must be positive")
        if self.target_border_count < 1:
            raise Error("ctr_target_border_count must be positive")
        if len(self.priors) == 0:
            raise Error("a ctr description needs at least one prior")
        if self.permutation_count < 1:
            raise Error("permutation_count must be positive")
        if self.permutation_block_size < 0:
            raise Error("permutation_block_size must be nonnegative")
        if (
            self.counter_calc_method != COUNTER_CALC_SKIP_TEST
            and self.counter_calc_method != COUNTER_CALC_FULL
        ):
            raise Error(
                "counter_calc_method must be COUNTER_CALC_SKIP_TEST or"
                " COUNTER_CALC_FULL"
            )
        check_ctr_prior_denom(self.prior_denom)
        check_ctr_complexity(self.max_ctr_complexity)


# ---------------------------------------------------------------------------
# Guards. Each refuses by name rather than ignoring.
# ---------------------------------------------------------------------------


def check_ctr_prior_denom(prior_denom: Float64) raises:
    """CatBoost's CPU rule, `ctr_helper.cpp:50`, by name.

        CB_ENSURE(denom == 1.0,
                  "Error: CPU could use only 1 as denom for ctrs currently");

    A prior is written `num:denom` in the option surface, and the denominator is
    accepted, checked and then discarded on the CPU. `ctr_train_bin` hard-codes
    the 1 in its `totalCount + 1`, so a caller who set anything else here would
    be silently ignored; this raises instead.
    """
    if prior_denom != CPU_PRIOR_DENOM:
        raise Error(
            "ctr prior_denom must be 1: CatBoost's CPU path refuses any other"
            " value at option load and the training formula hard-codes"
            " totalCount + 1"
        )


def check_ctr_border_type(border_selection_type: String) raises:
    """`ctr_helper.cpp:29`, by name.

        CB_ENSURE(... == EBorderSelectionType::Uniform,
                  "Error: CPU supports only uniform binarization for CTRS");

    It is a consequence of the training value already being a bucket index:
    there is no pass over CTR values in which a border-selection algorithm
    could run.
    """
    if border_selection_type != "Uniform":
        raise Error(
            "ctr binarization must be Uniform: the training CTR is already a"
            " bucket index, so there is no value pass for another border"
            " selection to run in"
        )


def check_ctr_complexity(max_ctr_complexity: Int) raises:
    """CatBoost's own bound on `max_ctr_complexity`.

    `TCatFeatureParams::Validate` (`cat_feature_options.cpp:266-271`):

        const ui32 ctrComplexityLimit = GetMaxTreeDepth();      // 16
        CB_ENSURE(MaxTensorComplexity.Get() < ctrComplexityLimit, ...);

    This used to refuse anything above 1, by name, so that the combinations lane
    would delete a refusal rather than discover an assumption. That lane has
    landed (catalog A30) and `ctr_combinations.mojo` now holds the binarized
    half of `calc_hashes`, the candidate enumeration from
    `greedy_tensor_search.cpp`, and `ctr_leaf_count_limit`'s top-K reindexing.
    The refusal that survives is `ctr_combinations.check_ctr_combination_trainer_support`,
    which refuses an ENABLED complexity above 1 because no grow loop drives the
    enumeration yet -- the same honest "unreached" statement
    `check_ctr_trainer_support` makes below.

    The bound is written out here rather than imported from
    `ctr_combinations.mojo`, because that module imports this one and a call in
    the other direction would close a cycle.
    """
    if max_ctr_complexity < 1:
        raise Error("max_ctr_complexity must be positive")
    if max_ctr_complexity >= MAX_CTR_COMPLEXITY_LIMIT:
        raise Error(
            "max_ctr_complexity must be below 16: CatBoost bounds it by"
            " GetMaxTreeDepth() at cat_feature_options.cpp:269"
        )


def check_ctr_trainer_support(params: CtrParams) raises:
    """Refuses an enabled bundle at a trainer boundary. See
    `check_ctr_model_support`, which is the refusal that now actually holds.

    The original reason -- "nothing appends CTR columns to a design matrix" --
    stopped being true on 2026-08-16. `ctr_columns.mojo` builds them,
    `binning.append_ctr_train_columns` appends them to a `BinnedMatrix`, and
    `BinMapper.transform` and `BinMapper.bin_row` append the inference half, so
    the mechanism reaches the histogram and reaches a score. What has NOT
    landed is the model file: `serialize._write_mapper` has no ctr section, so a
    fitted model would save without its tables and load scoring wrong.

    This overload is kept because it takes a `CtrParams`, which is A19's own
    bundle and is still what a caller holding one has. `check_ctr_model_support`
    is the one the dataset path calls, and it names the real blocker.
    """
    if params.enabled:
        raise Error(
            "ordered target statistics cannot be trained into a model yet:"
            " the columns build and score, but their fitted tables have no"
            " place in the model file, so a saved model would load without"
            " them (catalog A19, and A29 owns serialize.mojo)"
        )


def check_ctr_model_support(active: Bool) raises:
    """Refuses to turn CTR columns into a `Model`, and says why by name.

    The four fitted tables (`final_ctr_class_table`, `final_ctr_mean_table`,
    `final_ctr_counter_table`, and the counter denominator) are **model state**:
    they are read off the target, and a model that lost them keeps every tree
    that references a CTR column and then bins those columns as if the columns
    were absent. That is a wrong answer that looks like a right one, which is
    the one failure mode worth refusing loudly for.

    `serialize._write_mapper` writes `n_features`, `n_bins`, the edges, the edge
    offsets and the missing-bin table; `_write_categorical` writes the category
    tables. Neither writes a CTR section and neither reader reads one. Adding
    that section, bumping the format version, and calling
    `ctr_columns.check_ctr_serializable` from `save_model`,
    `save_multiclass_model` and `save_dataset` is catalog A29's work in
    `serialize.mojo`, which the CTR wiring lane does not own.

    Until then this raises at every `trainset.train_dataset*` entry point, so a
    CTR-carrying model is never produced and therefore never saved wrong.
    Deleting this guard is the export lane's first step, not a side effect of
    anything here.
    """
    if active:
        raise Error(
            "a dataset carrying ctr columns cannot be trained into a model:"
            " the fitted ctr tables are model state (catalog A19) and the"
            " mojotrees model format has no ctr section, so the model would"
            " save without them and load scoring wrong. serialize.mojo and the"
            " format version belong to the model-export lane (catalog A29)"
        )


# ---------------------------------------------------------------------------
# Shared validation for the ordered loops
# ---------------------------------------------------------------------------


def _check_ordered_inputs(
    categories: List[Int],
    permutation: List[Int],
    n_buckets: Int,
) raises:
    if n_buckets < 1:
        raise Error("n_buckets must be positive")
    var n = len(categories)
    if len(permutation) != n:
        raise Error("permutation length must match the row count")
    for i in range(n):
        var b = categories[i]
        if b < 0 or b >= n_buckets:
            raise Error("category bucket out of range")
    var seen = List[Bool]()
    seen.resize(n, False)
    for p in range(n):
        var r = permutation[p]
        if r < 0 or r >= n:
            raise Error("permutation entry out of range")
        if seen[r]:
            raise Error("permutation is not a bijection")
        seen[r] = True


def _check_classes(
    target_class_of_row: List[Int], n_rows: Int, n_classes: Int
) raises:
    if len(target_class_of_row) != n_rows:
        raise Error("target class length must match the row count")
    for i in range(n_rows):
        var c = target_class_of_row[i]
        if c < 0 or c >= n_classes:
            raise Error("target class out of range")


# ---------------------------------------------------------------------------
# The ordered loops
# ---------------------------------------------------------------------------


def ordered_ctr_borders_binary(
    categories: List[Int],
    target_class_of_row: List[Int],
    permutation: List[Int],
    n_buckets: Int,
    prior: Float64,
    ctr_border_count: Int,
    mut out: List[Int],
) raises:
    """`CalcOnlineCTRSimple` (`online_ctr.cpp:344`), the two-class `Borders`
    path, which is the default for a binary target and the one that matters.

    The inner loop (`online_ctr.cpp:300`) reads BEFORE it writes, in permutation
    order:

        goodCount  = elem[1];
        totalCount = elem[0] + elem[1];
        ++elem[permutedTargetClass[docIdx]];

    So row `i` sees the rows strictly before it in the permutation and never its
    own target, and the first row of a category sees `0 / 0` and gets the pure
    prior. That read-then-write ordering IS the leakage argument; swapping the
    two lines is the single edit that silently turns this back into ordinary
    target encoding.

    `out` is indexed by original row id. CatBoost writes into an array that is
    itself in permuted order; writing back through the permutation keeps one row
    order for the caller and is otherwise the same values.
    """
    var n = len(categories)
    _check_ordered_inputs(categories, permutation, n_buckets)
    _check_classes(target_class_of_row, n, SIMPLE_CLASSES_COUNT)
    var shift = ctr_shift(prior)
    var norm = ctr_norm(prior)
    var counts = List[Int]()
    counts.resize(n_buckets * 2, 0)
    out.clear()
    out.resize(n, 0)
    for p in range(n):
        var r = permutation[p]
        var b = categories[r]
        var good = counts[b * 2 + 1]
        var total = counts[b * 2] + counts[b * 2 + 1]
        out[r] = ctr_train_bin(
            Float64(good), total, prior, shift, norm, ctr_border_count
        )
        counts[b * 2 + target_class_of_row[r]] += 1


def ordered_ctr_classes(
    categories: List[Int],
    target_class_of_row: List[Int],
    permutation: List[Int],
    n_buckets: Int,
    n_classes: Int,
    ctr_type: Int,
    target_border_idx: Int,
    prior: Float64,
    ctr_border_count: Int,
    mut out: List[Int],
) raises:
    """`CalcOnlineCTRClasses` (`online_ctr.cpp:144`): `Buckets`, and `Borders`
    above two classes.

    The two types share every line except `UpdateGoodCount`
    (`online_ctr.cpp:115`):

        if (ctrType == ECtrType::Buckets) { *goodCount = curCount; }
        else                              { *goodCount -= curCount; }

    which CatBoost runs inside a loop over borders that walks `goodCount` down
    from the total. Reading that loop out, `Buckets` at border `k` is the count
    of class `k`, and `Borders` at border `k` is the count of every class above
    `k`. The denominator is the full count either way, which is what the
    inference reader does too (`ctr_calcer.py:44-64`).

    One value per `(border, prior)` pair is produced there; this call produces
    the one for `target_border_idx`, which is the caller's loop.
    """
    var n = len(categories)
    _check_ordered_inputs(categories, permutation, n_buckets)
    if n_classes < 1:
        raise Error("n_classes must be positive")
    _check_classes(target_class_of_row, n, n_classes)
    if ctr_type != CTR_BUCKETS and ctr_type != CTR_BORDERS:
        raise Error(
            "ordered_ctr_classes handles CTR_BUCKETS and CTR_BORDERS only"
        )
    var n_borders = ctr_target_border_count(ctr_type, n_classes)
    if target_border_idx < 0 or target_border_idx >= n_borders:
        raise Error("target_border_idx out of range for this ctr type")
    var shift = ctr_shift(prior)
    var norm = ctr_norm(prior)
    var counts = List[Int]()
    counts.resize(n_buckets * n_classes, 0)
    var totals = List[Int]()
    totals.resize(n_buckets, 0)
    out.clear()
    out.resize(n, 0)
    for p in range(n):
        var r = permutation[p]
        var b = categories[r]
        var base = b * n_classes
        var total = totals[b]
        var good: Int
        if ctr_type == CTR_BUCKETS:
            good = counts[base + target_border_idx]
        else:
            good = total
            for border in range(target_border_idx + 1):
                good -= counts[base + border]
        out[r] = ctr_train_bin(
            Float64(good), total, prior, shift, norm, ctr_border_count
        )
        counts[base + target_class_of_row[r]] += 1
        totals[b] = total + 1


def ordered_ctr_mean(
    categories: List[Int],
    target_class_of_row: List[Int],
    permutation: List[Int],
    n_buckets: Int,
    n_classes: Int,
    prior: Float64,
    ctr_border_count: Int,
    mut out: List[Int],
) raises:
    """`CalcOnlineCTRMean` (`online_ctr.cpp:437`).

    The accumulator is a `(Sum, Count)` pair and the increment is

        elem.Add(static_cast<float>(permutedTargetClass[docId]) / targetBorderCount)

    -- the running mean of the CLASS INDEX normalized to `[0, 1]`, not of the
    raw target. `targetBorderCount` here is `targetClassesCount - 1`
    (`online_ctr.cpp:762` passes exactly that), so at the default two classes it
    is 1 and the increment is the class index itself.

    `GetTargetBorderCount` returns 1 for this type, so it is one feature per
    prior whatever the class count.

    CatBoost's `TCtrMeanHistory::Sum` is a `float`. Ours is a `Float64` and only
    `ctr_train_bin` narrows, which is the one place this module is deliberately
    more accurate than its reference. It is also the one of the four types whose
    accumulator is not an integer, which is why CatBoost leaves this loop serial
    while parallelizing the others.
    """
    var n = len(categories)
    _check_ordered_inputs(categories, permutation, n_buckets)
    if n_classes < 2:
        raise Error("n_classes must be at least 2 for a binarized target mean")
    _check_classes(target_class_of_row, n, n_classes)
    var n_borders = n_classes - 1
    var shift = ctr_shift(prior)
    var norm = ctr_norm(prior)
    var sums = List[Float64]()
    sums.resize(n_buckets, 0.0)
    var counts = List[Int]()
    counts.resize(n_buckets, 0)
    out.clear()
    out.resize(n, 0)
    for p in range(n):
        var r = permutation[p]
        var b = categories[r]
        out[r] = ctr_train_bin(
            sums[b], counts[b], prior, shift, norm, ctr_border_count
        )
        sums[b] = sums[b] + Float64(target_class_of_row[r]) / Float64(n_borders)
        counts[b] = counts[b] + 1


def counter_ctr(
    categories: List[Int],
    n_buckets: Int,
    count_rows: Int,
    prior: Float64,
    ctr_border_count: Int,
    mut out: List[Int],
) raises:
    """`CalcOnlineCTRCounter` (`online_ctr.cpp:503`), and **it is not ordered.**

    `counterCTRTotal` is filled once by `CountOnlineCTRTotal`
    (`online_ctr.cpp:564`) over the whole array before any row is emitted, and
    the denominator is

        counterCTRDenominator = *MaxElement(counterCTRTotal.begin(), ...)

    -- the largest category's count, not the row count. So every row of a
    category gets the same value, `(fullCount + prior) / (maxCount + 1)`, and
    `IsPermutationDependentCtrType` returns false for this type. There is no
    permutation argument here because there is nothing for one to do, and a
    caller who passes one is doing something the source does not.

    `count_rows` is `counter_calc_method`. `SkipTest`, the default, counts the
    learn rows only; `Full` counts learn plus every test set
    (`online_ctr.cpp:723`, `sampleCount = hashArr.size()`). It is a train-time
    transduction switch and at its default CatBoost does not look at test
    features. Values are still emitted for every row of `categories`, which is
    how a test row gets a value under `SkipTest`.
    """
    var n = len(categories)
    if n_buckets < 1:
        raise Error("n_buckets must be positive")
    if count_rows < 0 or count_rows > n:
        raise Error("count_rows must lie between 0 and the row count")
    for i in range(n):
        if categories[i] < 0 or categories[i] >= n_buckets:
            raise Error("category bucket out of range")
    var shift = ctr_shift(prior)
    var norm = ctr_norm(prior)
    var counts = List[Int]()
    counts.resize(n_buckets, 0)
    for i in range(count_rows):
        counts[categories[i]] += 1
    var denom = 0
    for b in range(n_buckets):
        if counts[b] > denom:
            denom = counts[b]
    out.clear()
    out.resize(n, 0)
    for i in range(n):
        out[i] = ctr_train_bin(
            Float64(counts[categories[i]]),
            denom,
            prior,
            shift,
            norm,
            ctr_border_count,
        )


def counter_denominator(counts: List[Int]) -> Int:
    """`*MaxElement(counterCTRTotal.begin(), counterCTRTotal.end())`
    (`online_ctr.cpp:728`), exposed so the inference side can be handed the same
    number the training side used."""
    var d = 0
    for b in range(len(counts)):
        if counts[b] > d:
            d = counts[b]
    return d


# ---------------------------------------------------------------------------
# The inference half: static tables and their readers
# ---------------------------------------------------------------------------


def final_ctr_class_table(
    categories: List[Int],
    target_class_of_row: List[Int],
    n_buckets: Int,
    n_classes: Int,
    count_rows: Int,
    mut table: List[Int],
) raises:
    """`CalcFinalCtrsImpl`'s class arm (`online_ctr.cpp:926`), the table the
    model file carries.

        ++elem[targetClass[z]];       // elem = ctrIntArray + classes * elemId

    A plain loop over all rows, in dataset order, with NO permutation and NO
    prefix. That is the whole difference between what training sees and what
    inference sees, and it is why the two formulas cannot be shared.

    `table[bucket * n_classes + class]`, which is the layout `ctr_calcer.py`
    indexes.
    """
    var n = len(categories)
    if n_buckets < 1:
        raise Error("n_buckets must be positive")
    if n_classes < 1:
        raise Error("n_classes must be positive")
    if count_rows < 0 or count_rows > n:
        raise Error("count_rows must lie between 0 and the row count")
    _check_classes(target_class_of_row, n, n_classes)
    for i in range(n):
        if categories[i] < 0 or categories[i] >= n_buckets:
            raise Error("category bucket out of range")
    table.clear()
    table.resize(n_buckets * n_classes, 0)
    for z in range(count_rows):
        table[categories[z] * n_classes + target_class_of_row[z]] += 1


def predict_ctr_class(
    table: List[Int],
    n_classes: Int,
    bucket: Int,
    ctr_type: Int,
    target_border_idx: Int,
    prior_num: Float64,
    prior_denom: Float64,
    shift: Float64,
    scale: Float64,
) raises -> Float64:
    """`calc_ctrs`' `Buckets` and `Borders` arms
    (`catboost/libs/model/model_export/resources/ctr_calcer.py:44-66`).

    `bucket < 0` means the category was never seen on learn, which is a case
    training cannot reach and inference must. It answers `calc(0, 0)`, the pure
    prior (`ctr_calcer.py:35`).

    The two-class `Borders` shortcut in the source
    (`ctr_history[bucket*2+1]` over `ctr_history[bucket*2] + ctr_history[bucket*2+1]`)
    is the general branch specialized, and is reproduced by the general branch
    here rather than duplicated.
    """
    if n_classes < 1:
        raise Error("n_classes must be positive")
    if bucket < 0:
        return ctr_predict_value(0.0, 0.0, prior_num, prior_denom, shift, scale)
    if (bucket + 1) * n_classes > len(table):
        raise Error("bucket out of range for this table")
    var n_borders = ctr_target_border_count(ctr_type, n_classes)
    if target_border_idx < 0 or target_border_idx >= n_borders:
        raise Error("target_border_idx out of range for this ctr type")
    var base = bucket * n_classes
    var total = 0
    for c in range(n_classes):
        total += table[base + c]
    var good = 0
    if ctr_type == CTR_BUCKETS:
        good = table[base + target_border_idx]
    elif ctr_type == CTR_BORDERS:
        for c in range(target_border_idx + 1, n_classes):
            good += table[base + c]
    else:
        raise Error(
            "predict_ctr_class handles CTR_BUCKETS and CTR_BORDERS only"
        )
    return ctr_predict_value(
        Float64(good), Float64(total), prior_num, prior_denom, shift, scale
    )


def final_ctr_mean_table(
    categories: List[Int],
    target_class_of_row: List[Int],
    n_buckets: Int,
    n_classes: Int,
    count_rows: Int,
    mut sums: List[Float64],
    mut counts: List[Int],
) raises:
    """`CalcFinalCtrsImpl`'s `BinarizedTargetMeanValue` arm
    (`online_ctr.cpp:918`):
    `elem.Add(static_cast<float>(targetClass[z]) / targetBorderCount)`, over all
    rows and in dataset order."""
    var n = len(categories)
    if n_buckets < 1:
        raise Error("n_buckets must be positive")
    if n_classes < 2:
        raise Error("n_classes must be at least 2 for a binarized target mean")
    if count_rows < 0 or count_rows > n:
        raise Error("count_rows must lie between 0 and the row count")
    _check_classes(target_class_of_row, n, n_classes)
    for i in range(n):
        if categories[i] < 0 or categories[i] >= n_buckets:
            raise Error("category bucket out of range")
    var n_borders = n_classes - 1
    sums.clear()
    sums.resize(n_buckets, 0.0)
    counts.clear()
    counts.resize(n_buckets, 0)
    for z in range(count_rows):
        var b = categories[z]
        sums[b] = sums[b] + Float64(target_class_of_row[z]) / Float64(n_borders)
        counts[b] = counts[b] + 1


def predict_ctr_mean(
    sums: List[Float64],
    counts: List[Int],
    bucket: Int,
    prior_num: Float64,
    prior_denom: Float64,
    shift: Float64,
    scale: Float64,
) raises -> Float64:
    """`calc_ctrs`' mean arm (`ctr_calcer.py:37-39`):
    `ctr.calc(ctr_mean_history.sum, ctr_mean_history.count)`."""
    if bucket < 0:
        return ctr_predict_value(0.0, 0.0, prior_num, prior_denom, shift, scale)
    if bucket >= len(sums) or bucket >= len(counts):
        raise Error("bucket out of range for this table")
    return ctr_predict_value(
        sums[bucket],
        Float64(counts[bucket]),
        prior_num,
        prior_denom,
        shift,
        scale,
    )


def final_ctr_counter_table(
    categories: List[Int],
    n_buckets: Int,
    count_rows: Int,
    mut counts: List[Int],
) raises:
    """`CalcFinalCtrsImpl`'s counter arm (`online_ctr.cpp:921`),
    `++ctrIntArray[elemId]`. The denominator is `counter_denominator(counts)`,
    matching `result->CounterDenominator = *MaxElement(...)` at
    `online_ctr.cpp:935`.

    `FeatureFreq` differs here and only here -- its denominator is
    `totalSampleCount` rather than the max (`online_ctr.cpp:938`) -- and it is
    GPU-only and not built.
    """
    var n = len(categories)
    if n_buckets < 1:
        raise Error("n_buckets must be positive")
    if count_rows < 0 or count_rows > n:
        raise Error("count_rows must lie between 0 and the row count")
    for i in range(n):
        if categories[i] < 0 or categories[i] >= n_buckets:
            raise Error("category bucket out of range")
    counts.clear()
    counts.resize(n_buckets, 0)
    for z in range(count_rows):
        counts[categories[z]] += 1


def predict_ctr_counter(
    counts: List[Int],
    denominator: Int,
    bucket: Int,
    prior_num: Float64,
    prior_denom: Float64,
    shift: Float64,
    scale: Float64,
) raises -> Float64:
    """`calc_ctrs`' counter arm (`ctr_calcer.py:40-43`):
    `ctr.calc(ctr_total[bucket], denominator)`."""
    if bucket < 0:
        return ctr_predict_value(
            0.0, Float64(denominator), prior_num, prior_denom, shift, scale
        )
    if bucket >= len(counts):
        raise Error("bucket out of range for this table")
    return ctr_predict_value(
        Float64(counts[bucket]),
        Float64(denominator),
        prior_num,
        prior_denom,
        shift,
        scale,
    )


# ---------------------------------------------------------------------------
# Projection hashing. Built, tested, and used by nothing here.
# ---------------------------------------------------------------------------


def ctr_combination_hash(a: UInt64, b: UInt64) -> UInt64:
    """`CalcHash` (`catboost/libs/model/hash.h:11`), verbatim.

        MAGIC_MULT * (a + MAGIC_MULT * b)     // MAGIC_MULT = 0x4906ba494954cb65

    Unsigned 64-bit wraparound is the intended arithmetic in both languages.
    This is what a *combination* of categorical columns folds with, so it is the
    first thing the `max_ctr_complexity` lane needs; nothing in this file calls
    it, because a complexity-1 projection is one column and needs no fold.
    """
    return _CTR_HASH_MAGIC * (a + _CTR_HASH_MAGIC * b)


def ctr_projection_hash(values: List[UInt64]) -> UInt64:
    """`calc_hashes`' categorical loop (`ctr_calcer.py:9-19`):

        result = 0
        for cat_feature_index in transposed_cat_feature_indexes:
            result = calc_hash(result, hashed_cat_features[cat_feature_index])

    Order matters, and the seed is 0. A one-element projection is
    `calc_hash(0, v)` and NOT `v`, which is worth knowing before someone
    shortcuts the single-column case.

    What is deliberately missing, and what the combinations lane must add, is
    the second loop of `calc_hashes`: a projection may also carry FLOAT splits
    (`TProjection::BinFeatures`), each folded in as the 0 or 1 of a threshold
    test rather than as a value. A combination is not only a tuple of
    categorical columns.
    """
    var result = UInt64(0)
    for i in range(len(values)):
        result = ctr_combination_hash(result, values[i])
    return result
