"""CTR columns in a design matrix: the wiring between `ctr.mojo` and the binner.

Catalog A19 (the mechanism) and A30 (the complexity bound). `ctr.mojo` holds the
arithmetic, the four ordered loops, the four static tables and the readers over
them, all verified against CatBoost `master` on 2026-08-16 and all reached by
nothing. This module is the layer that turns a *categorical column of a binned
matrix* into *numeric columns of that same matrix*, and it is the piece A19's
`check_ctr_trainer_support` says is missing.

Scope, and it is deliberately narrow
------------------------------------
**Simple CTRs only: complexity 1, one categorical column per projection.**
Combinations stay off. `ctr.check_ctr_complexity` still enforces CatBoost's
`1 <= v < 16` and `ctr_combinations.check_ctr_combination_trainer_support` still
refuses an enabled complexity above 1, because no grow loop drives
`grow_tree_ctr_projections`; `SimpleCtrConfig.validate` calls both, so neither
guard can be skipped by coming through here.

One permutation, not CatBoost's three learning folds. CatBoost draws a fold per
tree (`train.cpp:208`, `ctr.ctr_fold_index`), which is a decision inside the
round loop; a `Dataset` is built once and trained on many times, so the columns
it carries are built from one permutation and that permutation is a property of
the dataset. `SimpleCtrConfig.permutation_index` selects it, `permutation_count`
is still recorded so a ported configuration reads the same, and the divergence is
written down rather than hidden.

The four columns, which is the number that costs
-------------------------------------------------
At CatBoost's CPU defaults one categorical column produces **four** numeric
columns, not one (`catboost_options.cpp::SetCtrDefaults`, and A19's table):

    Borders, priors {0, 0.5, 1}, one target border   -> 3 columns
    Counter, prior  {0}                              -> 1 column

`ctr.ctr_feature_count` computes the product per description and
`plan_ctr_columns` lays them out in `AllocateCtrData`'s order
(`online_ctr.cpp:741`): description, then target border, then prior.

Derived bounds, and nothing here is measured
--------------------------------------------
Let `n` be the learn row count and `C` the number of categorical columns wide
enough to escape the one-hot cutoff. Every figure below follows from the loop
structure and none of it was timed; this lane ran nothing.

- **Design matrix growth.** `4 * C` extra `UInt8` columns, so `4 * C * n` bytes
  added to `BinnedMatrix.bins`. At `n = 10^6` and `C = 10` that is 40 MB against
  the 50 MB a 50-feature base matrix costs, and it is the dominant cost of the
  feature and the reason `max_ctr_complexity = 4` is refused (A30 section 5
  prices combinations at 1.1 GB per tree at the same shape).
- **Element operations, once, at dataset construction.** One ordered pass per
  column (`4n`), plus one full count per column for `Counter` and for the final
  tables (`2n`), so about `6n` per categorical column and `6 * C * n` in total.
  It is paid **once per dataset**, not once per tree, which is the whole
  difference between complexity 1 here and CatBoost's per-tree recomputation.
- **Transient allocation.** One `List[Int]` of `n` for the extracted category
  codes per slot (`8n` bytes), one of `n` for the target classes, one of `n` for
  the permutation, and one of `n` reused for each column's output. Peak transient
  is about `8n * (C + 3)` bytes and every one of them is freed before the
  `Dataset` is returned.
- **Model state.** Per categorical column: a class table of
  `n_buckets * n_classes` `Int`s and a counter table of `n_buckets` `Int`s, with
  `n_buckets = n_categories + 1 <= max_bin`. At `max_bin = 255` and two classes
  that is at most `255 * 2 + 255 = 765` `Int`s per column, which is small and is
  **fitted from the target**, so it is model state and not a hyperparameter. See
  `CtrTables`.

Determinism
-----------
The permutation is `ctr.ctr_permutation`, a keyed block sort: block `b` takes the
key `splitmix64(stream + b)` and blocks sort by `(key, b)`. Its answer does not
depend on the order the keys were computed in, which is why it is identical
across `MOJOTREES_NUM_WORKERS` and across machines. **Nothing here consumes it in
a way that could undo that**: it is computed once, in full, before any column is
built, and every loop below reads `permutation[p]` by position. No call site
draws from a running stream and no loop here is parallel, so the columns are the
same bytes at every worker count. The ordered accumulation is serial for all four
CTR types, which is stricter than CatBoost needs for three of them and is the
guarantee A19 chose.

Train and predict are different, and this module keeps them apart
-----------------------------------------------------------------
This is the failure A19 warns about at length and it is worth restating where the
wiring lives.

- **Training** columns come from `build_ctr_train_columns`, which runs the
  ordered prefix of one permutation with the denominator `totalCount + 1` and
  emits a `ui8` bucket index directly (`ctr.ctr_train_bin`). `trainset._build_ctr`
  is the only caller, and it puts them in the matrix with
  `binning.append_ctr_columns`.
- **Inference** columns come from `ctr_predict_columns` / `ctr_predict_row`,
  which read the static tables `fit_ctr_tables` built over the whole learn set
  with the denominator `t + PriorDenom`, produce an unquantized float
  (`ctr.ctr_predict_value`), and only then take a bucket by CatBoost's
  `EmulateUi8Rounding` rule (`ctr.ctr_predict_bucket`). They are appended by
  `BinMapper.transform` and `BinMapper.bin_row`, which is what makes a fitted
  `Model` score a raw row without any edit to `model.mojo`.

`BinMapper.transform` cannot be the training path for this reason, and the order
of operations in `Dataset` enforces it: the base matrix is transformed while the
mapper still has no tables, the ordered columns are appended, the tables are
fitted, and only then are the tables attached to the mapper. A mapper with tables
attached always appends the *static* columns, which is exactly right for a
validation set, a prediction batch and a single scored row, and never runs during
the fit that produced it.

Unseen categories
-----------------
CatBoost's inference has a `bucket < 0` arm for a category absent from the learn
set, which answers `calc(0, 0)`, the pure prior (`ctr_calcer.py:35`). It is
**unreachable from this wiring**, and not because anything was dropped:
`categorical.CategoricalSpec.bin_of` already maps every missing, unknown and
table-evicted value to `UNKNOWN_BIN = 0`, so a scored row always has a bucket.
Bucket 0 is a real bucket with real counts, and when no training row landed there
its counts are zero and `ctr.ctr_predict_value` returns the pure prior anyway --
the same number by a different route. The `bucket < 0` arm is left in `ctr.mojo`
because a future caller with a genuine hash space (a combination) will reach it.

Imports
-------
`.ctr`, `.ctr_combinations`, and `std`. `.ctr` imports `.rng`, `.rng` imports
nothing, and `.ctr_combinations` imports only `.ctr`; so this module's whole
transitive import set is `{ctr, ctr_combinations, rng}` and it can be imported
from `binning.mojo` without touching the `efb -> binning -> tree_parameters_extra`
cycle `cegb.mojo` documents. **It deliberately does not import `.categorical`**,
even though it is about categorical columns: `categorical` imports
`tree_parameters_extra`, and taking a `CategoricalSpec` here would drag that in
behind `binning`. The category information arrives as two plain lists,
`is_categorical` and `n_categories`, which `binning.mojo` builds from the spec it
already holds. That is the same rule A19 followed and for the same reason.
"""

from .ctr import (
    CPU_PRIOR_DENOM,
    CTR_BINARIZED_TARGET_MEAN,
    CTR_BORDERS,
    CTR_BUCKETS,
    CTR_COUNTER,
    COUNTER_CALC_FULL,
    COUNTER_CALC_SKIP_TEST,
    DEFAULT_CTR_BORDER_COUNT,
    DEFAULT_CTR_SEED,
    DEFAULT_MAX_CTR_COMPLEXITY,
    DEFAULT_PERMUTATION_COUNT,
    PERMUTATION_BLOCK_SIZE_NOT_SET,
    SIMPLE_CLASSES_COUNT,
    check_ctr_complexity,
    check_ctr_prior_denom,
    counter_denominator,
    ctr_bucket_count,
    ctr_norm,
    ctr_permutation,
    ctr_predict_bucket,
    ctr_predict_scale,
    ctr_shift,
    ctr_target_border_count,
    ctr_train_bin,
    default_priors,
    final_ctr_class_table,
    final_ctr_counter_table,
    final_ctr_mean_table,
    ordered_ctr_borders_binary,
    ordered_ctr_classes,
    ordered_ctr_mean,
    predict_ctr_class,
    predict_ctr_counter,
    predict_ctr_mean,
    resolve_permutation_block_size,
    target_class,
    target_classes_count,
)
from .ctr_combinations import check_ctr_combination_trainer_support


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# `OneHotMaxSize` (`cat_feature_options.cpp`), CatBoost's CPU default. A
# categorical column is given CTRs only when it is *too wide to one-hot*, which
# is `IsPermutationNeeded`'s `hasCtrs` test spelled out:
# `CalcMaxCategoricalFeaturesUniqueValuesCountOnLearn() > OneHotMaxSize`
# (`learn_context.cpp:38`, and the feature-level test at
# `greedy_tensor_search.cpp:469`). A two-level column is one-hot and gets none.
comptime CTR_ONE_HOT_MAX_SIZE = 2

# The permutation this module uses by default. CatBoost keeps
# `permutation_count - 1 = 3` learning folds and draws one per tree; a dataset
# carries one set of columns, so it carries one permutation. Index 0 is the
# identity (`ctr_permutation`, matching `shuffle = (foldIdx != 0)`), which would
# make the ordered prefix dataset order and is a legitimate but degenerate
# choice; index 1 is the first genuinely permuted fold and is the default here.
comptime DEFAULT_CTR_PERMUTATION_INDEX = 1


# ---------------------------------------------------------------------------
# One CTR description
# ---------------------------------------------------------------------------


struct CtrDescription(Copyable, Movable):
    """`TCtrDescription` (`cat_feature_options.h:44`): a type, its priors, and
    its binarization.

    The binarization is a count only, because `ctr_helper.cpp:29` refuses
    anything but uniform on the CPU and `ctr.check_ctr_border_type` says so by
    name. The training CTR *is* a bucket index, so there is no value pass in
    which another border selection could run.
    """

    var ctr_type: Int
    var priors: List[Float64]
    var ctr_border_count: Int

    def __init__(
        out self,
        ctr_type: Int,
        var priors: List[Float64] = [],
        ctr_border_count: Int = DEFAULT_CTR_BORDER_COUNT,
    ) raises:
        """An empty `priors` takes `ctr.default_priors`, which is what
        `SetDefaultPriorsIfNeeded` (`catboost_options.cpp:1209`) does."""
        self.ctr_type = ctr_type
        self.ctr_border_count = ctr_border_count
        if len(priors) == 0:
            self.priors = default_priors(ctr_type)
        else:
            self.priors = priors^

    def n_priors(self) -> Int:
        return len(self.priors)

    def validate(self) raises:
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
        if len(self.priors) == 0:
            raise Error("a ctr description needs at least one prior")


def catboost_simple_ctr_defaults() raises -> List[CtrDescription]:
    """`SetCtrDefaults`' `default:` arm (`catboost_options.cpp`), the CPU
    `simple_ctr` for every loss except PairLogit.

        Borders with priors {0, 0.5, 1}   (3 columns at one target border)
        Counter with prior  {0}           (1 column)

    Four numeric columns per categorical column, which is A19's headline number
    and the reason CTRs are expensive. The PairLogit arm (`Counter` only) and the
    GPU arm (`FeatureFreq`) are not built.

    The list is constructed here rather than shared with a combinations list,
    because `SetCtrDefaults` warns twice that the two do not affect each other
    (`catboost_options.cpp:455-460`) and `ctr_combinations.CtrRouting` keeps them
    separate for the same reason.
    """
    var out = List[CtrDescription]()
    out.append(CtrDescription(CTR_BORDERS))
    out.append(CtrDescription(CTR_COUNTER))
    return out^


# ---------------------------------------------------------------------------
# The configuration a dataset carries
# ---------------------------------------------------------------------------


struct SimpleCtrConfig(Copyable, Movable):
    """Everything the dataset path needs to build simple CTR columns.

    `enabled` is False by default and a disabled config is read exactly once, by
    an `is_active()` test that returns immediately. Nothing is allocated, no pass
    is taken, and the binned matrix a disabled dataset holds is byte-identical to
    the one it held before this module existed.
    """

    var enabled: Bool
    var descriptions: List[CtrDescription]
    var target_borders: List[Float64]
    var counter_calc_method: Int
    var permutation_count: Int
    var permutation_index: Int
    var permutation_block_size: Int
    var has_time: Bool
    var one_hot_max_size: Int
    var max_ctr_complexity: Int
    var prior_denom: Float64
    var seed: Int

    def __init__(out self):
        """Disabled, and otherwise CatBoost's verified defaults, so a ported
        configuration reads the same even where this refuses to run."""
        self.enabled = False
        self.descriptions = List[CtrDescription]()
        self.target_borders = List[Float64]()
        self.counter_calc_method = COUNTER_CALC_SKIP_TEST
        self.permutation_count = DEFAULT_PERMUTATION_COUNT
        self.permutation_index = DEFAULT_CTR_PERMUTATION_INDEX
        self.permutation_block_size = PERMUTATION_BLOCK_SIZE_NOT_SET
        self.has_time = False
        self.one_hot_max_size = CTR_ONE_HOT_MAX_SIZE
        self.max_ctr_complexity = DEFAULT_MAX_CTR_COMPLEXITY
        self.prior_denom = CPU_PRIOR_DENOM
        self.seed = DEFAULT_CTR_SEED

    @staticmethod
    def disabled() -> SimpleCtrConfig:
        """The inactive bundle, spelled out at a call site."""
        return SimpleCtrConfig()

    @staticmethod
    def catboost_defaults(
        var target_borders: List[Float64] = [],
        permutation_index: Int = DEFAULT_CTR_PERMUTATION_INDEX,
        seed: Int = DEFAULT_CTR_SEED,
        max_ctr_complexity: Int = 1,
    ) raises -> SimpleCtrConfig:
        """An active bundle at CatBoost's CPU `simple_ctr` defaults.

        `max_ctr_complexity` defaults to **1** here and not to CatBoost's 4. A30
        section 5 prices 4 at about `1.6 * 10^9` element operations and 1.1 GB of
        CTR columns per tree at a million rows against `3 * 10^8` for the whole
        rest of the tree, and CatBoost itself silently resolves an unset value to
        1 for any fit under 200 iterations (`catboost_options.cpp:1046`,
        `ctr_combinations.resolve_max_ctr_complexity`). A30 item 6 says the place
        to take that position is the wiring lane's default, which is this line.
        `ctr.CtrParams` still records CatBoost's 4, so a port still reads the
        same.

        `target_borders` empty means "derive them", which
        `default_target_borders` will do for a two-valued label and will refuse
        for anything else.
        """
        var out = SimpleCtrConfig()
        out.enabled = True
        out.descriptions = catboost_simple_ctr_defaults()
        out.target_borders = target_borders^
        out.permutation_index = permutation_index
        out.seed = seed
        out.max_ctr_complexity = max_ctr_complexity
        return out^

    def is_active(self) -> Bool:
        return self.enabled

    def n_target_classes(self) -> Int:
        """`GetClassesCount`: `Borders.ysize() + 1`.

        Meaningful only once `target_borders` has been resolved. An empty list
        would answer 1, and one class makes `ctr_target_border_count` return 0
        for `Borders` -- so every column would silently vanish. `validate`
        refuses an empty list on an enabled bundle for exactly that reason, and
        `resolve_target_borders` is what fills it.
        """
        return target_classes_count(self.target_borders)

    def resolve_target_borders(mut self, label: List[Float64]) raises:
        """Fill `target_borders` from the label if the caller gave none.

        Called once, before anything is planned, because the plan's shape
        depends on the class count. After this the config records the borders it
        actually used, which is what a `Dataset` stores and what a subset of it
        inherits.

        For the default two-valued target the border is the midpoint of the two
        label values, so it carries no information beyond the label's support;
        that is why inheriting it into a fold is not the leak that inheriting
        bin edges would be. See `default_target_borders` for the source
        argument.
        """
        if not self.enabled:
            return
        if len(self.target_borders) != 0:
            return
        self.target_borders = default_target_borders(label)

    def validate(self) raises:
        """Every guard, none of them skippable by arriving through here.

        `check_ctr_complexity` is A19's bound (`1 <= v < 16`, CatBoost's own) and
        `check_ctr_combination_trainer_support` is A30's refusal of an enabled
        complexity above 1. Both are kept: the first is a range check that
        outlives every lane, the second is the honest statement that no grow loop
        drives the combination enumeration.
        """
        if not self.enabled:
            return
        if len(self.descriptions) == 0:
            raise Error("an enabled ctr config needs at least one description")
        for i in range(len(self.descriptions)):
            self.descriptions[i].validate()
        if len(self.target_borders) == 0:
            raise Error(
                "ctr target borders are unresolved: call"
                " resolve_target_borders(label) before planning. An empty list"
                " would mean one target class, and one class makes"
                " GetTargetBorderCount return 0 for Borders, so every column"
                " would silently disappear"
            )
        if self.permutation_count < 1:
            raise Error("permutation_count must be positive")
        if self.permutation_index < 0:
            raise Error("permutation_index must be nonnegative")
        if self.permutation_block_size < 0:
            raise Error("permutation_block_size must be nonnegative")
        if self.one_hot_max_size < 0:
            raise Error("one_hot_max_size must be nonnegative")
        if self.counter_calc_method == COUNTER_CALC_FULL:
            raise Error(
                "counter_calc_method='Full' is not implemented: it counts the"
                " learn rows plus every test set (online_ctr.cpp:723), and a"
                " Dataset holds learn rows only, so there is nothing here for"
                " the switch to include. SkipTest, CatBoost's default, is the"
                " only accepted value"
            )
        if self.counter_calc_method != COUNTER_CALC_SKIP_TEST:
            raise Error(
                "counter_calc_method must be COUNTER_CALC_SKIP_TEST or"
                " COUNTER_CALC_FULL"
            )
        check_ctr_prior_denom(self.prior_denom)
        check_ctr_complexity(self.max_ctr_complexity)
        check_ctr_combination_trainer_support(
            self.max_ctr_complexity, self.enabled
        )
        var n_classes = self.n_target_classes()
        for i in range(len(self.target_borders)):
            if i > 0 and self.target_borders[i] <= self.target_borders[i - 1]:
                raise Error("target borders must be strictly ascending")
        for i in range(len(self.descriptions)):
            ref d = self.descriptions[i]
            if d.ctr_type == CTR_BINARIZED_TARGET_MEAN and n_classes < 2:
                raise Error(
                    "BinarizedTargetMeanValue needs at least two target classes"
                )


def default_target_borders(label: List[Float64]) raises -> List[Float64]:
    """The one target border a default CTR fit needs, for a two-valued label.

    `ctr_target_border_count` defaults to 1 and the selection is `MinEntropy`
    (`cat_feature_options.cpp:230`), which is catalog A15's mechanism and not
    this lane's. It does not have to be, at the default: A30's appendix reads
    `TExactBinarizer` out and shows that at `maxBordersCount = 1` the banded DP
    does not run its main loop at all and collapses to

        threshold = argmin over i in [0, k-2] of P(W[i]) + P(W[k-1] - W[i])
        border    = (values[t] + values[t+1]) / 2

    over the `k` distinct target values. At `k = 2` the argmin is over the single
    index `i = 0`, so the border is exactly the midpoint of the two values and no
    entropy is evaluated. That is the whole of the default case for a binary
    target, which is the only target a default CTR comparison uses.

    `k > 2` raises rather than guessing. The general case is the real DP
    (`E_RLM2`, about seventy lines per A30's appendix) and belongs to whichever
    lane owns border selection; a caller with a multi-valued target passes
    `target_borders` explicitly.
    """
    var first = 0.0
    var second = 0.0
    var n_distinct = 0
    for i in range(len(label)):
        var v = label[i]
        if n_distinct == 0:
            first = v
            n_distinct = 1
        elif v == first:
            continue
        elif n_distinct == 1:
            second = v
            n_distinct = 2
        elif v == second:
            continue
        else:
            raise Error(
                "ctr target borders must be given explicitly for a target with"
                " more than two distinct values: the default"
                " ctr_target_border_count is 1 and its MinEntropy selection is"
                " only a midpoint when the target is two-valued (catalog A15"
                " owns the general case)"
            )
    if n_distinct != 2:
        raise Error(
            "ctr target borders need a target with exactly two distinct values,"
            " or an explicit target_borders list"
        )
    var lo = first
    var hi = second
    if second < first:
        lo = second
        hi = first
    var out = List[Float64]()
    out.append((lo + hi) * 0.5)
    return out^


def target_classes(
    label: List[Float64], borders: List[Float64]
) raises -> List[Int]:
    """`TTargetClassifier::GetTargetClass` over a whole label column.

    `ctr.target_class` is the four-line classifier verbatim; this is the loop
    around it, and it is the only place a raw label value is read by this
    module. Everything downstream sees class indices.
    """
    var out = List[Int](capacity=len(label))
    for i in range(len(label)):
        out.append(target_class(label[i], borders))
    return out^


# ---------------------------------------------------------------------------
# One produced column
# ---------------------------------------------------------------------------


struct CtrColumn(Copyable, Movable):
    """One numeric column produced from one categorical column.

    It is a `(projection, ctr type, target border, prior)` tuple, which is
    exactly what `AddCtrsToCandList` emits one candidate per
    (`greedy_tensor_search.cpp:400`) and what `AllocateCtrData` sizes
    (`online_ctr.cpp:741`).

    `shift` and `scale` are `CalcNormalization`'s pair folded the way
    `split.cpp:78-82` folds them: the training multiply by `borderCount` and the
    divide by `norm` become one inference `Scale`, so both formulas produce the
    same real number and only the training one truncates. At every default prior
    (0, 0.5, 1) `shift` is 0 and `scale` is `borderCount`.
    """

    var slot: Int
    """Index into `CtrTables.source_features`, i.e. which fitted table this
    column reads."""

    var source_feature: Int
    """The base feature id of the categorical column this came from."""

    var ctr_type: Int
    var target_border_idx: Int
    var prior_index: Int
    var prior: Float64
    var shift: Float64
    var norm: Float64
    var scale: Float64
    var ctr_border_count: Int

    def __init__(
        out self,
        slot: Int,
        source_feature: Int,
        ctr_type: Int,
        target_border_idx: Int,
        prior_index: Int,
        prior: Float64,
        ctr_border_count: Int,
    ) raises:
        self.slot = slot
        self.source_feature = source_feature
        self.ctr_type = ctr_type
        self.target_border_idx = target_border_idx
        self.prior_index = prior_index
        self.prior = prior
        self.shift = ctr_shift(prior)
        # `CalcNormalization` (`online_ctr.cpp:102`). `norm` is the training
        # divisor and `scale = borderCount / norm` is the inference multiplier
        # `split.cpp:81` folds it into; both are kept so neither side has to
        # recompute the pair per row. At every default prior (0, 0.5, 1) this is
        # `shift = 0, norm = 1, scale = borderCount`.
        self.norm = ctr_norm(prior)
        self.scale = ctr_predict_scale(ctr_border_count, self.norm)
        self.ctr_border_count = ctr_border_count

    def n_buckets(self) raises -> Int:
        """`GetBucketCount`'s CTR arm: `BorderCount + 1`, 16 at the default."""
        return ctr_bucket_count(self.ctr_border_count)


# ---------------------------------------------------------------------------
# The fitted state
# ---------------------------------------------------------------------------


struct CtrTables(Copyable, Movable):
    """The static tables a fitted model needs to score, plus the column layout.

    **This is model state, not configuration.** Every count in it was read off
    the target, so a model that lost it is a model that scores wrong -- silently,
    because the trees still reference the columns and the columns still get
    values. `CalcFinalCtrsAndSaveToModel` is where CatBoost puts the same thing
    (`online_ctr.h`), and CatBoost's `.cbm` carries it. mojotrees' text format
    does not yet: `serialize._write_mapper` writes edges, offsets, `n_features`,
    `n_bins` and the missing-bin table and nothing else, so a mapper carrying
    these would round-trip without them. `check_ctr_serializable` refuses by
    name, and `binning.BinMapper.attach_ctr` documents where the call has to go.

    Layout. Tables are indexed by **slot**, which is a position in
    `source_features` (the categorical base features that earned CTRs, ascending)
    and not a feature id. A slot's tables are:

    - `class_table[class_offsets[s] : class_offsets[s + 1]]`, laid out
      `bucket * n_classes + class`, which is the layout `ctr_calcer.py` indexes.
      Empty when no description of this fit needs a class table.
    - `mean_sums` / `mean_counts` over `mean_offsets`, the
      `BinarizedTargetMeanValue` pair.
    - `counter_counts[counter_offsets[s] : counter_offsets[s + 1]]` and
      `counter_denominator[s]`, the largest category's count
      (`online_ctr.cpp:935`).

    `columns` is the produced layout in `AllocateCtrData` order: source feature
    ascending, then description, then target border, then prior. Column `i` of
    this list is base feature `n_base_features + i` of the design matrix.
    """

    var active: Bool
    var n_base_features: Int
    var n_classes: Int
    var prior_denom: Float64
    var source_features: List[Int]
    var slot_buckets: List[Int]
    var columns: List[CtrColumn]
    var class_table: List[Int]
    var class_offsets: List[Int]
    var mean_sums: List[Float64]
    var mean_counts: List[Int]
    var mean_offsets: List[Int]
    var counter_counts: List[Int]
    var counter_offsets: List[Int]
    var counter_denominator: List[Int]

    var predict_lut: List[UInt8]
    """Column `c`'s bucket -> inference bin id, at
    `predict_lut[predict_lut_offsets[c] + bucket]`.

    Derived state, filled by `fit_ctr_tables` and changing no value: a CTR
    column's inference bin depends on the category bucket and on nothing else,
    because the tables are static. Precomputing it is what keeps
    `ctr_predict_row` a gather instead of a table walk per scored row -- without
    it, scoring one row would re-read `n_buckets * n_classes` counts per column.
    `ctr_predict_bin` is the reference this is built from and stays the
    definition.
    """

    var predict_lut_offsets: List[Int]
    """Start of column `c`'s lookup in `predict_lut`, length `n_columns + 1`."""

    def __init__(out self):
        self.active = False
        self.n_base_features = 0
        self.n_classes = 0
        self.prior_denom = CPU_PRIOR_DENOM
        self.source_features = List[Int]()
        self.slot_buckets = List[Int]()
        self.columns = List[CtrColumn]()
        self.class_table = List[Int]()
        self.class_offsets = List[Int]()
        self.mean_sums = List[Float64]()
        self.mean_counts = List[Int]()
        self.mean_offsets = List[Int]()
        self.counter_counts = List[Int]()
        self.counter_offsets = List[Int]()
        self.counter_denominator = List[Int]()
        self.predict_lut = List[UInt8]()
        self.predict_lut_offsets = List[Int]()

    @staticmethod
    def none() -> CtrTables:
        """The inactive value a mapper holds when CTRs are off.

        Fifteen empty containers and three scalars. It is the whole of the
        default arm's cost: no pass, no per-row allocation, and nothing that
        scales with the row or feature count.
        """
        return CtrTables()

    def is_active(self) -> Bool:
        return self.active

    def n_columns(self) -> Int:
        return len(self.columns)

    def n_slots(self) -> Int:
        return len(self.source_features)

    def total_features(self) -> Int:
        """Base features plus CTR columns: the width of the design matrix and of
        a binned row, and NOT the width of a raw row. A raw row stays
        `n_base_features` wide forever, which is what keeps every caller of
        `BinMapper.transform` and `bin_row` working unchanged."""
        return self.n_base_features + len(self.columns)

    def matches(self, other: CtrTables) raises -> Bool:
        """Whether two fitted CTR states score every row the same way.

        `BinMapper.matches` calls this, so `boosting.train_more` cannot take
        trees fitted against one CTR state onto a dataset carrying another --
        which would be the same class of silent error as a bin index meaning two
        different things.
        """
        if self.active != other.active:
            return False
        if not self.active:
            return True
        if self.n_base_features != other.n_base_features:
            return False
        if self.n_classes != other.n_classes:
            return False
        if self.prior_denom != other.prior_denom:
            return False
        if len(self.columns) != len(other.columns):
            return False
        for i in range(len(self.columns)):
            ref a = self.columns[i]
            ref b = other.columns[i]
            if a.slot != b.slot:
                return False
            if a.source_feature != b.source_feature:
                return False
            if a.ctr_type != b.ctr_type:
                return False
            if a.target_border_idx != b.target_border_idx:
                return False
            if a.prior != b.prior:
                return False
            if a.shift != b.shift:
                return False
            if a.norm != b.norm:
                return False
            if a.scale != b.scale:
                return False
            if a.ctr_border_count != b.ctr_border_count:
                return False
        if not _int_lists_equal(self.source_features, other.source_features):
            return False
        if not _int_lists_equal(self.slot_buckets, other.slot_buckets):
            return False
        if not _int_lists_equal(self.class_table, other.class_table):
            return False
        if not _int_lists_equal(self.class_offsets, other.class_offsets):
            return False
        if not _int_lists_equal(self.mean_counts, other.mean_counts):
            return False
        if not _int_lists_equal(self.mean_offsets, other.mean_offsets):
            return False
        if not _int_lists_equal(self.counter_counts, other.counter_counts):
            return False
        if not _int_lists_equal(self.counter_offsets, other.counter_offsets):
            return False
        if not _int_lists_equal(
            self.counter_denominator, other.counter_denominator
        ):
            return False
        if len(self.mean_sums) != len(other.mean_sums):
            return False
        for i in range(len(self.mean_sums)):
            if self.mean_sums[i] != other.mean_sums[i]:
                return False
        if not _int_lists_equal(
            self.predict_lut_offsets, other.predict_lut_offsets
        ):
            return False
        if len(self.predict_lut) != len(other.predict_lut):
            return False
        for i in range(len(self.predict_lut)):
            if self.predict_lut[i] != other.predict_lut[i]:
                return False
        return True


def _int_lists_equal(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def check_ctr_serializable(tables: CtrTables) raises:
    """Refuses to let fitted CTR state reach a writer that would drop it.

    The tables above are read off the target and a model without them scores
    wrong rather than failing, which is the one outcome worth refusing loudly
    for. `serialize._write_mapper` writes `n_features`, `n_bins`, the edges,
    the edge offsets and the missing-bin table; `_write_categorical` writes the
    category tables. Neither writes a CTR section, and neither reader reads one,
    so a saved CTR model would load with `CtrTables.none()`, keep every tree that
    references a CTR column, and bin those columns as if they were absent.

    **This lane cannot install the call.** `serialize.mojo` belongs to the
    model-export lane (catalog A29), which is the lane that has to add the
    section, bump the format version, and put this call at the top of
    `save_model`, `save_multiclass_model` and `save_dataset`. Until it does, the
    refusal that actually holds is `ctr.check_ctr_trainer_support` at the trainer
    boundary in `trainset.mojo`, which stops a CTR-carrying model from being
    produced at all. This function is here so that the export lane has the exact
    guard to call and so that the requirement is written down in the file that
    owns the state.
    """
    if tables.is_active():
        raise Error(
            "a model carrying fitted CTR tables cannot be saved: the tables are"
            " model state read off the target (catalog A19), and the mojotrees"
            " text format has no ctr section, so the file would load without"
            " them and score wrong. The model-export lane (catalog A29) owns"
            " serialize.mojo and the format version"
        )


# ---------------------------------------------------------------------------
# Planning the columns
# ---------------------------------------------------------------------------


def ctr_source_features(
    config: SimpleCtrConfig,
    is_categorical: List[Bool],
    n_categories: List[Int],
) raises -> List[Int]:
    """The categorical columns that earn CTRs, ascending.

    A column earns them when it is **too wide to one-hot**:
    `uniqueValues > one_hot_max_size`. That is the test at
    `greedy_tensor_search.cpp:469` and the same quantity `IsPermutationNeeded`
    uses to decide whether a permutation exists at all
    (`CalcMaxCategoricalFeaturesUniqueValuesCountOnLearn() > OneHotMaxSize`,
    `learn_context.cpp:38`). A19 flags this coupling as running in the direction
    people do not expect: it is the *presence of a wide categorical column*, not
    a user's request, that turns CatBoost's permutation on.

    `n_categories[f]` here is the count of *kept* categories,
    `CategoricalSpec.n_categories`, which is at most `max_bin - 1`. A column
    whose rare levels were evicted therefore looks narrower than the raw data
    was, and can fall under the one-hot cutoff where CatBoost's would not. That
    is a divergence and it is recorded rather than patched, because the eviction
    is `fit_categorical_spec`'s existing behavior and the bucket a CTR reads is
    the kept bucket either way.
    """
    var out = List[Int]()
    if not config.enabled:
        return out^
    if len(is_categorical) != len(n_categories):
        raise Error(
            "is_categorical and n_categories must have one entry per feature"
        )
    for f in range(len(is_categorical)):
        if not is_categorical[f]:
            continue
        if n_categories[f] > config.one_hot_max_size:
            out.append(f)
    return out^


def plan_ctr_columns(
    config: SimpleCtrConfig,
    is_categorical: List[Bool],
    n_categories: List[Int],
) raises -> CtrTables:
    """The column layout, with the tables still empty.

    Order is `AllocateCtrData`'s (`online_ctr.cpp:741`): source feature
    ascending, then description in the order the config lists them, then target
    border, then prior. At the CPU defaults with a binary target that is, per
    categorical column,

        (Borders, border 0, prior 0.0)
        (Borders, border 0, prior 0.5)
        (Borders, border 0, prior 1.0)
        (Counter, border 0, prior 0.0)

    -- the four of A19's headline count, in that order, and `ctr_feature_count`
    computes the same product independently.

    `n_buckets` for a slot is `n_categories + 1`: mojotrees numbers a kept
    category `i` as bin `i + 1` and reserves bin 0 (`categorical.UNKNOWN_BIN`)
    for missing, unknown and evicted values. Bucket 0 is therefore a real bucket
    with real statistics rather than CatBoost's absent-category case.
    """
    var tables = CtrTables()
    config.validate()
    if not config.enabled:
        return tables^
    var n_base = len(is_categorical)
    var sources = ctr_source_features(config, is_categorical, n_categories)
    var n_classes = config.n_target_classes()
    if n_classes < 1:
        raise Error("a ctr fit needs at least one target class")

    tables.active = True
    tables.n_base_features = n_base
    tables.n_classes = n_classes
    tables.prior_denom = config.prior_denom
    for s in range(len(sources)):
        var f = sources[s]
        tables.source_features.append(f)
        tables.slot_buckets.append(n_categories[f] + 1)
        for d in range(len(config.descriptions)):
            ref desc = config.descriptions[d]
            var n_borders = ctr_target_border_count(desc.ctr_type, n_classes)
            for border in range(n_borders):
                for p in range(len(desc.priors)):
                    tables.columns.append(
                        CtrColumn(
                            s,
                            f,
                            desc.ctr_type,
                            border,
                            p,
                            desc.priors[p],
                            desc.ctr_border_count,
                        )
                    )
    # A config that names no categorical column wide enough to escape the
    # one-hot cutoff produces no columns, and an active-but-empty state would
    # cost every later call an `is_active()` branch for nothing. Say inactive.
    if len(tables.columns) == 0:
        return CtrTables.none()
    return tables^


def ctr_columns_per_categorical(config: SimpleCtrConfig) raises -> Int:
    """How many numeric columns one categorical column produces.

    Four at CatBoost's CPU defaults with a binary target, computed from
    `ctr_target_border_count * n_priors` per description rather than asserted.
    """
    var n_classes = config.n_target_classes()
    var total = 0
    for d in range(len(config.descriptions)):
        ref desc = config.descriptions[d]
        total += ctr_target_border_count(desc.ctr_type, n_classes) * len(
            desc.priors
        )
    return total


# ---------------------------------------------------------------------------
# Fitting the static tables (the inference half)
# ---------------------------------------------------------------------------


def _slot_column(
    cat_bins: List[Int], n_rows: Int, slot: Int
) raises -> List[Int]:
    """One slot's category codes as their own list, which is the shape every
    `ctr.mojo` entry point takes."""
    var out = List[Int](capacity=n_rows)
    var base = slot * n_rows
    for r in range(n_rows):
        out.append(cat_bins[base + r])
    return out^


def _needs_class_table(tables: CtrTables) -> Bool:
    for i in range(len(tables.columns)):
        var t = tables.columns[i].ctr_type
        if t == CTR_BORDERS or t == CTR_BUCKETS:
            return True
    return False


def _needs_mean_table(tables: CtrTables) -> Bool:
    for i in range(len(tables.columns)):
        if tables.columns[i].ctr_type == CTR_BINARIZED_TARGET_MEAN:
            return True
    return False


def _needs_counter_table(tables: CtrTables) -> Bool:
    for i in range(len(tables.columns)):
        if tables.columns[i].ctr_type == CTR_COUNTER:
            return True
    return False


def fit_ctr_tables(
    mut tables: CtrTables,
    cat_bins: List[Int],
    n_rows: Int,
    target_class_of_row: List[Int],
) raises:
    """`CalcFinalCtrs` (`online_ctr.cpp:875`): the tables the model scores from.

    A plain loop over every learn row, **in dataset order, with no permutation
    and no prefix**. That is the whole difference from the training columns, and
    it is why the two formulas cannot share code. `ctr.final_ctr_class_table`,
    `final_ctr_mean_table` and `final_ctr_counter_table` are the three loops and
    each is a transcription of one arm of `CalcFinalCtrsImpl`.

    `count_rows` is the learn row count, because `counter_calc_method` is pinned
    to `SkipTest` (`SimpleCtrConfig.validate` refuses `Full`, which would count
    the test sets a `Dataset` does not hold).

    Only the table kinds this fit's descriptions actually read are built; the
    others stay empty, so a `Counter`-only fit pays no class table and a
    `Borders`-only fit pays no counter table.
    """
    if not tables.is_active():
        return
    var n_slots = tables.n_slots()
    if len(cat_bins) != n_slots * n_rows:
        raise Error("cat_bins length must equal n_slots * n_rows")
    if len(target_class_of_row) != n_rows:
        raise Error("target class length must match the row count")

    var want_class = _needs_class_table(tables)
    var want_mean = _needs_mean_table(tables)
    var want_counter = _needs_counter_table(tables)

    tables.class_table = List[Int]()
    tables.class_offsets = List[Int]()
    tables.mean_sums = List[Float64]()
    tables.mean_counts = List[Int]()
    tables.mean_offsets = List[Int]()
    tables.counter_counts = List[Int]()
    tables.counter_offsets = List[Int]()
    tables.counter_denominator = List[Int]()
    tables.class_offsets.append(0)
    tables.mean_offsets.append(0)
    tables.counter_offsets.append(0)

    for s in range(n_slots):
        var n_buckets = tables.slot_buckets[s]
        var column = _slot_column(cat_bins, n_rows, s)
        if want_class:
            var one = List[Int]()
            final_ctr_class_table(
                column,
                target_class_of_row,
                n_buckets,
                tables.n_classes,
                n_rows,
                one,
            )
            for i in range(len(one)):
                tables.class_table.append(one[i])
        tables.class_offsets.append(len(tables.class_table))

        if want_mean:
            var sums = List[Float64]()
            var counts = List[Int]()
            final_ctr_mean_table(
                column,
                target_class_of_row,
                n_buckets,
                tables.n_classes,
                n_rows,
                sums,
                counts,
            )
            for i in range(len(sums)):
                tables.mean_sums.append(sums[i])
                tables.mean_counts.append(counts[i])
        tables.mean_offsets.append(len(tables.mean_counts))

        if want_counter:
            var counts = List[Int]()
            final_ctr_counter_table(column, n_buckets, n_rows, counts)
            tables.counter_denominator.append(counter_denominator(counts))
            for i in range(len(counts)):
                tables.counter_counts.append(counts[i])
        else:
            tables.counter_denominator.append(0)
        tables.counter_offsets.append(len(tables.counter_counts))

    # The bucket -> bin lookup, built last because it reads every table above.
    # A CTR column's inference bin is a function of the category bucket alone --
    # the tables are static, which is the whole difference from the training
    # side -- so the answer for every bucket can be taken once here instead of
    # once per scored row. It changes no value; `ctr_predict_bin` is the
    # definition and this is a materialization of it.
    var lut = List[UInt8]()
    var lut_offsets = List[Int]()
    lut_offsets.append(0)
    for c in range(tables.n_columns()):
        var n_buckets = tables.slot_buckets[tables.columns[c].slot]
        for b in range(n_buckets):
            lut.append(UInt8(ctr_predict_bin(tables, c, b)))
        lut_offsets.append(len(lut))
    tables.predict_lut = lut^
    tables.predict_lut_offsets = lut_offsets^


# ---------------------------------------------------------------------------
# Building the training columns (the ordered half)
# ---------------------------------------------------------------------------


def ctr_train_permutation(
    config: SimpleCtrConfig, n_rows: Int
) raises -> List[Int]:
    """The one permutation this dataset's ordered columns are built from.

    `resolve_permutation_block_size` is `TFoldsCreationParams`' three-line rule
    (`learn_context.cpp:89`): an unset size becomes
    `min(256, n / 1000 + 1)` and an unpermuted fold gets the whole row count,
    which is CatBoost's way of writing the identity.

    `has_time` disables the permutation outright, matching `IsPermutationNeeded`'s
    first line, and the CTR degenerates to a prefix statistic in dataset order --
    the correct behavior for a genuinely time-ordered pool.

    The result is computed once, here, in full, before any column is built. Every
    loop below indexes it by position. That is what keeps `ctr_permutation`'s
    keyed-sort determinism intact at the call site: nothing consumes a stream,
    nothing draws in parallel, and no loop's answer depends on how many workers
    ran it.
    """
    var permuted = not config.has_time
    var block = resolve_permutation_block_size(
        config.permutation_block_size, n_rows, permuted
    )
    if not permuted:
        # `permutation_index = 0` is `ctr_permutation`'s identity branch, which
        # is exactly what an unpermuted fold means.
        return ctr_permutation(n_rows, block, config.seed, 0)
    return ctr_permutation(
        n_rows, block, config.seed, config.permutation_index
    )


def build_ctr_train_columns(
    tables: CtrTables,
    cat_bins: List[Int],
    n_rows: Int,
    target_class_of_row: List[Int],
    permutation: List[Int],
) raises -> List[UInt8]:
    """The training columns, column-major, `out[c * n_rows + r]`.

    One column per `CtrTables.columns` entry, each already a bucket index in
    `[0, ctr_border_count]` because `ctr.ctr_train_bin` returns what CatBoost's
    `CalcCTR` returns: a `ui8`. There is no binarization pass over CTR values
    here for the same reason there is none in CatBoost.

    Dispatch is `ComputeOnlineCTRs`' final `ExecRange` (`online_ctr.cpp:732`),
    which branches on the type **and** on the target class count:

    - `Borders` at exactly two classes -> `ordered_ctr_borders_binary`
      (`CalcOnlineCTRSimple`), the default path for a binary target
    - `Buckets`, and `Borders` above two classes -> `ordered_ctr_classes`
      (`CalcOnlineCTRClasses`)
    - `BinarizedTargetMeanValue` -> `ordered_ctr_mean` (`CalcOnlineCTRMean`)
    - `Counter` -> not ordered at all, see below

    **`Counter` is not permutation-dependent** (`IsPermutationDependentCtrType`
    returns false for it). `CalcOnlineCTRCounter` reads a table that
    `CountOnlineCTRTotal` filled once over the whole array *before any row was
    emitted*, and its denominator is the largest category's count, not the row
    count. So every row of a category gets the same value. That full count is
    taken here once per slot, before the column loop for that slot writes a
    single byte, and it is shared by every `Counter` prior of that slot; the
    permutation is not read on this arm at all. It is deliberately the same count
    `fit_ctr_tables` builds, because it is the same quantity -- train and predict
    differ for `Counter` only in that the training side truncates.
    """
    var out = List[UInt8]()
    if not tables.is_active():
        return out^
    var n_slots = tables.n_slots()
    if len(cat_bins) != n_slots * n_rows:
        raise Error("cat_bins length must equal n_slots * n_rows")
    if len(target_class_of_row) != n_rows:
        raise Error("target class length must match the row count")
    if len(permutation) != n_rows:
        raise Error("permutation length must match the row count")

    var n_cols = tables.n_columns()
    out.resize(n_cols * n_rows, 0)

    for s in range(n_slots):
        var n_buckets = tables.slot_buckets[s]
        var column = _slot_column(cat_bins, n_rows, s)

        # Filled lazily, once, and only if this slot has a Counter column.
        var counter_counts = List[Int]()
        var counter_denom = 0
        var counter_ready = False

        var bins = List[Int]()
        for c in range(n_cols):
            ref spec = tables.columns[c]
            if spec.slot != s:
                continue
            var dst = c * n_rows
            if spec.ctr_type == CTR_COUNTER:
                if not counter_ready:
                    final_ctr_counter_table(
                        column, n_buckets, n_rows, counter_counts
                    )
                    counter_denom = counter_denominator(counter_counts)
                    counter_ready = True
                # Every row of a category gets the same value, so the bucket
                # is a function of the bucket alone: compute it once per
                # category and gather. Derived, not measured, and it changes no
                # value.
                var lut = List[UInt8](capacity=n_buckets)
                for b in range(n_buckets):
                    lut.append(
                        UInt8(
                            ctr_train_bin(
                                Float64(counter_counts[b]),
                                counter_denom,
                                spec.prior,
                                spec.shift,
                                spec.norm,
                                spec.ctr_border_count,
                            )
                        )
                    )
                for r in range(n_rows):
                    out[dst + r] = lut[column[r]]
                continue
            if spec.ctr_type == CTR_BINARIZED_TARGET_MEAN:
                ordered_ctr_mean(
                    column,
                    target_class_of_row,
                    permutation,
                    n_buckets,
                    tables.n_classes,
                    spec.prior,
                    spec.ctr_border_count,
                    bins,
                )
            elif (
                spec.ctr_type == CTR_BORDERS
                and tables.n_classes == SIMPLE_CLASSES_COUNT
            ):
                ordered_ctr_borders_binary(
                    column,
                    target_class_of_row,
                    permutation,
                    n_buckets,
                    spec.prior,
                    spec.ctr_border_count,
                    bins,
                )
            else:
                ordered_ctr_classes(
                    column,
                    target_class_of_row,
                    permutation,
                    n_buckets,
                    tables.n_classes,
                    spec.ctr_type,
                    spec.target_border_idx,
                    spec.prior,
                    spec.ctr_border_count,
                    bins,
                )
            for r in range(n_rows):
                out[dst + r] = UInt8(bins[r])
    return out^


# ---------------------------------------------------------------------------
# Reading the static tables (the inference half)
# ---------------------------------------------------------------------------


def ctr_predict_bin(
    tables: CtrTables, column_index: Int, bucket: Int
) raises -> Int:
    """The inference bucket for one CTR column at one category bucket.

    **The definition.** `fit_ctr_tables` calls this once per (column, bucket)
    and stores the answers in `CtrTables.predict_lut`; `ctr_predict_row` and
    `ctr_predict_columns` then gather from that instead of coming back here, so
    scoring a row does not walk a table. This function is what the lookup means
    and the two gatherers are a materialization of it.

    Two steps, and keeping them apart is the point.

    1. The **value**, `TModelCtr::Calc` (`model/online_ctr.h:289`):
       `((c + PriorNum) / (t + PriorDenom) + Shift) * Scale`, unquantized, with
       `c` and `t` read from the static table over the whole learn set. This is a
       different formula over different data from the training one, which used
       the ordered prefix and the hard-coded denominator `t + 1`.
    2. The **bucket**, `ctr.ctr_predict_bucket`, which is `EmulateUi8Rounding`
       (`split.h:512`) solved for the bucket index. A tree here compares integer
       bin ids, so the epsilon that CatBoost puts on the *threshold* has to be
       put back on the *value*; `ctr_predict_bucket` documents the algebra.

    `Scale = borderCount / norm` (`split.cpp:81`) is why step 1's value is on the
    same scale as the training bucket index: the training multiply was folded
    into it, so the two produce the same real number from the same counts and
    only the training one truncated. They do not have the same counts, and are
    not supposed to.
    """
    if not tables.is_active():
        raise Error("this mapper carries no ctr tables")
    if column_index < 0 or column_index >= tables.n_columns():
        raise Error("ctr column index out of range")
    ref spec = tables.columns[column_index]
    var s = spec.slot
    var value: Float64
    if spec.ctr_type == CTR_COUNTER:
        var lo = tables.counter_offsets[s]
        var hi = tables.counter_offsets[s + 1]
        var counts = List[Int](capacity=hi - lo)
        for i in range(lo, hi):
            counts.append(tables.counter_counts[i])
        value = predict_ctr_counter(
            counts,
            tables.counter_denominator[s],
            bucket,
            spec.prior,
            tables.prior_denom,
            spec.shift,
            spec.scale,
        )
    elif spec.ctr_type == CTR_BINARIZED_TARGET_MEAN:
        var lo = tables.mean_offsets[s]
        var hi = tables.mean_offsets[s + 1]
        var sums = List[Float64](capacity=hi - lo)
        var counts = List[Int](capacity=hi - lo)
        for i in range(lo, hi):
            sums.append(tables.mean_sums[i])
            counts.append(tables.mean_counts[i])
        value = predict_ctr_mean(
            sums,
            counts,
            bucket,
            spec.prior,
            tables.prior_denom,
            spec.shift,
            spec.scale,
        )
    else:
        var lo = tables.class_offsets[s]
        var hi = tables.class_offsets[s + 1]
        var table = List[Int](capacity=hi - lo)
        for i in range(lo, hi):
            table.append(tables.class_table[i])
        value = predict_ctr_class(
            table,
            tables.n_classes,
            bucket,
            spec.ctr_type,
            spec.target_border_idx,
            spec.prior,
            tables.prior_denom,
            spec.shift,
            spec.scale,
        )
    return ctr_predict_bucket(value, spec.ctr_border_count)


def ctr_predict_row(
    tables: CtrTables, base_bins: List[Int]
) raises -> List[Int]:
    """The CTR bins for one scored row, given its base feature bins.

    `base_bins` is what `BinMapper.bin_row` produced for the raw features; a CTR
    column reads `base_bins[source_feature]`, which is a category bucket in
    `[0, n_categories]` with 0 meaning missing, unknown or evicted. The result is
    appended to `base_bins` by the caller, so a scored row's bin vector is
    `n_base_features + n_columns` long while its raw row stays
    `n_base_features` long.
    """
    var out = List[Int]()
    if not tables.is_active():
        return out^
    if len(base_bins) < tables.n_base_features:
        raise Error("base bins must cover every base feature")
    if len(tables.predict_lut_offsets) != tables.n_columns() + 1:
        raise Error(
            "these ctr tables were never fitted: call fit_ctr_tables before"
            " scoring through them"
        )
    out.reserve(tables.n_columns())
    for c in range(tables.n_columns()):
        ref spec = tables.columns[c]
        var bucket = base_bins[spec.source_feature]
        var lo = tables.predict_lut_offsets[c]
        var hi = tables.predict_lut_offsets[c + 1]
        if bucket < 0 or bucket >= hi - lo:
            raise Error("category bucket out of range for this ctr table")
        out.append(Int(tables.predict_lut[lo + bucket]))
    return out^


def ctr_predict_columns(
    tables: CtrTables, cat_bins: List[Int], n_rows: Int
) raises -> List[UInt8]:
    """The CTR columns for a whole scored matrix, column-major.

    `cat_bins` is slot-major, `cat_bins[s * n_rows + r]`, the same shape
    `build_ctr_train_columns` takes -- so the *only* difference between the two
    calls is which formula runs over it, which is the asymmetry stated once and
    then held by the type system's absence rather than by a comment.

    The per-bucket value depends on nothing but the bucket, so it is computed
    once per (column, bucket) into a lookup of `n_buckets` entries and then
    gathered per row. That is `n_buckets` calls to `ctr_predict_bin` and `n_rows`
    loads per column, instead of `n_rows` calls -- a derived reduction in work,
    not a measured one, and it changes no value.
    """
    var out = List[UInt8]()
    if not tables.is_active():
        return out^
    var n_slots = tables.n_slots()
    if len(cat_bins) != n_slots * n_rows:
        raise Error("cat_bins length must equal n_slots * n_rows")
    var n_cols = tables.n_columns()
    if len(tables.predict_lut_offsets) != n_cols + 1:
        raise Error(
            "these ctr tables were never fitted: call fit_ctr_tables before"
            " scoring through them"
        )
    out.resize(n_cols * n_rows, 0)
    for c in range(n_cols):
        ref spec = tables.columns[c]
        var n_buckets = tables.slot_buckets[spec.slot]
        var lo = tables.predict_lut_offsets[c]
        var src = spec.slot * n_rows
        var dst = c * n_rows
        for r in range(n_rows):
            var bucket = cat_bins[src + r]
            if bucket < 0 or bucket >= n_buckets:
                raise Error("category bucket out of range for this ctr table")
            out[dst + r] = tables.predict_lut[lo + bucket]
    return out^
