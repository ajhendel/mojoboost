"""Feature binning.

Maps raw feature values to small integer bins so that split finding can
operate on fixed-size histograms instead of sorted feature values.

`fit_bins` performs quantile (equal-frequency) binning, LightGBM style,
and returns a `BinMapper` whose stored edges let a trained model bin raw,
unseen feature values at prediction time. `bin_equal_width` remains as a
simple mapper-free alternative for experiments.

Ties
----
A quantile boundary is a rank, and a rank can land in the middle of a run of
equal values. When it does, the edge is cut where that run *ends*, not
skipped: `emit_quantile_edges` takes the next distinct value above the
boundary's value, and the strictly-increasing filter collapses the many
boundaries sharing one run down to the single edge that run earns.

This is the difference between a low-cardinality column being usable and
being invisible. A balanced binary column of 100k rows has 254 boundaries and
all of them land inside one of its two runs, so cutting only at ranks that
straddle a change gave it no edges at all, one bin, and no split any tree
could ever make on it; the same was true of a sparse column whose implicit
zeros outnumber its stored values. Columns of distinct values are unaffected
(the next distinct value above rank `idx - 1` is the value at rank `idx`), and
so are columns with at most `max_bins` values (every rank is a boundary, so
the run's end is one too).

Levels
------
The tie rule finds a run once a boundary lands in it, and a boundary is a
rank, so it cannot find a level too rare to hold one. Three rows in sixty
thousand sit between two boundaries however far their value is from anything
else, so they were binned with their neighbours: the level was in the data
and absent from the model.

So before any boundary is computed, a column is asked how many distinct
values it has. If there are no more of them than the feature has ordinary
bins, the boundaries are not consulted at all and the levels are walked in
order, cutting once a bin has accumulated `min_data_in_bin` rows. That is
LightGBM's `GreedyFindBin` in the `num_distinct_values <= max_bin` case. At
`min_data_in_bin` of 1 it is one bin per level whatever each level's
population; at the default of 3, which is LightGBM's, levels holding one or
two rows merge into their neighbour and levels holding three or more still
each take a bin of their own (see `min_data_in_bin` below, and
docs/LIGHTGBM_PARITY.md).

`collect_distinct` answers the question for the dense fit and
`distinct_levels_sorted` for the sparse one, which has sorted its stored
values already. The two are required to agree, because a sparse matrix and
its dense form must bin identically, and they are tested against each other.
**They do not agree at the default any more**: `sparse.fit_bins_csc` takes
neither `min_data_in_bin` nor `bin_construct_sample_cnt`, so it still fits at
1 and at every row. That is a live gap, not a design, and it is recorded in
docs/LIGHTGBM_PARITY.md.

A column with more levels than bins is refused within its first few hundred
rows, so an ordinary continuous column pays a few hundred probes to ask and
then takes the quantile path exactly as before, edges unchanged bit for bit.
Measured at 250,000 x 100 continuous, four workers, that cost 0.0276 s
against 0.0270 s without asking, which is inside the run-to-run spread. On a
column that does have few levels the levels path was much the cheaper of the
two when it was measured, because the boundary path pays for a bucket table
and a second pass to resolve its ties and this one is a single scan. **Those
low-cardinality numbers were taken at `min_data_in_bin` of 1 and no longer
describe the default.** The scan now also counts each level (`counts` out of
`collect_distinct`), which is one L1 increment per row and no second pass, so
the advantage should survive; the size of it has to be re-measured before it
is quoted again, and this module quotes no ratio for it.

Full parity is still a further step, and stops at the same place it did.
Past the bin budget LightGBM keeps counting, and spends the counts: a value
populous enough to deserve a bin of its own takes one (`is_big_count_value`),
and the target bin size is recomputed from what budget is left as it goes.
This binner uses quantile boundaries there, capped as LightGBM caps them (see
`min_data_in_bin` below), and does not implement the big-count rule.

min_data_in_bin
---------------
LightGBM's minimum population for a numerical bin, verified against
`GreedyFindBin` in `src/io/bin.cpp` on LightGBM master rather than against
any description of it. It enters in two places and this binner honors both:

- In the `num_distinct_values <= max_bin` branch, LightGBM walks the levels
  in order accumulating their counts and cuts only once the accumulator has
  reached `min_data_in_bin`, resetting it at each cut. So adjacent levels are
  merged until they hold enough rows between them, and the last bin may hold
  fewer (nothing merges backwards). The levels path here does exactly that.
- In the other branch, LightGBM first shrinks the budget itself:
  `max_bin = max(1, min(max_bin, total_cnt / min_data_in_bin))`. The quantile
  path here applies the same cap to `n_ordinary`, which puts at least
  `min_data_in_bin` values in the average bin. What it does *not* do is
  LightGBM's per-level greedy inside that budget, which is the same gap the
  paragraph above names.

The default is LightGBM's 3. 1 is still reachable and is the value at which
**both branches are the code they were before this option existed**: at 1
every level's count already clears the accumulator, so the levels path cuts
between every adjacent pair, and `total_cnt / 1` never binds because the
quantile branch is only reached when there are more distinct values than
bins. Both are guarded on `min_data_in_bin <= 1` so that setting runs the
same instructions rather than the same arithmetic, and
`tests/test_binning.mojo` proves a fit at 1 with `bin_construct_sample_cnt=0`
is edge for edge the fit this module produced before either option existed.

What 3 does to the number of bins, as a bound rather than a measurement. Let
a column have `L` levels holding `N` rows between them. At 1 it gets `L`
bins. At 3 every closed bin except possibly the last holds at least 3 rows,
and no closed bin can absorb more than 3 levels (three levels hold at least
three rows), so

    ceil(L / 3)  <=  bins  <=  min(L, floor(N / 3) + 1)

and the merge reaches **only levels holding one or two rows**: a level of 3
or more closes its bin on its own. A column of 20 levels in 250,000 rows is
untouched. A column of 136 singleton levels in 900 rows goes from 136 bins to
about 46. Sparse and near-unique integer columns move; ordinary
low-cardinality ones do not.

Counting the levels is not a pass any more. `collect_distinct` returns a
count per level from the scan it was already doing, at one L1 increment per
row, because at a default of 3 a separate `count_levels` pass would be paid
by every low-cardinality column rather than by the few that asked for the
option. `count_levels` remains as the independent reference the fused counts
are tested against.

Fitting from a sample
---------------------
`bin_construct_sample_cnt` fits the edges from a subsample of the rows rather
than from all of them, which is what LightGBM does. The default is LightGBM's
200,000. 0 means every row, which is what this module used to default to; a
positive value at or above `n_rows` also means every row, so nothing below
200,000 rows is sampled at all.

What the sample does and does not save, as arithmetic rather than a claim.
The matrix is column-major and the sample is a set of *rows*, so a sampled
column is a strided gather, not a shorter read. With a sampling rate `p` and
eight doubles to a 64-byte line, the fraction of lines a column still has to
touch is `1 - (1 - p)^8`: at 200,000 of 1,000,000 rows that is 83 percent of
the bytes for 20 percent of the values, and at 200,000 of 10,000,000 it is 15
percent of the bytes for 2 percent of the values. So the saving is in the
per-value work -- the `isnan` test, the distinct probe, and the rank
selection or sort that follows -- and only becomes a saving in memory traffic
well below a fifth. The gather also costs an indirection per value and
defeats the unit-stride prefetch the full read gets.

The sample is a set of row indices drawn **once per fit**, before any feature
is touched, and every feature is fit from the same rows -- LightGBM samples
rows, not values. It is drawn by Knuth's selection sampling with a
counter-based splitmix64 draw per row (`rng.uniform(seed * GOLDEN + r)`), so
the draw for row r does not depend on how many rows were selected before it,
the result is exactly `bin_construct_sample_cnt` indices in ascending order,
and it is identical at every worker count and on every machine on this
toolchain. It does not reproduce LightGBM's own indices: LightGBM draws from
a 32-bit LCG in `include/LightGBM/utils/random.h` seeded by
`data_random_seed`, and matching that stream is a separate promise this
module does not make.

Two consequences, both of them LightGBM's too, and both real:

- A level that no sampled row carries is not a level the fit can see, so a
  rare value can lose the bin it would have had. That is the whole trade.
- Missingness is decided from the sample as well: a column whose only `NaN`s
  fall outside the sample reserves no missing bin, and `transform` then bins
  those `NaN`s as 0.0 (the `missing_type = None` rule below).

Categorical features are unaffected: `categorical.fit_categorical_spec` fits
its tables from the full column, so a sampled fit cannot lose a category.

feature_pre_filter
------------------
LightGBM's, and off here rather than on. At `True`, `fit_bins` counts each
feature's bins on the same sample it fit the edges from and drops the features
that have no usable split at all: `need_filter` is `NeedFilter` from
`src/io/bin.cpp` and `filter_count` is the `filter_cnt` of
`src/io/dataset_loader.cpp`, which is `min_data_in_leaf` **scaled to the
sample** and truncated (`min_data_in_leaf * total_sample_size / num_data`, so 4
at the stock 20, 200,000 and 1,000,000, not 20). The survivors are
`BinMapper.usable`, which is LightGBM's `used_features`, and `transform`
carries the list onto the `BinnedMatrix` because a grower is handed the matrix
and nothing else.

Dropping is not skipping. LightGBM removes the feature from the Dataset, so it
is gone from the pool `feature_fraction` samples and the fraction is taken of
what is left (`src/treelearner/col_sampler.hpp`), which changes the trees and
not only their cost. `sampling.select_tree_features` takes the pool for exactly
that reason. What is *not* renumbered is the matrix: every column is still
binned and every feature keeps its id, so a dropped feature reads 0 in an
importance vector rather than vanishing from it, which is also what LightGBM
does (`GBDT::FeatureImportance` sizes its result by `num_total_features`).

Three things here are not LightGBM's, and the parity contract says so:

- The default. LightGBM's is `true`; this is `False`, because `False` has to
  remain the fit that preceded the option.
- At `False` mojotrees keeps one-bin features, which LightGBM drops from
  `used_features` whatever the flag says. So `False` is "keep everything",
  not LightGBM's `false`.
- All features filtered raises here. LightGBM warns and continues.

`sparse.fit_bins_csc` and `external_memory` build their own mappers and do not
take the flag, so a prefiltered fit is the dense path only.

Missing values
--------------
`NaN` is the missing marker for numerical features. A feature whose training
column contains at least one `NaN` reserves one extra bin, immediately above
its ordinary bins, for missing values; `missing_bin[f]` is that bin's index,
or -1 for a feature that reserves none. Reserving costs that feature one
ordinary bin out of the `max_bins` budget, LightGBM style. `NaN` never
reaches the ordinary quantile comparisons: it is routed to the reserved bin
before the binary search runs, and it is left out of the quantile
computation when the edges are fit.

A `NaN` presented at prediction time for a feature with no reserved bin is
binned as the value 0.0, which is what LightGBM does for a feature whose
`missing_type` is `None`. `use_missing=False` reserves nothing for any
feature, so every `NaN` is binned as 0.0, again matching LightGBM.

`+inf` and `-inf` are ordinary finite-side extremes, not missing values: bin
edges are clamped to +/-1e300 (LightGBM's `Common::AvoidInf`), so `+inf`
always lands in a feature's highest ordinary bin and `-inf` in bin 0.

Categorical features are not covered by any of this: they reserve no missing
bin and keep `missing_bin[f] = -1`, because `categorical.mojo` already sends
`NaN`, negatives, and unseen codes to its own bin 0.

Features named in `categorical_features` are binned by `categorical.mojo`
instead: they get no quantile edges at all, and their raw integer codes map
to bins through a fitted category table. The two paths never mix, so adding
a categorical column changes nothing about how the numerical columns are
binned.

Fitting and transforming parallelize across features: every feature's edge
computation (column copy + rank resolution) and bin assignment is independent
and writes only its own output range, so workers need no synchronization and
the result is bit-identical to the serial path. Inputs too small to
amortize task-scheduling overhead stay serial.

How the quantile boundaries are found
-------------------------------------
A quantile boundary needs two order statistics of the column, the values at
ranks `idx - 1` and `idx`, and there are at most `max_bins - 1` boundaries. A
full sort answers every possible rank, which is `n log n` per feature to
answer at most `2 * (max_bins - 1)` questions; above `SELECT_MIN_ROWS`
non-missing values `resolve_ranks` answers exactly those questions instead,
in two linear passes over the column:

1. Count every value into a bucket of an order-preserving 64-bit key,
   rebased on the column's own key range so the buckets straddle the values
   that are actually there rather than the whole double line. The same pass
   records whether a bucket saw one key or several.
2. Prefix-sum the counts to give each bucket its global rank range. A bucket
   holding no requested rank is dropped. A bucket holding one repeated key
   answers its ranks straight from that key, which is what makes constant,
   binary, and other low-cardinality columns cost nothing beyond the two
   passes. Only the rest are gathered into one compact buffer.
3. Resolve each gathered bucket: a small one is sorted, a large one is
   bucketed again on its own narrower key range, to `SELECT_MAX_DEPTH`.

The result is the same value at the same rank a full sort would have put
there, so the edges are identical; the module-level `sort` path is kept for
small columns and as the reference the tests compare against. Sorting is
still the worst case, in two places: a segment that reaches the depth budget,
and a level that would have gathered more than a quarter of what it was given
(which means its buckets separated nothing). Both sort one segment, which is
never more work than sorting the whole column was, and the quarter rule is
also what bounds the extra memory -- see `resolve_ranks`.

`emit_quantile_edges` is the single authority on how a resolved boundary
becomes an edge (midpoint, `_avoid_inf` clamp, strictly-increasing filter),
and `quantile_boundary_indices` on which boundaries exist at all. The dense
fast path, the dense sort path, and `sparse.fit_bins_csc` all go through
both, so there is one rule rather than three copies of it.

Forced splits
-------------
`map_forced_splits` turns a parsed forced-split tree (see
tree_parameters_extra.mojo) into bin space. It lives here because that is
where the fitted binning lives: the grower is handed a `BinnedMatrix`, which
carries bins and no edges, so a raw threshold has to be resolved before it
reaches growth.

Work estimates. The threshold that decides serial from parallel is stated in
histogram-op equivalents (see `parallel.plan_tasks`), and a row of binning is
not one of those. Fitting orders each column, so its estimate carries a
comparison count (the sort's, kept unchanged when the cheaper selected path
took over, so the measured serial/parallel crossover stays where it was
measured); transforming searches each value, so its estimate carries the
search depth. Counting one op per (feature, row) here would have
kept a stage that is 8 to 20 times dearer per row than a histogram
accumulate on the serial path well past the point where it pays for a
dispatch. None of this moves an edge or a bin: it decides only how the same
work is spread.
"""

from std.math import isnan, log
from std.memory import bitcast
from std.os import getenv

from .categorical import CategoricalSpec, fit_categorical_spec

# Ordered target statistics (catalog A19), the columns half. `ctr_columns`
# imports `.ctr` and `.ctr_combinations`; `.ctr` imports `.rng`; `.rng` imports
# nothing. So the whole transitive set added here is `{ctr_columns, ctr,
# ctr_combinations, rng}` and `rng` was already imported below -- nothing in
# that set reaches back into `binning`, `categorical` or
# `tree_parameters_extra`, so the `efb -> binning -> tree_parameters_extra`
# cycle is untouched. This was checked in the direction that matters: no file
# under `src/mojotrees` imports `binning` *from* `ctr*`, and `ctr_columns`
# deliberately takes `List[Bool]`/`List[Int]` rather than a `CategoricalSpec`
# so that it never needs `.categorical` (which does import
# `tree_parameters_extra`).
from .ctr_columns import (
    CtrTables,
    ctr_predict_columns,
    ctr_predict_row,
)
from .parallel import (
    _env_int,
    dispatch_feature_ranges,
    dispatch_feature_rows,
    dispatch_features,
    dispatch_rows,
)
from .rng import GOLDEN, uniform
from .tree_parameters_extra import ForcedSplits


def _log2_ceil(n: Int) -> Int:
    """Smallest k with `2 ** k >= n`, and 0 for n <= 1. Used only to weight
    work estimates, so an off-by-one costs a scheduling decision at worst."""
    var k = 0
    var v = 1
    while v < n:
        v += v
        k += 1
    return k

# LightGBM's Common::kMaxDouble: bin edges are clamped here so an infinite
# training value cannot produce an infinite (or non-increasing) edge.
comptime MAX_EDGE = 1e300

# A bin index is stored in a byte (`BinnedMatrix.bins` is `List[UInt8]`), so
# no binning may reserve more than this many bins. Every path that produces
# a `BinMapper` has to hold the ceiling, because `BinMapper.transform`
# narrows to `UInt8` while `BinMapper.bin_value` returns an `Int`: above it
# the two disagree silently, by a whole leaf rather than by a rounding step.
comptime MAX_BINS = 256

comptime DEFAULT_MIN_DATA_IN_BIN = 3
"""`fit_bins`'s minimum population for a numerical bin. LightGBM's default.

This was 1 until the stock-defaults decision, and 1 is still the value at
which the levels path cuts between every adjacent level and the quantile
budget is never capped. It is reachable, and `tests/test_binning.mojo` proves
that a fit at `min_data_in_bin=1` with `bin_construct_sample_cnt=0` is edge
for edge the fit this module produced before either option existed. What
changed is which of the two a caller who says nothing gets: mojotrees's
defaults are LightGBM's, because a default is what a user actually
experiences and a comparison between two libraries at their own defaults is
the only one a user can act on."""

comptime DEFAULT_BIN_CONSTRUCT_SAMPLE_CNT = 200_000
"""Rows `fit_bins` fits its edges from. LightGBM's default; 0 means every row.

Was 0. See `DEFAULT_MIN_DATA_IN_BIN` for why it is LightGBM's number now.
Below 200,000 rows this is the full-column fit either way, because a sample
that would cover the matrix is not drawn at all."""

comptime DEFAULT_DATA_RANDOM_SEED = 1
"""LightGBM's `data_random_seed`, which seeds its bin-construction sample.
The name and the default are carried over; the stream is not (see the module
docstring). Fixed, never derived from a clock or a global, so the sampled fit
that is now the default is the same fit on every machine and at every
`MOJOTREES_NUM_WORKERS`."""

comptime DEFAULT_MIN_DATA_IN_LEAF = 20
"""LightGBM's `min_data_in_leaf` default, repeated here because `filter_count`
scales it and this module cannot import the grower's parameter bundle without
closing the `efb -> binning -> tree_parameters_extra` cycle. A caller who sets
`min_data_in_leaf` must pass the same number to `fit_bins`, exactly as a
LightGBM user must reconstruct the Dataset after changing it."""


def filter_count(
    min_data_in_leaf: Int, total_sample_size: Int, num_data: Int
) raises -> Int:
    """LightGBM's `filter_cnt`, the row count the prefilter tests against.

    Verified against `src/io/dataset_loader.cpp`, which computes it twice
    with the same expression (`DatasetLoader::ConstructBinMappersFromTextData`
    and `DatasetLoader::CostructFromSampleData`):

        const data_size_t filter_cnt = static_cast<data_size_t>(
          static_cast<double>(config_.min_data_in_leaf * total_sample_size)
          / num_dist_data);

    So it is `min_data_in_leaf` scaled from the whole dataset down to the
    bin-construction sample, truncated toward zero, and *not* `min_data_in_leaf`
    itself. The multiplication is integer and only the division is floating
    point, which is reproduced here rather than approximated: at
    `min_data_in_leaf=20`, a 200,000-row sample and 1,000,000 rows it is 4, and
    at `total_sample_size == num_data` -- every fit that reads every row, which
    is every fit below `bin_construct_sample_cnt` rows -- it is exactly
    `min_data_in_leaf`.

    A zero or negative `num_data` has no scaling to do and is refused rather
    than divided by.
    """
    if num_data < 1:
        raise Error("num_data must be positive")
    if total_sample_size < 0:
        raise Error("total_sample_size must be nonnegative")
    if min_data_in_leaf < 0:
        raise Error("min_data_in_leaf must be nonnegative")
    return Int(
        Float64(min_data_in_leaf * total_sample_size) / Float64(num_data)
    )


def need_filter(
    cnt_in_bin: List[Int],
    total_cnt: Int,
    filter_cnt: Int,
    categorical: Bool = False,
) -> Bool:
    """LightGBM's `NeedFilter` from `src/io/bin.cpp`, transcribed.

    True when the feature has no usable split at all: no prefix of its bins
    leaves at least `filter_cnt` rows on both sides. LightGBM's loop stops one
    short of the last bin, so the last bin's population never enters a prefix
    and a reserved missing bin (which numerical binning puts last) only ever
    contributes to `total_cnt`.

    Categorical features take LightGBM's other branch: a categorical split is
    one category against the rest rather than a prefix, so with more than two
    bins there is always some partition to try and the answer is False without
    looking. With at most two bins the single bin (not the prefix) is tested,
    which for two bins is the same arithmetic.

    Note that the prefix sums are non-decreasing, so `total_cnt - sum_left` is
    non-increasing: the first prefix to reach `filter_cnt` is also the one with
    the most rows left on the right, and no later prefix can succeed where it
    failed. The loop is kept in LightGBM's shape anyway, because the bin counts
    are at most 256 numbers and a transcription is worth more here than the
    early exit.
    """
    var n = len(cnt_in_bin)
    if n < 2:
        return True
    if not categorical:
        var sum_left = 0
        for i in range(n - 1):
            sum_left += cnt_in_bin[i]
            if sum_left >= filter_cnt and total_cnt - sum_left >= filter_cnt:
                return False
        return True
    if n > 2:
        return False
    for i in range(n - 1):
        var sum_left = cnt_in_bin[i]
        if sum_left >= filter_cnt and total_cnt - sum_left >= filter_cnt:
            return False
    return True


def _avoid_inf(x: Float64) -> Float64:
    """LightGBM's `Common::AvoidInf`, clamping an edge into +/-1e300."""
    if isnan(x):
        return 0.0
    if x >= MAX_EDGE:
        return MAX_EDGE
    if x <= -MAX_EDGE:
        return -MAX_EDGE
    return x


comptime POSITIVE_INF = bitcast[DType.float64, 1](
    SIMD[DType.uint64, 1](0x7FF0000000000000)
)
"""`+inf` as a padding sentinel: `POSITIVE_INF < v` is false for every `v`
a search can be handed, `+inf` included, so a padded search table cannot
change a bin. Spelled from its bit pattern so it needs no library constant."""


# --------------------------------------------------------------------------
# Quantile boundaries: which ones exist, and how one becomes an edge.
# --------------------------------------------------------------------------


def quantile_boundary_indices(
    n_valid: Int, n_ordinary: Int, mut idxs: List[Int]
):
    """The rank `idx` of every quantile boundary that exists, ascending.

    Boundary `b` of `n_ordinary` bins sits at `idx = b * n_valid //
    n_ordinary`, and it exists only when `0 < idx < n_valid`: a boundary at
    rank 0 or at rank `n_valid` has nothing on one side of it. `idx` is
    non-decreasing in `b` and repeats are kept, because two boundaries
    landing on the same rank is exactly the duplicate case
    `emit_quantile_edges` filters, and dropping them here would change which
    edge is compared against which.

    The single authority on where the boundaries are. Every fitter, dense or
    sparse, exact or selected, asks this rather than recomputing the
    expression, so a boundary means one thing everywhere.
    """
    idxs.clear()
    for b in range(1, n_ordinary):
        var idx = b * n_valid // n_ordinary
        if idx <= 0 or idx >= n_valid:
            continue
        idxs.append(idx)


def emit_quantile_edges(
    below: List[Float64], above: List[Float64], mut out: List[Float64]
):
    """Turn resolved boundaries into strictly increasing bin edges.

    `below[j]` is the value at rank `idx - 1` of the j-th boundary
    `quantile_boundary_indices` reported, and `above[j]` is the smallest
    value in the column that is strictly greater than it, or `below[j]` again
    when there is none. An edge is the midpoint of the two, clamped by
    `_avoid_inf`.

    Why `above` is the next distinct value and not the value at rank `idx`.
    A boundary landing inside a run of equal values has no gap *at that rank*,
    but the column still has a gap: the run has to end somewhere. Cutting at
    the end of the run is what a tied quantile means, and taking the value at
    rank `idx` instead dropped the boundary entirely, which is how a balanced
    binary column used to come out of a 255-bin fit with no edges at all and
    therefore one unsplittable bin. The two rules agree exactly whenever the
    old one worked: with no ties `above` is the value at rank `idx`, and when
    every rank is a boundary (`n_valid <= n_ordinary`) the run's own end is
    also a boundary and the filter below collapses the pair.

    Two boundaries still produce no edge. One whose `below` is the column
    maximum has nothing above it, and one whose clamped midpoint fails to
    exceed the last edge kept would break the strictly-increasing invariant
    `bin_value` and `transform` search against; the second is also what
    collapses the many boundaries that share a run down to the one edge that
    run earns.

    The single authority on the edge rule; `out` is cleared first, so a
    caller can hand it the same reusable buffer for every feature.
    """
    # Non-raising, because the dense fitter calls this from inside a worker
    # closure that cannot raise. The two arrays describe the same boundaries
    # by construction; the clamp is here so a caller that got that wrong
    # loses boundaries rather than reading past the end of an array.
    var n = len(below)
    if len(above) < n:
        n = len(above)
    out.clear()
    for j in range(n):
        var lo = below[j]
        var hi = above[j]
        if hi <= lo:
            continue
        var edge = _avoid_inf((lo + hi) / 2.0)
        if len(out) > 0 and edge <= out[len(out) - 1]:
            continue
        out.append(edge)


# --------------------------------------------------------------------------
# CatBoost border selection. An ADDITIONAL mode, never a replacement.
#
# Everything below is opt-in and unreachable at `border_type=BORDER_QUANTILE`,
# which is `fit_bins`'s default and is the LightGBM `GreedyFindBin` port this
# file has always been. Verified against CatBoost `master`, read 2026-08-16;
# see `docs/design/CATBOOST_CATALOG.md` A15 for the full citation list and for
# the divergences. Line numbers below are into
# `library/cpp/grid_creator/binarization.cpp` unless another file is named.
#
# The one fact worth carrying in your head. CatBoost's `border_count` counts
# THRESHOLDS and this repo's `max_bin` counts BINS, so CatBoost's default 254
# and our 255 are the same 255-bin budget, and `max_borders` below is
# `n_ordinary - 1`. Given that, `GreedyLogSum` and our quantile fit produce
# the SAME NUMBER of borders, `min(max_borders, distinct - 1)`, and differ
# only in where they sit. The catalog shows why there is no third outcome.
# --------------------------------------------------------------------------

comptime BORDER_QUANTILE = 0
"""Ours: LightGBM's `GreedyFindBin`, equal-frequency. The default, and the
only value that leaves `fit_bins` on the instructions it ran before this
mode existed."""

comptime BORDER_GREEDY_LOG_SUM = 1
"""CatBoost `GreedyLogSum`, its own default (`MakeBinarizer`, :118)."""

comptime BORDER_GREEDY_MIN_ENTROPY = 2
"""CatBoost `GreedyMinEntropy` (`MakeBinarizer`, :120)."""

comptime BORDER_UNIFORM = 3
"""CatBoost `Uniform` (`TUniformBinarizer`, :1262)."""

comptime BORDER_MEDIAN = 4
"""CatBoost `Median` (`TMedianBinarizer`, :1201)."""

comptime BORDER_UNIFORM_AND_QUANTILES = 5
"""CatBoost `UniformAndQuantiles` (`TMedianPlusUniformBinarizer`, :1224)."""

comptime BORDER_MIN_ENTROPY = 6
"""CatBoost `MinEntropy`. Recognized by name and REFUSED; see
`check_border_type`."""

comptime BORDER_MAX_LOG_SUM = 7
"""CatBoost `MaxLogSum`. Recognized by name and REFUSED; see
`check_border_type`."""

comptime CATBOOST_DEFAULT_BORDER_COUNT = 254
"""CatBoost's CPU `border_count` default, from
`catboost/private/libs/options/data_processing_options.cpp:14-19`. Thresholds,
so it is this repo's `max_bin = 255`. Recorded, and read by nothing: our
default budget stays 255 bins because it already is this budget."""

comptime _CB_PENALTY_EPS = 1e-8
"""The `1e-8` inside CatBoost's penalties (`Penalty<>`, :174-186), kept
because it is what stops `log(0)` at a zero-weight bin."""

comptime _CB_NEG_INF = -POSITIVE_INF
"""What `CalcSplitScore` returns for a cut at either end of a bin
(:1400-1402), which is also how an unsplittable bin sinks to the bottom of
the heap."""


def parse_border_type(name: String) raises -> Int:
    """CatBoost's `feature_border_type` spellings, plus `quantile` for ours.

    CatBoost's own names are taken verbatim from `EBorderSelectionType`
    (`library/cpp/grid_creator/binarization.h:13-21`) so that a scenario file
    can pass through whatever the user gave CatBoost.
    """
    if name == "quantile":
        return BORDER_QUANTILE
    if name == "GreedyLogSum":
        return BORDER_GREEDY_LOG_SUM
    if name == "GreedyMinEntropy":
        return BORDER_GREEDY_MIN_ENTROPY
    if name == "Uniform":
        return BORDER_UNIFORM
    if name == "Median":
        return BORDER_MEDIAN
    if name == "UniformAndQuantiles":
        return BORDER_UNIFORM_AND_QUANTILES
    if name == "MinEntropy":
        return BORDER_MIN_ENTROPY
    if name == "MaxLogSum":
        return BORDER_MAX_LOG_SUM
    raise Error(
        "unknown border_type '",
        name,
        "'; expected quantile, GreedyLogSum, GreedyMinEntropy, Uniform,"
        " Median, UniformAndQuantiles, MinEntropy or MaxLogSum",
    )


def border_type_name(border_type: Int) raises -> String:
    """The name `parse_border_type` accepts, for messages and round-trips."""
    if border_type == BORDER_QUANTILE:
        return String("quantile")
    if border_type == BORDER_GREEDY_LOG_SUM:
        return String("GreedyLogSum")
    if border_type == BORDER_GREEDY_MIN_ENTROPY:
        return String("GreedyMinEntropy")
    if border_type == BORDER_UNIFORM:
        return String("Uniform")
    if border_type == BORDER_MEDIAN:
        return String("Median")
    if border_type == BORDER_UNIFORM_AND_QUANTILES:
        return String("UniformAndQuantiles")
    if border_type == BORDER_MIN_ENTROPY:
        return String("MinEntropy")
    if border_type == BORDER_MAX_LOG_SUM:
        return String("MaxLogSum")
    raise Error("unknown border_type ", border_type)


def check_border_type(border_type: Int) raises:
    """Accept a border type, or refuse it by name with the reason.

    `MinEntropy` and `MaxLogSum` are refused rather than approximated. They
    are `TExactBinarizer` (:1151), whose engine is the banded dynamic program
    at :192-694: nine solver modes, `O(distinct * bins)` time and
    `(bins - 2) * distinct` `size_t` of scratch. An approximation of either is
    already here under CatBoost's own name for it, `GreedyMinEntropy` and
    `GreedyLogSum`, which are the same two objectives optimized greedily
    (`BestWeightedSplit`, :1655-1667, pairs each exact type with its greedy
    twin under one penalty).
    """
    if border_type == BORDER_MIN_ENTROPY or border_type == BORDER_MAX_LOG_SUM:
        raise Error(
            "border_type '",
            border_type_name(border_type),
            "' is CatBoost's exact dynamic program (TExactBinarizer) and is"
            " not implemented; its greedy twin is '",
            String("GreedyMinEntropy") if border_type
            == BORDER_MIN_ENTROPY else String("GreedyLogSum"),
            "'",
        )
    if border_type < BORDER_QUANTILE or border_type > BORDER_MAX_LOG_SUM:
        raise Error("unknown border_type ", border_type)


@fieldwise_init
struct _CbBin(Copyable, Movable):
    """One bin of CatBoost's greedy split, `[start, end)` over distinct
    levels. `IFeatureBin` (:1320-1376) plus `TWeightedFeatureBin`'s cached
    best cut (:1428-1497).

    `best` is the level index of the better of the two cuts considered, and
    `score` is that cut's penalty gain, `_CB_NEG_INF` when the bin holds a
    single level and cannot be cut at all.
    """

    var start: Int
    var end: Int
    var best: Int
    var score: Float64


def _cb_penalty(border_type: Int, w: Float64) -> Float64:
    """`Penalty<MaxSumLog>(w) = -log(w + 1e-8)` and
    `Penalty<MinEntropy>(w) = w * log(w + 1e-8)`, :174-186."""
    if border_type == BORDER_GREEDY_MIN_ENTROPY:
        return w * log(w + _CB_PENALTY_EPS)
    return -log(w + _CB_PENALTY_EPS)


def _cb_split_score(
    b_start: Int,
    b_end: Int,
    p: Int,
    cum: List[Float64],
    border_type: Int,
) -> Float64:
    """`TWeightedFeatureBin::CalcSplitScore`, :1454-1471.

    `Penalty(L + R) - Penalty(L) - Penalty(R)` on the two sides' OBSERVATION
    weights, and `_CB_NEG_INF` for a cut at either end. `cum[i]` is the
    cumulative observation count through level `i`.
    """
    if p <= b_start or p >= b_end:
        return _CB_NEG_INF
    var left_bins = 0.0 if b_start == 0 else cum[b_start - 1]
    var lw = cum[p - 1] - left_bins
    var rw = cum[b_end - 1] - cum[p - 1]
    return (
        _cb_penalty(border_type, lw + rw)
        - _cb_penalty(border_type, lw)
        - _cb_penalty(border_type, rw)
    )


def _cb_update_best(mut b: _CbBin, cum: List[Float64], border_type: Int):
    """`TWeightedFeatureBin::UpdateBestSplitProperties`, :1473-1493.

    Only TWO cuts are ever considered, the two ends of the level holding the
    bin's median observation, and a tie goes left
    (`BestSplit = scoreLeft >= scoreRight ? lb : ub`). That is what makes this
    a recursive median split rather than an exhaustive search, and it is the
    single most surprising thing about `GreedyLogSum`.
    """
    var left_bins = 0.0 if b.start == 0 else cum[b.start - 1]
    var mid_w = 0.5 * (left_bins + cum[b.end - 1])
    # lower_bound over cum[start, end) for mid_w.
    var lo = b.start
    var hi = b.end
    while lo < hi:
        var m = lo + (hi - lo) // 2
        if cum[m] < mid_w:
            lo = m + 1
        else:
            hi = m
    # `mid_w <= cum[end - 1]` always, so `lo < end` always; the clamp is here
    # because this function must never read past a bin and cannot raise.
    var lb = lo if lo < b.end else b.end - 1
    var ub = lb + 1
    var s_left = _cb_split_score(b.start, b.end, lb, cum, border_type)
    var s_right = _cb_split_score(b.start, b.end, ub, cum, border_type)
    if s_left >= s_right:
        b.best = lb
        b.score = s_left
    else:
        b.best = ub
        b.score = s_right


def _cb_can_split(b: _CbBin) -> Bool:
    """`IFeatureBin::CanSplit`, :1353-1355."""
    return b.start != b.best and b.end != b.best


def _cb_better(a: _CbBin, b: _CbBin) -> Bool:
    """Strict heap order: higher score first, ties to the leftmost bin.

    DIVERGENCE, deliberate. `std::priority_queue`'s order among equal scores
    is unspecified in C++, so CatBoost has no behavior here to match. A total
    order is required because this project's rule is that a fit reproduce
    across `MOJOTREES_NUM_WORKERS` and across machines, and a tie CAN change
    the output: it decides which bin gets the last split the budget allows.
    """
    if a.score != b.score:
        return a.score > b.score
    return a.start < b.start


def _cb_heap_push(mut heap: List[_CbBin], var item: _CbBin):
    heap.append(item^)
    var i = len(heap) - 1
    while i > 0:
        var p = (i - 1) // 2
        if not _cb_better(heap[i], heap[p]):
            break
        var t = heap[i].copy()
        heap[i] = heap[p].copy()
        heap[p] = t^
        i = p


def _cb_heap_pop(mut heap: List[_CbBin]) -> _CbBin:
    var top = heap[0].copy()
    var last = heap[len(heap) - 1].copy()
    heap.resize(len(heap) - 1, top.copy())
    var n = len(heap)
    if n > 0:
        heap[0] = last^
        var i = 0
        while True:
            var l = 2 * i + 1
            var r = l + 1
            var best = i
            if l < n and _cb_better(heap[l], heap[best]):
                best = l
            if r < n and _cb_better(heap[r], heap[best]):
                best = r
            if best == i:
                break
            var t = heap[i].copy()
            heap[i] = heap[best].copy()
            heap[best] = t^
            i = best
    return top^


def _cb_greedy_borders(
    levels: List[Float64],
    cum: List[Float64],
    max_borders: Int,
    border_type: Int,
    mut heap: List[_CbBin],
    mut out: List[Float64],
):
    """`GreedySplit`, :1500-1520, over grouped levels.

    The loop is `while (splits.size() <= maxBordersCount && splits.top().CanSplit())`,
    so it stops either at `max_borders + 1` bins or once the best-scoring bin
    holds a single level. An unsplittable bin scores `_CB_NEG_INF` and sinks,
    so "the top cannot split" means every bin is a single level: the border
    count is exactly `min(max_borders, len(levels) - 1)` and never anything
    between.

    Grouped levels rather than raw sorted observations: CatBoost runs
    `TFeatureBin` on the raw array for a dense column (:1703-1712) and
    `TWeightedFeatureBin` on grouped levels for a sparse one (:1686-1702). The
    two were ported and compared and produce identical border sets; the
    grouped array is the shorter one. See catalog A11.
    """
    heap.clear()
    var n = len(levels)
    if n < 2 or max_borders < 1:
        return
    var root = _CbBin(0, n, 0, 0.0)
    _cb_update_best(root, cum, border_type)
    _cb_heap_push(heap, root^)
    while len(heap) <= max_borders and _cb_can_split(heap[0]):
        var top = _cb_heap_pop(heap)
        var left = _CbBin(top.start, top.best, 0, 0.0)
        _cb_update_best(left, cum, border_type)
        top.start = top.best
        _cb_update_best(top, cum, border_type)
        _cb_heap_push(heap, left^)
        _cb_heap_push(heap, top^)
    # `IFeatureBin::LeftBorder`, :1357-1371: every bin but the first
    # contributes the midpoint of the two levels its start sits between. The
    # heap is not in bin order, so the caller sorts.
    for i in range(len(heap)):
        var s = heap[i].start
        if s > 0:
            out.append(0.5 * levels[s - 1] + 0.5 * levels[s])


def _cb_regular_border(levels: List[Float64], v: Float64) -> Float64:
    """`RegularBorder`, :698-731, without the `initialBorders` snapping.

    "Border before the element with value `v`". Off the low end it returns
    `min(0.5 * front, 2 * front)` and off the high end
    `max(2 * back, back + 1)`, which are CatBoost's "always true" and "always
    false" degenerate borders; in between it is the midpoint of the two levels
    `v` falls between, with the wrong-side-rounding guard that pulls the
    midpoint back down to the lower level when rounding put it on the upper
    one.
    """
    var n = len(levels)
    # lower_bound: first level not less than v.
    var lo = 0
    var hi = n
    while lo < hi:
        var m = lo + (hi - lo) // 2
        if levels[m] < v:
            lo = m + 1
        else:
            hi = m
    if lo >= n:
        var back = levels[n - 1]
        var a = 2.0 * back
        var b = back + 1.0
        return _avoid_inf(a if a > b else b)
    if lo == 0:
        var front = levels[0]
        var a = 0.5 * front
        var b = 2.0 * front
        return _avoid_inf(a if a < b else b)
    var res = (levels[lo] + levels[lo - 1]) * 0.5
    if res == levels[lo]:
        res = levels[lo - 1]
    return _avoid_inf(res)


def _cb_level_at_rank(cum: List[Float64], rank: Int) -> Int:
    """The level holding observation `rank`: the first level whose cumulative
    end exceeds it. This is how a rank-addressed rule reaches a grouped
    array."""
    var target = Float64(rank)
    var lo = 0
    var hi = len(cum)
    while lo < hi:
        var m = lo + (hi - lo) // 2
        if cum[m] <= target:
            lo = m + 1
        else:
            hi = m
    return lo if lo < len(cum) else len(cum) - 1


def _cb_median_borders(
    levels: List[Float64],
    cum: List[Float64],
    n_valid: Int,
    max_borders: Int,
    mut out: List[Float64],
):
    """`GenerateMedianBorders`, :1046-1063.

    Equal-count ranks `(i + 1) * total / (max_borders + 1)`, clamped to the
    last observation, each turned into a border by `RegularBorder`, and each
    dropped when its value is the column minimum (`if (val1 != featureValues[0])`)
    because a border there would put every row on one side.
    """
    if max_borders < 1 or len(levels) < 2 or n_valid < 1:
        return
    var first = levels[0]
    for i in range(max_borders):
        var i1 = (i + 1) * n_valid // (max_borders + 1)
        if i1 > n_valid - 1:
            i1 = n_valid - 1
        var val1 = levels[_cb_level_at_rank(cum, i1)]
        if val1 != first:
            out.append(_cb_regular_border(levels, val1))


def _cb_uniform_borders(
    lo: Float64, hi: Float64, max_borders: Int, mut out: List[Float64]
):
    """`TUniformBinarizer::BestSplit`, :1262-1317.

    `minValue + (i + 1) * (maxValue - minValue) / (maxBordersCount + 1)`,
    inserted RAW. Note that this one does NOT go through `RegularBorder` while
    the uniform half of `UniformAndQuantiles` does (:1252); that asymmetry is
    CatBoost's, not a transcription slip.
    """
    if max_borders < 1 or not (lo < hi):
        return
    for i in range(max_borders):
        out.append(
            _avoid_inf(
                lo + Float64(i + 1) * (hi - lo) / Float64(max_borders + 1)
            )
        )


def _cb_uniform_and_quantiles_borders(
    levels: List[Float64],
    cum: List[Float64],
    n_valid: Int,
    max_borders: Int,
    mut out: List[Float64],
):
    """`TMedianPlusUniformBinarizer::BestSplit`, :1224-1260.

    Half the budget goes to uniform borders and the rest to median ones, and
    the halving is `halfBorders = maxBordersCount / 2` with the MEDIAN half
    getting `maxBordersCount - halfBorders`, so an odd budget favours the
    median half. Both halves land in one set, so a uniform border that
    coincides with a median one costs nothing and the result can be under
    budget for that reason alone. Unlike plain `Uniform`, the uniform half
    here is snapped by `RegularBorder`.
    """
    if max_borders < 1 or len(levels) < 2:
        return
    var half = max_borders // 2
    _cb_median_borders(levels, cum, n_valid, max_borders - half, out)
    var lo = levels[0]
    var hi = levels[len(levels) - 1]
    for i in range(half):
        var v = lo + Float64(i + 1) * (hi - lo) / Float64(half + 1)
        out.append(_cb_regular_border(levels, v))


def _cb_group_levels(
    mut col: List[Float64],
    n_valid: Int,
    mut levels: List[Float64],
    mut cum: List[Float64],
):
    """Sort a column in place and reduce it to ascending distinct levels with
    cumulative observation counts. CatBoost's `GroupAndSortValues`, :1613-1640,
    with unit weights and no default value.

    `NaN` is already gone: the caller drops it while building the column, and
    CatBoost drops it too, one layer up (`NSplitSelection::BestSplit` :90-97
    erases every `NaN` before any binarizer sees the values).
    """
    sort(col)
    levels.clear()
    cum.clear()
    var i = 0
    while i < n_valid:
        var v = col[i]
        var j = i + 1
        while j < n_valid and col[j] == v:
            j += 1
        levels.append(v)
        cum.append(Float64(j))
        i = j


def _cb_finalize(mut out: List[Float64]):
    """Sort and strictly deduplicate, which is CatBoost's `THashSet` plus
    `Sort` (`SetQuantization`, :903-904) and this repo's strictly increasing
    edge invariant in one pass. `_avoid_inf` clamps, as everywhere else here.

    CatBoost also purges `-0.0f` here (:897-902, "BestSplit might add negative
    zeros"), an artifact of its Float32 `0.5f * a + 0.5f * b`. Not ported:
    `-0.0` and `0.0` group into one level above, so no midpoint of two
    distinct Float64 levels can be `-0.0`, and the strict-increase filter
    would collapse the pair anyway.
    """
    var n = len(out)
    if n < 1:
        return
    sort(out)
    var w = 0
    for i in range(n):
        var e = _avoid_inf(out[i])
        if w > 0 and e <= out[w - 1]:
            continue
        out[w] = e
        w += 1
    out.resize(w, 0.0)


def catboost_borders(
    mut col: List[Float64],
    n_valid: Int,
    max_borders: Int,
    border_type: Int,
    mut levels: List[Float64],
    mut cum: List[Float64],
    mut heap: List[_CbBin],
    mut out: List[Float64],
):
    """CatBoost's border selection for one column, into `out` as this repo's
    strictly increasing bin edges.

    `col` holds the column's `n_valid` non-`NaN` values and is SORTED IN PLACE.
    `max_borders` is CatBoost's `border_count`, which is one less than the
    number of ordinary bins the caller is budgeting. `levels`, `cum` and
    `heap` are reusable scratch, so a caller fitting many features allocates
    once per worker rather than once per feature.

    Non-raising, like `emit_quantile_edges` and for the same reason: the dense
    fitter calls this from inside a worker closure. `border_type` is validated
    by `check_border_type` before any dispatch, so a refused or unknown type
    cannot arrive here; if one did it would leave `out` empty, which is one
    bin and no split rather than a wrong split.

    The border semantics line up without translation. CatBoost sends
    `value > border` right, and `BinMapper.bin_value` puts `v` in the first
    bin with `v <= edge`, so a CatBoost border IS one of our edges.
    """
    out.clear()
    if n_valid < 1 or max_borders < 1:
        return
    _cb_group_levels(col, n_valid, levels, cum)
    if len(levels) < 2:
        return
    if (
        border_type == BORDER_GREEDY_LOG_SUM
        or border_type == BORDER_GREEDY_MIN_ENTROPY
    ):
        _cb_greedy_borders(levels, cum, max_borders, border_type, heap, out)
    elif border_type == BORDER_UNIFORM:
        _cb_uniform_borders(
            levels[0], levels[len(levels) - 1], max_borders, out
        )
    elif border_type == BORDER_MEDIAN:
        _cb_median_borders(levels, cum, n_valid, max_borders, out)
    elif border_type == BORDER_UNIFORM_AND_QUANTILES:
        _cb_uniform_and_quantiles_borders(
            levels, cum, n_valid, max_borders, out
        )
    _cb_finalize(out)


# --------------------------------------------------------------------------
# Rank selection: the order statistics a quantile fit needs, without a full
# sort. See the module docstring for why and how.
# --------------------------------------------------------------------------

comptime SELECT_MIN_ROWS = 1 << 16
"""Non-missing values below which a feature is sorted outright. Two linear
passes plus a bucket table only pay for themselves once the column is much
larger than the handful of ranks a quantile fit asks about, and a small
column's sort is already cheap. Scheduling-only in the sense that matters:
both paths produce the same edges, and `tests/test_binning.mojo` asserts it
on both sides of this threshold."""

comptime SELECT_BUCKET_BITS = 16
comptime SELECT_BUCKETS = 1 << SELECT_BUCKET_BITS
"""Buckets one rebasing pass splits a segment into. The point of the pass is
to drop buckets, so there have to be far more of them than the at most
`2 * (max_bins - 1)` ranks being asked about, or every bucket holds a rank
and nothing is dropped. At 16 bits against 255 bins under a percent of the
buckets survive, which is what turns a sort of the column into a sort of a
few hundred short runs. The cost is a 512 KB count table per worker,
allocated only when a column is large enough to take this path."""

comptime SELECT_SMALL_SEGMENT = 1 << 12
"""A gathered bucket at or below this many values is sorted rather than
bucketed again. Below it the sort is a few microseconds and another pass
would cost more than it saves."""

comptime SELECT_MAX_DEPTH = 4
"""How many times a segment may be rebucketed before it is simply sorted.
Four levels of `SELECT_BUCKET_BITS` cover the whole 64-bit key, so the budget
only runs out on a column whose values crowd into one bucket at every scale
without being equal; sorting that bucket is exact and is never more work than
sorting the whole column."""

comptime DISTINCT_SLOTS = 1 << 11
comptime DISTINCT_SHIFT = UInt64(53)
"""Slots in the set `collect_distinct` uses to decide whether a column has
few enough distinct values to give each one its own bin, and the shift that
turns a scrambled key into one of them (64 - 11).

The largest set it ever holds is `MAX_BINS + 1` keys, the size at which the
answer is already no, so the table never loads past an eighth and a probe is
about one slot. It is 16 KB per worker, allocated on first use and then
reused for every column that worker sees."""


def env_select_min_rows() -> Int:
    """`SELECT_MIN_ROWS` under `MOJOTREES_BINNING_SELECT_MIN_ROWS`.

    Scheduling-only, in the same sense as the two variables in
    `parallel.mojo`: the two paths resolve the same order statistics, so this
    decides which one runs and nothing else. It exists so a test can assert
    that equality on data small enough to fit in a test, and so a benchmark
    can measure one path against the other without rebuilding. `0`, unset, or
    unparsable means the default.
    """
    var n = _env_int("MOJOTREES_BINNING_SELECT_MIN_ROWS", SELECT_MIN_ROWS)
    return SELECT_MIN_ROWS if n <= 0 else n


def order_key(v: Float64) -> UInt64:
    """An order-preserving `UInt64` view of a non-`NaN` double.

    IEEE-754 doubles compare as sign-magnitude integers, so flipping every
    bit of a negative pattern and the sign bit of a non-negative one turns
    the value order into unsigned integer order: `a < b` implies
    `order_key(a) < order_key(b)` for every pair of non-`NaN` doubles,
    infinities included.

    The one pair it separates that `<` does not is `-0.0` and `+0.0`, which
    are equal as values and get different keys. That cannot move an edge:
    the two only ever meet as a boundary's `below` and `above`, where
    `above <= below` holds either way round and `emit_quantile_edges` drops
    the boundary, and a midpoint taken against a neighbour is the same for
    both signs of zero.
    """
    var u = v.to_bits().cast[DType.uint64]()
    if (u >> 63) != 0:
        return u ^ 0xFFFF_FFFF_FFFF_FFFF
    return u | 0x8000_0000_0000_0000


def first_above_sorted(col: List[Float64], lo: Int, hi: Int, w: Float64) -> Int:
    """First rank in `[lo, hi)` of an ascending `col` whose value exceeds `w`,
    or `hi` when `w` is the largest value there.

    The sorted path's answer to "where does this run end": `col[lo - 1] == w`
    at every call site, so everything below `lo` is already at or under `w`.
    """
    var left = lo
    var right = hi
    while left < right:
        var mid = (left + right) // 2
        if col[mid] > w:
            right = mid
        else:
            left = mid + 1
    return left


def resolve_above_unsorted(
    col: List[Float64],
    below: List[Float64],
    mut above: List[Float64],
    mut cand: List[Float64],
    mut found: List[Int],
):
    """`above[j]` = the smallest value in `col` strictly greater than
    `below[j]`, or `below[j]` when there is none.

    The selected path's answer to the same question, for a column it has
    deliberately not sorted. `below` is ascending, so the values that exceed
    `below[j]` are exactly those assigned to index `j` or later by the search
    below, and one suffix-minimum sweep over the boundaries turns per-index
    minima into per-boundary answers. One pass over the column and a
    few-hundred-entry search per value, against a table small enough to stay
    in L1.

    Only called when a boundary actually landed on a tie: with no ties the
    value at rank `idx` is already the next distinct value, so a column of
    distinct values pays nothing for this at all.
    """
    var m = len(below)
    above.clear()
    if m <= 0:
        return
    cand.clear()
    cand.resize(m, 0.0)
    # 0/1 rather than Bool: the raw-pointer loads below are defined for
    # scalar element types only.
    found.clear()
    found.resize(m, 0)

    var bp = below.unsafe_ptr()
    var cp = cand.unsafe_ptr()
    var fp = found.unsafe_ptr()
    var vp = col.unsafe_ptr()
    for i in range(len(col)):
        var v = vp.unsafe_load(i)
        # Count of boundaries with `below < v`; the last of them is the only
        # index this value can be the immediate successor of.
        var left = 0
        var right = m
        while left < right:
            var mid = (left + right) // 2
            if bp.unsafe_load(mid) < v:
                left = mid + 1
            else:
                right = mid
        if left == 0:
            continue
        var j = left - 1
        if fp.unsafe_load(j) == 0 or v < cp.unsafe_load(j):
            cp.unsafe_store(j, v)
            fp.unsafe_store(j, 1)

    for t in range(m - 1):
        var j = m - 2 - t
        if fp.unsafe_load(j + 1) != 0:
            if (
                fp.unsafe_load(j) == 0
                or cp.unsafe_load(j + 1) < cp.unsafe_load(j)
            ):
                cp.unsafe_store(j, cp.unsafe_load(j + 1))
                fp.unsafe_store(j, 1)

    for j in range(m):
        if fp.unsafe_load(j) != 0:
            above.append(cp.unsafe_load(j))
        else:
            above.append(bp.unsafe_load(j))


comptime _KEY_EMPTY = UInt64(0)
comptime _KEY_MIXED = UInt64(0xFFFF_FFFF_FFFF_FFFF)
"""Two key values `order_key` cannot return, so they can mark a bucket that
has seen nothing and one that has seen two different keys. Both would need
their argument to be a `NaN`, which never reaches the key: `order_key(v) == 0`
requires the bit pattern `0xFFFF...`, and `== 0xFFFF...` requires
`0x7FFF...` or `0xFFFF...`, all three of them quiet `NaN`s."""


def value_from_key(k: UInt64) -> Float64:
    """The inverse of `order_key`, exact for every key it can produce.

    Only used where a bucket is known to hold one repeated key, so the value
    it rebuilds is bit for bit the value that went in: two doubles with the
    same bit pattern are the same double.
    """
    var u = k
    if (u >> 63) != 0:
        u = u ^ 0x8000_0000_0000_0000
    else:
        u = u ^ 0xFFFF_FFFF_FFFF_FFFF
    return bitcast[DType.float64, 1](SIMD[DType.uint64, 1](u))


def collect_distinct(
    col: List[Float64],
    n_valid: Int,
    limit: Int,
    mut table: List[UInt64],
    mut slots: List[Int],
    mut hits: List[Int],
    mut out: List[Float64],
    mut counts: List[Int],
) -> Bool:
    """The ascending distinct values of `col[0, n_valid)`, and how many rows
    each holds, when there are at most `limit` of them; `False`, with `out`
    and `counts` empty, as soon as there is one more.

    Counting here rather than in a second pass
    ------------------------------------------
    `min_data_in_bin` needs a count per level, and its default is now 3, so
    the levels branch needs those counts on every low-cardinality column
    rather than on the few that asked for the option. `count_levels` answers
    the same question in a separate pass, at a binary search over the level
    table per row; this loop already visits every row and already knows which
    slot the row's level lives in, so the count is one L1 increment on a line
    the probe just touched.

    The arithmetic, which is why this is not a tidiness change. At 250,000
    rows and 100 columns of 20 levels the scan does 25e6 probes. A separate
    counting pass would add 25e6 x ceil(log2(20)) = 1.25e8 dependent,
    branchy comparisons and a second streaming read of every column; the
    fused increment adds 25e6 L1 read-modify-writes and no second read.
    Neither number is measured, and the orchestrator has the clock.

    `hits` is per *slot* and `counts` is per *level*, ascending with `out`.
    `hits` is caller-owned scratch for the same reason `table` is: it is
    cleared by the slots that were recorded rather than by its length.

    This is what lets a column with few enough levels get a bin per level
    instead of a bin per quantile boundary, which is LightGBM's
    `GreedyFindBin` when `num_distinct_values <= max_bin` (see `fit_bins`).

    It answers no cheaply, which is what makes it affordable to ask on every
    column. A continuous column reaches `limit + 1` distinct values within
    its first few hundred rows and stops there, so the whole check costs a
    few hundred probes however many million rows the column has. A column
    that really does have few levels is scanned to the end, at about a probe
    per row: a multiply, a shift, and a load from a table small enough to sit
    in L1.

    Keys are `order_key`'s, so the set never compares two doubles for
    equality; two values are one level exactly when their bit patterns match,
    which is the same rule the rest of this module orders by. Zero is the
    single exception, normalized below.

    The table is left clean for the next call rather than rewritten at the
    start of one: at most `limit + 1` of its slots are ever filled, so
    clearing the recorded ones is bounded by the answer rather than by the
    table, and a fit over thousands of features does not rewrite 16 KB per
    feature.
    """
    out.clear()
    counts.clear()
    if limit < 1 or limit > MAX_BINS or n_valid < 1:
        return False
    if len(table) != DISTINCT_SLOTS or len(hits) != DISTINCT_SLOTS:
        table.clear()
        table.resize(DISTINCT_SLOTS, _KEY_EMPTY)
        hits.clear()
        hits.resize(DISTINCT_SLOTS, 0)
        slots.clear()
    var tp = table.unsafe_ptr()
    var hp = hits.unsafe_ptr()
    for i in range(len(slots)):
        tp.unsafe_store(slots[i], _KEY_EMPTY)
        hp.unsafe_store(slots[i], 0)
    slots.clear()

    var cp = col.unsafe_ptr()
    var full = False
    for r in range(n_valid):
        var v = cp.unsafe_load(r)
        if v == 0.0:
            # `-0.0` and `0.0` are one value to every comparison an edge is
            # built from, so they have to be one level here too; their bit
            # patterns differ, so the key alone would make them two. The
            # count follows the normalization, so both signs land on one
            # level and in one counter.
            v = 0.0
        var k = order_key(v)
        # Fibonacci scramble, then take the top bits: the low bits of a key
        # are a double's mantissa, and columns of round numbers share them.
        var s = Int((k * UInt64(0x9E37_79B9_7F4A_7C15)) >> DISTINCT_SHIFT)
        while True:
            var cur = tp.unsafe_load(s)
            if cur == k:
                hp.unsafe_store(s, hp.unsafe_load(s) + 1)
                break
            if cur == _KEY_EMPTY:
                if len(out) >= limit:
                    full = True
                    break
                tp.unsafe_store(s, k)
                hp.unsafe_store(s, 1)
                slots.append(s)
                out.append(v)
                break
            s = (s + 1) & (DISTINCT_SLOTS - 1)
        if full:
            break
    if full:
        out.clear()
        return False
    # At most `limit` values, so this is a sort of a few hundred at the very
    # most, once per column.
    sort(out)
    # The table is still populated, so each sorted level finds its own
    # counter by the probe that put it there. `limit` probes at the very
    # most, and every key is present, so the walk terminates.
    counts.resize(len(out), 0)
    var np = counts.unsafe_ptr()
    for j in range(len(out)):
        var k = order_key(out[j])
        var s = Int((k * UInt64(0x9E37_79B9_7F4A_7C15)) >> DISTINCT_SHIFT)
        while tp.unsafe_load(s) != k:
            s = (s + 1) & (DISTINCT_SLOTS - 1)
        np.unsafe_store(j, hp.unsafe_load(s))
    return True


def distinct_levels_sorted(
    sorted_col: List[Float64], n: Int, limit: Int, mut out: List[Float64]
) -> Bool:
    """`collect_distinct`'s answer for a column that is already sorted.

    Equal values are adjacent in a sorted column, so the levels are a walk
    and there is no set to build. `sparse.fit_bins_csc` sorts its stored
    values anyway and wants this one; the dense fit does not sort at all any
    more and wants the other.

    The two must agree on every column, or a sparse matrix and its dense form
    would bin differently, and `tests/test_sparse.mojo` requires them to be
    identical edge for edge. They agree because the only way two finite
    doubles can be one level here is bit equality, which is what
    `collect_distinct` keys on, with the single exception of `-0.0` and `0.0`
    that both normalize the same way. `tests/test_binning.mojo` checks the
    two against each other directly.
    """
    out.clear()
    if limit < 1 or limit > MAX_BINS or n < 1:
        return False
    for i in range(n):
        var v = sorted_col[i]
        if v == 0.0:
            v = 0.0
        if len(out) > 0 and out[len(out) - 1] == v:
            continue
        if len(out) >= limit:
            out.clear()
            return False
        out.append(v)
    return True


def count_levels(
    col: List[Float64],
    n_valid: Int,
    levels: List[Float64],
    mut counts: List[Int],
):
    """`counts[j]` = how many of `col[0, n_valid)` equal `levels[j]`.

    `levels` is what `collect_distinct` returned for this same column, so it
    is ascending and every value in the column is one of its entries; the
    search below therefore always lands inside the array and the counts sum
    to `n_valid`.

    **`fit_bins` does not call this.** `collect_distinct` counts as it scans
    (see its docstring for why a second pass was not affordable once
    `min_data_in_bin` defaulted to 3). What this is now is the independent
    reference those fused counts are checked against: it answers the same
    question by a different mechanism -- a binary search over the level table
    instead of a hash slot -- and `tests/test_binning.mojo` requires the two
    to agree count for count. Deleting it would leave the fused counter
    checked only against itself.

    `-0.0` needs no normalizing on the way in: it compares equal to `0.0`,
    which is the entry `collect_distinct` stored, so the search converges on
    the same slot for both signs.

    Non-raising, because the dense fitter calls this from inside a worker
    closure that cannot raise.
    """
    counts.clear()
    var m = len(levels)
    if m <= 0 or n_valid <= 0:
        return
    counts.resize(m, 0)
    var lp = levels.unsafe_ptr()
    var np = counts.unsafe_ptr()
    var vp = col.unsafe_ptr()
    for i in range(n_valid):
        var v = vp.unsafe_load(i)
        # First index whose level is not below `v`, which is `v`'s own level.
        # At most `log2(MAX_BINS)` steps over a table that fits in L1.
        var left = 0
        var right = m
        while left < right:
            var mid = (left + right) // 2
            if lp.unsafe_load(mid) < v:
                left = mid + 1
            else:
                right = mid
        if left < m:
            np.unsafe_store(left, np.unsafe_load(left) + 1)


def bin_construct_sample_rows(
    n_rows: Int, sample_cnt: Int, seed: Int
) raises -> List[Int]:
    """The rows a sampled edge fit reads, ascending, or empty for "every row".

    Knuth's selection sampling: row r of the `left` still to be considered is
    taken when `uniform < need / left`, which selects exactly `sample_cnt`
    rows in one pass and in ascending order. The `need >= left` shortcut at
    the end is what makes "exactly" true rather than "in expectation".

    Determinism, which is the property this has to have. The draw for row r is
    `uniform(seed * GOLDEN + r)`, a counter-based splitmix64 draw that depends
    on r alone: it does not read a clock, a global, or a running RNG state, so
    it is the same draw however many rows were selected before it. The loop is
    serial and runs once per fit, before any feature is dispatched, so the
    sample is identical at every `MOJOTREES_NUM_WORKERS`. `splitmix64` is
    64-bit integer arithmetic and the comparison is one multiply and one
    compare on `Float64`, with no accumulation for a compiler to reassociate,
    so it is the same set of rows on every machine on this toolchain.

    An empty result means every row, which is the caller's fast path: it is
    returned both for `sample_cnt <= 0` and for a sample that would cover the
    matrix anyway.
    """
    var out = List[Int]()
    if n_rows < 1 or sample_cnt <= 0 or sample_cnt >= n_rows:
        return out^
    out.reserve(sample_cnt)
    var base = UInt64(seed) * GOLDEN
    var need = sample_cnt
    for r in range(n_rows):
        if need <= 0:
            break
        var left = n_rows - r
        if need >= left:
            out.append(r)
            need -= 1
            continue
        if uniform(base + UInt64(r)) * Float64(left) < Float64(need):
            out.append(r)
            need -= 1
    return out^


def _bucket_shift(span: UInt64) -> UInt64:
    """Smallest `s` with `span >> s < SELECT_BUCKETS`, so a key range of
    `span` maps onto at most `SELECT_BUCKETS` buckets."""
    var s = UInt64(0)
    var v = span
    while v >= UInt64(SELECT_BUCKETS):
        v = v >> 1
        s += 1
    return s


def resolve_ranks(
    mut seg: List[Float64],
    rank_base: Int,
    ranks: List[Int],
    r_lo: Int,
    r_hi: Int,
    mut counts: List[Int],
    mut bucket_keys: List[UInt64],
    mut vals: List[Float64],
    depth: Int,
):
    """Fill `vals[i]` with the `ranks[i] - rank_base`-th smallest value of
    `seg`, for every `i` in `[r_lo, r_hi)`.

    `ranks` is ascending and every entry it is asked about lies inside
    `[rank_base, rank_base + len(seg))`. `seg` is scratch: it may be permuted
    or sorted, and the caller refills it. `counts` and `bucket_keys` are
    reusable bucket tables, grown here on first use and never read across
    calls. The answer is by definition the value a full sort would have left
    at that rank, so this is an implementation of `sort`-then-index and not an
    approximation of it.

    Memory. Beyond the two fixed bucket tables (`SELECT_BUCKETS` entries each,
    allocated once per worker), the extra storage is the gather buffer and one
    copy of the largest segment gathered into it. A bucket holding one
    repeated key is answered from the key and never gathered at all, and a
    level that would gather more than a quarter of what it was given sorts
    instead, so each level gathers at most `n / 4` and the levels below it a
    quarter of that again: the extra storage is bounded by two thirds of the
    segment, and the whole fit stays inside a small multiple of the column
    buffer it already had.
    """
    if r_lo >= r_hi:
        return
    var n = len(seg)
    if n <= 0:
        # Unreachable from the fitters: a rank inside the segment implies at
        # least one value in it. Non-raising because this runs inside a
        # worker closure, so an impossible call returns rather than throws.
        return

    # All equal: every rank in the segment has that value. A constant column
    # ends here on its first call; a binary or few-level column ends in the
    # equivalent per-bucket check further down, which answers a repeated key
    # without gathering the bucket at all.
    var first = seg[0]
    var constant = True
    for i in range(1, n):
        if seg[i] != first:
            constant = False
            break
    if constant:
        for i in range(r_lo, r_hi):
            vals[i] = first
        return

    if n <= SELECT_SMALL_SEGMENT or depth >= SELECT_MAX_DEPTH:
        sort(seg)
        for i in range(r_lo, r_hi):
            vals[i] = seg[ranks[i] - rank_base]
        return

    # Rebase the buckets on this segment's own key range, so they straddle
    # the values that are there. A fixed slice of the double line would put a
    # narrow column entirely in one bucket and make no progress.
    var kmin = order_key(seg[0])
    var kmax = kmin
    for i in range(1, n):
        var k = order_key(seg[i])
        if k < kmin:
            kmin = k
        if k > kmax:
            kmax = k
    var shift = _bucket_shift(kmax - kmin)
    var n_buckets = Int((kmax - kmin) >> shift) + 1

    if len(counts) < SELECT_BUCKETS:
        counts.resize(SELECT_BUCKETS, 0)
    if len(bucket_keys) < SELECT_BUCKETS:
        bucket_keys.resize(SELECT_BUCKETS, _KEY_EMPTY)
    var cp = counts.unsafe_ptr()
    var kp = bucket_keys.unsafe_ptr()
    for b in range(n_buckets):
        cp.unsafe_store(b, 0)
        kp.unsafe_store(b, _KEY_EMPTY)
    # One pass counts each bucket and records whether it saw one key or
    # several. The second half is what makes a low-cardinality column free:
    # every one of its buckets holds one repeated key, so every bucket is
    # answered from the key below and nothing is gathered, copied, or sorted.
    var sp = seg.unsafe_ptr()
    for i in range(n):
        var k = order_key(sp.unsafe_load(i))
        var b = Int((k - kmin) >> shift)
        cp.unsafe_store(b, cp.unsafe_load(b) + 1)
        var seen = kp.unsafe_load(b)
        if seen == _KEY_EMPTY:
            kp.unsafe_store(b, k)
        elif seen != k:
            kp.unsafe_store(b, _KEY_MIXED)

    # Walk buckets and requested ranks together, both ascending. A bucket
    # holding no requested rank is dropped; the rest are recorded with the
    # rank window they answer and, for the ones that need ordering, a slot in
    # the gather buffer.
    var seg_start = List[Int]()
    var seg_len = List[Int]()
    var seg_key = List[UInt64]()
    var seg_rank_base = List[Int]()
    var seg_r_lo = List[Int]()
    var seg_r_hi = List[Int]()
    var start = 0
    var ri = r_lo
    var gathered = 0
    for b in range(n_buckets):
        var c = cp.unsafe_load(b)
        var stop = start + c
        var lo = ri
        while ri < r_hi and ranks[ri] - rank_base < stop:
            ri += 1
        var key = kp.unsafe_load(b)
        if ri > lo:
            seg_key.append(key)
            seg_rank_base.append(rank_base + start)
            seg_r_lo.append(lo)
            seg_r_hi.append(ri)
            if key == _KEY_MIXED:
                seg_start.append(gathered)
                seg_len.append(c)
                cp.unsafe_store(b, gathered)
                gathered += c
            else:
                seg_start.append(-1)
                seg_len.append(0)
                cp.unsafe_store(b, -1)
        else:
            cp.unsafe_store(b, -1)
        start = stop

    # A level that would gather most of what it was given has not separated
    # anything, so it sorts instead: bounded memory, and no worse than the
    # sort this whole path exists to avoid. In practice this never fires --
    # the requested ranks are a few hundred against `SELECT_BUCKETS` buckets.
    if gathered * 4 >= n:
        sort(seg)
        for i in range(r_lo, r_hi):
            vals[i] = seg[ranks[i] - rank_base]
        return

    var hot = List[Float64](capacity=gathered)
    hot.resize(gathered, 0.0)
    var hp = hot.unsafe_ptr()
    if gathered > 0:
        for i in range(n):
            var v = sp.unsafe_load(i)
            var b = Int((order_key(v) - kmin) >> shift)
            var d = cp.unsafe_load(b)
            if d >= 0:
                hp.unsafe_store(d, v)
                cp.unsafe_store(b, d + 1)

    # Both tables are free again from here, so the recursion reuses them
    # rather than allocating a pair per level.
    for s in range(len(seg_r_lo)):
        if seg_key[s] != _KEY_MIXED:
            # One repeated key: every rank in the bucket is that value.
            var v = value_from_key(seg_key[s])
            for i in range(seg_r_lo[s], seg_r_hi[s]):
                vals[i] = v
            continue
        var slen = seg_len[s]
        var sub = List[Float64](capacity=slen)
        var base = seg_start[s]
        for i in range(slen):
            sub.append(hp.unsafe_load(base + i))
        resolve_ranks(
            sub,
            seg_rank_base[s],
            ranks,
            seg_r_lo[s],
            seg_r_hi[s],
            counts,
            bucket_keys,
            vals,
            depth + 1,
        )


def no_missing_bins(n_features: Int) -> List[Int]:
    """A per-feature missing-bin table for data with no missing support:
    every entry -1."""
    var out = List[Int](capacity=n_features)
    out.resize(n_features, -1)
    return out^


def _sized_missing_bins(var table: List[Int], n_features: Int) -> List[Int]:
    """`table` when it is already per-feature, otherwise an all -1 table."""
    if len(table) == n_features:
        return table^
    return no_missing_bins(n_features)


def all_features(n_features: Int) -> List[Int]:
    """`[0, 1, ..., n_features - 1]`, the unfiltered feature pool."""
    var out = List[Int](capacity=n_features)
    for f in range(n_features):
        out.append(f)
    return out^


# ---------------------------------------------------------------------------
# CTR columns in a binned matrix (catalog A19), the four helpers the two
# append paths share. All of them are no-ops on an inactive `CtrTables`.
# ---------------------------------------------------------------------------


def ctr_slot_columns[
    features_origin: ImmOrigin, //
](
    features: Span[Float64, features_origin],
    n_rows: Int,
    tables: CtrTables,
) raises -> List[Int]:
    """The category bucket of every CTR source column, slot-major, read from
    the RAW category codes.

    `out[s * n_rows + r]` is row `r`'s bucket for the categorical feature at
    slot `s`, which is `tables.source_features[s]`. Bucket 0 is missing or a
    code unseen at fit time; code `i` of `tables.slot_codes` for that slot is
    bucket `i + 1`. `CtrTables.bucket_of` is the lookup and is the only one,
    so the ordered build, the static fit and every scored row cannot disagree.

    This is the input both halves take: `ctr_columns.build_ctr_train_columns`
    runs the ordered prefix over it and `ctr_columns.ctr_predict_columns` runs
    the static table over it, so the *only* difference between train and
    predict is which function is handed this array.

    **This function used to read the BINNED matrix and that made the whole
    mechanism inert.** Its body was `out[dst + r] = Int(bins[src + r])`. A
    categorical column is truncated to `max_bin - 1` levels at fit time and
    every evicted level shares bin 0, so a CTR built from bins gave all of
    them one shared statistic: the columns carried no information the
    truncated column did not already carry. Measured 2026-08-16 on exactly the
    shape they are meant for, 400 near-uniform levels against a 254-entry
    table with roughly a third of rows in bin 0, 256,000 training rows:
    `auc 0.891191 -> 0.891210`, `ap 0.837608 -> 0.837575`. Two nulls, in
    opposite directions. At 3,000 levels with 15 rows each,
    `auc 0.681706 -> 0.681534` and `ap 0.576727 -> 0.577537`. Nulls again.

    That was circular and this is the fix: a target statistic is the mechanism
    for a column too wide to bin, so it must be computed BEFORE the binning
    that loses the levels. Sourced here from `categorical.distinct_category_codes`,
    which truncates nothing, a 200,000-level column produces 200,000
    distinguishable statistics, and the resulting CTR is a real value that
    then quantizes as an ORDINARY NUMERIC column. The cardinality moves out of
    the bin id, where one byte caps it, and into a value, where nothing does.

    Cost: `n_slots * n_rows` `Int`s, transient, plus one binary search per
    entry over the slot's code table.
    """
    var out = List[Int]()
    if not tables.is_active():
        return out^
    var n_slots = tables.n_slots()
    if len(features) < n_rows * tables.n_base_features:
        raise Error(
            "ctr_slot_columns needs the raw feature matrix, n_rows *"
            " n_base_features wide; a shorter span means a caller is still"
            " passing binned values, which is the defect this signature"
            " exists to make impossible"
        )
    out.resize(n_slots * n_rows, 0)
    for s in range(n_slots):
        var f = tables.source_features[s]
        var src = f * n_rows
        var dst = s * n_rows
        for r in range(n_rows):
            out[dst + r] = tables.bucket_of(s, features[src + r])
    return out^


def ctr_extend_cats(cats: CategoricalSpec, n_extra: Int) -> CategoricalSpec:
    """`cats` widened by `n_extra` numerical features.

    A CTR column is numeric: its value is already a bucket index in
    `[0, ctr_border_count]` and a tree thresholds it ordinally, which is the
    entire point of the mechanism. So the extra slots are `False` with an empty
    category slice, and the split search treats them as it treats any other
    binned numeric feature.
    """
    # A spec with no information at all (`CategoricalSpec.none()`, which a
    # `BinnedMatrix` built by the four-argument constructor holds) already
    # answers "numerical" for every index past its end, so widening it would
    # only invent slots. Leave it alone.
    if len(cats.offsets) == 0:
        return cats.copy()
    var flags = cats.is_categorical.copy()
    for _ in range(n_extra):
        flags.append(False)
    var offsets = cats.offsets.copy()
    var last = offsets[len(offsets) - 1]
    for _ in range(n_extra):
        offsets.append(last)
    return CategoricalSpec(flags^, cats.codes.copy(), offsets^)


def ctr_extend_missing(missing_bin: List[Int], n_extra: Int) -> List[Int]:
    """`missing_bin` widened by `n_extra` entries of -1.

    A CTR column has no missing value to reserve a bin for. Every row of a
    categorical column has a bucket -- missing raw values land in
    `categorical.UNKNOWN_BIN` -- so every row of a CTR column has a statistic,
    and there is no third state.
    """
    var out = missing_bin.copy()
    for _ in range(n_extra):
        out.append(-1)
    return out^


def ctr_extend_usable(
    usable: List[Int],
    n_base: Int,
    n_extra: Int,
    replaced: List[Int] = [],
) -> List[Int]:
    """`usable` widened with the CTR column ids, still ascending, with the
    columns in `replaced` taken out of it first.

    CTR columns are always usable: they exist to be split on, and a prefilter
    that dropped one would be dropping the mechanism rather than a trivial
    column. Every base id is below `n_base` and the new ids start at `n_base`,
    so appending keeps the list ascending, which `usable_features` requires.

    **`replaced` is the difference between a CTR that ACCOMPANIES its source
    column and one that REPLACES it, and the two are the two policy halves.**

    - Empty (the default) is ACCOMPANIMENT, which is the opt-in rule under
      `lossguide`: `CTR_SOURCE_BIN_OVERFLOW` adds statistics to a column whose
      category table overflowed and leaves the raw column searchable beside
      them. That is a measured configuration and it works, so nothing here
      moves it.
    - Non-empty is REPLACEMENT, which is CatBoost mode
      (`CTR_SOURCE_ONE_HOT_MAX_SIZE`). CatBoost has **no raw categorical split
      at all** above `one_hot_max_size`: `AddSimpleCtrs` sends the column to
      CTRs (`greedy_tensor_search.cpp:469`) and the one-hot candidate
      generator has already returned for it (`:182`), so the raw column is
      never a candidate. Dropping it from `usable` is how that is said here,
      because `usable` is the pool `sampling.select_tree_features` draws a
      tree's features from and a column outside it is offered to no split
      search.

    Ids in `replaced` that are not in `usable` are ignored rather than
    refused: a prefilter may already have removed one, and "remove it" is
    satisfied either way. The result stays ascending because removal preserves
    order and every appended id is at or above `n_base`.
    """
    var out = List[Int](capacity=len(usable) + n_extra)
    for i in range(len(usable)):
        var f = usable[i]
        var drop = False
        for k in range(len(replaced)):
            if replaced[k] == f:
                drop = True
                break
        if not drop:
            out.append(f)
    for i in range(n_extra):
        out.append(n_base + i)
    return out^


comptime ROW_MAJOR_PACK_MAX_BINS = 16
"""Most bins a feature may use and still be stored in half a byte.

Two features per byte, `(n + 1) / 2` bytes for `n` packable features. A bin id
is a small integer, so packing is lossless by construction: 16 bins are the
ids 0..15 and a nibble holds exactly those. A feature that reaches 17 bins
needs id 16, which does not fit, and stays a whole byte.
"""

comptime ROW_MAJOR_TILE_BYTES = 32 * 1024
"""Row-record bytes one transpose tile writes before moving on.

The transpose reads a column contiguously and writes it with a `row_stride`
stride, so the tile is sized by its *write* window: `tile_rows * row_stride`
bytes stay hot across the whole feature loop, and each of those bytes is
fetched once rather than once per feature. 32 KiB is a conservative floor for
a private L1, chosen on the same grounds as
`apple_cpu_policy.ASSUMED_L1D_BYTES`: too small costs some locality, too large
thrashes. Untuned, and no measurement here justifies a different number.
"""


comptime ROW_MAJOR_OFF = 0
"""`MOJOTREES_CPU_ROW_MAJOR=0`: never build the view, at any size."""

comptime ROW_MAJOR_ON = 1
"""`MOJOTREES_CPU_ROW_MAJOR=1`: always build it, budget or no budget."""

comptime ROW_MAJOR_AUTO = 2
"""`MOJOTREES_CPU_ROW_MAJOR` unset: build it when it fits the budget."""

comptime ROW_MAJOR_DEFAULT_BUDGET_MB = 1024
"""The memory budget, in mebibytes, and the paragraph that states the rule.

*The rule.* Under `auto` -- the default -- the row-major view is built when
`n_rows * row_stride <= budget` and **refused** otherwise, where the budget is
1 GiB and `row_stride` is the realized record width computed in
`BinnedMatrix.build_row_major`, not an estimate. A refusal is silent and
total: no allocation happens, `has_row_major()` stays false, and every
histogram build degrades to feature-major through
`apple_cpu_policy.resolve_bin_layout`, which is the layout that shipped. A
refusal never raises, because a dataset being large is not a user error.
`MOJOTREES_CPU_ROW_MAJOR_MAX_MB` moves the budget; `0` there means no budget
at all, and `MOJOTREES_CPU_ROW_MAJOR=1` overrides it outright.

*Why there is a budget when LightGBM has none.* LightGBM's `auto` allocates
both layouts unconditionally and mentions memory only in a log line
(`src/io/dataset.cpp`, "And if memory is not enough, you can set
`force_col_wise=true`"). The view is a **second full copy** of the bin ids, and
at 255 bins nothing packs into a nibble, so `row_stride == n_features` and the
bin-matrix footprint exactly doubles. Turning a fit that fit into a fit that
does not is a hard failure, not a regression; a slower fit is recoverable and
an OOM is not, so the default has to bound the absolute surprise.

*Why 1 GiB.* It is chosen, not measured, and it is chosen to admit every shape
this optimization was built for and refuse the ones that are dangerous. 1M x
50 is 50 MB, 1M x 200 is 200 MB, 10M x 100 is 1000 MB: all admitted. 10M x 200
is 2 GB and is refused. The bound is absolute rather than a fraction of free
memory for two reasons. There is no portable free-memory query here, and a
rule that read one would give different answers on two runs of the same fit on
the same machine, which makes a report of "it was fast yesterday"
unreproducible. This rule is a pure function of the data: the same matrix gets
the same answer on every machine and at every `MOJOTREES_NUM_WORKERS`.

*What it does not bound.* The feature-major matrix, the Float64 input the
caller may still be holding, the gradient and hessian planes, and the
histograms. It bounds one array, the one this file adds.
"""


def env_row_major_mode() raises -> Int:
    """`MOJOTREES_CPU_ROW_MAJOR`: unset (auto), `1` (force), `0` (never).

    Three states, and unset is not the same as `0`. Unset means **auto**: the
    view is built when it fits `row_major_budget_bytes()`, which is the rule
    written out on `ROW_MAJOR_DEFAULT_BUDGET_MB`. Explicit `1` builds it even
    above the budget, which is how a caller who knows their machine buys the
    layout on a 10M x 200 matrix. Explicit `0` never builds it, which is the
    off switch a bisection wants and the one thing that guarantees the fit
    allocates not one byte for this.

    Raises on an unrecognized value rather than quietly meaning auto, on
    `apple_cpu_policy.env_bin_layout`'s grounds: this campaign has already
    thrown away results from arms that silently ran the other configuration.
    """
    var s = getenv("MOJOTREES_CPU_ROW_MAJOR")
    if s.byte_length() == 0 or s == "auto":
        return ROW_MAJOR_AUTO
    if s == "0" or s == "off" or s == "false":
        return ROW_MAJOR_OFF
    if s == "1" or s == "on" or s == "true":
        return ROW_MAJOR_ON
    raise Error(
        'MOJOTREES_CPU_ROW_MAJOR must be "auto" (or unset), "1" or "0". Got "',
        s,
        '"',
    )


def row_major_budget_bytes() -> Int:
    """Bytes the row-major view may occupy under `auto`, 0 meaning no limit.

    `MOJOTREES_CPU_ROW_MAJOR_MAX_MB` in mebibytes, defaulting to
    `ROW_MAJOR_DEFAULT_BUDGET_MB`, where the rule and the reason for the
    number are written out. `MOJOTREES_CPU_ROW_MAJOR_MAX_MB=0` lifts the
    budget without forcing the view on a dataset that would not have got one.
    """
    return _env_int(
        "MOJOTREES_CPU_ROW_MAJOR_MAX_MB", ROW_MAJOR_DEFAULT_BUDGET_MB
    ) * 1024 * 1024


def row_major_fits_budget(n_rows: Int, row_stride: Int, max_bytes: Int) -> Bool:
    """The budget rule itself, in one place.

    `n_rows * row_stride <= max_bytes`, with `max_bytes <= 0` meaning there is
    no budget.

    A function rather than an inline compare so the rule has one definition,
    a test can assert the boundary without allocating the gigabyte that is
    being refused, and a caller can ask the question before committing to a
    layout. It takes the *realized* stride, so packing a feature into a nibble
    earns admission rather than being rounded away.
    """
    if max_bytes <= 0:
        return True
    if n_rows <= 0 or row_stride <= 0:
        return True
    return n_rows * row_stride <= max_bytes


def _row_major_widths(
    bins: List[UInt8], n_rows: Int, n_features: Int, mut width_of: List[Int]
) raises:
    """Each feature's observed bin count into `width_of[f]`, one parallel max
    per column.

    A free function rather than a method because the caller is `mut self` and
    a closure capturing a pointer whose origin is `self.bins` while `self` is
    mutable is a borrow Mojo will not carry into a parallel task. Taking the
    column array as its own argument gives the pointer its own origin.
    """
    var width_p = width_of.unsafe_ptr()
    var bins_p = bins.unsafe_ptr()

    def scan_feature(f: Int) {imm}:
        var col = f * n_rows
        var hi = 0
        for r in range(n_rows):
            var v = Int(bins_p.unsafe_load(col + r))
            if v > hi:
                hi = v
        width_p.unsafe_store(f, hi + 1)

    dispatch_features(scan_feature, n_features, n_features * n_rows)


def _row_major_fill(
    bins: List[UInt8],
    n_rows: Int,
    n_features: Int,
    stride: Int,
    byte_of: List[Int],
    shift_of: List[Int],
    mut rm: List[UInt8],
) raises:
    """The transpose itself, into a `rm` already sized and zeroed.

    Row-tiled so the strided writes stay inside one hot window across the
    whole feature loop; see `BinnedMatrix.build_row_major` for why the window
    is sized by `ROW_MAJOR_TILE_BYTES` and why this is a second pass rather
    than a fusion into the binning tile.
    """
    var bins_p = bins.unsafe_ptr()
    var rm_p = rm.unsafe_ptr()
    var byte_p = byte_of.unsafe_ptr()
    var shift_p = shift_of.unsafe_ptr()
    var tile = ROW_MAJOR_TILE_BYTES // stride
    if tile < 1:
        tile = 1

    def fill_rows(start: Int, end: Int) {imm}:
        var t0 = start
        while t0 < end:
            var t1 = t0 + tile
            if t1 > end:
                t1 = end
            for f in range(n_features):
                var col = f * n_rows
                var bo = byte_p.unsafe_load(f)
                var sh = shift_p.unsafe_load(f)
                # The record starts zeroed and a byte is written by at most
                # two features of the same row, and a row's record belongs to
                # exactly one task, so the OR that merges a pair of nibbles
                # needs no atomic. An unpacked feature has `sh == 0` and owns
                # its byte outright, so one store covers both cases with no
                # branch in the row loop.
                for r in range(t0, t1):
                    var v = Int(bins_p.unsafe_load(col + r))
                    var idx = r * stride + bo
                    rm_p.unsafe_store(
                        idx, rm_p.unsafe_load(idx) | UInt8(v << sh)
                    )
            t0 = t1

    dispatch_rows(fill_rows, n_rows, n_rows * n_features)


struct BinnedMatrix(Copyable, Movable):
    """Binned feature matrix, in one or two layouts of the same bytes.

    Bin for (row r, feature f) is stored at `bins[f * n_rows + r]`. `cats`
    records which features are categorical, so split finding knows to search
    category partitions rather than ordinal thresholds; an empty spec means
    every feature is numerical. `missing_bin[f]` is the bin reserved for
    missing values of feature f, or -1 when that feature reserves none.

    `usable` is the ascending pool a tree may split on, LightGBM's
    `used_features`. It is every feature unless the matrix came from a mapper
    fit with `feature_pre_filter=True`, and it rides on the matrix rather than
    on the mapper alone because a grower is handed the matrix and nothing else
    -- the same reason `map_forced_splits` exists. The columns are *not*
    renumbered: a filtered feature keeps its id, its column, and its slot in an
    importance vector.
    The row-major view
    ------------------
    `build_row_major` adds a second copy of the same bin ids laid out the
    other way round: one fixed-width **record** per row, `row_stride` bytes,
    holding every feature's bin for that row. Feature f's bin lives in byte
    `row_byte[f]` of the record, in the nibble selected by `row_shift[f]` and
    `row_mask[f]`. `row_bin_at(r, f) == bin_at(r, f)` for every cell; the two
    arrays are the same information and neither is authoritative.

    This is LightGBM's `MultiValBin`, the layout behind `force_row_wise`. It
    exists because a histogram build over a node's row list reads every
    feature of one row: feature-major that is one cache line per (row,
    feature), row-major it is one line per row for all of them.

    **Which builder reads which, and this paragraph is here for the GPU
    reader.** `histogram.build_histogram*` and every `_accumulate_subset*`
    kernel except the `_row_major` ones read `bins`, the feature-major array,
    and always have. Only `histogram.build_histogram_subset_row_major*` reads
    `row_bins`. **Every GPU path reads `bins`.** The device histogram kernels
    upload and scatter the feature-major array, `train_gpu`'s resident data
    plane holds the feature-major array, and nothing on the device has ever
    been handed `row_bins` -- a device scatter wants the coalesced column, not
    the record. So a GPU lane considering a group-major or feature-blocked
    device layout is deciding a *separate* question from this one, and turning
    `MOJOTREES_CPU_ROW_MAJOR` on cannot change a device result. It costs the
    device fit the memory below and nothing else.

    **Memory cost, stated in bytes because a user with a ceiling needs to find
    it without reading the source.** The row-major view is one extra copy of
    the bin matrix, `n_rows * row_stride` bytes, against the feature-major
    `n_rows * n_features` bytes it does not replace. At the headline shape,
    **1,000,000 rows by 50 features: 50 MB feature-major, plus 50 MB
    row-major with no feature packable, plus as little as 25 MB when every
    feature fits in 4 bits** -- so a fit's bin-matrix footprint goes from 50 MB
    to between 75 MB and 100 MB. Packing recovers half a byte per packable
    feature per row and nothing else: it does not shrink `bins`. `row_stride`
    and `row_major_bytes()` report the realized figure for a given dataset.

    **The view is built by default and refused above a budget.** The default
    is `auto`: build it when `n_rows * row_stride` is at most 1 GiB, skip it
    silently above that and run feature-major, which is what shipped.
    `ROW_MAJOR_DEFAULT_BUDGET_MB` states the rule and argues the number,
    `MOJOTREES_CPU_ROW_MAJOR_MAX_MB` moves it, and `MOJOTREES_CPU_ROW_MAJOR`
    forces either way (`1` builds above the budget, `0` never builds). At the
    shapes above: 1M x 50 and 1M x 200 are built, 10M x 200 is not.
    """

    var bins: List[UInt8]
    var n_rows: Int
    var n_features: Int
    var n_bins: Int
    var cats: CategoricalSpec
    var missing_bin: List[Int]
    var usable: List[Int]

    var row_bins: List[UInt8]
    """The record array, `n_rows * row_stride` bytes. Empty until
    `build_row_major` runs."""

    var row_stride: Int
    """Bytes in one row's record, 0 when the view is not built."""

    var row_byte: List[Int]
    """Byte offset of feature f inside a record."""

    var row_shift: List[Int]
    """Right shift applied to that byte for feature f: 0 for a whole byte or
    a low nibble, 4 for a high nibble."""

    var row_mask: List[Int]
    """Mask applied after the shift: 255 for a whole byte, 15 for a
    nibble."""

    var feature_bins: List[Int]
    """Bins feature f actually uses, `max observed bin + 1`.

    Observed from the data rather than declared by the mapper, so a feature
    that reserves 255 bins and uses four is compacted like a four-bin feature
    and packs like one. It is a function of `bins` alone, so two matrices with
    the same bytes always get the same layout.
    """

    var bin_offset: List[Int]
    """Cumulative `feature_bins`, length `n_features + 1`.

    LightGBM's `group_bin_boundaries_` idea: feature f's cells in a *compact*
    histogram start at `bin_offset[f]`, so a private accumulator costs
    `sum_f feature_bins[f]` cells rather than `n_features * n_bins`. Only the
    row-major blocked kernel's private partials use it; the output histogram
    keeps its `f * n_bins + b` shape, which every other file indexes with.
    """

    def __init__(
        out self,
        var bins: List[UInt8],
        n_rows: Int,
        n_features: Int,
        n_bins: Int,
    ):
        self.bins = bins^
        self.n_rows = n_rows
        self.n_features = n_features
        self.n_bins = n_bins
        self.cats = CategoricalSpec.none()
        self.missing_bin = no_missing_bins(n_features)
        self.usable = all_features(n_features)
        self.row_bins = []
        self.row_stride = 0
        self.row_byte = []
        self.row_shift = []
        self.row_mask = []
        self.feature_bins = []
        self.bin_offset = []

    def __init__(
        out self,
        var bins: List[UInt8],
        n_rows: Int,
        n_features: Int,
        n_bins: Int,
        var cats: CategoricalSpec,
        var missing_bin: List[Int] = [],
        var usable: List[Int] = [],
    ):
        """A `missing_bin` table of the wrong length (the empty default
        included) means no feature reserves a missing bin. An empty `usable`
        (the default) means nothing was prefiltered, so every feature is
        usable."""
        self.bins = bins^
        self.n_rows = n_rows
        self.n_features = n_features
        self.n_bins = n_bins
        self.cats = cats^
        self.missing_bin = _sized_missing_bins(missing_bin^, n_features)
        if len(usable) > 0:
            self.usable = usable^
        else:
            self.usable = all_features(n_features)
        self.row_bins = []
        self.row_stride = 0
        self.row_byte = []
        self.row_shift = []
        self.row_mask = []
        self.feature_bins = []
        self.bin_offset = []

    def bin_at(self, row: Int, feature: Int) -> Int:
        return Int(self.bins[feature * self.n_rows + row])

    def is_missing(self, row: Int, feature: Int) -> Bool:
        """Whether (row, feature) holds a missing value."""
        return self.bin_at(row, feature) == self.missing_bin[feature]

    def usable_features(self) -> List[Int]:
        """The pool `feature_fraction` draws from, ascending. Pass it to
        `sampling.select_tree_features`; it is every feature unless the fit
        prefiltered, and passing every feature is the same draw as passing
        nothing."""
        return self.usable.copy()

    def any_usable_categorical(self) -> Bool:
        """Whether any column a split search can be OFFERED is categorical.

        The searchability question, which is not the declaration question
        `CategoricalSpec.any_categorical` answers. A column can be declared
        categorical and be outside `usable`, in which case
        `sampling.select_tree_features` never draws it, no `features` list ever
        names it and no scan ever reaches it. That is exactly the state
        `binning.append_ctr_columns` leaves a CatBoost-mode source column in:
        its CTR columns REPLACE it, so it is binned and predicted through and
        never searched.

        Every guard that refuses a shape because a *category-partition search*
        would be reached must ask this and not the other one. Asking
        `any_categorical` refuses on the declaration, which is a refusal for a
        search that will not happen.

        Linear in the usable count, called once per tree at most, and the
        matrix owns both halves of the question, which is why it lives here
        rather than on `CategoricalSpec` -- the spec knows the flags and knows
        nothing about the pool.
        """
        for i in range(len(self.usable)):
            if self.cats.is_cat(self.usable[i]):
                return True
        return False

    def has_row_major(self) -> Bool:
        """Whether the row-major view is built and the right size.

        The size check is not paranoia: a `BinnedMatrix` is copied and
        serialized in several places, and a kernel that indexed a stale record
        array would read garbage bins rather than fail.
        """
        return (
            self.row_stride > 0
            and self.n_rows > 0
            and self.n_features > 0
            and len(self.row_bins) == self.n_rows * self.row_stride
            and len(self.row_byte) == self.n_features
            and len(self.row_shift) == self.n_features
            and len(self.row_mask) == self.n_features
            and len(self.feature_bins) == self.n_features
            and len(self.bin_offset) == self.n_features + 1
        )

    def row_bin_at(self, row: Int, feature: Int) -> Int:
        """The bin at (row, feature), read from the record array.

        Equal to `bin_at(row, feature)` for every cell of a built view. This
        is the reference the tests compare the kernels against; the kernels
        inline the same three operations with the per-feature constants
        hoisted out of the row loop.
        """
        var raw = Int(self.row_bins[row * self.row_stride + self.row_byte[feature]])
        return (raw >> self.row_shift[feature]) & self.row_mask[feature]

    def row_major_bytes(self) -> Int:
        """Bytes the row-major view occupies, 0 when it is not built. The
        number the memory paragraph above is about, for this dataset."""
        return len(self.row_bins)

    def packed_feature_count(self) -> Int:
        """Features stored in half a byte. 0 when the view is not built."""
        var n = 0
        for f in range(len(self.row_mask)):
            if self.row_mask[f] == 15:
                n += 1
        return n

    def compact_bin_count(self) -> Int:
        """`sum_f feature_bins[f]`, the cell count of a compact histogram over
        every feature. 0 when the view is not built."""
        if len(self.bin_offset) != self.n_features + 1:
            return 0
        return self.bin_offset[self.n_features]

    def build_row_major(mut self, max_bytes: Int = 0) raises:
        """Build (or rebuild) the row-major record array from `bins`.

        `max_bytes` is the memory budget, `0` (the default) meaning none, and
        it is checked against the **realized** `n_rows * row_stride` after the
        layout is decided and before anything is allocated. Over budget, this
        is a no-op that leaves the matrix in the state `drop_row_major` leaves
        it: nothing allocated, `has_row_major()` false, every histogram build
        degrading to feature-major. It does not raise. The rule and the number
        are on `ROW_MAJOR_DEFAULT_BUDGET_MB`; the policy that picks the
        argument is `env_row_major_mode` at the `transform` call site, so this
        method stays mechanism and the default keeps its old contract, which
        is "build it".

        A refusal still pays the width scan below -- reads only, no
        allocation -- because the exact answer needs the realized stride and
        the conservative one (`n_rows * n_features`, nothing packed) would
        refuse a low-cardinality matrix at twice its true cost. One scan of a
        matrix that is about to be trained on many times is the cheaper
        mistake.

        Three passes, none of which touches a Float64 and none of which can
        change a bin id:

        1. **Widths.** Each feature's observed bin count, one parallel max
           over its column. Derived from the data rather than from the
           `BinMapper`, so a `BinnedMatrix` assembled by `efb`, by
           `serialize`, or by a test gets the same treatment as one that came
           out of `transform`, and so a feature that reserves many bins and
           uses few is packed on what it uses.
        2. **Layout.** Features needing more than `ROW_MAJOR_PACK_MAX_BINS`
           bins take a whole byte each, in ascending feature order, at the
           front of the record; the rest take half a byte each, two to a byte,
           in ascending feature order, after them. `row_stride` is
           `unpacked + (packable + 1) / 2`. Splitting the record this way
           rather than interleaving keeps the assignment a two-line rule and
           costs nothing: a record is read through `row_byte[f]`, so nothing
           downstream cares what order the bytes are in, and at any feature
           count where the record spans more than a cache line the packed half
           is the dense half either way.
        3. **Transpose.** Row-tiled, `ROW_MAJOR_TILE_BYTES` of record window
           at a time, feature-major inside the tile so the *reads* stay
           sequential down each column and the *writes* stay inside a window
           that is still hot when the next feature arrives. Dispatched over
           disjoint ascending row ranges; a row's record is written by exactly
           one task, so the two features sharing a byte are written by the
           same task and the OR that merges them needs no atomic.

        The fused alternative -- writing the record inside `transform`'s
        binning tile -- was rejected on write traffic, and the arithmetic is
        worth keeping: that loop is dispatched per (feature, row range), so a
        fixed feature walks the whole record array with a `row_stride` stride.
        At 1,000,000 rows by 50 features the array is 50 MB and a 128-byte
        line holds two of that feature's bytes, so each of the 50 features
        pulls ~25 MB of lines, ~1.25 GB of read-for-ownership traffic against
        the 100 MB (one read of `bins`, one write of `row_bins`) the tiled
        pass moves. A second pass that moves a twelfth of the traffic is not a
        second pass worth fusing away.
        """
        var nf = self.n_features
        var nr = self.n_rows
        var byte_of = List[Int]()
        var shift_of = List[Int]()
        var mask_of = List[Int]()
        var width_of = List[Int]()
        var offsets = List[Int]()
        var rm = List[UInt8]()
        var stride = 0

        if nf > 0 and nr > 0 and len(self.bins) == nr * nf:
            width_of.resize(nf, 1)
            _row_major_widths(self.bins, nr, nf, width_of)

            byte_of.resize(nf, 0)
            shift_of.resize(nf, 0)
            mask_of.resize(nf, 255)
            var whole = 0
            for f in range(nf):
                if width_of[f] > ROW_MAJOR_PACK_MAX_BINS:
                    byte_of[f] = whole
                    whole += 1
            var packed = 0
            for f in range(nf):
                if width_of[f] <= ROW_MAJOR_PACK_MAX_BINS:
                    byte_of[f] = whole + (packed >> 1)
                    shift_of[f] = 4 if (packed & 1) == 1 else 0
                    mask_of[f] = 15
                    packed += 1
            stride = whole + ((packed + 1) >> 1)

            # The budget, checked here and nowhere else: after the stride is
            # real and before the one allocation that costs the memory.
            if not row_major_fits_budget(nr, stride, max_bytes):
                self.drop_row_major()
                return

            offsets.append(0)
            for f in range(nf):
                offsets.append(offsets[f] + width_of[f])

            rm.resize(nr * stride, UInt8(0))
            _row_major_fill(
                self.bins, nr, nf, stride, byte_of, shift_of, rm
            )

        self.row_bins = rm^
        self.row_stride = stride
        self.row_byte = byte_of^
        self.row_shift = shift_of^
        self.row_mask = mask_of^
        self.feature_bins = width_of^
        self.bin_offset = offsets^

    def drop_row_major(mut self):
        """Release the row-major view and its `row_major_bytes()`."""
        self.row_bins = []
        self.row_stride = 0
        self.row_byte = []
        self.row_shift = []
        self.row_mask = []
        self.feature_bins = []
        self.bin_offset = []


def _rebuild_row_major(mut matrix: BinnedMatrix, build_view: Bool) raises:
    """Drop the row-major view and rebuild it under the environment policy.

    The same three-way policy `BinMapper.transform` applies at the end of a fit:
    `auto` offers the budget, an explicit `1` passes none, an explicit `0` does
    not build. Factored out because appending CTR columns has to redo the
    decision -- the packing widths are a function of every column's realized bin
    count, so four new 16-bin columns can change which features fit in a nibble.
    """
    matrix.drop_row_major()
    var mode = ROW_MAJOR_OFF if not build_view else env_row_major_mode()
    if mode == ROW_MAJOR_ON:
        matrix.build_row_major(0)
    elif mode == ROW_MAJOR_AUTO:
        matrix.build_row_major(row_major_budget_bytes())


def append_ctr_columns(
    mut matrix: BinnedMatrix,
    ctr_bins: List[UInt8],
    n_ctr_columns: Int,
    build_view: Bool = True,
    replaced: List[Int] = [],
) raises:
    """Append CTR columns to a binned matrix, in place, whichever half made them.

    After this call `matrix.n_features` counts them, so every histogram builder,
    every split search and every grower sees ordinary binned numeric features and
    needs no edit: `histogram.build_histogram_into_scratch` reads
    `data.bins[f * n_rows + r]`, which is where these bytes now are, and
    `data.usable` now offers the new ids to a split search.

    `replaced` names base columns these CTR columns stand IN PLACE OF rather
    than beside; they are dropped from `matrix.usable` and no split search is
    offered them again. See `ctr_extend_usable` for which policy passes what
    and why the two halves differ. The columns themselves stay in `bins` and
    keep their ids: `usable` is a search pool and not a renumbering, so a
    replaced column is still binned, still predicted through and still occupies
    its own slot in an importance vector, exactly as a prefiltered one is.

    `ctr_bins` is column-major, `ctr_bins[c * n_rows + r]`, and every byte in it
    is already a bucket index -- the training half because `ctr.ctr_train_bin`
    returns CatBoost's `ui8` directly, the inference half because
    `ctr.ctr_predict_bucket` truncates the unquantized value. There is no
    binarization pass over CTR values on either side, for the reason
    `ctr.check_ctr_border_type` states.

    Two callers, and the difference between them is the whole mechanism.

    - **Training**: `trainset._build_ctr` hands it
      `ctr_columns.build_ctr_train_columns`' output -- the ordered prefix of one
      permutation, denominator `totalCount + 1`, read-before-write so row `i`
      never sees its own target. Swapping the read and the write inside
      `ctr.ordered_ctr_borders_binary` is the single edit that would silently
      turn this back into ordinary target encoding.
    - **Inference**: `BinMapper.transform` hands it
      `ctr_columns.ctr_predict_columns`' output -- the static tables over the
      whole learn set, denominator `t + PriorDenom`, truncated by
      `EmulateUi8Rounding`'s rule.

    A `BinMapper` can only ever make the second call: it holds no permutation
    and no target, so it *cannot* evaluate the training formula. That is the
    train/predict separation enforced by what the type owns rather than by a
    comment.
    """
    if n_ctr_columns < 1:
        return
    var n_rows = matrix.n_rows
    if len(ctr_bins) != n_ctr_columns * n_rows:
        raise Error("ctr column buffer must be n_ctr_columns * n_rows bytes")
    var n_base = matrix.n_features
    for i in range(len(ctr_bins)):
        matrix.bins.append(ctr_bins[i])
    matrix.n_features = n_base + n_ctr_columns
    matrix.cats = ctr_extend_cats(matrix.cats, n_ctr_columns)
    matrix.missing_bin = ctr_extend_missing(matrix.missing_bin, n_ctr_columns)
    matrix.usable = ctr_extend_usable(
        matrix.usable, n_base, n_ctr_columns, replaced
    )
    _rebuild_row_major(matrix, build_view)


struct BinMapper(Copyable, Movable):
    """Per-feature bin edges fit on training data.

    Feature f's edges are `edges[edge_offsets[f] : edge_offsets[f + 1]]`,
    strictly increasing. A value v maps to the first bin b whose edge
    satisfies v <= edge[b]; values above every edge map to the last bin.
    A feature with k edges uses k + 1 bins (k + 1 <= n_bins).

    Categorical features carry no edges; `cats` holds their category tables
    and `bin_value` routes them through it instead.

    A numerical feature with `missing_bin[f] >= 0` reserves that bin for
    missing values, so its k edges give ordinary bins 0..k and a missing bin
    at k + 1. `missing_bin[f] = -1` means no reservation.

    `usable` is the ascending list of feature ids a tree may split on, which is
    every feature unless `fit_bins` ran with `feature_pre_filter=True`. It is
    LightGBM's `used_features`: `Dataset::Construct` builds exactly this list
    (`src/io/dataset.cpp`, from `!BinMapper::is_trivial()`) and it is the pool
    `feature_fraction` samples from. It is *not* a renumbering. The matrix keeps
    every column and every feature keeps its own id, so a filtered feature is
    still binned, still predicted through, and still occupies its own slot in an
    importance vector -- which is what LightGBM does too, because
    `GBDT::FeatureImportance` sizes its result by `max_feature_idx_ + 1 =
    num_total_features` rather than by the used count.
    """

    var edges: List[Float64]
    var edge_offsets: List[Int]
    var n_features: Int
    var n_bins: Int
    var cats: CategoricalSpec
    var missing_bin: List[Int]
    var usable: List[Int]

    var ctr: CtrTables
    """Fitted ordered-target-statistic tables, catalog A19. `CtrTables.none()`
    unless `attach_ctr` put some here, which nothing does by default.

    **`n_features` does not count CTR columns.** A raw row and a raw matrix stay
    exactly as wide as the caller's data, which is what keeps every existing
    call site of `transform` and `bin_row` correct without an edit; the *binned*
    width is `n_total_features()`. That split is also the thing a consumer gets
    wrong: anything that walks features by `n_features` and then indexes by a
    tree's `feature[i]` reads past its own arrays on a CTR model, which is why
    `model_dump._build` refuses one by name.

    The tables are model state (they are read off the target) and they travel
    with the model: `serialize._write_ctr` writes them as format v5's optional
    `ctr` section and `serialize._read_ctr` reads them back inside
    `_read_mapper`, so a saved mapper and its tables cannot be separated.
    `ctr_columns.check_ctr_serializable` guards the one part of the state the
    file leaves out (the derived bucket -> bin lookup, rebuilt on load), and
    `ctr_columns.check_ctr_dataset_serializable` still refuses at the prepared
    table writer, which carries the *ordered training* columns and cannot
    describe them.
    """

    def __init__(
        out self,
        var edges: List[Float64],
        var edge_offsets: List[Int],
        n_features: Int,
        n_bins: Int,
    ):
        self.edges = edges^
        self.edge_offsets = edge_offsets^
        self.n_features = n_features
        self.n_bins = n_bins
        self.cats = CategoricalSpec.all_numerical(n_features)
        self.missing_bin = no_missing_bins(n_features)
        self.usable = all_features(n_features)
        self.ctr = CtrTables.none()

    def __init__(
        out self,
        var edges: List[Float64],
        var edge_offsets: List[Int],
        n_features: Int,
        n_bins: Int,
        var cats: CategoricalSpec,
        var missing_bin: List[Int] = [],
        var usable: List[Int] = [],
    ):
        """A `missing_bin` table of the wrong length (the empty default
        included) means no feature reserves a missing bin. An empty `usable`
        (the default) means no feature was prefiltered, so every feature is
        usable."""
        self.edges = edges^
        self.edge_offsets = edge_offsets^
        self.n_features = n_features
        self.n_bins = n_bins
        self.cats = cats^
        self.missing_bin = _sized_missing_bins(missing_bin^, n_features)
        if len(usable) > 0:
            self.usable = usable^
        else:
            self.usable = all_features(n_features)
        self.ctr = CtrTables.none()

    def n_total_features(self) -> Int:
        """Columns a binned matrix or a binned row has: the base features plus
        the CTR columns. Equal to `n_features` unless CTR tables are attached,
        so nothing that reads `n_features` today changes meaning."""
        if not self.ctr.is_active():
            return self.n_features
        return self.n_features + self.ctr.n_columns()

    def has_ctr(self) -> Bool:
        return self.ctr.is_active()

    def attach_ctr(mut self, var tables: CtrTables) raises:
        """Take ownership of the fitted CTR tables, after the training columns
        have already been built.

        **Order matters and this is the whole of the train/predict separation.**
        A mapper with tables attached appends the *static-table* CTR columns to
        everything it transforms or bins, which is right for a validation set, a
        prediction batch and a scored row and would be catastrophically wrong for
        the matrix the trees are grown on. The dataset path therefore transforms
        the base matrix while the mapper is still bare, appends the *ordered*
        columns with `append_ctr_train_columns`, fits the tables, and calls this
        last. After this call the mapper can no longer produce the training
        matrix, and nothing asks it to.

        A `Model` built from a mapper in this state predicts correctly and, as
        of format v5, saves correctly too: `serialize._write_ctr` writes these
        tables and `serialize._read_ctr` calls this method to put them back.
        The trainer-boundary refusal that used to stand in `trainset.mojo` is
        gone with the reason for it; `ctr.check_ctr_model_support` survives as
        the model-dump guard, which is a schema limit and not a save one.
        """
        if tables.is_active():
            if tables.n_base_features != self.n_features:
                raise Error(
                    "ctr tables were planned for a different feature count"
                )
            if tables.n_columns() > 0:
                var need = tables.columns[0].n_buckets()
                for i in range(tables.n_columns()):
                    var b = tables.columns[i].n_buckets()
                    if b > need:
                        need = b
                if need > self.n_bins:
                    raise Error(
                        "ctr columns need ",
                        need,
                        " bins and this binning reserves only ",
                        self.n_bins,
                        ": raise max_bin to at least ctr_border_count + 1",
                    )
        self.ctr = tables^

    def usable_features(self) -> List[Int]:
        """The pool `feature_fraction` draws from, ascending. Every feature
        unless the fit prefiltered."""
        return self.usable.copy()

    def drop_usable(mut self, features: List[Int]):
        """Take `features` out of the search pool, keeping it ascending.

        Called once, by `trainset._build_ctr`, for the CatBoost-mode source
        columns whose CTR columns REPLACE them. It is recorded on the MAPPER
        and not only on the training matrix so that the two agree: a matrix
        this mapper transforms later -- a validation set through
        `Dataset.from_reference`, a batch through `Model.predict_batch` --
        copies `self.usable` and then has its CTR ids appended, and without
        this it would offer a column the training matrix did not.

        `usable` is not written by `serialize`, so a mapper read back from a
        file has the default pool again. That is inert rather than wrong:
        nothing on the inference path consults `usable`, which is the pool
        `sampling.select_tree_features` draws from and is read at fit time
        only. Stated here rather than discovered, because the natural reading
        of "model state" would have expected it to survive.
        """
        var out = List[Int](capacity=len(self.usable))
        for i in range(len(self.usable)):
            var f = self.usable[i]
            var drop = False
            for k in range(len(features)):
                if features[k] == f:
                    drop = True
                    break
            if not drop:
                out.append(f)
        self.usable = out^

    def is_usable(self, feature: Int) -> Bool:
        """Whether `feature` survived the prefilter. Linear in the usable
        count, which is what keeps the representation a plain ascending list
        rather than a second per-feature table."""
        for i in range(len(self.usable)):
            if self.usable[i] == feature:
                return True
        return False

    def has_missing(self) -> Bool:
        """Whether any feature reserves a missing bin."""
        for f in range(self.n_features):
            if self.missing_bin[f] >= 0:
                return True
        return False

    # `raises` because the CTR comparison below walks two fitted table
    # sets and indexes them, and `CtrTables.matches` is declared
    # accordingly. All three callers -- model_editing._check_refit_dataset
    # and external_memory.update_external{,_multiclass} -- are already
    # raising, so this widens no contract.
    def matches(self, other: BinMapper) raises -> Bool:
        """Whether two mappers bin every value the same way.

        Equality of the fitted binning, not of the objects: same features,
        same edges, same missing reservations, same category tables. It is
        what lets a fitted model take more trees from a dataset that was
        binned separately (see boosting.train_more) without any chance of a
        bin index meaning two different things.
        """
        if self.n_features != other.n_features:
            return False
        if self.n_bins != other.n_bins:
            return False
        if len(self.edges) != len(other.edges):
            return False
        if len(self.edge_offsets) != len(other.edge_offsets):
            return False
        for i in range(len(self.edges)):
            if self.edges[i] != other.edges[i]:
                return False
        for i in range(len(self.edge_offsets)):
            if self.edge_offsets[i] != other.edge_offsets[i]:
                return False
        if len(self.missing_bin) != len(other.missing_bin):
            return False
        for i in range(len(self.missing_bin)):
            if self.missing_bin[i] != other.missing_bin[i]:
                return False
        # A prefiltered mapper and an unfiltered one bin every value the same
        # way but do not offer the same features to a tree, so `train_more`
        # must not accept one for the other.
        if len(self.usable) != len(other.usable):
            return False
        for i in range(len(self.usable)):
            if self.usable[i] != other.usable[i]:
                return False
        for f in range(self.n_features):
            if self.cats.is_cat(f) != other.cats.is_cat(f):
                return False
        if len(self.cats.codes) != len(other.cats.codes):
            return False
        for i in range(len(self.cats.codes)):
            if self.cats.codes[i] != other.cats.codes[i]:
                return False
        if len(self.cats.offsets) != len(other.cats.offsets):
            return False
        for i in range(len(self.cats.offsets)):
            if self.cats.offsets[i] != other.cats.offsets[i]:
                return False
        # Two mappers that bin the base features identically still disagree
        # about a CTR column if their fitted tables differ, and a CTR column's
        # bin id would then mean two different things -- the exact failure the
        # rest of this function exists to prevent. `train_more` must not accept
        # one for the other.
        if not self.ctr.matches(other.ctr):
            return False
        return True

    def bin_value(self, feature: Int, v: Float64) -> Int:
        if self.cats.is_cat(feature):
            return self.cats.bin_of(feature, v)
        var value = v
        if isnan(value):
            var mb = self.missing_bin[feature]
            if mb >= 0:
                return mb
            # No reserved bin: LightGBM bins NaN as 0.0 for a feature whose
            # missing_type is None.
            value = 0.0
        var lo = self.edge_offsets[feature]
        var left = lo
        var right = self.edge_offsets[feature + 1]
        while left < right:
            var mid = (left + right) // 2
            if value <= self.edges[mid]:
                right = mid
            else:
                left = mid + 1
        return left - lo

    def transform[
        features_origin: ImmOrigin, //
    ](
        self,
        features: Span[Float64, features_origin],
        n_rows: Int,
        build_view: Bool = True,
    ) raises -> BinnedMatrix:
        """Bin a column-major feature matrix (`features[f * n_rows + r]`).

        `features` is a borrowed view, so a caller holding the matrix
        somewhere other than a Mojo `List` -- the Python bindings hold
        NumPy's own buffer -- bins it in place instead of copying it first.
        A `List` converts implicitly, so passing one still works.

        `build_view=False` is the opt-out for a matrix that will be **scored
        and not histogrammed**: a prediction input, a validation set, a
        `predict_contrib` call. The row-major view is read by exactly one
        thing, `histogram.build_histogram_subset_row_major*`, so a matrix that
        never reaches a histogram builder pays a transpose pass and a second
        copy of its bin ids for nothing -- and the copy is the same doubling
        that `ROW_MAJOR_DEFAULT_BUDGET_MB` exists to bound, arriving on a code
        path the user did not think of as a fit. The default is `True`,
        meaning "apply the environment policy", because a call site that omits
        it is more likely to be a fit than a predict; the predict-side call
        sites that should pass `False` are named in the lane report rather
        than edited here, because they live in files this lane does not own.
        """
        if len(features) != n_rows * self.n_features:
            raise Error("features length must equal n_rows * n_features")
        var n_features = self.n_features
        var bins = List[UInt8](capacity=n_rows * n_features)
        bins.resize(n_rows * n_features, 0)

        # A per-feature search table, built once per call: each numerical
        # feature's edges padded up to a power of two with `+inf` sentinels.
        # That buys a fixed, branch-free descent per value in place of a
        # data-dependent loop whose every step is a mispredictable branch. A
        # sentinel never counts (`+inf < v` is false for every `v`, `+inf`
        # included), so padding cannot move a bin, and `bin_value` keeps the
        # plain search as the reference the tests compare against.
        #
        # The table is at most `n_features * n_bins` doubles, built in
        # `n_features * n_bins` work against `n_features * n_rows` searches.
        var pad = List[Float64](capacity=n_features * self.n_bins)
        var pad_offsets = List[Int](capacity=n_features + 1)
        var pad_half = List[Int](capacity=n_features)
        pad_offsets.append(0)
        for f in range(n_features):
            var lo = self.edge_offsets[f]
            var k = self.edge_offsets[f + 1] - lo
            # Smallest power of two strictly greater than k, so the descent's
            # largest reachable answer (k) is representable and a feature with
            # no edges still gets one sentinel to point at.
            var m = 1
            while m <= k:
                m += m
            for i in range(k):
                pad.append(self.edges[lo + i])
            for _ in range(k, m):
                pad.append(POSITIVE_INF)
            pad_offsets.append(len(pad))
            pad_half.append(m >> 1)

        var bins_p = bins.unsafe_ptr()
        var feat_p = features.unsafe_ptr()
        var pad_p = pad.unsafe_ptr()
        var poff_p = pad_offsets.unsafe_ptr()
        var half_p = pad_half.unsafe_ptr()
        var miss_p = self.missing_bin.unsafe_ptr()
        ref cats = self.cats

        def do_tile(f: Int, r_lo: Int, r_hi: Int) {imm}:
            var col = f * n_rows
            if cats.is_cat(f):
                for r in range(r_lo, r_hi):
                    bins_p.unsafe_store(
                        col + r,
                        UInt8(cats.bin_of(f, feat_p.unsafe_load(col + r))),
                    )
                return
            var pbase = poff_p.unsafe_load(f)
            var half = half_p.unsafe_load(f)
            var mb = miss_p.unsafe_load(f)
            for r in range(r_lo, r_hi):
                var v = feat_p.unsafe_load(col + r)
                # NaN is routed before any comparison, so it never takes part
                # in the quantile search (see `bin_value`).
                if isnan(v):
                    if mb >= 0:
                        bins_p.unsafe_store(col + r, UInt8(mb))
                        continue
                    v = 0.0
                # Count the edges strictly below `v`, which is the bin
                # `bin_value`'s search arrives at: it stops at the first edge
                # with `v <= edge`, and the edges are strictly increasing.
                var pos = 0
                var step = half
                while step > 0:
                    var nxt = pos + step
                    var go = pad_p.unsafe_load(pbase + nxt - 1) < v
                    pos = nxt if go else pos
                    step = step >> 1
                bins_p.unsafe_store(col + r, UInt8(pos))

        # A row costs one binary search over at most `n_bins` edges, not one
        # accumulate: about `log2(n_bins)` dependent compares, each on a hot
        # but data-dependent load.
        #
        # Split by feature *and* by rows. Binning a cell reads that cell and
        # writes that cell, so tiles are independent and the bins are the same
        # bytes however the tiles fall. With features to spare this is the
        # by-feature split it has always been; with only a handful of features
        # over a long history it is what keeps the other cores working.
        dispatch_feature_rows(
            do_tile,
            n_features,
            n_rows,
            n_features * n_rows * (1 + _log2_ceil(self.n_bins)),
        )
        var out = BinnedMatrix(
            bins^,
            n_rows,
            n_features,
            self.n_bins,
            self.cats.copy(),
            self.missing_bin.copy(),
            self.usable.copy(),
        )
        # The INFERENCE half of catalog A19, and it runs here and only here.
        #
        # `self.ctr` is `CtrTables.none()` unless `attach_ctr` was called, and
        # `attach_ctr` is called only *after* the fit that produced this mapper
        # has already built its training matrix. So this branch is dead during a
        # fit, by construction rather than by a flag, and it is live for exactly
        # the matrices that should get static-table columns: a validation set
        # binned through `Dataset.from_reference`, and every raw matrix handed to
        # `Model.predict_batch`.
        #
        # `ctr_predict_columns` reads the tables `CalcFinalCtrs` built over the
        # whole learn set with the denominator `t + PriorDenom` and truncates
        # with `EmulateUi8Rounding`'s rule. It is a different formula over
        # different data from the one that made the training columns, which is
        # A19's central warning and the reason the two paths do not share a line
        # of code.
        #
        # Cost when inactive: one `is_active()` load and a not-taken branch. No
        # allocation, no pass, and the matrix is byte-identical to the one this
        # function returned before CTRs existed.
        if self.ctr.is_active():
            # RAW values, not `out.bins`. The buckets a CTR is keyed on are
            # the un-truncated category codes, so the source here is the same
            # matrix `transform` was handed and never the one it just wrote.
            var cat = ctr_slot_columns(features, n_rows, self.ctr)
            var extra = ctr_predict_columns(self.ctr, cat, n_rows)
            append_ctr_columns(
                out, extra, self.ctr.n_columns(), build_view=False
            )
        # The row-major view, at fit time. Its widths come from the bins this
        # call just wrote, so it is built here rather than fused into the tile
        # above: the packing decision needs each feature's realized bin count,
        # which does not exist until the last tile has run. See
        # `BinnedMatrix.build_row_major` for the write traffic that makes the
        # second pass the cheap way round anyway.
        #
        # This is the policy half of the decision and the only place the
        # environment is read: `auto` (the default) offers the budget, an
        # explicit `1` passes no budget at all, and an explicit `0` does not
        # call the builder. Whether the view, once built, is the one the
        # histograms actually read is a *separate* decision made once per fit
        # by `histogram.choose_bin_layout_timed`; building it only makes both
        # arms runnable.
        var mode = ROW_MAJOR_OFF if not build_view else env_row_major_mode()
        if mode == ROW_MAJOR_ON:
            out.build_row_major(0)
        elif mode == ROW_MAJOR_AUTO:
            out.build_row_major(row_major_budget_bytes())
        return out^

    def bin_row(self, row: List[Float64]) raises -> List[Int]:
        """Bin one example for prediction.

        The raw row is `n_features` long -- the caller's own feature count,
        which CTRs do not change. The result is `n_total_features()` long,
        because a CTR column is not something a caller can supply: its value is
        a statistic of the training target and the mapper is the only thing that
        holds it.

        This is why `Model.predict`, `Model.predict_raw`, `Model.leaf_indices`
        and their multiclass counterparts score a CTR model with no edit to
        `model.mojo`: they all bin through here, and the trees index the vector
        this returns.
        """
        if len(row) != self.n_features:
            raise Error("row length must equal n_features")
        var out = List[Int](capacity=self.n_total_features())
        for f in range(self.n_features):
            out.append(self.bin_value(f, row[f]))
        if self.ctr.is_active():
            # The static-table half again, one row at a time, keyed on the RAW
            # row rather than on the bins just written: the bucket a CTR needs
            # is the un-truncated category code. Same tables, same formula and
            # same truncation as `ctr_predict_columns` takes in bulk, so a row
            # scored singly and the same row scored in a batch get the same bin.
            var extra = ctr_predict_row(self.ctr, row)
            for i in range(len(extra)):
                out.append(extra[i])
        return out^


def fit_bins[
    features_origin: ImmOrigin, //
](
    features: Span[Float64, features_origin],
    n_rows: Int,
    n_features: Int,
    max_bins: Int = 255,
    categorical_features: List[Int] = [],
    use_missing: Bool = True,
    min_data_in_bin: Int = DEFAULT_MIN_DATA_IN_BIN,
    bin_construct_sample_cnt: Int = DEFAULT_BIN_CONSTRUCT_SAMPLE_CNT,
    data_random_seed: Int = DEFAULT_DATA_RANDOM_SEED,
    feature_pre_filter: Bool = False,
    min_data_in_leaf: Int = DEFAULT_MIN_DATA_IN_LEAF,
    border_type: Int = BORDER_QUANTILE,
) raises -> BinMapper:
    """Fit quantile (equal-frequency) bin edges on a column-major feature
    matrix. Edges are midpoints between distinct values at quantile
    boundaries, so duplicate-heavy features simply use fewer bins.

    Feature indices listed in `categorical_features` are treated as
    integer-coded categoricals: they are excluded from quantile binning
    entirely and get a category table instead (see `categorical.mojo`).

    With `use_missing` (the default), a numerical feature whose column holds
    any `NaN` reserves its highest bin for missing values and fits its edges
    over the remaining `max_bins - 1` bins from the non-missing values alone,
    so `NaN` never enters a quantile comparison. `use_missing=False` reserves
    nothing and bins `NaN` as 0.0, matching LightGBM's `use_missing=false`.

    `min_data_in_bin` is LightGBM's minimum population for a numerical bin
    and `bin_construct_sample_cnt` fits the edges from a subsample of the
    rows, seeded by `data_random_seed`. Both change which bins exist, and
    both now default to **LightGBM's** values, 3 and 200,000, rather than to
    the 1 and 0 at which this function is the function it was before they
    existed. Passing `min_data_in_bin=1, bin_construct_sample_cnt=0` restores
    that fit exactly, and a test says so. The module docstring states what
    each honors and what it does not.

    `feature_pre_filter` is LightGBM's, and `min_data_in_leaf` is the number it
    scales (see `filter_count`); it has no other use here, so a caller who
    leaves the filter off never has to keep the two in step. At `True` the fit
    additionally counts each feature's bins on the same sample it fit the edges
    from, runs `need_filter` on those counts, and returns a `BinMapper` whose
    `usable` list omits every feature LightGBM's `Dataset::Construct` would
    have left out of `used_features` -- the trivial ones (one bin) and the
    filtered ones. **Off is the exact fit this function has always produced**:
    no counting pass runs, no allocation is made for it, and `usable` is every
    feature.

    Two divergences from LightGBM, both deliberate and both on the record:

    - LightGBM drops a one-bin feature from `used_features` whatever
      `feature_pre_filter` says. mojotrees drops nothing at `False`, because
      `False` has to stay bit-identical to the fit that preceded this argument
      and dropping constant columns from the `feature_fraction` pool is not a
      bit-identical change. So mojotrees's `False` is "keep everything", not
      LightGBM's `false`.
    - LightGBM warns and continues when every feature is trivial. There is
      nowhere to warn to here and a fit with no usable feature has no tree to
      grow, so this raises instead.

    `border_type` selects CatBoost's border selection instead of the quantile
    fit above. **`BORDER_QUANTILE` is the default and is this function
    unchanged**: the CatBoost work sits behind one branch per FEATURE inside
    the column loop, nothing per row, and its scratch buffers stay
    unallocated on that arm. Any other value is opt-in, bit-moving by
    construction (a different rule for where a border goes is a different
    model), and costs a full sort of each column where the default arm
    resolves a few hundred ranks by bucket selection. See
    `docs/design/CATBOOST_CATALOG.md` A15, `check_border_type` for the two
    types refused by name, and `parse_border_type` for the spellings.
    """
    if max_bins < 2 or max_bins > MAX_BINS:
        raise Error("max_bins must be in [2, ", MAX_BINS, "]")
    check_border_type(border_type)
    if border_type != BORDER_QUANTILE and min_data_in_bin > 1:
        # `min_data_in_bin` is LightGBM's and CatBoost has no analogue, so
        # honoring it on a CatBoost arm would mean inventing an interaction
        # neither library defines. Refused rather than silently ignored.
        raise Error(
            "min_data_in_bin is LightGBM's and has no CatBoost analogue; it"
            " cannot be combined with border_type '",
            border_type_name(border_type),
            "'. Pass min_data_in_bin=1",
        )
    if n_rows < 1:
        raise Error("n_rows must be positive")
    if len(features) != n_rows * n_features:
        raise Error("features length must equal n_rows * n_features")
    if min_data_in_bin < 1:
        raise Error("min_data_in_bin must be positive")
    if bin_construct_sample_cnt < 0:
        raise Error("bin_construct_sample_cnt must be non-negative")
    if feature_pre_filter and min_data_in_leaf < 0:
        raise Error("min_data_in_leaf must be nonnegative")

    var cats = fit_categorical_spec(
        features, n_rows, n_features, categorical_features, max_bins
    )

    # Each feature's edges land in its own fixed-stride scratch slice; a
    # serial pass then concatenates them in feature order, so the result is
    # identical whichever path ran.
    var max_edges = max_bins - 1
    var scratch = List[Float64](capacity=n_features * max_edges)
    scratch.resize(n_features * max_edges, 0.0)
    var counts = List[Int](capacity=n_features)
    counts.resize(n_features, 0)
    var missing_bin = no_missing_bins(n_features)

    var scratch_p = scratch.unsafe_ptr()
    var counts_p = counts.unsafe_ptr()
    var missing_p = missing_bin.unsafe_ptr()
    var feat_p = features.unsafe_ptr()
    var select_min_rows = env_select_min_rows()
    ref spec = cats

    # Drawn once, before any feature is dispatched, so every feature is fit
    # from the same rows and the sample cannot depend on the worker count.
    # Empty means every row, which is the default and keeps the column loop
    # below the loop it has always been.
    var sample = bin_construct_sample_rows(
        n_rows, bin_construct_sample_cnt, data_random_seed
    )
    var sampled = len(sample) > 0
    var n_sample = len(sample) if sampled else n_rows
    var sample_p = sample.unsafe_ptr()

    # LightGBM's prefilter state. `filter_cnt` is drawn once, from the same two
    # sizes for every feature, so it cannot depend on which task fit a column.
    # Both buffers are unallocated when the filter is off, which is what keeps
    # that path allocation-for-allocation the path it was.
    var n_flags = n_features if feature_pre_filter else 0
    var filter_cnt = 0
    if feature_pre_filter:
        filter_cnt = filter_count(min_data_in_leaf, n_sample, n_rows)
    var trivial = List[Int](capacity=n_flags)
    trivial.resize(n_flags, 0)
    var trivial_p = trivial.unsafe_ptr()

    def do_range(f_start: Int, f_end: Int) {imm}:
        # One set of scratch buffers per task rather than per feature: each is
        # emptied and refilled between features, keeping the capacity it
        # already grew, so a 500-feature fit does a handful of allocations per
        # worker instead of thousands. Every feature still sees exactly the
        # column it saw before, in the same order, so the edges are unchanged.
        # `bucket_counts` and `bucket_keys` are the rank-selection tables and
        # stay empty (and unallocated) on a task that never takes that path.
        # `n_sample` values at most, which is the sample when one was drawn
        # and `n_rows` when it was not. At the stock 200,000 against a
        # 1,000,000-row fit that is 1.6 MB reserved per task instead of 8 MB,
        # and the difference is first-touched memory, not just address space.
        var col = List[Float64](capacity=n_sample)
        var idxs = List[Int]()
        var ranks = List[Int]()
        var vals = List[Float64]()
        var below = List[Float64]()
        var above = List[Float64]()
        var cand = List[Float64]()
        var found = List[Int]()
        var edge_buf = List[Float64]()
        var bucket_counts = List[Int]()
        var bucket_keys = List[UInt64]()
        var dist_table = List[UInt64]()
        var dist_slots = List[Int]()
        var dist_hits = List[Int]()
        var dist_vals = List[Float64]()
        var level_counts = List[Int]()
        # CatBoost border-selection scratch. Empty and unallocated on the
        # default arm, which never touches these three.
        var cb_levels = List[Float64]()
        var cb_cum = List[Float64]()
        var cb_heap = List[_CbBin]()
        # Per-bin populations, refilled per feature and only when the prefilter
        # asked for them.
        var cnt_in_bin = List[Int]()
        for f in range(f_start, f_end):
            # Categorical columns never enter quantile binning.
            if spec.is_cat(f):
                counts_p.unsafe_store(f, 0)
                if feature_pre_filter:
                    # mojotrees's categorical layout is LightGBM's: bin 0 is
                    # the unknown/missing dummy and the i-th kept code is bin
                    # i + 1, so the counts line up bin for bin with the vector
                    # `NeedFilter` was written against.
                    var nb = spec.n_categories(f) + 1
                    cnt_in_bin.clear()
                    cnt_in_bin.resize(nb, 0)
                    var cbase = f * n_rows
                    if sampled:
                        for i in range(n_sample):
                            var r = sample_p.unsafe_load(i)
                            var b = spec.bin_of(
                                f, feat_p.unsafe_load(cbase + r)
                            )
                            if b >= 0 and b < nb:
                                cnt_in_bin[b] += 1
                    else:
                        for r in range(n_rows):
                            var b = spec.bin_of(
                                f, feat_p.unsafe_load(cbase + r)
                            )
                            if b >= 0 and b < nb:
                                cnt_in_bin[b] += 1
                    trivial_p.unsafe_store(
                        f,
                        1 if need_filter(
                            cnt_in_bin, n_sample, filter_cnt, True
                        ) else 0,
                    )
                continue
            # NaN is dropped before any ordering, so it never takes part in a
            # quantile comparison, and a column that has any gives up one bin
            # to hold its missing values.
            col.clear()
            var base = f * n_rows
            if sampled:
                # The same rows for every feature, in ascending order, so a
                # sampled fit differs from a full one only in which values it
                # sees and never in the order it sees them.
                for i in range(n_sample):
                    var v = feat_p.unsafe_load(
                        base + sample_p.unsafe_load(i)
                    )
                    if not isnan(v):
                        col.append(v)
            else:
                for r in range(n_rows):
                    var v = feat_p.unsafe_load(base + r)
                    if not isnan(v):
                        col.append(v)
            var n_valid = len(col)
            # Missingness is decided from what was read, which is the whole
            # column unless a sample was drawn. `n_sample == n_rows` when it
            # was not, so this is the test it has always been.
            var reserve = use_missing and n_valid < n_sample
            var n_ordinary = max_bins - 1 if reserve else max_bins

            below.clear()
            above.clear()
            if border_type != BORDER_QUANTILE:
                # CatBoost's border selection. `n_ordinary` BINS is
                # `n_ordinary - 1` BORDERS, and borders are what CatBoost's
                # `border_count` counts; see the section header above.
                catboost_borders(
                    col,
                    n_valid,
                    n_ordinary - 1,
                    border_type,
                    cb_levels,
                    cb_cum,
                    cb_heap,
                    edge_buf,
                )
            elif collect_distinct(
                col,
                n_valid,
                n_ordinary,
                dist_table,
                dist_slots,
                dist_hits,
                dist_vals,
                level_counts,
            ):
                # Few enough levels that every one of them can have its own
                # bin, so cut between each adjacent pair and let the budget
                # go unspent. Quantile boundaries are ranks, and a rank
                # cannot find a level that is too rare for any boundary to
                # land in: one row in a million gets no boundary of its own,
                # so it used to be swallowed by its neighbour however far
                # away that neighbour was. This is LightGBM's rule for the
                # same case.
                if min_data_in_bin <= 1:
                    for j in range(len(dist_vals) - 1):
                        below.append(dist_vals[j])
                        above.append(dist_vals[j + 1])
                else:
                    # LightGBM's `GreedyFindBin` in the same branch: walk the
                    # levels accumulating their counts and cut only once the
                    # accumulator has reached the minimum, resetting it at
                    # each cut. Adjacent levels merge until they hold enough
                    # rows between them; the final bin keeps whatever is left,
                    # because nothing merges backwards. `level_counts` came
                    # out of the scan above, not out of a second pass.
                    var acc = 0
                    for j in range(len(dist_vals) - 1):
                        acc += level_counts[j]
                        if acc >= min_data_in_bin:
                            below.append(dist_vals[j])
                            above.append(dist_vals[j + 1])
                            acc = 0
            else:
                # LightGBM's other branch shrinks the budget before it cuts:
                # `max_bin = max(1, min(max_bin, total_cnt / min_data_in_bin))`.
                # At the default of 1 the cap cannot bind here -- this branch
                # is reached only when the column has more distinct values
                # than ordinary bins, so `n_valid > n_ordinary` -- and the
                # guard keeps the default on the same instructions anyway.
                var n_ord = n_ordinary
                if min_data_in_bin > 1:
                    var cap = n_valid // min_data_in_bin
                    if cap < 1:
                        cap = 1
                    if cap < n_ord:
                        n_ord = cap
                quantile_boundary_indices(n_valid, n_ord, idxs)
                if n_valid >= select_min_rows:
                    # The ranks the boundaries ask about, ascending and
                    # unique. `idxs` is non-decreasing, so appending `idx - 1`
                    # then `idx` while each is past the last kept rank
                    # produces exactly that.
                    ranks.clear()
                    for j in range(len(idxs)):
                        var idx = idxs[j]
                        if len(ranks) == 0 or ranks[len(ranks) - 1] < idx - 1:
                            ranks.append(idx - 1)
                        if ranks[len(ranks) - 1] < idx:
                            ranks.append(idx)
                    vals.clear()
                    vals.resize(len(ranks), 0.0)
                    resolve_ranks(
                        col,
                        0,
                        ranks,
                        0,
                        len(ranks),
                        bucket_counts,
                        bucket_keys,
                        vals,
                        0,
                    )
                    # `idxs` and `ranks` both ascend, so one cursor walks them
                    # together instead of searching per boundary.
                    var rp = 0
                    var tied = False
                    for j in range(len(idxs)):
                        var idx = idxs[j]
                        while ranks[rp] < idx - 1:
                            rp += 1
                        below.append(vals[rp])
                        while ranks[rp] < idx:
                            rp += 1
                        above.append(vals[rp])
                        if vals[rp] <= below[j]:
                            tied = True
                    # A boundary inside a run of equal values needs the value
                    # the run ends at, which the rank it was handed does not
                    # know. Only a column that has such a boundary pays for
                    # the extra pass; one of distinct values already has its
                    # answer.
                    if tied:
                        resolve_above_unsorted(col, below, above, cand, found)
                else:
                    sort(col)
                    for j in range(len(idxs)):
                        var idx = idxs[j]
                        var w = col[idx - 1]
                        below.append(w)
                        var e = first_above_sorted(col, idx, n_valid, w)
                        above.append(col[e] if e < n_valid else w)
            if border_type == BORDER_QUANTILE:
                emit_quantile_edges(below, above, edge_buf)

            var out = scratch_p.unsafe_offset(f * max_edges)
            var n_out = len(edge_buf)
            for i in range(n_out):
                out.unsafe_store(i, edge_buf[i])
            counts_p.unsafe_store(f, n_out)
            # k edges give ordinary bins 0..k, so the missing bin is k + 1.
            missing_p.unsafe_store(f, n_out + 1 if reserve else -1)

            if feature_pre_filter:
                # LightGBM builds `cnt_in_bin` from the distinct values it
                # already counted; the edges are the same edges either way, so
                # counting the sample through them gives the same vector. The
                # reserved missing bin goes last, which is where LightGBM puts
                # its NaN bin, and `need_filter`'s prefix loop stops one short
                # of the last bin, so a missing-heavy column is not rescued by
                # its own missing rows.
                var nb = n_out + 2 if reserve else n_out + 1
                cnt_in_bin.clear()
                cnt_in_bin.resize(nb, 0)
                for i in range(n_valid):
                    var v = col[i]
                    # The bin `bin_value` lands on: the number of edges
                    # strictly below `v`.
                    var lo = 0
                    var hi = n_out
                    while lo < hi:
                        var mid = (lo + hi) // 2
                        if v <= edge_buf[mid]:
                            hi = mid
                        else:
                            lo = mid + 1
                    cnt_in_bin[lo] += 1
                if reserve:
                    cnt_in_bin[nb - 1] = n_sample - n_valid
                trivial_p.unsafe_store(
                    f,
                    1 if need_filter(
                        cnt_in_bin, n_sample, filter_cnt, False
                    ) else 0,
                )

    # A feature costs a copy of its column plus resolving order statistics in
    # it, so the estimate carries a comparison count rather than the row count
    # alone. It stays the sort's estimate: it decides only how the work is
    # spread, the selected path is the cheaper of the two, and keeping one
    # number keeps the serial/parallel crossover where it was measured.
    dispatch_feature_ranges(
        do_range, n_features, n_features * n_rows * (1 + _log2_ceil(n_rows))
    )

    var edges = List[Float64]()
    var offsets = List[Int](capacity=n_features + 1)
    offsets.append(0)
    for f in range(n_features):
        var base = f * max_edges
        for i in range(counts[f]):
            edges.append(scratch[base + i])
        offsets.append(len(edges))

    # `used_features` (src/io/dataset.cpp): the ascending ids that survived,
    # built serially in feature order from per-feature flags each task wrote
    # only into its own slots, so the list is the same at every worker count.
    var usable = List[Int]()
    if feature_pre_filter:
        for f in range(n_features):
            if trivial[f] == 0:
                usable.append(f)
        if len(usable) == 0:
            raise Error(
                "feature_pre_filter dropped every feature: none of them has a"
                " bin boundary leaving at least ",
                filter_cnt,
                " rows on both sides (min_data_in_leaf=",
                min_data_in_leaf,
                " scaled to a ",
                n_sample,
                "-row sample of ",
                n_rows,
                " rows). Lower min_data_in_leaf or min_data_in_bin, or turn"
                " the filter off. LightGBM warns and continues here; there is"
                " nothing to warn to and no tree to grow",
            )
    else:
        usable = all_features(n_features)
    return BinMapper(
        edges^, offsets^, n_features, max_bins, cats^, missing_bin^, usable^
    )


def map_forced_splits(
    mapper: BinMapper, forced: ForcedSplits
) raises -> ForcedSplits:
    """Turn a parsed forced-split tree's raw thresholds into bin ids.

    This is the one place a raw feature value and a fitted binning meet for
    forced splits, and it is why they live here rather than in
    tree_parameters_extra.mojo: the grower is handed a `BinnedMatrix`, which
    carries bins and no edges, so the mapping has to happen where the mapper
    is. The result is the same tree with `bins` filled, which
    `ExtraTreeParams.check_scalars` accepts and `tree.grow_tree` applies.

    Semantics. A forced node sends rows with `value <= threshold` left, and a
    tree node sends rows with `bin <= threshold_bin` left, so the bin is
    `mapper.bin_value(feature, threshold)`. That is exact at a bin boundary
    and snapped otherwise: a threshold strictly inside a bin's range sends
    that whole bin left, because a binned matrix cannot separate two values
    that share a bin. LightGBM snaps a forced threshold the same way, and the
    same approximation is what `max_bin` already imposes on every learned
    split.

    Two refusals, both because the structure cannot express the thing asked
    for rather than because it was not implemented:

    - a categorical feature, whose split is a set of categories and not an
      ordinal threshold (`parse_forced_splits` already refuses LightGBM's
      `cat_threshold` key for the same reason);
    - a `NaN` threshold, which would bin into the feature's missing bin and
      make the forced node route by missingness rather than by value.

    An empty tree maps to an empty tree, so a caller can apply this
    unconditionally.
    """
    if forced.is_empty():
        return ForcedSplits.none()
    forced.check_features(mapper.n_features)
    var bins = List[Int](capacity=forced.n_nodes())
    for i in range(forced.n_nodes()):
        var f = forced.nodes[i].feature
        if mapper.cats.is_cat(f):
            raise Error(
                "forced splits: feature ",
                f,
                " is categorical, and a categorical split is a set of"
                " categories rather than an ordinal threshold; forced"
                " categorical splits are not supported",
            )
        var v = forced.nodes[i].threshold
        if isnan(v):
            raise Error(
                "forced splits: feature ",
                f,
                " has a NaN threshold, which would route the node by"
                " missingness instead of by value",
            )
        bins.append(mapper.bin_value(f, v))
    return ForcedSplits(forced.nodes.copy(), bins^)


def bin_equal_width[
    features_origin: ImmOrigin, //
](
    features: Span[Float64, features_origin],
    n_rows: Int,
    n_features: Int,
    n_bins: Int,
) raises -> BinnedMatrix:
    """Bin a column-major feature matrix (`features[f * n_rows + r]`) into
    equal-width bins per feature. This mapper-free path has no missing-value
    support: it reserves no bin and sends `NaN` to bin 0."""
    if n_bins < 2 or n_bins > 256:
        raise Error("n_bins must be in [2, 256]")
    if len(features) != n_rows * n_features:
        raise Error("features length must equal n_rows * n_features")
    # The range pass reads row 0 before it loops, so a column with no rows has
    # to be rejected here rather than read: this path now writes through a raw
    # pointer, where an out-of-range read is silent rather than caught.
    # `fit_bins` rejects the same shape. A matrix with no features at all is
    # still fine, because nothing reads it.
    if n_features > 0 and n_rows < 1:
        raise Error("n_rows must be positive")

    # Sized up front and written by index rather than appended, which is what
    # lets the features run in parallel: each one owns the
    # `[f * n_rows, (f + 1) * n_rows)` slice of the output and nothing else.
    # Its two passes (the range, then the assignment) stay in ascending row
    # order, so every bin is what the serial loop produced.
    var bins = List[UInt8](capacity=n_rows * n_features)
    bins.resize(n_rows * n_features, 0)
    var bins_p = bins.unsafe_ptr()
    var feat_p = features.unsafe_ptr()

    def do_feature(f: Int) {imm}:
        var col = f * n_rows
        var lo = feat_p.unsafe_load(col)
        var hi = lo
        for r in range(1, n_rows):
            var v = feat_p.unsafe_load(col + r)
            if v < lo:
                lo = v
            if v > hi:
                hi = v
        var width = (hi - lo) / Float64(n_bins)
        for r in range(n_rows):
            var b: Int
            var v = feat_p.unsafe_load(col + r)
            if width <= 0.0 or isnan(v):
                b = 0
            else:
                b = Int((v - lo) / width)
                if b >= n_bins:
                    b = n_bins - 1
                if b < 0:
                    b = 0
            bins_p.unsafe_store(col + r, UInt8(b))

    # Two passes over each column, both of them a load and a compare or two.
    dispatch_features(do_feature, n_features, 2 * n_features * n_rows)
    return BinnedMatrix(bins^, n_rows, n_features, n_bins)
