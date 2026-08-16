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

from std.math import isnan
from std.memory import bitcast

from .categorical import CategoricalSpec, fit_categorical_spec
from .parallel import (
    _env_int,
    dispatch_feature_ranges,
    dispatch_feature_rows,
    dispatch_features,
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


struct BinnedMatrix(Copyable, Movable):
    """Column-major binned feature matrix.

    Bin for (row r, feature f) is stored at `bins[f * n_rows + r]`. `cats`
    records which features are categorical, so split finding knows to search
    category partitions rather than ordinal thresholds; an empty spec means
    every feature is numerical. `missing_bin[f]` is the bin reserved for
    missing values of feature f, or -1 when that feature reserves none.
    """

    var bins: List[UInt8]
    var n_rows: Int
    var n_features: Int
    var n_bins: Int
    var cats: CategoricalSpec
    var missing_bin: List[Int]

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

    def __init__(
        out self,
        var bins: List[UInt8],
        n_rows: Int,
        n_features: Int,
        n_bins: Int,
        var cats: CategoricalSpec,
        var missing_bin: List[Int] = [],
    ):
        """A `missing_bin` table of the wrong length (the empty default
        included) means no feature reserves a missing bin."""
        self.bins = bins^
        self.n_rows = n_rows
        self.n_features = n_features
        self.n_bins = n_bins
        self.cats = cats^
        self.missing_bin = _sized_missing_bins(missing_bin^, n_features)

    def bin_at(self, row: Int, feature: Int) -> Int:
        return Int(self.bins[feature * self.n_rows + row])

    def is_missing(self, row: Int, feature: Int) -> Bool:
        """Whether (row, feature) holds a missing value."""
        return self.bin_at(row, feature) == self.missing_bin[feature]


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
    """

    var edges: List[Float64]
    var edge_offsets: List[Int]
    var n_features: Int
    var n_bins: Int
    var cats: CategoricalSpec
    var missing_bin: List[Int]

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

    def __init__(
        out self,
        var edges: List[Float64],
        var edge_offsets: List[Int],
        n_features: Int,
        n_bins: Int,
        var cats: CategoricalSpec,
        var missing_bin: List[Int] = [],
    ):
        """A `missing_bin` table of the wrong length (the empty default
        included) means no feature reserves a missing bin."""
        self.edges = edges^
        self.edge_offsets = edge_offsets^
        self.n_features = n_features
        self.n_bins = n_bins
        self.cats = cats^
        self.missing_bin = _sized_missing_bins(missing_bin^, n_features)

    def has_missing(self) -> Bool:
        """Whether any feature reserves a missing bin."""
        for f in range(self.n_features):
            if self.missing_bin[f] >= 0:
                return True
        return False

    def matches(self, other: BinMapper) -> Bool:
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
        self, features: Span[Float64, features_origin], n_rows: Int
    ) raises -> BinnedMatrix:
        """Bin a column-major feature matrix (`features[f * n_rows + r]`).

        `features` is a borrowed view, so a caller holding the matrix
        somewhere other than a Mojo `List` -- the Python bindings hold
        NumPy's own buffer -- bins it in place instead of copying it first.
        A `List` converts implicitly, so passing one still works."""
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
        return BinnedMatrix(
            bins^,
            n_rows,
            n_features,
            self.n_bins,
            self.cats.copy(),
            self.missing_bin.copy(),
        )

    def bin_row(self, row: List[Float64]) raises -> List[Int]:
        """Bin one example (length n_features) for prediction."""
        if len(row) != self.n_features:
            raise Error("row length must equal n_features")
        var out = List[Int](capacity=self.n_features)
        for f in range(self.n_features):
            out.append(self.bin_value(f, row[f]))
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
    """
    if max_bins < 2 or max_bins > MAX_BINS:
        raise Error("max_bins must be in [2, ", MAX_BINS, "]")
    if n_rows < 1:
        raise Error("n_rows must be positive")
    if len(features) != n_rows * n_features:
        raise Error("features length must equal n_rows * n_features")
    if min_data_in_bin < 1:
        raise Error("min_data_in_bin must be positive")
    if bin_construct_sample_cnt < 0:
        raise Error("bin_construct_sample_cnt must be non-negative")

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
        for f in range(f_start, f_end):
            # Categorical columns never enter quantile binning.
            if spec.is_cat(f):
                counts_p.unsafe_store(f, 0)
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
            if collect_distinct(
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
            emit_quantile_edges(below, above, edge_buf)

            var out = scratch_p.unsafe_offset(f * max_edges)
            var n_out = len(edge_buf)
            for i in range(n_out):
                out.unsafe_store(i, edge_buf[i])
            counts_p.unsafe_store(f, n_out)
            # k edges give ordinary bins 0..k, so the missing bin is k + 1.
            missing_p.unsafe_store(f, n_out + 1 if reserve else -1)

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
    return BinMapper(
        edges^, offsets^, n_features, max_bins, cats^, missing_bin^
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
