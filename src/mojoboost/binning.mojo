"""Feature binning.

Maps raw feature values to small integer bins so that split finding can
operate on fixed-size histograms instead of sorted feature values.

`fit_bins` performs quantile (equal-frequency) binning, LightGBM style,
and returns a `BinMapper` whose stored edges let a trained model bin raw,
unseen feature values at prediction time. `bin_equal_width` remains as a
simple mapper-free alternative for experiments.

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
from .parallel import _env_int, dispatch_feature_ranges, dispatch_features
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

    `below[j]` and `above[j]` are the values at ranks `idx - 1` and `idx` of
    the j-th boundary `quantile_boundary_indices` reported, in the same
    order. An edge is the midpoint of the two, clamped by `_avoid_inf`.

    Two boundaries produce no edge. One whose two values are equal has no gap
    to cut (that is how a duplicate-heavy column ends up using fewer bins
    than `max_bins`), and one whose clamped midpoint fails to exceed the last
    edge kept would break the strictly-increasing invariant `bin_value` and
    `transform` search against. Repeated ranks (`n_valid < n_ordinary`) hit
    the first rule and clamped infinities the second.

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


def env_select_min_rows() -> Int:
    """`SELECT_MIN_ROWS` under `MOJOBOOST_BINNING_SELECT_MIN_ROWS`.

    Scheduling-only, in the same sense as the two variables in
    `parallel.mojo`: the two paths resolve the same order statistics, so this
    decides which one runs and nothing else. It exists so a test can assert
    that equality on data small enough to fit in a test, and so a benchmark
    can measure one path against the other without rebuilding. `0`, unset, or
    unparsable means the default.
    """
    var n = _env_int("MOJOBOOST_BINNING_SELECT_MIN_ROWS", SELECT_MIN_ROWS)
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

    def transform(
        self, features: List[Float64], n_rows: Int
    ) raises -> BinnedMatrix:
        """Bin a column-major feature matrix (`features[f * n_rows + r]`)."""
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

        def do_feature(f: Int) {imm}:
            var col = f * n_rows
            if cats.is_cat(f):
                for r in range(n_rows):
                    bins_p.unsafe_store(
                        col + r,
                        UInt8(cats.bin_of(f, feat_p.unsafe_load(col + r))),
                    )
                return
            var pbase = poff_p.unsafe_load(f)
            var half = half_p.unsafe_load(f)
            var mb = miss_p.unsafe_load(f)
            for r in range(n_rows):
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
        dispatch_features(
            do_feature,
            n_features,
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


def fit_bins(
    features: List[Float64],
    n_rows: Int,
    n_features: Int,
    max_bins: Int = 255,
    categorical_features: List[Int] = [],
    use_missing: Bool = True,
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
    nothing and bins `NaN` as 0.0, matching LightGBM's `use_missing=false`."""
    if max_bins < 2 or max_bins > MAX_BINS:
        raise Error("max_bins must be in [2, ", MAX_BINS, "]")
    if n_rows < 1:
        raise Error("n_rows must be positive")
    if len(features) != n_rows * n_features:
        raise Error("features length must equal n_rows * n_features")

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

    def do_range(f_start: Int, f_end: Int) {imm}:
        # One set of scratch buffers per task rather than per feature: each is
        # emptied and refilled between features, keeping the capacity it
        # already grew, so a 500-feature fit does a handful of allocations per
        # worker instead of thousands. Every feature still sees exactly the
        # column it saw before, in the same order, so the edges are unchanged.
        # `bucket_counts` and `bucket_keys` are the rank-selection tables and
        # stay empty (and unallocated) on a task that never takes that path.
        var col = List[Float64](capacity=n_rows)
        var idxs = List[Int]()
        var ranks = List[Int]()
        var vals = List[Float64]()
        var below = List[Float64]()
        var above = List[Float64]()
        var edge_buf = List[Float64]()
        var bucket_counts = List[Int]()
        var bucket_keys = List[UInt64]()
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
            for r in range(n_rows):
                var v = feat_p.unsafe_load(base + r)
                if not isnan(v):
                    col.append(v)
            var n_valid = len(col)
            var reserve = use_missing and n_valid < n_rows
            var n_ordinary = max_bins - 1 if reserve else max_bins

            quantile_boundary_indices(n_valid, n_ordinary, idxs)
            below.clear()
            above.clear()
            if n_valid >= select_min_rows:
                # The ranks the boundaries ask about, ascending and unique.
                # `idxs` is non-decreasing, so appending `idx - 1` then `idx`
                # while each is past the last kept rank produces exactly that.
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
                for j in range(len(idxs)):
                    var idx = idxs[j]
                    while ranks[rp] < idx - 1:
                        rp += 1
                    below.append(vals[rp])
                    while ranks[rp] < idx:
                        rp += 1
                    above.append(vals[rp])
            else:
                sort(col)
                for j in range(len(idxs)):
                    below.append(col[idxs[j] - 1])
                    above.append(col[idxs[j]])
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


def bin_equal_width(
    features: List[Float64], n_rows: Int, n_features: Int, n_bins: Int
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
