"""Categorical primitives for the sparse GPU path.

Two things live here, both of them device-side: they read a device-resident
sparse matrix and a category table and hand back statistics or a routing
mask, and neither changes how a tree is grown, how a model is stored, or how
a prediction is made.

1. **Device-side category-set membership**, through a pooled bitset rather
   than through four kernel arguments, so a grower can hold every live node's
   category set on the device at once. `apply_categorical_split_pooled`
   routes a split from the pool and then hands the partition and the range
   bookkeeping back to `GpuSparseHistogramBuilder.finish_split`, so the
   pooled and the four-argument forms share one split implementation.
2. **Device-side category statistics** for one categorical feature at one
   node, straight from the compressed column, with the implicit zeros folded
   in by the same subtraction `gpu_sparse.mojo` uses for the default bin.
   `GpuCategoryStats` owns the one output buffer that needs and drives the
   builder's own node-total reduction, so the two accumulations quantize
   identically.

Where the semantics live
------------------------
Not here. `sparse.default_category_bin`, `sparse.absent_is_unknown`, and
`sparse.check_sparse_categorical_semantics` state what an absent entry of a
categorical column means, and they live in `sparse.mojo` because they are
facts about that representation, not about a device: a CPU caller has to be
able to ask them without importing a GPU module, and
`SparseBinnedMatrix.validate` enforces them on every matrix that enters
training on either backend. They are imported below and re-exported from
here, so a device-side caller reads one name in one place.

The short version, because the rest of this module depends on it: an absent
entry of a categorical feature is the value 0.0, which is *category code 0*,
so it takes category 0's bin when the fitted table kept that code and
`UNKNOWN_BIN` when it did not -- and in the second case every absent row
routes right at every categorical node of the feature, along with the
missing, unseen, and dropped ones. `absent_is_unknown` is how a caller finds
out which of the two a column got.

What is *not* here: the split search. `gpu_split_search.mojo` already runs
LightGBM's one-vs-rest and sorted category searches on the device, over a
histogram, and it neither knows nor cares whether that histogram came from a
dense matrix or a compressed one. Nothing in this module duplicates it.

Bin 0 must never enter a category set
-------------------------------------
The device routing rule (`gpu_active_rows._row_goes_left`, mirrored by
`gpu_sparse._sparse_side_entries_kernel`) tests the bitset bit for a row's
bin with no special case for bin 0: a set with bit 0 raised would send every
missing, unseen, and dropped row *left*, silently reversing the documented
default. The searches never produce such a set, but a set can also arrive
from a caller, a deserialized model, or a binding, so `cat_bitset_from_codes`
refuses to build one and `check_cat_bitset` refuses to accept one.
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, global_idx, thread_idx
from std.math import round
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from .categorical import (
    CAT_BITSET_WORDS,
    CAT_MAX_BINS,
    UNKNOWN_BIN,
    CatBitset,
    CategoricalSpec,
    cat_add,
    cat_contains,
    cat_empty,
)
from .gpu_sparse import (
    SIDE_LEFT,
    SIDE_RIGHT,
    TOT_COUNT,
    TOT_GRAD,
    TOT_HESS,
    GpuSparseHistogramBuilder,
)
from .gpu_tiling import DeviceCaps, derive_block_threads
from .histogram_sparse import SparseNodeEntries
from .sparse import (
    SparseBinnedMatrix,
    absent_is_unknown,
    check_sparse_categorical_semantics,
    default_category_bin,
)

comptime CAT_STAT_BINS = CAT_MAX_BINS


# --- Semantics ------------------------------------------------------------


def default_category_bin(cats: CategoricalSpec, feature: Int) raises -> Int:
    """The bin an absent entry of a categorical feature falls in.

    Identical to `sparse.default_bins(mapper)[feature]` for a categorical
    column, computed from the category table alone so a caller holding a
    `SparseBinnedMatrix` and no mapper can check the matrix against it.
    """
    if not cats.is_cat(feature):
        raise Error("feature is not categorical")
    return cats.bin_of(feature, 0.0)


def absent_is_unknown(cats: CategoricalSpec, feature: Int) raises -> Bool:
    """Whether absent entries of `feature` land in the unknown bin.

    True when category code 0 is not in the fitted table, in which case every
    row without a stored entry routes right at every categorical node of this
    feature, together with the missing and unseen rows. See the module
    docstring: this is a modelling fact, not a device detail.
    """
    return default_category_bin(cats, feature) == UNKNOWN_BIN


def check_sparse_categorical_semantics(
    data: SparseBinnedMatrix
) raises -> List[Int]:
    """Check a binned sparse matrix against the categorical rules, and report
    which categorical features send their absent rows to the unknown bin.

    Three things must hold for every categorical feature, and all three are
    properties a malformed producer could break without any individual number
    looking wrong:

    - the column's `default_bin` must be the bin of category code 0, because
      an absent entry is the value 0.0 and nothing else;
    - the column must reserve no missing bin: a categorical feature routes
      missing values to `UNKNOWN_BIN` by construction, and a reserved bin
      would give it a second, contradictory missing route;
    - every stored bin must be inside `[0, n_categories]`, since a bin past
      the table indexes a category that was never fitted.

    Returns the ascending feature ids for which `absent_is_unknown` holds.
    """
    var flagged = List[Int]()
    for f in range(data.n_features):
        if not data.cats.is_cat(f):
            continue
        var expected = default_category_bin(data.cats, f)
        if Int(data.default_bin[f]) != expected:
            raise Error(
                "categorical column's default bin is not the bin of"
                " category code 0"
            )
        if data.missing_bin[f] >= 0:
            raise Error(
                "categorical feature must not reserve a missing bin"
            )
        var n_cat = data.cats.n_categories(f)
        for i in range(data.col_offsets[f], data.col_offsets[f + 1]):
            if Int(data.bin[i]) > n_cat:
                raise Error("stored bin is past the fitted category table")
        if expected == UNKNOWN_BIN:
            flagged.append(f)
    return flagged^


# --- Category sets --------------------------------------------------------


def check_cat_bitset(bitset: CatBitset, n_categories: Int) raises:
    """Reject a category set that cannot mean what it says.

    Bit 0 must be clear (see the module docstring: a raised bit 0 reverses
    the routing of every missing, unseen, and dropped row) and no bit past
    the fitted table may be raised, since it would name a category that does
    not exist.
    """
    if n_categories < 0 or n_categories + 1 > CAT_MAX_BINS:
        raise Error("category count out of range for a 256-bit set")
    if (bitset[0] & UInt64(1)) != 0:
        raise Error(
            "category set contains the unknown bin, which must always route"
            " right"
        )
    for b in range(n_categories + 1, CAT_MAX_BINS):
        if cat_contains(bitset, b):
            raise Error("category set contains a bin past the fitted table")


def cat_bitset_from_codes(
    cats: CategoricalSpec, feature: Int, codes: List[Int]
) raises -> CatBitset:
    """The bin-indexed category set for a list of raw category codes.

    Raises on a code the fitted table does not hold, rather than mapping it
    to `UNKNOWN_BIN`: `bin_of` returns bin 0 for an unknown code, and adding
    bin 0 to the set is exactly the mistake `check_cat_bitset` exists to
    catch. A caller that wants unknown codes ignored has to drop them itself
    and say so.
    """
    if not cats.is_cat(feature):
        raise Error("feature is not categorical")
    var out = cat_empty()
    for i in range(len(codes)):
        var bin = cats.bin_of(feature, Float64(codes[i]))
        if bin == UNKNOWN_BIN:
            raise Error(
                "category code is not in the fitted table; it cannot be"
                " placed in a split set"
            )
        cat_add(out, bin)
    return out


def codes_from_cat_bitset(
    cats: CategoricalSpec, feature: Int, bitset: CatBitset
) raises -> List[Int]:
    """The ascending raw category codes a set selects.

    The inverse of `cat_bitset_from_codes`, and the form a binding or a model
    dump wants: bin ids are an artifact of the fitted table, codes are what
    the caller gave us.
    """
    if not cats.is_cat(feature):
        raise Error("feature is not categorical")
    var n_cat = cats.n_categories(feature)
    check_cat_bitset(bitset, n_cat)
    var begin = cats.offsets[feature]
    var out = List[Int]()
    for b in range(1, n_cat + 1):
        if cat_contains(bitset, b):
            out.append(cats.codes[begin + b - 1])
    return out^


struct CatSetPool(Movable):
    """A device-resident pool of 256-bit category sets, keyed by slot.

    `gpu_predict.mojo` already concatenates a *model's* category sets into
    one device pool for inference. This is the training-time twin: the sets a
    grower is about to route by, uploaded once per batch of splits instead of
    riding along as four `UInt64` kernel arguments per split. The layout is
    the same one `gpu_predict` uses -- `CAT_BITSET_WORDS` words per set, and
    a slot's absolute word offset is what a kernel indexes with -- so the two
    pools are interchangeable in a kernel that takes an offset.

    The pool is built on the host, where the searches produce their sets, and
    uploaded whole. Sets are never edited in place on the device: a set is a
    decision, and a decision that changed under a running kernel would be
    unreproducible.
    """

    var ctx: DeviceContext
    var words: List[UInt64]
    var pool_dev: DeviceBuffer[DType.uint64]
    var capacity: Int
    var uploaded: Int

    def __init__(out self, ctx: DeviceContext, capacity: Int) raises:
        """Room for `capacity` sets, allocated once."""
        if capacity < 1:
            raise Error("category set pool needs positive capacity")
        self.ctx = ctx
        self.capacity = capacity
        self.uploaded = 0
        self.words = List[UInt64]()
        self.pool_dev = ctx.enqueue_create_buffer[DType.uint64](
            capacity * CAT_BITSET_WORDS
        )

    def n_sets(self) -> Int:
        return len(self.words) // CAT_BITSET_WORDS

    def clear(mut self):
        """Drop every staged set. The device buffer keeps its allocation."""
        self.words.clear()
        self.uploaded = 0

    def push(mut self, bitset: CatBitset, n_categories: Int) raises -> Int:
        """Stage one set and return its word offset into the pool.

        The offset, not the slot index, because that is what a kernel
        indexes with and what `gpu_predict`'s `NODE_CAT` already stores.
        """
        check_cat_bitset(bitset, n_categories)
        if self.n_sets() >= self.capacity:
            raise Error("category set pool is full")
        var offset = len(self.words)
        for w in range(CAT_BITSET_WORDS):
            self.words.append(bitset[w])
        return offset

    def get(self, offset: Int) raises -> CatBitset:
        """The staged set at a word offset, host side."""
        if offset < 0 or offset + CAT_BITSET_WORDS > len(self.words):
            raise Error("category set offset out of range")
        var out = CatBitset(0)
        for w in range(CAT_BITSET_WORDS):
            out[w] = self.words[offset + w]
        return out

    def contains(self, offset: Int, bin: Int) raises -> Bool:
        """Host mirror of the device membership test, including its lack of
        a special case for bin 0: what the kernel does is what this does, so
        a set that would misroute is visible here too. `check_cat_bitset`
        on the way in is what keeps that from mattering."""
        if bin < 0 or bin >= CAT_MAX_BINS:
            return False
        if offset < 0 or offset + CAT_BITSET_WORDS > len(self.words):
            raise Error("category set offset out of range")
        return (
            (self.words[offset + (bin >> 6)] >> UInt64(bin & 63)) & 1
        ) != 0

    def upload(mut self) raises:
        """Copy every staged set to the device and wait.

        Synchronizes, because the copy reads a host `List` this struct owns
        and the next `push` would move it.
        """
        if len(self.words) == 0:
            self.uploaded = 0
            return
        self.ctx.enqueue_copy(
            dst_buf=self.pool_dev, src_ptr=self.words.unsafe_ptr()
        )
        self.ctx.synchronize()
        self.uploaded = len(self.words)


# --- Kernels --------------------------------------------------------------


def _cat_pool_side_kernel(
    order: MutPointer[Int32, MutAnyOrigin],
    entry_row: MutPointer[Int32, MutAnyOrigin],
    entry_bin: MutPointer[UInt8, MutAnyOrigin],
    ranges: MutPointer[Int32, MutAnyOrigin],
    pool: MutPointer[UInt64, MutAnyOrigin],
    side: MutPointer[UInt8, MutAnyOrigin],
    node: Int32,
    n_features: Int32,
    feature: Int32,
    set_offset: Int32,
):
    """`gpu_sparse._sparse_side_entries_kernel`'s categorical arm, with the
    set read from the pool instead of from four kernel arguments.

    Same rule and same result, so the two are interchangeable for a
    categorical split; this one exists so a grower holding many live nodes'
    sets on the device does not have to carry each set through the launch
    arity. Rows without a stored entry never reach here and keep the default
    side, which for a categorical feature is the side of category code 0's
    bin (or of the unknown bin when code 0 was not kept).
    """
    var f = Int(feature)
    var rbase = 2 * (Int(node) * Int(n_features) + f)
    var lo = Int(ranges[unsafe_offset=rbase][0])
    var hi = Int(ranges[unsafe_offset = rbase + 1][0])
    var i = lo + global_idx.x
    if i >= hi:
        return

    var e = Int(order[unsafe_offset=i][0])
    var r = Int(entry_row[unsafe_offset=e][0])
    var bin = Int(entry_bin[unsafe_offset=e])
    var word = pool[unsafe_offset = Int(set_offset) + (bin >> 6)][0]
    var goes_left = ((word >> UInt64(bin & 63)) & 1) != 0
    side[unsafe_offset=r] = SIDE_LEFT if goes_left else SIDE_RIGHT


def _category_stats_kernel(
    order: MutPointer[Int32, MutAnyOrigin],
    entry_row: MutPointer[Int32, MutAnyOrigin],
    entry_bin: MutPointer[UInt8, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    ranges: MutPointer[Int32, MutAnyOrigin],
    out_stats: MutPointer[Int32, MutAnyOrigin],
    node: Int32,
    n_features: Int32,
    feature: Int32,
    n_bins: Int32,
    entries_per_tile: Int32,
    g_scale: Float32,
    h_scale: Float32,
):
    """Per-category fixed-point statistics for one categorical feature.

    `out_stats` is `[grad | hess | count]`, `n_bins` cells each, indexed by
    the feature's own bin ids: bin 0 is the unknown bin and bins
    `1 ..= n_categories` are the kept categories. `grid.x` tiles the
    feature's entry window; the shared partial and the integer atomics are
    the same as in the histogram kernel, so this is exactly one column of
    that histogram and the two agree bit for bit.

    The implicit zeros are not here. `_category_default_fill_kernel` folds
    them into the default bin afterwards, which for a categorical feature is
    category code 0's bin, or the unknown bin when code 0 was not kept.
    """
    var f = Int(feature)
    var tid = thread_idx.x
    var nb = Int(n_bins)
    var rbase = 2 * (Int(node) * Int(n_features) + f)
    var lo = Int(ranges[unsafe_offset=rbase][0])
    var hi = Int(ranges[unsafe_offset = rbase + 1][0])

    var sg = stack_allocation[
        CAT_STAT_BINS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var sh = stack_allocation[
        CAT_STAT_BINS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var sc = stack_allocation[
        CAT_STAT_BINS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()

    var b = tid
    while b < nb:
        sg[unsafe_offset=b] = 0
        sh[unsafe_offset=b] = 0
        sc[unsafe_offset=b] = 0
        b += block_dim.x
    barrier()

    var tile_begin = lo + block_idx.x * Int(entries_per_tile)
    var tile_end = tile_begin + Int(entries_per_tile)
    if tile_end > hi:
        tile_end = hi

    var i = tile_begin + tid
    while i < tile_end:
        var e = Int(order[unsafe_offset=i][0])
        var r = Int(entry_row[unsafe_offset=e][0])
        var bin = Int(entry_bin[unsafe_offset=e])
        _ = Atomic.fetch_add(
            sg.unsafe_offset(bin),
            Int32(round(grad[unsafe_offset=r][0] * g_scale)),
        )
        _ = Atomic.fetch_add(
            sh.unsafe_offset(bin),
            Int32(round(hess[unsafe_offset=r][0] * h_scale)),
        )
        _ = Atomic.fetch_add(sc.unsafe_offset(bin), Int32(1))
        i += block_dim.x
    barrier()

    b = tid
    while b < nb:
        if sc[unsafe_offset=b][0] != 0:
            _ = Atomic.fetch_add(
                out_stats.unsafe_offset(b), sg[unsafe_offset=b][0]
            )
            _ = Atomic.fetch_add(
                out_stats.unsafe_offset(nb + b), sh[unsafe_offset=b][0]
            )
            _ = Atomic.fetch_add(
                out_stats.unsafe_offset(2 * nb + b), sc[unsafe_offset=b][0]
            )
        b += block_dim.x


def _category_default_fill_kernel(
    out_stats: MutPointer[Int32, MutAnyOrigin],
    totals: MutPointer[Int32, MutAnyOrigin],
    n_bins: Int32,
    default_bin: Int32,
):
    """Fold the node's implicit zeros into the categorical default bin.

    One thread. The leftover after every stored entry is the rows with no
    entry for this feature, which are the rows of category code 0, so this is
    the categorical statement of the same subtraction the histogram path
    makes -- and the place where "absent means category 0" becomes an actual
    number rather than a docstring.
    """
    if thread_idx.x != 0 or block_idx.x != 0:
        return
    var nb = Int(n_bins)
    var sum_g = Int32(0)
    var sum_h = Int32(0)
    var sum_c = Int32(0)
    for b in range(nb):
        sum_g += out_stats[unsafe_offset=b][0]
        sum_h += out_stats[unsafe_offset = nb + b][0]
        sum_c += out_stats[unsafe_offset = 2 * nb + b][0]
    var d = Int(default_bin)
    out_stats[unsafe_offset=d] = (
        out_stats[unsafe_offset=d][0]
        + (totals[unsafe_offset=TOT_GRAD][0] - sum_g)
    )
    out_stats[unsafe_offset = nb + d] = (
        out_stats[unsafe_offset = nb + d][0]
        + (totals[unsafe_offset=TOT_HESS][0] - sum_h)
    )
    out_stats[unsafe_offset = 2 * nb + d] = (
        out_stats[unsafe_offset = 2 * nb + d][0]
        + (totals[unsafe_offset=TOT_COUNT][0] - sum_c)
    )


# --- Host-side statistics -------------------------------------------------


@fieldwise_init
struct CategoryStats(Copyable, Movable):
    """One categorical feature's per-bin statistics at one node.

    Indexed by bin id: 0 is the unknown bin (missing, unseen, dropped, and
    -- when category code 0 was not kept -- absent), and `1 ..= n_categories`
    are the kept categories in ascending code order. That is exactly the
    layout `categorical.find_best_categorical_split` reads, so these arrays
    drop straight into it as the `base = 0` slice of a histogram.
    """

    var grad: List[Float64]
    var hess: List[Float64]
    var count: List[Int]
    var feature: Int
    var n_categories: Int
    var absent_bin: Int
    """The bin the node's absent rows were folded into: category code 0's
    bin, or `UNKNOWN_BIN` when code 0 is not in the fitted table."""

    def total_count(self) -> Int:
        var out = 0
        for b in range(len(self.count)):
            out += self.count[b]
        return out

    def unknown_count(self) -> Int:
        """Rows that will route right at any categorical split of this
        feature, whatever set the search picks."""
        return self.count[UNKNOWN_BIN]


def category_stats_host(
    data: SparseBinnedMatrix,
    order: List[Int],
    node: SparseNodeEntries,
    rows: List[Int],
    grad: List[Float64],
    hess: List[Float64],
    feature: Int,
) raises -> CategoryStats:
    """`_category_stats_kernel` + `_category_default_fill_kernel` on the
    host, in Float64.

    The reference the device pair is compared against, and a usable primitive
    in its own right for a caller with no device. Same structure: accumulate
    the node's stored entries, then give the default bin whatever the node
    total did not account for.
    """
    if not data.cats.is_cat(feature):
        raise Error("feature is not categorical")
    if len(grad) != data.n_rows or len(hess) != data.n_rows:
        raise Error("gradient/hessian length must equal n_rows")
    var n_bins = data.n_bins
    var g = List[Float64](capacity=n_bins)
    g.resize(n_bins, 0.0)
    var h = List[Float64](capacity=n_bins)
    h.resize(n_bins, 0.0)
    var c = List[Int](capacity=n_bins)
    c.resize(n_bins, 0)

    var total_g = 0.0
    var total_h = 0.0
    for i in range(len(rows)):
        var r = rows[i]
        if r < 0 or r >= data.n_rows:
            raise Error("row index out of range")
        total_g += grad[r]
        total_h += hess[r]
    var total_c = len(rows)

    var stored_g = 0.0
    var stored_h = 0.0
    var stored_c = 0
    for i in range(node.starts[feature], node.ends[feature]):
        var e = order[i]
        if e < 0 or e >= data.nnz():
            raise Error("entry index out of range")
        var r = data.row_index[e]
        var b = Int(data.bin[e])
        g[b] += grad[r]
        h[b] += hess[r]
        c[b] += 1
        stored_g += grad[r]
        stored_h += hess[r]
        stored_c += 1

    var d = Int(data.default_bin[feature])
    g[d] += total_g - stored_g
    h[d] += total_h - stored_h
    c[d] += total_c - stored_c
    return CategoryStats(
        g^,
        h^,
        c^,
        feature,
        data.cats.n_categories(feature),
        d,
    )


def category_stats_from_fixed(
    raw: List[Int32],
    n_bins: Int,
    g_scale: Float64,
    h_scale: Float64,
    feature: Int,
    n_categories: Int,
    absent_bin: Int,
) raises -> CategoryStats:
    """Convert the device's downloaded fixed-point planes into
    `CategoryStats`, the same conversion `histogram_from_host` makes for a
    whole histogram."""
    if len(raw) != 3 * n_bins:
        raise Error("fixed-point stats must hold three planes of n_bins")
    if g_scale == 0.0 or h_scale == 0.0:
        raise Error("fixed-point scales must be nonzero")
    var g = List[Float64](capacity=n_bins)
    var h = List[Float64](capacity=n_bins)
    var c = List[Int](capacity=n_bins)
    var g_inv = 1.0 / g_scale
    var h_inv = 1.0 / h_scale
    for b in range(n_bins):
        g.append(Float64(raw[b]) * g_inv)
        h.append(Float64(raw[n_bins + b]) * h_inv)
        c.append(Int(raw[2 * n_bins + b]))
    return CategoryStats(g^, h^, c^, feature, n_categories, absent_bin)


def enqueue_category_stats[
    order_origin: MutOrigin,
    row_origin: MutOrigin,
    bin_origin: MutOrigin,
    grad_origin: MutOrigin,
    hess_origin: MutOrigin,
    range_origin: MutOrigin,
    total_origin: MutOrigin,
    out_origin: MutOrigin, //
](
    ctx: DeviceContext,
    caps: DeviceCaps,
    order: MutPointer[Int32, order_origin],
    entry_row: MutPointer[Int32, row_origin],
    entry_bin: MutPointer[UInt8, bin_origin],
    grad: MutPointer[Float32, grad_origin],
    hess: MutPointer[Float32, hess_origin],
    ranges: MutPointer[Int32, range_origin],
    totals: MutPointer[Int32, total_origin],
    out_stats: MutPointer[Int32, out_origin],
    node: Int,
    n_features: Int,
    feature: Int,
    n_bins: Int,
    default_bin: Int,
    n_entries: Int,
    g_scale: Float64,
    h_scale: Float64,
) raises:
    """Enqueue the two category-statistics kernels for one feature.

    Takes raw pointers rather than a builder so it composes with whatever
    owns the buffers: every argument is something `GpuSparseHistogramBuilder`
    already holds, and nothing here allocates. `out_stats` must be zeroed and
    `totals` filled by the caller (the builder's totals kernel does both for
    a node), which is what keeps this a primitive rather than a second
    pipeline.

    Does not transfer or synchronize.
    """
    if n_bins < 1 or n_bins > CAT_MAX_BINS:
        raise Error("category statistics need 1..256 bins")
    if default_bin < 0 or default_bin >= n_bins:
        raise Error("default bin out of range")
    var threads = derive_block_threads(caps)
    if n_entries > 0:
        var per_tile = n_entries
        var tiles = 1
        # One tile per multiprocessor at most: the window is one column of
        # one node, so oversubscribing it costs more launch overhead than the
        # parallelism buys.
        if n_entries > threads * caps.sm_count:
            tiles = caps.sm_count
            per_tile = (n_entries + tiles - 1) // tiles
        ctx.enqueue_function[_category_stats_kernel](
            order,
            entry_row,
            entry_bin,
            grad,
            hess,
            ranges,
            out_stats,
            Int32(node),
            Int32(n_features),
            Int32(feature),
            Int32(n_bins),
            Int32(per_tile),
            Float32(g_scale),
            Float32(h_scale),
            grid_dim=tiles,
            block_dim=threads,
        )
    ctx.enqueue_function[_category_default_fill_kernel](
        out_stats,
        totals,
        Int32(n_bins),
        Int32(default_bin),
        grid_dim=1,
        block_dim=threads,
    )
