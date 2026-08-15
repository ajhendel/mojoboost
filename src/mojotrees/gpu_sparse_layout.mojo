"""Device layout and cost model for sparse GPU primitives.

Pure host arithmetic. Nothing here opens a `DeviceContext`, allocates a
buffer, or launches a kernel, so the layout decisions, the per-node cost
model, and the node bookkeeping are all reasonable about without a GPU, the
way `gpu_tiling.mojo` is reasonable about without one. `gpu_sparse.mojo` and
`gpu_categorical.mojo` consume this module; it consumes neither of them.

What a sparse dataset looks like on a device
--------------------------------------------
The device layout is the `SparseBinnedMatrix` of `sparse.mojo` with its index
widths narrowed to what the kernels index with:

    row_index    Int32[nnz]        stored entry -> row
    bin          UInt8[nnz]        stored entry -> bin
    col_offsets  Int32[n_features + 1]
    default_bin  UInt8[n_features] bin holding the implicit zero
    order        Int32[nnz]        entry permutation, grouped by node
    scratch      Int32[nnz]        scatter destination for the partition
    ranges       Int32[2 * max_nodes * n_features]  per-node entry windows
    side         UInt8[n_rows]     per-row split side scratch

`order`, `scratch`, and `ranges` are the device mirror of `SparseEntryOrder`
and `SparseNodeEntries` in `histogram_sparse.mojo`. The CPU grower keeps the
same three things in host memory and the same invariant holds on both sides:
a node's entries for one feature are a contiguous window of `order`, that
window is a sub-window of the parent's, and entries stay in ascending row
order inside it because every partition is stable.

`side` is the one buffer with no CPU counterpart. It exists because a sparse
split routes rows through the split feature's *stored entries* alone (every
row of the node takes the side of `default_bin[f]` unless it has an entry),
so the side has to be materialized per row before rows and entries can be
partitioned by it. One byte per row, allocated once per session.

Semantics carried over unchanged
--------------------------------
Every rule `sparse.mojo` fixes is a rule this layout keeps, because the
layout stores exactly the same numbers:

- an **absent** entry is the value 0.0 and bins to `default_bin[f]`;
- an **explicitly stored zero**, and any stored value that happens to bin to
  `default_bin[f]`, is kept in the structure and accumulated as a stored
  entry, which is indistinguishable in the result from having dropped it;
- a stored **NaN** is missing, bins to the feature's reserved missing bin,
  and is never a default bin, so missing rows stay distinguishable from
  absent ones.

Nothing here reinterprets zero as missing, and there is no parameter that
would (`zero_as_missing` is not implemented anywhere in mojotrees).

What this module refuses to decide
----------------------------------
Whether the sparse device path beats the dense one on a given dataset is a
measurement, not an inequality that can be derived here. `sparse_node_cost`
and `dense_node_cost` count the work each path does in units a benchmark can
weigh (bin reads, entry reads, row reads, kernel launches, host
synchronizations), and `decide_sparse` combines them *only* when the caller
hands over per-unit costs it actually measured. With no measurements the
verdict is `SPARSE_UNDECIDED`, and it stays undecided: there is no default
threshold, no density heuristic, and no automatic fallback that would pick a
path on this module's authority. See
`docs/GPU_SPARSE_CATEGORICAL_DESIGN.md` for the benchmark that has to run
before a threshold is allowed to exist.

What this module *does* decide
------------------------------
Whether a given dataset can run on the device path at all, and whether a
training path exists to run it from. `SparseGpuCapability` at the bottom of
this file is the one record that answers both, `check_sparse_gpu_histograms`
is the raising form for a caller wanting the primitives, and
`check_sparse_gpu_training` is the raising form for an explicit
`device="gpu"` request on sparse input -- which today always raises, because
no trainer drives these primitives. A capability record is how that stays
honest: nothing here reports GPU support and then runs on the CPU.
"""

from .categorical import CAT_MAX_BINS, CategoricalSpec
from .efb import FeatureBundling
from .gpu_objectives_native import DEFAULT_MAX_NODES
from .gpu_tiling import (
    DeviceCaps,
    HistogramTiling,
    derive_tiling,
    shared_bytes_for,
)
from .histogram_sparse import SparseNodeEntries
from .sparse import SparseBinnedMatrix, check_sparse_categorical_semantics

# Entry ids, row ids, offsets, and node ids all cross into the kernels as
# Int32, so every one of them is bounded by the same limit the dense GPU
# path already imposes on rows (`histogram_gpu.MAX_ROWS`).
comptime SPARSE_MAX_INDEX = Int(Int32.MAX)

# The widest histogram the GPU backend accepts, as `UInt8` bins allow.
comptime SPARSE_MAX_BINS = 256

# Bytes per element of each device buffer, so the accounting below is one
# table rather than a scatter of literals.
comptime BYTES_INDEX = 4
comptime BYTES_BIN = 1
comptime BYTES_SIDE = 1

# Default ceiling on the node ids a tree may use, `gpu_objectives_native`'s.
# The device range table is sized by it at construction, since a split must
# not allocate.


# --- Kernel block sizes and their shared-memory cost ----------------------
#
# The bounds live here rather than in `gpu_sparse.mojo` because they are what
# the support check has to reason about, and a support check that reads
# different numbers from the kernels it is clearing would be worse than no
# check at all. `gpu_sparse.mojo` defines its local names from these.

# The segmented entry partition keeps one Int32 per thread.
comptime SPARSE_SCAN_MAX_THREADS = 1024

# The two reducing kernels (node totals, default-bin completion) keep three
# Int32 planes per thread, so the same 1024 bound would ask for 12 KB per
# threadgroup. See `sparse_kernel_shared_bytes`.
comptime SPARSE_REDUCE_MAX_THREADS = 256


def sparse_kernel_shared_bytes(n_bins: Int) -> Int:
    """The largest shared-memory request any kernel in this module makes.

    Not `gpu_tiling.shared_bytes_for(n_bins)`, which is what the dense tiling
    policy uses, and which is the wrong figure here for two reasons:

    - it scales with `n_bins`, but a `stack_allocation` takes a *compile-time*
      extent, so the accumulation kernel asks for `3 * SPARSE_MAX_BINS`
      Int32 whatever `n_bins` turns out to be. At `n_bins = 16` the policy
      figure is 192 bytes and the real request is 3072;
    - it describes the accumulation kernel only, and the entry partition asks
      for more than any of them (`SPARSE_SCAN_MAX_THREADS` Int32), with a
      size that does not depend on `n_bins` at all.

    Taking the maximum at runtime rather than folding it into a `comptime`
    keeps it honest if any one of the three bounds is later changed alone.
    """
    var most = shared_bytes_for(n_bins)
    var accumulation = 3 * SPARSE_MAX_BINS * BYTES_INDEX
    if accumulation > most:
        most = accumulation
    var reduction = 3 * SPARSE_REDUCE_MAX_THREADS * BYTES_INDEX
    if reduction > most:
        most = reduction
    var partition = SPARSE_SCAN_MAX_THREADS * BYTES_INDEX
    if partition > most:
        most = partition
    return most


# --- Hard support limits --------------------------------------------------
#
# These are the combinations the device path cannot express at all, as
# opposed to the ones it merely might lose on. Every one of them raises
# rather than degrading, because silently densifying or silently dropping a
# feature is exactly the failure mode this lane exists to avoid.

comptime SPARSE_OK = 0
comptime SPARSE_TOO_MANY_BINS = 1
comptime SPARSE_TOO_MANY_ROWS = 2
comptime SPARSE_TOO_MANY_ENTRIES = 3
comptime SPARSE_TOO_MANY_FEATURES = 4
comptime SPARSE_SHARED_MEMORY = 5
comptime SPARSE_EMPTY = 6
comptime SPARSE_CATEGORIES_EXCEED_BINS = 7
comptime SPARSE_BUNDLED_CATEGORICAL = 8
comptime SPARSE_CATEGORY_SET_OVERFLOW = 9


def sparse_support_name(reason: Int) -> String:
    """Human-readable form of a support code, for error messages and for a
    handoff table that has to name the case it is talking about."""
    if reason == SPARSE_OK:
        return String("supported")
    if reason == SPARSE_TOO_MANY_BINS:
        return String("more than 256 bins")
    if reason == SPARSE_TOO_MANY_ROWS:
        return String("more rows than an Int32 index holds")
    if reason == SPARSE_TOO_MANY_ENTRIES:
        return String("more stored entries than an Int32 index holds")
    if reason == SPARSE_TOO_MANY_FEATURES:
        return String("more features than an Int32 index holds")
    if reason == SPARSE_SHARED_MEMORY:
        return String(
            "device shared memory too small for this module's widest kernel"
        )
    if reason == SPARSE_EMPTY:
        return String("empty matrix")
    if reason == SPARSE_CATEGORIES_EXCEED_BINS:
        return String("categorical feature has more categories than bins")
    if reason == SPARSE_BUNDLED_CATEGORICAL:
        return String("categorical feature inside a multi-member bundle")
    if reason == SPARSE_CATEGORY_SET_OVERFLOW:
        return String(
            "categorical feature has more categories than the 256-bit split"
            " set holds"
        )
    return String("unknown")


def sparse_support(
    caps: DeviceCaps,
    n_rows: Int,
    n_features: Int,
    n_bins: Int,
    nnz: Int,
) -> Int:
    """Whether a shape is expressible on the device path, as a code.

    Does not raise, so a caller can report the reason rather than catching.
    `check_sparse_support` is the raising form.
    """
    if n_rows < 1 or n_features < 1 or n_bins < 1:
        return SPARSE_EMPTY
    if n_bins > SPARSE_MAX_BINS:
        return SPARSE_TOO_MANY_BINS
    if n_rows > SPARSE_MAX_INDEX:
        return SPARSE_TOO_MANY_ROWS
    if n_features > SPARSE_MAX_INDEX:
        return SPARSE_TOO_MANY_FEATURES
    # `>=` rather than `>`: `col_offsets[n_features]` is nnz itself, and it
    # is an Int32 like every other offset.
    if nnz < 0 or nnz >= SPARSE_MAX_INDEX:
        return SPARSE_TOO_MANY_ENTRIES
    if sparse_kernel_shared_bytes(n_bins) > caps.max_shared_memory_per_block:
        return SPARSE_SHARED_MEMORY
    return SPARSE_OK


def check_sparse_support(
    caps: DeviceCaps,
    n_rows: Int,
    n_features: Int,
    n_bins: Int,
    nnz: Int,
) raises:
    """Raising form of `sparse_support`."""
    var reason = sparse_support(caps, n_rows, n_features, n_bins, nnz)
    if reason != SPARSE_OK:
        raise Error(
            "sparse GPU path does not support this dataset: "
            + sparse_support_name(reason)
        )


def categorical_support(
    cats: CategoricalSpec, n_features: Int, n_bins: Int
) -> Int:
    """The categorical half of the support check, as a code.

    A categorical feature needs one bin per kept category plus the reserved
    unknown bin, and the split search's category set is a 256-bit mask
    (`categorical.CAT_MAX_BINS`), so both bounds have to hold before any
    device work starts. This mirrors the check `gpu_split_search` already
    makes on the dense path; it is repeated here so a sparse caller fails at
    layout time rather than at launch time.

    Non-raising, like `sparse_support`, so the capability record below can
    report the reason rather than catch it. `check_categorical_support` is
    the raising form.
    """
    for f in range(n_features):
        if not cats.is_cat(f):
            continue
        var n_cat = cats.n_categories(f)
        if n_cat >= n_bins:
            return SPARSE_CATEGORIES_EXCEED_BINS
        if n_cat + 1 > CAT_MAX_BINS:
            return SPARSE_CATEGORY_SET_OVERFLOW
    return SPARSE_OK


def check_categorical_support(
    cats: CategoricalSpec, n_features: Int, n_bins: Int
) raises:
    """Raising form of `categorical_support`."""
    var reason = categorical_support(cats, n_features, n_bins)
    if reason != SPARSE_OK:
        raise Error(
            "sparse GPU path does not support this dataset: "
            + sparse_support_name(reason)
        )


# --- Device buffer accounting --------------------------------------------


@fieldwise_init
struct SparseDeviceLayout(Copyable, Movable):
    """Element counts and byte counts of every buffer the sparse device path
    allocates for one dataset.

    Sized by the dataset and by `max_nodes`, never by a node, so growing a
    tree allocates nothing. `bytes()` is what a memory budget compares, and
    `dense_bytes()` is what the same dataset would cost as the dense
    `bins` matrix the existing `GpuHistogramBuilder` uploads, so the two are
    directly comparable.
    """

    var n_rows: Int
    var n_features: Int
    var n_bins: Int
    var nnz: Int
    var max_nodes: Int

    def entry_index_elems(self) -> Int:
        """`row_index`, `order`, and `scratch`, one Int32 per stored entry."""
        return 3 * self.nnz

    def entry_bin_elems(self) -> Int:
        return self.nnz

    def column_index_elems(self) -> Int:
        return self.n_features + 1

    def range_elems(self) -> Int:
        """Two Int32 (start, end) per (node, feature)."""
        return 2 * self.max_nodes * self.n_features

    def side_elems(self) -> Int:
        return self.n_rows

    def bytes(self) -> Int:
        """Total device bytes for the sparse structure itself.

        Excludes the gradient, hessian, histogram, and active-row buffers,
        which the sparse path shares with the dense one unchanged.
        """
        return (
            BYTES_INDEX * self.entry_index_elems()
            + BYTES_BIN * self.entry_bin_elems()
            + BYTES_INDEX * self.column_index_elems()
            + BYTES_BIN * self.n_features
            + BYTES_INDEX * self.range_elems()
            + BYTES_SIDE * self.side_elems()
        )

    def dense_bytes(self) -> Int:
        """What the same dataset costs as the dense `UInt8` bin matrix."""
        return BYTES_BIN * self.n_rows * self.n_features

    def density(self) -> Float64:
        """Stored entries over cells. Reported, never thresholded on."""
        return Float64(self.nnz) / Float64(self.n_rows * self.n_features)

    def bytes_ratio(self) -> Float64:
        """Sparse device bytes over dense device bytes. Above 1.0 the sparse
        layout is the larger of the two, which happens well before the
        density reaches 1: an entry costs 9 bytes here (row, bin, and its
        slot in both permutation buffers) against 1 byte dense."""
        return Float64(self.bytes()) / Float64(self.dense_bytes())

    @staticmethod
    def of(
        data: SparseBinnedMatrix, max_nodes: Int = DEFAULT_MAX_NODES
    ) raises -> SparseDeviceLayout:
        if max_nodes < 1:
            raise Error("max_nodes must be positive")
        return SparseDeviceLayout(
            data.n_rows,
            data.n_features,
            data.n_bins,
            data.nnz(),
            max_nodes,
        )


# --- Per-node cost model --------------------------------------------------


@fieldwise_init
struct SparseNodeCost(Copyable, Movable):
    """The work one node's histogram costs, in units a benchmark can weigh.

    Kept as separate counts rather than collapsed into one number, because
    the four units have wildly different per-unit costs on a GPU (a
    coalesced bin read is nothing next to a kernel launch, and a host
    synchronization is worth thousands of launches) and only a measurement
    knows the ratios.
    """

    var bin_reads: Int
    """Random-access reads of the binned data: `node_rows * n_active` dense,
    `entries_in_node` sparse."""

    var row_reads: Int
    """Sequential reads of the gradient/hessian arrays through the active-row
    permutation. Dense pays this inside the bin reads; sparse pays it once
    more, for the node-total reduction."""

    var launches: Int
    var host_syncs: Int


def dense_node_cost(node_rows: Int, n_active: Int) -> SparseNodeCost:
    """What `GpuHistogramBuilder.build_leaf` costs for one node.

    One accumulation kernel (plus a reduction under `STRATEGY_TILED`, not
    counted here since the strategy is the caller's) and one download.
    """
    return SparseNodeCost(node_rows * n_active, 0, 1, 1)


def sparse_node_cost(
    node_rows: Int, entries_in_node: Int
) -> SparseNodeCost:
    """What `GpuSparseHistogramBuilder.build_leaf` costs for one node.

    Three kernels: the node-total reduction over the node's row range, the
    entry accumulation over its entry windows, and the default-bin
    completion. One download, exactly as the dense path.

    The reduction over `node_rows` is the price of the subtraction trick: the
    default bin is `total - stored`, and `total` has to be summed in the same
    fixed point the entries are accumulated in, which means one pass over the
    node's rows. It is a *sequential* pass over two Float32 arrays, against
    the dense path's `n_active` random bin reads per row, which is why the
    two are counted in different units.
    """
    return SparseNodeCost(entries_in_node, node_rows, 3, 1)


@fieldwise_init
struct MeasuredCosts(Copyable, Movable):
    """Per-unit costs, in nanoseconds, that somebody measured on the device
    the decision is being made for.

    All four default to zero, which means *unmeasured*, and an unmeasured
    cost model returns `SPARSE_UNDECIDED` rather than a guess. There is
    deliberately no default set of numbers: they are device-specific by an
    order of magnitude (an M4's unified memory and an H100's HBM do not agree
    on any of these), and a constant baked in here would become the automatic
    threshold this lane is not allowed to build.
    """

    var ns_per_bin_read: Float64
    var ns_per_row_read: Float64
    var ns_per_launch: Float64
    var ns_per_host_sync: Float64

    @staticmethod
    def unmeasured() -> MeasuredCosts:
        return MeasuredCosts(0.0, 0.0, 0.0, 0.0)

    def is_measured(self) -> Bool:
        """Whether every unit the model uses has a positive cost. A partly
        filled set is treated as unmeasured: mixing a measured bin read with
        a guessed launch cost is how a threshold gets built by accident."""
        return (
            self.ns_per_bin_read > 0.0
            and self.ns_per_row_read > 0.0
            and self.ns_per_launch > 0.0
            and self.ns_per_host_sync > 0.0
        )

    def evaluate(self, cost: SparseNodeCost) -> Float64:
        """Nanoseconds this cost model predicts for one node."""
        return (
            Float64(cost.bin_reads) * self.ns_per_bin_read
            + Float64(cost.row_reads) * self.ns_per_row_read
            + Float64(cost.launches) * self.ns_per_launch
            + Float64(cost.host_syncs) * self.ns_per_host_sync
        )


comptime SPARSE_UNDECIDED = 0
comptime SPARSE_PREFER_SPARSE = 1
comptime SPARSE_PREFER_DENSE = 2
comptime SPARSE_UNSUPPORTED = 3


def sparse_verdict_name(verdict: Int) -> String:
    if verdict == SPARSE_UNDECIDED:
        return String("undecided")
    if verdict == SPARSE_PREFER_SPARSE:
        return String("sparse")
    if verdict == SPARSE_PREFER_DENSE:
        return String("dense")
    return String("unsupported")


@fieldwise_init
struct SparseVerdict(Copyable, Movable):
    """The outcome of a comparison, with the numbers it was made from.

    `verdict` is `SPARSE_UNDECIDED` unless the caller supplied measurements,
    and `SPARSE_UNSUPPORTED` when the shape cannot run at all. `sparse_ns`
    and `dense_ns` are zero in both of those cases.
    """

    var verdict: Int
    var support: Int
    var sparse_ns: Float64
    var dense_ns: Float64

    def is_decided(self) -> Bool:
        return (
            self.verdict == SPARSE_PREFER_SPARSE
            or self.verdict == SPARSE_PREFER_DENSE
        )


def decide_sparse(
    caps: DeviceCaps,
    layout: SparseDeviceLayout,
    node_rows: Int,
    entries_in_node: Int,
    n_active: Int,
    costs: MeasuredCosts,
) -> SparseVerdict:
    """Compare the two paths for one representative node.

    Returns `SPARSE_UNDECIDED` whenever `costs` is unmeasured, whatever the
    density is. That is the whole point of the signature: a caller that wants
    a decision has to produce the measurements, and a caller that has none
    gets no decision to lean on.

    Even with measurements this is a *node* comparison, not a training-run
    comparison: it ignores the once-per-session upload, the per-split
    partition, and the range download, all of which the handoff's staged plan
    requires benchmarking separately before a policy consumes any of this.
    """
    var support = sparse_support(
        caps,
        layout.n_rows,
        layout.n_features,
        layout.n_bins,
        layout.nnz,
    )
    if support != SPARSE_OK:
        return SparseVerdict(SPARSE_UNSUPPORTED, support, 0.0, 0.0)
    if not costs.is_measured():
        return SparseVerdict(SPARSE_UNDECIDED, support, 0.0, 0.0)
    var sparse_ns = costs.evaluate(
        sparse_node_cost(node_rows, entries_in_node)
    )
    var dense_ns = costs.evaluate(dense_node_cost(node_rows, n_active))
    var verdict = (
        SPARSE_PREFER_SPARSE if sparse_ns < dense_ns else SPARSE_PREFER_DENSE
    )
    return SparseVerdict(verdict, support, sparse_ns, dense_ns)


# --- Entry tiling ---------------------------------------------------------


def derive_entry_tiling(
    caps: DeviceCaps,
    max_entries: Int,
    n_slots: Int,
    n_bins: Int,
    strategy: Int,
    max_partial_cells: Int = 0,
) raises -> HistogramTiling:
    """Launch geometry for an entry-oriented histogram.

    The same policy `gpu_tiling.derive_tiling` computes for rows, with the
    tiled axis reinterpreted: `rows_per_tile` counts *stored entries* per
    tile and `n_tiles` is `grid.y` over a feature's entry window rather than
    over a row range. Nothing in the policy depends on what the tiled axis
    means, so reusing it keeps one place where block size, occupancy, and the
    partial-buffer budget are decided.

    `max_entries` must be the largest entry count over the active features of
    this node, not the total: `grid.y` is uniform across `grid.x`, so a
    feature with fewer entries simply leaves its tail tiles with nothing to
    do. That imbalance is the known weakness of this geometry and it is
    proportional to how uneven the column occupancies are; see the handoff.

    One inherited surprise: because the policy is reused verbatim,
    `MOJOTREES_GPU_ROW_TILE` forces the tile size here too, and it then means
    *entries* per tile rather than rows per tile. Reusing one override for
    one policy is better than a second variable that could disagree with it,
    but a benchmark that sets it should know it is setting both.
    """
    var n = max_entries
    if n < 1:
        n = 1
    return derive_tiling(
        caps, n, n_slots, n_bins, strategy, max_partial_cells
    )


# --- Node entry-window bookkeeping ---------------------------------------


struct SparseRangeTable(Copyable, Movable):
    """Node id -> per-feature entry window, for one tree.

    The host mirror of the device range buffer. Node ids are handed out in
    ascending order by the grower, so this is a `List` indexed by node id
    that grows by appending, exactly like `LeafRangeTable` in
    `gpu_active_rows.mojo`, and a split costs at most two appends.

    Two modes, both exact on the device and differing only in what the host
    believes:

    - `split` takes the per-feature midpoints the device computed and gives
      each child its true window. Requires downloading `n_features` Int32
      per split.
    - `bound_split` gives both children the parent's whole window as an
      *upper bound*. The device table still holds the true windows and the
      kernels still read those, so histograms are unchanged; only the launch
      geometry is over-provisioned, by whatever the split did not separate.
      Nothing is downloaded and nothing synchronizes.

    `is_exact` records which mode a node's window came from, so a caller
    cannot mistake a bound for a count.
    """

    var nodes: List[SparseNodeEntries]
    var exact: List[Bool]
    var root: SparseNodeEntries
    var n_features: Int
    var nnz: Int

    def __init__(out self, n_features: Int, nnz: Int):
        self.nodes = List[SparseNodeEntries]()
        self.exact = List[Bool]()
        self.root = SparseNodeEntries.empty(n_features)
        self.n_features = n_features
        self.nnz = nnz

    def reset_root(mut self, data: SparseBinnedMatrix) raises:
        """Start a new tree with every stored entry at node 0, grouped by
        feature exactly as the CSC layout already groups it."""
        if data.n_features != self.n_features:
            raise Error("range table and matrix disagree on n_features")
        if data.nnz() != self.nnz:
            raise Error("range table and matrix disagree on nnz")
        self.nodes.clear()
        self.exact.clear()
        self.root = SparseNodeEntries.root(data)
        self.nodes.append(self.root.copy())
        self.exact.append(True)

    def reset_root_offsets(mut self, col_offsets: List[Int]) raises:
        """`reset_root` for a caller that kept the CSC column offsets rather
        than the matrix. Same table, and the same one the device's root
        seeding kernel writes from the same offsets."""
        if len(col_offsets) != self.n_features + 1:
            raise Error("col_offsets must have length n_features + 1")
        if col_offsets[self.n_features] != self.nnz:
            raise Error("col_offsets must end at nnz")
        var starts = List[Int](capacity=self.n_features)
        var ends = List[Int](capacity=self.n_features)
        for f in range(self.n_features):
            starts.append(col_offsets[f])
            ends.append(col_offsets[f + 1])
        self.nodes.clear()
        self.exact.clear()
        self.root = SparseNodeEntries(starts^, ends^)
        self.nodes.append(self.root.copy())
        self.exact.append(True)

    def compact_root(mut self, mids: List[Int]) raises:
        """Narrow node 0's windows to `[start, mids[f])`.

        This is what a bagged tree needs: the root's row range covers only
        the bag, but the root's entry windows start out covering whole
        columns, entries of out-of-bag rows included. Accumulating those
        would break the subtraction that fills the default bin, since the
        node totals cover the bag alone and the leftover would come out
        negative. One partition of the root's windows by bag membership
        fixes it, and this records the result.
        """
        if len(mids) != self.n_features:
            raise Error("midpoint list must have one entry per feature")
        if len(self.nodes) != 1:
            raise Error("root compaction must happen before the first split")
        var starts = List[Int](capacity=self.n_features)
        var ends = List[Int](capacity=self.n_features)
        for f in range(self.n_features):
            var lo = self.root.starts[f]
            var hi = self.root.ends[f]
            if mids[f] < lo or mids[f] > hi:
                raise Error("entry midpoint is outside the root's window")
            starts.append(lo)
            ends.append(mids[f])
        # The out-of-bag entries are no longer reachable from any live node,
        # so the columns stop being tiled by the live windows and the
        # invariant check would (correctly) fail against the original
        # columns. The compacted root becomes the reference it compares to.
        self.root = SparseNodeEntries(starts^, ends^)
        self.nodes[0] = self.root.copy()

    def n_nodes(self) -> Int:
        return len(self.nodes)

    def get(self, node: Int) raises -> SparseNodeEntries:
        if node < 0 or node >= len(self.nodes):
            raise Error("node has no sparse entry window")
        return self.nodes[node].copy()

    def is_exact(self, node: Int) raises -> Bool:
        if node < 0 or node >= len(self.exact):
            raise Error("node has no sparse entry window")
        return self.exact[node]

    def max_entries(self, node: Int, features: List[Int]) raises -> Int:
        """The largest per-feature entry count of `node` over `features`
        (empty means every feature). This is what `derive_entry_tiling`
        takes, since `grid.y` is uniform across the active features."""
        var window = self.get(node)
        var use_all = len(features) == 0
        var n = self.n_features if use_all else len(features)
        var out = 0
        for i in range(n):
            var f = i if use_all else features[i]
            if f < 0 or f >= self.n_features:
                raise Error("feature index out of range")
            var count = window.ends[f] - window.starts[f]
            if count > out:
                out = count
        return out

    def total_entries(self, node: Int) raises -> Int:
        return self.get(node).n_entries()

    def _grow_to(mut self, node: Int):
        while len(self.nodes) <= node:
            self.nodes.append(SparseNodeEntries.empty(self.n_features))
            self.exact.append(True)

    def _check_ids(self, parent: Int, left: Int, right: Int) raises:
        if left < 0 or right < 0 or parent < 0:
            raise Error("node ids must be nonnegative")
        if left == parent or right == parent or left == right:
            raise Error(
                "child node ids must differ from the parent and each other"
            )

    def split(
        mut self, parent: Int, left: Int, right: Int, mids: List[Int]
    ) raises:
        """Hand the parent's windows to its children at the per-feature
        midpoints the device partition produced.

        `mids[f]` is the first index of feature f's right-going entries, so
        the left child owns `[start, mid)` and the right child `[mid, end)`.
        Both must sit inside the parent's window, which is what makes the
        children's windows sub-windows of the parent's for every feature.
        """
        var window = self.get(parent)
        var parent_exact = self.is_exact(parent)
        self._check_ids(parent, left, right)
        if len(mids) != self.n_features:
            raise Error("midpoint list must have one entry per feature")
        var left_entries = SparseNodeEntries.empty(self.n_features)
        var right_entries = SparseNodeEntries.empty(self.n_features)
        for f in range(self.n_features):
            var lo = window.starts[f]
            var hi = window.ends[f]
            var mid = mids[f]
            if mid < lo or mid > hi:
                raise Error("entry midpoint is outside the parent's window")
            left_entries.starts[f] = lo
            left_entries.ends[f] = mid
            right_entries.starts[f] = mid
            right_entries.ends[f] = hi
        self._grow_to(left)
        self._grow_to(right)
        if self.nodes[left].n_entries() != 0:
            raise Error("left child already owns an entry window")
        if self.nodes[right].n_entries() != 0:
            raise Error("right child already owns an entry window")
        self.nodes[left] = left_entries^
        self.nodes[right] = right_entries^
        # A midpoint the device computed sits inside the *true* window, and a
        # bound window contains the true one, so splitting a bound at a true
        # midpoint yields two windows that each still contain their child's
        # true window. Exactness therefore propagates rather than being
        # recovered: a child of a bound parent is still only bounded.
        self.exact[left] = parent_exact
        self.exact[right] = parent_exact
        self.nodes[parent] = SparseNodeEntries.empty(self.n_features)

    def bound_split(
        mut self, parent: Int, left: Int, right: Int
    ) raises:
        """Give both children the parent's window as an upper bound.

        For launch geometry only. The device table holds the true windows and
        every kernel reads them, so a histogram built under a bound is the
        same histogram; the cost is `grid.y` tiles that find their feature's
        window already exhausted and exit. Successive bound splits compound,
        so a deep subtree grown entirely this way is launched as if every
        node still held the root's entries.
        """
        var window = self.get(parent)
        self._check_ids(parent, left, right)
        self._grow_to(left)
        self._grow_to(right)
        if self.nodes[left].n_entries() != 0:
            raise Error("left child already owns an entry window")
        if self.nodes[right].n_entries() != 0:
            raise Error("right child already owns an entry window")
        self.nodes[left] = window.copy()
        self.nodes[right] = window.copy()
        self.exact[left] = False
        self.exact[right] = False
        self.nodes[parent] = SparseNodeEntries.empty(self.n_features)

    def flatten(self, max_nodes: Int) raises -> List[Int32]:
        """The table as the flat `[start, end]` pairs the device buffer
        holds, indexed `2 * (node * n_features + feature)`.

        Used to seed the device table for a new tree, and by a validation
        path that wants to compare the host's belief against the device's.
        """
        if max_nodes < len(self.nodes):
            raise Error("max_nodes is smaller than the live node count")
        var out = List[Int32](capacity=2 * max_nodes * self.n_features)
        out.resize(2 * max_nodes * self.n_features, Int32(0))
        for node in range(len(self.nodes)):
            var window = self.nodes[node].copy()
            for f in range(self.n_features):
                var base = 2 * (node * self.n_features + f)
                out[base] = Int32(window.starts[f])
                out[base + 1] = Int32(window.ends[f])
        return out^

    def check_invariants(self) raises:
        """Per feature, the live windows must tile that feature's whole
        column of the permutation, with no gap and no overlap.

        Only meaningful on an all-exact table: a bound window deliberately
        overlaps its sibling's, which is why `is_exact` exists and why this
        refuses to run on a table holding one.
        """
        for node in range(len(self.exact)):
            if not self.exact[node]:
                raise Error(
                    "cannot check tiling on a table holding bound windows"
                )
        for f in range(self.n_features):
            var covered = 0
            for i in range(len(self.nodes)):
                var a_lo = self.nodes[i].starts[f]
                var a_hi = self.nodes[i].ends[f]
                if a_hi <= a_lo:
                    continue
                if a_lo < self.root.starts[f] or a_hi > self.root.ends[f]:
                    raise Error("entry window escapes the feature's column")
                covered += a_hi - a_lo
                for k in range(i + 1, len(self.nodes)):
                    var b_lo = self.nodes[k].starts[f]
                    var b_hi = self.nodes[k].ends[f]
                    if b_hi <= b_lo:
                        continue
                    if a_lo < b_hi and b_lo < a_hi:
                        raise Error("entry windows overlap")
            # Disjointness plus the right total is coverage, since every
            # window sits inside the feature's own column.
            if covered != self.root.ends[f] - self.root.starts[f]:
                raise Error(
                    "entry windows do not cover the feature's column"
                )


# --- EFB compatibility ----------------------------------------------------


@fieldwise_init
struct BundleCompatibility(Copyable, Movable):
    """Whether a fitted EFB plan can be trained on through this layout, and
    what recovering an original feature's histogram then costs.

    A bundled matrix is an ordinary `SparseBinnedMatrix` (`efb.bundle_csc`
    returns one), so the device path accumulates it with no change at all:
    the kernels are bin-agnostic and a bundle's shared bin 0 is just that
    column's `default_bin`. The compatibility question is entirely about what
    happens *after* the histogram comes back.
    """

    var support: Int
    var n_bundles: Int
    var max_bundle_bins: Int
    var lossless: Bool
    var recovery_is_exact: Bool
    """Whether `efb.unbundle_histogram` recovers every member's local
    histogram exactly. True iff the plan drops no stored entry: a collision
    the member lost is folded into that member's default bin, which is the
    approximation `max_conflict_rate` buys and which no device change can
    undo."""


def check_bundle_compatibility(
    caps: DeviceCaps, plan: FeatureBundling, cats: CategoricalSpec
) raises -> BundleCompatibility:
    """Check a fitted bundling plan against the device path's limits.

    `cats` is the *source* matrix's categorical spec, the one the plan was
    fitted against, not the bundled matrix's.

    Two things are verified beyond the usual shape limits:

    - a categorical feature must be a singleton bundle. `efb.mojo` already
      guarantees this unconditionally, and the guarantee is what keeps a
      singleton's category table and its reserved unknown bin valid under
      identity encoding. It is checked rather than assumed, because a plan
      arrives here as data and a bundled categorical column would be
      silently wrong rather than loudly wrong: its bins would be offset into
      a shared range, so bin 0 would stop being the unknown bin and every
      unseen category would start routing left.
    - the plan's collision count is reported, because a lossy plan makes
      `unbundle_histogram` approximate and that has to reach the caller as a
      fact about the model, not as a device detail.
    """
    var max_bins = plan.max_bundle_bins()
    var support = sparse_support(
        caps,
        plan.n_rows,
        plan.n_bundles(),
        max_bins,
        plan.bundled_entries,
    )
    for b in range(plan.n_bundles()):
        if plan.bundle_size(b) == 1:
            continue
        for k in range(plan.bundle_start[b], plan.bundle_start[b + 1]):
            if cats.is_cat(plan.members[k]):
                return BundleCompatibility(
                    SPARSE_BUNDLED_CATEGORICAL,
                    plan.n_bundles(),
                    max_bins,
                    False,
                    False,
                )
    var lossless = plan.is_lossless()
    return BundleCompatibility(
        support,
        plan.n_bundles(),
        max_bins,
        lossless,
        lossless,
    )


# --- The capability record --------------------------------------------
#
# One value that answers, for one dataset and one device, what the sparse
# device path can and cannot do -- and says so in a form a trainer, a
# binding, or a device policy can report without catching an exception. It
# exists because the alternative is what the repository had: three separate
# checks in three modules, none of them consulted by anything, and a policy
# elsewhere asserting a fact ("there is no sparse GPU histogram kernel")
# that stopped being true.
#
# The honest distinction it draws, and the reason `training` is a separate
# field from `histograms`:
#
# - the *primitives* are real. `GpuSparseHistogramBuilder` uploads a
#   compressed matrix, builds a node histogram, and partitions rows and
#   entries at a split; `gpu_categorical` computes per-category statistics
#   and routes a categorical split from a device-resident set.
# - the *training path* is not. No grower drives them, no `device="gpu"`
#   reaches them, and nothing here will report that it does. A request that
#   asks for sparse GPU training gets an error naming the missing piece,
#   never a silent fall back to the CPU trainer while claiming the device.


def sparse_gpu_training_is_wired() -> Bool:
    """Whether a sparse GPU *training* path exists.

    Constant `False` today, and deliberately a function rather than a
    comment: every place that must not claim sparse GPU training reads it,
    so wiring a trainer is one edit here plus the trainer, and forgetting to
    flip it fails closed rather than open.
    """
    return False


@fieldwise_init
struct SparseGpuCapability(Copyable, Movable):
    """What the sparse device path can do with one dataset on one device.

    `support` and `categorical` are `sparse_support` / `categorical_support`
    codes, `SPARSE_OK` when the shape is expressible. `histograms` is whether
    the device primitives can run at all, and `training` is whether a
    training path exists to run them from -- two different questions, kept
    apart so neither can be read as the other.

    `unknown_absent` lists the categorical features whose absent rows land in
    the unknown bin (see `sparse.absent_is_unknown`). That is a modelling
    fact rather than a limit, so it never blocks anything; it rides along
    because this is the record a caller already has in hand when it decides
    what to report.
    """

    var support: Int
    var categorical: Int
    var histograms: Bool
    var training: Bool
    var unknown_absent: List[Int]

    def blocked_reason(self) -> Int:
        """The first code that makes this dataset unrunnable on the device
        path, or `SPARSE_OK` when the shape is fine."""
        if self.support != SPARSE_OK:
            return self.support
        return self.categorical

    def explain(self) -> String:
        """One line, suitable for an error message or a report."""
        var reason = self.blocked_reason()
        if reason != SPARSE_OK:
            return String(
                "sparse GPU path unavailable: ", sparse_support_name(reason)
            )
        if not self.training:
            return String(
                "sparse GPU primitives are available for this dataset but no"
                " sparse GPU training path is wired; training runs on the"
                " CPU sparse trainer"
            )
        return String("sparse GPU training available")


def sparse_gpu_capability(
    caps: DeviceCaps, data: SparseBinnedMatrix
) raises -> SparseGpuCapability:
    """The capability record for one binned sparse matrix on one device.

    Validates the matrix first, through `SparseBinnedMatrix.validate`: a
    record derived from a structurally broken matrix would describe a dataset
    that cannot be trained on at all, on either backend.
    """
    data.validate()
    var flagged = check_sparse_categorical_semantics(data)
    var support = sparse_support(
        caps, data.n_rows, data.n_features, data.n_bins, data.nnz()
    )
    var categorical = categorical_support(
        data.cats, data.n_features, data.n_bins
    )
    var runnable = support == SPARSE_OK and categorical == SPARSE_OK
    return SparseGpuCapability(
        support,
        categorical,
        runnable,
        runnable and sparse_gpu_training_is_wired(),
        flagged^,
    )


def check_sparse_gpu_histograms(
    caps: DeviceCaps, data: SparseBinnedMatrix
) raises -> SparseGpuCapability:
    """Raise unless the sparse device *primitives* can run on this dataset,
    and return the record either way.

    What `GpuSparseHistogramBuilder`'s constructor asks, in a form a caller
    can ask before it opens a device context.
    """
    var capability = sparse_gpu_capability(caps, data)
    if not capability.histograms:
        raise Error(capability.explain())
    return capability^


def check_sparse_gpu_training(
    caps: DeviceCaps, data: SparseBinnedMatrix
) raises -> SparseGpuCapability:
    """Raise unless a sparse GPU *training* path exists for this dataset.

    This is what an explicit `device="gpu"` request on sparse input must go
    through. Today it always raises, because `sparse_gpu_training_is_wired`
    is False: an unsupported shape raises with its own reason, and a
    perfectly supported one raises saying the primitives exist but nothing
    drives them. Neither answer is "we ran it on the CPU and called it the
    device".
    """
    var capability = check_sparse_gpu_histograms(caps, data)
    if not capability.training:
        raise Error(capability.explain())
    return capability^
