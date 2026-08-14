"""Who owns a histogram, when it dies, and how much memory the live set can
hold.

Tree growth already caches histograms, in two places that do not know about
each other: the CPU grower keeps one per frontier leaf in `_LeafState` and
recycles buffers through `_HistPool` (tree.mojo), and the GPU grower keeps
one per frontier leaf in `_GpuLeafState` after downloading it (train_gpu.mojo).
Both rely on the same lifetime rule without stating it: a parent's histogram
is dead the instant sibling subtraction has read it. This module states that
rule, gives every cached histogram a key that says exactly which
(dataset, gradients, feature set, tree, node) it describes, and bounds the
bytes the live set can reach.

It is the bookkeeping half of the hybrid scheduler in
`hybrid_leaf_scheduler.mojo`: once a histogram may be produced by either
device, "is this buffer still the right answer?" stops being obvious from
the call site and has to be a checkable property of a key.

**This module holds no histogram buffers.** It holds keys, sizes, origins,
and counters, the way `PoolLedger` and `ResidencyLedger` in gpu_runtime.mojo
hold slot capacities rather than device buffers. It imports nothing from the
GPU layer and opens no device, so all of it is exercisable on a CPU-only
machine.

What is cacheable, and what is not
----------------------------------
Stated as a negative result, because it is one. Beyond the parent/sibling
subtraction both growers already exploit, there is no histogram reuse
available in this trainer:

- **Across rounds: none.** A histogram is a sum of that round's gradients.
  Round `i + 1` refreshes every gradient, so every histogram from round `i`
  is wrong, not stale-but-usable. The fixed-point scales (`g_scale`,
  `h_scale` in histogram_gpu.mojo) are re-derived per round from the round's
  own magnitude sums, so even the dequantization constant changes.
- **Across trees within a round: none.** `begin_tree` reseeds the active-row
  permutation and node ids restart at 0, so a node id from the previous tree
  names different rows. Under bagging the tree's rows differ outright.
- **Across feature sets: none.** A histogram's shape is always the dataset's
  full `n_features * n_bins`, with the slices of features outside the active
  set left at zero (`set_features` in histogram_gpu.mojo,
  `build_histogram_into`'s `features` argument in histogram.mojo). A
  histogram built under feature set A therefore reads as "these features had
  no rows" under feature set B, which is silently wrong rather than
  detectably wrong. That is the invalidation this module exists to make
  detectable.
- **Across per-node feature draws: full reuse, already.**
  `feature_fraction_bynode` narrows the *search*, not the accumulation, so
  one histogram serves any per-node draw. Re-searching the same histogram
  under different monotone bounds is free for the same reason. Neither needs
  a cache key of its own, and neither invalidates one.
- **Parent and sibling: the whole of it.** Build the smaller child, subtract
  for the larger, drop the parent. That is what both growers do and what the
  bound below is sized for.

So the cache's job is not to find reuse that nobody has found. It is to make
the existing lifetime rule enforceable across two producers, and to refuse a
lookup that a single-device grower could never have made wrong.

Epochs
------
Three counters, each bumped by exactly one event, together with the dataset
identity, decide whether a key is still live:

- `round_epoch` -- bumped whenever gradients change: `upload_gradients`,
  `fill_gradients_device`, and each class's `fill_softmax_gradients_device`
  in a multiclass round. This is the counter that also stamps the
  fixed-point scales.
- `tree_epoch` -- bumped by `begin_tree`, which reseeds the row permutation
  and restarts node ids.
- `feature_epoch` -- bumped by a `set_features` call that actually changed
  the active set. `set_features` itself returns early when the set is
  unchanged, and this counter mirrors that: an unchanged set must not
  invalidate anything, or every tree would drop its root histogram.

Invalidation is therefore not an operation. A key either matches the current
epochs or it does not, and `staleness` says which counter disagreed.

Provenance
----------
Two producers can build the same node's histogram, and their outputs are
interchangeable only under a stated arithmetic. `ORIGIN_GPU_FIXED` is the
device's fixed-point Int32 accumulation dequantized on the host;
`ORIGIN_CPU_REPLICA` is a host accumulation through the *same* fixed-point
pipeline and the same scales, which is the only CPU origin that can stand in
for a device one; `ORIGIN_CPU_FLOAT64` is `build_histogram_subset_into`'s
plain Float64 sum, which is a different number and produces a different
tree. `ORIGIN_SUBTRACTED` is a sibling obtained by
`subtract_histogram_into`, which inherits the arithmetic of its two
operands.

`origins_are_subtractable` is what keeps that honest: subtracting a Float64
CPU histogram from a dequantized device parent is arithmetically defined and
semantically meaningless, and it is exactly the mistake a hybrid grower can
make without noticing.

Memory bound
------------
A histogram is three planes of `n_features * n_bins`: Float64 gradients,
Float64 hessians, and `Int` counts, which is `HISTOGRAM_BYTES_PER_CELL`
bytes per cell on a 64-bit target. The live set is bounded at
`num_leaves + 1` histograms, and `capacity_bound_bytes` proves the `+ 1`:
leaf-wise growth holds one per frontier leaf, and during a split the parent
is still live while both children exist, which is the peak.
"""

from .parallel import _env_int


# --- Provenance -----------------------------------------------------------

comptime ORIGIN_UNKNOWN = 0
comptime ORIGIN_GPU_FIXED = 1
comptime ORIGIN_CPU_REPLICA = 2
comptime ORIGIN_CPU_FLOAT64 = 3
comptime ORIGIN_SUBTRACTED = 4
comptime N_ORIGINS = 5


def origin_name(origin: Int) -> String:
    if origin == ORIGIN_GPU_FIXED:
        return String("gpu-fixed")
    if origin == ORIGIN_CPU_REPLICA:
        return String("cpu-replica")
    if origin == ORIGIN_CPU_FLOAT64:
        return String("cpu-float64")
    if origin == ORIGIN_SUBTRACTED:
        return String("subtracted")
    return String("unknown")


def origin_is_quantized(origin: Int) -> Bool:
    """Whether this origin's values passed through the Int32 fixed-point
    pipeline and were dequantized by a round's scales.

    `ORIGIN_SUBTRACTED` is deliberately absent: a difference is quantized
    exactly when both its operands were, which `origins_are_subtractable`
    checks and a single flag cannot express.
    """
    return origin == ORIGIN_GPU_FIXED or origin == ORIGIN_CPU_REPLICA


def origins_are_subtractable(parent: Int, child: Int) -> Bool:
    """Whether `parent - child` is a histogram of the sibling rather than a
    mixture of two arithmetics.

    Both operands must have come through the same accumulation. A
    dequantized device parent minus a Float64 host child is not the sibling:
    the two disagree at Float32 precision on every bin, and the difference
    carries that disagreement into a gain, a leaf value, and the split
    chosen from it.

    `ORIGIN_SUBTRACTED` is accepted on either side because a subtracted
    histogram is only ever admitted after this same check passed for the
    operands that produced it, which `HistogramCache.admit` enforces by
    taking the origin from the caller and never inventing one.
    """
    if parent == ORIGIN_UNKNOWN or child == ORIGIN_UNKNOWN:
        return False
    if parent == ORIGIN_SUBTRACTED or child == ORIGIN_SUBTRACTED:
        return True
    if origin_is_quantized(parent) != origin_is_quantized(child):
        return False
    return True


# --- Staleness ------------------------------------------------------------

comptime FRESH = 0
comptime STALE_ABSENT = 1
comptime STALE_DATASET = 2
comptime STALE_ROUND = 3
comptime STALE_TREE = 4
comptime STALE_FEATURES = 5
comptime STALE_SHAPE = 6
comptime STALE_RANGE = 7
comptime N_STALENESS = 8


def staleness_name(code: Int) -> String:
    if code == FRESH:
        return String("fresh")
    if code == STALE_ABSENT:
        return String("absent")
    if code == STALE_DATASET:
        return String("dataset-changed")
    if code == STALE_ROUND:
        return String("gradients-changed")
    if code == STALE_TREE:
        return String("tree-changed")
    if code == STALE_FEATURES:
        return String("feature-set-changed")
    if code == STALE_SHAPE:
        return String("shape-mismatch")
    if code == STALE_RANGE:
        return String("row-range-changed")
    return String("unknown")


# --- Sizes ----------------------------------------------------------------

# Float64 gradient + Float64 hessian + Int count per (feature, bin) cell, the
# layout of `Histogram` in histogram.mojo on a 64-bit target.
comptime HISTOGRAM_BYTES_PER_CELL = 24

# The device-side fixed-point planes are Int32 rather than Float64/Int, so a
# download is half the host footprint: `[grad | hess | count]`, 4 bytes each.
comptime FIXED_BYTES_PER_CELL = 12


def histogram_cells(n_features: Int, n_bins: Int) raises -> Int:
    if n_features < 1 or n_bins < 1:
        raise Error("a histogram needs at least one feature and one bin")
    return n_features * n_bins


def histogram_bytes(n_features: Int, n_bins: Int) raises -> Int:
    """Host bytes one `Histogram` of this shape occupies."""
    return HISTOGRAM_BYTES_PER_CELL * histogram_cells(n_features, n_bins)


def fixed_download_bytes(n_features: Int, n_bins: Int) raises -> Int:
    """Bytes `download_raw` moves for one node, whatever that node's row
    count is.

    Independent of the node's size, and that is the point: it is the term a
    tiny leaf cannot amortize, and the reason a hybrid scheduler is worth
    modelling at all. The full `n_features` is used rather than the active
    slot count because `out_dev` is allocated and copied at the dataset's
    full shape even when `set_features` has narrowed the launch.
    """
    return FIXED_BYTES_PER_CELL * histogram_cells(n_features, n_bins)


def capacity_bound_bytes(
    num_leaves: Int, n_features: Int, n_bins: Int
) raises -> Int:
    """Host bytes the live histogram set can reach during leaf-wise growth.

    `num_leaves + 1` histograms. Leaf-wise growth keeps one per frontier
    leaf; the peak is inside a split, where the parent is still live (the
    subtraction is reading it) while both children exist. One split is in
    flight at a time in both growers, so the excess is one histogram and not
    one per level.

    A hybrid grower does not raise this bound: it changes which device
    accumulated a buffer, not how many buffers are live, and the sibling it
    obtains by subtraction is a buffer the single-device grower also held.
    """
    if num_leaves < 1:
        raise Error("num_leaves must be positive")
    return (num_leaves + 1) * histogram_bytes(n_features, n_bins)


# --- Keys -----------------------------------------------------------------


@fieldwise_init
struct CacheEpochs(Copyable, Movable):
    """The counters that decide whether a cached histogram still describes
    anything.

    `dataset_id` is a caller-supplied identity for the binned matrix; the
    session layer already computes one (`bins_fingerprint` in
    gpu_runtime.mojo) and this module takes it as an opaque integer rather
    than recomputing it or importing the GPU layer to reach it. Zero means
    "not identified", which never matches a stamped key, so an unstamped
    cache is an empty cache rather than a wrongly-hit one.
    """

    var dataset_id: UInt64
    var round_epoch: Int
    var tree_epoch: Int
    var feature_epoch: Int

    @staticmethod
    def start(dataset_id: UInt64) -> CacheEpochs:
        """Epoch counters for a fresh training session. All three start at
        zero and only ever increase, so a key from any earlier state of the
        session compares unequal."""
        return CacheEpochs(dataset_id, 0, 0, 0)

    def begin_round(mut self):
        """Gradients (and the fixed-point scales derived from them) have
        changed. One call per `upload_gradients` /
        `fill_gradients_device`, and one per class in a softmax round, since
        each class refills the same gradient buffers."""
        self.round_epoch += 1

    def begin_tree(mut self):
        """A new tree: the row permutation is reseeded and node ids restart
        at 0."""
        self.tree_epoch += 1

    def set_features(mut self, changed: Bool):
        """A `set_features` call. `changed` is that call's own
        already-computed answer (histogram_gpu.mojo returns early when the
        active set is unchanged); passing False bumps nothing, so an
        unchanged set cannot invalidate the root histogram that was built
        under it."""
        if changed:
            self.feature_epoch += 1

    def matches(self, other: CacheEpochs) -> Bool:
        return (
            self.dataset_id == other.dataset_id
            and self.round_epoch == other.round_epoch
            and self.tree_epoch == other.tree_epoch
            and self.feature_epoch == other.feature_epoch
        )


@fieldwise_init
struct HistogramKey(Copyable, Movable):
    """What one cached histogram describes.

    The epochs answer "of what gradients, of what tree, of what feature
    set". The node id and its half-open active-row window answer "of what
    rows": the node id alone is not enough, because a wiring mistake that
    reuses an id inside one tree would otherwise hit rather than raise, and
    the window is the identity `LeafRangeTable` already maintains
    (gpu_active_rows.mojo). `n_rows` is redundant with the window on the GPU
    path and is the only row identity the CPU path has, since the CPU grower
    carries row *lists* rather than ranges; it is carried so both producers
    can stamp a key.
    """

    var epochs: CacheEpochs
    var node: Int
    var range_begin: Int
    var range_end: Int
    var n_rows: Int
    var n_features: Int
    var n_bins: Int

    @staticmethod
    def range_node(
        epochs: CacheEpochs,
        node: Int,
        range_begin: Int,
        range_end: Int,
        n_features: Int,
        n_bins: Int,
    ) raises -> HistogramKey:
        """A key for a node the GPU grower owns, whose rows are the
        half-open window `[range_begin, range_end)` of the active-row
        permutation."""
        if node < 0:
            raise Error("node id must be nonnegative")
        if range_begin < 0 or range_end < range_begin:
            raise Error("active-row window is not a valid range")
        return HistogramKey(
            epochs,
            node,
            range_begin,
            range_end,
            range_end - range_begin,
            n_features,
            n_bins,
        )

    @staticmethod
    def list_node(
        epochs: CacheEpochs,
        node: Int,
        n_rows: Int,
        n_features: Int,
        n_bins: Int,
    ) raises -> HistogramKey:
        """A key for a node whose rows are a host-side list rather than a
        device window. The window is recorded as empty, which no
        `range_node` key can equal for a node that owns rows, so the two
        producers cannot collide on one key by accident."""
        if node < 0:
            raise Error("node id must be nonnegative")
        if n_rows < 0:
            raise Error("row count must be nonnegative")
        return HistogramKey(epochs, node, 0, 0, n_rows, n_features, n_bins)

    def shape_matches(self, n_features: Int, n_bins: Int) -> Bool:
        return self.n_features == n_features and self.n_bins == n_bins

    def matches(self, other: HistogramKey) -> Bool:
        return (
            self.epochs.matches(other.epochs)
            and self.node == other.node
            and self.range_begin == other.range_begin
            and self.range_end == other.range_end
            and self.n_rows == other.n_rows
            and self.n_features == other.n_features
            and self.n_bins == other.n_bins
        )

    def bytes(self) raises -> Int:
        return histogram_bytes(self.n_features, self.n_bins)


def staleness(key: HistogramKey, now: CacheEpochs) -> Int:
    """Why `key` is no longer usable, or `FRESH`.

    Ordered from the coarsest disagreement to the finest, so the reported
    code names the outermost thing that changed rather than a consequence of
    it: a new round is reported as `STALE_ROUND` even though the tree
    counter has usually moved too.
    """
    if key.epochs.dataset_id != now.dataset_id:
        return STALE_DATASET
    if key.epochs.round_epoch != now.round_epoch:
        return STALE_ROUND
    if key.epochs.tree_epoch != now.tree_epoch:
        return STALE_TREE
    if key.epochs.feature_epoch != now.feature_epoch:
        return STALE_FEATURES
    return FRESH


def subtraction_staleness(
    parent: HistogramKey, child: HistogramKey
) -> Int:
    """Why `parent - child` is not the sibling's histogram, or `FRESH`.

    Checks only what the keys can see. That the child really is a child of
    the parent is a tree-shape fact the grower knows and a key does not, so
    it is the caller's precondition; what is checked here is that the two
    describe the same gradients, the same tree, the same feature set, the
    same shape, and that the child's window sits inside the parent's.
    """
    if not parent.epochs.matches(child.epochs):
        return staleness(child, parent.epochs)
    if not parent.shape_matches(child.n_features, child.n_bins):
        return STALE_SHAPE
    if parent.range_end > parent.range_begin:
        if (
            child.range_begin < parent.range_begin
            or child.range_end > parent.range_end
        ):
            return STALE_RANGE
    if child.n_rows > parent.n_rows:
        return STALE_RANGE
    return FRESH


def sibling_key(
    parent: HistogramKey, child: HistogramKey, sibling_node: Int
) raises -> HistogramKey:
    """The key the subtracted sibling gets.

    Derived rather than constructed by the caller so a sibling cannot be
    filed under epochs its operands do not share. The window is the
    complement of the child's inside the parent's, which is exactly what the
    stable partition produced: the two children tile the parent's range with
    the left child first.
    """
    var why = subtraction_staleness(parent, child)
    if why != FRESH:
        raise Error(
            String(
                "cannot subtract these histograms: ", staleness_name(why)
            )
        )
    if sibling_node < 0:
        raise Error("node id must be nonnegative")
    if parent.range_end <= parent.range_begin:
        return HistogramKey.list_node(
            parent.epochs,
            sibling_node,
            parent.n_rows - child.n_rows,
            parent.n_features,
            parent.n_bins,
        )
    var begin: Int
    var end: Int
    if child.range_begin == parent.range_begin:
        begin = child.range_end
        end = parent.range_end
    else:
        begin = parent.range_begin
        end = child.range_begin
    return HistogramKey.range_node(
        parent.epochs,
        sibling_node,
        begin,
        end,
        parent.n_features,
        parent.n_bins,
    )


# --- The ledger -----------------------------------------------------------


@fieldwise_init
struct CacheEntry(Copyable, Movable):
    """One live histogram, as bookkeeping: what it describes, how it was
    produced, and how many bytes it holds. The buffer itself stays with
    whoever owns it (a frontier leaf, a `_HistPool` slot); this is the
    record that says the buffer is still the answer to a question."""

    var key: HistogramKey
    var origin: Int
    var bytes: Int


struct HistogramCache(Copyable, Movable):
    """A ledger of live histograms with a byte ceiling.

    Deliberately not a store. Handing buffers to a cache would mean moving
    them out of the frontier states that already own them, which is a change
    to two growers this lane must not touch. What a ledger can do without
    owning anything is the part that stops being obvious once two devices
    produce histograms: refuse a lookup whose key has gone stale, refuse a
    subtraction that mixes arithmetics, and hold the live set to a bound
    that can be checked against `capacity_bound_bytes`.

    Sized for tens of entries (one per frontier leaf, `num_leaves` at most),
    so lookup is a linear scan. That is the same tradeoff
    `LeafRangeTable.check_invariants` makes and for the same reason: at this
    size the scan is cheaper than the structure that would avoid it, and it
    is what a test wants to read.
    """

    var entries: List[CacheEntry]
    var max_bytes: Int
    var live_bytes: Int
    var peak_bytes: Int
    var hits: Int
    var misses: Int
    var stale_drops: Int
    var admits: Int
    var releases: Int

    def __init__(out self, max_bytes: Int = 0) raises:
        """`max_bytes` of 0 means unbounded, which is the setting for a
        grower that already bounds its own frontier; a positive value makes
        `admit` refuse rather than let the live set grow past it."""
        if max_bytes < 0:
            raise Error("cache byte ceiling must be nonnegative")
        self.entries = List[CacheEntry]()
        self.max_bytes = max_bytes
        self.live_bytes = 0
        self.peak_bytes = 0
        self.hits = 0
        self.misses = 0
        self.stale_drops = 0
        self.admits = 0
        self.releases = 0

    def n_entries(self) -> Int:
        return len(self.entries)

    def find(self, key: HistogramKey) -> Int:
        """Index of the entry `key` names, or -1. Counts nothing: `lookup`
        is the counting form, and an invariant check should not move a hit
        rate."""
        for i in range(len(self.entries)):
            if self.entries[i].key.matches(key):
                return i
        return -1

    def lookup(mut self, key: HistogramKey, now: CacheEpochs) -> Int:
        """Index of a *usable* entry for `key`, or -1.

        A key that has gone stale never hits, whether or not an entry with
        the same node id is present, which is the whole guarantee this
        module offers a hybrid grower.
        """
        if staleness(key, now) != FRESH:
            self.misses += 1
            return -1
        var i = self.find(key)
        if i < 0:
            self.misses += 1
            return -1
        self.hits += 1
        return i

    def origin_of(self, index: Int) raises -> Int:
        if index < 0 or index >= len(self.entries):
            raise Error("cache entry index out of range")
        return self.entries[index].origin

    def key_of(self, index: Int) raises -> HistogramKey:
        if index < 0 or index >= len(self.entries):
            raise Error("cache entry index out of range")
        return self.entries[index].key.copy()

    def admit(
        mut self, key: HistogramKey, origin: Int, now: CacheEpochs
    ) raises:
        """Record a histogram that has just been produced.

        Refuses an origin of `ORIGIN_UNKNOWN`: a hybrid grower has two
        producers and a third path that subtracts, so "somebody built this"
        is not a provenance and cannot be checked against later.
        """
        if origin <= ORIGIN_UNKNOWN or origin >= N_ORIGINS:
            raise Error("a cached histogram needs a known origin")
        var why = staleness(key, now)
        if why != FRESH:
            raise Error(
                String(
                    "cannot admit a histogram that is already stale: ",
                    staleness_name(why),
                )
            )
        if self.find(key) >= 0:
            raise Error("this histogram is already in the cache")
        var size = key.bytes()
        if self.max_bytes > 0 and self.live_bytes + size > self.max_bytes:
            raise Error(
                "admitting this histogram would exceed the cache byte"
                " ceiling; raise the ceiling or release the parent first"
            )
        self.entries.append(CacheEntry(key.copy(), origin, size))
        self.live_bytes += size
        if self.live_bytes > self.peak_bytes:
            self.peak_bytes = self.live_bytes
        self.admits += 1

    def release(mut self, key: HistogramKey) raises:
        """Drop one entry: the split parent whose subtraction has been read,
        or a frontier leaf whose tree has ended."""
        var i = self.find(key)
        if i < 0:
            raise Error("this histogram is not in the cache")
        self.live_bytes -= self.entries[i].bytes
        self.releases += 1
        var last = len(self.entries) - 1
        if i != last:
            self.entries[i] = self.entries[last].copy()
        _ = self.entries.pop()

    def drop_stale(mut self, now: CacheEpochs) -> Int:
        """Drop every entry that no longer describes anything and return how
        many went. A round or tree boundary makes this the whole cache; it
        is a sweep rather than a clear so that a boundary which invalidated
        nothing costs nothing."""
        var dropped = 0
        var i = 0
        while i < len(self.entries):
            if staleness(self.entries[i].key, now) != FRESH:
                self.live_bytes -= self.entries[i].bytes
                var last = len(self.entries) - 1
                if i != last:
                    self.entries[i] = self.entries[last].copy()
                _ = self.entries.pop()
                dropped += 1
            else:
                i += 1
        self.stale_drops += dropped
        return dropped

    def clear(mut self):
        self.entries.clear()
        self.live_bytes = 0

    def within_bound(self, num_leaves: Int) raises -> Bool:
        """Whether the live set is inside `capacity_bound_bytes` for this
        leaf budget. False is a bug in the caller's lifetime handling, not a
        pressure signal: leaf-wise growth cannot hold more than
        `num_leaves + 1` histograms unless something failed to release a
        parent."""
        if len(self.entries) == 0:
            return True
        var shape = self.entries[0].key.copy()
        return self.live_bytes <= capacity_bound_bytes(
            num_leaves, shape.n_features, shape.n_bins
        )

    def check_subtraction(
        mut self, parent: HistogramKey, child: HistogramKey
    ) raises -> Int:
        """Validate a sibling subtraction against the ledger and return the
        parent's entry index.

        Both operands must be present, their keys must agree (see
        `subtraction_staleness`), and their origins must share an
        arithmetic. This is the check that a single-device grower gets for
        free and a hybrid one does not.
        """
        var pi = self.find(parent)
        if pi < 0:
            raise Error("the parent histogram is not in the cache")
        var ci = self.find(child)
        if ci < 0:
            raise Error("the child histogram is not in the cache")
        var why = subtraction_staleness(parent, child)
        if why != FRESH:
            raise Error(
                String(
                    "cannot subtract these histograms: ",
                    staleness_name(why),
                )
            )
        if not origins_are_subtractable(
            self.entries[pi].origin, self.entries[ci].origin
        ):
            raise Error(
                String(
                    "cannot subtract a ",
                    origin_name(self.entries[ci].origin),
                    " histogram from a ",
                    origin_name(self.entries[pi].origin),
                    " one: the two accumulate in different arithmetics, so"
                    " the difference is not the sibling",
                )
            )
        return pi

    def report(self) -> String:
        """One line for a trace or a bug report."""
        return String(
            "entries=",
            len(self.entries),
            " live=",
            self.live_bytes,
            "B peak=",
            self.peak_bytes,
            "B hits=",
            self.hits,
            " misses=",
            self.misses,
            " admits=",
            self.admits,
            " releases=",
            self.releases,
            " stale_drops=",
            self.stale_drops,
        )


# --- Environment ----------------------------------------------------------


def env_cache_bound_bytes() -> Int:
    """`MOJOBOOST_HIST_CACHE_BYTES`, following the `MOJOBOOST_` contract in
    parallel.mojo: a byte ceiling for the live histogram set, or 0 (the
    default) for unbounded.

    Present so a memory-constrained run can make the bound an error rather
    than a swap storm, and so a test can force the refusal path. It tunes no
    behavior otherwise: nothing here evicts, so the ceiling can only refuse.
    """
    return _env_int("MOJOBOOST_HIST_CACHE_BYTES", 0)
