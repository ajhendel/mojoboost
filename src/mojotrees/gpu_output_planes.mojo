"""Output-plane layout and class-batch planning for multiclass training.

Pure host arithmetic. Nothing here opens a `DeviceContext`, allocates a
buffer, or launches a kernel, so every layout decision, every memory
formula, and every batching bound below is reasonable about (and testable)
without a GPU, the way `gpu_tiling.mojo` and `gpu_sparse_layout.mojo` are.
`gpu_multiclass_batch.mojo` consumes this module; it consumes nothing of
mojotrees but `parallel._env_int`.

The layout question this module answers
---------------------------------------
A softmax round holds two different per-(row, class) quantities, and they
want opposite layouts:

    raw scores, probabilities    row-major, `x[r * n_classes + k]`
    gradients, hessians          class-major, `x[k * n_rows + r]`

Row-major wins for scores because every consumer of a score reduces *over
classes within one row*: the softmax max-subtraction, the denominator, the
log-loss of the true class, and the argmax of a prediction all read
`n_classes` contiguous floats. That is also the layout the host trainer
already documents (`raw[r * n_classes + k]` in `boosting.mojo`), the layout
`GpuObjectiveState.raw_dev`/`prob_dev` already carry, and the layout
`gpu_predict._predict_kernel` already writes. None of that changes here, and
nothing here may change it: prediction shapes are contract.

Class-major wins for derivatives because every consumer of a gradient reduces
*over rows within one class*: the histogram kernels, the magnitude reduction
that sets the fixed-point scale, and the leaf-value sums. A class-major plane
is exactly the `Float32[n_rows]` buffer those kernels already take, at the
offset `k * n_rows`, so batching C classes into one allocation costs the
existing kernels a base-pointer offset and nothing else. Row-major gradients
would instead need a stride argument threaded through every histogram kernel
and would make each kernel's row loop read with stride `n_classes`, which is
the one access pattern a coalesced Float32 load cannot absorb.

So the transpose happens exactly once per round, inside the batched gradient
kernel: it reads row-major probabilities and writes class-major derivatives.
That is the whole layout design, and `plane_index` below is the only place
either convention is spelled.

What a batch buys, quantitatively
---------------------------------
Growing `k_count` class trees together can share three things, and it is
worth being precise about which, because they have very different sizes:

1. **Bin reads.** Only when the batched classes read the *same rows in the
   same order* through the *same active feature set*. That is true at a
   round's root (every class tree of a round starts from the one shared bag
   or the identity permutation) and false at every deeper node, because each
   class picks its own split. When it holds and `classes_per_block` classes
   share a threadgroup, the round reads the binned matrix
   `ceil(n_classes / classes_per_block)` times instead of `n_classes` times.
   This is the largest saving available and the only one that scales with
   `n_rows * n_features`.

2. **Bin counts.** The count plane of a histogram does not depend on the
   class, so a batch over a shared row range needs one count plane, not
   `k_count` of them. That is a third of the output bytes and a third of the
   shared memory, which is what lets more classes share a threadgroup.

3. **Launches and host synchronizations.** Always available, shared rows or
   not. `n_classes` separate magnitude reductions with `n_classes` readbacks
   become one launch and one readback; `n_classes` small deep-node
   histograms become one launch with `n_classes` times the parallelism.
   This is the saving that matters for the tail of a tree, where each node's
   own row count is too small to fill the device.

Nothing here decides whether a batch is *faster* on a given device. The
formulas below count bytes and passes; `bench/apple/multiclass_batch_plan.json`
is where the measurements go.

What this module refuses to decide
----------------------------------
It never reorders classes. A batch is a contiguous ascending run of class
ids, batches run in ascending order, and `class_at` is the only mapping from
a batch slot to a class. That keeps the serialized tree order
(`trees[round * n_classes + k]`, `round_tree_slot` below) and the per-tree
feature-subsampling seed identical to what the sequential loop produces, on
either backend, at any batch size. A batching scheme that cannot hold that
invariant is not admissible, and there is no parameter here that relaxes it.
"""

from .binning import MAX_BINS
from .parallel import _env_int


# Layout codes for a per-(row, output) plane.
comptime PLANE_ROW_MAJOR = 0
"""`x[r * n_outputs + k]`. Raw scores, probabilities, predictions."""
comptime PLANE_CLASS_MAJOR = 1
"""`x[k * n_rows + r]`. Gradients, hessians, per-class row weights."""

# Every device-side plane this module sizes is Float32 (scores, gradients)
# or Int32 (histogram cells, row indices). Apple GPUs have no Float64, so
# there is no wider case to size for.
comptime BYTES_PER_F32 = 4
comptime BYTES_PER_I32 = 4

# One row of a batch costs a gradient and a hessian.
comptime BYTES_PER_GRAD_ROW = 8

# One row of a per-class active-row permutation costs the permutation, the
# scatter scratch, and the scan offsets, which is what `GpuActiveRows`
# allocates per row (`rows_dev`, `scratch_dev`, `offsets_dev`).
comptime BYTES_PER_ROW_INDEX_SET = 12

# A histogram cell is one Int32 in one of the three planes.
comptime BYTES_PER_HIST_CELL = 4


# Mirrors `gpu_objectives_native.SUM_BLOCKS`. Duplicated rather than
# imported so this module stays free of the device imports that module
# carries; `magnitude_partial_bytes` is the only user, and a test that
# imports both can assert they agree.
comptime SUM_BLOCKS_MIRROR = 256

# Classes that may share one threadgroup's shared memory in the batched
# shared-row histogram kernel. The kernel allocates
# `SHARED_CLASS_CAP * MAX_BINS` Int32 for gradients and the same for
# hessians, plus one `MAX_BINS` count plane: at the cap that is
# 4 * 256 * 4 * 2 + 256 * 4 = 9216 bytes, inside the 16 KiB every supported
# backend advertises (`gpu_tiling.FALLBACK_SHARED_MEMORY_PER_BLOCK`).
# Raising this needs a bin-capacity-parameterized kernel, which is the
# subject of `gpu_histogram_specializations.mojo`, not of a constant here.
comptime SHARED_CLASS_CAP = 4

# Ceiling on the device memory one class batch may hold in gradient planes,
# histogram output, and per-class row indices. 512 MiB is deliberately well
# under a unified-memory machine's budget: the training buffers this shares
# a device with (the binned matrix, the active rows, the partial histogram)
# are all sized independently and this is the one that scales with the class
# count. `MOJOTREES_GPU_CLASS_BATCH_BYTES` overrides it.
comptime CLASS_BATCH_BUDGET_BYTES = 512 << 20


def _ceil_div(a: Int, b: Int) -> Int:
    return (a + b - 1) // b


@always_inline
def row_major_index(r: Int, k: Int, n_outputs: Int) -> Int:
    """`x[r * n_outputs + k]`, the score layout."""
    return r * n_outputs + k


@always_inline
def class_major_index(r: Int, k: Int, n_rows: Int) -> Int:
    """`x[k * n_rows + r]`, the derivative layout."""
    return k * n_rows + r


@fieldwise_init
struct OutputPlanes(Copyable, Movable):
    """The shape and layout of one per-(row, output) device plane.

    Carrying the layout as a value rather than as a convention is what keeps
    the two conventions from being confused at a call site: a plane knows
    whether `index(r, k)` strides by `n_outputs` or by `n_rows`, and
    `class_plane_is_contiguous` answers the one question the histogram
    kernels actually ask.
    """

    var n_rows: Int
    var n_outputs: Int
    var layout: Int

    @staticmethod
    def scores(n_rows: Int, n_outputs: Int) raises -> OutputPlanes:
        """Raw scores, probabilities, and predictions: row-major, always.
        This is a serialized-contract layout, not a choice."""
        var p = OutputPlanes(n_rows, n_outputs, PLANE_ROW_MAJOR)
        p.check()
        return p^

    @staticmethod
    def derivatives(n_rows: Int, n_outputs: Int) raises -> OutputPlanes:
        """Gradients and hessians: class-major, so each class's plane is the
        contiguous `Float32[n_rows]` the histogram kernels already read."""
        var p = OutputPlanes(n_rows, n_outputs, PLANE_CLASS_MAJOR)
        p.check()
        return p^

    def check(self) raises:
        if self.n_rows < 1:
            raise Error("output planes need at least one row")
        if self.n_outputs < 1:
            raise Error("output planes need at least one output")
        if self.layout != PLANE_ROW_MAJOR and (
            self.layout != PLANE_CLASS_MAJOR
        ):
            raise Error("unknown output-plane layout")

    @always_inline
    def index(self, r: Int, k: Int) -> Int:
        """The flat offset of `(row, output)` in this plane's layout."""
        if self.layout == PLANE_CLASS_MAJOR:
            return class_major_index(r, k, self.n_rows)
        return row_major_index(r, k, self.n_outputs)

    @always_inline
    def stride_over_outputs(self) -> Int:
        """Distance between two outputs of one row."""
        if self.layout == PLANE_CLASS_MAJOR:
            return self.n_rows
        return 1

    @always_inline
    def stride_over_rows(self) -> Int:
        """Distance between two rows of one output."""
        if self.layout == PLANE_CLASS_MAJOR:
            return 1
        return self.n_outputs

    @always_inline
    def class_plane_is_contiguous(self) -> Bool:
        """True when output `k` occupies one unbroken `n_rows` run, which is
        what lets an existing per-row kernel take `offset_of_output(k)` as
        its base pointer with no other change."""
        return self.layout == PLANE_CLASS_MAJOR

    def offset_of_output(self, k: Int) raises -> Int:
        """The start of output `k`'s contiguous plane. Only class-major
        planes have one."""
        if k < 0 or k >= self.n_outputs:
            raise Error("output index out of range")
        if not self.class_plane_is_contiguous():
            raise Error(
                "a row-major plane has no contiguous per-output run; index"
                " it with `index(r, k)`"
            )
        return k * self.n_rows

    @always_inline
    def n_scores(self) -> Int:
        return self.n_rows * self.n_outputs

    @always_inline
    def bytes(self) -> Int:
        """Float32 bytes one plane of this shape occupies."""
        return self.n_scores() * BYTES_PER_F32


@fieldwise_init
struct BatchEligibility(Copyable, Movable):
    """What the classes of a candidate batch actually share.

    Both flags are facts about the caller's state at the moment a batch is
    formed, not policy: `shared_rows` is true when every class in the batch
    reads the same active-row window in the same order, and
    `shared_features` is true when every class scans the same active feature
    set in the same slot order. Only their conjunction licenses sharing bin
    reads, because a threadgroup that reads one bin column serves several
    classes only if all of them want that column for those rows.
    """

    var shared_rows: Bool
    var shared_features: Bool

    @staticmethod
    def round_root(feature_subsampling: Bool) -> BatchEligibility:
        """The root of a softmax round. Every class tree of the round starts
        on the same rows (`_boost_rounds_multiclass` draws one bag or GOSS
        sample per round, before any class's tree, exactly so this holds), so
        `shared_rows` is unconditionally true. Feature subsampling draws once
        per tree with seed `round * n_classes + k`, so the classes share a
        feature set only when subsampling is off."""
        return BatchEligibility(True, not feature_subsampling)

    @staticmethod
    def deeper_node() -> BatchEligibility:
        """Any node below a round's root. Each class chose its own split, so
        neither the rows nor (under subsampling) the features agree. Launch
        batching still applies; bin sharing does not."""
        return BatchEligibility(False, False)

    @always_inline
    def bin_reads_shared(self) -> Bool:
        return self.shared_rows and self.shared_features

    @always_inline
    def counts_shared(self) -> Bool:
        """The count plane depends on the rows and the bins, never on the
        class, so a shared row window is the whole condition."""
        return self.shared_rows


def gradient_plane_bytes(n_rows: Int, batch: Int) -> Int:
    """Gradients and hessians for `batch` classes, class-major."""
    return BYTES_PER_GRAD_ROW * n_rows * batch


def score_plane_bytes(n_rows: Int, n_classes: Int) -> Int:
    """Raw scores plus softmax probabilities, row-major. Sized by the class
    count and never by the batch: the whole score matrix is resident for the
    whole run, because a round's probabilities are a reduction over every
    class of a row."""
    return 2 * BYTES_PER_F32 * n_rows * n_classes


def histogram_batch_cells(
    n_features: Int, n_bins: Int, batch: Int, shared_counts: Bool
) -> Int:
    """Int32 cells in a batched histogram output.

    With a shared count plane the layout is
    `[grad_0 .. grad_{K-1} | hess_0 .. hess_{K-1} | count]`, so `2K + 1`
    planes; otherwise each class carries its own `[grad | hess | count]`
    triple, so `3K`.
    """
    var planes = 3 * batch
    if shared_counts:
        planes = 2 * batch + 1
    return planes * n_features * n_bins


def histogram_batch_bytes(
    n_features: Int, n_bins: Int, batch: Int, shared_counts: Bool
) -> Int:
    return BYTES_PER_HIST_CELL * histogram_batch_cells(
        n_features, n_bins, batch, shared_counts
    )


def partial_batch_cells(
    n_tiles: Int,
    n_slots: Int,
    n_bins: Int,
    batch: Int,
    shared_counts: Bool,
) -> Int:
    """Int32 cells a tiled batched accumulation would need for its partials:
    the batched output shape, once per row tile."""
    return n_tiles * histogram_batch_cells(
        n_slots, n_bins, batch, shared_counts
    )


def magnitude_partial_bytes(batch: Int) -> Int:
    """The batched magnitude reduction's partials: two planes of
    `SUM_BLOCKS` Float32 per class. Independent of `n_rows`, which is the
    point of reducing on the device."""
    return 2 * BYTES_PER_F32 * SUM_BLOCKS_MIRROR * batch


def active_rows_batch_bytes(n_rows: Int, batch: Int, shared_rows: Bool) -> Int:
    """Row-index memory a batch needs. A batch over a shared row window
    borrows the one permutation the caller already has and costs nothing;
    a batch whose classes have diverged needs one permutation per class."""
    if shared_rows:
        return 0
    return BYTES_PER_ROW_INDEX_SET * n_rows * batch


def class_batch_bytes(
    n_rows: Int,
    n_features: Int,
    n_bins: Int,
    n_tiles: Int,
    batch: Int,
    shared_counts: Bool,
    shared_rows: Bool,
) -> Int:
    """Total device bytes one resident class batch occupies.

    The four terms, in the order they matter as the class count grows:

        gradients   8 * n_rows * K              scales with rows
        row indices 12 * n_rows * K             scales with rows, 0 if shared
        partials    4 * n_tiles * (2K+1 or 3K) * n_features * n_bins
        output      4 * (2K+1 or 3K) * n_features * n_bins
        magnitudes  2048 * K                    constant per class

    At a million rows the first two dominate everything else by two orders
    of magnitude, which is why `plan_class_batches` bounds the batch by rows
    and not by the histogram shape.
    """
    var total = gradient_plane_bytes(n_rows, batch)
    total += active_rows_batch_bytes(n_rows, batch, shared_rows)
    total += histogram_batch_bytes(n_features, n_bins, batch, shared_counts)
    total += BYTES_PER_HIST_CELL * partial_batch_cells(
        n_tiles, n_features, n_bins, batch, shared_counts
    )
    total += magnitude_partial_bytes(batch)
    return total


def classes_per_block(
    n_bins: Int, shared_bytes_per_block: Int, shared_counts: Bool
) raises -> Int:
    """How many classes can share one threadgroup's partial histogram.

    Each class needs a gradient and a hessian bin plane; the count plane is
    shared or per class with them. The result is clamped to
    `SHARED_CLASS_CAP`, which is what the kernel statically allocates: a
    device with more shared memory than the cap assumes does not get a wider
    kernel by arithmetic here.
    """
    if n_bins < 1 or n_bins > MAX_BINS:
        raise Error("bin count out of range")
    if shared_bytes_per_block < 1:
        raise Error("shared memory per block must be positive")
    var per_class = 2 * n_bins * BYTES_PER_HIST_CELL
    var fixed = 0
    if shared_counts:
        fixed = n_bins * BYTES_PER_HIST_CELL
    else:
        per_class += n_bins * BYTES_PER_HIST_CELL
    if shared_bytes_per_block <= fixed:
        raise Error(
            "device shared memory too small for a batched partial histogram"
        )
    var fits = (shared_bytes_per_block - fixed) // per_class
    if fits < 1:
        raise Error(
            "device shared memory too small for a batched partial histogram"
        )
    if fits > SHARED_CLASS_CAP:
        fits = SHARED_CLASS_CAP
    return fits


@fieldwise_init
struct ClassBatchPlan(Copyable, Movable):
    """How one round's classes are grouped, and what each group costs.

    A plan is a partition of `0 .. n_classes-1` into contiguous ascending
    runs of at most `batch_size`, plus the two facts a kernel launch needs
    (`per_block`, `shared_counts`) and the byte total a caller can check
    against its own budget. It is a value: forming it allocates nothing and
    touches no device.
    """

    var n_classes: Int
    var batch_size: Int
    """Classes resident in device memory at once. Bounded by the memory
    budget."""
    var per_block: Int
    """Classes sharing one threadgroup's bin reads. 1 when the batch is not
    eligible for bin sharing. Bounded by shared memory and
    `SHARED_CLASS_CAP`."""
    var shared_counts: Bool
    var shared_rows: Bool
    var bytes_per_batch: Int

    def check(self) raises:
        if self.n_classes < 2:
            raise Error("a class batch plan needs at least two classes")
        if self.batch_size < 1 or self.batch_size > self.n_classes:
            raise Error("batch size out of range")
        if self.per_block < 1 or self.per_block > self.batch_size:
            raise Error("classes per block out of range")
        if self.shared_counts and not self.shared_rows:
            raise Error(
                "counts can only be shared across classes reading the same"
                " rows"
            )

    @always_inline
    def n_batches(self) -> Int:
        return _ceil_div(self.n_classes, self.batch_size)

    def batch_begin(self, b: Int) raises -> Int:
        """First class id of batch `b`. Batches are ascending and
        contiguous, which is the whole ordering guarantee."""
        if b < 0 or b >= self.n_batches():
            raise Error("batch index out of range")
        return b * self.batch_size

    def batch_count(self, b: Int) raises -> Int:
        """Classes in batch `b`. Only the last batch is short."""
        var begin = self.batch_begin(b)
        var left = self.n_classes - begin
        if left < self.batch_size:
            return left
        return self.batch_size

    def class_at(self, b: Int, slot: Int) raises -> Int:
        """The class id a batch slot carries. The only batch-slot-to-class
        mapping there is, and it is monotone in `(b, slot)`, so results
        collected by ascending slot are results in ascending class order."""
        var count = self.batch_count(b)
        if slot < 0 or slot >= count:
            raise Error("batch slot out of range")
        return self.batch_begin(b) + slot

    def bin_passes_per_round(self) -> Int:
        """How many times a round reads the binned matrix at the level the
        plan applies to. `n_classes` sequentially; `ceil(n_classes /
        per_block)` when classes share threadgroups. This is the number the
        benchmark has to move."""
        return _ceil_div(self.n_classes, self.per_block)

    def launches_per_level(self) -> Int:
        """Histogram launches one level of the round costs: one per batch,
        whether or not bin reads are shared."""
        return self.n_batches()

    def is_sequential(self) -> Bool:
        """True when the plan degenerates to the existing one-class-at-a-time
        loop, which is what a memory-starved device or an ineligible level
        resolves to. A caller may then skip the batched path entirely rather
        than pay its bookkeeping for no sharing."""
        return self.batch_size == 1


def env_class_batch() -> Int:
    """`MOJOTREES_GPU_CLASS_BATCH`: force a batch size, 0 for auto. Matches
    the `MOJOTREES_` override contract in `parallel.mojo` and
    `gpu_tiling.mojo`, and exists for the same reason: a benchmark has to be
    able to pin the geometry a measurement was taken at."""
    return _env_int("MOJOTREES_GPU_CLASS_BATCH", 0)


def env_class_batch_budget() -> Int:
    """`MOJOTREES_GPU_CLASS_BATCH_BYTES`, defaulting to
    `CLASS_BATCH_BUDGET_BYTES`."""
    return _env_int(
        "MOJOTREES_GPU_CLASS_BATCH_BYTES", CLASS_BATCH_BUDGET_BYTES
    )


def plan_class_batches(
    n_classes: Int,
    n_rows: Int,
    n_features: Int,
    n_bins: Int,
    n_tiles: Int,
    shared_bytes_per_block: Int,
    eligibility: BatchEligibility,
    budget_bytes: Int = 0,
    requested_batch: Int = 0,
) raises -> ClassBatchPlan:
    """Group a round's classes into batches that fit the device.

    `budget_bytes` of 0 reads `MOJOTREES_GPU_CLASS_BATCH_BYTES` and then the
    default budget; `requested_batch` of 0 reads `MOJOTREES_GPU_CLASS_BATCH`
    and then derives the batch from the budget. A forced batch is still
    clamped to `n_classes`, and still has to fit: forcing a size the device
    cannot hold raises rather than silently shrinking, because a benchmark
    that asked for a geometry must not be handed a different one.

    The bound is one division. Per-class cost is affine in the batch size
    (`class_batch_bytes(K) = a * K + b`, with `b` the shared count plane and
    the shared row indices), so the largest admissible `K` is
    `(budget - b) / a`, and the loop below just walks it down from
    `n_classes` rather than inverting the formula in closed form; the class
    counts this runs at are at most a few thousand.
    """
    if n_classes < 2:
        raise Error("multiclass batching needs at least two classes")
    if n_rows < 1:
        raise Error("class batching needs at least one row")
    if n_features < 1:
        raise Error("class batching needs at least one feature")
    if n_bins < 1 or n_bins > MAX_BINS:
        raise Error("bin count out of range")
    if n_tiles < 1:
        raise Error("tile count must be positive")

    var budget = budget_bytes
    if budget <= 0:
        budget = env_class_batch_budget()
    if budget <= 0:
        budget = CLASS_BATCH_BUDGET_BYTES

    var shared_counts = eligibility.counts_shared()
    var shared_rows = eligibility.shared_rows

    var per_block = 1
    if eligibility.bin_reads_shared():
        per_block = classes_per_block(
            n_bins, shared_bytes_per_block, shared_counts
        )

    var forced = requested_batch
    if forced <= 0:
        forced = env_class_batch()

    var batch: Int
    if forced > 0:
        batch = forced
        if batch > n_classes:
            batch = n_classes
        var cost = class_batch_bytes(
            n_rows,
            n_features,
            n_bins,
            n_tiles,
            batch,
            shared_counts,
            shared_rows,
        )
        if cost > budget:
            raise Error(
                "the forced class batch does not fit the device budget;"
                " raise MOJOTREES_GPU_CLASS_BATCH_BYTES or lower"
                " MOJOTREES_GPU_CLASS_BATCH"
            )
    else:
        batch = n_classes
        while batch > 1:
            var cost = class_batch_bytes(
                n_rows,
                n_features,
                n_bins,
                n_tiles,
                batch,
                shared_counts,
                shared_rows,
            )
            if cost <= budget:
                break
            batch -= 1

    # A batch narrower than a threadgroup's class capacity caps the sharing:
    # a block cannot serve a class that is not resident.
    if per_block > batch:
        per_block = batch

    var bytes = class_batch_bytes(
        n_rows, n_features, n_bins, n_tiles, batch, shared_counts, shared_rows
    )
    var plan = ClassBatchPlan(
        n_classes, batch, per_block, shared_counts, shared_rows, bytes
    )
    plan.check()
    return plan^


@always_inline
def round_tree_slot(round: Int, k: Int, n_classes: Int) -> Int:
    """Where class `k`'s tree of `round` sits in the serialized ensemble:
    `trees[round * n_classes + k]`, round-major, which is what
    `MulticlassBooster` documents, what `flatten_multiclass` uploads, and
    what `gpu_predict._predict_kernel` indexes. Batching changes when a tree
    is grown, never where it is stored."""
    return round * n_classes + k


@always_inline
def tree_feature_seed(round: Int, k: Int, n_classes: Int) -> Int:
    """The seed the grower draws class `k`'s feature subsample with. The
    same integer as `round_tree_slot`, and deliberately so: it is the
    absolute tree index, so a batched run draws exactly the feature sets a
    sequential run of the same ensemble would have drawn, and a continued run
    draws what an uninterrupted one would have."""
    return round * n_classes + k


def check_batch_order(plan: ClassBatchPlan) raises:
    """Assert the plan enumerates every class exactly once, in ascending
    order, across its batches.

    This is the invariant every ordering guarantee in this lane rests on, and
    it is cheap enough to check at plan time rather than to argue about: walk
    the batches in order, walk each batch's slots in order, and require the
    class ids to be `0, 1, ... n_classes-1` with no gap and no repeat.
    """
    var expected = 0
    for b in range(plan.n_batches()):
        var count = plan.batch_count(b)
        if count < 1:
            raise Error("a class batch must not be empty")
        for slot in range(count):
            if plan.class_at(b, slot) != expected:
                raise Error(
                    "class batches must enumerate classes in ascending order"
                )
            expected += 1
    if expected != plan.n_classes:
        raise Error("class batches must cover every class exactly once")
