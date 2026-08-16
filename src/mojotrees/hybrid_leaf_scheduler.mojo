"""Where one leaf's histogram is accumulated: the device that owns the rows,
or the host that already owns the bins.

The GPU grower builds every node's histogram on the device and downloads it
(`build_leaf` in histogram_gpu.mojo). The cost of that download does not
depend on the node: `download_raw` copies the whole
`3 * n_features * n_bins` fixed-point buffer and synchronizes, whether the
node owns a million rows or four. The cost of accumulating does depend on the
node, and on the host it is `node_rows * n_active_features` scattered adds
over a binned matrix the trainer is already holding in host memory. Deep in a
leaf-wise tree most nodes are small, so a hybrid grower would keep the large
leaves on the device and let the host build the small ones, paying the
transfer only for the row ids it needs.

This module decides which, and refuses to decide it from reasoning alone.

**Default off, and off for a stated reason.** Nothing in this repository has
measured a host histogram build against a device one on any hardware, so
`HybridCosts.unmeasured()` is the only cost model this module can construct
and it declines every leaf. That mirrors `crossover_rules()` in
device_policy.mojo and `CrossoverInputs.min_cells` in apple_gpu_policy.mojo:
the inputs a measurement would key on are reported here so the benchmark has
somewhere to put its answer, and the answer is not invented. See
`docs/design/HYBRID_TRAINING.md` for the experiments that would fill it in.

The three things that make this decidable at all
------------------------------------------------
1. **The binned matrix is already host-resident.** `train_gpu` holds the
   caller's `BinnedMatrix` for the whole fit (it walks it per round in
   `tree.predict_row`), so a host build reads bins at no transfer cost. The
   device copy is a second copy, not the only one.
2. **The parent histogram is already host-resident** on the default split
   path. `grow_tree_gpu` downloads every node's histogram and subtracts
   host-side, so a host-built child's sibling comes out of the arithmetic the
   grower already performs. Under `SPLIT_SEARCH_DEVICE` it does not: both
   children are built and searched on the device and no histogram crosses,
   so there is no host parent to subtract from and this module declines
   every leaf (`DECLINE_NO_HOST_PARENT`).
3. **Only the row ids are device-owned.** Gradients and hessians are host
   lists on the `upload_gradients` path. What a host build needs and does not
   have is the node's slice of the active-row permutation. On the
   device-objective path (`fill_gradients_device`) the gradients never exist
   on the host at all, and pulling them back per round would undo that path's
   whole purpose, so this module declines there too
   (`DECLINE_GRADIENTS_ON_DEVICE`).

How the host gets the rows
--------------------------
The obvious answer -- read back this node's `4 * node_rows` bytes -- is not
available. `DeviceContext.enqueue_copy` copies the *whole* source buffer, so
a sub-range readback needs either a device buffer allocated per call or an
API this project does not have; `download_rows` therefore moves `4 * n_rows`
whatever the node's size, which would make a per-node readback cost more than
the download it was meant to avoid.

So the design does not use one. The host takes **one whole-permutation
snapshot per tree** (`snapshot_nanos`) and maintains it itself:

- Every node alive when the snapshot was taken has its rows at
  `snapshot[begin : end]`, for free -- the ranges are already host-tracked in
  `LeafRangeTable`.
- A split re-partitions `rows_dev[parent.begin : parent.end]` and provably
  touches nothing outside it (gpu_active_rows.mojo states this invariant), so
  the host keeps the snapshot valid by mirroring that one partition on its own
  copy with `partition_range_host` -- the same stable, buffer-order rule, which
  agrees with the device index for index. Cost: one partition over the
  parent's rows (`host_materialize_nanos`), and it materializes *both*
  children at once.
- The root needs no snapshot at all: unbagged its rows are the identity
  permutation, bagged they are the host's own bag.

The snapshot is charged to the one leaf that first elects to use it, in full
(`DECLINE_SNAPSHOT_NOT_PAID` when that leaf's saving cannot cover it). Every
later host leaf in the tree reads it for nothing. That is deliberately
conservative -- it can never make a tree slower than the pure-device path,
whereas amortizing over an expected leaf count would be a guess about a
distribution nobody has measured.

Substitutability: what a host build has to reproduce
----------------------------------------------------
A host histogram is interchangeable with a device one only if it is the same
number. It is not, by default: the device accumulates fixed-point Int32 from
Float32 gradients scaled by the round's `g_scale`/`h_scale` and dequantizes
on download, while `build_histogram_subset_into` sums Float64 directly. The
two agree to Float32 precision, which is enough for a benchmark and not
enough for a substitution -- a tree grown from a mixture of the two is a
third tree, different from both homogeneous ones.

So the module names four modes and only one of them substitutes:

- `MODE_OFF` (default) -- every leaf goes to the device. No behavior change
  of any kind.
- `MODE_MIRROR` -- the host build runs *in addition to* the device build and
  its output is compared; the device's histogram is the one used. Strictly
  slower, and the only mode that can establish the bitwise claim below.
- `MODE_REPLICA` -- the host build substitutes, and must go through the same
  fixed-point pipeline: convert each gradient to Float32 exactly as
  `stage_gradients` does, multiply by the round's Float32 scale, round to
  Int32, accumulate in Int32 (integer addition is associative, so the
  accumulation order cannot matter), and dequantize by the same `1 / scale`.
  Available only once `MODE_MIRROR` has shown the two agree bit for bit on
  the target hardware, which is what `quantized_replica_verified` records.
  The replica kernel is not in this lane: this module says what it must be,
  and declines until a caller declares it verified.
- `MODE_HOST_FLOAT64` -- the host build substitutes with plain Float64
  accumulation. This changes the fit. It is named so it can be benchmarked
  as the deliberately different algorithm it is, never enabled as an
  optimization of the shipped one.

Determinism
-----------
Two rules, both structural rather than advisory:

- **Placement is a pure function of its arguments.** `place_leaf` reads node
  row counts, dataset shape, active feature count, launch counts, the
  snapshot flag, and the cost coefficients. It reads no clock, no queue
  depth, and no completion signal. The snapshot flag is the only argument
  that is tree state rather than tree shape, and it advances with the
  grower's own deterministic node order. Two runs on one machine, and a run
  with the host build racing the device against one where it does not, place
  every leaf identically.
- **Completion order is not consumption order.** `WaveBarrier` is the
  bookkeeping that enforces it: the grower announces the nodes of a wave in
  the order it will consume them, the builds complete in whatever order the
  hardware delivers, and `announced()` returns the announcement order
  regardless. `require_ready` refuses a commit while any build is
  outstanding, so no gain comparison can be made from a partial frontier.
  Between them, which device finished first cannot reach the best-gain scan,
  the tie-break, or the chosen split.

The three answers, and where each comes from
--------------------------------------------
A node's histogram is read, built on the host, or built on the device, and
this module returns exactly one of those:

- **Reuse** (`PLACE_REUSE`) is decided first and without the cost model,
  because a build that does not happen cannot lose a comparison. It is
  decided from a `ReuseOffer` the caller makes, never from a search here:
  the sibling of a committed split, which both growers already subtract, or
  a cache entry `histogram_cache_policy.HistogramCache.lookup` returned
  fresh. `plan_split` states the sibling's placement rather than leaving it
  implied, which is what makes a split plan cover the whole split.
- **Host** and **device** are the comparison below, unchanged, and today it
  always answers device for want of a measured cost model.

None of the three changes a split. Reuse hands back the histogram the grower
would have produced; a host build substitutes only in a mode that says so;
and the decision is a pure function of counts, so it cannot depend on which
device finished first.

Where the launch count comes from
---------------------------------
`LeafWork.gpu_launches` is what the device path costs before it touches a
row, and it depends on the strategy the geometry resolved to: one launch for
the atomic path, two for the tiled one. This module does not know that rule
and must not learn it. `LeafWork.for_tiling` takes a resolved
`gpu_tiling.HistogramTiling` and reads the count off it, and a caller holding
an `apple_histogram_policy.HistogramPlan` passes `plan.tiling()`. That is the
whole of this module's coupling to the launch policy: it charges for the
launches the policy will issue, and decides nothing about them.

What this module does not do
----------------------------
It does not race the host against the device. A placement is a
*substitution*: the leaf is built on the host instead of the device, never
on both for throughput (`MODE_MIRROR` builds on both, but for comparison,
and it is documented as strictly slower). That is a design choice and not
an omission. On unified memory the two share one memory bus, so a host
build running concurrently with a bandwidth-bound device accumulation
competes for the same bytes per second; the only work concurrency could
add is the launch-bound small-leaf tail, which is exactly what the
substitution already captures without the contention. The overlap
experiment and the constraint on it are HYBRID_TRAINING.md §9 E5.

It does not accumulate a histogram, own a buffer, touch a device, or edit a
grower. It does not decide CPU-versus-GPU *training* -- that is
`device_policy.mojo`, and a hybrid leaf schedule presupposes the GPU trainer
was already chosen. It does not decide what is reusable; that argument is
made once, in `histogram_cache_policy.mojo`, along with histogram lifetimes,
cache keys, and invalidation.
"""

from std.os import getenv

from .gpu_tiling import HistogramTiling
from .histogram_cache_policy import (
    FRESH,
    ORIGIN_CPU_FLOAT64,
    ORIGIN_CPU_REPLICA,
    ORIGIN_GPU_FIXED,
    ORIGIN_SUBTRACTED,
    ORIGIN_UNKNOWN,
    STALE_ABSENT,
    fixed_download_bytes,
    histogram_cells,
    origin_name,
    origins_are_subtractable,
    staleness_name,
)
from .parallel import _env_int


# --- Modes ----------------------------------------------------------------

comptime MODE_OFF = 0
comptime MODE_MIRROR = 1
comptime MODE_REPLICA = 2
comptime MODE_HOST_FLOAT64 = 3
comptime N_MODES = 4


def mode_name(mode: Int) -> String:
    if mode == MODE_MIRROR:
        return String("mirror")
    if mode == MODE_REPLICA:
        return String("replica")
    if mode == MODE_HOST_FLOAT64:
        return String("float64")
    return String("off")


def mode_substitutes(mode: Int) -> Bool:
    """Whether a host build in this mode *replaces* the device build.

    `MODE_MIRROR` does not: it builds on both and uses the device's, which
    is why it is the only mode that can be enabled before the bitwise claim
    is established.
    """
    return mode == MODE_REPLICA or mode == MODE_HOST_FLOAT64


def mode_preserves_fit(mode: Int) -> Bool:
    """Whether this mode grows the same tree the pure-GPU grower grows.

    True for `MODE_OFF` by construction and for `MODE_MIRROR` because the
    device's histogram is the one consumed. True for `MODE_REPLICA` only
    under its verification precondition, which is a property of the run
    rather than of the mode, so this returns False for it: a caller that has
    verified the replica knows more than this function does and says so
    through `HybridContext.quantized_replica_verified`.
    """
    return mode == MODE_OFF or mode == MODE_MIRROR


def host_origin_for(mode: Int) raises -> Int:
    """The provenance a host build carries in this mode, for
    `HistogramCache.admit`."""
    if mode == MODE_REPLICA:
        return ORIGIN_CPU_REPLICA
    if mode == MODE_HOST_FLOAT64:
        return ORIGIN_CPU_FLOAT64
    raise Error("this mode produces no substitutable host histogram")


def env_hybrid_mode() -> Int:
    """`MOJOTREES_HYBRID_LEAVES`, following the `MOJOTREES_` contract in
    parallel.mojo: `mirror`, `replica`, `float64`, or `off` (the default,
    and the value of anything unrecognized)."""
    var s = getenv("MOJOTREES_HYBRID_LEAVES")
    if s == "mirror":
        return MODE_MIRROR
    if s == "replica":
        return MODE_REPLICA
    if s == "float64":
        return MODE_HOST_FLOAT64
    return MODE_OFF


def env_hybrid_costs() raises -> HybridCosts:
    """The cost model `MOJOTREES_HYBRID_COSTS` selects: `apple-m4` for the
    calibrated Apple M4 coefficients, anything else (the default included)
    for the unmeasured model that declines every leaf.

    A separate switch from `MOJOTREES_HYBRID_LEAVES` on purpose: the mode
    says what a host build would be allowed to do, the costs say whether one
    is worth running, and a run that sets the mode without the costs gets
    the observable `DECLINE_COSTS_UNMEASURED` rather than a placement made
    from another machine's numbers."""
    var s = getenv("MOJOTREES_HYBRID_COSTS")
    if s == "apple-m4":
        return HybridCosts.apple_m4()
    return HybridCosts.unmeasured()


# --- The replica claim, as builder state ----------------------------------
#
# `GpuHistogramBuilder.replica_state` records whether the host fixed-point
# replica has been shown to reproduce that device's histograms bit for bit.
# The grower's mirror comparison sets it; `MODE_REPLICA` substitutes only at
# `REPLICA_VERIFIED`, and `REPLICA_REFUTED` retires hybrid scheduling for
# the rest of the fit. Held on the builder rather than here so one fit
# verifies once, and so a placement stays a pure function of its arguments.

comptime REPLICA_UNTESTED = 0
comptime REPLICA_VERIFIED = 1
comptime REPLICA_REFUTED = 2


# --- Placement and refusals -----------------------------------------------

comptime PLACE_GPU = 0
comptime PLACE_CPU = 1
comptime PLACE_BOTH = 2
comptime PLACE_REUSE = 3
"""Neither device builds this node: a histogram that is already the right
answer is read instead. The two cases are the sibling of a committed split,
which both growers already subtract, and a cache entry
`histogram_cache_policy.HistogramCache.lookup` returned as fresh. Reuse is
the cheapest placement by construction and so is decided before the cost
comparison, but only from a fact the caller reports; nothing here searches
for reuse, and `histogram_cache_policy.mojo` is where what is reusable at all
is argued."""


def placement_name(place: Int) -> String:
    if place == PLACE_CPU:
        return String("cpu")
    if place == PLACE_BOTH:
        return String("both")
    if place == PLACE_REUSE:
        return String("reuse")
    return String("gpu")


comptime ACCEPTED = 0
comptime DECLINE_MODE_OFF = 1
comptime DECLINE_NO_HOST_PARENT = 2
comptime DECLINE_GRADIENTS_ON_DEVICE = 3
comptime DECLINE_REPLICA_UNVERIFIED = 4
comptime DECLINE_COSTS_UNMEASURED = 5
comptime DECLINE_NO_HOST_BINS = 6
comptime DECLINE_EMPTY_NODE = 7
comptime DECLINE_TRANSFER_DOMINATES = 8
comptime DECLINE_SLOWER_ON_HOST = 9
comptime DECLINE_SNAPSHOT_NOT_PAID = 10
comptime REUSED_FRESH_HISTOGRAM = 11
comptime N_DECLINE_REASONS = 12


def decline_name(reason: Int) -> String:
    if reason == ACCEPTED:
        return String("accepted")
    if reason == DECLINE_MODE_OFF:
        return String("hybrid-scheduling-off")
    if reason == DECLINE_NO_HOST_PARENT:
        return String("device-split-search-keeps-no-host-parent")
    if reason == DECLINE_GRADIENTS_ON_DEVICE:
        return String("gradients-are-device-resident")
    if reason == DECLINE_REPLICA_UNVERIFIED:
        return String("fixed-point-replica-not-verified")
    if reason == DECLINE_COSTS_UNMEASURED:
        return String("no-measured-cost-model")
    if reason == DECLINE_NO_HOST_BINS:
        return String("binned-matrix-not-host-resident")
    if reason == DECLINE_EMPTY_NODE:
        return String("node-owns-no-rows")
    if reason == DECLINE_TRANSFER_DOMINATES:
        return String("state-transfer-costs-more-than-the-histogram")
    if reason == DECLINE_SLOWER_ON_HOST:
        return String("modelled-slower-on-the-host")
    if reason == DECLINE_SNAPSHOT_NOT_PAID:
        return String("first-host-leaf-cannot-pay-for-the-snapshot")
    if reason == REUSED_FRESH_HISTOGRAM:
        return String("reused-a-histogram-that-is-already-the-answer")
    return String("unknown")


# --- Where a node's rows come from ----------------------------------------

comptime ROWS_HOST_IDENTITY = 0
comptime ROWS_HOST_BAG = 1
comptime ROWS_HOST_MIRROR = 2
comptime ROWS_HOST_SNAPSHOT = 3
comptime ROWS_DEVICE_COPY = 4
comptime N_ROW_SOURCES = 5


def row_source_name(source: Int) -> String:
    if source == ROWS_HOST_IDENTITY:
        return String("host-identity")
    if source == ROWS_HOST_BAG:
        return String("host-bag")
    if source == ROWS_HOST_MIRROR:
        return String("host-mirror")
    if source == ROWS_HOST_SNAPSHOT:
        return String("host-snapshot")
    return String("device-copy")


def row_source_is_host(source: Int) -> Bool:
    """Whether the host already holds this node's row ids, so a host build
    moves nothing *for this node*.

    Four cases do. The unbagged root's rows are the identity permutation,
    which `begin_tree` writes with a kernel and the host can reproduce
    without reading anything. A bagged root's rows are the bag, which the
    host drew and handed to `begin_tree` in that order. A node whose
    ancestor is already host-materialized has its rows derived by re-running
    the same stable partition (`partition_range_host` in
    gpu_active_rows.mojo is that reference model, index for index), which is
    `ROWS_HOST_MIRROR`. And `ROWS_HOST_SNAPSHOT` reads a whole-permutation
    snapshot the host already took, which costs this node nothing because
    the snapshot is a per-tree cost (see `snapshot_nanos`).

    `ROWS_DEVICE_COPY` is the only source that charges a transfer to the
    node itself: a per-range readback of exactly the node's window
    (`GpuHistogramBuilder.readback_range`, a `create_sub_buffer` view plus
    one `enqueue_copy` and one synchronize). The design was written believing
    that readback was not expressible and built the snapshot sources around
    that; it is, and the GPU grower plans every leaf as `ROWS_DEVICE_COPY`,
    which charges each leaf its own `4 * node_rows` bytes and nothing
    amortized across the tree.
    """
    return source != ROWS_DEVICE_COPY


def row_transfer_bytes(source: Int, node_rows: Int) raises -> Int:
    """Device-to-host bytes a host build of this node must move first.

    Row ids are Int32 on the device (`rows_dev` in gpu_active_rows.mojo), so
    a device copy is four bytes per row and nothing else: the bins are
    already on the host, the gradients are already on the host on the path
    this module accepts, and the histogram never leaves the host once it is
    built there.
    """
    if node_rows < 0:
        raise Error("row count must be nonnegative")
    if source < 0 or source >= N_ROW_SOURCES:
        raise Error("unknown row source")
    if row_source_is_host(source):
        return 0
    return 4 * node_rows


# --- Cost model -----------------------------------------------------------

comptime _NANOS_UNMEASURED = 0


def _ceil_div(a: Int, b: Int) -> Int:
    return (a + b - 1) // b


struct HybridCosts(Copyable, Movable):
    """Machine coefficients a placement decision needs, and the citation for
    where they were measured.

    All rates are nanoseconds per unit of a *thousand* somethings, so the
    arithmetic stays in integers and a decision is reproducible bit for bit
    on any machine that reads the same coefficients. Nothing here is a
    threshold: a threshold says "a node under N rows goes to the host", and
    what these say is how long each path takes, from which the comparison
    follows. That distinction is the reason this module can ship a cost
    model at all while `crossover_rules()` ships no rules.

    `measured` is the gate. `unmeasured()` is the only constructor this
    repository can honestly call today, and every placement it produces is
    `PLACE_GPU` with `DECLINE_COSTS_UNMEASURED`.
    """

    var launch_nanos: Int
    """Fixed cost of enqueuing and running one histogram kernel, exclusive
    of the row work: the term a four-row leaf pays in full.

    Fixed is an approximation on Metal, and it fails in one direction. Every
    `enqueue_function` there becomes its own command buffer and the queue
    holds 64 of them, with no way to raise it (measured by disassembly,
    `docs/GPU_PORTABILITY.md` section 6.2). While the queue has room the
    enqueue half is nearly free; once it is full the host blocks until an
    older buffer completes. So a coefficient calibrated on a short launch
    stream *underestimates* the enqueue cost of a long one, which biases a
    placement toward the device on exactly the launch-bound tail this module
    exists to catch. Calibrate it against a stream of realistic length, and
    treat a measured `launch_nanos` as valid only for streams no longer than
    the one it was fitted on."""

    var sync_nanos: Int
    """Host round trip for one `ctx.synchronize()`. Paid once per node by
    the device path (`download_raw`) and once by a host build whose rows
    have to be copied back, so it cancels except where the row source is
    host-side."""

    var transfer_nanos_per_kib: Int
    """Device-to-host copy rate through the pinned staging buffers, in
    nanoseconds per KiB. One rate for both directions of interest here: the
    fixed-point histogram download and the row-id readback go through the
    same path.

    A rate per KiB is the byte term only. On Metal the wait is the larger
    half and it does not scale with bytes: `enqueue_copy` drains the whole
    queue before it moves anything, in either direction (measured by
    disassembly, `docs/GPU_PORTABILITY.md` section 6.1). `sync_nanos` below
    is where that wait is charged, and a calibration must not fold it into
    this rate or a small transfer will look nearly free."""

    var device_nanos_per_krow_slot: Int
    """Device accumulation, per thousand (row, active-feature) pairs."""

    var host_nanos_per_krow_slot: Int
    """Host accumulation, per thousand (row, active-feature) pairs, over a
    *contiguous* node (the root): the scattered-add inner loop of
    `_accumulate_subset` with its bin reads streaming."""

    var host_scatter_nanos_per_krow_slot: Int
    """The same accumulation over a *scattered* node: rows far enough apart
    that, in the feature-major bin layout (`bin_at` reads
    `bins[feature * n_rows + row]`), every bin read is a fresh cache line
    and the prefetcher has nothing to follow. The calibration measures
    leaves at several row gaps and this is the *largest* rate it saw
    (memory-latency bound, roughly five times the contiguous rate on the
    calibration machine). A small leaf of a large dataset looks like this
    to the host and not like the root, which is why the contiguous rate
    alone misplaced such leaves; `host_slot_nanos_per_k` ramps between the
    two by the node's mean row gap."""

    var host_fixed_nanos: Int
    """What a host build costs before it touches a row: the quantization
    dispatch, the per-feature task fan-out, and the dequantization pass
    (`histogram.build_histogram_subset_replica_into`). Measured as a
    one-row build less the zeroing pass. Tens of microseconds on the
    calibration machine, which is a real fraction of the device's fixed
    cost and is what stops a tiny leaf from being modelled as free."""

    var host_partition_nanos_per_krow: Int
    """Host mirror of one split's stable partition, per thousand rows of the
    node being partitioned. This is what materializing a node's rows costs
    once a snapshot exists: `partition_range_host`'s flag/prefix/scatter over
    the parent's own range, and nothing else."""

    var host_zero_nanos_per_kcell: Int
    """Host zeroing pass, per thousand histogram cells. Paid by every host
    build (`Histogram.reset`) and independent of the node's rows, which is
    what stops a tiny leaf from being free on the host."""

    var convert_nanos_per_kcell: Int
    """Fixed-point to Float64 conversion after a download
    (`histogram_from_host`), per thousand cells. Paid by every device build
    and likewise independent of the node's rows."""

    var measured: Bool
    var evidence_id: String
    """Where the numbers came from: a benchmark file, a document section, a
    commit. A measured cost model without one is refused, exactly as
    `CrossoverEvidence` refuses a rule without a citation."""

    var measured_on: String

    def __init__(out self):
        """The unmeasured model: every coefficient zero, `measured` False.

        The no-argument form is the *only* one that can produce an
        unmeasured cost model, and the argument-taking one below is the only
        one that can produce a measured one, so there is no path to a set of
        coefficients without a citation.
        """
        self.launch_nanos = _NANOS_UNMEASURED
        self.sync_nanos = _NANOS_UNMEASURED
        self.transfer_nanos_per_kib = _NANOS_UNMEASURED
        self.device_nanos_per_krow_slot = _NANOS_UNMEASURED
        self.host_nanos_per_krow_slot = _NANOS_UNMEASURED
        self.host_scatter_nanos_per_krow_slot = _NANOS_UNMEASURED
        self.host_fixed_nanos = _NANOS_UNMEASURED
        self.host_partition_nanos_per_krow = _NANOS_UNMEASURED
        self.host_zero_nanos_per_kcell = _NANOS_UNMEASURED
        self.convert_nanos_per_kcell = _NANOS_UNMEASURED
        self.measured = False
        self.evidence_id = String("")
        self.measured_on = String("")

    def __init__(
        out self,
        launch_nanos: Int,
        sync_nanos: Int,
        transfer_nanos_per_kib: Int,
        device_nanos_per_krow_slot: Int,
        host_nanos_per_krow_slot: Int,
        host_scatter_nanos_per_krow_slot: Int,
        host_fixed_nanos: Int,
        host_partition_nanos_per_krow: Int,
        host_zero_nanos_per_kcell: Int,
        convert_nanos_per_kcell: Int,
        var evidence_id: String,
        var measured_on: String,
    ) raises:
        if evidence_id.byte_length() == 0:
            raise Error(
                "a measured cost model needs an evidence identifier; cite"
                " the benchmark that produced these coefficients"
            )
        if (
            launch_nanos < 0
            or sync_nanos < 0
            or transfer_nanos_per_kib < 0
            or device_nanos_per_krow_slot < 0
            or host_nanos_per_krow_slot < 0
            or host_scatter_nanos_per_krow_slot < 0
            or host_fixed_nanos < 0
            or host_partition_nanos_per_krow < 0
            or host_zero_nanos_per_kcell < 0
            or convert_nanos_per_kcell < 0
        ):
            raise Error("cost coefficients must be nonnegative")
        if host_scatter_nanos_per_krow_slot < host_nanos_per_krow_slot:
            raise Error(
                "a scattered node cannot accumulate faster than a contiguous"
                " one; the two host rates are swapped or mismeasured"
            )
        self.launch_nanos = launch_nanos
        self.sync_nanos = sync_nanos
        self.transfer_nanos_per_kib = transfer_nanos_per_kib
        self.device_nanos_per_krow_slot = device_nanos_per_krow_slot
        self.host_nanos_per_krow_slot = host_nanos_per_krow_slot
        self.host_scatter_nanos_per_krow_slot = host_scatter_nanos_per_krow_slot
        self.host_fixed_nanos = host_fixed_nanos
        self.host_partition_nanos_per_krow = host_partition_nanos_per_krow
        self.host_zero_nanos_per_kcell = host_zero_nanos_per_kcell
        self.convert_nanos_per_kcell = convert_nanos_per_kcell
        self.measured = True
        self.evidence_id = evidence_id^
        self.measured_on = measured_on^

    @staticmethod
    def unmeasured() -> HybridCosts:
        """The cost model for a machine nobody has calibrated.

        Every coefficient is zero and `measured` is False, so no comparison
        is attempted and every leaf stays on the device. The only measured
        alternative is `apple_m4` below, and adding another must cite a
        benchmark the same way; see `docs/design/HYBRID_TRAINING.md`.
        """
        return HybridCosts()

    @staticmethod
    def apple_m4() raises -> HybridCosts:
        """Experiment E1 of docs/design/HYBRID_TRAINING.md §9, run on an
        Apple M4: `pixi run bench-hybrid-costs` at 500k rows x 50 features
        x 255 bins, minimum over five interleaved trials, 2026-08-15 (the
        2026-08-14 run lacked the scattered-leaf rate; all coefficients
        come from one run because this machine's timings drift between
        time windows). Full output in the evidence file.

        Selected only by an explicit `MOJOTREES_HYBRID_COSTS=apple-m4`
        (see `env_hybrid_costs`), never inferred from the hardware: on any
        other machine these numbers are a guess, and a guessed cost model
        can misplace work even though it can never change a tree.
        """
        return HybridCosts(
            62128,  # launch_nanos
            20560,  # sync_nanos
            334,  # transfer_nanos_per_kib
            160,  # device_nanos_per_krow_slot
            1178,  # host_nanos_per_krow_slot
            5516,  # host_scatter_nanos_per_krow_slot
            41280,  # host_fixed_nanos
            15374,  # host_partition_nanos_per_krow
            11,  # host_zero_nanos_per_kcell
            10024,  # convert_nanos_per_kcell
            String("bench/results/apple_m4_hybrid_costs_2026-08-15.md"),
            String("Apple M4, macOS, 500k x 50 x 255"),
        )

    def cite(self) -> String:
        if not self.measured:
            return String("unmeasured")
        return String(self.evidence_id, " on ", self.measured_on)


@fieldwise_init
struct LeafWork(Copyable, Movable):
    """The shape of one node's histogram build. Every field is a count the
    grower already has; none of them is a timing."""

    var node: Int

    var node_rows: Int
    """Rows this node owns. The GPU grower has it exactly, from the parent
    histogram's integer counts (`_count_left` in train_gpu.mojo), before the
    partition has run."""

    var n_slots: Int
    """Active features, which is what either device actually accumulates:
    `len(builder.active)` after `set_features`."""

    var n_features: Int
    """The dataset's full feature count, which is what a download costs:
    `out_dev` is allocated and copied at the full shape whatever the active
    set is."""

    var n_bins: Int

    var dataset_rows: Int
    """The binned matrix's column stride. It decides how far apart a host
    build's bin reads land, which is the difference between a streaming scan
    and one cache miss per read; `host_slot_nanos_per_k` prices that."""

    var row_source: Int

    var materialize_rows: Int
    """Rows the host must re-partition to bring this node's row ids into
    existence on its own copy of the permutation, or 0 when they are already
    there.

    Zero for a node that was alive when the snapshot was taken (its rows are
    `snapshot[begin : end]` already) and for the root under either host-known
    seeding. Non-zero for a node created by a split the host has not yet
    mirrored, where the cost is one `partition_range_host` over the *parent's*
    row count — so this field carries the parent's rows, not this node's. See
    `host_materialize_nanos`.
    """

    var gpu_launches: Int
    """Kernel launches the device path would issue for this node: one for
    the atomic strategy, two for the tiled one (partial accumulate, then
    reduce). Taken from the resolved tiling rather than re-derived, so this
    module holds no copy of `gpu_tiling.mojo`'s rules.

    `node_of` defaults it to one, which is the atomic path and the shape of
    every small node. A caller that has resolved a geometry should not use
    that default: `for_tiling` below takes the count from
    `gpu_tiling.launches_for_strategy`, and a caller holding an
    `apple_histogram_policy.HistogramPlan` passes `plan.tiling()`.
    """

    @staticmethod
    def node_of(
        node: Int,
        node_rows: Int,
        n_slots: Int,
        n_features: Int,
        n_bins: Int,
        dataset_rows: Int,
        row_source: Int = ROWS_HOST_SNAPSHOT,
        gpu_launches: Int = 1,
        materialize_rows: Int = 0,
    ) raises -> LeafWork:
        """A node backed by the tree's active-row snapshot, which is every
        node but the root; see `tree_row_source`."""
        if node < 0:
            raise Error("node id must be nonnegative")
        if node_rows < 0:
            raise Error("row count must be nonnegative")
        if n_slots < 1 or n_slots > n_features:
            raise Error("active feature count is outside [1, n_features]")
        if dataset_rows < 1:
            raise Error("dataset must have at least one row")
        if row_source < 0 or row_source >= N_ROW_SOURCES:
            raise Error("unknown row source")
        if gpu_launches < 1:
            raise Error("a device build issues at least one launch")
        if materialize_rows < 0:
            raise Error("materialization row count must be nonnegative")
        if materialize_rows > 0 and not row_source_is_host(row_source):
            raise Error(
                "only a host row source materializes rows by partitioning;"
                " a device copy delivers them whole"
            )
        _ = histogram_cells(n_features, n_bins)
        return LeafWork(
            node,
            node_rows,
            n_slots,
            n_features,
            n_bins,
            dataset_rows,
            row_source,
            materialize_rows,
            gpu_launches,
        )

    @staticmethod
    def for_tiling(
        node: Int,
        node_rows: Int,
        n_slots: Int,
        n_features: Int,
        n_bins: Int,
        dataset_rows: Int,
        tiling: HistogramTiling,
        row_source: Int = ROWS_HOST_SNAPSHOT,
        materialize_rows: Int = 0,
    ) raises -> LeafWork:
        """A node whose device path has already been planned.

        The connected form of `node_of`: the launch count comes from the
        strategy the geometry resolved to rather than from this module's
        default, so the cost model charges the launches the device will
        actually issue. A caller holding an
        `apple_histogram_policy.HistogramPlan` passes `plan.tiling()`, which
        is the same geometry projected onto the descriptor every launch site
        takes.
        """
        return LeafWork.node_of(
            node,
            node_rows,
            n_slots,
            n_features,
            n_bins,
            dataset_rows,
            row_source,
            tiling.launches(),
            materialize_rows,
        )

    def row_slot_kops(self) -> Int:
        """Thousands of (row, active-feature) accumulations. Rounded up, so
        a node smaller than a thousand pairs still costs something."""
        return _ceil_div(self.node_rows * self.n_slots, 1000)

    def cell_kcells(self) raises -> Int:
        """Thousands of histogram cells, rounded up."""
        return _ceil_div(histogram_cells(self.n_features, self.n_bins), 1000)

    def download_bytes(self) raises -> Int:
        return fixed_download_bytes(self.n_features, self.n_bins)

    def transfer_bytes(self) raises -> Int:
        return row_transfer_bytes(self.row_source, self.node_rows)


@fieldwise_init
struct HybridContext(Copyable, Movable):
    """Everything about the *run* that a per-leaf decision depends on.

    Constructed once per tree, not per node, so a placement cannot drift
    inside a tree as a queue fills or a thread pool warms.
    """

    var mode: Int

    var device_split_search: Bool
    """True when `SPLIT_SEARCH_DEVICE` resolved. Then no histogram is
    downloaded, there is no host parent to subtract from, and every leaf is
    declined."""

    var gradients_host_resident: Bool
    """True on the `upload_gradients` path, where `grad` and `hess` are
    Float64 host lists the trainer already holds. False on the device
    objective path, where they exist only as device Float32 and pulling them
    back per round would cost `8 * n_rows` to save a per-node download."""

    var bins_host_resident: Bool
    """True whenever the caller still holds the `BinnedMatrix` it trained
    from, which both GPU trainers do for the whole fit."""

    var quantized_replica_verified: Bool
    """True once `MODE_MIRROR` has shown, on this hardware, that the host
    fixed-point replica reproduces the device histogram bit for bit. Gates
    `MODE_REPLICA`, because the claim is about a Float32 multiply on two
    different units and cannot be assumed."""

    var guard_transfer_dominates: Bool
    """Refuse a host build whose row readback costs more than the
    accumulation it enables, even when the head-to-head comparison still
    favors the host.

    Deliberately conservative and deliberately separable. The head-to-head
    can favor a host build that spends most of its time in a transfer, since
    it is competing against a device path that also transfers; this guard
    says that is not a trade worth making on a first integration, and a
    benchmark that disagrees can clear the flag rather than edit the model.

    Inert under the snapshot sources, which charge the node no transfer at
    all; it bites only on `ROWS_DEVICE_COPY`, the per-range readback the GPU
    grower actually uses. There the "transfer" is a fixed synchronize plus
    four bytes a row, and the device path pays that same synchronize on its
    histogram download, so the guard refuses exactly the small leaves the
    comparison exists to move: the grower clears it by default and
    `MOJOTREES_HYBRID_GUARD_TRANSFER=1` restores it for an A/B.
    """

    var n_active_rows: Int
    """Rows this tree grows on: the bag, or the dataset. Fixed for the whole
    tree by `begin_tree`, and the only input `snapshot_nanos` needs."""

    var snapshot_taken: Bool
    """Whether the host already holds a snapshot of this tree's active-row
    permutation.

    This is the one piece of *tree state* a placement depends on, and it is
    passed in rather than remembered so `place_leaf` stays a pure function of
    its arguments. It advances deterministically with the grower's node
    order: false until the first accepted host leaf pays for the snapshot,
    true for the rest of the tree, and false again at the next `begin_tree`.
    Nothing about it depends on timing or on which device finished first.
    """

    var costs: HybridCosts

    @staticmethod
    def disabled() -> HybridContext:
        """The shipped configuration: hybrid scheduling off, no cost model.
        Every `place_leaf` returns `PLACE_GPU`."""
        return HybridContext(
            MODE_OFF,
            False,
            True,
            True,
            False,
            True,
            1,
            False,
            HybridCosts.unmeasured(),
        )

    @staticmethod
    def from_env(
        device_split_search: Bool,
        gradients_host_resident: Bool,
        bins_host_resident: Bool,
        n_active_rows: Int = 1,
    ) -> HybridContext:
        """The environment's mode over the run's facts, still with no cost
        model: `MOJOTREES_HYBRID_LEAVES=replica` selects the mode and is
        still declined for want of measured coefficients. That is the
        intended behavior of the switch today, and the reason the switch
        exists before the numbers do: it makes the decline reason
        observable."""
        return HybridContext(
            env_hybrid_mode(),
            device_split_search,
            gradients_host_resident,
            bins_host_resident,
            False,
            True,
            n_active_rows,
            False,
            HybridCosts.unmeasured(),
        )

    def with_snapshot(self, taken: Bool) -> HybridContext:
        """This context with the snapshot flag set, which is how a grower
        advances the one bit of tree state a placement reads. Returns a new
        value rather than mutating, so a context handed to `place_leaf`
        cannot change underneath it."""
        return HybridContext(
            self.mode,
            self.device_split_search,
            self.gradients_host_resident,
            self.bins_host_resident,
            self.quantized_replica_verified,
            self.guard_transfer_dominates,
            self.n_active_rows,
            taken,
            self.costs.copy(),
        )


# --- Estimates ------------------------------------------------------------


def device_leaf_nanos(work: LeafWork, costs: HybridCosts) raises -> Int:
    """Modelled cost of building this node's histogram on the device and
    getting it to the host.

    Four terms, in the order `build_leaf` pays them: the launches, the
    accumulation, the fixed-size download plus its synchronization, and the
    fixed-point conversion. Two of the four do not scale with the node,
    which is the asymmetry the whole module exists to exploit.
    """
    if not costs.measured:
        return 0
    var total = costs.launch_nanos * work.gpu_launches
    total += work.row_slot_kops() * costs.device_nanos_per_krow_slot
    total += (
        _ceil_div(work.download_bytes(), 1024) * costs.transfer_nanos_per_kib
    )
    total += costs.sync_nanos
    total += work.cell_kcells() * costs.convert_nanos_per_kcell
    return total


def host_transfer_nanos(work: LeafWork, costs: HybridCosts) raises -> Int:
    """Modelled cost of getting this node's state to the host: the row-id
    readback and the synchronization it forces. Zero when the host already
    holds the rows."""
    if not costs.measured:
        return 0
    var bytes = work.transfer_bytes()
    if bytes == 0:
        return 0
    return (
        _ceil_div(bytes, 1024) * costs.transfer_nanos_per_kib
        + costs.sync_nanos
    )


def snapshot_nanos(n_active_rows: Int, costs: HybridCosts) raises -> Int:
    """Modelled cost of taking the host's snapshot of this tree's active-row
    permutation: one whole-buffer `download_rows` and its synchronization.

    `4 * n_active_rows` bytes, once per tree, and thereafter every node alive
    at that moment has its rows for free (`snapshot[begin : end]`) and every
    node created after it costs one host partition of its parent's range
    (`host_materialize_nanos`).

    Priced so a grower that maintains a snapshot can compare it against the
    per-range readback (`ROWS_DEVICE_COPY`, `host_transfer_nanos`). The GPU
    grower uses the readback and never takes a snapshot: per node it moves
    `4 * node_rows` bytes instead of `4 * n_active_rows`, and nothing is
    amortized across leaves.
    """
    if n_active_rows < 1:
        raise Error("a tree must grow on at least one row")
    if not costs.measured:
        return 0
    return (
        _ceil_div(4 * n_active_rows, 1024) * costs.transfer_nanos_per_kib
        + costs.sync_nanos
    )


def host_materialize_nanos(work: LeafWork, costs: HybridCosts) -> Int:
    """Modelled cost of bringing this node's row ids into existence on the
    host's own copy: one `partition_range_host` over the parent's range, or
    zero when the snapshot already holds them.

    The partition is stable in buffer order and so reproduces the device's
    result index for index, which is what lets the host keep its snapshot
    valid without re-reading it. It rewrites `snapshot[parent.begin :
    parent.end]` in place, so after it runs *both* children are materialized
    and neither pays again.
    """
    if not costs.measured:
        return 0
    return (
        _ceil_div(work.materialize_rows, 1000)
        * costs.host_partition_nanos_per_krow
    )


comptime _SCATTER_SATURATION_GAP = 256
"""Mean row gap, in rows, at which a node is charged the full scattered
rate. Bins are one byte and feature-major, so at a gap of 64 every bin read
is a fresh cache line and at a few hundred the prefetcher and the
translation cache have stopped helping; the calibration's per-gap rates
(bench/bench_hybrid_costs.mojo) sit under the ramp this fixes at every gap
it measured, so the ramp is conservative for the host."""


def host_slot_nanos_per_k(work: LeafWork, costs: HybridCosts) -> Int:
    """Host accumulation rate for *this* node, per thousand (row, feature)
    pairs: from the contiguous rate up to the scattered rate as the node's
    rows thin out.

    The node's rows sit in the device's stable buffer order, and a stable
    partition of an identity (or bag) permutation keeps them ascending, so
    per feature the host reads `node_rows` bytes spread over a
    `dataset_rows`-byte column. Mean gap `dataset_rows / node_rows` rows;
    the rate ramps linearly from the contiguous rate at gap one to the
    scattered rate at `_SCATTER_SATURATION_GAP` and stays there. A model,
    not a measurement of every gap; its endpoints are what E1 measures, and
    it is fitted to sit above the intermediate points E1 also measures.
    """
    if work.node_rows < 1:
        return costs.host_nanos_per_krow_slot
    var gap_permille = _ceil_div(
        1000 * work.dataset_rows, work.node_rows * _SCATTER_SATURATION_GAP
    )
    if gap_permille > 1000:
        gap_permille = 1000
    var spread = (
        costs.host_scatter_nanos_per_krow_slot - costs.host_nanos_per_krow_slot
    )
    return costs.host_nanos_per_krow_slot + spread * gap_permille // 1000


def host_build_nanos(work: LeafWork, costs: HybridCosts) raises -> Int:
    """Modelled cost of the host accumulation itself, transfer excluded: the
    fixed dispatch price of a host build, the zeroing pass over every cell,
    and the scattered adds over the node's own rows at the rate the node's
    row density earns it (`host_slot_nanos_per_k`)."""
    if not costs.measured:
        return 0
    return (
        costs.host_fixed_nanos
        + work.cell_kcells() * costs.host_zero_nanos_per_kcell
        + work.row_slot_kops() * host_slot_nanos_per_k(work, costs)
    )


def host_leaf_nanos(work: LeafWork, costs: HybridCosts) raises -> Int:
    """Modelled cost of building this node's histogram on the host, end to
    end, *excluding* any once-per-tree snapshot.

    Three terms: getting the rows (a device readback, or nothing),
    materializing them on the host's copy if a split created them after the
    snapshot, and the accumulation. No download and no conversion: the
    histogram is produced where it will be read.
    """
    return (
        host_transfer_nanos(work, costs)
        + host_materialize_nanos(work, costs)
        + host_build_nanos(work, costs)
    )


# --- The decision ---------------------------------------------------------


@fieldwise_init
struct Placement(Copyable, Movable):
    """Where one node's histogram is built, why, and the numbers behind it.

    Carried in full even when the answer is `PLACE_GPU`, so a trace can show
    a leaf that was two nanoseconds from going the other way and a
    regression can show which term moved.
    """

    var node: Int
    var device: Int
    var reason: Int
    var device_nanos: Int
    var host_nanos: Int
    var transfer_nanos: Int
    var transfer_bytes: Int
    var download_bytes: Int
    var origin: Int
    """The provenance a build under this placement will carry, for
    `HistogramCache.admit`. `ORIGIN_GPU_FIXED` whenever the device's
    histogram is the one consumed, which includes `MODE_MIRROR`."""

    var takes_snapshot: Bool
    """True when accepting this placement requires the host to take the
    tree's active-row snapshot first, which is the case for exactly one leaf
    per tree: the first one that could pay for it. The caller does the
    `download_rows` and then passes `ctx.with_snapshot(True)` for the rest of
    the tree."""

    def is_host(self) -> Bool:
        return self.device == PLACE_CPU

    def builds_on_host(self) -> Bool:
        """Whether a host accumulation runs at all, which `MODE_MIRROR`
        makes true without making `is_host` true."""
        return self.device == PLACE_CPU or self.device == PLACE_BOTH

    def is_reuse(self) -> Bool:
        """Whether nothing is accumulated for this node at all. The modelled
        costs are still carried, so a trace can show what the reuse
        avoided."""
        return self.device == PLACE_REUSE


@fieldwise_init
struct ReuseOffer(Copyable, Movable):
    """What the caller's histogram bookkeeping already holds for this node.

    A fact, not a search. `histogram_cache_policy.HistogramCache.lookup`
    answers "is this buffer still the right answer for this
    (dataset, round, tree, feature set, node)?" and returns a staleness code
    with it; this carries that answer to the placement so a node whose
    histogram exists is not built a second time on either device.

    Nothing here decides what is reusable. That argument is made once, in
    `histogram_cache_policy.mojo`, and its conclusion is narrow: beyond the
    parent/sibling subtraction both growers already exploit there is no reuse
    in this trainer, because every round refreshes every gradient. So the
    offer a caller can honestly make is the sibling of a committed split, and
    `none()` is what everything else gets.
    """

    var available: Bool
    var origin: Int
    """Where the offered histogram came from, for
    `origins_are_subtractable`: a device build, a host replica, a Float64
    host build, or a subtraction."""

    var staleness: Int
    """`FRESH`, or the `histogram_cache_policy` code saying why not."""

    @staticmethod
    def none() -> ReuseOffer:
        """No histogram is held for this node, which is the default and the
        state of every node under the shipped configuration."""
        return ReuseOffer(False, ORIGIN_UNKNOWN, STALE_ABSENT)

    @staticmethod
    def held(origin: Int, staleness: Int) -> ReuseOffer:
        """A cache entry the caller found. Still refused unless it is fresh
        and carries a known origin."""
        return ReuseOffer(True, origin, staleness)

    @staticmethod
    def subtracted_from(parent_origin: Int, direct_origin: Int) -> ReuseOffer:
        """The sibling of a committed split, which is the one reuse this
        trainer actually has.

        Offered only when the parent and the directly built child came
        through the same arithmetic: a dequantized device parent minus a
        Float64 host child is not the sibling, and
        `origins_are_subtractable` is the check that says so. When they do
        not agree the sibling has to be built like any other node, which is
        what `none()` here produces.
        """
        if not origins_are_subtractable(parent_origin, direct_origin):
            return ReuseOffer.none()
        return ReuseOffer(True, ORIGIN_SUBTRACTED, FRESH)

    def usable(self) -> Bool:
        return (
            self.available
            and self.staleness == FRESH
            and self.origin != ORIGIN_UNKNOWN
        )


def decline_reason(ctx: HybridContext, work: LeafWork) raises -> Int:
    """Why this node will not be built on the host, or `ACCEPTED`.

    A ladder, in the order a reader should ask the questions: is hybrid
    scheduling on at all, can this run's state model support it, is the
    substitution licensed, are there numbers to decide with, and only then
    the numbers themselves. Each gate names the outermost thing that is
    missing, so a decline reason points at the work that would remove it
    rather than at a symptom.
    """
    if ctx.mode == MODE_OFF:
        return DECLINE_MODE_OFF
    if ctx.device_split_search:
        return DECLINE_NO_HOST_PARENT
    if not ctx.gradients_host_resident:
        return DECLINE_GRADIENTS_ON_DEVICE
    if not ctx.bins_host_resident:
        return DECLINE_NO_HOST_BINS
    if ctx.mode == MODE_REPLICA and not ctx.quantized_replica_verified:
        return DECLINE_REPLICA_UNVERIFIED
    if not ctx.costs.measured:
        return DECLINE_COSTS_UNMEASURED
    if work.node_rows < 1:
        # An empty node's histogram is all zeros on either device and the
        # grower's shape rules will refuse to split it anyway; sending it
        # anywhere new is churn, not a decision.
        return DECLINE_EMPTY_NODE

    var build = host_build_nanos(work, ctx.costs)
    var transfer = host_transfer_nanos(work, ctx.costs)
    if ctx.guard_transfer_dominates and transfer > build:
        return DECLINE_TRANSFER_DOMINATES
    var host = host_leaf_nanos(work, ctx.costs)
    var device = device_leaf_nanos(work, ctx.costs)
    if host >= device:
        # `>=` rather than `>`: a modelled tie stays on the device, which is
        # the path that would have run without this module. A cost model
        # that cannot separate two paths must not move work.
        return DECLINE_SLOWER_ON_HOST
    if work.row_source == ROWS_HOST_SNAPSHOT and not ctx.snapshot_taken:
        # This node wants to read a snapshot that does not exist yet, so it
        # is the leaf that would pay for taking it. It may do so only if its
        # own saving covers the whole snapshot; every later host leaf in this
        # tree then reads the snapshot for free.
        #
        # Conservative on purpose. Charging the first leaf the full price
        # cannot make a tree slower than the pure-device path, whereas
        # amortizing over an *expected* leaf count would be a guess about a
        # distribution nobody has measured. It does mean a tree whose savings
        # are spread thinly across many leaves never snapshots at all; E4 and
        # E6 in docs/design/HYBRID_TRAINING.md are the experiments that would
        # say whether that matters.
        if device - host < snapshot_nanos(ctx.n_active_rows, ctx.costs):
            return DECLINE_SNAPSHOT_NOT_PAID
    return ACCEPTED


def place_leaf(ctx: HybridContext, work: LeafWork) raises -> Placement:
    """Decide where one node's histogram is built.

    Pure: node counts, dataset shape, run configuration, snapshot state, and
    cost coefficients in; a placement out. No clock is read, no queue is
    polled, and no completion signal is consulted.

    `ctx.snapshot_taken` is the one input that is tree *state* rather than
    tree shape, and it is an argument rather than a remembered field for
    exactly that reason: two calls with the same arguments give the same
    answer, and the state advances only with the grower's own deterministic
    node order. Which device finished first cannot reach it.

    `MODE_MIRROR` is the one mode whose accepted placement is `PLACE_BOTH`:
    the host build runs, its output is compared, and the device's histogram
    is still the one the grower consumes. Its `origin` is therefore
    `ORIGIN_GPU_FIXED`, not a host origin.
    """
    var why = decline_reason(ctx, work)
    var device_nanos = device_leaf_nanos(work, ctx.costs)
    var host_nanos = host_leaf_nanos(work, ctx.costs)
    var transfer = host_transfer_nanos(work, ctx.costs)
    var place = PLACE_GPU
    var origin = ORIGIN_GPU_FIXED
    var takes_snapshot = False
    if why == ACCEPTED:
        if mode_substitutes(ctx.mode):
            place = PLACE_CPU
            origin = host_origin_for(ctx.mode)
        else:
            place = PLACE_BOTH
        takes_snapshot = (
            work.row_source == ROWS_HOST_SNAPSHOT and not ctx.snapshot_taken
        )
    return Placement(
        work.node,
        place,
        why,
        device_nanos,
        host_nanos,
        transfer,
        work.transfer_bytes(),
        work.download_bytes(),
        origin,
        takes_snapshot,
    )


def place_leaf_with_reuse(
    ctx: HybridContext, work: LeafWork, offer: ReuseOffer
) raises -> Placement:
    """Decide where one node's histogram comes from, reuse included.

    Three answers, in the order they are cheap: read a histogram that is
    already the right answer, build it on the host, or build it on the
    device. The first is decided from the caller's offer alone and never
    from the cost model, because a build that does not happen cannot be
    slower than one that does; the other two are `place_leaf`'s comparison,
    unchanged.

    Reuse changes no split. The reused histogram is the one the grower would
    have produced -- a subtraction both growers already perform, or a cache
    entry whose key says it describes this exact
    (dataset, gradients, feature set, tree, node) -- so the gain scan sees
    the same cells and picks the same split. What it changes is only whether
    an accumulation ran.
    """
    if not offer.usable():
        return place_leaf(ctx, work)
    return Placement(
        work.node,
        PLACE_REUSE,
        REUSED_FRESH_HISTOGRAM,
        device_leaf_nanos(work, ctx.costs),
        host_leaf_nanos(work, ctx.costs),
        host_transfer_nanos(work, ctx.costs),
        work.transfer_bytes(),
        work.download_bytes(),
        offer.origin,
        False,
    )


@fieldwise_init
struct SplitPlan(Copyable, Movable):
    """How the two children of one committed split get their histograms.

    The direct/subtracted division is not this module's decision and is not
    changed by it: both growers already build the smaller child and subtract
    for the larger, and the tie goes to the left child (`n_left <= n_right`
    in `grow_tree_gpu`). What is decided here is only where that one direct
    build runs. The subtracted sibling is host arithmetic in either case,
    because the parent's histogram is host-resident on the path this module
    accepts.

    Both children carry a placement, so a plan says what happens to the whole
    split rather than to half of it. The sibling's is normally `PLACE_REUSE`
    with `ORIGIN_SUBTRACTED`, which is the reuse this trainer already
    performs and is now stated instead of assumed. It is not reuse when the
    parent and the direct child came through different arithmetic -- a
    dequantized device parent and a Float64 host child, which is exactly what
    `MODE_HOST_FLOAT64` produces -- and then the sibling is placed like any
    other node.
    """

    var direct_node: Int
    var direct_rows: Int
    var subtracted_node: Int
    var subtracted_rows: Int
    var direct_is_left: Bool
    var placement: Placement
    var subtracted_placement: Placement

    def builds_on_host(self) -> Bool:
        return self.placement.builds_on_host()

    def sibling_is_subtracted(self) -> Bool:
        """Whether the sibling comes out of the parent by subtraction, which
        is the arithmetic the grower already performs."""
        return (
            self.subtracted_placement.is_reuse()
            and self.subtracted_placement.origin == ORIGIN_SUBTRACTED
        )

    def builds(self) -> Int:
        """Histogram accumulations this split costs, on either device. One
        under the shipped configuration, two when the sibling cannot be
        subtracted, and one more when a mirrored host build runs alongside a
        device one."""
        var n = 0
        if not self.placement.is_reuse():
            n += 1
            if self.placement.device == PLACE_BOTH:
                n += 1
        if not self.subtracted_placement.is_reuse():
            n += 1
            if self.subtracted_placement.device == PLACE_BOTH:
                n += 1
        return n


def plan_split(
    ctx: HybridContext,
    left_node: Int,
    n_left: Int,
    right_node: Int,
    n_right: Int,
    n_slots: Int,
    n_features: Int,
    n_bins: Int,
    dataset_rows: Int,
    left_row_source: Int = ROWS_HOST_SNAPSHOT,
    right_row_source: Int = ROWS_HOST_SNAPSHOT,
    gpu_launches: Int = 1,
    parent_materialized: Bool = True,
    parent_origin: Int = ORIGIN_GPU_FIXED,
) raises -> SplitPlan:
    """Plan the histogram builds a committed split needs.

    Row counts come from the parent histogram's exact integer counts, which
    the grower has before the partition runs, so this plans without waiting
    for any device work to finish. That is what lets a host build of the
    small child overlap the device partition rather than follow it.

    `parent_materialized` is False when the parent's range has not yet been
    partitioned on the host's copy of the permutation, in which case the
    direct child charges one host partition over the parent's rows
    (`n_left + n_right`). That partition materializes *both* children at
    once, so the sibling never charges it again.

    `parent_origin` is where the parent's histogram came from, and it decides
    whether the sibling can be subtracted from it at all. It defaults to
    `ORIGIN_GPU_FIXED`, which is what the GPU grower holds: a device build,
    downloaded and dequantized.
    """
    if n_left < 0 or n_right < 0:
        raise Error("child row counts must be nonnegative")
    var direct_is_left = n_left <= n_right
    var direct_node: Int
    var direct_rows: Int
    var other_node: Int
    var other_rows: Int
    var source: Int
    if direct_is_left:
        direct_node = left_node
        direct_rows = n_left
        other_node = right_node
        other_rows = n_right
        source = left_row_source
    else:
        direct_node = right_node
        direct_rows = n_right
        other_node = left_node
        other_rows = n_left
        source = right_row_source
    var materialize = 0
    if row_source_is_host(source) and not parent_materialized:
        materialize = n_left + n_right
    var work = LeafWork.node_of(
        direct_node,
        direct_rows,
        n_slots,
        n_features,
        n_bins,
        dataset_rows,
        source,
        gpu_launches,
        materialize,
    )
    var direct = place_leaf(ctx, work)

    # The sibling. Its rows are whatever is left of the parent's, so the
    # partition the direct child may have charged has already materialized
    # them and it charges nothing; what it costs, if it cannot be subtracted,
    # is an ordinary build of its own row count.
    var other_source = right_row_source
    if not direct_is_left:
        other_source = left_row_source
    var other_work = LeafWork.node_of(
        other_node,
        other_rows,
        n_slots,
        n_features,
        n_bins,
        dataset_rows,
        other_source,
        gpu_launches,
        0,
    )
    var sibling = place_leaf_with_reuse(
        ctx,
        other_work,
        ReuseOffer.subtracted_from(parent_origin, direct.origin),
    )
    return SplitPlan(
        direct_node,
        direct_rows,
        other_node,
        other_rows,
        direct_is_left,
        direct^,
        sibling^,
    )


def child_row_source(parent_source: Int) -> Int:
    """The row source a child inherits.

    A child of a node whose rows the host already holds also has host rows:
    the host partitions the parent's list with the same stable rule the
    device applies to the same range, so the two agree index for index
    (`partition_range_host` is that reference model). Whether the parent's
    rows came from the identity seeding, the bag, an earlier mirror, or the
    tree's snapshot makes no difference to the child, which is why all four
    collapse to `ROWS_HOST_MIRROR` here.

    A child of a device-owned node is device-owned. Nothing here promotes a
    node to host ownership; only the snapshot does that, tree-wide, and after
    it is taken every node alive at that moment becomes
    `ROWS_HOST_SNAPSHOT`.
    """
    if row_source_is_host(parent_source):
        return ROWS_HOST_MIRROR
    return ROWS_DEVICE_COPY


def tree_row_source(is_root: Bool, bagged: Bool) -> Int:
    """The row source a node gets from the tree's shape alone.

    The root is host-known either way: the identity permutation unbagged, the
    host's own bag otherwise. Every other node is snapshot-backed, whether or
    not the snapshot has been taken yet -- taking it is a cost
    (`DECLINE_SNAPSHOT_NOT_PAID` decides whether a leaf can afford it), not a
    different source.

    `ROWS_DEVICE_COPY` is therefore never produced here: this is the row
    source under the *snapshot* design. The GPU grower does not use it; it
    passes `ROWS_DEVICE_COPY` for every leaf to `plan_split`, because the
    per-range readback this function was written believing impossible
    (`GpuHistogramBuilder.readback_range`) exists and needs no snapshot.
    """
    if is_root:
        if bagged:
            return ROWS_HOST_BAG
        return ROWS_HOST_IDENTITY
    return ROWS_HOST_SNAPSHOT


# --- Deterministic commitment ---------------------------------------------


struct WaveBarrier(Copyable, Movable):
    """The bookkeeping that keeps completion order out of split selection.

    A hybrid grower has two producers running at once: the device's kernels
    for one child and the host's accumulation for another, or the host's
    build overlapping the device's partition. Which finishes first depends
    on the machine, the queue, and the thread pool. None of that may reach
    the frontier.

    So the grower announces the nodes of a wave in the order it will consume
    them -- for a split, the direct child then its subtracted sibling, which
    is the order `grow_tree_gpu` already inserts them in -- completes them in
    whatever order they finish, and reads `announced()` to consume. The
    announcement order is a property of the tree, not of the hardware.
    `require_ready` refuses a commit while any build is outstanding, so a
    best-gain scan cannot see a frontier whose entries are partly from the
    previous wave.

    `completion_order()` exists only so a trace can show that the two orders
    differed, which is the observation that proves the barrier was doing
    something. Nothing may branch on it.
    """

    var nodes: List[Int]
    var done: List[Bool]
    var completions: List[Int]
    var waves: Int

    def __init__(out self):
        self.nodes = List[Int]()
        self.done = List[Bool]()
        self.completions = List[Int]()
        self.waves = 0

    def open(mut self) raises:
        """Start a new wave, discarding the previous one's record."""
        self.nodes.clear()
        self.done.clear()
        self.completions.clear()
        self.waves += 1

    def expect(mut self, node: Int) raises:
        """Announce a node this wave will build, in consumption order."""
        if node < 0:
            raise Error("node id must be nonnegative")
        for i in range(len(self.nodes)):
            if self.nodes[i] == node:
                raise Error("this node is already announced in this wave")
        self.nodes.append(node)
        self.done.append(False)

    def complete(mut self, node: Int) raises:
        """Record that a node's histogram is finished, whichever device
        finished it."""
        for i in range(len(self.nodes)):
            if self.nodes[i] == node:
                if self.done[i]:
                    raise Error("this node has already completed")
                self.done[i] = True
                self.completions.append(node)
                return
        raise Error("this node was not announced in this wave")

    def outstanding(self) -> Int:
        var n = 0
        for i in range(len(self.done)):
            if not self.done[i]:
                n += 1
        return n

    def ready(self) -> Bool:
        return self.outstanding() == 0

    def require_ready(self) raises:
        """Refuse to proceed while any announced build is unfinished. Called
        immediately before the best-gain scan."""
        if not self.ready():
            raise Error(
                "a split cannot be selected while histogram builds are"
                " outstanding: completion order would decide the frontier"
            )

    def announced(self) -> List[Int]:
        """The wave's nodes in the order the grower will consume them. The
        only order any caller may use."""
        return self.nodes.copy()

    def completion_order(self) -> List[Int]:
        """The order the builds actually finished. Diagnostics only."""
        return self.completions.copy()

    def order_was_permuted(self) -> Bool:
        """Whether completion order differed from announcement order in this
        wave. True is not a problem; it is the evidence that the barrier is
        load-bearing."""
        if len(self.completions) != len(self.nodes):
            return False
        for i in range(len(self.nodes)):
            if self.nodes[i] != self.completions[i]:
                return True
        return False


# --- Reporting ------------------------------------------------------------


def _yes_no(value: Bool) -> String:
    if value:
        return String("yes")
    return String("no")


def describe_placement(placement: Placement) -> String:
    """One line for a trace or a bug report: where the node went, why, and
    the two modelled costs it went there on."""
    return String(
        "node=",
        placement.node,
        " where=",
        placement_name(placement.device),
        " why=",
        decline_name(placement.reason),
        " device_ns=",
        placement.device_nanos,
        " host_ns=",
        placement.host_nanos,
        " transfer_ns=",
        placement.transfer_nanos,
        " transfer=",
        placement.transfer_bytes,
        "B download=",
        placement.download_bytes,
        "B origin=",
        origin_name(placement.origin),
        " takes_snapshot=",
        _yes_no(placement.takes_snapshot),
    )


def describe_offer(offer: ReuseOffer) -> String:
    """One line for a trace: what the caller held for this node and whether
    it was usable."""
    return String(
        "held=",
        _yes_no(offer.available),
        " origin=",
        origin_name(offer.origin),
        " staleness=",
        staleness_name(offer.staleness),
        " usable=",
        _yes_no(offer.usable()),
    )


def describe_context(ctx: HybridContext) -> String:
    return String(
        "mode=",
        mode_name(ctx.mode),
        " substitutes=",
        _yes_no(mode_substitutes(ctx.mode)),
        " preserves_fit=",
        _yes_no(mode_preserves_fit(ctx.mode)),
        " host_gradients=",
        _yes_no(ctx.gradients_host_resident),
        " host_bins=",
        _yes_no(ctx.bins_host_resident),
        " device_split_search=",
        _yes_no(ctx.device_split_search),
        " replica_verified=",
        _yes_no(ctx.quantized_replica_verified),
        " active_rows=",
        ctx.n_active_rows,
        " snapshot=",
        _yes_no(ctx.snapshot_taken),
        " costs=",
        ctx.costs.cite(),
    )


def env_hybrid_trace() -> Int:
    """`MOJOTREES_HYBRID_TRACE`: nonzero makes a caller print one
    `describe_placement` line per node. Reporting only."""
    return _env_int("MOJOTREES_HYBRID_TRACE", 0)
