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
   have is the node's slice of the active-row permutation, which is
   `4 * node_rows` bytes device-to-host. On the device-objective path
   (`fill_gradients_device`) the gradients never exist on the host at all,
   and pulling them back per round would undo that path's whole purpose, so
   this module declines there too (`DECLINE_GRADIENTS_ON_DEVICE`).

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

- **Placement is a pure function of shape.** `place_leaf` reads node row
  counts, dataset shape, active feature count, launch counts, and the cost
  coefficients. It reads no clock, no queue depth, and no completion signal.
  Two runs on one machine, and a run with the host build racing the device
  against one where it does not, place every leaf identically.
- **Completion order is not consumption order.** `WaveBarrier` is the
  bookkeeping that enforces it: the grower announces the nodes of a wave in
  the order it will consume them, the builds complete in whatever order the
  hardware delivers, and `announced()` returns the announcement order
  regardless. `require_ready` refuses a commit while any build is
  outstanding, so no gain comparison can be made from a partial frontier.
  Between them, which device finished first cannot reach the best-gain scan,
  the tie-break, or the chosen split.

What this module does not do
----------------------------
It does not accumulate a histogram, own a buffer, touch a device, or edit a
grower. It does not decide CPU-versus-GPU *training* -- that is
`device_policy.mojo`, and a hybrid leaf schedule presupposes the GPU trainer
was already chosen. Histogram lifetimes, cache keys, and invalidation are
`histogram_cache_policy.mojo`.
"""

from std.os import getenv

from .histogram_cache_policy import (
    ORIGIN_CPU_FLOAT64,
    ORIGIN_CPU_REPLICA,
    ORIGIN_GPU_FIXED,
    fixed_download_bytes,
    histogram_cells,
    origin_name,
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
    """`MOJOBOOST_HYBRID_LEAVES`, following the `MOJOBOOST_` contract in
    parallel.mojo: `mirror`, `replica`, `float64`, or `off` (the default,
    and the value of anything unrecognized)."""
    var s = getenv("MOJOBOOST_HYBRID_LEAVES")
    if s == "mirror":
        return MODE_MIRROR
    if s == "replica":
        return MODE_REPLICA
    if s == "float64":
        return MODE_HOST_FLOAT64
    return MODE_OFF


# --- Placement and refusals -----------------------------------------------

comptime PLACE_GPU = 0
comptime PLACE_CPU = 1
comptime PLACE_BOTH = 2


def placement_name(place: Int) -> String:
    if place == PLACE_CPU:
        return String("cpu")
    if place == PLACE_BOTH:
        return String("both")
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
comptime N_DECLINE_REASONS = 10


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
    return String("unknown")


# --- Where a node's rows come from ----------------------------------------

comptime ROWS_HOST_IDENTITY = 0
comptime ROWS_HOST_BAG = 1
comptime ROWS_HOST_MIRROR = 2
comptime ROWS_DEVICE_COPY = 3
comptime N_ROW_SOURCES = 4


def row_source_name(source: Int) -> String:
    if source == ROWS_HOST_IDENTITY:
        return String("host-identity")
    if source == ROWS_HOST_BAG:
        return String("host-bag")
    if source == ROWS_HOST_MIRROR:
        return String("host-mirror")
    return String("device-copy")


def row_source_is_host(source: Int) -> Bool:
    """Whether the host already holds this node's row ids, so a host build
    moves nothing.

    Three cases do. The unbagged root's rows are the identity permutation,
    which `begin_tree` writes with a kernel and the host can reproduce
    without reading anything. A bagged root's rows are the bag, which the
    host drew and handed to `begin_tree` in that order. And a node whose
    ancestor was already built on the host can have its rows derived on the
    host by re-running the same stable partition
    (`partition_range_host` in gpu_active_rows.mojo is that reference model,
    index for index), which is the `ROWS_HOST_MIRROR` case: the transfer is
    paid once at the root of a host-owned subtree and never again inside it.
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
    of the row work: the term a four-row leaf pays in full."""

    var sync_nanos: Int
    """Host round trip for one `ctx.synchronize()`. Paid once per node by
    the device path (`download_raw`) and once by a host build whose rows
    have to be copied back, so it cancels except where the row source is
    host-side."""

    var transfer_nanos_per_kib: Int
    """Device-to-host copy rate through the pinned staging buffers, in
    nanoseconds per KiB. One rate for both directions of interest here: the
    fixed-point histogram download and the row-id readback go through the
    same path."""

    var device_nanos_per_krow_slot: Int
    """Device accumulation, per thousand (row, active-feature) pairs."""

    var host_nanos_per_krow_slot: Int
    """Host accumulation, per thousand (row, active-feature) pairs: the
    scattered-add inner loop of `_accumulate_subset`."""

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
            or host_zero_nanos_per_kcell < 0
            or convert_nanos_per_kcell < 0
        ):
            raise Error("cost coefficients must be nonnegative")
        self.launch_nanos = launch_nanos
        self.sync_nanos = sync_nanos
        self.transfer_nanos_per_kib = transfer_nanos_per_kib
        self.device_nanos_per_krow_slot = device_nanos_per_krow_slot
        self.host_nanos_per_krow_slot = host_nanos_per_krow_slot
        self.host_zero_nanos_per_kcell = host_zero_nanos_per_kcell
        self.convert_nanos_per_kcell = convert_nanos_per_kcell
        self.measured = True
        self.evidence_id = evidence_id^
        self.measured_on = measured_on^

    @staticmethod
    def unmeasured() -> HybridCosts:
        """The only cost model this repository can construct today.

        Every coefficient is zero and `measured` is False, so no comparison
        is attempted and every leaf stays on the device. Replacing this with
        real coefficients is a deliberate edit that must cite a benchmark;
        see `docs/design/HYBRID_TRAINING.md`.
        """
        return HybridCosts()

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
    """The binned matrix's column stride. Carried because it decides how far
    apart a host build's bin reads land, which is the difference between a
    cache-resident scan and a scattered one; no term below uses it yet, and
    a fitted cost model may need it."""

    var row_source: Int
    var gpu_launches: Int
    """Kernel launches the device path would issue for this node: one for
    the atomic strategy, two for the tiled one (partial accumulate, then
    reduce). Taken from the resolved tiling rather than re-derived, so this
    module holds no copy of `gpu_tiling.mojo`'s rules."""

    @staticmethod
    def node_of(
        node: Int,
        node_rows: Int,
        n_slots: Int,
        n_features: Int,
        n_bins: Int,
        dataset_rows: Int,
        row_source: Int = ROWS_DEVICE_COPY,
        gpu_launches: Int = 1,
    ) raises -> LeafWork:
        """A node whose rows must be copied back for a host build, which is
        every node until the caller establishes otherwise."""
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
        _ = histogram_cells(n_features, n_bins)
        return LeafWork(
            node,
            node_rows,
            n_slots,
            n_features,
            n_bins,
            dataset_rows,
            row_source,
            gpu_launches,
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
    """

    var costs: HybridCosts

    @staticmethod
    def disabled() -> HybridContext:
        """The shipped configuration: hybrid scheduling off, no cost model.
        Every `place_leaf` returns `PLACE_GPU`."""
        return HybridContext(
            MODE_OFF, False, True, True, False, True, HybridCosts.unmeasured()
        )

    @staticmethod
    def from_env(
        device_split_search: Bool,
        gradients_host_resident: Bool,
        bins_host_resident: Bool,
    ) -> HybridContext:
        """The environment's mode over the run's facts, still with no cost
        model: `MOJOBOOST_HYBRID_LEAVES=replica` selects the mode and is
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
            HybridCosts.unmeasured(),
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


def host_build_nanos(work: LeafWork, costs: HybridCosts) raises -> Int:
    """Modelled cost of the host accumulation itself, transfer excluded: the
    zeroing pass over every cell plus the scattered adds over the node's own
    rows."""
    if not costs.measured:
        return 0
    return (
        work.cell_kcells() * costs.host_zero_nanos_per_kcell
        + work.row_slot_kops() * costs.host_nanos_per_krow_slot
    )


def host_leaf_nanos(work: LeafWork, costs: HybridCosts) raises -> Int:
    """Modelled cost of building this node's histogram on the host, end to
    end. No download and no conversion: the histogram is produced where it
    will be read."""
    return host_transfer_nanos(work, costs) + host_build_nanos(work, costs)


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

    def is_host(self) -> Bool:
        return self.device == PLACE_CPU

    def builds_on_host(self) -> Bool:
        """Whether a host accumulation runs at all, which `MODE_MIRROR`
        makes true without making `is_host` true."""
        return self.device == PLACE_CPU or self.device == PLACE_BOTH


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
    if work.node_rows < 1:
        # An empty node's histogram is all zeros on either device and the
        # grower's shape rules will refuse to split it anyway; sending it
        # anywhere new is churn, not a decision.
        return DECLINE_EMPTY_NODE
    if not ctx.costs.measured:
        return DECLINE_COSTS_UNMEASURED

    var build = host_build_nanos(work, ctx.costs)
    var transfer = host_transfer_nanos(work, ctx.costs)
    if ctx.guard_transfer_dominates and transfer > build:
        return DECLINE_TRANSFER_DOMINATES
    if build + transfer >= device_leaf_nanos(work, ctx.costs):
        # `>=` rather than `>`: a modelled tie stays on the device, which is
        # the path that would have run without this module. A cost model
        # that cannot separate two paths must not move work.
        return DECLINE_SLOWER_ON_HOST
    return ACCEPTED


def place_leaf(ctx: HybridContext, work: LeafWork) raises -> Placement:
    """Decide where one node's histogram is built.

    Pure: node counts, dataset shape, run configuration, and cost
    coefficients in; a placement out. No clock is read, no queue is polled,
    and no earlier placement is consulted, so replaying a tree's nodes in
    any order reproduces every placement.

    `MODE_MIRROR` is the one mode whose accepted placement is `PLACE_BOTH`:
    the host build runs, its output is compared, and the device's histogram
    is still the one the grower consumes. Its `origin` is therefore
    `ORIGIN_GPU_FIXED`, not a host origin.
    """
    var why = decline_reason(ctx, work)
    var device_nanos = device_leaf_nanos(work, ctx.costs)
    var host_build = host_build_nanos(work, ctx.costs)
    var transfer = host_transfer_nanos(work, ctx.costs)
    var place = PLACE_GPU
    var origin = ORIGIN_GPU_FIXED
    if why == ACCEPTED:
        if mode_substitutes(ctx.mode):
            place = PLACE_CPU
            origin = host_origin_for(ctx.mode)
        else:
            place = PLACE_BOTH
    return Placement(
        work.node,
        place,
        why,
        device_nanos,
        host_build + transfer,
        transfer,
        work.transfer_bytes(),
        work.download_bytes(),
        origin,
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
    """

    var direct_node: Int
    var direct_rows: Int
    var subtracted_node: Int
    var subtracted_rows: Int
    var direct_is_left: Bool
    var placement: Placement

    def builds_on_host(self) -> Bool:
        return self.placement.builds_on_host()


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
    left_row_source: Int = ROWS_DEVICE_COPY,
    right_row_source: Int = ROWS_DEVICE_COPY,
    gpu_launches: Int = 1,
) raises -> SplitPlan:
    """Plan the one histogram build a committed split needs.

    Row counts come from the parent histogram's exact integer counts, which
    the grower has before the partition runs, so this plans without waiting
    for any device work to finish. That is what lets a host build of the
    small child overlap the device partition rather than follow it.
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
    var work = LeafWork.node_of(
        direct_node,
        direct_rows,
        n_slots,
        n_features,
        n_bins,
        dataset_rows,
        source,
        gpu_launches,
    )
    return SplitPlan(
        direct_node,
        direct_rows,
        other_node,
        other_rows,
        direct_is_left,
        place_leaf(ctx, work),
    )


def child_row_source(parent_source: Int) -> Int:
    """The row source a child inherits.

    A child of a node whose rows the host already holds also has host rows:
    the host partitions the parent's list with the same stable rule the
    device applies to the same range, so the two agree index for index
    (`partition_range_host` is that reference model). This is what makes a
    host-owned subtree pay its readback once, at the subtree's root, rather
    than at every node inside it.

    A child of a device-owned node is device-owned. Nothing here promotes a
    node to host ownership; only an accepted `place_leaf` does that, and it
    is the caller that records the promotion by passing
    `ROWS_HOST_MIRROR` for that node's children.
    """
    if row_source_is_host(parent_source):
        return ROWS_HOST_MIRROR
    return ROWS_DEVICE_COPY


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

    def open(mut self):
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
        " costs=",
        ctx.costs.cite(),
    )


def env_hybrid_trace() -> Int:
    """`MOJOBOOST_HYBRID_TRACE`: nonzero makes a caller print one
    `describe_placement` line per node. Reporting only."""
    return _env_int("MOJOBOOST_HYBRID_TRACE", 0)
