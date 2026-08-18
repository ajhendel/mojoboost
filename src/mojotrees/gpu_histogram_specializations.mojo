"""Primitives for specializing the GPU histogram path.

The pieces a specialized histogram kernel would be built out of, and nothing
that launches one. Every function here is arithmetic over Ints and Bools: none
of them opens a `DeviceContext`, allocates, launches a kernel, or reads the
environment. That is deliberate, and it is what lets the whole specialization
story be reasoned about (and later tested) on a machine with no accelerator.

Three specializations are described here, in increasing order of how much
they demand of the kernel side:

1. **Bin-capacity classes.** This one has landed, and the description below
   is now a description of the range histogram family rather than of a gap.
   `gpu_active_rows._range_hist_atomic_kernel` and `_range_hist_partial_kernel`
   take a `BIN_CAP` comptime parameter and allocate three Int32 planes of
   `GROUP * BIN_CAP` cells, so a 32-bin histogram no longer occupies the
   3 KiB a 256-bin one does. That family's ladder is the four capacities in
   `gpu_tiling` (32, 64, 128, 256) rather than the five here, and its
   footprint is `gpu_tiling.histogram_shared_bytes`, which carries the
   feature-group width this module's `kernel_shared_bytes` does not.
   `bin_capacity_for` here still rounds to the next power of two from 16, and
   is still what `gpu_portability` and `gpu_vendor_policy` price a
   single-slot block with; the two ladders differ only at 16, where the
   kernel family's narrowest rung is 32, so this module's figure is the
   smaller one and no caller may treat it as an upper bound on a real launch.
   Reconciling the two ladders is a follow-up, not a correctness bug: nothing
   here selects a launch.

2. **Packed bin loads.** The row loop reads `bins[f * n_rows + r]` where `r`
   comes from the active-row permutation, so it is a gather and not a
   contiguous read. It is contiguous exactly when the node's slice of the
   permutation is an ascending run of consecutive row ids, which is a
   property of the permutation and has to be reported by the caller, never
   assumed. `plan_packed_window` computes the aligned span of such a run, and
   `pack4_bins`/`unpack_bin` are the portable pack and extract. See the
   capability boundary below for what a wide load would have to prove first.
   The window arithmetic is only valid on a one-byte-per-cell, one-feature-
   per-block matrix, so `plan_packed_window_for` takes a
   `BinStorageDescriptor` and refuses every other layout outright rather than
   planning a window that would read the wrong bytes.

3. **Batched small leaves, planned elsewhere.** A node whose histogram
   launches fewer threadgroups than the device can hold leaves the device
   underfilled, and a tree's frontier is full of such nodes. The kernels and
   the plan for that live in `gpu_leaf_batching.mojo`, which concatenates the
   batch's row tiles onto one flat `grid.y` axis and binary searches it per
   threadgroup. This module deliberately contains **no** batch planner: an
   earlier draft of it planned a `grid.z` batch with a uniform tile count per
   leaf, which wastes every threadgroup past a small leaf's end and rests on
   a portable `grid.z` bound this project has never established. The packed
   tile axis has neither problem, so the duplicate was removed rather than
   maintained beside it. What survives here is the policy-side question of
   *whether* a frontier wants batching, which is
   `apple_histogram_policy.batching_declined_reason`.

What is deliberately not here
-----------------------------
No vendor intrinsic of any kind, Apple's included. The rule this module
enforces structurally is that a non-portable operation may only be reached
through a flag on `DeviceHistogramCapabilities`, whose `portable()` value
answers false to every one of them, and only where a portable implementation
of the same arithmetic sits next to it producing the identical integers. The
portable implementation is the definition; the fast path is an optimization
that has to reproduce it.

No subgroup width is assumed anywhere. `DeviceHistogramCapabilities` carries
`subgroup_width` and its portable value is 0, meaning unknown, which is the
only honest answer on a backend that refuses `WARP_SIZE` (Metal does). No
function here divides by it, and none should: `WARP_GRANULARITY` in
`gpu_tiling.mojo` is a launch-rounding granularity chosen to be a multiple of
every supported backend's width, not a claim about any device's width.

No unsourced grid bound either, for the same reason. `MAX_GRID_DIM_Y` in
`gpu_tiling.mojo` is 65535 because CUDA caps `grid.y` there and the other
backends allow more; there is no comparable figure for `grid.z` in this
project, so nothing here carries one. An earlier draft did, by analogy, and
that was the same mistake as assuming a subgroup width.

No policy. Which specialization to use for which device and shape is
`apple_histogram_policy.mojo`, which is also where the default-off gate
lives. Nothing here selects anything.

The storage descriptor
----------------------
`BinStorageDescriptor` is this lane's single answer to "what does one
(row, feature) cell of the device-resident bin matrix actually cost to read".
It is produced by `gpu_binned_layout.BinLayoutPlan.storage_descriptor` (the
one module allowed to *choose* a storage width) and consumed here, so a
specialization never re-derives a width, a byte offset, or a shared-memory
footprint from a bin count it happened to be handed. `plan_packed_window`
below used to assume one byte per cell silently; `plan_packed_window_for`
takes the descriptor and refuses every layout that assumption is false for.

Two constants used to be mirrored from `gpu_tiling.mojo` so this module could
stand alone while it landed alongside concurrent work on that file. That was
the pre-integration arrangement handoffs/performance_14_gpu_histogram.md (deleted, recover with git log --all --diff-filter=D -- handoffs/performance_14_gpu_histogram.md)
recorded; `BYTES_PER_PARTIAL_CELL` is now imported from `gpu_tiling.mojo`
rather than copied. `MAX_BINS` is `binning.MAX_BINS`, the one bin ceiling
(a bin index is a byte); it is imported here and re-exported to
`gpu_active_rows.mojo`, `gpu_leaf_batching.mojo`, and
`gpu_binned_layout.mojo`, which import it from this module.
"""

from .binning import MAX_BINS
from .gpu_tiling import BYTES_PER_PARTIAL_CELL


# The bin ceiling the GPU backend enforces, and the width the shipping
# kernels allocate in threadgroup memory unconditionally. This is the lane's
# single definition; `gpu_active_rows`, `gpu_leaf_batching`, and
# `gpu_binned_layout` import it from here.


# A histogram is three planes (gradient, hessian, count), each one Int32 per
# bin. Both numbers are properties of the layout `histogram_gpu.mojo` and
# `gpu_active_rows.mojo` already use, not choices made here.
comptime PLANES_PER_HISTOGRAM = 3
comptime BYTES_PER_PLANE_CELL = 4

# The smallest capacity a specialized kernel is instantiated at. Below 16
# bins the threadgroup memory saved is already immaterial next to the launch
# itself, and every additional instantiation is compile time paid by every
# build on every backend.
comptime BIN_CAPACITY_MIN = 16

# 16, 32, 64, 128, 256.
comptime BIN_CLASS_COUNT = 5

# Bytes a lane loads at once on the packed path, and the alignment that path
# requires. Four is the width every backend loads as one 32-bit word.
comptime PACK_LANES = 4

# An aligned body shorter than this is not worth a separate code path: the
# head and tail loops would dominate whatever the body saved.
comptime MIN_PACKED_BODY_QUADS = 4


# --- Bin capacity classes ---


def bin_capacity_for(n_bins: Int) raises -> Int:
    """The capacity a kernel specialized for `n_bins` bins is instantiated
    at: the next power of two at or above `n_bins`, never below
    `BIN_CAPACITY_MIN` and never above `MAX_BINS`.

    A capacity rather than the bin count itself because the specialization is
    a compile-time parameter, and five instantiations covering every bin
    count is the trade this makes against 256 of them. The five capacities
    are exactly the ones the `max_bin` values this project runs against land
    in: 15 and 16 bins both class as 16, 255 and 256 both as 256.
    """
    if n_bins < 1:
        raise Error("histogram needs a positive bin count")
    if n_bins > MAX_BINS:
        raise Error(
            "GPU histogram supports at most ", MAX_BINS, " bins"
        )
    var capacity = BIN_CAPACITY_MIN
    while capacity < n_bins:
        capacity *= 2
    return capacity


def bin_class_index(capacity: Int) -> Int:
    """Position of a capacity in the class ladder (0 for 16 through 4 for
    256), or -1 for a value that is not one of the five."""
    var candidate = BIN_CAPACITY_MIN
    for i in range(BIN_CLASS_COUNT):
        if candidate == capacity:
            return i
        candidate *= 2
    return -1


def bin_class_capacity(index: Int) raises -> Int:
    """The capacity at a ladder position."""
    if index < 0 or index >= BIN_CLASS_COUNT:
        raise Error(
            "bin class index must be 0 through ", BIN_CLASS_COUNT - 1
        )
    var capacity = BIN_CAPACITY_MIN
    for _ in range(index):
        capacity *= 2
    return capacity


def bin_class_max_bin(index: Int) raises -> Int:
    """The `max_bin` a class is named after: 15, 31, 63, 127, 255.

    One bin of the capacity is the missing bin in every dataset this backend
    bins, so a `max_bin` of 255 produces 256 bins and classes at capacity
    256. Reported so a benchmark or a handoff can name a class the way a user
    would set the parameter.
    """
    return bin_class_capacity(index) - 1


def kernel_shared_bytes(capacity: Int) -> Int:
    """Threadgroup memory one block of a kernel specialized at `capacity`
    occupies: three Int32 planes of `capacity` entries."""
    return PLANES_PER_HISTOGRAM * BYTES_PER_PLANE_CELL * capacity


def unspecialized_kernel_shared_bytes() -> Int:
    """Threadgroup memory one block occupies under the pre-parameterization
    allocation: three `MAX_BINS`-wide Int32 planes, at every bin count.

    This is what `_range_hist_atomic_kernel` and `_range_hist_partial_kernel`
    allocated before they took a `BIN_CAP` parameter, and it is what a build
    reporting `KernelFeatures.specialized_bin_kernels = False` is priced at.
    It does not depend on `n_bins`, which was the whole reason the
    bin-capacity specialization existed. Kept, and kept named for what it is,
    because it is also the baseline the default feature group is derived
    against: `gpu_tiling.free_feature_group` widens a group only as far as
    this many bytes per slot already bought.
    """
    return kernel_shared_bytes(MAX_BINS)


def modeled_shared_bytes(n_bins: Int) -> Int:
    """Threadgroup memory an ideal block sized exactly to `n_bins` would
    need: `n_bins * 12`.

    An idealization and no longer a mirror of anything.
    `gpu_tiling.shared_bytes_for` used to compute this and now reports the
    capacity-rounded footprint the kernel family really allocates for one
    feature slot, so the two agree only when `n_bins` is itself a ladder
    value. This survives as the denominator `shared_bytes_unmodeled` measures
    the old allocation's waste against.
    """
    return n_bins * BYTES_PER_PARTIAL_CELL


def shared_bytes_unmodeled(n_bins: Int) -> Int:
    """Bytes of threadgroup memory the pre-parameterization allocation
    occupied beyond an ideal `n_bins`-wide one, at `n_bins` bins.

    Zero at 256 bins and 2688 at 32 bins. This is the waste the `BIN_CAP`
    parameter removed, stated so a handoff or a benchmark can name it; it is
    no longer a correction anyone has to apply to a residency figure, because
    `apple_histogram_policy.mojo` prices the compiled kernel directly.
    """
    var unmodeled = unspecialized_kernel_shared_bytes() - modeled_shared_bytes(
        n_bins
    )
    if unmodeled < 0:
        return 0
    return unmodeled


@fieldwise_init
struct SharedHistogramLayout(Copyable, Movable):
    """Where the three planes of one block's partial histogram sit inside a
    single threadgroup allocation of `capacity` cells each.

    The shipping kernels take three separate `stack_allocation` calls, which
    is equivalent; this describes the one-allocation form a specialized
    kernel would use, so the offsets are stated once here rather than
    recomputed at each of the three use sites (zero, accumulate, flush).

    "A specialized kernel WOULD use" became "one does" on 2026-08-17 and then
    became "none does" again the same day. `gpu_leaf_batching._plan_hist_kernel`
    took one flat allocation of `units * 3 * (n_bins + 1)` cells so that a
    threadgroup could hold several feature slots and several private copies of
    each, with a PAD word on the plane stride that this struct does not carry,
    without which the copies of one bin all land in one memory bank. That kernel
    measured null or worse on every arm and was removed; see
    `docs/design/DECLINED_OPTIMIZATIONS.md` rows E11 and E12. So this layout
    describes the single-copy case, which is the only case there is.
    """

    var capacity: Int
    var grad_offset: Int
    var hess_offset: Int
    var count_offset: Int
    var total_cells: Int
    """Int32 entries in the whole allocation."""

    var total_bytes: Int


def layout_for(capacity: Int) raises -> SharedHistogramLayout:
    """The shared-memory layout for one specialized capacity."""
    if bin_class_index(capacity) < 0:
        raise Error(
            "shared layout needs one of the ",
            BIN_CLASS_COUNT,
            " bin capacities",
        )
    return SharedHistogramLayout(
        capacity,
        0,
        capacity,
        2 * capacity,
        PLANES_PER_HISTOGRAM * capacity,
        kernel_shared_bytes(capacity),
    )


# --- Bin storage descriptor ---
#
# The one description of the device-resident bin matrix every specialization
# in this lane reads. `gpu_binned_layout.mojo` is the only module allowed to
# choose the storage; this module only reports what a chosen storage costs and
# refuses the specializations it invalidates.

comptime BIN_STORAGE_PACKED = 0
"""Sub-byte packing, 1 through 7 bits per cell. A cell is *not* a byte, so
every byte-index assumption elsewhere (`bins[f * n_rows + r]`, the four-lane
packed window) is wrong for it and has to go through
`gpu_bin_packing.element_byte_offset` and `element_bit_shift` instead."""

comptime BIN_STORAGE_U8 = 1
"""One `UInt8` per cell. What `GpuHistogramBuilder` uploads today, and the
only storage the shipping kernels index."""

comptime BIN_STORAGE_U16 = 2
"""Two bytes per cell. Reachable only by a dataset with more than 256 bins,
which `binning.fit_bins` and `BinnedMatrix.bins: List[UInt8]` cannot express,
so no plan in this repository resolves to it. Named rather than omitted so a
descriptor can report the truth instead of silently claiming `UInt8`."""

comptime BIN_STORAGE_WIDER = 3
"""Four or more bytes per cell. Same status as `BIN_STORAGE_U16`."""


def bin_storage_name(storage: Int) -> String:
    if storage == BIN_STORAGE_PACKED:
        return String("packed-subbyte")
    if storage == BIN_STORAGE_U8:
        return String("u8")
    if storage == BIN_STORAGE_U16:
        return String("u16")
    if storage == BIN_STORAGE_WIDER:
        return String("wider")
    return String("unknown")


def storage_for_width(element_bits: Int) raises -> Int:
    """The storage class a per-cell bit width lands in.

    Truthful in both directions: a width of 8 is `BIN_STORAGE_U8` and nothing
    else, a width below 8 is packed and must not be read as bytes, and a width
    above 8 is reported as the wide storage it is rather than rounded down.
    Widths at or below zero are a caller error, not a storage class.
    """
    if element_bits < 1:
        raise Error("a bin cell must be at least one bit wide")
    if element_bits < 8:
        return BIN_STORAGE_PACKED
    if element_bits == 8:
        return BIN_STORAGE_U8
    if element_bits <= 16:
        return BIN_STORAGE_U16
    return BIN_STORAGE_WIDER


def storage_bytes_per_element(storage: Int) raises -> Int:
    """Whole bytes one cell occupies, or 0 for the sub-byte storage, whose
    cells do not occupy a whole number of bytes at all."""
    if storage == BIN_STORAGE_PACKED:
        return 0
    if storage == BIN_STORAGE_U8:
        return 1
    if storage == BIN_STORAGE_U16:
        return 2
    if storage == BIN_STORAGE_WIDER:
        return 4
    raise Error("unknown bin storage class")


def storage_is_byte_addressable(storage: Int) -> Bool:
    """Whether cell `i` of a stream starts at byte `i * bytes_per_element`.
    False for the packed storage, whose cells start at bit offsets."""
    return storage != BIN_STORAGE_PACKED


def storage_is_shipping(storage: Int) -> Bool:
    """Whether the kernels compiled into this build can read this storage.

    Exactly one storage class is: the `UInt8` matrix every shipping kernel
    indexes as `bins[f * n_rows + r]`. Everything else needs a kernel that
    does not exist yet, which is why `BinStorageDescriptor.check_shipping`
    refuses rather than letting a caller upload bytes no kernel can decode.
    """
    return storage == BIN_STORAGE_U8


@fieldwise_init
struct BinStorageDescriptor(Copyable, Movable):
    """How one dataset's bins are stored on the device, and what that costs.

    Small on purpose: everything a histogram kernel needs to know about the
    bin matrix, and nothing about how the layout was chosen. `block_features`
    is the blocking factor `G` (1 for the feature-major matrix in use today),
    which is what decides whether a row's cells are contiguous and how much
    threadgroup memory one block's partial histograms need.

    The two marker flags are not decoration. Packing never renumbers a bin
    (`gpu_bin_packing` refuses to), so the missing bin keeps its id and a
    categorical bin keeps its position in the 256-bit `CatBitset` *provided
    the chosen width can represent it*. A width too narrow for a feature's
    missing bin or its highest category bin would drop the marker silently,
    which is the one way this lane could train on a different dataset than the
    caller binned. `gpu_binned_layout.check_markers_preserved` is what sets
    these, and `check` refuses a descriptor that admits either loss.
    """

    var storage: Int
    var element_bits: Int
    """Bits one (row, feature) cell occupies. 8 for the shipping matrix."""

    var dataset_rows: Int
    var n_features: Int
    var n_bins: Int
    var block_features: Int
    """`G`: features stored together, row-major inside the block. 1 for the
    feature-major matrix, `n_features` for a fully row-major one."""

    var passthrough: Bool
    """Whether the device buffer *is* `BinnedMatrix.bins`: cell (r, f) lives at
    byte `f * dataset_rows + r`, no packing pass ran, and no decode
    instruction executes."""

    var missing_marker_preserved: Bool
    var categorical_marker_preserved: Bool

    @staticmethod
    def dense_u8(
        dataset_rows: Int, n_features: Int, n_bins: Int
    ) raises -> BinStorageDescriptor:
        """The matrix `GpuHistogramBuilder` uploads today.

        This is the conservative descriptor: a caller that has made no layout
        decision passes it and gets exactly the behaviour the shipping kernels
        already have, with the markers preserved because width 8 holds every
        id a `UInt8` bin matrix can contain.
        """
        var d = BinStorageDescriptor(
            BIN_STORAGE_U8,
            8,
            dataset_rows,
            n_features,
            n_bins,
            1,
            True,
            True,
            True,
        )
        d.check()
        return d^

    def check(self) raises:
        """Refuse a descriptor that is internally inconsistent or that admits
        a lost marker. Cheap, and called by every producer here."""
        if self.dataset_rows < 1:
            raise Error("a bin storage descriptor needs at least one row")
        if self.n_features < 1:
            raise Error("a bin storage descriptor needs at least one feature")
        if self.n_bins < 1 or self.n_bins > MAX_BINS:
            raise Error("the GPU backend supports 1 to 256 bins")
        if self.block_features < 1 or self.block_features > self.n_features:
            raise Error("block feature count out of range")
        if self.storage != storage_for_width(self.element_bits):
            raise Error(
                "bin storage class does not match its element width"
            )
        if not self.missing_marker_preserved:
            raise Error(
                "the chosen bin width cannot represent a feature's missing"
                " bin; widen the feature or keep the 8-bit layout"
            )
        if not self.categorical_marker_preserved:
            raise Error(
                "the chosen bin width cannot represent a categorical"
                " feature's highest category bin; widen the feature or keep"
                " the 8-bit layout"
            )
        if self.passthrough and not self.is_dense_feature_major_u8():
            raise Error(
                "only a one-feature-per-block 8-bit layout is a passthrough"
            )

    def check_shipping(self) raises:
        """Refuse a storage no kernel in this build can read.

        The explicit failure the integration contract asks for: a caller that
        resolves to a packed or wide layout is told that the kernels index
        `UInt8` cells, rather than uploading a buffer the kernels would
        misread as bytes and train on.
        """
        self.check()
        if not storage_is_shipping(self.storage):
            raise Error(
                "the GPU histogram kernels read one UInt8 per (row, feature)"
                " cell; this dataset resolved to "
                + bin_storage_name(self.storage)
                + " storage, which needs a decoding kernel that is not"
                " compiled in. Keep the 8-bit feature-major layout"
            )
        if self.block_features != 1:
            raise Error(
                "the GPU histogram kernels index bins[f * n_rows + r], which"
                " is a one-feature-per-block layout; a blocked layout needs a"
                " kernel that is not compiled in"
            )

    def is_dense_feature_major_u8(self) -> Bool:
        """Whether cell (r, f) is byte `f * dataset_rows + r`, which is the
        assumption every byte-index formula in this module rests on."""
        return self.storage == BIN_STORAGE_U8 and self.block_features == 1

    def bytes_per_element(self) raises -> Int:
        return storage_bytes_per_element(self.storage)

    def row_bits_per_block(self) -> Int:
        """Bits one row of one block occupies: `G * element_bits`. What
        decides whether a block's row lands inside one memory sector."""
        return self.block_features * self.element_bits

    def dense_bytes(self) -> Int:
        """What the same matrix costs as the `UInt8` buffer in use today."""
        return self.dataset_rows * self.n_features

    def bin_capacity(self) raises -> Int:
        """The specialization capacity a kernel reading this descriptor is
        instantiated at.

        Bounded by the descriptor's *storage* as well as by `n_bins`: a cell
        of `w` bits can only produce ids `0 .. 2^w - 1`, so a 4-bit feature
        never needs a 256-wide partial histogram however large `n_bins` is.
        """
        var reachable = self.n_bins
        if self.element_bits < 8:
            var by_width = 1 << self.element_bits
            if by_width < reachable:
                reachable = by_width
        return bin_capacity_for(reachable)

    def kernel_shared_bytes_per_block(self) raises -> Int:
        """Threadgroup memory one block of a *specialized* kernel needs to
        hold one partial histogram per feature of the storage block."""
        return self.block_features * kernel_shared_bytes(self.bin_capacity())

    def shipping_shared_bytes_per_block(self) -> Int:
        """Threadgroup memory one block occupied under the
        pre-parameterization allocation, which did not depend on the
        descriptor at all: three `MAX_BINS`-wide Int32 planes. The range
        histogram family allocates
        `gpu_tiling.histogram_shared_bytes(bin_cap, group)` instead, which
        this cannot report because a descriptor carries no group width."""
        return unspecialized_kernel_shared_bytes()


# --- The capability boundary ---


@fieldwise_init
struct DeviceHistogramCapabilities(Copyable, Movable):
    """Device facts a specialized histogram path is allowed to depend on.

    Every field is a report, never an inference, and `portable()` is the
    answer for a device that has told us nothing. A non-portable operation
    may be reached only through a flag here, and only where the portable
    arithmetic producing the identical integers sits beside it. That is the
    entire mechanism keeping a vendor intrinsic out of the common path.
    """

    var subgroup_width: Int
    """Lanes that execute in lockstep, or 0 for unknown, which is the value
    on any backend that does not answer the query (Metal rejects
    `WARP_SIZE`). Nothing in this package divides by it or assumes it, and
    the launch rounding in `gpu_tiling.mojo` is a portable granularity rather
    than a width claim."""

    var wide_byte_loads: Bool
    """Whether a four-byte aligned load of the bin matrix has been shown to
    beat four one-byte loads on this device. False everywhere until a
    measurement exists: `pack4_bins` below is the portable implementation and
    it is the one that runs."""

    var unified_memory: Bool
    """Whether the device budget is shared with the host. Carried so a caller
    reading capabilities alone can size a partial buffer; the budget fraction
    itself is `apple_gpu_policy.partial_budget_bytes`."""

    @staticmethod
    def portable() -> DeviceHistogramCapabilities:
        """A device that has reported nothing beyond being launchable. No
        specialization that needs a device fact is available from here.

        There is deliberately no grid-dimension field. An earlier draft
        carried a portable `grid.z` bound for a batched leaf launch, which was
        a number this project has never established on Metal, CUDA, and HIP
        alike. The batched path that survives (`gpu_leaf_batching.mojo`) packs
        its tiles onto `grid.y`, whose portable bound *is* known and already
        lives in `gpu_tiling.MAX_GRID_DIM_Y`, so nothing needs the third axis
        or a guess about it.
        """
        return DeviceHistogramCapabilities(0, False, False)


@fieldwise_init
struct KernelFeatures(Copyable, Movable):
    """Which specialized kernel variants are compiled into this build.

    Separate from device capabilities on purpose: a device may be perfectly
    able to run a batched launch while the batched kernel does not exist. A
    plan may only select a variant that is both supported by the device and
    present here, and `none()` is what every caller gets until the variants
    land and are measured.
    """

    var specialized_bin_kernels: Bool
    """A histogram kernel instantiated per bin capacity, so its threadgroup
    allocation is `kernel_shared_bytes(capacity)` rather than the fixed
    `MAX_BINS` width."""

    var packed_bin_loads: Bool
    """A row loop that reads four bins per load over a contiguous, aligned
    run of row ids."""

    var batched_leaf_kernel: Bool
    """The batched histogram kernels in `gpu_leaf_batching.mojo` are compiled
    in and validated: several leaves in one launch, their row tiles packed
    onto one `grid.y` axis, each leaf writing its own output slice.

    **IT GATES ONE OF THE TWO CALLERS AND NOT THE OTHER, which is worth
    knowing before this flag is trusted as a reach test.**
    `gpu_leaf_batching.admit_frontier_batch` refuses a HOST-staged frontier
    batch without it. The DEVICE-written plan the oblivious level build runs
    (`GpuLeafBatcher.enqueue_device_plan_batch_fused` and its subtracting arm,
    reached from `histogram_gpu.enqueue_desc_level_children`) does
    not consult it at all, so `batched_leaf_kernel = False` does not mean a fit
    ran no batched kernel. Recorded 2026-08-17 by the GPU histogram lane rather
    than corrected, because making the device path consult the flag would
    change what a symmetric fit can run and is not a docstring's decision.

    Two further accumulation kernels lived on that same device-plan path for
    part of 2026-08-17, `_plan_hist_kernel` behind `MOJOTREES_GPU_HIST_LEAN`,
    `..._PRIVATE`, `..._ROW_SPLIT` and `..._GROUP`, and a pair-indexed
    subtracting kernel behind `MOJOTREES_GPU_HIST_PAIR_GRID`. Both were
    bit-identical to the shipping pair and both measured null or worse, so both
    were removed the same day. See `docs/design/DECLINED_OPTIMIZATIONS.md` rows
    E11 to E13."""

    @staticmethod
    def none() -> KernelFeatures:
        """What this build has today: the portable kernels only."""
        return KernelFeatures(False, False, False)

    @staticmethod
    def all_present() -> KernelFeatures:
        """Every variant compiled in. For a benchmark that has them and wants
        to plan against them; never a default."""
        return KernelFeatures(True, True, True)

    def any(self) -> Bool:
        return (
            self.specialized_bin_kernels
            or self.packed_bin_loads
            or self.batched_leaf_kernel
        )


# --- Packed bin loads ---


@always_inline
def pack4_bins(b0: UInt8, b1: UInt8, b2: UInt8, b3: UInt8) -> UInt32:
    """Four consecutive bin bytes as one 32-bit word, low byte first.

    The portable implementation of the packed load: the caller reads four
    bytes however the backend reads bytes, and this states the packing. A
    backend that has *shown* a single aligned 32-bit load to be faster (see
    `DeviceHistogramCapabilities.wide_byte_loads`) substitutes that load and
    must produce this exact word, little end first, so `unpack_bin` below
    keeps working unchanged and the histogram stays bit-identical.
    """
    return (
        UInt32(b0)
        | (UInt32(b1) << UInt32(8))
        | (UInt32(b2) << UInt32(16))
        | (UInt32(b3) << UInt32(24))
    )


@always_inline
def unpack_bin(word: UInt32, lane: Int) -> Int:
    """Bin id in one lane of a packed word, `lane` in 0 through 3."""
    return Int((word >> UInt32(8 * lane)) & UInt32(0xFF))


@fieldwise_init
struct PackedLoadWindow(Copyable, Movable):
    """How one contiguous run of row ids splits into a scalar head, a packed
    body, and a scalar tail.

    Element indices are into the bin matrix, which is feature-major: the
    element for row `r` of feature `f` is at `f * dataset_rows + r`. The
    alignment that matters is therefore of `f * dataset_rows + first_row`,
    which depends on the feature unless `dataset_rows` is itself a multiple
    of `PACK_LANES`. `plan_packed_window` requires that multiple, so one
    window describes every feature in a launch rather than one per feature.

    **This arithmetic assumes one byte per (row, feature) cell**, which is
    what `GpuHistogramBuilder` uploads today. It is not a safe assumption in
    general: `gpu_bin_packing.mojo` and `gpu_binned_layout.mojo` describe
    sub-byte bin widths, under which an element index is no longer a byte
    offset and a four-row group is no longer four bytes. If either layout is
    adopted for the training path, this window is wrong rather than merely
    unhelpful, and it must be re-derived against the packed stream's own
    offset arithmetic (`gpu_bin_packing.element_byte_offset` and
    `element_bit_shift`) before the packed path is enabled on it.
    """

    var usable: Bool
    """False when the run is too short, unaligned in a way this window does
    not describe, or not a run at all. Then the caller runs the scalar loop
    it runs today, over the whole count."""

    var head_count: Int
    """Rows before the aligned body, read one byte at a time."""

    var body_quads: Int
    """Packed loads in the body, each covering `PACK_LANES` rows."""

    var tail_count: Int
    """Rows after the body, read one byte at a time."""

    var reason: Int
    """Why `usable` is false, as a `WINDOW_*` code below; `WINDOW_OK` when it
    is true."""

    def covered(self) -> Int:
        """Rows this window accounts for, which is the run's whole count
        whether or not the body is usable."""
        return self.head_count + PACK_LANES * self.body_quads + self.tail_count


comptime WINDOW_OK = 0
comptime WINDOW_NOT_A_RUN = 1
comptime WINDOW_STRIDE_UNALIGNED = 2
comptime WINDOW_TOO_SHORT = 3
comptime WINDOW_STORAGE_NOT_BYTES = 4
"""The bin matrix is not one `UInt8` per cell, so an element index is not a
byte offset and a four-row group is not four bytes. Reported rather than
silently mis-planned; see `plan_packed_window_for`."""


def window_reason_name(reason: Int) -> String:
    if reason == WINDOW_OK:
        return String("ok")
    if reason == WINDOW_NOT_A_RUN:
        return String("rows_not_a_contiguous_run")
    if reason == WINDOW_STRIDE_UNALIGNED:
        return String("column_stride_unaligned")
    if reason == WINDOW_TOO_SHORT:
        return String("body_too_short")
    if reason == WINDOW_STORAGE_NOT_BYTES:
        return String("storage_is_not_one_byte_per_cell")
    return String("unknown")


def plan_packed_window(
    dataset_rows: Int,
    first_row: Int,
    count: Int,
    rows_are_contiguous_run: Bool,
) raises -> PackedLoadWindow:
    """Split a node's rows into head, packed body, and tail.

    `rows_are_contiguous_run` is the caller's assertion that the node's slice
    of the active-row permutation holds `first_row` through
    `first_row + count - 1` in ascending order. It is a property of the
    permutation and cannot be derived here, so it is required rather than
    guessed; a false answer costs only the packed path, while a wrong true
    answer would read the wrong rows, which is why nothing infers it.

    An unusable window still reports the full count in `head_count`, so a
    caller can run the same scalar loop over it without a second branch.
    """
    if dataset_rows < 1 or count < 0 or first_row < 0:
        raise Error("packed window needs a nonnegative run inside the matrix")
    if first_row + count > dataset_rows:
        raise Error("packed window runs past the end of a column")

    if not rows_are_contiguous_run:
        return PackedLoadWindow(False, count, 0, 0, WINDOW_NOT_A_RUN)
    if dataset_rows % PACK_LANES != 0:
        # Column `f` starts at `f * dataset_rows`, so an unaligned stride
        # gives every feature a different alignment and one window cannot
        # describe the launch. Per-feature windows are possible and are
        # deliberately not built here; see the handoff.
        return PackedLoadWindow(False, count, 0, 0, WINDOW_STRIDE_UNALIGNED)

    var head = (PACK_LANES - (first_row % PACK_LANES)) % PACK_LANES
    if head > count:
        head = count
    var body_quads = (count - head) // PACK_LANES
    var tail = count - head - PACK_LANES * body_quads
    if body_quads < MIN_PACKED_BODY_QUADS:
        return PackedLoadWindow(False, count, 0, 0, WINDOW_TOO_SHORT)
    return PackedLoadWindow(True, head, body_quads, tail, WINDOW_OK)


def plan_packed_window_for(
    storage: BinStorageDescriptor,
    first_row: Int,
    count: Int,
    rows_are_contiguous_run: Bool,
) raises -> PackedLoadWindow:
    """`plan_packed_window` against a storage descriptor instead of against a
    bare row count.

    This is the form a caller should use, and the reason it exists is stated
    in `PackedLoadWindow`'s own docstring: the four-lane window arithmetic is
    only correct when one cell is one byte and one column starts at
    `f * dataset_rows`. Handing it a packed or blocked layout does not make it
    slower, it makes it read the wrong bytes. So the descriptor decides, and a
    layout the window cannot describe comes back unusable with
    `WINDOW_STORAGE_NOT_BYTES` rather than as a plan.

    `dataset_rows` comes from the descriptor rather than from the caller, so a
    window can never be planned against a row count the matrix does not have.
    """
    storage.check()
    if not storage.is_dense_feature_major_u8():
        return PackedLoadWindow(False, count, 0, 0, WINDOW_STORAGE_NOT_BYTES)
    return plan_packed_window(
        storage.dataset_rows, first_row, count, rows_are_contiguous_run
    )


def features_admit(
    features: KernelFeatures,
    device: DeviceHistogramCapabilities,
    storage: BinStorageDescriptor,
) raises -> Bool:
    """Whether the packed-load specialization may run at all for this build,
    device, and storage.

    All three have to agree, and the storage is the one a caller is most
    likely to forget: a build with `packed_bin_loads` compiled in, on a device
    that reports `wide_byte_loads`, still may not use it on a sub-byte or
    blocked matrix. Reported as a Bool because selection is
    `apple_histogram_policy`'s job, not this module's.
    """
    storage.check()
    if not features.packed_bin_loads:
        return False
    if not device.wide_byte_loads:
        return False
    return storage.is_dense_feature_major_u8()
