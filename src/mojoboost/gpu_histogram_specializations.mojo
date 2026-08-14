"""Primitives for specializing the GPU histogram path.

The pieces a specialized histogram kernel would be built out of, and nothing
that launches one. Every function here is arithmetic over Ints and Bools: no
`DeviceContext`, no allocation, no kernel, no environment read. That is
deliberate, and it is what lets the whole specialization story be reasoned
about (and later tested) on a machine with no accelerator.

Three specializations are described here, in increasing order of how much
they demand of the kernel side:

1. **Bin-capacity classes.** The shipping kernels in `gpu_active_rows.mojo`
   allocate three `MAX_BINS`-wide Int32 arrays in threadgroup memory
   regardless of `n_bins`, so a 32-bin histogram occupies exactly as much
   threadgroup memory as a 256-bin one: 3 KiB. `gpu_tiling.shared_bytes_for`
   models the footprint as `n_bins * 12`, which is the footprint a kernel
   sized to its bin count *would* have. The two agree only at 256 bins.
   `bin_capacity_for` rounds a bin count up to the next power of two (16, 32,
   64, 128, 256, covering the `max_bin` values 15, 31, 63, 127 and 255 that
   LightGBM parity keeps this project in), and `kernel_shared_bytes` gives
   the footprint a kernel instantiated at that capacity really has. Closing
   that gap is the only one of the three specializations that needs no new
   device primitive at all, only a comptime parameter on the existing
   kernels.

2. **Packed bin loads.** The row loop reads `bins[f * n_rows + r]` where `r`
   comes from the active-row permutation, so it is a gather and not a
   contiguous read. It is contiguous exactly when the node's slice of the
   permutation is an ascending run of consecutive row ids, which is a
   property of the permutation and has to be reported by the caller, never
   assumed. `plan_packed_window` computes the aligned span of such a run, and
   `pack4_bins`/`unpack_bin` are the portable pack and extract. See the
   capability boundary below for what a wide load would have to prove first.

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

Two constants are mirrored from `gpu_tiling.mojo` so this module stands alone
while it lands alongside concurrent work on that file, following the pattern
`apple_gpu_policy.mojo` established. They are marked, and
handoffs/performance_14_gpu_histogram.md records that they collapse into
imports at integration.
"""


# --- Mirrors of gpu_tiling.mojo. ---

# The bin ceiling the GPU backend enforces, and the width the shipping
# kernels allocate in threadgroup memory unconditionally.
comptime MAX_BINS = 256

# Int32 gradient + Int32 hessian + Int32 count per (tile, feature, bin).
comptime BYTES_PER_PARTIAL_CELL = 12

# --- End mirrors. ---


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


def _ceil_div(a: Int, b: Int) -> Int:
    return (a + b - 1) // b


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
    """Threadgroup memory one block of the shipping kernels occupies, at
    every bin count: three `MAX_BINS`-wide Int32 planes.

    This is what `_range_hist_atomic_kernel` and `_range_hist_partial_kernel`
    allocate today, and it does not depend on `n_bins`, which is the whole
    reason the bin-capacity specialization exists.
    """
    return kernel_shared_bytes(MAX_BINS)


def modeled_shared_bytes(n_bins: Int) -> Int:
    """Threadgroup memory `gpu_tiling.shared_bytes_for` models one block as
    needing. Mirrored here so the two numbers can be compared without
    importing the host-side module."""
    return n_bins * BYTES_PER_PARTIAL_CELL


def shared_bytes_unmodeled(n_bins: Int) -> Int:
    """Bytes of threadgroup memory the shipping kernels occupy that the
    tiling model does not account for, at `n_bins` bins.

    Zero at 256 bins and 2688 at 32 bins. Any residency figure derived from
    `modeled_shared_bytes` is optimistic by this much until the kernels take
    a capacity parameter; `apple_histogram_policy.mojo` uses whichever of the
    two footprints matches the kernel actually compiled in.
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
    onto one `grid.y` axis, each leaf writing its own output slice."""

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


def window_reason_name(reason: Int) -> String:
    if reason == WINDOW_OK:
        return String("ok")
    if reason == WINDOW_NOT_A_RUN:
        return String("rows_not_a_contiguous_run")
    if reason == WINDOW_STRIDE_UNALIGNED:
        return String("column_stride_unaligned")
    if reason == WINDOW_TOO_SHORT:
        return String("body_too_short")
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
