"""Portability invariants for the GPU kernels, checkable without a GPU.

The equivalence suites (test_backend_equivalence.mojo, test_gpu_training.mojo)
prove the GPU path is correct on whatever accelerator is running them. They
cannot prove it is correct on an accelerator that is not present, and this
project has only ever executed those suites on Apple Metal.

This file covers the gap that leaves. The launch parameters and numeric
bounds the kernels depend on are host-side arithmetic over compile-time
constants and the tiling policy, so the limits that differ between Metal,
CUDA, and HIP can be asserted anywhere, including on a CPU-only CI runner. A
change that would build and pass on Metal while exceeding a CUDA or HIP limit
fails here instead of failing silently on hardware nobody in CI has.

The limits below are the smallest guaranteed across the three backends, not
what any one device reports. Where a real device reports more,
`gpu_tiling.mojo` reads the actual attribute at runtime and uses it; these
floors are what the code may assume when a backend answers nothing.

Scope, against the neighboring test file: `test_gpu_tiling.mojo` checks that
`derive_tiling` is internally consistent for a given `DeviceCaps`. This file
checks that its output is launchable on the weakest device each backend
allows, and that the fixed-point accumulator cannot overflow. Neither
replaces running on the hardware; docs/GPU_VALIDATION.md holds the runner
instructions and the record of what has actually been executed.

**What this file asserts against.** `gpu_portability.mojo` is where the
portable floors are written down and where the gates that enforce them live,
and it is what `histogram_gpu.mojo` calls when a builder opens and whenever
a launch geometry changes. So the checks below run the real gates rather
than re-deriving their arithmetic: a floor that moves in the module moves
here, and a gate that stops refusing something fails here. The three
constants this file used to define for itself are now read from the module
and asserted against the published backend limits instead, because two
copies of a portability floor is exactly the drift the file exists to catch.
"""

from std.testing import assert_equal, assert_raises, assert_true, TestSuite

from mojotrees.apple_gpu_policy import API_CUDA, API_HIP, API_METAL, API_UNKNOWN
from mojotrees.gpu_histogram_specializations import KernelFeatures
from mojotrees.gpu_portability import (
    GRID_AXES,
    MIN_SHARED_MEMORY_PER_BLOCK,
    N_REQUIREMENTS,
    contract_for,
    require_bins_supported,
    require_device_can_host_kernels,
    require_histogram_launchable,
)
from mojotrees.gpu_tiling import (
    MAX_GRID_DIM_Y,
    STRATEGY_ATOMIC,
    STRATEGY_TILED,
    DeviceCaps,
    derive_tiling,
    shared_bytes_for,
)
from mojotrees.histogram_gpu import MAX_BINS, MAX_ROWS

# Smallest shared memory per threadgroup guaranteed across the backends.
# CUDA guarantees 48 KiB per block, AMD LDS is 64 KiB per workgroup, and
# Apple threadgroup memory is 16 KiB on the oldest supported GPUs. The Apple
# floor is the binding one.
comptime PORTABLE_SHARED_BYTES_PER_BLOCK = 16384

# Largest threads per block guaranteed everywhere. CUDA and HIP both cap at
# 1024; Metal's limit is per-pipeline but never below 1024 on Apple silicon.
comptime PORTABLE_MAX_THREADS_PER_BLOCK = 1024

# Wavefront granularity. AMD is 64 wide, NVIDIA and Apple are 32, so a block
# that is a multiple of 64 is a whole number of wavefronts everywhere and
# never launches a partially masked one.
comptime PORTABLE_WARP_GRANULARITY = 64

# Bytes the histogram kernels actually reserve in shared memory: three
# MAX_BINS-long Int32 planes, sized by the compile-time maximum rather than
# by the dataset's bin count. `gpu_portability` computes the same number from
# its own terms, and one of the tests below pins the two together.
comptime SHARED_BYTES_RESERVED = 3 * MAX_BINS * 4

# What the shipping path carries: nothing in the launch sequence reads an API
# name, so `API_UNKNOWN` and its portable floor is the contract every real
# launch is checked against today.
comptime _SHIPPING_API = API_UNKNOWN

# The variants this build compiles in, mirroring
# `histogram_gpu.build_kernel_features()`. Kept as a literal rather than
# imported so a change there has to be noticed here.
comptime _COMPILED_VARIANTS = KernelFeatures(False, False, True)

# The fixed-point scale in histogram_gpu.mojo targets 2^30, half the Int32
# range, for the sum of scaled magnitudes before rounding.
comptime FIXED_POINT_TARGET = 1 << 30

comptime INT32_MAX = Int(Int32.MAX)

# The weakest device the portable floors describe, and a large discrete GPU,
# so every shape below is resolved for both ends of the range.
comptime _MIN_SM_COUNT = 4


def _weakest_device() -> DeviceCaps:
    return DeviceCaps(
        _MIN_SM_COUNT,
        PORTABLE_MAX_THREADS_PER_BLOCK,
        PORTABLE_SHARED_BYTES_PER_BLOCK,
    )


def _large_device() -> DeviceCaps:
    return DeviceCaps(304, 1024, 65536)


def test_reserved_shared_memory_fits_the_portable_floor() raises:
    """The kernels reserve three MAX_BINS-long Int32 planes whatever the
    dataset's bin count, so the reservation is identical on every launch and
    must fit the smallest threadgroup allocation any supported backend
    offers.

    `gpu_tiling.shared_bytes_for` sizes by the dataset's bins, which is the
    smaller number. That is the right input for the policy's own budget, but
    it is not what the kernel asks the driver for, so the floor has to be
    checked against the reservation. On a device whose shared memory sits
    between the two, the policy would accept a shape the launch cannot
    satisfy; the portable floor is what keeps that region empty."""
    assert_true(
        SHARED_BYTES_RESERVED <= PORTABLE_SHARED_BYTES_PER_BLOCK,
        "per-block shared histogram exceeds the portable floor",
    )
    # Enough headroom for more than one resident block, or occupancy is
    # capped at one block per multiprocessor by shared memory alone.
    assert_true(
        SHARED_BYTES_RESERVED * 2 <= PORTABLE_SHARED_BYTES_PER_BLOCK,
        "shared histogram leaves no room for a second resident block",
    )
    # The policy's per-shape estimate is never above the reservation, so a
    # shape it accepts is never larger than what the kernel asks for.
    assert_true(
        shared_bytes_for(MAX_BINS) <= SHARED_BYTES_RESERVED,
        "tiling policy budgets more shared memory than the kernel reserves",
    )


def test_bin_count_fits_the_kernel_index_type() raises:
    """Bins arrive from the binner as UInt8 and index shared memory, so 256
    is the hard ceiling on every backend."""
    assert_true(MAX_BINS <= 256, "more bins than a UInt8 bin value can encode")


def test_module_floor_matches_the_kernel_reservation() raises:
    """`gpu_portability.MIN_SHARED_MEMORY_PER_BLOCK` is what
    `require_device_can_host_kernels` refuses a device for, and it has to be
    the same number the kernels actually reserve.

    The module derives it from `PLANES_PER_HISTOGRAM * BYTES_PER_PLANE_CELL *
    MAX_BINS` and this file derives it as `3 * MAX_BINS * 4`. Pinning them
    together is what keeps the gate honest: a plane count or a cell width
    that changed in one place and not the other would otherwise let the gate
    admit a device the kernels cannot launch on."""
    assert_equal(
        MIN_SHARED_MEMORY_PER_BLOCK,
        SHARED_BYTES_RESERVED,
        "the portability floor and the kernel reservation have diverged",
    )
    assert_true(
        MIN_SHARED_MEMORY_PER_BLOCK <= PORTABLE_SHARED_BYTES_PER_BLOCK,
        "the floor the gate enforces exceeds what every backend guarantees",
    )


def test_every_covered_backend_promises_the_shipping_primitives() raises:
    """Each of the four covered API codes must yield a contract that provides
    every primitive the kernels use, and the two grid axes they launch on.

    This is the check that fails when a backend is added to the table with a
    primitive left False: the shipping kernels use all six unconditionally,
    so a contract that does not promise them describes a device this package
    cannot target, and it should be refused at the table rather than at a
    launch on hardware."""
    var apis = [API_UNKNOWN, API_METAL, API_CUDA, API_HIP]
    for a in range(len(apis)):
        var contract = contract_for(apis[a])
        assert_equal(
            contract.api, apis[a], "contract_for returned another backend"
        )
        assert_true(
            contract.grid_axes >= GRID_AXES,
            "a covered backend must offer the grid axes the kernels launch on",
        )
        assert_equal(
            contract.launch_granularity,
            PORTABLE_WARP_GRANULARITY,
            "launch granularity is not the portable wavefront multiple",
        )
        assert_true(
            contract.max_grid_dim_y <= MAX_GRID_DIM_Y,
            "a contract promises more grid.y than the tightest backend cap",
        )
        for r in range(N_REQUIREMENTS):
            assert_true(
                contract.provides(r),
                "a covered backend does not promise a primitive the shipping"
                " kernels use unconditionally",
            )
        # Float64 device arithmetic is off everywhere, including on backends
        # that have it: one source, and Apple silicon sets the floor.
        assert_true(
            not contract.device_float64_permitted,
            "a backend permits device Float64 that the shared source cannot"
            " emit",
        )


def test_the_shipping_gate_admits_the_weakest_device() raises:
    """The gate `histogram_gpu` calls on every builder and every geometry
    change must accept the portable floor.

    This is the direction that matters: a gate that refuses the weakest
    supported device refuses the path that ships. Running the real
    `require_device_can_host_kernels` and `require_histogram_launchable`
    here, rather than re-deriving their arithmetic, is what makes this a
    test of the gate instead of a second copy of it."""
    var contract = contract_for(_SHIPPING_API)
    var caps = _weakest_device()
    require_device_can_host_kernels(contract, caps)

    var shapes = [10_000, 20, 100_000, 100, 1_000_000, 20]
    var i = 0
    while i + 1 < len(shapes):
        var tiling = derive_tiling(caps, shapes[i], shapes[i + 1], MAX_BINS)
        require_histogram_launchable(
            contract,
            caps,
            tiling,
            shapes[i + 1],
            MAX_BINS,
            # `KernelFeatures` is `Copyable, Movable` and not
            # `ImplicitlyCopyable`, so a comptime value of it cannot cross
            # into a runtime argument on its own.
            materialize[_COMPILED_VARIANTS](),
        )
        i += 2


def test_the_gate_refuses_a_device_below_the_floor() raises:
    """A gate that never refuses anything is not a gate.

    The two properties `require_device_can_host_kernels` exists to enforce
    are the threadgroup memory the kernels allocate and a threadgroup at
    least one launch granularity wide. A device short of either cannot run
    any histogram this package builds, and finding that out at the first
    launch would waste the upload that preceded it."""
    var contract = contract_for(_SHIPPING_API)

    var starved = DeviceCaps(
        _MIN_SM_COUNT,
        PORTABLE_MAX_THREADS_PER_BLOCK,
        MIN_SHARED_MEMORY_PER_BLOCK - 1,
    )
    with assert_raises():
        require_device_can_host_kernels(contract, starved)

    var narrow = DeviceCaps(
        _MIN_SM_COUNT,
        PORTABLE_WARP_GRANULARITY - 1,
        PORTABLE_SHARED_BYTES_PER_BLOCK,
    )
    with assert_raises():
        require_device_can_host_kernels(contract, narrow)


def test_the_gate_refuses_an_unsupported_bin_count() raises:
    """`require_bins_supported` is the one limit `histogram_gpu` stopped
    spelling out for itself, so it has to hold here. Bins index a
    `MAX_BINS`-wide threadgroup plane by a UInt8 value; neither zero nor
    more than `MAX_BINS` is representable."""
    require_bins_supported(1)
    require_bins_supported(MAX_BINS)
    with assert_raises():
        require_bins_supported(0)
    with assert_raises():
        require_bins_supported(MAX_BINS + 1)


def test_derived_block_shape_is_launchable_on_every_backend() raises:
    """Whatever the policy derives has to be a legal block size everywhere,
    and a whole number of wavefronts so AMD launches no partially masked
    wave."""
    var shapes = [10_000, 20, 100_000, 100, 50_000, 400, 1_000_000, 20]
    var devices = [_weakest_device(), _large_device()]

    for d in range(len(devices)):
        var caps = devices[d].copy()
        var i = 0
        while i + 1 < len(shapes):
            var tiling = derive_tiling(
                caps, shapes[i], shapes[i + 1], MAX_BINS
            )
            assert_true(
                tiling.block_threads <= PORTABLE_MAX_THREADS_PER_BLOCK,
                "derived block size exceeds the portable limit",
            )
            assert_true(
                tiling.block_threads <= caps.max_threads_per_block,
                "derived block size exceeds the device's own limit",
            )
            assert_equal(
                tiling.block_threads % PORTABLE_WARP_GRANULARITY,
                0,
                "derived block size is not whole 64-wide wavefronts",
            )
            i += 2


def test_derived_grid_dims_stay_within_portable_limits() raises:
    """Row tiles land in `grid.y`, which CUDA caps at 65535 while HIP and
    Metal allow more. The policy clamps to that cap, so the property to hold
    is that no dataset the builder accepts can push past it, including the
    largest row count that fits in the Int32 the kernels index with."""
    var devices = [_weakest_device(), _large_device()]
    var row_counts = [1, 1_000, 1_000_000, 100_000_000, MAX_ROWS]

    for d in range(len(devices)):
        var caps = devices[d].copy()
        for r in range(len(row_counts)):
            var tiling = derive_tiling(caps, row_counts[r], 8, MAX_BINS)
            assert_true(
                tiling.n_tiles >= 1,
                "every shape must launch at least one row tile",
            )
            assert_true(
                tiling.n_tiles <= MAX_GRID_DIM_Y,
                "grid.y exceeds CUDA's cap for a row count the builder"
                " accepts",
            )
            # Tiles must cover every row, or rows go unaccumulated.
            assert_true(
                tiling.n_tiles * tiling.rows_per_tile >= row_counts[r],
                "row tiles do not cover the dataset",
            )

    # Features land in grid.x, which is 2^31 - 1 everywhere, and the
    # partition kernel is a plain 1D launch over rows into the same
    # dimension. Neither can overflow for anything the builder accepts.
    var partition_blocks = (MAX_ROWS + PORTABLE_WARP_GRANULARITY - 1) // (
        PORTABLE_WARP_GRANULARITY
    )
    assert_true(
        partition_blocks <= INT32_MAX,
        "partition launch exceeds the portable grid.x limit",
    )


def test_both_strategies_resolve_on_the_weakest_device() raises:
    """Neither accumulation strategy may become unreachable on a small
    device. The atomic path is the fallback that needs no partial buffer, and
    the tiled path must still produce a legal launch when forced."""
    var caps = _weakest_device()

    var atomic = derive_tiling(caps, 100_000, 50, MAX_BINS, STRATEGY_ATOMIC)
    assert_equal(atomic.strategy, STRATEGY_ATOMIC)
    assert_equal(
        atomic.partial_cells, 0, "the atomic strategy needs no partial buffer"
    )

    var tiled = derive_tiling(caps, 100_000, 50, MAX_BINS, STRATEGY_TILED)
    assert_equal(tiled.strategy, STRATEGY_TILED)
    assert_true(
        tiled.partial_cells > 0, "the tiled strategy needs a partial buffer"
    )
    assert_true(
        tiled.n_tiles <= MAX_GRID_DIM_Y,
        "forcing the tiled strategy must not exceed CUDA's grid.y cap",
    )


def test_fixed_point_accumulation_cannot_overflow_int32() raises:
    """The scale is chosen so the sum of scaled magnitudes reaches at most
    2^30 before rounding. Each row's value rounds to nearest, adding up to
    0.5 per row, so the worst-case accumulated magnitude is
    2^30 + n_rows / 2 and that must stay inside Int32.

    Solving for the row count gives the ceiling asserted here. It is above a
    billion rows, so no realistic dataset approaches it, but it is the actual
    bound and the reason the scale targets half the Int32 range rather than
    all of it. Determinism depends on this: once accumulation saturates or
    wraps, integer addition stops being the order-independent operation the
    whole bit-exactness argument rests on."""
    var headroom = INT32_MAX - FIXED_POINT_TARGET
    var safe_max_rows = 2 * headroom

    assert_true(
        FIXED_POINT_TARGET + safe_max_rows // 2 <= INT32_MAX,
        "worst-case fixed-point accumulation overflows Int32",
    )
    # Far beyond any dataset that fits in memory, which is why the builder
    # does not enforce it separately.
    assert_true(
        safe_max_rows > 1_000_000_000,
        "fixed-point row ceiling has dropped into a realistic dataset size",
    )
    # Half the Int32 range is what buys that headroom; a larger target would
    # shrink the safe row count proportionally.
    assert_true(
        FIXED_POINT_TARGET * 2 <= INT32_MAX + 1,
        "fixed-point target exceeds half the Int32 range",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
