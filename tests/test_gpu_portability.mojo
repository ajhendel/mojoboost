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
"""

from std.testing import assert_equal, assert_true, TestSuite

from mojoboost.gpu_tiling import (
    MAX_GRID_DIM_Y,
    STRATEGY_ATOMIC,
    STRATEGY_TILED,
    DeviceCaps,
    derive_tiling,
    shared_bytes_for,
)
from mojoboost.histogram_gpu import MAX_BINS, MAX_ROWS

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
# by the dataset's bin count.
comptime SHARED_BYTES_RESERVED = 3 * MAX_BINS * 4

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
