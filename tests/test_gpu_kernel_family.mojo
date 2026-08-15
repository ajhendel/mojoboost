"""The parameterized range histogram family: selection rules and identity.

`gpu_active_rows._range_hist_atomic_kernel` and `_range_hist_partial_kernel`
are one kernel each, instantiated over a feature-group ladder (1, 2, 4, 8, 16)
and a bin-capacity ladder (32, 64, 128, 256). Two kinds of claim follow from
that, and this file holds both.

The host half is arithmetic over synthetic `DeviceCaps` and runs on every
machine: which capacity a bin count resolves to, which group widths have a
kernel, what a block's threadgroup footprint is, and the one rule that lets
the default group widen without a measurement, namely that
`free_feature_group` never returns a width whose footprint exceeds what the
pre-parameterization allocation already paid at the same baseline.

The device half is the identity claim, and it is bit-exact rather than
approximate: accumulation is fixed-point Int32 and integer addition does not
care in what order it happens, so every instantiation, both accumulation
strategies, and both gradient sources have to produce the same integers. It
is asserted over three bin counts (32, 64, and 256, which resolve to three of
the four capacities), four active-feature counts (1, 3, 8, and 17, so a tail
block owning one slot and a tail block owning many are both exercised), and a
node whose row range is a strict subset of the active prefix rather than the
whole of it. The fused sibling subtraction is checked the same way, against a
host subtraction of the same two histograms.

The device half skips (passing) with no accelerator; the host half runs
everywhere.
"""

from std.sys import has_accelerator
from std.testing import assert_equal, assert_raises, assert_true, TestSuite
from max.gpu.host import DeviceBuffer, DeviceContext

from mojotrees.binning import BinnedMatrix
from mojotrees.gpu_active_rows import (
    FEATURE_GROUP_MAX,
    GpuActiveRows,
    RowRouting,
)
from mojotrees.gpu_tiling import (
    HIST_BIN_CAP_MAX,
    HIST_BIN_CAP_MIN,
    HIST_FEATURE_GROUP_MAX,
    STRATEGY_ATOMIC,
    STRATEGY_TILED,
    DeviceCaps,
    free_feature_group,
    feature_group_for_residency,
    histogram_bin_capacity,
    histogram_shared_bytes,
    is_feature_group_width,
    query_device_caps,
    shared_bytes_for,
)
from support import _splitmix64


# --- Host half: the two ladders and the rules over them ---


def _unspecialized_block_bytes(group: Int) -> Int:
    """What one block of the pre-parameterization allocation cost at feature
    group `group`: three `MAX_BINS`-wide Int32 planes per owned slot, at every
    bin count. The baseline the free-footprint rule is measured against, spelt
    out here rather than imported so the test would notice the definition
    moving underneath it."""
    return 3 * group * 256 * 4


def test_bin_capacity_is_the_smallest_rung_at_or_above_the_bin_count() raises:
    """The capacity ladder, exhaustively over every bin count a UInt8 bin can
    take. Nothing may resolve below the narrowest rung, because there is no
    kernel below it, and nothing at or under 256 may resolve above 256."""
    assert_equal(HIST_BIN_CAP_MIN, 32)
    assert_equal(HIST_BIN_CAP_MAX, 256)
    for n_bins in range(1, 257):
        var cap = histogram_bin_capacity(n_bins)
        assert_true(cap >= HIST_BIN_CAP_MIN)
        assert_true(cap <= HIST_BIN_CAP_MAX)
        assert_true(cap >= n_bins)
        # Smallest such rung: halving it would drop below the bin count, or
        # below the floor.
        assert_true(cap == HIST_BIN_CAP_MIN or (cap // 2) < n_bins)
        # A power of two.
        assert_equal(cap & (cap - 1), 0)
    assert_equal(histogram_bin_capacity(1), 32)
    assert_equal(histogram_bin_capacity(32), 32)
    assert_equal(histogram_bin_capacity(33), 64)
    assert_equal(histogram_bin_capacity(64), 64)
    assert_equal(histogram_bin_capacity(65), 128)
    assert_equal(histogram_bin_capacity(128), 128)
    assert_equal(histogram_bin_capacity(129), 256)
    assert_equal(histogram_bin_capacity(256), 256)


def test_shared_bytes_reports_the_capacity_a_kernel_allocates() raises:
    """`shared_bytes_for` is the footprint of one feature slot at the capacity
    the kernel is instantiated at, so it agrees with `histogram_shared_bytes`
    at group 1 and is never below an ideal `n_bins`-wide figure."""
    for n_bins in range(1, 257):
        var cap = histogram_bin_capacity(n_bins)
        assert_equal(shared_bytes_for(n_bins), histogram_shared_bytes(cap, 1))
        # The old model, `n_bins * 12`, is what the capacity rounds up from.
        assert_true(shared_bytes_for(n_bins) >= n_bins * 12)
    # Three Int32 planes, and linear in both arguments.
    assert_equal(histogram_shared_bytes(32, 1), 3 * 32 * 4)
    assert_equal(histogram_shared_bytes(256, 2), 3 * 2 * 256 * 4)
    assert_equal(histogram_shared_bytes(64, 8), 3 * 8 * 64 * 4)


def test_only_ladder_widths_are_feature_groups() raises:
    """A width with no instantiation cannot launch, so 3 is refused rather
    than rounded, and so is anything above the top rung."""
    assert_equal(FEATURE_GROUP_MAX, HIST_FEATURE_GROUP_MAX)
    assert_equal(HIST_FEATURE_GROUP_MAX, 16)
    var rungs = [1, 2, 4, 8, 16]
    for i in range(len(rungs)):
        assert_true(is_feature_group_width(rungs[i]))
    var refused = [-1, 0, 3, 5, 6, 7, 9, 12, 17, 32]
    for i in range(len(refused)):
        assert_true(not is_feature_group_width(refused[i]))


def test_the_free_group_rule_never_costs_more_than_the_old_default() raises:
    """The claim that lets the default widen without a measurement.

    Whatever residency a build had before the kernels took a capacity
    parameter, it had while paying three `MAX_BINS`-wide planes per owned
    slot. A group whose real footprint is at or below that number cannot
    reduce the resident blocks per core on any device, whatever its
    threadgroup budget, so widening to it is not an occupancy regression and
    needs no per-device measurement. This asserts exactly that, over every
    capacity and both baselines the package uses.
    """
    var baselines = [1, 2]
    var caps = [32, 64, 128, 256]
    for bi in range(len(baselines)):
        var baseline = baselines[bi]
        var budget = _unspecialized_block_bytes(baseline)
        for ci in range(len(caps)):
            var cap = caps[ci]
            var group = free_feature_group(cap, baseline)
            # It is a width a kernel exists at.
            assert_true(is_feature_group_width(group))
            # It costs no more than the baseline already cost.
            assert_true(histogram_shared_bytes(cap, group) <= budget)
            # It is the widest such rung: the next one up would exceed it, or
            # there is no next one up.
            assert_true(
                group == HIST_FEATURE_GROUP_MAX
                or histogram_shared_bytes(cap, 2 * group) > budget
            )
            # It is never narrower than the baseline, since the capacity can
            # only be at or below the width the baseline was paid at.
            assert_true(group >= baseline)
    # The concrete answers the package ships: at the full width the rule is
    # the identity, and each halving of the capacity doubles the free group.
    assert_equal(free_feature_group(256, 1), 1)
    assert_equal(free_feature_group(256, 2), 2)
    assert_equal(free_feature_group(128, 1), 2)
    assert_equal(free_feature_group(128, 2), 4)
    assert_equal(free_feature_group(64, 1), 4)
    assert_equal(free_feature_group(64, 2), 8)
    assert_equal(free_feature_group(32, 1), 8)
    assert_equal(free_feature_group(32, 2), 16)
    # A baseline that is not a rung is a caller error, not a rounding.
    with assert_raises():
        _ = free_feature_group(64, 3)
    with assert_raises():
        _ = free_feature_group(0, 1)


def test_residency_group_fits_the_reported_threadgroup_budget() raises:
    """The other selection helper, on synthetic devices: the widest group
    whose footprint times the residency target still fits what the device
    reported, and never a width without a kernel."""
    # Apple M4 reports these exactly; the second is an A100-shaped device.
    var devices = [DeviceCaps(10, 1024, 32768), DeviceCaps(108, 1024, 49152)]
    var caps = [32, 64, 128, 256]
    for di in range(len(devices)):
        var dev = devices[di].copy()
        for ci in range(len(caps)):
            var cap = caps[ci]
            for resident in range(1, 9):
                var group = feature_group_for_residency(dev, cap, resident)
                assert_true(is_feature_group_width(group))
                # Either it fits at the target residency, or it is the
                # narrowest rung there is and nothing narrower could be tried.
                var need = resident * histogram_shared_bytes(cap, group)
                assert_true(
                    need <= dev.max_shared_memory_per_block or group == 1
                )
                # Widest such: the next rung up does not fit.
                assert_true(
                    group == HIST_FEATURE_GROUP_MAX
                    or resident * histogram_shared_bytes(cap, 2 * group)
                    > dev.max_shared_memory_per_block
                )
    # A 32 KiB budget holds one 256-bin block at group 8 and not at 16.
    var m4 = DeviceCaps(10, 1024, 32768)
    assert_equal(feature_group_for_residency(m4, 256, 1), 8)
    assert_equal(feature_group_for_residency(m4, 256, 2), 4)
    assert_equal(feature_group_for_residency(m4, 32, 1), 16)
    with assert_raises():
        _ = feature_group_for_residency(m4, 256, 0)


# --- Device half ---


def _make_data(
    n_rows: Int, n_features: Int, n_bins: Int
) raises -> BinnedMatrix:
    """A column-major binned matrix of pseudorandom bins, as
    test_gpu_active_rows.mojo builds one."""
    var bins = List[UInt8](capacity=n_rows * n_features)
    for f in range(n_features):
        for r in range(n_rows):
            var v = _splitmix64(UInt64(f * n_rows + r) + 0x51ED2701)
            bins.append(UInt8(Int(v % UInt64(n_bins))))
    return BinnedMatrix(bins^, n_rows, n_features, n_bins)


def _fixed_scale(values: List[Float64]) -> Float32:
    var total = 0.0
    for i in range(len(values)):
        total += abs(values[i])
    if total < 1e-12:
        total = 1e-12
    return Float32(Float64(1 << 30) / total)


def _upload_bins(
    mut ctx: DeviceContext, data: BinnedMatrix
) raises -> DeviceBuffer[DType.uint8]:
    # The comptime guard keeps the device instantiation out of CPU-only
    # builds: module-level helpers compile unconditionally.
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var buf = ctx.enqueue_create_buffer[DType.uint8](len(data.bins))
        ctx.enqueue_copy(dst_buf=buf, src_ptr=data.bins.unsafe_ptr())
        ctx.synchronize()
        return buf^


def _identity_case(n_bins: Int, n_features: Int) raises:
    """Every instantiation, both strategies, and both gradient sources build
    the same integer histogram for one node.

    The node is deliberately not the root: a split routes part of the active
    prefix away first, so the range read is `[begin, end)` with `begin > 0`
    and `count < n_active`, which is the shape a grower actually asks for and
    the one a kernel that ignored the window would still pass a root-only
    test against.
    """
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var n_rows = 3000
        var data = _make_data(n_rows, n_features, n_bins)
        var grad = List[Float64](capacity=n_rows)
        var hess = List[Float64](capacity=n_rows)
        for r in range(n_rows):
            grad.append(
                Float64(Int(_splitmix64(UInt64(r) + 7) % 2000)) * 0.001 - 1.0
            )
            hess.append(
                Float64(Int(_splitmix64(UInt64(r) + 991) % 1000)) * 0.001
                + 0.25
            )
        var g_scale = _fixed_scale(grad)
        var h_scale = _fixed_scale(hess)

        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var bins = _upload_bins(ctx, data)
        var rows = GpuActiveRows(ctx, n_rows, n_features, n_bins, caps)
        rows.begin_tree()

        var grad32 = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
        var hess32 = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
        for r in range(n_rows):
            grad32.unsafe_ptr().unsafe_store(r, Float32(grad[r]))
            hess32.unsafe_ptr().unsafe_store(r, Float32(hess[r]))
        var grad_dev = ctx.enqueue_create_buffer[DType.float32](n_rows)
        var hess_dev = ctx.enqueue_create_buffer[DType.float32](n_rows)
        ctx.enqueue_copy(dst_buf=grad_dev, src_ptr=grad32.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=hess_dev, src_ptr=hess32.unsafe_ptr())

        # A non-identity slot permutation, so a kernel that indexed by
        # `block_idx.x` instead of following `feat_ids` would show.
        var feat_dev = ctx.enqueue_create_buffer[DType.int32](n_features)
        with feat_dev.map_to_host() as host:
            var dst = host.unsafe_ptr()
            for s in range(n_features):
                dst.unsafe_store(s, Int32((n_features - 1) - s))

        # Split the root so the node under test owns a strict subset.
        _ = rows.partition(
            bins.unsafe_ptr(), 0, 1, 2, RowRouting.numerical(0, n_bins // 2)
        )
        var node = 2
        var window = rows.range_of(node)
        assert_true(window.begin > 0)
        assert_true(window.count() > 0)
        assert_true(window.count() < rows.n_active())

        var hist_size = n_features * n_bins
        var cells = 3 * hist_size
        var ref_dev = ctx.enqueue_create_buffer[DType.int32](cells)
        var arm_dev = ctx.enqueue_create_buffer[DType.int32](cells)
        var host_ref = ctx.enqueue_create_host_buffer[DType.int32](cells)
        var host_arm = ctx.enqueue_create_host_buffer[DType.int32](cells)

        var cap = histogram_bin_capacity(n_bins)
        var strategies = [STRATEGY_ATOMIC, STRATEGY_TILED]
        var groups = [1, 2, 4, 8, 16]

        # The reference: narrowest group, atomic accumulation, Float32
        # gradients. That is the arm nearest what shipped before this family
        # existed, so every other arm is compared against the old behavior and
        # not merely against its neighbor.
        var ref_tiling = rows.range_tiling(
            caps, node, n_features, STRATEGY_ATOMIC, 1
        )
        # The atomic path writes no partials, but a live buffer has to be
        # named rather than passed as a temporary whose allocation the call
        # would outlive.
        var ref_part = ctx.enqueue_create_buffer[DType.int32](1)
        rows.set_feature_group(1)
        rows.set_quantized_gradients(False)
        rows.enqueue_range_histogram(
            ref_tiling,
            node,
            bins.unsafe_ptr(),
            grad_dev.unsafe_ptr(),
            hess_dev.unsafe_ptr(),
            feat_dev.unsafe_ptr(),
            ref_dev.unsafe_ptr(),
            ref_part.unsafe_ptr(),
            n_features,
            g_scale,
            h_scale,
        )
        ctx.enqueue_copy(dst_ptr=host_ref.unsafe_ptr(), src_buf=ref_dev)
        ctx.synchronize()
        var populated = 0
        var total_count = 0
        for i in range(cells):
            if host_ref.unsafe_ptr().unsafe_load(i) != 0:
                populated += 1
        for b in range(hist_size):
            total_count += Int(
                host_ref.unsafe_ptr().unsafe_load(2 * hist_size + b)
            )
        # A comparison of all-zero buffers would pass for the wrong reason,
        # and a window the kernel ignored would count every active row.
        assert_true(populated > 0)
        assert_equal(total_count, n_features * window.count())

        var arms = 0
        for si in range(len(strategies)):
            var strategy = strategies[si]
            var tiling = rows.range_tiling(
                caps, node, n_features, strategy, 1 << 20
            )
            assert_equal(tiling.strategy, strategy)
            var part_cells = tiling.partial_cells
            if part_cells < 1:
                part_cells = 1
            var part_dev = ctx.enqueue_create_buffer[DType.int32](
                3 * part_cells
            )
            for gi in range(len(groups)):
                var group = groups[gi]
                # A rung whose footprint this device cannot hold is refused by
                # `set_feature_group`, which is the behavior under test
                # elsewhere; here it is simply not an arm.
                if (
                    histogram_shared_bytes(cap, group)
                    > caps.max_shared_memory_per_block
                ):
                    continue
                rows.set_feature_group(group)
                assert_equal(rows.feature_group, group)
                for q in range(2):
                    rows.set_quantized_gradients(q == 1)
                    rows.enqueue_range_histogram(
                        tiling,
                        node,
                        bins.unsafe_ptr(),
                        grad_dev.unsafe_ptr(),
                        hess_dev.unsafe_ptr(),
                        feat_dev.unsafe_ptr(),
                        arm_dev.unsafe_ptr(),
                        part_dev.unsafe_ptr(),
                        n_features,
                        g_scale,
                        h_scale,
                    )
                    ctx.enqueue_copy(
                        dst_ptr=host_arm.unsafe_ptr(), src_buf=arm_dev
                    )
                    ctx.synchronize()
                    var a = host_ref.unsafe_ptr()
                    var b2 = host_arm.unsafe_ptr()
                    for i in range(cells):
                        assert_equal(
                            Int(a.unsafe_load(i)), Int(b2.unsafe_load(i))
                        )
                    arms += 1
        # Every rung that fits, both strategies, both gradient sources.
        assert_true(arms >= 8)
        rows.set_feature_group(1)
        rows.set_quantized_gradients(False)


def test_every_instantiation_agrees_at_32_bins() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        _identity_case(32, 1)
        _identity_case(32, 3)
        _identity_case(32, 8)
        _identity_case(32, 17)


def test_every_instantiation_agrees_at_64_bins() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        _identity_case(64, 3)
        _identity_case(64, 17)


def test_every_instantiation_agrees_at_256_bins() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        _identity_case(256, 3)
        _identity_case(256, 17)


def _fused_subtract_case(strategy: Int, group: Int) raises:
    """The fused sibling subtraction, at one strategy and one group width.

    A build that folds the subtraction in has to leave the parent slot holding
    exactly what a plain build followed by a host subtraction leaves. Both are
    fixed-point Int32 under one scale, so the comparison is bit for bit. At a
    group above 1 a block owns several slices and has to subtract each of
    them, and the tail block one to `GROUP - 1`, which is the case a
    per-block rather than per-slice subtraction would get wrong.
    """
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var n_rows = 2048
        var n_features = 7
        var n_bins = 64
        var data = _make_data(n_rows, n_features, n_bins)
        var grad = List[Float64](capacity=n_rows)
        var hess = List[Float64](capacity=n_rows)
        for r in range(n_rows):
            grad.append(
                Float64(Int(_splitmix64(UInt64(r) + 7) % 2000)) * 0.001 - 1.0
            )
            hess.append(
                Float64(Int(_splitmix64(UInt64(r) + 991) % 1000)) * 0.001
                + 0.25
            )
        var g_scale = _fixed_scale(grad)
        var h_scale = _fixed_scale(hess)

        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var bins = _upload_bins(ctx, data)
        var rows = GpuActiveRows(ctx, n_rows, n_features, n_bins, caps)
        if (
            histogram_shared_bytes(histogram_bin_capacity(n_bins), group)
            > caps.max_shared_memory_per_block
        ):
            return
        rows.set_feature_group(group)
        rows.begin_tree()

        var grad32 = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
        var hess32 = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
        for r in range(n_rows):
            grad32.unsafe_ptr().unsafe_store(r, Float32(grad[r]))
            hess32.unsafe_ptr().unsafe_store(r, Float32(hess[r]))
        var grad_dev = ctx.enqueue_create_buffer[DType.float32](n_rows)
        var hess_dev = ctx.enqueue_create_buffer[DType.float32](n_rows)
        ctx.enqueue_copy(dst_buf=grad_dev, src_ptr=grad32.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=hess_dev, src_ptr=hess32.unsafe_ptr())

        var feat_dev = ctx.enqueue_create_buffer[DType.int32](n_features)
        with feat_dev.map_to_host() as host:
            for f in range(n_features):
                host.unsafe_ptr().unsafe_store(f, Int32(f))

        var hist_size = n_features * n_bins
        var cells = 3 * hist_size
        # Two slots of one pool, as the resident frontier holds them: slot 0
        # is the parent to derive from, slot 1 the child to build.
        var pool = ctx.enqueue_create_buffer[DType.int32](2 * cells)
        var host_pool = ctx.enqueue_create_host_buffer[DType.int32](2 * cells)

        _ = rows.partition(
            bins.unsafe_ptr(), 0, 1, 2, RowRouting.numerical(0, n_bins // 2)
        )
        var node = 2
        assert_true(rows.range_of(node).count() > 0)

        var tiling = rows.range_tiling(
            caps, node, n_features, strategy, 1 << 20
        )
        assert_equal(tiling.strategy, strategy)
        var part_cells = tiling.partial_cells
        if part_cells < 1:
            part_cells = 1
        var part_dev = ctx.enqueue_create_buffer[DType.int32](3 * part_cells)

        rows.enqueue_range_histogram(
            tiling,
            node,
            bins.unsafe_ptr(),
            grad_dev.unsafe_ptr(),
            hess_dev.unsafe_ptr(),
            feat_dev.unsafe_ptr(),
            pool.unsafe_ptr().unsafe_offset(cells),
            part_dev.unsafe_ptr(),
            n_features,
            g_scale,
            h_scale,
        )
        ctx.enqueue_copy(dst_ptr=host_pool.unsafe_ptr(), src_buf=pool)
        ctx.synchronize()
        var child = List[Int32](capacity=cells)
        var nonzero = 0
        for i in range(cells):
            var v = host_pool.unsafe_ptr().unsafe_load(cells + i)
            child.append(v)
            if v != 0:
                nonzero += 1
        assert_true(nonzero > 0)

        # Seed a parent with values a subtraction cannot hide in, then build
        # the same node again with the subtraction folded in.
        var seed = List[Int32](capacity=cells)
        with pool.map_to_host() as host:
            var dst = host.unsafe_ptr()
            for i in range(cells):
                var v = Int32(1000 + 7 * i)
                seed.append(v)
                dst.unsafe_store(i, v)
            for i in range(cells):
                dst.unsafe_store(cells + i, Int32(0))

        rows.enqueue_range_histogram(
            tiling,
            node,
            bins.unsafe_ptr(),
            grad_dev.unsafe_ptr(),
            hess_dev.unsafe_ptr(),
            feat_dev.unsafe_ptr(),
            pool.unsafe_ptr().unsafe_offset(cells),
            part_dev.unsafe_ptr(),
            n_features,
            g_scale,
            h_scale,
            -cells,
            True,
        )
        ctx.enqueue_copy(dst_ptr=host_pool.unsafe_ptr(), src_buf=pool)
        ctx.synchronize()
        var got = host_pool.unsafe_ptr()
        for i in range(cells):
            # The child slot is the same histogram as the unfused build.
            assert_equal(Int(got.unsafe_load(cells + i)), Int(child[i]))
            # The parent slot is the seed minus exactly that histogram.
            assert_equal(
                Int(got.unsafe_load(i)), Int(seed[i]) - Int(child[i])
            )


def test_fused_subtraction_is_exact_at_every_width() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var groups = [1, 2, 4, 8, 16]
        for gi in range(len(groups)):
            _fused_subtract_case(STRATEGY_ATOMIC, groups[gi])
            _fused_subtract_case(STRATEGY_TILED, groups[gi])


def test_a_width_without_a_kernel_is_refused() raises:
    """`set_feature_group` refuses two different things: a width off the
    ladder, which has no instantiation, and a rung whose threadgroup
    footprint exceeds what this device reported, which would compile and then
    fail at the launch."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        var narrow = GpuActiveRows(ctx, 64, 4, 16, caps)
        assert_equal(narrow.bin_cap, 32)
        with assert_raises():
            narrow.set_feature_group(0)
        with assert_raises():
            narrow.set_feature_group(3)
        with assert_raises():
            narrow.set_feature_group(FEATURE_GROUP_MAX + 1)
        # Every rung is affordable at 32 bins on any device that can host the
        # kernels at all, since 16 slots of 32 bins is what 2 slots of 256
        # already cost.
        narrow.set_feature_group(16)
        assert_equal(narrow.feature_group, 16)

        # At the full width, a rung the device cannot hold is refused where it
        # is asked for.
        var wide = GpuActiveRows(ctx, 64, 4, 256, caps)
        assert_equal(wide.bin_cap, 256)
        var rungs = [1, 2, 4, 8, 16]
        for i in range(len(rungs)):
            var group = rungs[i]
            var need = histogram_shared_bytes(256, group)
            if need > caps.max_shared_memory_per_block:
                with assert_raises():
                    wide.set_feature_group(group)
            else:
                wide.set_feature_group(group)
                assert_equal(wide.feature_group, group)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
